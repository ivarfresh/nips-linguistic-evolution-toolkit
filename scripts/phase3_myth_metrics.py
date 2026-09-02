"""Per-myth metrics for the Phase 3 seed pool.

Applies the *canonical* analyses already in this repo to the 10 seed myths in
`data/phase3/seed_manifest.json`:

  - **Abstractness** via `analyses.myth_compression_curves.myth_metrics`:
    n_tokens, n_types, TTR, Shannon entropy, mean sentence length, n_sentences.

  - **Cooperativeness** via `analyses.cooperativity_analysis.analyze_cooperativity`:
    dictionary-based word-category counting (collective / individual / connected /
    disconnected / giving / taking) → mutuality_ratio and cooperative_pct.

Writes both metric blocks back to each seed entry and prints a per-pool table.

Pure local computation. No API cost.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyses.myth_compression_curves import myth_metrics
from analyses.cooperativity_analysis import analyze_cooperativity


def fmt(values, dec=2):
    if not values:
        return "—"
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.{dec}f} (±{sd:.{dec}f})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/phase3/seed_manifest.json",
        help="Seed manifest JSON",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    # Build myths_by_agent input for analyze_cooperativity: each pool is an
    # "agent", each seed is a "round" (index 0..n-1).
    myths_by_pool = {
        pool: {i: seed["text"] for i, seed in enumerate(seeds)}
        for pool, seeds in manifest.get("seeds", {}).items()
    }
    coop_scores = analyze_cooperativity(myths_by_pool)

    # Write back per-seed metrics + coop scores
    for pool, seeds in manifest.get("seeds", {}).items():
        for i, seed in enumerate(seeds):
            seed["abstractness"] = myth_metrics(seed["text"])
            seed["cooperativity"] = coop_scores[pool][i]

    out_path = args.out or args.manifest
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote metrics back to {out_path}\n")

    # ---- Aggregate report ----
    pools = list(manifest["seeds"].keys())

    print("=" * 118)
    print("ABSTRACTNESS (analyses/myth_compression_curves.py)")
    print("=" * 118)
    print(f"{'Pool':12}  {'n':>3}  {'n_tokens':>16}  {'TTR':>16}  {'entropy_bits':>18}  {'mean_sent_len':>18}  {'n_sentences':>14}")
    print("-" * 118)
    for pool in pools:
        seeds = manifest["seeds"][pool]
        ab = [s["abstractness"] for s in seeds]
        print(
            f"{pool:12}  {len(seeds):>3}  "
            f"{fmt([a['n_tokens'] for a in ab], 1):>16}  "
            f"{fmt([a['ttr'] for a in ab], 3):>16}  "
            f"{fmt([a['entropy_bits'] for a in ab], 3):>18}  "
            f"{fmt([a['mean_sentence_length'] for a in ab], 2):>18}  "
            f"{fmt([a['n_sentences'] for a in ab], 1):>14}"
        )
    print()

    print("=" * 130)
    print("COOPERATIVENESS (analyses/cooperativity_analysis.py — dictionary-based word counts)")
    print("=" * 130)
    print(f"{'Pool':12}  {'n':>3}  {'mutuality_ratio':>18}  {'cooperative_pct':>20}  {'uncooperative_pct':>20}  {'collective+connected+giving':>30}  {'indiv+disc+taking':>22}")
    print("-" * 130)
    for pool in pools:
        seeds = manifest["seeds"][pool]
        cs = [s["cooperativity"] for s in seeds]
        coop_total = [c["collective"] + c["connected"] + c["giving"] for c in cs]
        uncoop_total = [c["individual"] + c["disconnected"] + c["taking"] for c in cs]
        print(
            f"{pool:12}  {len(seeds):>3}  "
            f"{fmt([c['mutuality_ratio'] for c in cs], 2):>18}  "
            f"{fmt([c['cooperative_pct'] for c in cs], 2):>20}  "
            f"{fmt([c['uncooperative_pct'] for c in cs], 2):>20}  "
            f"{fmt(coop_total, 1):>30}  "
            f"{fmt(uncoop_total, 1):>22}"
        )
    print()

    # Per-myth breakdown for cooperativity (so reader can verify)
    print("=" * 110)
    print("Per-seed cooperativity word counts")
    print("=" * 110)
    print(f"{'Pool':10}  {'rep':>3}  {'agent':8}  {'words':>5}  {'collec':>6}  {'indiv':>5}  {'conn':>5}  {'disc':>5}  {'give':>5}  {'take':>5}  {'mut_ratio':>9}  {'coop_pct':>8}")
    print("-" * 110)
    for pool in pools:
        for i, seed in enumerate(manifest["seeds"][pool]):
            c = seed["cooperativity"]
            rep_idx = seed.get("source_run", "").split("rep")[-1].split("_")[0] if seed.get("source_run") else "?"
            print(
                f"{pool:10}  {i:>3}  {seed.get('agent_id', '?'):8}  "
                f"{c['total_words']:>5}  {c['collective']:>6}  {c['individual']:>5}  "
                f"{c['connected']:>5}  {c['disconnected']:>5}  {c['giving']:>5}  {c['taking']:>5}  "
                f"{c['mutuality_ratio']:>9.2f}  {c['cooperative_pct']:>8.2f}"
            )

    print()
    print("Reading the cooperativity numbers:")
    print("  - 'collective' words: together, shared, mutual, partnership, community, ...")
    print("  - 'connected':       relationship, trust, reciprocity, friendship, generosity, ...")
    print("  - 'giving':          give, gave, offered, returned, shared, generous, gift, ...")
    print("  - 'individual':      alone, solitary, self, isolated, lonely, ...")
    print("  - 'disconnected':    betrayal, isolation, conflict, rivalry, ...")
    print("  - 'taking':          took, kept, withheld, refused, hoarded, ...")
    print("  - mutuality_ratio = (collective+connected+giving) / (individual+disconnected+taking + 1)")
    print("  - cooperative_pct = (collective+connected+giving) / total_words × 100")


if __name__ == "__main__":
    main()
