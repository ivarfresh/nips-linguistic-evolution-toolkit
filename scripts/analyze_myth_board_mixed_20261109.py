#!/usr/bin/env python3
"""Board vs partner-myth in the 4 GPT-5 Nano + 4 Sonnet 4.5 population (2026-11-09).

Pairs every board run (myth_board_pilot_20261107, stage mixed_board) with its seed twin from
Table 1 (mixed_pop_game_myth_gpt4_sonnet4_n5 and _n5ext) by replicate id and reports, for the
whole group and per model family: final resources (mean of the agents' balances), the paired
difference, a paired t-test, and the mean send decision per investor -> trustee family.

    python3 scripts/analyze_myth_board_mixed_20261109.py [--data-root data/json] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
BOARD_RECEIPT = "noise_experiments/myth_board_pilot_20261107/mixed_board_completion_receipt.json"
TWIN_SETS = {
    "mixed_pop_game_myth_gpt4_sonnet4_n5": "noise_experiments/mixed_model_populations_20260918",
    "mixed_pop_game_myth_gpt4_sonnet4_n5ext": "noise_experiments/table1_n10_extension_20261001",
}
FAMILY = {"openai/gpt-5-nano": "Nano", "anthropic/claude-sonnet-4.5": "Sonnet"}


def load_finals(data_root):
    board, twins = {}, {}
    receipt = data_root / BOARD_RECEIPT
    if receipt.exists():
        for f in json.loads(receipt.read_text())["finals"]:
            d = json.loads((ROOT / f["path"]).read_text())
            board[d["run_metadata"]["replicate_id"]] = d
    for name, folder in TWIN_SETS.items():
        for p in (data_root / folder / name).rglob(f"{name}_*.json"):
            if p.name.endswith(".results.json"):
                continue
            d = json.loads(p.read_text())
            assert d["run_metadata"]["agent_models"]
            twins[d["run_metadata"]["replicate_id"]] = d
    return board, twins


def family_means(d):
    models = d["run_metadata"]["agent_models"]
    balances = d["game_data"]["balances"]
    out = {"all": statistics.mean(balances.values())}
    for fam in set(FAMILY.values()):
        out[fam] = statistics.mean(v for a, v in balances.items() if FAMILY[models[a]] == fam)
    return out


def sends(d):
    """Mean send decision per investor family -> trustee family over all rounds."""
    models = d["run_metadata"]["agent_models"]
    acc = {}
    for entry in d["conversation_history"]:
        for dy in entry.get("dyads") or []:
            if dy.get("sent_decision") is None:
                continue
            key = f"{FAMILY[models[dy['investor']]]}->{FAMILY[models[dy['trustee']]]}"
            acc.setdefault(key, []).append(dy["sent_decision"])
    return {k: statistics.mean(v) for k, v in sorted(acc.items())}


def msd(xs):
    return f"{statistics.mean(xs):.1f} (±{statistics.stdev(xs):.1f})" if len(xs) > 1 else f"{xs[0]:.1f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(ROOT / "data/json"))
    p.add_argument("--out", help="directory for CSV tables")
    args = p.parse_args()
    board, twins = load_finals(Path(args.data_root))
    reps = sorted(set(board) & set(twins))
    print(f"board runs {sorted(board)}; twins {sorted(twins)}; paired {reps}")
    rows = []
    for r in reps:
        b, t = family_means(board[r]), family_means(twins[r])
        rows.append({"replicate": r, **{f"board_{k}": v for k, v in b.items()}, **{f"partner_{k}": v for k, v in t.items()},
                     **{f"diff_{k}": b[k] - t[k] for k in b}})
    if rows:
        print("\n| Agents | Partner myth | Board | Board minus partner | pairs up | paired t p |")
        print("|---|---|---|---|---|---|")
        for k in ("all", "Nano", "Sonnet"):
            bs, ts = [x[f"board_{k}"] for x in rows], [x[f"partner_{k}"] for x in rows]
            diffs = [x[f"diff_{k}"] for x in rows]
            pval = stats.ttest_rel(bs, ts).pvalue if len(rows) > 1 else float("nan")
            print(f"| {k} | {msd(ts)} | {msd(bs)} | {statistics.mean(diffs):+.1f} | {sum(d > 0 for d in diffs)}/{len(diffs)} | {pval:.3f} |")
        print("\nPer replicate (all agents): " + ", ".join(f"rep{x['replicate']} {x['partner_all']:.1f}->{x['board_all']:.1f}" for x in rows))
        print("\n| Investor -> trustee | Partner myth, mean send | Board, mean send |")
        print("|---|---|---|")
        keys = sorted({k for r in reps for k in sends(twins[r])})
        for k in keys:
            ts = [sends(twins[r]).get(k) for r in reps]
            bs = [sends(board[r]).get(k) for r in reps]
            print(f"| {k} | {statistics.mean(x for x in ts if x is not None):.2f} | {statistics.mean(x for x in bs if x is not None):.2f} |")
    else:
        print("\nTwins only (no board finals yet):")
        for r in sorted(twins):
            print(r, {k: round(v, 1) for k, v in family_means(twins[r]).items()}, {k: round(v, 2) for k, v in sends(twins[r]).items()})
    if args.out and rows:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "board_vs_partner_by_replicate.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out / 'board_vs_partner_by_replicate.csv'}")


if __name__ == "__main__":
    main()
