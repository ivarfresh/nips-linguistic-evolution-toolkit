#!/usr/bin/env python3
"""Partner-marked board vs partner myth in Sonnet 4.5 populations with two defectors (2026-11-09).

Pairs every partner-board run (myth_board_pilot_20261107, stage partner_board) with its seed
twin from the September defectors25 cells (negative_only_reasoning_rerun_population_*_claude_n5)
by task order and replicate, and reports for the six ordinary agents: final resources, the
mean send decision to a defector and to another ordinary agent (overall and by round third),
and the trustee's return share to defectors and ordinary investors.

    python3 scripts/analyze_myth_board_partner_20261109.py [--data-root data/json] [--out DIR]

With --out the tables are written and the folder's provenance.json is rewritten.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
BOARD_RECEIPT = "noise_experiments/myth_board_pilot_20261107/partner_board_completion_receipt.json"
TWIN_ROOT = "noise_experiments/negative_only_crossmodel_reasoning_rerun_20260909"
TWIN_SETS = {"game_myth": "negative_only_reasoning_rerun_population_game_myth_claude_n5",
             "myth_game": "negative_only_reasoning_rerun_population_myth_game_claude_n5"}
TWIN_PARAMS = "noisy8_crossmodel_negative_defectors25_twotask_r3"
THIRDS = {"rounds 1-3": (1, 3), "rounds 4-7": (4, 7), "rounds 8-10": (8, 10)}


def load_finals(data_root):
    board, twins = {}, {}
    receipt = data_root / BOARD_RECEIPT
    if receipt.exists():
        for f in json.loads(receipt.read_text())["finals"]:
            path = data_root.parent.parent / f["path"]
            d = json.loads(path.read_text())
            d["_path"] = path
            board[(f["task_order"], d["run_metadata"]["replicate_id"])] = d
    for shape, name in TWIN_SETS.items():
        for p in (data_root / TWIN_ROOT / name).rglob(f"{name}_*.json"):
            if p.name.endswith(".results.json") or TWIN_PARAMS not in str(p):
                continue
            d = json.loads(p.read_text())
            d["_path"] = p
            twins[(shape, d["run_metadata"]["replicate_id"])] = d
    return board, twins


def defectors_of(d):
    return set(d["game_data"]["defector_agent_ids"])


def ordinary_mean(d):
    bad = defectors_of(d)
    return statistics.mean(v for a, v in d["game_data"]["balances"].items() if a not in bad)


def sends(d, lo=1, hi=10):
    """Mean send decision by ordinary investors, to defectors and to ordinary trustees."""
    bad = defectors_of(d)
    to_bad, to_good = [], []
    for entry in d["conversation_history"]:
        if not lo <= entry["round"] <= hi:
            continue
        for dy in entry.get("dyads") or []:
            if dy["investor"] in bad or dy.get("sent_decision") is None:
                continue
            (to_bad if dy["trustee"] in bad else to_good).append(dy["sent_decision"])
    return (statistics.mean(to_bad) if to_bad else float("nan"),
            statistics.mean(to_good) if to_good else float("nan"))


def msd(xs):
    return f"{statistics.mean(xs):.2f} (±{statistics.stdev(xs):.2f})" if len(xs) > 1 else f"{xs[0]:.2f}"


# Fields that differ between the runs this folder is built from, each with its reason
# (scripts/check_safeguards.py rejects any undeclared difference).
_SEED = "Seeds follow the replicate; a board run and its twin share them (asserted by the launcher)."
ALLOWED_DIFFERENCES = {
    "protocol.myth.board": "The board is the manipulation: on in the board runs, absent in the partner-myth twins.",
    "protocol.myth.board_seed": "Per-run shuffle order of the board; absent in the twins.",
    "protocol.myth.later_rounds_template": "Board runs use the board version of the later-round myth prompt.",
    "implementation": "Board runs use later src/ code (the board feature); scripts/run_myth_board_pilot.py asserts "
                      "before launch that a board run's inputs differ from its twin's only in the board settings.",
    "replicate.identity.experiment": "Board runs and their twins are separate experiment sets.",
    "replicate.identity.replicate_id": "Ten replicates, paired by replicate id.",
    "replicate.identity.output_path": "Each run has its own output path.",
    "replicate.noise_seed": _SEED, "replicate.pairing_seed": _SEED, "replicate.run_seed": _SEED,
    "replicate.defector_seed": _SEED, "replicate.random_defection_seed": _SEED,
    "protocol.simulation.task_order": "Both task orders are run; every table reports them separately as well as together.",
}


def write_provenance(out, finals):
    """Record the board runs and their twins and hash every file in the folder. Run last, also after a
    README edit: scripts/check_safeguards.py compares the hashes of all tracked files."""
    sys.path.insert(0, str(ROOT))
    from analyses._shared import write_output_provenance
    write_output_provenance(out, sorted(Path(p).resolve() for p in finals), allowed_differences=ALLOWED_DIFFERENCES)
    print(f"provenance: {len(finals)} runs -> {Path(out) / 'provenance.json'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(ROOT / "data/json"))
    p.add_argument("--out", help="directory for CSV tables")
    args = p.parse_args()
    board, twins = load_finals(Path(args.data_root))
    keys = sorted(set(board) & set(twins))
    print(f"board runs {len(board)}; twins {len(twins)}; paired {len(keys)}")
    if not keys:
        print("\nTwins only:")
        for k in sorted(twins):
            print(k, f"ordinary {ordinary_mean(twins[k]):.1f}", "send to defector/ordinary %.2f / %.2f" % sends(twins[k]))
        return
    rows = []
    for k in keys:
        b, t = board[k], twins[k]
        row = {"task_order": k[0], "replicate": k[1], "defectors": ",".join(sorted(defectors_of(b))),
               "partner_ordinary": ordinary_mean(t), "board_ordinary": ordinary_mean(b)}
        for label, (lo, hi) in {"all": (1, 10), **THIRDS}.items():
            tb, tg = sends(t, lo, hi)
            bb, bg = sends(b, lo, hi)
            row.update({f"partner_to_defector_{label}": tb, f"partner_to_ordinary_{label}": tg,
                        f"board_to_defector_{label}": bb, f"board_to_ordinary_{label}": bg})
        rows.append(row)
    for shape in ("game_myth", "myth_game", "both"):
        sub = [r for r in rows if shape == "both" or r["task_order"] == shape]
        if not sub:
            continue
        print(f"\n### {shape} (n={len(sub)} pairs)")
        print("| Measure | Partner myth | Partner-marked board | Difference | pairs | paired t p |")
        print("|---|---|---|---|---|---|")
        for label, pk, bk in (("Ordinary agents' final resources", "partner_ordinary", "board_ordinary"),
                              ("Send to a defector, all rounds", "partner_to_defector_all", "board_to_defector_all"),
                              ("Send to an ordinary agent, all rounds", "partner_to_ordinary_all", "board_to_ordinary_all"),
                              *((f"Send to a defector, {t}", f"partner_to_defector_{t}", f"board_to_defector_{t}") for t in THIRDS),
                              *((f"Send to an ordinary agent, {t}", f"partner_to_ordinary_{t}", f"board_to_ordinary_{t}") for t in THIRDS)):
            ps, bs = [r[pk] for r in sub], [r[bk] for r in sub]
            diffs = [b - a for a, b in zip(ps, bs)]
            pval = stats.ttest_rel(bs, ps).pvalue if len(sub) > 1 else float("nan")
            print(f"| {label} | {msd(ps)} | {msd(bs)} | {statistics.mean(diffs):+.2f} | {sum(x < 0 for x in diffs)} down / {sum(x > 0 for x in diffs)} up | {pval:.3f} |")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "partner_board_vs_partner_myth_by_replicate.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out / 'partner_board_vs_partner_myth_by_replicate.csv'}")
        write_provenance(out, [d["_path"] for k in keys for d in (board[k], twins[k])])


if __name__ == "__main__":
    main()
