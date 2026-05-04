#!/usr/bin/env python3
"""Print a clean control × noise matrix from the v4_direct_provider_controls dir.

Usage:
    python3 print_control_matrix.py

Reads the canonical *.json simulation files under
data/json/noise_experiments/v4_direct_provider_controls/<experiment_set>/...
and prints mean ± std of mean cumulative balance at round 10 per
(experiment_set, task_order, noise_label, informed) cell.

Does NOT depend on the build_cell_summary.py pipeline — useful for a quick
look while the official cell_summary.csv pipeline is being re-run.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "json" / "noise_experiments"

CONTROL_DIRS = [
    "v4_direct_provider_A1_partner_myth",       # A1 (real partner myth) — pos5/neg5/bootstrap baselines
    "v4_direct_provider_A1_no_noise",           # A1 × no_noise
    "v4_direct_provider_A1_adversarial_bootstrap",
    "v4_direct_provider_A1_targeted_bootstrap",
    "v4_direct_provider_targeted_bootstrap",    # cooperative content × bootstrap (no inj)
    "v4_direct_provider_baseline",              # the no-injection no-noise baselines
    "v4_direct_provider_controls",              # all C1/C2/C3 + new 2026-05-03 batches
]


def balance_r10(path: Path) -> float | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    history = data.get("conversation_history", [])
    rounds = [r for r in history if r.get("sent") is not None]
    if len(rounds) < 10:
        return None
    target = rounds[9]
    bal = target.get("balances") or {}
    a1, a2 = bal.get("Agent_1"), bal.get("Agent_2")
    if a1 is None or a2 is None:
        return None
    return float((a1 + a2) / 2.0)


def collect(model_filter: str | None = "gpt-5-nano", versions: list[str] | None = None):
    cells = defaultdict(list)
    for version in versions or CONTROL_DIRS:
        vroot = DATA_ROOT / version
        if not vroot.exists():
            continue
        for f in vroot.rglob("*.json"):
            if "results" in f.name or "checkpoint" in f.name:
                continue
            parts = f.relative_to(vroot).parts
            if len(parts) < 4:
                continue
            experiment, model, task_order = parts[0], parts[1], parts[2]
            if model_filter and model != model_filter:
                continue
            noise_cond = parts[3] if len(parts) >= 5 else "no_noise"
            informed = "informed" in noise_cond
            noise_label = (noise_cond.replace("_informed", "")
                                     .replace("noisy_", "")
                                     .replace("default", "no_noise"))
            bal = balance_r10(f)
            if bal is None:
                continue
            cells[(version, experiment, model, task_order, noise_label, informed)].append(bal)
    return cells


def fmt_cell(values):
    if not values:
        return "—"
    n = len(values)
    m = statistics.mean(values)
    s = statistics.stdev(values) if n > 1 else 0.0
    return f"{m:5.2f} ± {s:4.2f} (n={n})"


def main():
    parser = argparse.ArgumentParser(description="Print a compact cell matrix from selected noise-experiment dirs.")
    parser.add_argument(
        "--model",
        default="gpt-5-nano",
        help="Model directory to include; use 'all' to disable filtering.",
    )
    parser.add_argument(
        "--versions",
        default=",".join(CONTROL_DIRS),
        help="Comma-separated version dirs under data/json/noise_experiments.",
    )
    args = parser.parse_args()

    model_filter = None if args.model == "all" else args.model
    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    cells = collect(model_filter=model_filter, versions=versions)
    rows = sorted(cells.items())

    print(f"# model filter: {model_filter or 'all'}")
    print(f"# versions: {', '.join(versions)}")
    print(f"{'experiment':50s}  {'task':9s}  {'noise':14s}  {'inf':3s}  cell")
    print("-" * 110)
    cur_exp = None
    for (version, experiment, model, task_order, noise_label, informed), values in rows:
        if experiment != cur_exp:
            print()
            cur_exp = experiment
        inf = "Y" if informed else "N"
        print(f"{experiment:50s}  {task_order:9s}  {noise_label:14s}  {inf:3s}  {fmt_cell(values)}")
    print()
    print(f"# total cells: {len(rows)}")


if __name__ == "__main__":
    main()
