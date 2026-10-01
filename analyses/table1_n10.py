"""Paper Table 1 (tab:mixed-vs-average) at n = 10 runs per cell.

Pools the original runs (mixed dyads n=6, mixed populations and September
single-model groups n=5) with the 2026-10-01 extension
(data/json/noise_experiments/table1_n10_extension_20261001, quarantine/
excluded), extracts per-run final resources with the existing extractors in
scripts/analyze_mixed_model_{dyads,populations}.py, and runs the unchanged
contrast, bootstrap and Welch/Holm code from analyses/mixed_vs_average.py
(PR #10). Writes only to docs/figures/mixed_vs_average_n10_20261001/; the
existing figure folders are not touched.

    python analyses/table1_n10.py [--mixed-vs-average PATH]
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.analyze_mixed_model_dyads as dy  # noqa: E402
import scripts.analyze_mixed_model_populations as po  # noqa: E402
from analyses._shared import load_simulation_runs  # noqa: E402

EXT = ROOT / "data/json/noise_experiments/table1_n10_extension_20261001"
OUT = ROOT / "docs/figures/mixed_vs_average_n10_20261001"
NO_DEFECTOR = {2: {"noisy2_crossmodel_negative_game_r3", "noisy2_crossmodel_negative_twotask_r3"},
               8: {"noisy8_crossmodel_negative_game_r3", "noisy8_crossmodel_negative_twotask_r3"}}


def finals(pattern, n_agents):
    return sorted(p for p in EXT.rglob(pattern)
                  if "quarantine" not in p.parts and "worker_logs" not in p.parts
                  and not p.name.endswith(dy.NON_FINAL) and p.parent.name in NO_DEFECTOR[n_agents])


def load(pools, allowed):
    runs = {}
    for paths in pools:
        if paths:
            runs.update(load_simulation_runs(paths, allowed_differences=allowed))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mixed-vs-average", default=str(ROOT.parent / "nips-worktrees/mixed-vs-average/analyses/mixed_vs_average.py"),
                    help="path to analyses/mixed_vs_average.py (PR #10)")
    args = ap.parse_args()

    d_mixed, d_sept = dy.final_paths()
    p_mixed, p_sept = po.final_paths()
    pools = {
        "dyad": [d_mixed, d_sept, finals("mixed_dyad_*rep0*.json", 2), finals("negative_only_reasoning_rerun_dyad_*rep0*.json", 2)],
        "pop": [p_mixed, p_sept, finals("mixed_pop_*rep0*.json", 8), finals("negative_only_reasoning_rerun_population_*rep0*.json", 8)],
    }
    counts = {k: [len(x) for x in v] for k, v in pools.items()}
    print("finals (old mixed, old single, new mixed, new single):", counts)
    assert counts == {"dyad": [54, 45, 36, 45], "pop": [90, 45, 90, 45]}, counts

    runs = load(pools["dyad"], dy.ALLOWED)
    dec = pd.DataFrame([row for paths in pools["dyad"] for p in paths for row in dy.extract(p, runs[str(p.resolve())])])
    runs = load(pools["pop"], po.ALLOWED)
    agent_rows = []
    for paths in pools["pop"]:
        for p in paths:
            agent_rows += po.extract(p, runs[str(p.resolve())])[1]
    agents = pd.DataFrame(agent_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    dec.to_csv(OUT / "dyad_decisions.csv", index=False)
    agents.to_csv(OUT / "population_agent_finals.csv", index=False)

    for module, name, frame, plotter in ((dy, "dyads", dec, dy.plot_boxplot_grid), (po, "populations", agents, po.plot_boxplot_grid)):
        module.OUTPUT = OUT / name
        module.OUTPUT.mkdir(parents=True, exist_ok=True)
        plotter(frame)

    last = dec.sort_values("round").groupby("path").tail(1)
    dyads = last.assign(v=last["total_balance"] / 2)[["path", "composition", "task_order", "v"]]
    pops = (agents.groupby(["path", "composition", "task_order"])["final_balance"].mean()
            .reset_index().rename(columns={"final_balance": "v"}))
    per_run = pd.concat([dyads, pops], ignore_index=True)
    n = per_run.groupby(["composition", "task_order"]).size()
    print("runs per cell: min", n.min(), "max", n.max())

    spec = importlib.util.spec_from_file_location("mixed_vs_average", args.mixed_vs_average)
    mva = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mva)
    mva.per_run_values = lambda: per_run
    mva.OUT = OUT
    mva.main()


if __name__ == "__main__":
    main()
