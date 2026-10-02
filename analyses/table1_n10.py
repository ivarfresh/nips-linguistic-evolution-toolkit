"""Paper Table 1 (tab:mixed-vs-average) at n = 10 runs per cell.

Pools the original runs (mixed dyads n=6, mixed populations and September
single-model groups n=5) with the 2026-10-01 extension
(data/json/noise_experiments/table1_n10_extension_20261001, quarantine/
excluded), extracts per-run final resources with the existing extractors in
scripts/analyze_mixed_model_{dyads,populations}.py, and runs the unchanged
contrast, bootstrap and Welch/Holm code from analyses/mixed_vs_average.py
(PR #10). Writes only to docs/figures/mixed_vs_average_n10_20261001/ (table,
LaTeX rows, every mid-tier figure, provenance.json); the existing figure
folders are not touched.

    python analyses/table1_n10.py [--mixed-vs-average PATH]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.analyze_mixed_model_dyads as dy  # noqa: E402
import scripts.analyze_mixed_model_populations as po  # noqa: E402
from analyses._shared import load_simulation_runs  # noqa: E402
from analyses._mixed_model_provenance import POOL_REASON  # noqa: E402
from src.experiment_condition import output_provenance  # noqa: E402

EXT = ROOT / "data/json/noise_experiments/table1_n10_extension_20261001"
OUT = ROOT / "docs/figures/mixed_vs_average_n10_20261001"
NO_DEFECTOR = {2: {"noisy2_crossmodel_negative_game_r3", "noisy2_crossmodel_negative_twotask_r3"},
               8: {"noisy8_crossmodel_negative_game_r3", "noisy8_crossmodel_negative_twotask_r3"}}
EXPECTED = {"dyad": [54, 45, 36, 45], "pop": [90, 45, 90, 45]}
ORDER = ["Sonnet + GPT (dyad)", "Sonnet + Gemini (dyad)", "Gemini + GPT (dyad)",
         "1 Gemini + 7 GPT", "2 Gemini + 6 GPT", "4 Gemini + 4 GPT",
         "1 GPT + 7 Sonnet", "2 GPT + 6 Sonnet", "4 GPT + 4 Sonnet"]
_ORIGINAL_FINALS = (dy.final_paths, po.final_paths)


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def finals(pattern, n_agents):
    return sorted(p for p in EXT.rglob(pattern)
                  if "quarantine" not in p.parts and "worker_logs" not in p.parts
                  and not p.name.endswith(dy.NON_FINAL) and p.parent.name in NO_DEFECTOR[n_agents])


def n10_pools():
    """Final paths per size: [original mixed, original single-model, new mixed, new single-model]."""
    d_mixed, d_sept = _ORIGINAL_FINALS[0]()
    p_mixed, p_sept = _ORIGINAL_FINALS[1]()
    pools = {
        "dyad": [d_mixed, d_sept, finals("mixed_dyad_*rep0*.json", 2), finals("negative_only_reasoning_rerun_dyad_*rep0*.json", 2)],
        "pop": [p_mixed, p_sept, finals("mixed_pop_*rep0*.json", 8), finals("negative_only_reasoning_rerun_population_*rep0*.json", 8)],
    }
    counts = {k: [len(x) for x in v] for k, v in pools.items()}
    if counts != EXPECTED:
        raise SystemExit(f"finals (old mixed, old single, new mixed, new single) {counts}, expected {EXPECTED}")
    return pools


def use_n10_final_paths(pools):
    """Point final_paths() of the dyad/population scripts at the n=10 pools (their provenance helpers call it)."""
    dy.final_paths = lambda: (pools["dyad"][0] + pools["dyad"][2], pools["dyad"][1] + pools["dyad"][3])
    po.final_paths = lambda: (pools["pop"][0] + pools["pop"][2], pools["pop"][1] + pools["pop"][3])


def load(pools, allowed):
    runs = {}
    for paths in pools:
        if paths:
            runs.update(load_simulation_runs(paths, allowed_differences=allowed))
    return runs


def write_table_rows(table: pd.DataFrame) -> None:
    """LaTeX rows for tab:mixed-vs-average; bold = uncorrected Welch p < 0.05 (the caption's rule)."""
    cell = {(r.group, r.task_order): r for r in table.itertuples()}
    lines = []
    for i, group in enumerate(ORDER):
        parts = []
        for to in ("game", "game_myth", "myth_game"):
            r = cell[(group, to)]
            value = f"\\mathbf{{{r.diff:.1f}}}" if r.welch_p < 0.05 else f"{r.diff:.1f}"
            parts.append(f"${value}$ [${r.ci_lo:.1f}$, ${r.ci_hi:.1f}$]")
        lines.append(f"    {group:<22} & " + " & ".join(parts) + r" \\")
        if i == 2:
            lines.append(r"    \midrule")
    (OUT / "table_rows.tex").write_text("\n".join(lines) + "\n")


def tracked_or_new(path: Path, start: float) -> bool:
    """Outputs of this run, plus tracked files; untracked copies made by hand stay out of the manifest."""
    if path.stat().st_mtime >= start:
        return True
    return subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT,
                          capture_output=True).returncode == 0


def write_provenance(pools, start: float) -> None:
    mixed = [p.resolve() for size in ("dyad", "pop") for p in pools[size][0] + pools[size][2]]
    single = [p.resolve() for size in ("dyad", "pop") for p in pools[size][1] + pools[size][3]]
    manifest = OUT / "provenance.json"
    outputs = sorted(p for p in OUT.rglob("*") if p.is_file() and p != manifest
                     and not p.name.startswith(".") and tracked_or_new(p, start))
    document = output_provenance(mixed + single, outputs, {**dy.ALLOWED, **po.ALLOWED}, output_root=OUT,
                                 pools={"mixed": mixed, "september": single}, pool_reason=POOL_REASON)
    manifest.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"provenance: {len(mixed + single)} runs, {len(outputs)} outputs -> {manifest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mixed-vs-average", default=str(ROOT / "analyses/mixed_vs_average.py"),
                    help="path to analyses/mixed_vs_average.py (PR #10)")
    args = ap.parse_args()
    start = time.time() - 1

    pools = n10_pools()
    runs = load(pools["dyad"], dy.ALLOWED)
    dec = pd.DataFrame([row for paths in pools["dyad"] for p in paths for row in dy.extract(p, runs[str(p.resolve())])])
    runs = load(pools["pop"], po.ALLOWED)
    game_rows, agent_rows = [], []
    for paths in pools["pop"]:
        for p in paths:
            g, a = po.extract(p, runs[str(p.resolve())])
            game_rows += g
            agent_rows += a
    games, agents = pd.DataFrame(game_rows), pd.DataFrame(agent_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    dec.to_csv(OUT / "dyad_decisions.csv", index=False)
    agents.to_csv(OUT / "population_agent_finals.csv", index=False)
    games.to_csv(OUT / "population_games.csv", index=False)

    for module, name, frame, plotter in ((dy, "dyads", dec, dy.plot_boxplot_grid), (po, "populations", agents, po.plot_boxplot_grid)):
        module.OUTPUT = OUT / name
        module.OUTPUT.mkdir(parents=True, exist_ok=True)
        plotter(frame)
    round_means = dec.groupby(["composition", "task_order", "round"], sort=False).agg(
        sent_mean=("sent", "mean"), return_proportion_mean=("return_proportion", "mean"),
        zero_receipt_rate=("zero_receipt", "mean")).reset_index()
    dy.OUTPUT = OUT / "dyads"
    dy.plot(dec, round_means)
    # The paper's 2-agent figure is the dyad resource boxplot under its Overleaf name.
    shutil.copyfile(OUT / "dyads/resources_boxplots.png", OUT / "dyads/mixed-model-simulation-2-agent.png")
    po.OUTPUT = OUT / "populations"
    po.LADDER_N_NOTE = "mean ± sd over 10 runs"
    po.plot_ladder(agents)

    use_n10_final_paths(pools)
    for script, sub in (("mixed_model_cooperation_per_round.py", "cooperation_per_round"),
                        ("mixed_dyad_family_split.py", "family_split")):
        mod = load_module(ROOT / "analyses" / script)
        mod.DYADS, mod.POPULATIONS = OUT / "dyad_decisions.csv", OUT / "population_games.csv"
        mod.OUTPUT = OUT / sub
        mod.EXPECTED_DYAD_RUNS, mod.EXPECTED_POPULATION_RUNS = dec["path"].nunique(), games["path"].nunique()
        mod.DYAD_N_NOTE = ("n = 10 runs per cell." if "per_round" in script else "n = 10 runs per group")
        mod.POP_N_NOTE = ("n = 10 runs per panel." if "per_round" in script else "n = 10 per point")
        mod.main()

    last = dec.sort_values("round").groupby("path").tail(1)
    dyads = last.assign(v=last["total_balance"] / 2)[["path", "composition", "task_order", "v"]]
    pops = (agents.groupby(["path", "composition", "task_order"])["final_balance"].mean()
            .reset_index().rename(columns={"final_balance": "v"}))
    per_run = pd.concat([dyads, pops], ignore_index=True)
    n = per_run.groupby(["composition", "task_order"]).size()
    print("runs per cell: min", n.min(), "max", n.max())

    mva = load_module(Path(args.mixed_vs_average))
    mva.per_run_values = lambda: per_run
    mva.OUT = OUT
    mva.main()
    write_table_rows(pd.read_csv(OUT / "mixed_vs_average.csv"))
    write_provenance(pools, start)


if __name__ == "__main__":
    main()
