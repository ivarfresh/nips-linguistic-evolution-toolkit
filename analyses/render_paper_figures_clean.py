#!/usr/bin/env python3
"""Re-render the paper's supplementary figures without plot titles (Edward, 6 Oct meeting).

Each figure is re-drawn by the script that made it, with matplotlib patched so that:
  * figure titles (suptitle) are not drawn,
  * figure-level footer notes (fig.text, e.g. "Each dot = one run ...") are not drawn,
  * panel subtitles keep their first line only when the extra lines are run-count notes
    ("homogeneous · n = 10 per box" -> dropped), so panels show just the group name,
  * only the figures the paper uses are saved, into --out under their Overleaf paths;
    every other savefig call is skipped, so no tracked figure is overwritten.
Each script runs in its own process, so style settings cannot leak between figures.
Data, colours and layout are otherwise unchanged. The scripts may still rewrite their
CSV side outputs; run from a scratch checkout and discard those.

API keys are blanked for the run, so nothing here can spend money.

Usage: python3 analyses/render_paper_figures_clean.py --out ~/Downloads/no_title_figures
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import re
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ("OPENROUTER_API_KEY", "TOGETHER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
    os.environ[key] = ""

import matplotlib  # noqa: E402
matplotlib.use("Agg")
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

N10 = ROOT / "docs/figures/mixed_vs_average_n10_20261001"
SCRATCH = ROOT / "data/analysis/_clean_render_scratch"
TARGETS: dict[str, str] = {}  # basename the script saves -> path under --out (set per job)
OUT = Path()
SAVED: list[str] = []
DROP_AXES_TITLES = [False]  # for scripts that put the figure title in a single panel's set_title

_text, _set_title, _savefig = Figure.text, Axes.set_title, Figure.savefig


def no_suptitle(self, *args, **kwargs):
    return None


def text(self, *args, **kwargs):
    # supxlabel/supylabel draw through Figure.text from _suplabels; keep those, drop footers.
    if any(f.function == "_suplabels" for f in inspect.stack()[1:4]):
        return _text(self, *args, **kwargs)
    return None


def set_title(self, label, *args, **kwargs):
    if DROP_AXES_TITLES[0]:
        return None
    # Drop run-count lines ("homogeneous · n = 10 per box"); strip run counts from the rest
    # ("Opus 5 (frontier, n = 5 runs)" -> "Opus 5 (frontier)") so model names stay.
    lines = [l for l in str(label).split("\n") if not re.search(r"(^|·)\s*(homogeneous|mixed)\b|n\s*=\s*\d+\s*per box", l)]
    lines = [re.sub(r",?\s*n\s*=\s*\d+(\s*(runs?|per (box|panel|point)))?", "", l).replace("()", "").strip() for l in lines]
    # "Opus 5 (frontier)" / "vs Sonnet 4.5 (September)" -> "Opus 5 vs Sonnet 4.5" (the legend says which is which).
    lines = [re.sub(r"\s*\(frontier\)", "", l) for l in lines if l]
    label = "\n".join(lines).replace("\nvs ", " vs ")
    return _set_title(self, label, *args, **kwargs)


LABEL_FIXES = [
    ("frontier arm (left box)", "frontier model (left box)"),
    ("September reference (right box)", "main model, same provider (right box)"),
    (" (high)", ""),
]


def clean_label(text: str) -> str:
    """Paper wording: no 'September', no '(high)', Sol always with its GPT version."""
    for old, new in LABEL_FIXES:
        text = text.replace(old, new)
    text = re.sub(r"\s*\(?September\)?", "", text)
    return re.sub(r"(GPT-[\d.]+ )?\bSol\b", lambda m: m.group(0) if m.group(1) else "GPT-5.6 Sol", text)


def savefig(self, fname, *args, **kwargs):
    from matplotlib.text import Text
    for t in self.findobj(Text):
        if t.get_text():
            t.set_text(clean_label(t.get_text()))
    name = Path(str(fname)).name
    if name not in TARGETS:
        # Side figures the script reads back go to the scratch folder; anything else is skipped.
        if SCRATCH in Path(str(fname)).resolve().parents:
            return _savefig(self, fname, *args, **kwargs)
        return None
    target = OUT / TARGETS[name]
    target.parent.mkdir(parents=True, exist_ok=True)
    kwargs["bbox_inches"] = "tight"
    _savefig(self, target, *args, **kwargs)
    SAVED.append(str(target))


Figure.suptitle, Figure.text, Axes.set_title, Figure.savefig = no_suptitle, text, set_title, savefig


def run_script(path: str, argv: list[str], targets: dict[str, str], drop_axes_titles: bool = False) -> None:
    DROP_AXES_TITLES[0] = drop_axes_titles
    TARGETS.clear()
    TARGETS.update(targets)
    sys.argv = [path, *argv]
    try:
        runpy.run_path(str(ROOT / path), run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0):
            raise


def load_module(path: str):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_n10(path: str, targets: dict[str, str]) -> None:
    """Scripts that analyses/table1_n10.py re-points at the n = 10 tables."""
    import pandas as pd
    TARGETS.clear()
    TARGETS.update(targets)
    import analyses.table1_n10 as t1
    t1.use_n10_final_paths(t1.n10_pools())  # as table1_n10.py does before calling these scripts
    mod = load_module(path)
    mod.DYADS, mod.POPULATIONS = N10 / "dyad_decisions.csv", N10 / "population_games.csv"
    mod.OUTPUT = SCRATCH
    mod.OUTPUT.mkdir(parents=True, exist_ok=True)
    mod.EXPECTED_DYAD_RUNS = pd.read_csv(mod.DYADS)["path"].nunique()
    mod.EXPECTED_POPULATION_RUNS = pd.read_csv(mod.POPULATIONS)["path"].nunique()
    mod.main()


def run_marker_rates(targets: dict[str, str]) -> None:
    """Cross-family signature words: replot from the saved n = 10 rates table (no re-analysis)."""
    import pandas as pd
    os.environ["LINGUISTIC_DATASET"] = "september_n10"
    TARGETS.clear()
    TARGETS.update(targets)
    mod = load_module("analyses/linguistic_uptake.py")
    mod.plot_marker_rates(pd.read_csv(mod.FIGS / "marker_rates_by_round.csv"))


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--job", type=int, help="run one job in this process (used internally)")
    args = ap.parse_args()
    OUT = args.out.expanduser()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    jobs = [
        lambda: run_script("analyses/figure2_noise_comparison.py", ["--out", str(SCRATCH)],
                           {"boxplots_no_noise.png": "figures/boxplots_no_noise.png",
                            "boxplots_noise_uninformed.png": "figures/boxplots_noise_uninformed.png"}),
        lambda: run_script("analyses/noise_strength_bridge_20260916.py", ["--out", str(SCRATCH)],
                           {"resource_boxplots.png": "figures/noise_U(-2,0)_boxplot.png"}),
        lambda: run_script("analyses/resources_final_boxplots.py",
                           ["--provenance", str(ROOT / "docs/figures/negative_only_crossmodel_reasoning_rerun_20260909_resources_boxplots/provenance.json"),
                            "--out", str(SCRATCH)],
                           {"dyad_all_agents_boxplots.png": "figures/dyad_all_agents_boxplots.png",
                            "population_ordinary_agents_boxplots.png": "figures/population_ordinary_agents_boxplots.png",
                            "population_all_agents_boxplots.png": "figures/population_all_agents_boxplots.png"}),
        lambda: run_n10("analyses/mixed_dyad_family_split.py",
                        {"2-agent-mixed-model-simulation-split.png": "figures/mid_tier_n10/2-agent-mixed-model-simulation-split.png",
                         "mixed-model-simulation-8-agent-split.png": "figures/mid_tier_n10/mixed-model-simulation-8-agent-split.png"}),
        lambda: run_n10("analyses/mixed_model_cooperation_per_round.py",
                        {"fig7_dyads_send_per_round.png": "figures/mid_tier_n10/fig7_dyads_send_per_round.png",
                         "fig7_dyads_return_per_round.png": "figures/mid_tier_n10/fig7_dyads_return_per_round.png"}),
        lambda: run_script("scripts/analyze_frontier_rerun.py", [],
                           {"resources_boxplots.png": "figures/frontier_resources_boxplots_final.png"}),
        lambda: run_script("scripts/analyze_frontier_main_mixed_20260928.py", [],
                           {"frontier_mixed_populations_resources_boxplots.png": "figures/frontier_mixed_populations_resources_boxplots.png"}),
        lambda: run_script("scripts/analyze_frontier_update_20260928.py", [],
                           {"frontier_update_set_resources_boxplots.png": "figures/frontier_update_set_resources_boxplots.png"}),
        lambda: run_script("scripts/analyze_frontier_defector_populations_20261002.py", [],
                           {"resources_boxplots.png": "figures/Frontier_defector_run/resources_boxplots.png"}),
        lambda: run_script("analyses/frontier_defector_population_figures.py", [],
                           {"fig8_populations_send_per_round.png": "figures/Frontier_defector_run/fig8_populations_send_per_round.png"}),
        lambda: run_script("analyses/plot_slide678_rerun.py", ["--out-dir", str(SCRATCH)],
                           {"slide678_cell_means.png": "figures/ablation_run_8-agent.png"}, drop_axes_titles=True),
        lambda: run_script("analyses/plot_slide678_rerun.py",
                           ["--run-root", str(ROOT / "data/json/noise_experiments/slide678_dyad_rerun_20260917"),
                            "--out-dir", str(SCRATCH), "--ceiling", "150",
                            "--population", "2-agent fixed dyad", "--output-stem", "slide678_dyad_cell_means"],
                           {"slide678_dyad_cell_means.png": "figures/ablation_run_2-agent.png"}, drop_axes_titles=True),
        lambda: run_script("analyses/myth_replay_plots.py", [],
                           {"replay_ablation_style.png": "figures/replay_ablation/replay_ablation_style.png"}),
        lambda: run_marker_rates({"cross_family_marker_rates.png": "figures/mid_tier_n10/cross_family_marker_rates.png"}),
    ]
    if args.job is not None:
        jobs[args.job]()
        print("\n".join(SAVED))
        return
    # One fresh process per figure script, as each was originally run: matplotlib style
    # settings (e.g. grid lines) from one script must not leak into the next.
    import subprocess
    for i in range(len(jobs)):
        subprocess.run([sys.executable, __file__, "--out", str(OUT), "--job", str(i)], check=True)


if __name__ == "__main__":
    main()
