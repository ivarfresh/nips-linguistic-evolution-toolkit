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


CONTEXT = {"gemini": "Gemini-3.7", "models": True}  # set per job: what a bare "Gemini" means in that figure
MODEL = re.compile(r"\b(?:(?:Claude )?Sonnet(?:[ -]4\.5)?|(?:Claude )?Opus(?:[ -]5\.5|[ -]5)?"
                   r"|Gemini(?:[ -]3\.7(?: Flash)?|[ -]3\.1(?: Pro)?)?|GPT-5\.6[ -]Sol|GPT-6[ -]Sol"
                   r"|GPT-5[ -][Nn]ano|Sol(?: \(high\))?|GPT(?![-\w.]))")


def model_name(m: re.Match) -> str:
    t = m.group(0)
    if "Sonnet" in t:
        return "Sonnet-4.5"
    if "Opus" in t:
        return "Opus-5.5" if "5.5" in t else "Opus-5"
    if t.startswith("Gemini"):
        return "Gemini-3.1" if "3.1" in t else "Gemini-3.7" if "3.7" in t else CONTEXT["gemini"]
    if "GPT-6" in t:
        return "GPT-6-Sol"
    if "Sol" in t:
        return "GPT-5.6-Sol"
    return "GPT-5-Nano"


def clean_label(text: str) -> str:
    """Paper figure wording: family-version model names, dyads/populations, single-model/mixed."""
    text = text.replace("frontier arm (left box)", "frontier model (left box)")
    text = text.replace("September reference (right box)", "main model, same provider (right box)")
    text = re.sub(r"\s*\(?September\)?", "", text)
    text = text.replace("GPT-5-nano", "GPT-5-Nano")
    if CONTEXT["models"]:
        text = MODEL.sub(model_name, text)
    text = text.replace("Homogeneous dyads\n(single-model controls)", "Single-model dyads")
    text = text.replace("Single model", "Single-model")
    text = re.sub(r"\bhomogeneous\b", "single-model", text)
    text = re.sub(r"\bHomogeneous\b", "Single-model", text)
    text = re.sub(r"\b2-agent fixed dyad\b", "dyad", text)
    text = re.sub(r"\b8-agent rotating population\b", "population", text)
    text = re.sub(r"\b2-agent(\s+)(single-model|mixed)", r"\2\1dyads", text)
    text = re.sub(r"\b8-agent(\s+)(single-model|mixed)", r"\2\1populations", text)
    text = text.replace("(2-agent)", "(dyads)").replace("(8-agent)", "(populations)")
    text = re.sub(r"^2 agents$", "Dyads", text)
    return re.sub(r"^8 agents$", "Populations", text)


def savefig(self, fname, *args, **kwargs):
    from matplotlib.text import Text
    from matplotlib.ticker import FixedLocator
    for ax in self.axes:  # fixed tick labels are produced by the formatter at draw time; re-set them renamed
        for axis in (ax.xaxis, ax.yaxis):
            if not isinstance(axis.get_major_locator(), FixedLocator):
                continue
            old = [t.get_text() for t in axis.get_ticklabels()]
            new = [clean_label(t) for t in old]
            if new != old:
                axis.set_ticklabels(new)
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


def run_script(path: str, argv: list[str], targets: dict[str, str], drop_axes_titles: bool = False,
               gemini: str = "Gemini-3.7", models: bool = True, env: dict[str, str] | None = None) -> None:
    if not (ROOT / path).exists():  # e.g. a script still on an unmerged branch
        print(f"SKIPPED (script not on this branch): {path}")
        return
    DROP_AXES_TITLES[0] = drop_axes_titles
    CONTEXT.update(gemini=gemini, models=models)
    os.environ.update(env or {})
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
    CONTEXT.update(gemini="Gemini-3.7", models=True)
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
    CONTEXT.update(gemini="Gemini-3.7", models=True)
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
                           {"resources_boxplots.png": "figures/frontier_resources_boxplots_final.png"}, gemini="Gemini-3.1"),
        lambda: run_script("scripts/analyze_frontier_main_mixed_20260928.py", [],
                           {"frontier_mixed_populations_resources_boxplots.png": "figures/frontier_mixed_populations_resources_boxplots.png"}, gemini="Gemini-3.1"),
        lambda: run_script("scripts/analyze_frontier_update_20260928.py", [],
                           {"frontier_update_set_resources_boxplots.png": "figures/frontier_update_set_resources_boxplots.png"}, gemini="Gemini-3.1"),
        lambda: run_script("scripts/analyze_frontier_defector_populations_20261002.py", [],
                           {"resources_boxplots.png": "figures/Frontier_defector_run/resources_boxplots.png"}, gemini="Gemini-3.1"),
        lambda: run_script("analyses/frontier_defector_population_figures.py", [],
                           {"fig8_populations_send_per_round.png": "figures/Frontier_defector_run/fig8_populations_send_per_round.png"}, gemini="Gemini-3.1"),
        lambda: run_script("analyses/plot_slide678_rerun.py", ["--out-dir", str(SCRATCH)],
                           {"slide678_cell_means.png": "figures/ablation_run_8-agent.png"}, drop_axes_titles=True, models=False),
        lambda: run_script("analyses/plot_slide678_rerun.py",
                           ["--run-root", str(ROOT / "data/json/noise_experiments/slide678_dyad_rerun_20260917"),
                            "--out-dir", str(SCRATCH), "--ceiling", "150",
                            "--population", "2-agent fixed dyad", "--output-stem", "slide678_dyad_cell_means"],
                           {"slide678_dyad_cell_means.png": "figures/ablation_run_2-agent.png"}, drop_axes_titles=True, models=False),
        lambda: run_script("analyses/myth_replay_plots.py", [],
                           {"replay_ablation_style.png": "figures/replay_ablation/replay_ablation_style.png"}),
        lambda: run_marker_rates({"cross_family_marker_rates.png": "figures/mid_tier_n10/cross_family_marker_rates.png"}),
        # Main-text figures and the figures this branch draws itself.
        lambda: run_script("analyses/dyad_resources_boxplots_clean.py", [],
                           {"dyad_resources_boxplots.png": "figures/mid_tier_n10/mid_tier_n10_resources_boxplots.png"}),
        lambda: run_script("analyses/population_ladder_nine_panels.py", [],
                           {"population_resources_boxplots_two_rows.png": "figures/mid_tier_n10/population_resources_boxplots_nine_panels.png",
                            "population_send_per_round_two_rows.png": "figures/mid_tier_n10/population_send_per_round_nine_panels.png",
                            "population_return_per_round_two_rows.png": "figures/mid_tier_n10/fig8_populations_return_per_round.png"}),
        lambda: run_script("analyses/frontier_mixed_dyads_per_round.py", [],
                           {"frontier_mixed_dyads_send_per_round.png": "figures/frontier_mixed_dyads_send_per_round.png",
                            "frontier_mixed_dyads_send_per_round_split.png": "figures/frontier_mixed_dyads_send_per_round_split.png"},
                           gemini="Gemini-3.1"),
        lambda: run_script("analyses/paper_figures_extra.py", [],
                           {"moral_label_shares_populations.pdf": "figures/mid_tier_n10/moral_label_shares_populations.pdf",
                            "moral_label_shares_notitle.png": "figures/mid_tier_n10/moral_label_shares_notitle.png",
                            "frontier_mixed_dyads_resources_boxplots.png": "figures/frontier_mixed_dyads_resources_boxplots.png"}),
        lambda: run_script("analyses/plot_word_adoption_panel.py", [],
                           {"language_reuse_words_only.png": "figures/mid_tier_n10/language_reuse_words_only.png"}),
        lambda: run_script("analyses/linguistic_uptake.py", ["--replot"],
                           {"language_reuse_shown_vs_unseen.png": "figures/mid_tier_n10/language_reuse_shown_vs_unseen.png"},
                           env={"LINGUISTIC_DATASET": "september_n10"}),
    ] + [
        (lambda size=size, axes=axes: run_script(
            "analyses/plot_myth_trajectory_panel.py", ["--size", str(size), "--axes", axes],
            {f"myth_map_trajectories_{size}agent_pooled.png": f"figures/mid_tier_n10/myth_map_trajectories_{size}agent_pooled.png",
             f"myth_time_map_{size}agent_pooled.png": f"figures/mid_tier_n10/myth_time_map_{size}agent_pooled.png"},
            env={"LINGUISTIC_DATASET": "september_n10"}))
        for size in (8, 2) for axes in ("pca", "time")
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
