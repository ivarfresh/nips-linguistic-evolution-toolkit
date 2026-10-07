#!/usr/bin/env python3
"""Paper figures that had no script in the repo, redrawn from saved tables in their current layout.

  moral_label_shares_populations.pdf  paper Figure 6: moral label shares per round, eight-agent
                                      populations, single-model (top) vs mixed (bottom), one
                                      column per model; both myth task orders pooled.
  moral_label_shares_notitle.png      supplement: the same for every family x setting.
  frontier_mixed_dyads_resources_boxplots.png
                                      paper Figure 4a: final resources in mixed frontier dyads
                                      (the 54 audited runs of scripts/analyze_frontier_main_mixed_20260928.py).

Model names follow the paper's figure convention (family-version: Sonnet-4.5, GPT-5-Nano, ...).
Inputs: docs/figures/linguistic_analysis_n10_20261001/moral_label_shares.csv and the audited
frontier finals. Run through analyses/render_paper_figures_clean.py. No API calls.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHARES = ROOT / "docs/figures/linguistic_analysis_n10_20261001/moral_label_shares.csv"
OUTPUT = ROOT / "data/analysis/_clean_render_scratch"  # the copies the paper uses come via render_paper_figures_clean.py
LABELS = ["be generous", "be fair", "be cautious"]
COLORS = ["#4a9a5c", "#d9aa3a", "#a83236"]  # as the current Figure 6
# Figure 4a y-axis range. Per-run dots span 50.0-75.0 (54 runs), so (45, 80) shows every dot.
# Override for a one-off render with FIG4A_YLIM="lo,hi" in the environment.
FIG4A_YLIM = (45, 80)  # paper choice, 2026-10-08
NAME = {"Gemini": "Gemini-3.7", "Sonnet": "Sonnet-4.5", "GPT": "GPT-5-Nano"}
SETTING = {"2-agent homogeneous": "single-model dyads", "2-agent mixed": "mixed dyads",
           "8-agent homogeneous": "single-model populations", "8-agent mixed": "mixed populations"}


def pooled_shares() -> pd.DataFrame:
    """Pool the two myth task orders, weighting each by its number of labelled myths."""
    d = pd.read_csv(SHARES)
    cols = [f"share_{l}" for l in LABELS]
    w = d[cols].mul(d["n_myths"], axis=0).assign(setting=d["setting"], family=d["family"], round=d["round"], n=d["n_myths"])
    g = w.groupby(["setting", "family", "round"], as_index=False).sum(numeric_only=True)
    g[cols] = g[cols].div(g["n"], axis=0)
    return g


def stacked(ax, rows: pd.DataFrame, edge: str = "white") -> None:
    rows = rows.sort_values("round")
    bottom = np.zeros(len(rows))
    for label, color in zip(LABELS, COLORS):
        v = rows[f"share_{label}"].to_numpy()
        ax.bar(rows["round"], v, bottom=bottom, color=color, edgecolor=edge, linewidth=0.8, width=0.85, label=label)
        bottom += v
    ax.set_ylim(0, 1)
    ax.set_xticks([1, 5, 10])
    ax.spines[["top", "right"]].set_visible(False)


def figure6(g: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 13}):
        fig, axes = plt.subplots(2, 3, figsize=(8, 6), sharex=True, sharey=True)
        for r, (setting, row_label) in enumerate([("8-agent homogeneous", "single-model"), ("8-agent mixed", "mixed")]):
            for c, family in enumerate(["Gemini", "Sonnet", "GPT"]):
                ax = axes[r, c]
                stacked(ax, g[(g["setting"] == setting) & (g["family"] == family)])
                if r == 0:
                    ax.set_title(NAME[family], fontsize=16)
                if r == 1:
                    ax.set_xlabel("round", fontsize=15)
            axes[r, 0].set_ylabel(row_label, fontsize=15)
            axes[r, 0].set_yticks([0, 0.5, 1], ["0", ".5", "1"])
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=15, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(OUTPUT / "moral_label_shares_populations.pdf", bbox_inches="tight")
        plt.close(fig)


def supplement(g: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from analyses._shared import configure_matplotlib
    configure_matplotlib()
    settings = ["2-agent homogeneous", "2-agent mixed", "8-agent homogeneous", "8-agent mixed"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True, sharey=True)
    for r, family in enumerate(["Sonnet", "Gemini", "GPT"]):
        for c, setting in enumerate(settings):
            ax = axes[r, c]
            stacked(ax, g[(g["setting"] == setting) & (g["family"] == family)])
            ax.set_title(f"{NAME[family]}, {SETTING[setting]}", fontsize=10)
            ax.set_xticks([2, 4, 6, 8, 10])
            if r == 2:
                ax.set_xlabel("round")
        axes[r, 0].set_ylabel("share of myths")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / "moral_label_shares_notitle.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig4a_ylim() -> tuple[float, float]:
    raw = os.environ.get("FIG4A_YLIM")
    if not raw:
        return FIG4A_YLIM
    lo, hi = (float(x) for x in raw.split(","))
    return lo, hi


def figure4a() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scripts.analyze_frontier_main_mixed_20260928 import BOX_COLORS, DOT_COLORS, TASK_ORDERS, ORDER_LABELS, load
    df, _, _ = load()
    per_run = (df[(df["setting"] == "mixed") & (df["num_agents"] == 2)]
               .groupby(["panel", "task_order", "path"], as_index=False)["resources"].mean())
    if per_run["path"].nunique() != 54:
        raise SystemExit("expected 54 mixed frontier dyads")
    ylim = fig4a_ylim()
    lo, hi = per_run["resources"].min(), per_run["resources"].max()
    print(f"Figure 4a per-run dots: min {lo:.2f}, max {hi:.2f}; y-axis {ylim[0]:g}-{ylim[1]:g}")
    if lo < ylim[0] or hi > ylim[1]:
        raise SystemExit(f"Figure 4a y-axis {ylim} would hide dots (range {lo:.2f}-{hi:.2f})")
    panels = [("Opus 5 + Sol", "Opus-5 + GPT-5.6-Sol"), ("Opus 5 + Gemini", "Opus-5 + Gemini-3.1"),
              ("Gemini + Sol", "Gemini-3.1 + GPT-5.6-Sol")]
    fig, axes = plt.subplots(1, 3, figsize=(5.2, 4.4), sharey=True)
    for ax, (panel, title) in zip(axes, panels):
        for pos, task_order in enumerate(TASK_ORDERS, 1):
            v = np.sort(per_run[(per_run["panel"] == panel) & (per_run["task_order"] == task_order)]["resources"].to_numpy())
            ax.boxplot(v, positions=[pos], widths=.6, patch_artist=True, showfliers=False, whis=1.5,
                       boxprops=dict(facecolor=BOX_COLORS[pos - 1], edgecolor="#666666", linewidth=1.2),
                       medianprops=dict(color="#222222", linewidth=2), whiskerprops=dict(color="#666666", linewidth=1.2),
                       capprops=dict(color="#666666", linewidth=1.2))
            ax.scatter(pos + np.linspace(-.12, .12, len(v)), v, s=22, c=DOT_COLORS[pos - 1], edgecolors="white", linewidths=.5, zorder=3)
        ax.set_title(title.replace(" + ", " +\n"), fontsize=9.5, fontweight="bold")
        ax.set_xticks([])
        ax.set_xlim(.4, 3.6)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Resources per agent")
    fig.legend(handles=[Patch(facecolor=BOX_COLORS[i], edgecolor="#666666", label=ORDER_LABELS[i]) for i in range(3)],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6)
    fig.savefig(OUTPUT / "frontier_mixed_dyads_resources_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    g = pooled_shares()
    figure6(g)
    supplement(g)
    figure4a()
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
