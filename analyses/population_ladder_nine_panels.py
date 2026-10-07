#!/usr/bin/env python3
"""Paper Figure 3 and its per-round companion with nine panels instead of ten.

The ten-panel grids (scripts/analyze_mixed_model_populations.py plot_boxplot_grid and
analyses/mixed_model_cooperation_per_round.py) show the all-GPT population twice: at the
start of the Gemini-among-GPT row and at the end of the GPT-among-Sonnet row. Edward
(6 Oct meeting) asked for nine panels and no plot titles. Same data, same colours.

Two layouts, to choose from:
  *_two_rows.png  the existing two rows; the second row (no duplicate 8 GPT) is centred
  *_one_row.png   one row ordered by number of GPT agents: 8 Gemini ... 8 GPT ... 8 Sonnet
  *_grid.png      3 x 3: single-model populations, then one row per ladder

Inputs (n = 10 tables written by analyses/table1_n10.py):
  docs/figures/mixed_vs_average_n10_20261001/population_agent_finals.csv
  docs/figures/mixed_vs_average_n10_20261001/population_games.csv
Outputs: docs/figures/population_ladder_nine_panels_20261006/
No API calls.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses._shared import configure_matplotlib  # noqa: E402
import analyses.mixed_model_cooperation_per_round as pr  # noqa: E402
import scripts.analyze_mixed_model_populations as po  # noqa: E402

N10 = ROOT / "docs/figures/mixed_vs_average_n10_20261001"
OUTPUT = ROOT / "docs/figures/population_ladder_nine_panels_20261006"
EXPECTED_RUNS = 270  # 9 compositions x 3 task orders x 10

TWO_ROWS = [
    ("Gemini among GPT", ["8 GPT", "1 Gemini + 7 GPT", "2 Gemini + 6 GPT", "4 Gemini + 4 GPT", "8 Gemini"]),
    ("GPT among Sonnet", ["8 Sonnet", "1 GPT + 7 Sonnet", "2 GPT + 6 Sonnet", "4 GPT + 4 Sonnet", None]),
]
GRID = [
    ("Single model", ["8 GPT", "8 Gemini", "8 Sonnet"]),
    ("Gemini among GPT", ["1 Gemini + 7 GPT", "2 Gemini + 6 GPT", "4 Gemini + 4 GPT"]),
    ("GPT among Sonnet", ["1 GPT + 7 Sonnet", "2 GPT + 6 Sonnet", "4 GPT + 4 Sonnet"]),
]
ONE_ROW = [(None, ["8 Gemini", "4 Gemini + 4 GPT", "2 Gemini + 6 GPT", "1 Gemini + 7 GPT", "8 GPT",
                   "4 GPT + 4 Sonnet", "2 GPT + 6 Sonnet", "1 GPT + 7 Sonnet", "8 Sonnet"])]
SHORT_ORDERS = ["Game\nonly", "Game\n→ Myth", "Myth\n→ Game"]


def label(composition: str) -> str:
    return composition.replace(" + ", " +\n") if "+" in composition else composition


def draw_box(ax, per_run: pd.DataFrame, composition: str) -> None:
    for pos, task_order in enumerate(po.TASK_ORDERS, 1):
        v = np.sort(per_run[(per_run["composition"] == composition) & (per_run["task_order"] == task_order)]["final_balance"].to_numpy())
        ax.boxplot(v, positions=[pos], widths=.52, patch_artist=True, showfliers=False, whis=1.5,
                   boxprops=dict(facecolor=po.BOX_COLORS[pos - 1], edgecolor="#666666"),
                   medianprops=dict(color="#222222", linewidth=1.6),
                   whiskerprops=dict(color="#666666"), capprops=dict(color="#666666"))
        ax.scatter(pos + np.linspace(-.1, .1, len(v)), v, s=26, c=po.DOT_COLORS[pos - 1], edgecolors="white", linewidths=.6, zorder=3)
    ax.set_xticks([1, 2, 3], SHORT_ORDERS, fontsize=8)
    ax.set_xlim(.5, 3.5)
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=.22)


def draw_round(ax, stats: pd.DataFrame, composition: str, metric: str = "send") -> None:
    d = stats[(stats["composition"] == composition) & (stats["metric"] == metric)]
    for task_order in pr.TASK_ORDERS:
        t = d[d["task_order"] == task_order].sort_values("round")
        x, mean, sd = t["round"].to_numpy(), t["mean"].to_numpy(), np.nan_to_num(t["sd"].to_numpy())
        color = pr.LINE_COLORS[task_order]
        ax.fill_between(x, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=2, marker="o", ms=3)
    if metric == "return":
        ax.axhline(1 / 3, color="#bbbbbb", linewidth=0.9, linestyle="--", zorder=0)
        ax.axhline(0.5, color="#bbbbbb", linewidth=0.9, linestyle=":", zorder=0)
    ax.set_xticks([1, 5, 10])
    ax.set_xlim(0.6, 10.4)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Round")
    ax.grid(alpha=0.2)


def plot(layout, draw, data, ylabel: str, filename: str, legend: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ncols = len(layout[0][1])
    width = 2.4 * ncols if len(layout) == 1 else 3.6 * ncols if ncols > 3 else 3.4 * ncols
    fig = plt.figure(figsize=(width, 3.6 * len(layout) + 0.4))
    # Half-column grid so a shorter row (the two-row layout's bottom row) can be centred.
    grid = fig.add_gridspec(len(layout), 2 * ncols)
    first = None
    for r, (row_label, compositions) in enumerate(layout):
        panels = [c for c in compositions if c is not None]
        offset = ncols - len(panels)
        for i, composition in enumerate(panels):
            ax = fig.add_subplot(grid[r, offset + 2 * i: offset + 2 * i + 2], sharey=first)
            first = first or ax
            draw(ax, data, composition)
            ax.set_title(label(composition), fontsize=10.5, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            if i:
                ax.tick_params(labelleft=False)
            elif row_label:
                ax.set_ylabel(row_label, fontsize=11, fontweight="bold", labelpad=10)
    fig.supylabel(ylabel, fontsize=11.5, x=0.004)
    bottom = 0.0
    if legend:
        handles = [Line2D([], [], color=pr.LINE_COLORS[t], linewidth=2, marker="o", ms=4, label=pr.ORDER_LABELS[t]) for t in pr.TASK_ORDERS]
        if "return" in filename:
            handles += [Line2D([], [], color="#bbbbbb", linestyle="--", label="1/3 = sender breaks even"),
                        Line2D([], [], color="#bbbbbb", linestyle=":", label="1/2 = gain split evenly")]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=10)
        bottom = 0.1 if len(layout) == 1 else 0.05 if len(layout) == 2 else 0.04
    fig.tight_layout(rect=(0.015, bottom, 1, 1), h_pad=1.6, w_pad=0.8)
    fig.savefig(OUTPUT / f"{filename}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    agents = pd.read_csv(N10 / "population_agent_finals.csv")
    games = pd.read_csv(N10 / "population_games.csv")
    if agents["path"].nunique() != EXPECTED_RUNS or games["path"].nunique() != EXPECTED_RUNS:
        raise SystemExit("Unexpected run counts in the n = 10 population tables")
    per_run = agents.groupby(["composition", "task_order", "path"], sort=False)["final_balance"].mean().reset_index()
    stats = pr.per_round_stats(pr.run_round_ratios(games, "population"))
    for name, layout in (("two_rows", TWO_ROWS), ("one_row", ONE_ROW), ("grid", GRID)):
        plot(layout, draw_box, per_run, "Resources per agent", f"population_resources_boxplots_{name}", legend=False)
        plot(layout, draw_round, stats, "Send fraction (sent / $5)", f"population_send_per_round_{name}", legend=True)
    plot(TWO_ROWS, lambda ax, d, c: draw_round(ax, d, c, "return"), stats, "Return ratio (returned / received)",
         "population_return_per_round_two_rows", legend=True)


if __name__ == "__main__":
    main()
