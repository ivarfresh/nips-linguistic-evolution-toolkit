#!/usr/bin/env python3
"""Figure for the self-inspiration sentence ablation (2026-11-27), in the paper's formats.

Left, in the format of paper Figure 5a (analyses/plot_word_adoption_panel.py): share of the
shown myth's new-to-the-reader words that the reader adopts (dark) against a comparable unseen
myth (light), with and without the sentence, for single-model dyads (Sonnet-4.5) and mixed
dyads (Sonnet-4.5 with GPT-5-Nano or Gemini-3.7; both readers). Bars: means over runs with 95%
CI; above each pair, the Holm-corrected paired Wilcoxon test and the number of runs in which
the shown myth scores higher.
Right, in the format of paper Figure 3 (analyses/population_ladder_nine_panels.py): final
resources per agent, one box per task order (same colours), with (filled) and without
(hatched) the sentence; dots are runs.
Reads per_run.csv and mixed_per_run.csv from analyses/no_self_line_dyads.py and
analyses/no_self_line_mixed_dyads.py.

  python3 analyses/plot_no_self_line_dyads.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses._shared import configure_matplotlib  # noqa: E402
from analyses.linguistic_uptake import holm, stars  # noqa: E402

OUT = ROOT / "docs/research/no_self_line_dyads_20261127"
SHOWN, UNSEEN = "#444444", "#bbbbbb"  # Figure 5a
BOX = {"game_myth": "#e99675", "myth_game": "#72b6a1"}  # Figure 3 (scripts/analyze_mixed_model_populations.py)
DOT = {"game_myth": "#fc8d62", "myth_game": "#66c2a5"}
ORDER_LABEL = {"game_myth": "Game → Myth", "myth_game": "Myth → Game"}
ARM_LABEL = {"line": "with\nsentence", "no_line": "without\nsentence"}


def main() -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    single = pd.read_csv(OUT / "per_run.csv").assign(pair="sonnet+sonnet")
    mixed = pd.read_csv(OUT / "mixed_per_run.csv")
    mixed["unseen_uptake"] = mixed["partner_uptake"] - mixed["borrowing"]
    keys = ["pair", "task_order", "arm", "replicate"]
    mixed_runs = mixed.groupby(keys)[["partner_uptake", "unseen_uptake", "resources"]].mean().reset_index()
    runs = pd.concat([single[keys + ["partner_uptake", "unseen_uptake", "resources"]], mixed_runs])

    fig = plt.figure(figsize=(13.5, 4.15))
    grid = fig.add_gridspec(1, 5, width_ratios=[2.6, 0.32, 1, 1, 1], wspace=0.1)

    # ---- left: Figure 5a format
    ax = fig.add_subplot(grid[0, 0])
    groups = [("single-model\ndyads", runs["pair"] == "sonnet+sonnet"), ("mixed dyads\nshown other\nfamily", runs["pair"] != "sonnet+sonnet")]
    cells = [(f"{name}\n{ARM_LABEL[arm].replace(chr(10), ' ')}", runs[mask & (runs["arm"] == arm)]) for name, mask in groups for arm in ("line", "no_line")]
    p = holm(np.array([stats.wilcoxon(g["partner_uptake"] - g["unseen_uptake"]).pvalue for _, g in cells]))
    for x, (_, g) in enumerate(cells):
        top = 0
        for dx, col, color in ((-0.18, "partner_uptake", SHOWN), (0.18, "unseen_uptake", UNSEEN)):
            v = 100 * g[col]
            ci = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            ax.bar(x + dx, v.mean(), width=0.34, color=color, yerr=ci, capsize=3, error_kw=dict(lw=1, ecolor="#666666"))
            top = max(top, v.mean() + ci)
        higher = int((g["partner_uptake"] > g["unseen_uptake"]).sum())
        ax.annotate(f"{stars(p[x])}\n{higher}/{len(g)}", (x, top), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(cells)), [name for name, _ in cells], fontsize=8)
    ax.set_ylabel("% of new words adopted")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.legend(handles=[Patch(color=SHOWN, label="shown myth"), Patch(color=UNSEEN, label="unseen myth")], loc="upper right", fontsize=8)
    ax.text(-0.1, 1.03, "a", transform=ax.transAxes, fontweight="bold", fontsize=12)

    # ---- right: Figure 3 format
    first = None
    titles = {"sonnet+sonnet": "2 Sonnet-4.5", "sonnet+gpt": "1 Sonnet-4.5 +\n1 GPT-5-Nano", "sonnet+gemini": "1 Sonnet-4.5 +\n1 Gemini-3.7"}
    for i, (pair, title) in enumerate(titles.items()):
        bx = fig.add_subplot(grid[0, i + 2], sharey=first)
        first = first or bx
        pos = 0
        for order in ("game_myth", "myth_game"):
            for arm in ("line", "no_line"):
                pos += 1
                v = np.sort(runs[(runs["pair"] == pair) & (runs["task_order"] == order) & (runs["arm"] == arm)]["resources"].to_numpy())
                assert len(v) == 5, (pair, order, arm)
                bx.boxplot(v, positions=[pos], widths=.52, patch_artist=True, showfliers=False, whis=1.5,
                           boxprops=dict(facecolor=BOX[order], edgecolor="#666666", hatch=None if arm == "line" else "////"),
                           medianprops=dict(color="#222222", linewidth=1.6),
                           whiskerprops=dict(color="#666666"), capprops=dict(color="#666666"))
                bx.scatter(pos + np.linspace(-.1, .1, len(v)), v, s=26, c=DOT[order], edgecolors="white", linewidths=.6, zorder=3)
        bx.set_xticks([1, 2, 3, 4], ["with", "without"] * 2, fontsize=8)
        bx.set_xlim(.5, 4.5)
        bx.set_ylim(0, 80)
        bx.grid(axis="y", alpha=.22)
        bx.set_title(title, fontsize=10.5, fontweight="bold")
        bx.spines[["top", "right"]].set_visible(False)
        if i:
            bx.tick_params(labelleft=False)
        else:
            bx.set_ylabel("Resources per agent", fontsize=11.5)
            bx.text(-0.36, 1.03, "b", transform=bx.transAxes, fontweight="bold", fontsize=12)
    handles = [Patch(facecolor=BOX[o], edgecolor="#666666", label=ORDER_LABEL[o]) for o in BOX]
    handles += [Patch(facecolor="white", edgecolor="#666666", label="with the sentence"),
                Patch(facecolor="white", edgecolor="#666666", hatch="////", label="without the sentence")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.7, -0.09))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"no_self_line_transmission_cooperation.{ext}", dpi=200, bbox_inches="tight")
    print(OUT / "no_self_line_transmission_cooperation.png")


if __name__ == "__main__":
    main()
