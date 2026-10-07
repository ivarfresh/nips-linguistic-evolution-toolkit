#!/usr/bin/env python3
"""Paper Figure 2 (main-model dyads) without the plot title, subtitles or footer.

Same data and colours as scripts/analyze_mixed_model_dyads.py plot_boxplot_grid at n = 10
(docs/figures/mixed_vs_average_n10_20261001/dyads/resources_boxplots.png); styled like the
nine-panel Figure 3 (analyses/population_ladder_nine_panels.py). Edward, 6 Oct meeting:
remove plot titles.

Input:  docs/figures/mixed_vs_average_n10_20261001/dyad_decisions.csv (analyses/table1_n10.py)
Output: docs/figures/paper_figures_clean_20261007/dyad_resources_boxplots.{png,pdf}
No API calls.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses._shared import configure_matplotlib  # noqa: E402
from analyses.population_ladder_nine_panels import draw_box  # noqa: E402

DECISIONS = ROOT / "docs/figures/mixed_vs_average_n10_20261001/dyad_decisions.csv"
OUTPUT = ROOT / "docs/figures/paper_figures_clean_20261007"
EXPECTED_RUNS = 180  # 6 pairings x 3 task orders x 10
ROWS = [
    ("Single model", ["Sonnet+Sonnet", "GPT+GPT", "Gemini+Gemini"]),
    ("Mixed", ["Sonnet+GPT", "Sonnet+Gemini", "Gemini+GPT"]),
]


def main() -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    decisions = pd.read_csv(DECISIONS)
    if decisions["path"].nunique() != EXPECTED_RUNS:
        raise SystemExit("Unexpected run count in the n = 10 dyad table")
    finals = decisions[decisions["round"] == 10]
    # Resources per agent = the pair's total balance after round 10, halved (as the original figure).
    per_run = finals.assign(final_balance=finals["total_balance"] / 2)[["composition", "task_order", "path", "final_balance"]]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 7.6), sharey=True)
    for axrow, (row_label, compositions) in zip(axes, ROWS):
        for ax, composition in zip(axrow, compositions):
            draw_box(ax, per_run, composition)
            ax.set_title(composition.replace("+", " + "), fontsize=11, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
        axrow[0].set_ylabel(row_label, fontsize=11, fontweight="bold", labelpad=10)
    fig.supylabel("Resources per agent", fontsize=11.5, x=0.004)
    fig.tight_layout(rect=(0.015, 0, 1, 1), h_pad=1.6, w_pad=0.8)
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT / f"dyad_resources_boxplots.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
