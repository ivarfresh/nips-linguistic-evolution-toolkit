#!/usr/bin/env python3
"""Variance-reduction summary figure.

Per (model x noise x informed x myth-order) cell, show the variance
ratio (myth-present / game-only) with bootstrap 95% CIs on a log axis.
A ratio < 1 means myth-writing reduces across-seed variance
(consolidation); > 1 means it increases variance (destabilisation).

Output: analysis/figures/variance_ratios.png + .pdf
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).parent
SUMMARIES = ANALYSIS_DIR / "cell_summaries"
FIG_DIR = ANALYSIS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

MODEL_ORDER = ["claude-sonnet-4.5", "gpt-5-nano"]
NOISE_ORDER = ["positive", "negative_5", "bootstrap", "deterministic_max"]
NOISE_LABELS = {
    "positive": "positive",
    "negative_5": "negative",
    "bootstrap": "bootstrap",
    "deterministic_max": "det. max",
}
MODEL_COLORS = {"claude-sonnet-4.5": "#d97706", "gpt-5-nano": "#0ea5e9"}


def main():
    deltas = pd.read_csv(SUMMARIES / "deltas.csv")
    classifications = pd.read_csv(
        SUMMARIES / "deltas.csv",
        dtype={"classification": str},
        keep_default_na=False,
    )["classification"]
    deltas["classification"] = classifications.values

    deltas["model_rank"] = deltas["model"].map(
        {m: i for i, m in enumerate(MODEL_ORDER)}
    ).fillna(99)
    deltas["noise_rank"] = deltas["noise_label"].map(
        {n: i for i, n in enumerate(NOISE_ORDER)}
    ).fillna(99)
    deltas["task_rank"] = deltas["myth_task_order"].map(
        {"game_myth": 0, "myth_game": 1}
    ).fillna(99)
    deltas = deltas.sort_values(["model_rank", "noise_rank", "informed", "task_rank"])

    # Drop missing/zero entries.
    deltas = deltas[deltas["var_ratio_myth_over_game"] > 0].copy()
    deltas = deltas[~deltas["var_ratio_ci_lo"].isna()]

    deltas["row_label"] = deltas.apply(
        lambda r: f"{r['model'].split('-')[0]} | "
                  f"{NOISE_LABELS.get(r['noise_label'], r['noise_label'])}"
                  + (" (inf)" if r["informed"] else "")
                  + f" | {r['myth_task_order']}",
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(deltas) + 1.5))

    y = np.arange(len(deltas))
    vr = deltas["var_ratio_myth_over_game"].values
    lo = deltas["var_ratio_ci_lo"].values
    hi = deltas["var_ratio_ci_hi"].values

    # Color by classification family
    def class_color(c):
        if c in ("lift+consolidation", "consolidation"):
            return "#16a34a"
        if c == "lift":
            return "#a3a3a3"
        if c == "null":
            return "#737373"
        if c in ("harmful", "harmful-lock-in"):
            return "#ea580c"
        if c in ("destabilizing", "lift+destabilizing"):
            return "#dc2626"
        return "#9ca3af"

    colors = [class_color(c) for c in deltas["classification"]]

    # Shade ratio < 1 = consolidation region
    ax.axvspan(1e-3, 1, alpha=0.08, color="#16a34a", zorder=0)
    ax.axvline(1, color="black", linewidth=0.8, zorder=1)
    ax.axvspan(1, 100, alpha=0.05, color="#dc2626", zorder=0)

    for yi, (m, l, h, c) in enumerate(zip(vr, lo, hi, colors)):
        ax.plot([l, h], [yi, yi], color=c, linewidth=1.5, alpha=0.7)
        ax.plot(m, yi, "o", color=c, markersize=6, markeredgecolor="white",
                markeredgewidth=0.8, zorder=10)

    ax.set_xscale("log")
    ax.set_xlim(0.005, 30)
    ax.set_yticks(y)
    ax.set_yticklabels(deltas["row_label"], fontsize=8)
    ax.set_xlabel("Variance ratio (myth-present / game-only) — log scale\n"
                  "<1 = consolidation; >1 = destabilization (95% bootstrap CI)",
                  fontsize=9)
    ax.set_title("Across-seed variance ratio per cell\n"
                 "v4_direct_provider, N=15 seeds/cell",
                 fontsize=11)
    ax.grid(axis="x", which="major", alpha=0.3)
    ax.invert_yaxis()

    # Annotate consolidation/destabilization halves
    ax.text(0.05, -0.5, "← consolidation (variance ↓)", fontsize=9,
            color="#16a34a", fontweight="bold")
    ax.text(3, -0.5, "destabilization (variance ↑) →", fontsize=9,
            color="#dc2626", fontweight="bold")

    # Legend for classification colors
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#16a34a", markersize=8,
                   label="lift+consolidation / consolidation"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#737373", markersize=8,
                   label="null / lift"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#ea580c", markersize=8,
                   label="harmful"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#dc2626", markersize=8,
                   label="destabilizing"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True)

    plt.tight_layout()
    out = FIG_DIR / "variance_ratios.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(FIG_DIR / "variance_ratios.pdf", bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
