#!/usr/bin/env python3
"""Linguistic-dynamics figure: lag-1 cross-agent correlation + embedding
cosine convergence slope, per cell.

Two-panel figure. Left panel: per-cell mean lag-1 max-direction Pearson r
with bootstrap 95% CIs. Right panel: per-cell embedding cosine slope per
round with bootstrap 95% CIs. Both panels share the same row ordering.

Output: analysis/figures/linguistic_dynamics.png + .pdf
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

NOISE_ORDER = ["positive", "negative_5", "bootstrap", "deterministic_max"]
MODEL_ORDER = ["claude-sonnet-4.5", "gpt-5-nano"]
NOISE_LABELS = {
    "positive": "positive",
    "negative_5": "negative",
    "bootstrap": "bootstrap",
    "deterministic_max": "det. max",
}
MODEL_COLORS = {"claude-sonnet-4.5": "#d97706", "gpt-5-nano": "#0ea5e9"}


def build_row_label(r):
    base = f"{r['model'].split('-')[0]} | {NOISE_LABELS.get(r['noise_label'], r['noise_label'])}"
    if r["informed"]:
        base += " (inf)"
    base += f" | {r['task_order']}"
    return base


def main():
    lag = pd.read_csv(SUMMARIES / "lag_summary.csv")
    emb = pd.read_csv(SUMMARIES / "embedding_summary.csv")

    # Normalize types and sort.
    for df in (lag, emb):
        df["model_rank"] = df["model"].map(
            {m: i for i, m in enumerate(MODEL_ORDER)}
        ).fillna(99)
        df["noise_rank"] = df["noise_label"].map(
            {n: i for i, n in enumerate(NOISE_ORDER)}
        ).fillna(99)
        df["task_rank"] = df["task_order"].map(
            {"game_myth": 0, "myth_game": 1}
        ).fillna(99)

    # Use the lag summary's row ordering as canonical.
    lag = lag.sort_values(["model_rank", "noise_rank", "informed", "task_rank"])

    # Build per-row labels and align embeddings to the same order.
    lag["row_label"] = lag.apply(build_row_label, axis=1)
    emb["row_label"] = emb.apply(build_row_label, axis=1)
    emb_idx = emb.set_index("row_label")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 0.32 * len(lag) + 1.8), sharey=True,
    )

    y = np.arange(len(lag))
    colors = [MODEL_COLORS[m] for m in lag["model"]]

    # Panel 1: lag-1 max-direction Pearson r
    ax1.errorbar(
        lag["lag1_max_mean"], y,
        xerr=[
            lag["lag1_max_mean"] - lag["lag1_max_ci_lo"],
            lag["lag1_max_ci_hi"] - lag["lag1_max_mean"],
        ],
        fmt="o", ecolor="gray", capsize=2, markersize=5,
        markerfacecolor="white", markeredgewidth=1.2,
    )
    for yi, c in zip(y, colors):
        ax1.plot(lag["lag1_max_mean"].iloc[yi], yi, "o",
                 color=c, markersize=5, zorder=10)
    ax1.axvline(0, color="black", linewidth=0.6, alpha=0.5)
    ax1.axvline(0.5, color="red", linewidth=0.6, linestyle="--", alpha=0.4)
    ax1.set_yticks(y)
    ax1.set_yticklabels(lag["row_label"], fontsize=8)
    ax1.set_xlabel("Lag-1 cross-agent Pearson r (per-dyad max direction)\n95% bootstrap CI; red dashed = |r|=0.5", fontsize=9)
    ax1.set_title("(a) Cooperativity-language lag-1 correlation", fontsize=10)
    ax1.grid(axis="x", alpha=0.3)
    ax1.invert_yaxis()
    ax1.set_xlim(-0.1, 0.85)

    # Panel 2: embedding cosine slope per round
    slopes = []
    slope_lo = []
    slope_hi = []
    for label in lag["row_label"]:
        if label in emb_idx.index:
            row = emb_idx.loc[label]
            slopes.append(row["cos_between_slope_per_round"])
            slope_lo.append(row["cos_between_slope_ci_lo"])
            slope_hi.append(row["cos_between_slope_ci_hi"])
        else:
            slopes.append(np.nan)
            slope_lo.append(np.nan)
            slope_hi.append(np.nan)
    slopes = np.array(slopes)
    slope_lo = np.array(slope_lo)
    slope_hi = np.array(slope_hi)
    valid = ~np.isnan(slopes)
    ax2.errorbar(
        slopes[valid], y[valid],
        xerr=[
            (slopes - slope_lo)[valid],
            (slope_hi - slopes)[valid],
        ],
        fmt="o", ecolor="gray", capsize=2, markersize=5,
        markerfacecolor="white", markeredgewidth=1.2,
    )
    for yi in y[valid]:
        c = colors[yi]
        ax2.plot(slopes[yi], yi, "o", color=c, markersize=5, zorder=10)
    ax2.axvline(0, color="black", linewidth=0.6, alpha=0.5)
    ax2.set_xlabel("Cosine slope per round (between-agent myth convergence)\n95% bootstrap CI; >0 = convergence", fontsize=9)
    ax2.set_title("(b) Embedding-cosine convergence slope", fontsize=10)
    ax2.grid(axis="x", alpha=0.3)
    ax2.invert_yaxis()

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=MODEL_COLORS[m], markersize=8,
                   label=m.split("-")[0])
        for m in MODEL_ORDER
    ]
    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9,
    )
    fig.suptitle(
        "Linguistic dynamics inside the myth chain — robust across cells\n"
        "v4_direct_provider, N=15 seeds/cell, two models",
        fontsize=11, y=1.04,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG_DIR / "linguistic_dynamics.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(FIG_DIR / "linguistic_dynamics.pdf", bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
