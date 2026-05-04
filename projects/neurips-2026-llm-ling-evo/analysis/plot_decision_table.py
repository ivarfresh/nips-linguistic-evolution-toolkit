#!/usr/bin/env python3
"""Single decision-table heatmap figure for §3.

Rows = (model, noise_label, informed); cols = myth_task_order.
Cell colour encodes the 3x3 classification.  Cell text = Δmean and
var_ratio with brackets.

Output: figures/decision_table.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm

ANALYSIS_DIR = Path(__file__).parent
SUMMARIES = ANALYSIS_DIR / "cell_summaries"
FIG_DIR = ANALYSIS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

CLASS_ORDER = [
    "lift+consolidation", "lift", "consolidation",
    "null",
    "lift+destabilizing", "pure_noise",
    "harmful", "harmful-lock-in", "destabilizing",
    "missing",
]

CLASS_COLORS = {
    "lift+consolidation":  "#1a9850",
    "lift":                "#66bd63",
    "consolidation":       "#a6d96a",
    "null":                "#f7f7f7",
    "lift+destabilizing":  "#fee08b",
    "pure_noise":          "#fdae61",
    "harmful":             "#f46d43",
    "harmful-lock-in":     "#d73027",
    "destabilizing":       "#a50026",
    "missing":             "#cccccc",
}


def main():
    deltas = pd.read_csv(SUMMARIES / "deltas.csv")
    deltas["classification"] = pd.read_csv(
        SUMMARIES / "deltas.csv",
        dtype={"classification": str},
        keep_default_na=False,
    )["classification"].values

    # Build a row label per (model, noise_label, informed).
    deltas["row_label"] = deltas.apply(
        lambda r: f"{r['model'].split('-')[0]}/{r['noise_label']}"
                  + (" (inf)" if r["informed"] else ""),
        axis=1,
    )
    # Stable ordering: by model first, then by noise type.
    noise_order = ["positive", "negative_5", "bootstrap", "deterministic_max"]
    model_order = ["claude-sonnet-4.5", "gpt-5-nano"]
    deltas["model_rank"] = deltas["model"].map(
        {m: i for i, m in enumerate(model_order)}
    ).fillna(99)
    deltas["noise_rank"] = deltas["noise_label"].map(
        {n: i for i, n in enumerate(noise_order)}
    ).fillna(99)
    row_keys = (
        deltas[["row_label", "model_rank", "noise_rank", "informed"]]
        .drop_duplicates()
        .sort_values(["model_rank", "noise_rank", "informed"])
    )
    row_labels = row_keys["row_label"].tolist()
    col_labels = ["game→myth", "myth→game"]
    col_map = {"game_myth": 0, "myth_game": 1}

    # Build matrices: classification (string), and annotation text.
    n_rows, n_cols = len(row_labels), len(col_labels)
    classes = np.full((n_rows, n_cols), "missing", dtype=object)
    annot = np.full((n_rows, n_cols), "—", dtype=object)
    for _, r in deltas.iterrows():
        if r["row_label"] not in row_labels:
            continue
        if r["myth_task_order"] not in col_map:
            continue
        i = row_labels.index(r["row_label"])
        j = col_map[r["myth_task_order"]]
        classes[i, j] = r["classification"]
        dm = f"Δμ={r['delta_mean']:+.1f}\n[{r['delta_mean_ci_lo']:+.1f},{r['delta_mean_ci_hi']:+.1f}]"
        if pd.notna(r["var_ratio_myth_over_game"]):
            vr = (
                f"\nσ²r={r['var_ratio_myth_over_game']:.2f}"
                f"\n[{r['var_ratio_ci_lo']:.2f},{r['var_ratio_ci_hi']:.2f}]"
            )
        else:
            vr = "\nσ²r=—"
        annot[i, j] = dm + vr

    # Numeric grid for matplotlib.
    cls_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    Z = np.array(
        [[cls_idx.get(classes[i, j], len(CLASS_ORDER) - 1)
          for j in range(n_cols)] for i in range(n_rows)],
        dtype=float,
    )

    cmap = ListedColormap([CLASS_COLORS[c] for c in CLASS_ORDER])
    norm = BoundaryNorm(np.arange(len(CLASS_ORDER) + 1) - 0.5, len(CLASS_ORDER))

    fig, ax = plt.subplots(figsize=(7.5, 0.7 * n_rows + 1.5))
    ax.imshow(Z, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(
                j, i, annot[i, j],
                ha="center", va="center", fontsize=7,
                color=("black" if classes[i, j] in
                       {"null", "lift", "consolidation", "lift+destabilizing",
                        "pure_noise"}
                       else "white"),
            )

    ax.set_title(
        "Myth effect by (model × noise × myth-order)\n"
        "v4_direct_provider — N=15 seeds/cell — bootstrap 95% CIs",
        fontsize=11,
    )
    ax.set_xlabel("Myth task order", fontsize=10)

    # Legend.
    handles = []
    for c in CLASS_ORDER:
        if c == "missing":
            continue
        handles.append(
            plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[c], label=c)
        )
    ax.legend(
        handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=8, frameon=False,
    )

    plt.tight_layout()
    out = FIG_DIR / "decision_table.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(FIG_DIR / "decision_table.pdf", bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
