#!/usr/bin/env python3
"""Paper Figure 5b: the myth map at round 10, single-model vs mixed populations.

One dot is one round-10 myth, placed on the same PCA map as myth_convergence_map.py
(all 300 n = 10 myth runs) and coloured by model family. Dashed outlines show where
each family's round-1 myths sat in the same runs, so the panel shows change, not
only the end state. One cell only, never pooled: 8-agent populations, Myth → Game
(the main task order); the other cells are in trajectories_*.png and
family_time_map_*.png. Numbers to cite come from the 768-d tables in the same folder
(family_separation.csv, significance.csv), not from this 2-D picture. No API calls.

    LINGUISTIC_DATASET=september_n10 python3 analyses/plot_myth_map_panel.py
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses import linguistic_datasets  # noqa: E402
from analyses.myth_convergence_map import N10_OUT, Map, load, write_provenance  # noqa: E402  (also sets the matplotlib style)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SIZE, TASK_ORDER = 8, "myth_game"
FIRST, LAST = 1, 10


def main() -> None:
    ds = linguistic_datasets.get("september_n10")
    myths, emb = load(ds)
    m = Map(myths, emb)
    cell = myths[(myths["size"] == SIZE) & (myths.task_order == TASK_ORDER)]
    shown = cell[cell["round"].isin([FIRST, LAST])]
    # zoom to this cell's myths; the outermost 0.2% on each side fall outside the frame
    xr = (shown.x.quantile(0.002) - 0.04, shown.x.quantile(0.998) + 0.04)
    yr = (shown.y.quantile(0.002) - 0.04, shown.y.quantile(0.998) + 0.04)

    fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.6), sharex=True, sharey=True)
    counts = {}
    for ax, mixed, letter in zip(axs, (False, True), ("single-model", "mixed")):
        runs = cell[cell.mixed == mixed]
        for fam in ds.families:
            f = runs[runs.family == fam]
            if f.empty:
                continue
            color = ds.colors[fam]
            first, last = f[f["round"] == FIRST], f[f["round"] == LAST]
            ax.contour(m.gx, m.gy, m.density(first), levels=[m.density(first).max() * 0.35],
                       colors=[color], linewidths=1.1, linestyles="--")
            ax.scatter(last.x, last.y, s=9, color=color, alpha=0.6, lw=0)
            ax.contour(m.gx, m.gy, m.density(last), levels=[m.density(last).max() * 0.35],
                       colors=[color], linewidths=1.4)
            counts.setdefault(fam, []).append(len(last))  # per panel; quoted in the caption, not drawn
        ax.set_title(f"{letter.capitalize()} populations ({runs.run_id.nunique()} runs)", fontsize=9, pad=3)
        ax.set_xlim(*xr)
        ax.set_ylim(*yr)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(m.xlabel, fontsize=9)
    axs[0].set_ylabel(m.ylabel, fontsize=9)
    handles = [Line2D([], [], marker="o", ls="", color=ds.colors[fam],
                      label=fam) for fam in counts]
    handles += [Line2D([], [], color="#555555", lw=1.4, label="round 10"),
                Line2D([], [], color="#555555", lw=1.1, ls="--", label="round 1")]
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="lower center", ncol=5,
               handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(N10_OUT / "myth_map_round10_8agent_myth_game.png", dpi=300)
    plt.close(fig)
    print(f"wrote {N10_OUT / 'myth_map_round10_8agent_myth_game.png'}; round-10 myths (single, mixed): {counts}")
    write_provenance(ds, N10_OUT)  # the folder's provenance.json hashes this figure too


if __name__ == "__main__":
    main()
