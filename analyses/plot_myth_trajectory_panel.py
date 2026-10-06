#!/usr/bin/env python3
"""Paper Figure 5b: how each family's myths move over ten rounds, single-model vs mixed populations.

The trajectory view of myth_convergence_map.py (trajectories_8agent.png), on the
same PCA map of all 300 n = 10 myth runs, with two changes for the paper: the two
task orders are pooled into one panel per mixing condition, and the background is
coloured by family (where that family's round-10 myths end up) instead of one blue
density. Line = family average per round, light -> dark = round 1 -> 10; big dots =
rounds 1, 5, 10; small dots = each run's family average at round 10.
8-agent populations only. The task-order split is in trajectories_8agent.png;
numbers to cite are the 768-d ones in significance.csv. No API calls.

    LINGUISTIC_DATASET=september_n10 python3 analyses/plot_myth_trajectory_panel.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses import linguistic_datasets  # noqa: E402
from analyses.myth_convergence_map import (MARK_ROUNDS, N10_OUT, ROUNDS, Map, load, shade,  # noqa: E402
                                           write_provenance)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.interpolate import make_interp_spline  # noqa: E402

SIZE = 8
OUT_NAME = "myth_map_trajectories_8agent_pooled.png"


def main() -> None:
    ds = linguistic_datasets.get("september_n10")
    myths, emb = load(ds)
    m = Map(myths, emb)
    cell = myths[myths["size"] == SIZE]
    xr = (cell.x.quantile(0.002) - 0.03, cell.x.quantile(0.998) + 0.03)  # outermost 0.2% fall outside the frame
    yr = (cell.y.quantile(0.002) - 0.03, cell.y.quantile(0.998) + 0.03)
    rounds = np.array(ROUNDS)
    t = np.linspace(rounds[0], rounds[-1], 120)

    fig, axs = plt.subplots(2, 1, figsize=(3.4, 5.6), sharex=True, sharey=True)
    for ax, mixed in zip(axs, (False, True)):
        runs = cell[cell.mixed == mixed]
        ax.set_facecolor("#f2f7fd")  # the pale blue of trajectories_*.png, sampled from that figure
        for fam in ds.families:  # background: where each family's round-10 myths end up
            last = runs[(runs.family == fam) & (runs["round"] == ROUNDS[-1])]
            if last.empty:
                continue
            d = m.density(last, bw=0.3)
            cmap = LinearSegmentedColormap.from_list(fam, [shade(ds.colors[fam], 0), ds.colors[fam]])
            ax.contourf(m.gx, m.gy, d, levels=np.linspace(d.max() * 0.15, d.max(), 6), cmap=cmap, alpha=0.45)
        for fam in ds.families:
            f = runs[runs.family == fam]
            if f.empty:
                continue
            color = ds.colors[fam]
            per_run = f[f["round"] == ROUNDS[-1]].groupby("run_id")[["x", "y"]].mean()
            ax.scatter(per_run.x, per_run.y, s=6, color=color, alpha=0.8, lw=0.3, edgecolor="white", zorder=3)
            path = f.groupby("round")[["x", "y"]].mean().reindex(rounds)
            sx = make_interp_spline(rounds, path.x, k=3)(t)
            sy = make_interp_spline(rounds, path.y, k=3)(t)
            for a in range(len(t) - 1):
                ax.plot(sx[a:a + 2], sy[a:a + 2], color=shade(color, a / len(t)), lw=3,
                        solid_capstyle="round", zorder=4)
            ax.annotate("", xy=(sx[-1], sy[-1]), xytext=(sx[-8], sy[-8]), zorder=5,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0, mutation_scale=16))
            for r in MARK_ROUNDS:
                ax.scatter(*path.loc[r], s=40, color=shade(color, (r - 1) / (rounds[-1] - 1)),
                           edgecolor="k", lw=0.8, zorder=6)
            ax.text(*path.loc[1], f"  {fam}", fontsize=8, color=color, fontweight="bold", zorder=7, va="center")
        kind = "Single-model" if not mixed else "Mixed"
        ax.set_title(f"{kind} populations", fontsize=9, pad=3)  # run counts go in the caption
        print(f"{kind}: {runs.run_id.nunique()} runs")
        ax.set_xlim(*xr)
        ax.set_ylim(*yr)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(m.ylabel, fontsize=8)
    axs[1].set_xlabel(m.xlabel, fontsize=8)
    fig.tight_layout()
    fig.savefig(N10_OUT / OUT_NAME, dpi=300)
    plt.close(fig)
    print(f"wrote {N10_OUT / OUT_NAME}")
    write_provenance(ds, N10_OUT)  # the folder's provenance.json hashes this figure too


if __name__ == "__main__":
    main()
