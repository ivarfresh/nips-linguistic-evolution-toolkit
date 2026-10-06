#!/usr/bin/env python3
"""Paper Figure 5b: how each family's myths move over ten rounds, single-model vs mixed populations.

The trajectory view of myth_convergence_map.py (trajectories_8agent.png), on the
same PCA map of all 300 n = 10 myth runs, with two changes for the paper: the two
task orders are pooled into one panel per mixing condition, and the background is
coloured by family (where that family's round-10 myths end up) instead of one blue
density. Line = family average per round, faintest background band -> dark = round 1 -> 10; big dots =
rounds 1, 5, 10; small dots = each run's family average at round 10.
8-agent populations (paper Figure 5b) by default; --size 2 draws the dyads (supplementary).
The task-order split is in trajectories_<n>agent.png;
numbers to cite are the 768-d ones in significance.csv. No API calls.

    LINGUISTIC_DATASET=september_n10 python3 analyses/plot_myth_trajectory_panel.py            # populations
    LINGUISTIC_DATASET=september_n10 python3 analyses/plot_myth_trajectory_panel.py --size 2   # dyads
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses import linguistic_datasets  # noqa: E402
from analyses.myth_convergence_map import (MARK_ROUNDS, N10_OUT, ROUNDS, Map, load, shade,  # noqa: E402
                                           write_provenance)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from scipy.interpolate import make_interp_spline  # noqa: E402

GROUP = {8: "populations", 2: "dyads"}


BG = "#f2f7fd"  # the pale blue of trajectories_*.png, sampled from that figure
BAND_ALPHA = 0.45


def band_colors(color, n_bands: int = 5):
    """Opaque on-screen colour of each background band, faintest first (contourf colours each band at
    its level midpoint, then it is blended at BAND_ALPHA over BG)."""
    cmap = LinearSegmentedColormap.from_list("", [shade(color, 0), color])
    bg = np.array(to_rgb(BG))
    return [BAND_ALPHA * np.array(cmap((i + 0.5) / n_bands)[:3]) + (1 - BAND_ALPHA) * bg for i in range(n_bands)]


def tone(color, t: float):
    """t=0: the faintest visible background band of this family (round 1); t=1: the family colour at
    40% brightness (round 10)."""
    start, end = band_colors(color)[0], np.array(to_rgb(color)) * 0.4
    return tuple(start + (end - start) * t)
PATH_LW = 2.4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", type=int, choices=sorted(GROUP), default=8)
    size = ap.parse_args().size
    out_name = f"myth_map_trajectories_{size}agent_pooled.png"
    ds = linguistic_datasets.get("september_n10")
    myths, emb = load(ds)
    m = Map(myths, emb)
    cell = myths[myths["size"] == size]
    rounds = np.array(ROUNDS)
    t = np.linspace(rounds[0], rounds[-1], 120)

    fig, axs = plt.subplots(1, 2, figsize=(3.35, 1.75), sharex=True, sharey=True)  # drawn at column width
    for ax, mixed in zip(axs, (False, True)):
        runs = cell[cell.mixed == mixed]
        ax.set_facecolor(BG)
        for fam in ds.families:  # background: where each family's round-10 myths end up
            last = runs[(runs.family == fam) & (runs["round"] == ROUNDS[-1])]
            if last.empty:
                continue
            d = m.density(last, bw=0.3)
            cmap = LinearSegmentedColormap.from_list(fam, [shade(ds.colors[fam], 0), ds.colors[fam]])
            ax.contourf(m.gx, m.gy, d, levels=np.linspace(d.max() * 0.15, d.max(), 6), cmap=cmap, alpha=BAND_ALPHA)
        for fam in ds.families:
            f = runs[runs.family == fam]
            if f.empty:
                continue
            color = ds.colors[fam]
            per_run = f[f["round"] == ROUNDS[-1]].groupby("run_id")[["x", "y"]].mean()
            ax.scatter(per_run.x, per_run.y, s=3, color=color, alpha=0.8, lw=0.2, edgecolor="white", zorder=3)
            path = f.groupby("round")[["x", "y"]].mean().reindex(rounds)
            sx = make_interp_spline(rounds, path.x, k=3)(t)
            sy = make_interp_spline(rounds, path.y, k=3)(t)
            ax.plot(sx, sy, color="k", lw=PATH_LW + 1.1, solid_capstyle="round", zorder=4)  # outline
            for a in range(len(t) - 1):  # faintest background band -> dark = round 1 -> 10
                ax.plot(sx[a:a + 2], sy[a:a + 2], color=tone(color, a / len(t)), lw=PATH_LW,
                        solid_capstyle="round", zorder=4.5)
            # no arrowhead: the spline's last few points wiggle, so a head points the wrong way;
            # light -> dark and the round 1/5/10 dots give the direction
            for r in MARK_ROUNDS:
                ax.scatter(*path.loc[r], s=14, color=tone(color, (r - 1) / (rounds[-1] - 1)),
                           edgecolor="k", lw=0.5, zorder=6)
            ax.text(*path.loc[1], f" {fam}", fontsize=5, color=color, fontweight="bold", zorder=7, va="center")
        kind = "Single-model" if not mixed else "Mixed"
        ax.set_title(f"{kind} {GROUP[size]}", fontsize=5.5, pad=2)  # run counts go in the caption
        print(f"{kind}: {runs.run_id.nunique()} runs")
        ax.set_xlim(*m.xr)  # full-corpus extent, as in trajectories_*.png
        ax.set_ylim(*m.yr)
        ax.tick_params(labelsize=4.5, length=1.5, width=0.4, pad=1)  # panel (a) prints ~5 pt at column width
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        ax.set_xlabel(m.xlabel, fontsize=5, labelpad=1.5)
    axs[0].set_ylabel(m.ylabel, fontsize=5, labelpad=1.5)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(N10_OUT / out_name, dpi=600)
    plt.close(fig)
    print(f"wrote {N10_OUT / out_name}")
    write_provenance(ds, N10_OUT)  # the folder's provenance.json hashes this figure too


if __name__ == "__main__":
    main()
