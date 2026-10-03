#!/usr/bin/env python3
"""Joint round-1 myth map: September models (Sonnet 4.5, Gemini 3.7 Flash, GPT-5 Nano) beside the
frontier models (Opus 5, Gemini 3.1 Pro, GPT-5.6 Sol).

Does each frontier model open with the same kind of myth as its smaller sibling from the same lab?
Round-1 myths of both corpora are embedded with all-mpnet-base-v2 (the cached vectors of each
corpus) and projected onto the first two principal components of the round-1 myths of BOTH
corpora, so this map is not the one in the per-corpus folders. Siblings share a hue: September
models are filled circles, frontier models open triangles.

This is the only place the two corpora meet: they come from different models, request profiles
and dates, so nothing here is pooled into a statistic about either corpus.

Outputs (in the frontier myth-map folder):
  joint_round1.png                     the map, one panel per task order
  joint_round1_centroid_distance.csv   768-d cosine distance between the six families' round-1 centroids
  joint_round1_confusion.csv           six-way classifier, 5 folds split by run: true family × predicted

    python3 analyses/myth_map_joint_round1.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses import linguistic_datasets  # noqa: E402
from analyses.myth_convergence_map import TASK_ORDERS, load  # noqa: E402
from analyses.myth_map_significance import FRONTIER_OUT, frontier_provenance  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402

SIBLING = {"Opus": "Sonnet", "GeminiPro": "Gemini", "Sol": "GPT"}  # same lab


def round1(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    ds = linguistic_datasets.get(name)
    myths, emb = load(ds)
    keep = (myths["round"] == 1).to_numpy()
    r1 = myths[keep].reset_index(drop=True).assign(corpus=name, color=lambda d: d.family.map(ds.colors))
    return r1, emb[keep]


def main() -> None:
    sep, e_sep = round1("september")
    fro, e_fro = round1("frontier")
    myths = pd.concat([sep, fro], ignore_index=True)
    emb = np.vstack([e_sep, e_fro])
    order = [*linguistic_datasets.get("september").families, *linguistic_datasets.get("frontier").families]

    pca = PCA(2, svd_solver="full").fit(emb)
    xy = pca.transform(emb)
    myths["x"], myths["y"] = xy[:, 0], xy[:, 1]
    var = pca.explained_variance_ratio_ * 100
    pad = 0.05
    xr = (myths.x.min() - pad, myths.x.max() + pad)
    yr = (myths.y.min() - pad, myths.y.max() + pad)
    gx, gy = np.mgrid[xr[0]:xr[1]:200j, yr[0]:yr[1]:200j]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6.4), sharex=True, sharey=True)
    for ax, to in zip(axs, TASK_ORDERS):
        cell = myths[myths.task_order == to]
        for fam in order:
            pts = cell[cell.family == fam]
            color = pts.color.iloc[0]
            frontier = fam in SIBLING
            label = f"{fam} (n={len(pts)})" + (f", sibling of {SIBLING[fam]}" if frontier else "")
            if frontier:
                ax.scatter(pts.x, pts.y, s=22, marker="^", facecolors="none", edgecolors=color, lw=1.1, alpha=0.85, label=label)
            else:
                ax.scatter(pts.x, pts.y, s=14, color=color, alpha=0.55, lw=0, label=label)
            kde = gaussian_kde(np.vstack([pts.x, pts.y]))
            ax.contour(gx, gy, kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape), levels=2, colors=[color],
                       linewidths=1.4, linestyles="--" if frontier else "-")
        ax.set_title(f"{TASK_ORDERS[to]}: round-1 myths, "
                     + ("written before any game" if to == "myth_game" else "written after one round of play"), fontsize=11)
        ax.set_xlabel(f"PC1 ({var[0]:.0f}% of variance)")
        ax.set_xlim(*xr)
        ax.set_ylim(*yr)
        ax.legend(frameon=False, fontsize=8.5, loc="best")
    axs[0].set_ylabel(f"PC2 ({var[1]:.0f}% of variance)")
    fig.suptitle("Do frontier models open with the same kind of myth as their smaller siblings? Filled circles, solid "
                 "rings: September models.\nOpen triangles, dashed rings: frontier models; siblings from the same lab "
                 "share a colour. One map fitted to the round-1 myths of both sets.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FRONTIER_OUT / "joint_round1.png", dpi=160)
    plt.close(fig)

    # centroid distances (768-d), per task order
    rows = []
    for to in TASK_ORDERS:
        cell = myths[myths.task_order == to]
        cent = {f: emb[cell.index[cell.family == f]].mean(0) for f in order}
        cent = {f: v / np.linalg.norm(v) for f, v in cent.items()}
        for a in order:
            rows.append(dict(task_order=to, family=a, **{b: round(1 - cent[a] @ cent[b], 4) for b in order}))
    pd.DataFrame(rows).to_csv(FRONTIER_OUT / "joint_round1_centroid_distance.csv", index=False)

    # six-way classifier, folds split by run
    conf = []
    for to in TASK_ORDERS:
        cell = myths[myths.task_order == to]
        pred = cross_val_predict(LogisticRegression(max_iter=3000), emb[cell.index], cell.family,
                                 groups=cell.run_id, cv=GroupKFold(5))
        tab = pd.crosstab(cell.family, pred).reindex(index=order, columns=order, fill_value=0)
        tab.insert(0, "task_order", to)
        conf.append(tab)
    pd.concat(conf).to_csv(FRONTIER_OUT / "joint_round1_confusion.csv", index_label="true_family")
    print(pd.concat(conf).to_string())
    print(pd.DataFrame(rows).to_string(index=False))

    frontier_provenance(FRONTIER_OUT, fro.path.unique())  # adds the September pools for this figure


if __name__ == "__main__":
    main()
