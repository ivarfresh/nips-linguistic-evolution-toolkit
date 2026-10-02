#!/usr/bin/env python3
"""Map of myth space: where each model family's myths start and how they move.

Every myth is embedded (all-mpnet-base-v2, the same vectors as the linguistic
analysis) and projected onto the first two principal components of the whole
corpus. PCA is linear, so a family's average position on the map is the map
position of its average embedding: the drawn paths are real averages, not
dots joined in a non-linear layout. The two components keep about a quarter of
the variance, so the README numbers are computed in the full 768-d space.

Figures (one PCA for all of them, so positions are comparable):
  round1_<task_order>.png      round-1 myths, coloured by family and by moral
  trajectories_<n>agent.png    family-average path over rounds 1-10, single-model
                               vs mixed runs, one column per task order
Tables:
  family_separation.csv        silhouette of family labels by cell and round (768-d cosine; 2-D map)
  partner_convergence.csv      mixed dyads: cross-family similarity within a run vs
                               between runs of the same pairing, task order and round
  round1_morals.csv            judge moral label counts for round-1 myths

provenance.json (September only) lists the 156 myth runs, as in linguistic_provenance.py.

Inputs (gitignored, written by linguistic_corpus.py, linguistic_uptake.py and
myth_moral_judge.py): <data>/myths.csv, <data>/embeddings_mpnet.npy,
<data>/moral_labels_z-ai__glm-5.2.csv.

    python analyses/myth_convergence_map.py
    python analyses/myth_convergence_map.py --dataset frontier   # -> <frontier data>/myth_map
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses._shared import cached_embeddings, configure_matplotlib  # noqa: E402
from analyses import linguistic_datasets  # noqa: E402

configure_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from scipy.interpolate import make_interp_spline  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

JUDGE = "z-ai__glm-5.2"
MORALS = {"be generous": "#e6ab02", "be fair": "#666666", "be cautious": "#e7298a"}
TASK_ORDERS = {"myth_game": "Myth → Game", "game_myth": "Game → Myth"}
ROUND1_NOTE = {"myth_game": "written before any game is played",
               "game_myth": "written after the first round of the game"}
ROUNDS = tuple(range(1, 11))
MARK_ROUNDS = (1, 5, 10)
SEPTEMBER_OUT = ROOT / "docs/figures/myth_convergence_map_20261002"
MIN_WORDS = 20  # as in linguistic_uptake


def load(ds) -> tuple[pd.DataFrame, np.ndarray]:
    myths = pd.read_csv(ds.data / "myths.csv")
    myths["text"] = myths["text"].fillna("")  # same texts as linguistic_uptake, so the cache is shared
    emb = cached_embeddings(ds.data / "embeddings_mpnet.npy", myths["text"].tolist(),
                            batch_size=64, show_progress_bar=True)
    valid = (myths["n_words"] >= MIN_WORDS).to_numpy()  # one GPT myth in the mixed dyads is an empty response
    myths, emb = myths[valid].reset_index(drop=True), emb[valid]
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    labels = pd.read_csv(ds.data / f"moral_labels_{JUDGE}.csv")[["run_id", "agent", "round", "label"]]
    merged = myths.merge(labels, on=["run_id", "agent", "round"], how="left", validate="one_to_one")
    assert len(merged) == len(myths) and merged.label.notna().all(), "every myth needs a moral label"
    cells = merged.groupby(["size", "mixed", "task_order", "family"])["round"].apply(frozenset)  # one panel line each
    assert (cells == frozenset(ROUNDS)).all(), f"every family path needs myths in rounds {ROUNDS[0]}-{ROUNDS[-1]}"
    return merged, emb


def shade(color, t: float):
    """t=0 light, t=1 full colour."""
    c = np.array(to_rgb(color))
    return tuple(1 - (1 - c) * (0.25 + 0.75 * t))


class Map:
    def __init__(self, myths: pd.DataFrame, emb: np.ndarray):
        pca = PCA(2).fit(emb)
        xy = pca.transform(emb)
        myths["x"], myths["y"] = xy[:, 0], xy[:, 1]
        var = pca.explained_variance_ratio_ * 100
        self.xlabel, self.ylabel = f"PC1 ({var[0]:.0f}% of variance)", f"PC2 ({var[1]:.0f}% of variance)"
        pad = 0.05
        self.xr = (myths.x.min() - pad, myths.x.max() + pad)
        self.yr = (myths.y.min() - pad, myths.y.max() + pad)
        self.gx, self.gy = np.mgrid[self.xr[0]:self.xr[1]:200j, self.yr[0]:self.yr[1]:200j]

    def density(self, pts: pd.DataFrame, bw=None) -> np.ndarray:
        kde = gaussian_kde(np.vstack([pts.x, pts.y]), bw_method=bw)
        return kde(np.vstack([self.gx.ravel(), self.gy.ravel()])).reshape(self.gx.shape)


def plot_round1(myths, m: Map, ds, task_order: str, out: Path) -> None:
    r1 = myths[(myths["round"] == 1) & (myths.task_order == task_order)]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True)
    panels = [("family", {f: ds.colors[f] for f in ds.families}, "Coloured by model family"),
              ("label", MORALS, "Coloured by the myth's moral (GLM-5.2 judge)")]
    for ax, (col, palette, title) in zip(axs, panels):
        for key, color in palette.items():
            pts = r1[r1[col] == key]
            if pts.empty:
                continue
            ax.scatter(pts.x, pts.y, s=14, color=color, alpha=0.7, lw=0, label=f"{key} (n={len(pts)})")
            if col == "family":
                ax.contour(m.gx, m.gy, m.density(pts), levels=3, colors=[color], linewidths=1)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=9)
        ax.set_xlabel(m.xlabel)
        ax.set_xlim(*m.xr)
        ax.set_ylim(*m.yr)
    axs[0].set_ylabel(m.ylabel)
    fig.suptitle(f"Where the myths start: round-1 myths, {ROUND1_NOTE[task_order]} "
                 f"({TASK_ORDERS[task_order]} runs, single-model + mixed)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"round1_{task_order}.png", dpi=160)
    plt.close(fig)


def plot_trajectories(myths, m: Map, ds, size: int, out: Path) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True)
    for i, mixed in enumerate([False, True]):
        for j, task_order in enumerate(TASK_ORDERS):
            ax = axs[i, j]
            cell = myths[(myths["size"] == size) & (myths.mixed == mixed) & (myths.task_order == task_order)]
            ax.set_xlim(*m.xr)
            ax.set_ylim(*m.yr)
            if i == 1:
                ax.set_xlabel(m.xlabel)
            if j == 0:
                ax.set_ylabel(m.ylabel)
            if cell.empty:
                ax.set_title("no runs")
                continue
            # background: where this panel's round-10 myths end up
            ax.contourf(m.gx, m.gy, m.density(cell[cell["round"] == ROUNDS[-1]], bw=0.25), levels=12, cmap="Blues", alpha=0.85)
            rounds = np.array(ROUNDS)
            t = np.linspace(rounds[0], rounds[-1], 120)
            for fam in ds.families:
                f = cell[cell.family == fam]
                if f.empty:
                    continue
                color = ds.colors[fam]
                path = f.groupby("round")[["x", "y"]].mean().reindex(rounds)
                for r in MARK_ROUNDS:  # each run's family average: the spread between runs
                    per_run = f[f["round"] == r].groupby("run_id")[["x", "y"]].mean()
                    ax.scatter(per_run.x, per_run.y, s=7, color="k", alpha=0.55, lw=0, zorder=3)
                sx = make_interp_spline(rounds, path.x, k=3)(t)
                sy = make_interp_spline(rounds, path.y, k=3)(t)
                for a in range(len(t) - 1):
                    ax.plot(sx[a:a + 2], sy[a:a + 2], color=shade(color, a / len(t)), lw=5,
                            solid_capstyle="round", zorder=4)
                ax.annotate("", xy=(sx[-1], sy[-1]), xytext=(sx[-8], sy[-8]), zorder=5,
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=0, mutation_scale=28))
                for r in MARK_ROUNDS:
                    ax.scatter(*path.loc[r], s=90, color=shade(color, (r - 1) / (rounds[-1] - 1)),
                               edgecolor="k", lw=1, zorder=6)
                ax.text(*path.loc[1], f"  {fam}", fontsize=10, color=color, fontweight="bold", zorder=7, va="center")
            kind = "Mixed-model" if mixed else "Single-model"
            ax.set_title(f"{kind} runs · {TASK_ORDERS[task_order]}  (n={cell.run_id.nunique()} runs)", fontsize=11)
    fig.suptitle(f"How myths move over 10 rounds ({size}-agent runs). Line = family average, light → dark = "
                 f"round 1 → 10; big dots = rounds 1, 5, 10;\nblack dots = each run's family average at those "
                 f"rounds; blue background = where round-10 myths end up", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"trajectories_{size}agent.png", dpi=160)
    plt.close(fig)


def family_separation(myths, emb) -> pd.DataFrame:
    """Family silhouette in 768-d (cite this) and on the 2-D map (shows how much the map flatters it)."""
    rows = []
    for (size, mixed, task_order), cell in myths.groupby(["size", "mixed", "task_order"]):
        for r in MARK_ROUNDS:
            sel = cell[cell["round"] == r]
            rows.append(dict(size=size, mixed=mixed, task_order=task_order, round=r, n_myths=len(sel),
                             silhouette_cosine=silhouette_score(emb[sel.index], sel.family, metric="cosine"),
                             silhouette_map_2d=silhouette_score(sel[["x", "y"]], sel.family)))
    return pd.DataFrame(rows)


def partner_convergence(myths, emb) -> pd.DataFrame:
    """Mixed dyads: are partners' myths closer than different-family myths from other runs?"""
    rows = []
    dyads = myths[myths.mixed & (myths["size"] == 2)]
    for (composition, task_order, r), sel in dyads.groupby(["composition", "task_order", "round"]):
        sim = emb[sel.index] @ emb[sel.index].T
        fam = sel.family.to_numpy(dtype=object)
        run = sel.run_id.to_numpy(dtype=object)
        cross = fam[:, None] != fam[None, :]
        same_run = run[:, None] == run[None, :]
        # per run: its partners' similarity minus its myths' similarity to other runs' other-family myths
        per_run = [sim[cross & same_run & (run[:, None] == rid)].mean() - sim[cross & ~same_run & (run[:, None] == rid)].mean()
                   for rid in np.unique(run)]
        rows.append(dict(composition=composition, task_order=task_order, round=r,
                         n_runs=len(per_run), n_runs_closer=int(np.sum(np.array(per_run) > 0)),
                         same_run=sim[cross & same_run].mean(), other_run=sim[cross & ~same_run].mean()))
    out = pd.DataFrame(rows)
    out["difference"] = out.same_run - out.other_run
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=sorted(linguistic_datasets.DATASETS), default=None)
    ap.add_argument("--out", type=Path, help="default: the September figure folder, or <dataset data>/myth_map")
    args = ap.parse_args()
    ds = linguistic_datasets.get(args.dataset)
    if args.out is None:  # never let another corpus overwrite the September figures
        args.out = SEPTEMBER_OUT if ds.name == "september" else ds.data / "myth_map"
    args.out.mkdir(parents=True, exist_ok=True)

    myths, emb = load(ds)
    m = Map(myths, emb)
    for task_order in TASK_ORDERS:
        plot_round1(myths, m, ds, task_order, args.out)
    for size in sorted(myths["size"].unique(), reverse=True):
        plot_trajectories(myths, m, ds, size, args.out)

    family_separation(myths, emb).round(3).to_csv(args.out / "family_separation.csv", index=False)
    partner_convergence(myths, emb).round(3).to_csv(args.out / "partner_convergence.csv", index=False)
    r1 = myths[myths["round"] == 1]
    (r1.groupby(["task_order", "family"]).label.value_counts().unstack(fill_value=0)
       .to_csv(args.out / "round1_morals.csv"))
    print(f"wrote {args.out} ({m.xlabel}, {m.ylabel})")
    if args.out == SEPTEMBER_OUT:  # the same 156 myth runs as the linguistic analysis; run after README edits too
        from analyses import linguistic_provenance
        linguistic_provenance.main(args.out)


if __name__ == "__main__":
    main()
