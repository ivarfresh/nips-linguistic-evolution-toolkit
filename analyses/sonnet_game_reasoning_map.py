#!/usr/bin/env python3
"""Map of Sonnet's game reasoning: does writing myths change how Sonnet talks about play?

The game prompt asks only for a JSON decision. Sonnet 4.5 adds a short
explanation anyway (81% of its game replies); GPT-5 Nano and Gemini 3.7 Flash
return JSON only, so this map is Sonnet-only. It covers every Sonnet decision in
the 111 September runs that contain Sonnet (validated run tables behind Figures
7 and 8): game only, game -> myth and myth -> game, with Sonnet partners, GPT
partners or Gemini partners.

The text is the reply with the JSON decision removed; replies under MIN_WORDS
words are dropped, and of a retried move only the last reply (the one the game
used) is kept. Texts are embedded with all-mpnet-base-v2 and projected onto
the first two principal components of the whole set, as in
myth_convergence_map.py. Senders and receivers are drawn in separate rows
because the roles alternate by round and read differently.

Figures:
  round1.png                 round-1 explanations by task order and role
  trajectories_<n>agent.png  task-order average path over rounds 1-10
Tables:
  texts_per_cell.csv         how many replies carry text, per cell
  distance_from_game_only.csv  768-d cosine distance of each myth condition's
                             centroid from the game-only centroid, per round,
                             with the game-only split-half distance as the noise floor
  distinctive_words.csv      words most over-used in each myth condition
                             relative to game only (log-odds, informative prior)

provenance.json lists the full validated September run set (linguistic_provenance
with_game_only), of which the 111 runs containing Sonnet are read.

    python3 analyses/sonnet_game_reasoning_map.py
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses._shared import cached_embeddings, configure_matplotlib  # noqa: E402
from analyses.linguistic_corpus import RUN_TABLES, agent_families  # noqa: E402

configure_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from scipy.interpolate import make_interp_spline  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

DATA = ROOT / "data/analysis/sonnet_game_reasoning_20261002"
OUT = ROOT / "docs/figures/sonnet_game_reasoning_20261002"
MIN_WORDS = 10
ROUNDS = tuple(range(1, 11))
MARK_ROUNDS = (1, 5, 10)
TASK_ORDERS = {"game": ("Game only", "#7f7f7f"), "game_myth": ("Game → Myth", "#1f78b4"),
               "myth_game": ("Myth → Game", "#e6550d")}
ROLES = {"investor": "Sender", "trustee": "Receiver"}
SETTING_LABEL = {2: {"Sonnet only": "Sonnet + Sonnet", "with GPT": "Sonnet + GPT", "with Gemini": "Sonnet + Gemini"},
                 8: {"Sonnet only": "8 Sonnet", "with GPT": "Sonnet among 1, 2 or 4 GPT"}}
DECISION = re.compile(r"```(?:json)?\s*\{.*?\}\s*```|\{[^{}]*[\"'](?:send|return)[\"'][^{}]*\}", re.S)
AMOUNT = re.compile(r"[\"'](send|return)[\"']\s*:\s*([0-9.]+)")
TOKEN = re.compile(r"[a-z][a-z'-]+")


def setting_of(composition: str, size: int) -> str:
    if "GPT" in composition:
        return "with GPT"
    if "Gemini" in composition:
        return "with Gemini"
    return "Sonnet only"


def extract() -> pd.DataFrame:
    rows = []
    for size, table in RUN_TABLES.items():
        runs = pd.read_csv(table).drop_duplicates("path")
        runs = runs[runs.composition.str.contains("Sonnet")]
        for path, composition, task_order in runs[["path", "composition", "task_order"]].itertuples(index=False):
            run = json.loads((ROOT / path).read_text())
            fams = agent_families(run)
            for agent_id, agent in run["agents"].items():
                if fams[agent_id] != "Sonnet":
                    continue
                for event in agent["interaction_history"]:
                    meta = event["metadata"]
                    if meta.get("task") != "game":
                        continue
                    content = event["response"].get("content") or ""
                    amount = AMOUNT.search(content)
                    text = re.sub(r"\s+", " ", DECISION.sub(" ", content)).strip()
                    rows.append(dict(path=path, run_id=Path(path).stem, size=size, composition=composition,
                                     setting=setting_of(composition, size), task_order=task_order,
                                     round=meta["round"], agent=agent_id, role=meta["role"],
                                     partner_family=fams[meta["opponent_id"]],
                                     amount=float(amount.group(2)) if amount else np.nan,
                                     text=text, n_words=len(text.split())))
    df = pd.DataFrame(rows)
    # a retried move is logged twice under the same move; the last reply is the decision the game used
    return df.drop_duplicates(["run_id", "agent", "round"], keep="last").reset_index(drop=True)


def shade(color, t: float):
    c = np.array(to_rgb(color))
    return tuple(1 - (1 - c) * (0.25 + 0.75 * t))


class Map:
    def __init__(self, df: pd.DataFrame, emb: np.ndarray):
        pca = PCA(2).fit(emb)
        xy = pca.transform(emb)
        df["x"], df["y"] = xy[:, 0], xy[:, 1]
        var = pca.explained_variance_ratio_ * 100
        self.xlabel, self.ylabel = f"PC1 ({var[0]:.0f}% of variance)", f"PC2 ({var[1]:.0f}% of variance)"
        pad = 0.05
        self.xr = (df.x.min() - pad, df.x.max() + pad)
        self.yr = (df.y.min() - pad, df.y.max() + pad)
        self.gx, self.gy = np.mgrid[self.xr[0]:self.xr[1]:200j, self.yr[0]:self.yr[1]:200j]

    def density(self, pts, bw=None):
        kde = gaussian_kde(np.vstack([pts.x, pts.y]), bw_method=bw)
        return kde(np.vstack([self.gx.ravel(), self.gy.ravel()])).reshape(self.gx.shape)

    def frame(self, ax, i, j):
        ax.set_xlim(*self.xr)
        ax.set_ylim(*self.yr)
        if j == 0:
            ax.set_ylabel(self.ylabel)
        ax.set_xlabel(self.xlabel)


def plot_round1(df, m: Map) -> None:
    r1 = df[df["round"] == 1]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True)
    for j, (role, role_name) in enumerate(ROLES.items()):
        ax = axs[j]
        for order, (name, color) in TASK_ORDERS.items():
            pts = r1[(r1.role == role) & (r1.task_order == order)]
            ax.scatter(pts.x, pts.y, s=16, color=color, alpha=0.65, lw=0, label=f"{name} (n={len(pts)})")
            if len(pts) > 5:
                ax.contour(m.gx, m.gy, m.density(pts), levels=3, colors=[color], linewidths=1)
        ax.set_title(f"{role_name}s, round 1")
        ax.legend(frameon=False, fontsize=9)
        m.frame(ax, 0, j)
    fig.suptitle("Sonnet's round-1 game explanations. Game → Myth differs from Game only by one line ('Take any "
                 "myths written in this session into account'),\nwith no myth written yet; Myth → Game has written "
                 "a myth first", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "round1.png", dpi=160)
    plt.close(fig)


def plot_trajectories(df, m: Map, size: int) -> None:
    sub = df[df["size"] == size]
    settings = [s for s in ("Sonnet only", "with GPT", "with Gemini") if s in set(sub.setting)]
    fig, axs = plt.subplots(2, len(settings), figsize=(6.6 * len(settings), 11), sharex=True, sharey=True, squeeze=False)
    for i, (role, role_name) in enumerate(ROLES.items()):
        for j, setting in enumerate(settings):
            ax = axs[i, j]
            cell = sub[(sub.role == role) & (sub.setting == setting)]
            ax.contourf(m.gx, m.gy, m.density(cell[cell["round"] == ROUNDS[-1]], bw=0.25), levels=12, cmap="Greys", alpha=0.6)
            for order, (name, color) in TASK_ORDERS.items():
                f = cell[cell.task_order == order]
                path = f.groupby("round")[["x", "y"]].mean().reindex(ROUNDS)
                assert path.notna().all().all(), (size, setting, role, order)
                for r in MARK_ROUNDS:
                    per_run = f[f["round"] == r].groupby("run_id")[["x", "y"]].mean()
                    ax.scatter(per_run.x, per_run.y, s=7, color=color, alpha=0.5, lw=0, zorder=3)
                t = np.linspace(ROUNDS[0], ROUNDS[-1], 120)
                sx = make_interp_spline(ROUNDS, path.x, k=3)(t)
                sy = make_interp_spline(ROUNDS, path.y, k=3)(t)
                for a in range(len(t) - 1):
                    ax.plot(sx[a:a + 2], sy[a:a + 2], color=shade(color, a / len(t)), lw=4.5,
                            solid_capstyle="round", zorder=4)
                ax.annotate("", xy=(sx[-1], sy[-1]), xytext=(sx[-8], sy[-8]), zorder=5,
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=0, mutation_scale=26))
                for r in MARK_ROUNDS:
                    ax.scatter(*path.loc[r], s=80, color=shade(color, (r - 1) / (ROUNDS[-1] - 1)),
                               edgecolor="k", lw=1, zorder=6)
            ax.set_title(f"{role_name}s · {SETTING_LABEL[size][setting]}  (n={cell.run_id.nunique()} runs)", fontsize=11)
            m.frame(ax, i, j)
    handles = [plt.Line2D([], [], color=c, lw=4, label=n) for n, c in TASK_ORDERS.values()]
    axs[0, 0].legend(handles=handles, loc="lower left", frameon=False, fontsize=10)
    fig.suptitle(f"How Sonnet's game explanations move over 10 rounds ({size}-agent runs). Line = task-order "
                 f"average, light → dark = round 1 → 10;\nbig dots = rounds 1, 5, 10; small dots = each run's "
                 f"average at those rounds; grey background = where round-10 explanations end up", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / f"trajectories_{size}agent.png", dpi=160)
    plt.close(fig)


def unit(v):
    return v / np.linalg.norm(v)


def distance_from_game_only(df, emb, seed=20261002) -> pd.DataFrame:
    """Cosine distance between each condition's centroid and the game-only centroid.

    The noise floor splits the game-only runs into two random halves (by run)
    and measures the same distance between the halves, 200 times.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (size, setting, role, r), cell in df.groupby(["size", "setting", "role", "round"]):
        base = cell[cell.task_order == "game"]
        c0 = unit(emb[base.index].mean(0))
        runs = base.run_id.unique()
        floor = []
        for _ in range(200):
            half = set(rng.permutation(runs)[: len(runs) // 2])
            a = base.run_id.isin(half).to_numpy()
            floor.append(1 - unit(emb[base.index[a]].mean(0)) @ unit(emb[base.index[~a]].mean(0)))
        row = dict(size=size, setting=setting, role=ROLES[role], round=r, split_half_floor=np.mean(floor))
        for order in ("game_myth", "myth_game"):
            sel = cell[cell.task_order == order]
            row[order] = 1 - unit(emb[sel.index].mean(0)) @ c0
        rows.append(row)
    return pd.DataFrame(rows)


def distinctive_words(df, top=15) -> pd.DataFrame:
    """Weighted log-odds with an informative Dirichlet prior (Monroe et al. 2008), each myth order vs game only."""
    from collections import Counter
    counts = {o: Counter(w for t in df[df.task_order == o].text for w in TOKEN.findall(t.lower())) for o in TASK_ORDERS}
    prior = sum(counts.values(), Counter())
    a0 = sum(prior.values())
    rows = []
    for order in ("game_myth", "myth_game"):
        c1, c2 = counts[order], counts["game"]
        n1, n2 = sum(c1.values()), sum(c2.values())
        scores = []
        for w, aw in prior.items():
            if aw < 20:
                continue
            l1 = np.log((c1[w] + aw) / (n1 + a0 - c1[w] - aw))
            l2 = np.log((c2[w] + aw) / (n2 + a0 - c2[w] - aw))
            z = (l1 - l2) / np.sqrt(1 / (c1[w] + aw) + 1 / (c2[w] + aw))
            scores.append((w, z, c1[w] / n1 * 1000, c2[w] / n2 * 1000))
        scores.sort(key=lambda s: s[1])
        for direction, chosen in (("more in " + order, scores[::-1][:top]), ("more in game only", scores[:top])):
            for w, z, f1, f2 in chosen:
                rows.append(dict(comparison=f"{order} vs game", direction=direction, word=w, z=z,
                                 per_1000_words_myth_condition=f1, per_1000_words_game_only=f2))
    return pd.DataFrame(rows)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df = extract()
    (df.assign(has_text=df.n_words >= MIN_WORDS)
       .groupby(["size", "setting", "task_order"])
       .agg(replies=("text", "size"), with_text=("has_text", "sum"), median_words=("n_words", "median"))
       .reset_index().to_csv(OUT / "texts_per_cell.csv", index=False))
    df = df[df.n_words >= MIN_WORDS].reset_index(drop=True)
    df.drop(columns=["x", "y"], errors="ignore").to_csv(DATA / "texts.csv", index=False)
    emb = cached_embeddings(DATA / "embeddings_mpnet.npy", df.text.tolist(), batch_size=64, show_progress_bar=True)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    m = Map(df, emb)
    plot_round1(df, m)
    for size in sorted(df["size"].unique(), reverse=True):
        plot_trajectories(df, m, size)
    distance_from_game_only(df, emb).round(4).to_csv(OUT / "distance_from_game_only.csv", index=False)
    distinctive_words(df).round(3).to_csv(OUT / "distinctive_words.csv", index=False)
    # manifest: the full validated run set (game-only included), of which the Sonnet runs are read
    from analyses import linguistic_provenance
    linguistic_provenance.main(OUT, with_game_only=True)
    print(f"{len(df)} texts; wrote {OUT} ({m.xlabel}, {m.ylabel})")


if __name__ == "__main__":
    main()
