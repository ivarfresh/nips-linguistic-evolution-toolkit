#!/usr/bin/env python3
"""Significance tests and two time views for the myth map (myth_convergence_map.py).

The PCA map keeps 92% of the between-family differences but only 13% of the
between-round ones, so it understates change over time. This script adds:

  convergence_over_rounds.png   mean cosine distance between a myth of family A
                                and a myth of family B, per round, in the full
                                768-d space; single-model runs dashed, mixed runs
                                solid; 95% bands from resampling runs
  family_time_map_<n>agent.png  a map with chosen axes: x separates the families
                                (largest between-family spread), y is the average
                                round-1 -> round-10 direction within a family
  significance.csv              every test in the README's significance table
  variance.csv                  what share of each kind of difference each view keeps

Same corpus and filters as myth_convergence_map.py. Runs are the unit of every
test. No API calls.

    python3 analyses/myth_map_significance.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses import linguistic_datasets  # noqa: E402
from analyses.myth_convergence_map import ROUNDS, SEPTEMBER_OUT, TASK_ORDERS, load, shade  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (backend set by myth_convergence_map)
from scipy import stats  # noqa: E402
from scipy.interpolate import make_interp_spline  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402

SEED = 20261003
N_BOOT = 1000
PAIRS = [("Sonnet", "GPT"), ("Sonnet", "Gemini"), ("Gemini", "GPT")]


def unit(v):
    return v / np.linalg.norm(v)


def group_of(row) -> str:
    """Cell for the distance plot: single-model runs pool across families; mixed runs by family pair."""
    if not row.mixed:
        return "single"
    fams = sorted(set(row.composition.replace("+", " ").split()) & {"Sonnet", "GPT", "Gemini"})
    return "+".join(fams)


# --------------------------------------------------------------------------- distances over rounds

def run_sums(myths, emb):
    """Per (task order, size, group, round, run, family): sum of unit embeddings and count."""
    keys = ["task_order", "size", "group", "round", "run_id", "family"]
    out = {}
    for key, ix in myths.groupby(keys).indices.items():
        out[key] = (emb[ix].sum(0), len(ix))
    return out


def cross_distance(sums, runs_a, runs_b, key_a, key_b):
    """Mean cosine distance over all (A myth, B myth) pairs = 1 - (sum_A . sum_B) / (n_A n_B)."""
    sa = sum(sums[key_a(r)][0] for r in runs_a)
    na = sum(sums[key_a(r)][1] for r in runs_a)
    sb = sum(sums[key_b(r)][0] for r in runs_b)
    nb = sum(sums[key_b(r)][1] for r in runs_b)
    return 1 - sa @ sb / (na * nb)


def distance_curves(myths, emb, rng) -> pd.DataFrame:
    sums = run_sums(myths, emb)
    rows = []
    for (to, size, group), cell in myths.groupby(["task_order", "size", "group"]):
        for a, b in PAIRS:
            if group != "single" and group != "+".join(sorted((a, b))):
                continue
            runs_a = cell[cell.family == a].run_id.unique()
            runs_b = cell[cell.family == b].run_id.unique()
            if len(runs_a) == 0 or len(runs_b) == 0:
                continue
            same_runs = group != "single"  # mixed: resample runs once, both families come along
            for r in ROUNDS:
                ka = lambda run, r=r: (to, size, group, r, run, a)  # noqa: E731
                kb = lambda run, r=r: (to, size, group, r, run, b)  # noqa: E731
                ok_a = [x for x in runs_a if ka(x) in sums]
                ok_b = [x for x in runs_b if kb(x) in sums]
                est = cross_distance(sums, ok_a, ok_b, ka, kb)
                boots = []
                for _ in range(N_BOOT):
                    if same_runs:
                        pick = rng.choice(ok_a, len(ok_a))
                        ba, bb = [x for x in pick if kb(x) in sums and ka(x) in sums], None
                        boots.append(cross_distance(sums, ba, ba, ka, kb))
                    else:
                        boots.append(cross_distance(sums, rng.choice(ok_a, len(ok_a)), rng.choice(ok_b, len(ok_b)), ka, kb))
                lo, hi = np.percentile(boots, [2.5, 97.5])
                rows.append(dict(task_order=to, size=size, group=group, pair=f"{a}–{b}", round=r,
                                 distance=est, ci_low=lo, ci_high=hi, n_runs=len(set(ok_a) | set(ok_b))))
    return pd.DataFrame(rows)


def plot_distance_curves(curves, ds, out: Path) -> None:
    pair_color = {"Sonnet–GPT": "#7570b3", "Sonnet–Gemini": "#1b9e77", "Gemini–GPT": "#d95f02"}
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for i, size in enumerate([2, 8]):
        for j, to in enumerate(TASK_ORDERS):
            ax = axs[i, j]
            cell = curves[(curves["size"] == size) & (curves.task_order == to)]
            for (group, pair), c in cell.groupby(["group", "pair"]):
                single = group == "single"
                color = pair_color[pair]
                ax.fill_between(c["round"], c.ci_low, c.ci_high, color=color, alpha=0.10 if single else 0.18, lw=0)
                ax.plot(c["round"], c.distance, color=color, lw=2, ls="--" if single else "-", marker="o", ms=3,
                        label=f"{pair}, {'single-model runs' if single else 'mixed runs'} (n={c.n_runs.iloc[0]})")
            ax.set_title(f"{size} agents · {TASK_ORDERS[to]}", fontsize=11)
            ax.set_xticks(list(ROUNDS))
            ax.grid(alpha=0.25)
            if i == 1:
                ax.set_xlabel("Round")
            if j == 0:
                ax.set_ylabel("Mean cosine distance between\na myth of each family (768-d)")
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Do the families' myths converge? Distance between a myth of one family and a myth of the other, "
                 "per round.\nDashed: families in separate single-model runs. Solid: families in the same mixed runs. "
                 "Bands: 95% intervals from resampling runs.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "convergence_over_rounds.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- map with chosen axes

def chosen_axes(myths, emb):
    """x: the direction holding the most between-family spread (top axis of the size-weighted family centroids).
    y: mean within-family round-1 -> round-10 shift, made orthogonal to x."""
    mu = emb.mean(0)
    g = myths.groupby("family").indices
    dev = np.stack([np.sqrt(len(ix)) * (emb[ix].mean(0) - mu) for ix in g.values()])
    x_dir = np.linalg.svd(dev, full_matrices=False)[2][0]
    shifts = [unit(emb[(myths.family == f).to_numpy() & (myths["round"] == ROUNDS[-1]).to_numpy()].mean(0)
                   - emb[(myths.family == f).to_numpy() & (myths["round"] == ROUNDS[0]).to_numpy()].mean(0))
              for f in ("Sonnet", "GPT", "Gemini")]
    y_dir = np.mean(shifts, axis=0)
    y_dir = unit(y_dir - (y_dir @ x_dir) * x_dir)
    return x_dir, y_dir


def share_kept(myths, emb, basis) -> dict:
    """Share of the between-family and between-round sums of squares that lie in the span of `basis`."""
    mu = emb.mean(0)
    q, _ = np.linalg.qr(np.stack(basis).T)
    out = {}
    for name, key in (("between_family", "family"), ("between_round", "round")):
        g = myths.groupby(key).indices
        dev = np.stack([emb[ix].mean(0) - mu for ix in g.values()])
        w = np.array([len(ix) for ix in g.values()])
        out[name] = float((w[:, None] * (dev @ q) ** 2).sum() / (w[:, None] * dev ** 2).sum())
    return out


def plot_family_time_map(myths, emb, ds, x_dir, y_dir, size: int, out: Path) -> None:
    myths = myths.assign(fx=emb @ x_dir, ty=emb @ y_dir)
    fig, axs = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    for i, mixed in enumerate([False, True]):
        for j, to in enumerate(TASK_ORDERS):
            ax = axs[i, j]
            cell = myths[(myths["size"] == size) & (myths.mixed == mixed) & (myths.task_order == to)]
            for fam in ds.families:
                f = cell[cell.family == fam]
                if f.empty:
                    continue
                color = ds.colors[fam]
                path = f.groupby("round")[["fx", "ty"]].mean().reindex(ROUNDS)
                per_run = f[f["round"].isin([1, 10])].groupby(["run_id", "round"])[["fx", "ty"]].mean().reset_index()
                ax.scatter(per_run.fx, per_run.ty, s=8, color=color, alpha=0.35, lw=0, zorder=2)
                t = np.linspace(ROUNDS[0], ROUNDS[-1], 120)
                sx = make_interp_spline(ROUNDS, path.fx, k=3)(t)
                sy = make_interp_spline(ROUNDS, path.ty, k=3)(t)
                for a in range(len(t) - 1):
                    ax.plot(sx[a:a + 2], sy[a:a + 2], color=shade(color, a / len(t)), lw=4.5, solid_capstyle="round", zorder=4)
                ax.annotate("", xy=(sx[-1], sy[-1]), xytext=(sx[-8], sy[-8]), zorder=5,
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=0, mutation_scale=26))
                for r in (1, 5, 10):
                    ax.scatter(*path.loc[r], s=80, color=shade(color, (r - 1) / 9), edgecolor="k", lw=1, zorder=6)
                ax.text(*path.loc[1], f"  {fam}", fontsize=10, color=color, fontweight="bold", va="center", zorder=7)
            kind = "Mixed-model" if mixed else "Single-model"
            ax.set_title(f"{kind} runs · {TASK_ORDERS[to]}  (n={cell.run_id.nunique()} runs)", fontsize=11)
            ax.grid(alpha=0.2)
            if i == 1:
                ax.set_xlabel("Family axis (direction of largest between-family spread)")
            if j == 0:
                ax.set_ylabel("Time axis (average round-1 → round-10 shift)")
    fig.suptitle(f"Myths on axes chosen for the question ({size}-agent runs): left-right separates the families, "
                 f"up-down is the direction myths move over the game.\nLine = family average, light → dark = round "
                 f"1 → 10; small dots = each run's family average at rounds 1 and 10", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / f"family_time_map_{size}agent.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- tests

def add(rows, claim, test, statistic, p, n, note=""):
    rows.append(dict(claim=claim, test=test, statistic=statistic, p=p, n=n, note=note))


def tests(myths, emb, curves, rng) -> pd.DataFrame:
    rows = []
    # 1. family from a round-1 myth
    for to in TASK_ORDERS:
        s = myths[(myths["round"] == 1) & (myths.task_order == to)]
        pred = cross_val_predict(LogisticRegression(max_iter=2000), emb[s.index], s.family,
                                 groups=s.run_id, cv=GroupKFold(5))
        acc = float((pred == s.family).mean())
        base = float(s.family.value_counts(normalize=True).max())
        p = stats.binomtest(int((pred == s.family).sum()), len(s), base, alternative="greater").pvalue
        add(rows, f"Round-1 myths reveal the family ({TASK_ORDERS[to]})", "logistic regression, 5 folds split by run",
            f"accuracy {acc:.3f} vs {base:.3f} largest-family guess", p, f"{len(s)} myths",
            "binomial p treats myths as independent; the run-level test below does not")
    # 2. run-level permutation of family labels (single-model runs)
    for to in TASK_ORDERS:
        s = myths[(myths["round"] == 1) & (myths.task_order == to) & (~myths.mixed)]
        runs = s.groupby("run_id")
        fam = runs.family.first().to_numpy()
        cent = np.stack([unit(emb[s.index[ix]].mean(0)) for ix in runs.indices.values()])  # positions -> row labels
        sim = cent @ cent.T
        off = ~np.eye(len(fam), dtype=bool)

        def gap(f):
            same = (f[:, None] == f[None, :]) & off
            return sim[same].mean() - sim[off & ~same].mean()

        obs = gap(fam)
        null = np.array([gap(rng.permutation(fam)) for _ in range(10000)])
        add(rows, f"Round-1 families differ, run level ({TASK_ORDERS[to]})", "permute family labels over single-model runs",
            f"within − between family similarity {obs:.3f}", (np.sum(null >= obs) + 1) / 10001, f"{len(fam)} runs")
    # 3. morals by family, and Sonnet generous by task order
    for to in TASK_ORDERS:
        s = myths[(myths["round"] == 1) & (myths.task_order == to)]
        chi = stats.chi2_contingency(pd.crosstab(s.family, s.label))
        add(rows, f"Round-1 moral depends on family ({TASK_ORDERS[to]})", "chi-square, family × moral",
            f"χ² = {chi[0]:.1f}, df = {chi[2]}", chi[1], f"{len(s)} myths")
    son = myths[(myths["round"] == 1) & (myths.family == "Sonnet")]
    tab = pd.crosstab(son.task_order, son.label == "be generous")
    add(rows, "One round of play makes Sonnet's opening less generous", "Fisher's exact, task order × generous",
        f"generous {tab.loc['myth_game', True]}/{tab.loc['myth_game'].sum()} (Myth → Game) vs "
        f"{tab.loc['game_myth', True]}/{tab.loc['game_myth'].sum()} (Game → Myth)",
        stats.fisher_exact(tab.to_numpy())[1], f"{len(son)} myths")
    # 4. single-model families drift apart: change in distance r1 -> r10 with bootstrap CI
    sing = curves[curves.group == "single"]
    for (to, size), c in sing.groupby(["task_order", "size"]):
        d1 = c[c["round"] == 1].set_index("pair")
        d10 = c[c["round"] == 10].set_index("pair")
        for pair in d1.index:
            change = d10.distance[pair] - d1.distance[pair]
            sep = (d10.ci_low[pair] > d1.ci_high[pair])
            add(rows, f"Single-model {pair} drift apart ({size} agents, {TASK_ORDERS[to]})",
                "round 10 vs round 1 distance, 95% run-bootstrap intervals",
                f"{d1.distance[pair]:.3f} → {d10.distance[pair]:.3f} ({change:+.3f})", np.nan,
                f"{d1.n_runs[pair]} runs", "intervals do not overlap" if sep else "intervals overlap")
    # 5. dyad partner convergence, per run
    dy = myths[myths.mixed & (myths["size"] == 2)]
    per_run = []
    for (comp, to, r), sel in dy.groupby(["composition", "task_order", "round"]):
        sim = emb[sel.index] @ emb[sel.index].T
        fam = sel.family.to_numpy(dtype=object)
        run = sel.run_id.to_numpy(dtype=object)
        cross = fam[:, None] != fam[None, :]
        same = run[:, None] == run[None, :]
        for rid in np.unique(run):
            mine = run[:, None] == rid
            if (cross & same & mine).any():
                per_run.append(dict(comp=comp, to=to, round=r, run=rid,
                                    diff=sim[cross & same & mine].mean() - sim[cross & ~same & mine].mean()))
    w = pd.DataFrame(per_run).pivot_table(index=["comp", "to", "run"], columns="round", values="diff")
    r10, delta = w[10], w[10] - w[1]
    add(rows, "Mixed dyads: partners' myths closer than other runs' at round 10", "sign test over runs",
        f"mean +{r10.mean():.3f} (±{r10.std():.3f}); {(r10 > 0).sum()}/{len(r10)} runs > 0",
        stats.binomtest(int((r10 > 0).sum()), len(r10)).pvalue, f"{len(r10)} runs")
    add(rows, "Mixed dyads: partner closeness grows from round 1 to 10", "Wilcoxon signed-rank over runs",
        f"mean +{delta.mean():.3f} (±{delta.std():.3f}); {(delta > 0).sum()}/{len(delta)} runs > 0",
        stats.wilcoxon(delta).pvalue, f"{len(delta)} runs")
    for (comp, to), x in r10.groupby(level=["comp", "to"]):
        add(rows, f"  {comp}, {TASK_ORDERS[to]}: partner closeness at round 10", "sign test over runs",
            f"mean +{x.mean():.3f}; {(x > 0).sum()}/{len(x)} runs > 0", stats.binomtest(int((x > 0).sum()), len(x)).pvalue,
            f"{len(x)} runs", "6 runs: the smallest possible two-sided p is 0.031")
    # 6. mixing stops the drift: round-10 family distance, single-model vs mixed runs (same pair, size, task order)
    r10c = curves[curves["round"] == 10]
    for (to, size, pair), c in r10c.groupby(["task_order", "size", "pair"]):
        if len(c) < 2:  # pair only in single-model runs (Sonnet–Gemini has no 8-agent mixed runs)
            continue
        sg, mx = c[c.group == "single"].iloc[0], c[c.group != "single"].iloc[0]
        add(rows, f"Round 10, {pair}: mixed runs closer than single-model runs ({size} agents, {TASK_ORDERS[to]})",
            "95% run-bootstrap intervals", f"single {sg.distance:.3f} vs mixed {mx.distance:.3f} ({mx.distance - sg.distance:+.3f})",
            np.nan, f"{sg.n_runs} + {mx.n_runs} runs",
            "intervals do not overlap" if mx.ci_high < sg.ci_low else "intervals overlap")
    # 7. mixed populations: family distance r1 -> r10
    mix8 = curves[(curves.group != "single") & (curves["size"] == 8)]
    for (to, pair), c in mix8.groupby(["task_order", "pair"]):
        d1, d10 = c[c["round"] == 1].iloc[0], c[c["round"] == 10].iloc[0]
        add(rows, f"8-agent mixed {pair}: family distance round 1 → 10 ({TASK_ORDERS[to]})",
            "95% run-bootstrap intervals", f"{d1.distance:.3f} → {d10.distance:.3f} ({d10.distance - d1.distance:+.3f})",
            np.nan, f"{d1.n_runs} runs",
            "intervals do not overlap" if (d10.ci_high < d1.ci_low or d10.ci_low > d1.ci_high) else "intervals overlap")
    return pd.DataFrame(rows)


def main() -> None:
    ds = linguistic_datasets.get("september")
    out = SEPTEMBER_OUT
    rng = np.random.default_rng(SEED)
    myths, emb = load(ds)
    myths["group"] = [group_of(r) for r in myths.itertuples()]

    curves = distance_curves(myths, emb, rng)
    curves.round(4).to_csv(out / "convergence_over_rounds.csv", index=False)
    plot_distance_curves(curves, ds, out)

    x_dir, y_dir = chosen_axes(myths, emb)
    for size in (8, 2):
        plot_family_time_map(myths, emb, ds, x_dir, y_dir, size, out)

    pca = PCA(2, svd_solver="full").fit(emb)
    var = pd.DataFrame([dict(view="PCA map (PC1 + PC2)", **share_kept(myths, emb, list(pca.components_))),
                        dict(view="Chosen axes (family + time)", **share_kept(myths, emb, [x_dir, y_dir]))])
    var.round(3).to_csv(out / "variance.csv", index=False)
    tests(myths, emb, curves, rng).to_csv(out / "significance.csv", index=False, float_format="%.3g")
    print(var.to_string(index=False))

    from analyses import linguistic_provenance
    linguistic_provenance.main(out)


if __name__ == "__main__":
    main()
