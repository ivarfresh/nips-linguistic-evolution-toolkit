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

    python3 analyses/myth_map_significance.py                      # September
    python3 analyses/myth_map_significance.py --dataset frontier   # Opus 5, Gemini 3.1 Pro, GPT-5.6 Sol
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
FRONTIER_OUT = ROOT / "docs/figures/myth_convergence_map_frontier_20261003"
# family pairs by position in the dataset's family order, with a colour per position, so a frontier
# pair is drawn like its September counterpart (Opus–Sol like Sonnet–GPT, and so on)
PAIR_SLOTS = [((0, 2), "#7570b3"), ((0, 1), "#1b9e77"), ((1, 2), "#d95f02")]


def family_pairs(ds):
    return [(ds.families[i], ds.families[j]) for (i, j), _ in PAIR_SLOTS]


def pair_colors(ds):
    return {f"{ds.families[i]}–{ds.families[j]}": c for (i, j), c in PAIR_SLOTS}


def unit(v):
    return v / np.linalg.norm(v)


def group_of(row, families) -> str:
    """Cell for the distance plot: single-model runs pool across families; mixed runs by the families they hold."""
    if not row.mixed:
        return "single"
    return "+".join(sorted(set(row.composition.replace("+", " ").split()) & set(families)))


# --------------------------------------------------------------------------- distances over rounds

def run_sums(myths, emb):
    """Per (task order, size, group, round, run, family): sum of unit embeddings and count."""
    keys = ["task_order", "size", "group", "round", "run_id", "family"]
    return {key: (emb[ix].sum(0), len(ix)) for key, ix in myths.groupby(keys).indices.items()}


def cross_distance(sums, runs_a, runs_b, key_a, key_b, mixed: bool) -> float:
    """Mean cosine distance over (A myth, B myth) pairs from DIFFERENT runs.

    For unit vectors the mean similarity over all pairs is (sum_A . sum_B) / (n_A n_B).
    Mixed runs hold both families, so the same-run (partner) pairs are taken out:
    single-model and mixed curves then compare the same kind of pair. Partner
    pairs are tested separately (dyad partner rows in significance.csv).
    """
    sa = sum(sums[key_a(r)][0] for r in runs_a)
    na = sum(sums[key_a(r)][1] for r in runs_a)
    sb = sum(sums[key_b(r)][0] for r in runs_b)
    nb = sum(sums[key_b(r)][1] for r in runs_b)
    dot, pairs = sa @ sb, na * nb
    if mixed:  # runs_a is runs_b (one resample); drop every same-run A x B block
        # a run drawn k times enters sa and sb k times each, so its own block sits in the product k^2 times
        from collections import Counter
        for r, k in Counter(runs_a).items():
            dot -= k * k * (sums[key_a(r)][0] @ sums[key_b(r)][0])
            pairs -= k * k * sums[key_a(r)][1] * sums[key_b(r)][1]
    return 1 - dot / pairs


def distance_curves(myths, emb, ds, rng):
    """Family-pair distance per round, with run-bootstrap samples kept for paired comparisons.

    Each bootstrap draw resamples runs once and evaluates every round on that draw
    (mixed: one draw of runs; single-model: one draw per family), so the change
    from round 1 to 10 is bootstrapped as a paired quantity.
    Returns (curves table, {(task_order, size, group, pair): array boots x rounds}).
    """
    sums = run_sums(myths, emb)
    rows, boots_by_cell = [], {}
    for (to, size, group), cell in myths.groupby(["task_order", "size", "group"]):
        mixed = group != "single"
        for a, b in family_pairs(ds):
            if mixed and not {a, b} <= set(group.split("+")):
                continue
            runs_a = cell[cell.family == a].run_id.unique()
            runs_b = cell[cell.family == b].run_id.unique()
            if len(runs_a) == 0 or len(runs_b) == 0:
                continue

            def ka(run, r):
                return (to, size, group, r, run, a)

            def kb(run, r):
                return (to, size, group, r, run, b)

            def curve(ra, rb):
                out = []
                for r in ROUNDS:
                    ra_r = [x for x in ra if (ka(x, r) in sums) and (not mixed or kb(x, r) in sums)]
                    rb_r = ra_r if mixed else [x for x in rb if kb(x, r) in sums]
                    if not ra_r or not rb_r:  # a draw of only the run missing this round's myth
                        out.append(np.nan)
                        continue
                    out.append(cross_distance(sums, ra_r, rb_r, lambda x: ka(x, r), lambda x: kb(x, r), mixed))
                return np.array(out)

            est = curve(runs_a, runs_b)
            draws = []
            for _ in range(N_BOOT):
                ra = rng.choice(runs_a, len(runs_a))
                rb = ra if mixed else rng.choice(runs_b, len(runs_b))
                draws.append(curve(ra, rb))
            draws = np.array(draws)
            boots_by_cell[(to, size, group, f"{a}–{b}")] = draws
            lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
            n_runs = len(runs_a) if mixed else f"{len(runs_a)} + {len(runs_b)}"
            for i, r in enumerate(ROUNDS):
                rows.append(dict(task_order=to, size=size, group=group, pair=f"{a}–{b}", round=r,
                                 distance=est[i], ci_low=lo[i], ci_high=hi[i], n_runs=n_runs))
    return pd.DataFrame(rows), boots_by_cell


def plot_distance_curves(curves, ds, out: Path) -> None:
    pair_color = pair_colors(ds)
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for i, size in enumerate([2, 8]):
        for j, to in enumerate(TASK_ORDERS):
            ax = axs[i, j]
            cell = curves[(curves["size"] == size) & (curves.task_order == to)]
            for (group, pair), c in cell.groupby(["group", "pair"]):
                single = group == "single"
                color = pair_color[pair]
                ax.fill_between(c["round"], c.ci_low, c.ci_high, color=color, alpha=0.10 if single else 0.18, lw=0)
                ax.plot(c["round"], c.distance, color=color, lw=2, ls=(0, (5, 3)) if single else "-", marker="o", ms=3)
            ax.set_title(f"{size} agents · {TASK_ORDERS[to]}", fontsize=11)
            ax.set_xticks(list(ROUNDS))
            ax.grid(alpha=0.25)
            if i == 1:
                ax.set_xlabel("Round")
            if j == 0:
                ax.set_ylabel("Mean cosine distance between\na myth of each family (768-d)")
            # legend: colour = family pair, line style = run type (drawn without markers so the dashes show)
            pairs = list(dict.fromkeys(cell.pair))
            handles = [plt.Line2D([], [], color=pair_color[p], lw=3, label=p) for p in pairs]
            handles += [plt.Line2D([], [], color="0.35", lw=2, ls=(0, (5, 3)), label="single-model runs (families never meet)"),
                        plt.Line2D([], [], color="0.35", lw=2, label="mixed runs (families play together)")]
            ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left", handlelength=4)
    fig.suptitle("Do the families' myths converge? Distance between a myth of one family and a myth of the other, "
                 "per round (pairs from different runs only).\nDashed: families in separate single-model runs. Solid: "
                 "families in mixed runs. Bands: 95% intervals from resampling runs.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "convergence_over_rounds.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- map with chosen axes

def chosen_axes(myths, emb, ds):
    """x: the direction holding the most between-family spread (top axis of the size-weighted family centroids).
    y: mean within-family round-1 -> round-10 shift, made orthogonal to x."""
    mu = emb.mean(0)
    g = myths.groupby("family").indices
    dev = np.stack([np.sqrt(len(ix)) * (emb[ix].mean(0) - mu) for ix in g.values()])
    x_dir = np.linalg.svd(dev, full_matrices=False)[2][0]
    shifts = [unit(emb[(myths.family == f).to_numpy() & (myths["round"] == ROUNDS[-1]).to_numpy()].mean(0)
                   - emb[(myths.family == f).to_numpy() & (myths["round"] == ROUNDS[0]).to_numpy()].mean(0))
              for f in ds.families]
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
    pad = 0.04
    xr = (myths.fx.min() - pad, myths.fx.max() + pad)
    yr = (myths.ty.min() - pad, myths.ty.max() + pad)
    gx, gy = np.mgrid[xr[0]:xr[1]:200j, yr[0]:yr[1]:200j]
    fig, axs = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    for i, mixed in enumerate([False, True]):
        for j, to in enumerate(TASK_ORDERS):
            ax = axs[i, j]
            cell = myths[(myths["size"] == size) & (myths.mixed == mixed) & (myths.task_order == to)]
            end = cell[cell["round"] == ROUNDS[-1]]  # background: where this panel's round-10 myths end up
            kde = stats.gaussian_kde(np.vstack([end.fx, end.ty]), bw_method=0.25)
            ax.contourf(gx, gy, kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape), levels=12, cmap="Blues", alpha=0.85)
            ax.set_xlim(*xr)
            ax.set_ylim(*yr)
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
                 f"1 → 10; small dots = each run's family average at rounds 1 and 10; blue background = where round-10 myths end up",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / f"family_time_map_{size}agent.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- tests

def add(rows, claim, test, statistic, p, n, note=""):
    rows.append(dict(claim=claim, test=test, statistic=statistic, p=p, n=n, note=note))


def tests(myths, emb, curves, boots, ds, rng) -> pd.DataFrame:
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
            f"within − between family similarity {obs:.3f}", (np.sum(null >= obs) + 1) / 10001, f"{len(fam)} runs",
            "1e-4 is the smallest p 10,000 shuffles can give")
    # 3. morals by family, and Sonnet generous by task order
    for to in TASK_ORDERS:
        s = myths[(myths["round"] == 1) & (myths.task_order == to)]
        chi = stats.chi2_contingency(pd.crosstab(s.family, s.label))
        add(rows, f"Round-1 moral depends on family ({TASK_ORDERS[to]})", "chi-square, family × moral",
            f"χ² = {chi[0]:.1f}, df = {chi[2]}", chi[1], f"{len(s)} myths")
    for fam in ds.families:
        one = myths[(myths["round"] == 1) & (myths.family == fam)]
        tab = pd.crosstab(one.task_order, one.label == "be generous").reindex(columns=[False, True], fill_value=0)
        add(rows, f"One round of play changes {fam}'s 'be generous' openings", "Fisher's exact, task order × generous",
            f"generous {tab.loc['myth_game', True]}/{tab.loc['myth_game'].sum()} (Myth → Game) vs "
            f"{tab.loc['game_myth', True]}/{tab.loc['game_myth'].sum()} (Game → Myth)",
            stats.fisher_exact(tab.to_numpy())[1], f"{len(one)} myths")
    # 4. change in family distance from round 1 to 10, paired run bootstrap, every cell
    for (to, size, group, pair), draws in boots.items():
        c = curves[(curves.task_order == to) & (curves["size"] == size) & (curves.group == group) & (curves.pair == pair)]
        d1, d10 = c[c["round"] == 1].distance.iloc[0], c[c["round"] == 10].distance.iloc[0]
        lo, hi = np.nanpercentile(draws[:, -1] - draws[:, 0], [2.5, 97.5])
        kind = "single-model" if group == "single" else "mixed"
        add(rows, f"{kind.capitalize()} {pair}: family distance round 1 → 10 ({size} agents, {TASK_ORDERS[to]})",
            "paired run bootstrap of the change (different-run pairs)",
            f"{d1:.3f} → {d10:.3f} ({d10 - d1:+.3f}); 95% interval [{lo:+.3f}, {hi:+.3f}]", np.nan,
            f"{c.n_runs.iloc[0]} runs", "interval excludes 0" if lo > 0 or hi < 0 else "interval includes 0")
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
    # 6. round 10: single-model vs mixed distance, same pair, size and task order (independent run sets)
    for (to, size, group, pair), mx in boots.items():
        if group == "single" or (to, size, "single", pair) not in boots:
            continue
        sg = boots[(to, size, "single", pair)]
        c = curves[(curves.task_order == to) & (curves["size"] == size) & (curves.pair == pair) & (curves["round"] == 10)]
        d_s, d_m = c[c.group == "single"].distance.iloc[0], c[c.group != "single"].distance.iloc[0]
        lo, hi = np.nanpercentile(mx[:, -1] - sg[:, -1], [2.5, 97.5])
        add(rows, f"Round 10, {pair}: mixed vs single-model distance ({size} agents, {TASK_ORDERS[to]})",
            "run bootstrap of the difference (different-run pairs)",
            f"single {d_s:.3f} vs mixed {d_m:.3f} ({d_m - d_s:+.3f}); 95% interval [{lo:+.3f}, {hi:+.3f}]", np.nan,
            f"{c[c.group == 'single'].n_runs.iloc[0]} vs {c[c.group != 'single'].n_runs.iloc[0]} runs",
            "interval excludes 0" if lo > 0 or hi < 0 else "interval includes 0")
    return pd.DataFrame(rows)


def september_pools() -> tuple[dict, dict]:
    """The September myth-run finals as (pools, allowed differences), as linguistic_provenance records them."""
    from scripts import analyze_mixed_model_dyads as dyads
    from scripts import analyze_mixed_model_populations as populations
    used = {(ROOT / p).resolve() for p in load(linguistic_datasets.get("september"))[0].path.unique()}
    dm, dsep = dyads.final_paths()
    pm, psep = populations.final_paths()
    mixed = [p.resolve() for p in dm + pm if p.resolve() in used]
    single = [p.resolve() for p in dsep + psep if p.resolve() in used]
    if {*mixed, *single} != used:
        raise SystemExit("September myth runs and run finals disagree")
    return {"september_mixed": mixed, "september_homogeneous": single}, {**dyads.ALLOWED, **populations.ALLOWED}


def frontier_provenance(out: Path, myths_paths) -> None:
    """provenance.json for the frontier folder: the audited frontier finals behind its myths, plus the
    September myth runs when the joint round-1 figure (myth_map_joint_round1.py) is in the folder."""
    import json
    from src.experiment_condition import output_provenance
    from scripts.analyze_frontier_main_mixed_20260928 import load as frontier_load
    from scripts.analyze_frontier_update_20260928 import ALLOWED, POOL_REASON
    _, homo, mixed = frontier_load()  # receipt-audited finals
    used = {(ROOT / p).resolve() for p in myths_paths}
    homo = [p.resolve() for p in homo if p.resolve() in used]
    mixed = [p.resolve() for p in mixed if p.resolve() in used]
    if {*homo, *mixed} != used:
        raise SystemExit(f"{len(used - {*homo, *mixed})} frontier myth runs are not audited finals")
    pools, allowed, reason = {"frontier_homogeneous": homo, "frontier_mixed": mixed}, dict(ALLOWED), POOL_REASON
    if (out / "joint_round1.png").exists():
        sep, sep_allowed = september_pools()
        pools.update(sep)
        allowed.update(sep_allowed)
        reason += (" The September pools (Sonnet 4.5, Gemini 3.7 Flash, GPT-5 Nano) are read only by joint_round1.*, "
                   "which places both corpora's round-1 myths on one map; no statistic pools them.")
    outputs = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "provenance.json" and not p.name.startswith("."))
    runs = [p for paths in pools.values() for p in paths]
    document = output_provenance(runs, outputs, allowed, output_root=out, pools=pools, pool_reason=reason)
    (out / "provenance.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"provenance: {len(runs)} runs, {len(outputs)} outputs -> {out / 'provenance.json'}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=["september", "frontier"], default="september")
    args = ap.parse_args()
    ds = linguistic_datasets.get(args.dataset)
    out = SEPTEMBER_OUT if ds.name == "september" else FRONTIER_OUT
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    myths, emb = load(ds)
    myths["group"] = [group_of(r, ds.families) for r in myths.itertuples()]

    curves, boots = distance_curves(myths, emb, ds, rng)
    curves.round(4).to_csv(out / "convergence_over_rounds.csv", index=False)
    plot_distance_curves(curves, ds, out)

    x_dir, y_dir = chosen_axes(myths, emb, ds)
    for size in (8, 2):
        plot_family_time_map(myths, emb, ds, x_dir, y_dir, size, out)

    pca = PCA(2, svd_solver="full").fit(emb)
    var = pd.DataFrame([dict(view="PCA map (PC1 + PC2)", **share_kept(myths, emb, list(pca.components_))),
                        dict(view="Chosen axes (family + time)", **share_kept(myths, emb, [x_dir, y_dir]))])
    var.round(3).to_csv(out / "variance.csv", index=False)
    tests(myths, emb, curves, boots, ds, rng).to_csv(out / "significance.csv", index=False, float_format="%.3g")
    print(var.to_string(index=False))

    if ds.name == "september":
        from analyses import linguistic_provenance
        linguistic_provenance.main(out)
    else:
        frontier_provenance(out, myths.path.unique())


if __name__ == "__main__":
    main()
