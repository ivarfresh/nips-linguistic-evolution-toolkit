#!/usr/bin/env python3
"""Lévi-Strauss structure of the myths at round 1 and how it changes over rounds.

Reads the judge output of analyses/myth_structure_judge.py (one structure per
myth: mythemes, oppositions, resolution, mediator, arc, role inversion) and
asks four questions, never pooling task orders or the September and frontier
corpora:

1. Round 1. Which oppositions, resolutions, mediators and arcs do the families
   start from? (single-model runs; run-level means)
2. Change by round. How the shares move from round 1 to round 10, per family,
   size and task order. Round 1 is written under a different prompt from
   rounds 2-10 (the first myth has no partner myth to read), so a step between
   rounds 1 and 2 is partly the prompt.
3. Columns. Lévi-Strauss lined mythemes up into "columns" that say the same
   thing. Here every mytheme's abstract-role version (no character names) is
   embedded (all-mpnet-base-v2) and clustered once over the whole corpus,
   with k fixed by silhouette on a random sample before any round is looked
   at. A myth "has" a column if any of its mythemes falls in it.
4. Transmission. Does an agent take up the oppositions and columns of the
   myth it was SHOWN, more than those of a myth it was not shown? The control
   myths are those of analyses/linguistic_uptake.py (8-agent: other same-family
   agents of the same run and round; dyads: the same-family agent of other runs
   in the same cell), so shared prompts, models and drift cancel. Adoption
   counts only features absent from the agent's own earlier myths, because the
   prompt tells it to build on its own previous myth. Only rounds 1, 2, 5, 6, 9
   and 10 are coded, so the shown-myth test covers rounds 2, 6 and 10, and an
   agent's "earlier myths" are its earlier CODED myths.
   Plus convergence: how similar myths within a run are, minus the similarity
   of myths from different runs of the same cell at the same round.

No API calls.
  python3 analyses/myth_structure_analysis.py --dataset september_n10
  python3 analyses/myth_structure_analysis.py --dataset frontier
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses._shared import cached_embeddings, configure_matplotlib  # noqa: E402
from analyses import linguistic_datasets  # noqa: E402
from analyses.linguistic_uptake import holm, null_candidates  # noqa: E402

JUDGE = "deepseek__deepseek-v4-flash_r1-2-5-6-9-10"  # judge model + coded rounds (1,2 / 5,6 / 9,10)
MIN_WORDS = 20
K_GRID = [8, 10, 12, 15, 20, 25, 30]
CAT_ORDER = ["giving_hoarding", "trust_fear", "cooperation_betrayal", "punishment_forgiveness",
             "individual_community", "scarcity_abundance", "clarity_distortion", "honesty_deception",
             "order_chaos", "life_death", "high_low", "light_dark", "human_divine", "nature_culture", "other"]
BINARY = {  # measure -> (column, value) ; share of myths where column == value
    "resolution: one pole wins": ("main_resolution", "one_pole_wins"),
    "resolution: balanced middle": ("main_resolution", "balanced"),
    "resolution: mediated": ("main_resolution", "mediated"),
    "mediator present": ("mediator_present", True),
    "arc: stable order": ("arc", "stable"),
    "arc: improves": ("arc", "improves"),
    "arc: restored": ("arc", "restored"),
    "role inversion": ("role_inversion", True),
}
# Holm correction runs over these headline measures only. With 10 runs per cell the smallest exact
# Wilcoxon p is 0.002, so correcting over every column and category (~70 tests) could never reach
# 0.05; the first run (2026-10-10) showed this. Mediator and resolution are left out because the
# DeepSeek judge counts the gift-carrying river as a mediator (60% agreement with GLM on 20 myths).
# Columns and the remaining categories keep raw p only and are exploratory.
HEADLINE = ["main: giving_hoarding", "main: trust_fear", "main: cooperation_betrayal", "any: clarity_distortion",
            "any: punishment_forgiveness", "arc: improves", "arc: stable order", "role inversion"]
TASK_LABEL = {"game_myth": "Game→Myth", "myth_game": "Myth→Game"}


# --------------------------------------------------------------------------- data

def load(ds, tag: str) -> tuple[pd.DataFrame, list[dict]]:
    myths = pd.read_csv(ds.data / "myths.csv")
    myths["valid"] = myths["n_words"] >= MIN_WORDS
    myths = myths.reset_index(drop=True)
    flat = pd.read_csv(ds.data / f"myth_structure_{tag}.csv")
    flat = flat[flat["status"] == "ok"]
    keep = ["run_id", "round", "agent", "main_category", "main_resolution", "mediator_present",
            "mediator_type", "arc", "role_inversion"]
    myths = myths.merge(flat[keep], on=["run_id", "round", "agent"], how="left")
    myths["coded"] = myths["main_category"].notna()
    recs = {}
    with (ds.data / f"myth_structure_{tag}.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if r["status"] == "ok":
                recs[(r["run_id"], r["round"], r["agent"])] = r["structure"]
    structs = [recs.get(k) for k in zip(myths["run_id"], myths["round"], myths["agent"])]
    myths["valid"] &= myths["coded"]
    for c in CAT_ORDER:
        myths[f"has_{c}"] = [bool(s) and any(o["category"] == c for o in s["oppositions"]) for s in structs]
    return myths, structs


# --------------------------------------------------------------------------- columns

def fit_columns(ds, myths: pd.DataFrame, structs: list[dict], figs: Path) -> list[frozenset]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    owner, texts = [], []
    for i, s in enumerate(structs):
        for m in (s or {}).get("mythemes", []):
            owner.append(i)
            texts.append(m["roles"].lower().strip())
    emb = cached_embeddings(ds.data / "embeddings_mythemes_roles_mpnet.npy", texts, batch_size=256,
                            show_progress_bar=True)
    rng = np.random.default_rng(0)
    sample = rng.choice(len(texts), size=min(8000, len(texts)), replace=False)
    sil = {k: silhouette_score(emb[sample], KMeans(k, n_init=4, random_state=0).fit_predict(emb[sample]),
                               metric="cosine") for k in K_GRID}
    k = max(sil, key=sil.get)
    print("silhouette by k:", {kk: round(v, 4) for kk, v in sil.items()}, "-> k =", k)
    km = KMeans(k, n_init=10, random_state=0).fit(emb)
    labels = km.labels_
    # exemplars: the mythemes nearest each centroid, plus the most frequent exact strings
    rows = []
    cent = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    texts_arr = np.array(texts)
    for c in range(k):
        idx = np.flatnonzero(labels == c)
        near = idx[np.argsort(-(emb[idx] @ cent[c]))[:8]]
        common = pd.Series(texts_arr[idx]).value_counts().head(5)
        rows.append({"column": c, "n_mythemes": len(idx), "share_of_myths": None,
                     "nearest_centroid": " | ".join(dict.fromkeys(texts_arr[near])),
                     "most_common": " | ".join(f"{t} ({n})" for t, n in common.items())})
    cols: list[set] = [set() for _ in range(len(myths))]
    for i, c in zip(owner, labels):
        cols[i].add(int(c))
    ex = pd.DataFrame(rows)
    has = np.array([[c in s for c in range(k)] for s in cols])[myths["valid"].to_numpy()]
    ex["share_of_myths"] = has.mean(axis=0).round(3)
    ex.sort_values("share_of_myths", ascending=False).to_csv(figs / "columns_exemplars.csv", index=False)
    pd.Series(sil, name="silhouette").rename_axis("k").to_csv(figs / "columns_silhouette.csv")
    return [frozenset(s) for s in cols]


# --------------------------------------------------------------------------- 1 + 2

def per_run_shares(myths: pd.DataFrame, by: list[str], col_sets: list[frozenset], k: int) -> pd.DataFrame:
    df = myths[myths["valid"]].copy()
    measures = {}
    for c in CAT_ORDER:
        measures[f"main: {c}"] = df["main_category"] == c
        measures[f"any: {c}"] = df[f"has_{c}"]
    for name, (col, val) in BINARY.items():
        measures[name] = df[col] == val
    for c in range(k):
        measures[f"column {c}"] = [c in col_sets[i] for i in df.index]
    wide = pd.DataFrame(measures, index=df.index).astype(float)
    wide = pd.concat([df[by + ["run_id"]], wide], axis=1)
    return wide.groupby(by + ["run_id"]).mean().reset_index()


def summarize(per_run: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    vals = [c for c in per_run.columns if c not in by + ["run_id"]]
    long = per_run.melt(id_vars=by + ["run_id"], value_vars=vals, var_name="measure", value_name="share")
    g = long.groupby(by + ["measure"])["share"]
    out = g.agg(mean="mean", sd="std", n_runs="count").reset_index()
    out["se"] = out["sd"] / np.sqrt(out["n_runs"])
    return out


def round_trend_tests(per_run: pd.DataFrame, cell: list[str]) -> pd.DataFrame:
    """Per run, the slope of each share over coded rounds 2-10 (the window with one prompt), then a
    Wilcoxon test of the slopes against zero per cell; Holm across the HEADLINE measures within each
    cell (other measures: raw p only). Also a Wilcoxon test of round 10 against round 1, per run."""
    vals = [c for c in per_run.columns if c not in cell + ["run_id", "round"]]
    late = per_run[per_run["round"] >= 2]
    rows = []
    for keys, g in late.groupby(cell):
        keys = keys if isinstance(keys, tuple) else (keys,)
        slopes = {}
        for run, gr in g.groupby("run_id"):
            if gr["round"].nunique() < 3:
                continue
            x = gr["round"].to_numpy(float)
            slopes[run] = {m: np.polyfit(x, gr[m].to_numpy(float), 1)[0] for m in vals}
        s = pd.DataFrame(slopes).T
        r1 = per_run[(per_run["round"] == 1)].merge(pd.DataFrame([dict(zip(cell, keys))]), on=cell)
        r10 = per_run[(per_run["round"] == 10)].merge(pd.DataFrame([dict(zip(cell, keys))]), on=cell)
        block = []
        for m in vals:
            v = s[m].dropna() if m in s else pd.Series(dtype=float)
            p = stats.wilcoxon(v).pvalue if len(v) >= 5 and (v != 0).any() else np.nan
            paired = r1[["run_id", m]].merge(r10[["run_id", m]], on="run_id", suffixes=("_1", "_10")).dropna()
            d = paired[f"{m}_10"] - paired[f"{m}_1"]
            p_1v10 = stats.wilcoxon(d).pvalue if len(d) >= 5 and (d != 0).any() else np.nan
            block.append({**dict(zip(cell, keys)), "measure": m, "headline": m in HEADLINE, "n_runs": len(v),
                          "round1_mean": r1[m].mean(), "round1_sd": r1[m].std(),
                          "round10_mean": r10[m].mean(), "round10_sd": r10[m].std(),
                          "slope_per_round_2to10": v.mean(), "slope_sd": v.std(), "p": p,
                          "p_round1_vs_10": p_1v10})
        block = pd.DataFrame(block)
        head = block["headline"].to_numpy()
        for col in ("p", "p_round1_vs_10"):
            block[f"{col}_holm"] = np.nan
            block.loc[head, f"{col}_holm"] = holm(block.loc[head, col].to_numpy())
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- 4

def own_history_sets(myths: pd.DataFrame, feats: list[frozenset]) -> list[frozenset]:
    hist: list[frozenset] = [frozenset()] * len(myths)
    for _, grp in myths.groupby(["run_id", "agent"]):
        seen: set = set()
        for i in grp.sort_values("round").index:
            hist[i] = frozenset(seen)
            seen |= feats[i]
    return hist


def transmission(myths: pd.DataFrame, cat_sets, col_sets) -> pd.DataFrame:
    cands = null_candidates(myths)
    hist_cat, hist_col = own_history_sets(myths, cat_sets), own_history_sets(myths, col_sets)
    rows = []
    for i, js in cands.items():
        p, nulls = js[0], js[1:]

        def adoption(feats, hist, j):
            new, pool = feats[i] - hist[i], feats[j] - hist[i]
            return len(new & pool) / len(pool) if pool else np.nan

        def match(j):
            return float(myths.at[i, "main_category"] == myths.at[j, "main_category"])

        def jacc(feats, j):
            u = feats[i] | feats[j]
            return len(feats[i] & feats[j]) / len(u) if u else np.nan

        row = myths.loc[i]
        rec = {"run_id": row.run_id, "size": row["size"], "mixed": row.mixed, "family": row.family,
               "task_order": row.task_order, "round": row["round"],
               "same_family_parent": myths.at[p, "family"] == row.family}
        for name, fn in {"main_opposition_match": match,
                         "opposition_adoption": lambda j: adoption(cat_sets, hist_cat, j),
                         "column_adoption": lambda j: adoption(col_sets, hist_col, j),
                         "column_jaccard": lambda j: jacc(col_sets, j)}.items():
            rec[f"{name}_shown"] = fn(p)
            rec[f"{name}_null"] = np.nanmean([fn(j) for j in nulls]) if nulls else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def transmission_summary(children: pd.DataFrame) -> pd.DataFrame:
    metrics = ["main_opposition_match", "opposition_adoption", "column_adoption", "column_jaccard"]
    for m in metrics:
        children[f"{m}_excess"] = children[f"{m}_shown"] - children[f"{m}_null"]
    by = ["size", "mixed", "task_order"]
    per_run = children.groupby(by + ["run_id"])[[f"{m}_{s}" for m in metrics for s in ("shown", "null", "excess")]].mean()
    rows = []
    for keys, g in per_run.reset_index().groupby(by):
        for m in metrics:
            v = g[f"{m}_excess"].dropna()
            rows.append({**dict(zip(by, keys)), "metric": m, "n_runs": len(v),
                         "shown_mean": g[f"{m}_shown"].mean(), "null_mean": g[f"{m}_null"].mean(),
                         "excess_mean": v.mean(), "excess_sd": v.std(),
                         "ci_half": stats.t.ppf(0.975, len(v) - 1) * v.std() / np.sqrt(len(v)) if len(v) > 1 else np.nan,
                         "runs_positive": int((v > 0).sum()),
                         "p": stats.wilcoxon(v).pvalue if len(v) >= 5 and (v != 0).any() else np.nan})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for m in metrics:
        sel = out["metric"] == m
        out.loc[sel, "p_holm"] = holm(out.loc[sel, "p"].to_numpy())
    return out


def convergence(myths: pd.DataFrame, col_sets) -> pd.DataFrame:
    """Mean pairwise column-set Jaccard within a run, minus between runs of the same cell, per round."""
    df = myths[myths["valid"] & ~myths["mixed"]]

    def jac(a, b):
        u = a | b
        return len(a & b) / len(u) if u else np.nan

    rows = []
    for (fam, size, task, rnd), g in df.groupby(["family", "size", "task_order", "round"]):
        runs = {r: gr.index.to_list() for r, gr in g.groupby("run_id")}
        within = {r: np.nanmean([jac(col_sets[a], col_sets[b]) for a, b in itertools.combinations(ix, 2)])
                  for r, ix in runs.items() if len(ix) > 1}
        for r, w in within.items():
            others = [j for rr, ix in runs.items() if rr != r for j in ix]
            between = np.nanmean([jac(col_sets[a], col_sets[b]) for a in runs[r] for b in others]) if others else np.nan
            rows.append({"family": fam, "size": size, "task_order": task, "round": rnd, "run_id": r,
                         "within": w, "between": between, "excess": w - between})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- plots

def plot_round1(summary: pd.DataFrame, families, figs: Path) -> None:
    import matplotlib.pyplot as plt
    r1 = summary[(summary["round"] == 1)]
    meas = [f"any: {c}" for c in CAT_ORDER] + list(BINARY)
    sizes = sorted(r1["size"].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes) + 1.5, 7.5), sharey=True)
    for ax, size in zip(np.atleast_1d(axes), sizes):
        cols = [(f, t) for f in families for t in TASK_LABEL]
        mat = np.full((len(meas), len(cols)), np.nan)
        for j, (f, t) in enumerate(cols):
            s = r1[(r1["family"] == f) & (r1["size"] == size) & (r1["task_order"] == t)].set_index("measure")["mean"]
            mat[:, j] = [s.get(m, np.nan) for m in meas]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for (a, b), v in np.ndenumerate(mat):
            if not np.isnan(v):
                ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 else "black")
        short = {"game_myth": "G→M", "myth_game": "M→G"}
        ax.set_xticks(range(len(cols)), [f"{f}\n{short[t]}" for f, t in cols], fontsize=7)
        ax.set_yticks(range(len(meas)), [m.replace("any: ", "opposition: ").replace("_", " / ") for m in meas],
                      fontsize=8)
        ax.set_title("dyads" if size == 2 else "populations", fontsize=10)
        ax.axhline(len(CAT_ORDER) - 0.5, color="k", lw=0.8)
    fig.colorbar(im, ax=axes, shrink=0.5, label="share of round-1 myths (mean over runs)\nG→M = Game→Myth, M→G = Myth→Game")
    fig.savefig(figs / "round1_structure.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_trends(summary: pd.DataFrame, families, colors, figs: Path, measures: list[str], name: str) -> None:
    import matplotlib.pyplot as plt
    sizes = sorted(summary["size"].unique())
    fig, axes = plt.subplots(len(measures), len(families) * len(sizes),
                             figsize=(2.3 * len(families) * len(sizes), 1.7 * len(measures)),
                             sharex=True, sharey="row", squeeze=False)
    for col, (size, fam) in enumerate(itertools.product(sizes, families)):
        for row, m in enumerate(measures):
            ax = axes[row, col]
            for t, ls in (("game_myth", "-"), ("myth_game", "--")):
                s = summary[(summary["family"] == fam) & (summary["size"] == size) & (summary["task_order"] == t)
                            & (summary["measure"] == m)].sort_values("round")
                if s.empty:
                    continue
                ax.plot(s["round"], s["mean"], ls, color=colors[fam], lw=1.4, label=TASK_LABEL[t])
                ax.fill_between(s["round"], s["mean"] - s["se"], s["mean"] + s["se"], color=colors[fam], alpha=0.15)
            ax.axvline(1.5, color="0.7", lw=0.6)
            if row == 0:
                ax.set_title(f"{fam}, {'dyads' if size == 2 else 'populations'}", fontsize=8)
            if col == 0:
                ax.set_ylabel(m.replace("main: ", "central: ").replace("_", " / "), fontsize=7)
            ax.tick_params(labelsize=6)
    axes[0, 0].legend(fontsize=6, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("round", fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_transmission(ts: pd.DataFrame, figs: Path) -> None:
    import matplotlib.pyplot as plt
    metrics = ["main_opposition_match", "opposition_adoption", "column_adoption"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.4), sharey=False)
    for ax, m in zip(axes, metrics):
        s = ts[ts["metric"] == m].reset_index(drop=True)
        labels = [f"{'dyad' if r['size'] == 2 else 'pop'}{' mixed' if r['mixed'] else ''}\n{TASK_LABEL[r['task_order']]}"
                  for _, r in s.iterrows()]
        ax.bar(range(len(s)), s["excess_mean"], yerr=s["ci_half"], color="0.55", capsize=3)
        for x, (_, r) in enumerate(s.iterrows()):
            p = r["p_holm"]
            mark = "n/a" if np.isnan(p) else "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
            ax.text(x, max(r["excess_mean"] + (r["ci_half"] or 0), 0) + 0.002, mark, ha="center", fontsize=7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(s)), labels, fontsize=6)
        ax.set_title(m.replace("_", " "), fontsize=9)
    axes[0].set_ylabel("shown myth minus unseen myth\n(run means, 95% CI)", fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "transmission_shown_vs_unseen.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(conv: pd.DataFrame, families, colors, figs: Path) -> None:
    import matplotlib.pyplot as plt
    sizes = sorted(conv["size"].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.5 * len(sizes), 3.2), sharey=True, squeeze=False)
    for ax, size in zip(axes[0], sizes):
        for fam in families:
            for t, ls in (("game_myth", "-"), ("myth_game", "--")):
                g = conv[(conv["family"] == fam) & (conv["size"] == size) & (conv["task_order"] == t)]
                if g.empty:
                    continue
                s = g.groupby("round")["excess"].agg(["mean", "std", "count"])
                se = s["std"] / np.sqrt(s["count"])
                ax.plot(s.index, s["mean"], ls, color=colors[fam], label=f"{fam} {TASK_LABEL[t]}")
                ax.fill_between(s.index, s["mean"] - se, s["mean"] + se, color=colors[fam], alpha=0.12)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title("dyads" if size == 2 else "populations", fontsize=9)
        ax.set_xlabel("round")
    axes[0, 0].set_ylabel("column overlap within run\nminus between runs (Jaccard)", fontsize=8)
    axes[0, -1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "convergence_columns.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=sorted(linguistic_datasets.DATASETS), default="september_n10")
    ap.add_argument("--tag", help="judge-output tag (default: <dataset>_<JUDGE>); a non-default tag "
                    "writes figures to a scratch folder under the data directory")
    args = ap.parse_args()
    ds = linguistic_datasets.get(args.dataset)
    tag = args.tag or f"{ds.name}_{JUDGE}"
    figs = (ROOT / "docs/figures/myth_structure_20261010" / ds.name if not args.tag
            else ds.data / f"myth_structure_scratch_{args.tag}")
    figs.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    myths, structs = load(ds, tag)
    coded_rounds = sorted(myths.loc[myths["coded"], "round"].unique())
    in_scope = myths["round"].isin(coded_rounds) & (myths["n_words"] >= MIN_WORDS)
    print(f"{ds.name}: {myths['valid'].sum()} coded myths in rounds {coded_rounds}; "
          f"{(in_scope & ~myths['coded']).sum()} myths in those rounds without a valid code")

    col_sets = fit_columns(ds, myths, structs, figs)
    k = 1 + max(c for s in col_sets for c in s)
    cat_sets = [frozenset(o["category"] for o in s["oppositions"]) if s else frozenset() for s in structs]

    # 1 + 2: single-model runs by family; mixed runs by composition, separately
    homo = myths[~myths["mixed"]]
    cell = ["family", "size", "task_order"]
    per_run = per_run_shares(homo, cell + ["round"], col_sets, k)
    summary = summarize(per_run, cell + ["round"])
    summary.to_csv(figs / "by_round_single_model.csv", index=False)
    trends = round_trend_tests(per_run, cell)
    trends.to_csv(figs / "trend_tests_single_model.csv", index=False)
    mixed = myths[myths["mixed"]]
    if len(mixed):
        pr_mixed = per_run_shares(mixed, ["composition", "family", "size", "task_order", "round"], col_sets, k)
        summarize(pr_mixed, ["composition", "family", "size", "task_order", "round"]).to_csv(
            figs / "by_round_mixed.csv", index=False)

    fams = [f for f in ds.families if f in set(homo["family"])]
    plot_round1(summary, fams, figs)
    top_cats = [f"main: {c}" for c in CAT_ORDER
                if summary[summary["measure"] == f"main: {c}"]["mean"].max() >= 0.10]
    plot_trends(summary, fams, ds.colors, figs, top_cats, "trend_central_opposition")
    plot_trends(summary, fams, ds.colors, figs, list(BINARY), "trend_resolution_mediator_arc")
    top_cols = (summary[summary["measure"].str.startswith("column ")].groupby("measure")["mean"].mean()
                .sort_values(ascending=False).index[:10].to_list())
    plot_trends(summary, fams, ds.colors, figs, top_cols, "trend_columns_top10")

    # 4: convergence and transmission
    conv = convergence(myths, col_sets)
    conv.to_csv(figs / "convergence_columns_per_run.csv", index=False)
    plot_convergence(conv, fams, ds.colors, figs)
    children = transmission(myths, cat_sets, col_sets)
    if children.empty:
        raise SystemExit("no coded child-parent pairs; transmission skipped")
    children.to_csv(ds.data / f"myth_structure_transmission_children_{tag}.csv", index=False)
    ts = transmission_summary(children)
    ts.to_csv(figs / "transmission_summary.csv", index=False)
    plot_transmission(ts, figs)
    print(f"-> {figs}")


if __name__ == "__main__":
    main()
