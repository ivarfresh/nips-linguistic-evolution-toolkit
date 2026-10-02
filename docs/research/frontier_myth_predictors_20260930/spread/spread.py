#!/usr/bin/env python3
"""Moral and word spread, September vs frontier (lens: spread, 2026-09-30).

Ports the September item-1 and item-3 analyses (analyses/linguistic_uptake.py,
analyses/moral_carryover.py) to any linguistic dataset and splits every result by
setting x population x family x task order. It calls the scripts' functions, never
their main(), so nothing is written to the shared data directories; all output goes to
<this folder>/<dataset>/.

  python3 spread.py --dataset september      # reproduces the committed September numbers, then writes rows
  python3 spread.py --dataset frontier
  python3 spread.py --dataset frontier --labels moral_labels_deepseek__deepseek-v4-flash.csv   # robustness

Measures (child = a myth written after being shown another agent's myth; parent = that
shown myth; unseen = a comparable same-family myth from the parent's round the child
was not shown, as in September):
  1. Word adoption: share of the parent's words new to the child that the child now uses,
     minus the same for unseen myths.
  2. Moral label uptake: P(child label == parent label) minus the same for unseen myths.
     Run-level Wilcoxon (September method) plus a within-stratum permutation test that
     re-draws the parent among the unseen candidates (usable when a stratum has 5 runs,
     where the Wilcoxon floor is p = 0.0625).
     Future-myth placebo: the parent author's NEXT myth (written in the child's round,
     never shown to the child) against unseen same-family myths of that round. In 8-agent
     runs the shown author is last round's partner, so author and child have just played
     each other; the future myth shares that game but not the shown text. A placebo about as
     large as the shown effect therefore means the match comes from the shared game, not
     from reading. (Copying in both directions reaches the future myth only second-hand.)
     R3 (8-agent myth->game): shown author != current partner, excess regressed on the
     t-1 pair game (author sent/5 or returned, reader's move, author-was-investor), SE by
     run; the intercept (centred covariates) is the adjusted uptake.
  3. Carryover (per family): own latest and shown generous-vs-fair label -> next send/5 or
     return proportion, own last move in that role, agent-within-run + round FE, SE by run.
     8-agent: rows whose shown author is the current partner dropped.
  4. Reverse: cooperation in a game -> P(next myth 'be generous'), own previous label,
     agent-within-run + round FE.
  5. Label shares by family x population x task order x round; generous share round 10
     minus round 1 per run.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import warnings

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="frontier", choices=["september", "frontier"])
ap.add_argument("--labels", default="moral_labels_z-ai__glm-5.2.csv")
ap.add_argument("--perms", type=int, default=2000)
ap.add_argument("--seed", type=int, default=20260930)
ap.add_argument("--skip-pooled-carryover", action="store_true", help="skip the slow all-family carryover tables")
ARGS = ap.parse_args()
os.environ["LINGUISTIC_DATASET"] = ARGS.dataset  # _DS is resolved at import

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from analyses import linguistic_uptake as lu  # noqa: E402
from analyses import moral_carryover as mc  # noqa: E402

TAG = "" if "glm" in ARGS.labels else "_" + ARGS.labels.replace("moral_labels_", "").replace(".csv", "")
OUT = HERE / ARGS.dataset
OUT.mkdir(parents=True, exist_ok=True)
lu.FIGS = mc.FIGS = OUT  # figures from the ported plot functions land here, never in DATA/figs
DATA = lu.DATA  # read only
FAMILIES = list(lu.FAMILIES)
ORDER = {"myth_game": "myth→game", "game_myth": "game→myth"}
GEN, FAIR = "be generous", "be fair"
SEPT_FIGS = ROOT / "docs/figures/linguistic_analysis_20260923"


# ----------------------------------------------------------------------------- helpers

def load_embeddings(name: str, texts: list[str]) -> np.ndarray:
    """np.load a cached embedding after checking its fingerprint; never re-encode (the
    September cache sits in a shared, read-only directory)."""
    digest = hashlib.sha256(b"all-mpnet-base-v2")
    for t in texts:
        digest.update(b"\0" + t.encode())
    path = DATA / (name + ".sha256")
    if not path.exists():  # September's text embeddings predate the stamp; the repro asserts below vouch for them
        e = np.load(DATA / name)
        if len(e) != len(texts):
            raise SystemExit(f"{name}: {len(e)} rows for {len(texts)} texts")
        print(f"{name}: no fingerprint stamp; row count matches, validated by the reproduction checks")
        return e
    if path.read_text().strip() != digest.hexdigest():
        raise SystemExit(f"{name}: fingerprint mismatch; refusing to re-embed into {DATA}")
    return np.load(DATA / name)


def population(row) -> str:
    comp = row["composition"]
    if row["size"] == 2:
        fams = comp.split("+")
        return f"{fams[0]} dyads" if fams[0] == fams[1] else f"{comp} dyads"
    return comp


def wilcoxon(vals) -> float:
    r = pd.Series(vals).dropna().round(10)
    if len(r) >= 5 and (r != 0).any():
        return float(stats.wilcoxon(r).pvalue)
    return np.nan


def run_stats(df: pd.DataFrame, metric: str) -> dict:
    per_run = df.groupby("run_id")[metric].mean().dropna()
    return {"effect": per_run.mean(), "sd": per_run.std(ddof=1), "p": wilcoxon(per_run),
            "n_runs": len(per_run), "runs_positive": int((per_run.round(10) > 0).sum()), "n_obs": int(df[metric].notna().sum())}


def boot_ci(vals, n=5000, seed=20260930):
    vals = np.asarray(pd.Series(vals).dropna(), float)
    if len(vals) < 2:
        return np.nan, np.nan
    draws = np.random.default_rng(seed).choice(vals, size=(n, len(vals))).mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


# ----------------------------------------------------------------------------- data

myths, dec = mc.load(ARGS.labels)
if TAG:  # robustness judge: its labels, but the GLM-5.2 one-sentence morals (the cached summary embeddings)
    glm = pd.read_csv(DATA / "moral_labels_z-ai__glm-5.2.csv")[mc.KEY + ["summary"]]
    myths = myths.drop(columns="summary").merge(glm, on=mc.KEY, how="left")
myths["population"] = myths.apply(population, axis=1)
dec["population"] = [population({"composition": c, "size": s}) for c, s in zip(dec["composition"], dec["size"])]
print(f"[{ARGS.dataset}] {len(myths)} myths, labelled {myths['label'].notna().sum()}, runs {myths['run_id'].nunique()}")
IDX = lu.index_of(myths)
CANDS = lu.null_candidates(myths)
FAM = myths["family"].to_numpy()
LAB = myths["label"].to_numpy(object)


def same(i, j) -> float:
    a, b = LAB[i], LAB[j]
    return float(a == b) if isinstance(a, str) and isinstance(b, str) else np.nan


def unseen_at(i: int, author: str, rnd: int, fam: str) -> list[int]:
    """Comparable unseen myths for child i at round rnd, built like lu.null_candidates."""
    row = myths.loc[i]
    if row["size"] == 8:
        return [j for j in BY_RUN_ROUND.get((row.run_id, rnd), [])
                if AGENT[j] not in (row.agent, author) and FAM[j] == fam]
    return [j for j in BY_CELL_ROUND.get((row.composition, row.task_order, rnd), [])
            if RUN[j] != row.run_id and FAM[j] == fam and (row.mixed or AGENT[j] == author)]


_valid = myths[myths["valid"]]
BY_RUN_ROUND = {k: g.index.to_list() for k, g in _valid.groupby(["run_id", "round"])}
BY_CELL_ROUND = {k: g.index.to_list() for k, g in _valid.groupby(["composition", "task_order", "round"])}
AGENT, RUN = myths["agent"].to_numpy(object), myths["run_id"].to_numpy(object)


# ----------------------------------------------------------------------------- 1. word adoption

words = [lu.content_words(t) for t in myths["text"]]
hist = lu.own_history(myths, words)
emb = load_embeddings("embeddings_mpnet.npy", myths["text"].tolist())
children = lu.child_table(myths, words, hist, emb, CANDS)
children["setting"] = [mc.setting_of(s, m) for s, m in zip(children["size"], children["mixed"])]
children["population"] = children.apply(population, axis=1)
children.to_csv(OUT / "word_uptake_children.csv", index=False)

if ARGS.dataset == "september":  # reproduce the committed item-1 table before anything else
    ours = lu.run_level_summary(children, ["size", "exposure"])
    ref = pd.read_csv(SEPT_FIGS / "reuse_summary.csv")
    m = ref.merge(ours, on=["size", "exposure"], suffixes=("_ref", "_ours"))
    assert len(m) == len(ref) == 5
    for c in ["adopt_parent_mean", "adopt_null_mean", "adopt_excess_mean", "cos_excess_mean", "n_runs"]:
        assert np.allclose(m[c + "_ref"], m[c + "_ours"]), c
    print("REPRO word adoption OK:\n" + m[["size", "exposure", "n_runs_ours", "adopt_parent_mean_ours",
                                          "adopt_null_mean_ours"]].round(3).to_string(index=False))

word_rows = []
w = children.assign(exposure=np.where(children["family"] == children["parent_family"], "same family", "other family"))
for keys, g in w.groupby(["setting", "population", "family", "parent_family", "exposure", "task_order"]):
    rec = dict(zip(["setting", "population", "family", "parent_family", "exposure", "task_order"], keys))
    for mtr in ["adopt_parent", "adopt_null", "adopt_excess", "cos_excess"]:
        s = run_stats(g, mtr)
        rec[f"{mtr}_mean"], rec[f"{mtr}_sd"] = s["effect"], s["sd"]
        if mtr.endswith("excess"):
            rec[f"{mtr}_p"], rec[f"{mtr}_runs_positive"] = s["p"], s["runs_positive"]
    rec["n_runs"], rec["n_children"] = g["run_id"].nunique(), len(g)
    word_rows.append(rec)
# pooled rows as in September's table: per setting x exposure, both orders and per order
for keys, g in w.groupby(["setting", "exposure"]):
    for to, h in [("both", g)] + list(g.groupby("task_order")):
        rec = {"setting": keys[0], "population": f"all {keys[0]} runs", "family": "all", "parent_family": "all",
               "exposure": keys[1], "task_order": to}
        for mtr in ["adopt_parent", "adopt_null", "adopt_excess", "cos_excess"]:
            s = run_stats(h, mtr)
            rec[f"{mtr}_mean"], rec[f"{mtr}_sd"] = s["effect"], s["sd"]
            if mtr.endswith("excess"):
                rec[f"{mtr}_p"], rec[f"{mtr}_runs_positive"] = s["p"], s["runs_positive"]
        rec["n_runs"], rec["n_children"] = h["run_id"].nunique(), len(h)
        word_rows.append(rec)
word = pd.DataFrame(word_rows)
word.to_csv(OUT / f"word_adoption_by_stratum.csv", index=False)


# ----------------------------------------------------------------------------- 2. moral label uptake

has_sum = myths["summary"].notna().to_numpy()
semb = load_embeddings("embeddings_moral_summary_mpnet.npy", myths["summary"].fillna("").astype(str).tolist())
dec_idx = {(r.run_id, r.round, r.agent): r for r in dec.itertuples(index=False)}
up = []
for i, js in CANDS.items():
    p, nulls = js[0], [j for j in js[1:] if has_sum[j]]
    if not (has_sum[i] and has_sum[p]) or not nulls:
        continue  # same selection as mc.summary_measures
    row = myths.loc[i]
    author, rp = myths.at[p, "agent"], int(myths.at[p, "round"])
    lab_null = [same(i, j) for j in nulls if isinstance(LAB[j], str)]
    rec = {"idx": i, "run_id": row.run_id, "setting": row.setting, "population": row.population,
           "size": row["size"], "family": row.family, "parent_family": FAM[p], "round": row["round"],
           "task_order": row.task_order, "agent": row.agent, "author": author,
           "label": LAB[i], "parent_label": LAB[p],
           "same_label_shown": same(i, p), "same_label_unseen": np.mean(lab_null) if lab_null and isinstance(LAB[i], str) else np.nan,
           "moral_cos_excess": float(semb[i] @ semb[p]) - float(np.mean(semb[nulls] @ semb[i])),
           "author_is_current_partner": row.partner_this_round == author}
    # future-myth placebo: the author's myth of the child's round, never shown to the child
    f = IDX.get((row.run_id, rp + 1, author))
    if f is not None and myths.at[f, "valid"] and isinstance(LAB[f], str) and isinstance(LAB[i], str):
        fn = [j for j in unseen_at(i, author, rp + 1, FAM[p]) if isinstance(LAB[j], str)]
        if fn:
            rec["same_label_future"] = same(i, f)
            rec["same_label_future_unseen"] = np.mean([same(i, j) for j in fn])
            rec["author_read_child_before_future"] = myths.at[f, "exposed_author"] == row.agent
    # partner-matched null (8-agent): unseen authors who, like the shown author, played an agent of the
    # child's family (not the child) in round rp, so both sides just came out of the same kind of game
    if row["size"] == 8 and isinstance(LAB[i], str):
        def matched(js):
            out = []
            for j in js:
                dj = dec_idx.get((row.run_id, rp, AGENT[j]))
                if dj is not None and dj.partner != row.agent and dj.partner_family == row.family and isinstance(LAB[j], str):
                    out.append(j)
            return out
        mn = matched(nulls)
        if mn:
            rec["same_label_unseen_matched"] = np.mean([same(i, j) for j in mn])
        if "same_label_future" in rec:
            mf = matched(unseen_at(i, author, rp + 1, FAM[p]))
            if mf:
                rec["same_label_future_unseen_matched"] = np.mean([same(i, j) for j in mf])
    # the t-1 game between author and child (8-agent: exposure = last round's partner)
    da, dc = dec_idx.get((row.run_id, rp, author)), dec_idx.get((row.run_id, rp, row.agent))
    if da is not None and dc is not None and da.partner == row.agent:
        rec["author_was_investor"] = float(da.role == "investor")
        rec["author_coop_t1"], rec["child_coop_t1"] = da.coop, dc.coop
    up.append(rec)
uptake = pd.DataFrame(up)
uptake["same_label_excess"] = uptake["same_label_shown"] - uptake["same_label_unseen"]
uptake["future_excess"] = uptake.get("same_label_future", np.nan) - uptake.get("same_label_future_unseen", np.nan)
uptake["matched_excess"] = uptake["same_label_shown"] - uptake.get("same_label_unseen_matched", np.nan)
uptake["future_matched_excess"] = uptake.get("same_label_future", np.nan) - uptake.get("same_label_future_unseen_matched", np.nan)
uptake["exposure"] = np.where(uptake["family"] == uptake["parent_family"], "same family", "other family")
uptake.to_csv(OUT / f"moral_uptake_children{TAG}.csv", index=False)

if ARGS.dataset == "september" and not TAG:  # reproduce +4.8 / +5.9 / +0.7
    ref = pd.read_csv(SEPT_FIGS / "moral_uptake_by_task_order.csv")
    ours = mc.run_summary(uptake, ["setting", "exposure", "task_order"], ["moral_cos_excess", "same_label_excess"])
    m = ref.merge(ours, on=["setting", "exposure", "task_order"], suffixes=("_ref", "_ours"))
    assert len(m) == len(ref) == len(ours)
    for c in ["n_runs", "same_label_excess_mean", "same_label_excess_p", "moral_cos_excess_mean"]:
        assert np.allclose(m[c + "_ref"], m[c + "_ours"]), c
    print("REPRO moral uptake OK:\n" + m[["setting", "exposure", "task_order", "n_runs_ours",
                                         "same_label_excess_mean_ours", "same_label_excess_p_ours"]].round(4).to_string(index=False))

RNG = np.random.default_rng(ARGS.seed)


def perm_p(sub: pd.DataFrame) -> tuple[float, float]:
    """Re-draw each child's parent among its unseen candidates; two-sided p for the mean same-label rate."""
    rows = [(int(i), [j for j in CANDS[int(i)][1:] if has_sum[j] and isinstance(LAB[j], str)])
            for i in sub["idx"] if isinstance(LAB[int(i)], str)]
    rows = [(i, js) for i, js in rows if js]
    if len(rows) < 10:
        return np.nan, np.nan
    obs = np.mean([same(i, CANDS[i][0]) for i, _ in rows])
    lab_c = np.array([LAB[i] for i, _ in rows], object)
    null = np.empty(ARGS.perms)
    for k in range(ARGS.perms):
        null[k] = np.mean(lab_c == np.array([LAB[js[RNG.integers(len(js))]] for _, js in rows], object))
    mu = null.mean()
    return obs - mu, (np.sum(np.abs(null - mu) >= abs(obs - mu) - 1e-12) + 1) / (ARGS.perms + 1)


def r3_adjusted(sub: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf
    s = sub[~sub["author_is_current_partner"]].dropna(subset=["same_label_excess", "author_coop_t1", "child_coop_t1"])
    if s["run_id"].nunique() < 5 or len(s) < 20:
        return {}
    s = s.assign(**{c: s[c] - s[c].mean() for c in ["author_coop_t1", "child_coop_t1", "author_was_investor"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols("same_label_excess ~ author_coop_t1 + child_coop_t1 + author_was_investor", data=s).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(s["run_id"])[0]})
    ci = fit.conf_int().loc["Intercept"]
    return {"effect": fit.params["Intercept"], "ci_low": ci[0], "ci_high": ci[1], "p": fit.pvalues["Intercept"],
            "n_runs": s["run_id"].nunique(), "n_obs": int(fit.nobs)}


def label_ceiling(sub: pd.DataFrame) -> bool:
    """Children's labels (or their parents') almost all one label: agreement is fixed by the base rate."""
    for col in ("label", "parent_label"):
        v = sub[col].dropna()
        if len(v) and v.value_counts(normalize=True).iloc[0] >= 0.95:
            return True
    return False


mor_rows = []
strata = [(k, g) for k, g in uptake.groupby(["setting", "population", "family", "parent_family", "exposure", "task_order"])]
# pooled strata matching September's committed table (families pooled within a setting, never across orders)
strata += [((st, f"all {st} runs", "all", "all", ex, to), g)
           for (st, ex, to), g in uptake.groupby(["setting", "exposure", "task_order"])]
# mixed population, exposure pooled over families (family pairs within it are listed separately)
for (st, pop, ex, to), g in uptake[uptake["setting"].str.contains("mixed")].groupby(["setting", "population", "exposure", "task_order"]):
    if g["family"].nunique() > 1 or g["parent_family"].nunique() > 1:
        strata.append(((st, pop, "all", "all", ex, to), g))
for keys, g in strata:
    rec = dict(zip(["setting", "population", "family", "parent_family", "exposure", "task_order"], keys))
    s = run_stats(g, "same_label_excess")
    rec.update({"uptake_" + k: v for k, v in s.items()})
    rec["uptake_ci_low"], rec["uptake_ci_high"] = boot_ci(g.groupby("run_id")["same_label_excess"].mean())
    rec["uptake_perm_effect"], rec["uptake_perm_p"] = perm_p(g)
    rec["label_ceiling"] = label_ceiling(g)
    s = run_stats(g, "moral_cos_excess")
    rec["moral_cos_excess_mean"], rec["moral_cos_excess_sd"], rec["moral_cos_excess_p"] = s["effect"], s["sd"], s["p"]
    s = run_stats(g, "future_excess")
    rec.update({"future_" + k: v for k, v in s.items()})
    for mtr in ("matched_excess", "future_matched_excess"):
        s = run_stats(g, mtr)
        rec[mtr + "_mean"], rec[mtr + "_sd"], rec[mtr + "_p"], rec[mtr + "_n_runs"], rec[mtr + "_n_obs"] = (
            s["effect"], s["sd"], s["p"], s["n_runs"], s["n_obs"])
    rec["future_author_read_child_share"] = g["author_read_child_before_future"].mean() if "author_read_child_before_future" in g else np.nan
    if g["size"].iloc[0] == 8 and keys[5] == "myth_game":
        rec.update({"r3_" + k: v for k, v in r3_adjusted(g).items()})
    mor_rows.append(rec)
moral = pd.DataFrame(mor_rows)
moral.to_csv(OUT / f"moral_uptake_by_stratum{TAG}.csv", index=False)
print(moral[(moral["family"] == "all")][["setting", "population", "exposure", "task_order", "uptake_n_runs",
                                         "uptake_effect", "uptake_p", "uptake_perm_p", "future_effect", "future_p",
                                         "r3_effect", "r3_p"]].round(4).to_string(index=False))


# ----------------------------------------------------------------------------- 3 + 4. carryover and reverse

d = mc.decision_table(myths, dec)
d["run_agent"] = d["run_id"] + "|" + d["agent"]
d["population"] = [population({"composition": c, "size": s}) for c, s in zip(d["composition"], d["size"])]


def fe_fit(formula: str, sub: pd.DataFrame, terms: list[str]):
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(sub["run_id"])[0]})
    ci = fit.conf_int()
    return {t: (fit.params[t], ci.loc[t, 0], ci.loc[t, 1], fit.pvalues[t]) for t in terms if t in fit.params}, int(fit.nobs)


carry_rows, rev_rows = [], []
groups = []
for (fam, st, pop), g in d.groupby(["family", "setting", "population"]):
    groups.append((fam, st, pop, g))
for (fam, st), g in d[d["setting"] == "2-agent mixed"].groupby(["family", "setting"]):
    groups.append((fam, st, "all mixed dyads", g))
for fam, st, pop, g in groups:
    for to, h0 in list(h for h in g.groupby("task_order")) + [("both", g)]:
        for role in ("investor", "trustee"):
            h = h0[h0["role"] == role]
            base = {"family": fam, "setting": st, "population": pop, "task_order": to, "role": role,
                    "coop_sd": h["coop"].std(ddof=1), "n_runs_total": h["run_id"].nunique()}
            # carryover
            c = h.dropna(subset=["coop", "own_label", "shown_label", "coop_lag_same_role"])
            c = c[c["own_label"].isin([GEN, FAIR]) & c["shown_label"].isin([GEN, FAIR])]
            if st.startswith("8-agent"):
                c = c[c["shown_author"] != c["partner"]]
            c = c.assign(own_gen=(c["own_label"] == GEN).astype(float), shown_gen=(c["shown_label"] == GEN).astype(float))
            vary = {t: int((c.groupby("run_agent")[t].nunique() > 1).sum()) for t in ("own_gen", "shown_gen")}
            rec = dict(base, n_agents_own_varies=vary["own_gen"], n_agents_shown_varies=vary["shown_gen"],
                       own_gen_share=c["own_gen"].mean(), shown_gen_share=c["shown_gen"].mean())
            if c["run_id"].nunique() >= 3 and len(c) >= 20 and c["coop"].std() > 0:
                fe = " + C(run_agent) + C(round)" + (" + C(task_order):C(round)" if to == "both" else "")
                res, n = fe_fit("coop ~ own_gen + shown_gen + coop_lag_same_role" + fe, c, ["own_gen", "shown_gen"])
                for t, (b, lo, hi, p) in res.items():
                    rec[f"{t}_coef"], rec[f"{t}_ci_low"], rec[f"{t}_ci_high"], rec[f"{t}_p"] = b, lo, hi, p
                rec["n_obs"], rec["n_runs"] = n, c["run_id"].nunique()
            carry_rows.append(rec)
            # reverse: this game -> next myth generous
            r = h.dropna(subset=["coop", "label_after", "own_label"])
            r = r.assign(gen_after=(r["label_after"] == GEN).astype(float), gen_before=(r["own_label"] == GEN).astype(float))
            rrec = dict(base, gen_after_share=r["gen_after"].mean())
            if r["run_id"].nunique() >= 3 and len(r) >= 20 and r["coop"].std() > 0 and r["gen_after"].std() > 0:
                fe = " + C(run_agent) + C(round)" + (" + C(task_order):C(round)" if to == "both" else "")
                res, n = fe_fit("gen_after ~ coop + gen_before" + fe, r, ["coop"])
                if "coop" in res:
                    b, lo, hi, p = res["coop"]
                    rrec.update(coef=b, ci_low=lo, ci_high=hi, p=p, n_obs=n, n_runs=r["run_id"].nunique())
                # September spec (run + round FE) for comparison
                res, n = fe_fit("gen_after ~ coop + gen_before + C(run_id) + C(round)", r, ["coop"])
                if "coop" in res:
                    rrec["coef_runfe"], _, _, rrec["p_runfe"] = res["coop"]
            rev_rows.append(rrec)
carry = pd.DataFrame(carry_rows)
rev = pd.DataFrame(rev_rows)
carry.to_csv(OUT / f"carryover_by_family{TAG}.csv", index=False)
rev.to_csv(OUT / f"reverse_by_family{TAG}.csv", index=False)
# September-comparable pooled tables (all families, as committed)
if not ARGS.skip_pooled_carryover:
    pooled = mc.carryover_models(d.drop(columns=["run_agent"]))
    pooled.to_csv(OUT / f"carryover_models_pooled{TAG}.csv", index=False)
    mc.reverse_models(d).to_csv(OUT / f"reverse_models_pooled{TAG}.csv", index=False)
if ARGS.dataset == "september" and not TAG and not ARGS.skip_pooled_carryover:
    ref = pd.read_csv(SEPT_FIGS / "moral_carryover_models.csv")
    k = ["fe", "setting", "role", "model", "predictor", "level"]
    ours = pd.read_csv(OUT / "carryover_models_pooled.csv")  # CSV round-trip: empty levels read back as NaN
    m = ref.dropna(subset=["coef"]).merge(ours, on=k, suffixes=("_ref", "_ours"))
    assert len(m) == ref["coef"].notna().sum() and np.allclose(m["coef_ref"], m["coef_ours"])
    print("REPRO carryover models OK")


# ----------------------------------------------------------------------------- 5. label shares

shares = mc.label_shares(myths.assign(setting=myths["population"]))
shares = shares.rename(columns={"setting": "population"})
shares.to_csv(OUT / f"moral_label_shares_by_population{TAG}.csv", index=False)
drift_rows = []
for (pop, fam, to), g in myths.dropna(subset=["label"]).groupby(["population", "family", "task_order"]):
    per = g.assign(gen=(g["label"] == GEN).astype(float)).groupby(["run_id", "round"])["gen"].mean().unstack()
    if 1 not in per or 10 not in per:
        continue
    diff = (per[10] - per[1]).dropna()
    drift_rows.append({"population": pop, "setting": g["setting"].iloc[0], "family": fam, "task_order": to,
                       "gen_share_all": g["label"].eq(GEN).mean(), "fair_share_all": g["label"].eq(FAIR).mean(),
                       "cautious_share_all": g["label"].eq("be cautious").mean(),
                       "gen_r1": per[1].mean(), "gen_r10": per[10].mean(), "change_mean": diff.mean(),
                       "change_sd": diff.std(ddof=1), "p": wilcoxon(diff), "n_runs": len(diff), "n_myths": len(g)})
drift = pd.DataFrame(drift_rows)
drift.to_csv(OUT / f"moral_generous_change_r1_r10{TAG}.csv", index=False)
print(drift.round(3).to_string(index=False))
if not TAG:
    mc.plot_label_shares(myths)  # -> OUT/moral_label_shares.png
print(f"wrote {OUT}")
