#!/usr/bin/env python3
"""H1 norm alignment, model stage (port of the September norm_alignment_fast.py,
robustness.py and artefact_checks.py), per stratum.

Estimator unchanged from September: fixed effects absorbed by alternating-projection
demeaning, CR1 run-clustered SEs, absorbed FE levels counted in the small-sample K.
Primary spec: outcome ~ z(alignment) + both players' own levels (label, send rule as
categories; 0-10 send score linear) + both players' lagged moves (any role, same role)
+ run x pair-family FE + round FE (8-agent: run x round FE).

What is new for the frontier port:
  * strata: all 2-agent / all 8-agent (September's pooled primary), 2-agent homogeneous
    per family, 2-agent mixed per pairing, 8-agent homogeneous per family, 8-agent
    mixed per composition; Holm within each stratum over its estimable primary tests.
  * estimability from CONTRIBUTING clusters: runs in which the FE-demeaned outcome and
    the FE-demeaned alignment measure both still vary. Fewer than MIN_G -> underpowered
    (CR1 SEs are too small with few clusters); zero -> not estimable.
  * giving gap = 1 - return proportion wherever send is locked at $5, so in strata
    whose send has no within-cell variation the gap test duplicates the return test and
    is dropped from Holm; where send varies in fewer than MIN_G runs the gap test is
    marked underpowered (it is the return test plus those few runs).

ALIGN_DATASET=september|frontier (default frontier). Needs norm_alignment.py's games table.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import norm_alignment as na

DATASET = na.DATASET
OUT = na.OUT
BIG = na.BIG
MEASURES = ["same_label", "moral_cos", "rule_index", "give_align", "myth_cos"]
OUTCOMES = na.OUTCOMES
CATS = ["inv_label", "tru_label", "inv_send_rule", "tru_send_rule"]
NUMS = ["inv_give", "tru_give"]
LAGS = ["inv_lag_any", "tru_lag_any", "inv_lag_same_f", "inv_lag_same_missing", "tru_lag_same_f", "tru_lag_same_missing"]
MIN_G = 8          # contributing clusters below this -> underpowered
TOL = 1e-9
MIN_RESID_DF = 20  # fewer residual df after absorbing FE -> saturated fit, SEs meaningless


# --------------------------------------------------------------------------- estimator

def demean(M: np.ndarray, groups: list[np.ndarray], tol=1e-10, max_iter=2000) -> np.ndarray:
    M = M.astype(float).copy()
    if not groups:
        return M - M.mean(axis=0)
    for _ in range(max_iter):
        prev = M.copy()
        for g in groups:
            cnt = np.bincount(g)
            for k in range(M.shape[1]):
                M[:, k] -= (np.bincount(g, M[:, k]) / cnt)[g]
        if np.max(np.abs(M - prev)) < tol:
            break
    return M


def feols(df: pd.DataFrame, y: str, x: list[str], cats: list[str] = (), fes: list[str] = (),
          cluster: str | None = "run_id", key: str | None = None) -> dict | None:
    """Same estimator as September. Adds g_eff: clusters where demeaned y and demeaned key vary."""
    cols = [y] + list(x) + list(cats) + list(fes) + ([cluster] if cluster else [])
    d = df.dropna(subset=list(dict.fromkeys(cols)))
    if len(d) < 30:
        return None
    parts = [d[list(x)].astype(float)]
    for c in cats:
        parts.append(pd.get_dummies(d[c].astype(str), prefix=c, drop_first=True, dtype=float))
    X = pd.concat(parts, axis=1)
    if not fes:
        X.insert(0, "const", 1.0)
    groups = [pd.factorize(d[f])[0] for f in fes]
    Z = demean(np.column_stack([d[y].to_numpy(float), X.to_numpy()]), groups)
    yy, XX = Z[:, 0], Z[:, 1:]
    # contributing runs: raw outcome AND raw key regressor both vary inside some finest-FE group
    # (8-agent: run x round; 2-agent: run x pair-family; else run). Demeaned values are no use
    # here: with two-way FE a locked run still gets nonzero residuals from the round means.
    key = key or x[0]
    within = "runround" if "runround" in fes else ("cell" if "cell" in fes else "run_id")
    kv = d[key].astype(float) if key in d else pd.Series(0.0, index=d.index)
    grp = d[within]
    vy_g = d[y].astype(float).groupby(grp).transform("std").fillna(0) > TOL
    vx_g = kv.groupby(grp).transform("std").fillna(0) > TOL
    g_eff = int(d.loc[vy_g & vx_g, "run_id"].nunique())
    g_y, g_x = int(d.loc[vy_g, "run_id"].nunique()), int(d.loc[vx_g, "run_id"].nunique())
    keep = np.abs(XX).max(axis=0) > TOL
    XX, names = XX[:, keep], X.columns[keep]
    B = np.linalg.pinv(XX.T @ XX)
    beta = B @ XX.T @ yy
    e = yy - XX @ beta
    n, k = XX.shape
    k_df = k + sum(len(np.unique(gr)) - 1 for gr in groups) + (1 if groups else 0)
    if cluster:
        cl = pd.factorize(d[cluster])[0]
        G = cl.max() + 1
        S = np.zeros((G, k))
        np.add.at(S, cl, XX * e[:, None])
        V = B @ (S.T @ S) @ B * (G / (G - 1)) * ((n - 1) / max(n - k_df, 1))
        dof = G - 1
    else:  # HC1
        V = B @ ((XX * e[:, None] ** 2).T @ XX) @ B * n / max(n - k_df, 1)
        dof, G = max(n - k_df, 1), np.nan
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    res = {}
    for i, nm in enumerate(names):
        t = beta[i] / se[i] if se[i] > 0 else np.nan
        q = stats.t.ppf(0.975, dof)
        res[nm] = {"coef": beta[i], "se": se[i], "ci_low": beta[i] - q * se[i], "ci_high": beta[i] + q * se[i],
                   "p": 2 * stats.t.sf(abs(t), dof) if not np.isnan(t) else np.nan}
    return {"terms": res, "n_games": n, "n_runs": d["run_id"].nunique(), "data": d, "G": G,
            "g_eff": g_eff, "g_y": g_y, "g_x": g_x, "resid_df": n - k_df}


# --------------------------------------------------------------------------- data

def load_games() -> pd.DataFrame:
    g = pd.read_csv(BIG / "norm_alignment_games.csv")
    give = na.judge_measures().set_index(na.KEY)["give_send_glm"]
    rb = np.where(g["task_order"] == "myth_game", g["round"], g["round"] - 1)
    ra = np.where(g["task_order"] == "myth_game", g["round"] + 1, g["round"])
    for w, col in (("inv", "investor"), ("tru", "trustee")):
        g[f"{w}_give"] = [give.get(k, np.nan) for k in zip(g["run_id"], rb, g[col])]
        g[f"{w}_give_after"] = [give.get(k, np.nan) for k in zip(g["run_id"], ra, g[col])]
    g["give_align"] = 1 - (g["inv_give"] - g["tru_give"]).abs() / 10
    g["give_align_after"] = 1 - (g["inv_give_after"] - g["tru_give_after"]).abs() / 10
    g["pair_give"] = (g["inv_give"] + g["tru_give"]) / 2
    ds = na.judge_measures().set_index(na.KEY)["label_ds"]
    g["inv_label_ds"] = [ds.get(k, np.nan) for k in zip(g["run_id"], rb, g["investor"])]
    g["tru_label_ds"] = [ds.get(k, np.nan) for k in zip(g["run_id"], rb, g["trustee"])]
    g["cell"] = g["run_id"] + "|" + g["pair_family"]
    g["runround"] = g["run_id"] + "|" + g["round"].astype(str)
    g["inv_id"] = g["run_id"] + "|" + g["investor"]
    g["tru_id"] = g["run_id"] + "|" + g["trustee"]
    return g


def add_nonpartner_placebo(g: pd.DataFrame) -> pd.DataFrame:
    """8-agent: investor's mean alignment with the 6 agents it is NOT playing (same myth round)."""
    myths, emb, semb = na.load_myths()
    give = na.judge_measures().set_index(na.KEY)["give_send_glm"]
    myths["give"] = [give.get(k, np.nan) for k in zip(myths["run_id"], myths["round"], myths["agent"])]
    by_rr = {k: v.index.to_numpy() for k, v in myths.groupby(["run_id", "round"])}
    idx = {(r, t, a): k for k, (r, t, a) in enumerate(zip(myths["run_id"], myths["round"], myths["agent"]))}
    out = {m: np.full(len(g), np.nan) for m in MEASURES}
    for n, r in enumerate(g.itertuples(index=False)):
        if r.size != 8:
            continue
        rb = r.round if r.task_order == "myth_game" else r.round - 1
        i = idx.get((r.run_id, rb, r.investor))
        if i is None:
            continue
        others = [j for j in by_rr.get((r.run_id, rb), []) if myths.at[j, "agent"] not in (r.investor, r.trustee)]
        vals = {m: [] for m in out}
        for j in others:
            pm = na.pair_measures(myths, emb, semb, i, j)
            for m in ["same_label", "moral_cos", "rule_index", "myth_cos"]:
                vals[m].append(pm.get(m, np.nan))
            vals["give_align"].append(1 - abs(myths.at[i, "give"] - myths.at[j, "give"]) / 10)
        for m in out:
            v = [x for x in vals[m] if not pd.isna(x)]
            out[m][n] = np.mean(v) if v else np.nan
    for m in out:
        g[f"np_{m}"] = out[m]
    return g


def z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std() if s.std() > 0 else s * np.nan


def fe_for(size: int, kind: str) -> list[str]:
    if kind == "cell":
        return ["cell", "runround"] if size == 8 else ["cell", "round"]
    if kind == "agents":
        return ["inv_id", "tru_id", "round"]
    raise ValueError(kind)


# --------------------------------------------------------------------------- strata

def strata(g: pd.DataFrame) -> list[dict]:
    """(name, population, family, setting, mask). 'all' strata pool families (September primary)."""
    out = [{"stratum": "2-agent all", "setting": "2-agent", "population": "all 2-agent runs (pooled)",
            "family": "pooled", "mask": g["size"] == 2},
           {"stratum": "8-agent all", "setting": "8-agent", "population": "all 8-agent runs (pooled)",
            "family": "pooled", "mask": g["size"] == 8}]
    for setting in ["2-agent homogeneous", "2-agent mixed", "8-agent homogeneous", "8-agent mixed"]:
        for comp in sorted(g.loc[g["setting"] == setting, "composition"].unique()):
            m = (g["setting"] == setting) & (g["composition"] == comp)
            if setting.startswith("2"):
                fams = comp.split("+")
                pop = f"{fams[0]} dyads" if setting.endswith("homogeneous") else f"{comp} dyads"
            else:
                pop = comp
            fam = comp.split("+")[0] if setting == "2-agent homogeneous" else (
                comp.split(" ", 1)[1] if setting == "8-agent homogeneous" else "mixed")
            out.append({"stratum": f"{setting} | {comp}", "setting": setting, "population": pop, "family": fam, "mask": m})
    return out


def send_runs(d: pd.DataFrame, size: int) -> int:
    """Runs in which send still varies inside a finest-FE group (run x round / run x pair-family)."""
    within = "runround" if size == 8 else "cell"
    v = d.groupby(within)["sent_frac"].transform("std").fillna(0) > TOL
    return int(d.loc[v, "run_id"].nunique())


def send_locked(d: pd.DataFrame, size: int) -> bool:
    """True if send has no variation left after the primary FE (so gap == 1 - return)."""
    return send_runs(d, size) == 0


def status_of(r: dict) -> str:
    if isinstance(r.get("skip_reason"), str):
        return r["skip_reason"].split(" (")[0]
    if pd.isna(r.get("coef")) or pd.isna(r.get("g_eff")) or pd.isna(r.get("p")):
        return "not estimable: ceiling"
    if r["g_eff"] == 0:
        return "not estimable: ceiling"
    if r["g_eff"] < MIN_G:
        return "underpowered"
    if not pd.isna(r.get("holm_p")) and r["holm_p"] < 0.05:
        return "yes"
    if r["p"] < 0.05:
        return "suggestive"
    return "no detectable effect"


# --------------------------------------------------------------------------- analyses

def fit_one(d, m, y, fe="cell", side=None, lags=True, levels=True, cats=None):
    d = d.dropna(subset=[m]).copy()
    if d.empty:
        return None, "not testable by design (no games with this measure)"
    d["zx"] = z(d[m])
    if d["zx"].isna().all():
        return None, f"not estimable: ceiling (measure {m} constant)"
    x = ["zx"]
    if side:
        d["zside"] = z(d[side])
        if d["zside"].isna().all():
            return None, f"not estimable: ceiling (side measure {side} constant)"
        x.append("zside")
    if lags:
        x += LAGS
    if levels:
        x += NUMS
    size = int(d["size"].iloc[0])
    f = feols(d, y, x, cats=(cats if cats is not None else CATS) if levels else [], fes=fe_for(size, fe), key="zx")
    if f is None:
        return None, f"not testable by design (n={len(d)} games < 30)"
    if "zx" not in f["terms"]:
        return f, f"not estimable: ceiling ({m} has no variation within cells)"
    return f, None


def row_from(f, skip, base: dict, d: pd.DataFrame, m: str, term="zx") -> dict:
    r = dict(base)
    r["skip_reason"] = skip if isinstance(skip, str) else np.nan
    r.update({"raw_mean": d[m].mean() if m in d else np.nan, "raw_sd": d[m].std() if m in d else np.nan})
    if f is not None and term in f["terms"]:
        r.update(f["terms"][term])
        r.update({"n_games": f["n_games"], "n_runs": f["n_runs"], "g_eff": f["g_eff"], "g_y": f["g_y"], "g_x": f["g_x"],
                  "resid_df": f["resid_df"]})
        if f["resid_df"] < MIN_RESID_DF and not isinstance(r.get("skip_reason"), str):
            r["skip_reason"] = f"underpowered (fixed effects saturate the data: residual df {f['resid_df']})"
    else:
        r.update({"n_games": len(d), "n_runs": d["run_id"].nunique(), "g_eff": 0, "skip_reason": skip})
    return r


def primary_models(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = g[g["has_myths_before"]]
    for s in strata(base):
        d = base[s["mask"]]
        if d.empty:
            continue
        size = int(d["size"].iloc[0])
        ns = send_runs(d, size)
        locked = ns == 0
        c_gs = d[["giving_gap", "sent_frac"]].corr().iloc[0, 1] if d["sent_frac"].std() > 0 else np.nan
        c_gr = d[["giving_gap", "return_proportion"]].corr().iloc[0, 1]
        meta = {k: v for k, v in s.items() if k != "mask"}
        for y in OUTCOMES:
            for m in MEASURES:
                f, skip = fit_one(d, m, y)
                if y == "sent_frac" and locked:
                    skip = "not estimable: ceiling (send locked within cells)"
                if y == "giving_gap" and locked:
                    skip = "not estimable: ceiling (send locked; gap = 1 - return, duplicates the return test)"
                elif y == "giving_gap" and ns < MIN_G:
                    skip = (f"underpowered (send varies in only {ns} runs; elsewhere gap = 1 - return, "
                            "so the gap test restates the return test plus those few runs)")
                r = row_from(f, skip, {**meta, "measure": m, "outcome": y, "spec": "primary"}, d, m)
                if skip and f is not None and "zx" in f["terms"]:
                    r["skip_reason"] = skip
                r["corr_gap_send"], r["corr_gap_return"] = c_gs, c_gr
                rows.append(r)
    out = pd.DataFrame(rows)
    out["holm_p"] = np.nan
    out["holm_family_size"] = np.nan
    for st, idx in out.groupby("stratum").groups.items():
        sub = out.loc[idx]
        ok = sub["skip_reason"].isna() & sub["p"].notna() & (sub["g_eff"] >= MIN_G)
        if ok.sum():
            out.loc[sub.index[ok], "holm_p"] = multipletests(sub.loc[ok, "p"], method="holm")[1]
            out.loc[sub.index, "holm_family_size"] = int(ok.sum())
    # September's own family: 30 tests over the two pooled strata, no estimability filter
    pooled = out["stratum"].isin(["2-agent all", "8-agent all"]) & out["p"].notna()
    out.loc[pooled, "holm_p_sept30"] = multipletests(out.loc[pooled, "p"], method="holm")[1]
    if DATASET == "september":  # the pooled strata ARE September's declared 30-test family
        out.loc[pooled, "holm_p"] = out.loc[pooled, "holm_p_sept30"]
    out["status"] = [status_of(r) for r in out.to_dict("records")]
    return out


def secondary_models(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = g[g["has_myths_before"]]
    have_ds = base["same_label_ds"].notna().any()
    for s in strata(base):
        d = base[s["mask"]]
        if d.empty:
            continue
        meta = {k: v for k, v in s.items() if k != "mask"}
        size = int(d["size"].iloc[0])
        ns = send_runs(d, size)
        for y in OUTCOMES:
            if (y == "sent_frac" and ns == 0) or (y == "giving_gap" and ns < MIN_G):
                continue
            for m in MEASURES:
                specs = [("agent FE (investor, trustee within run) + round", dict(fe="agents")),
                         ("no controls", dict(lags=False, levels=False)),
                         ("lags, no levels", dict(levels=False))]
                if m != "myth_cos":
                    specs.append(("side by side with whole-myth similarity", dict(side="myth_cos")))
                if size == 8:
                    specs.append(("vs non-partner placebo", dict(side=f"np_{m}")))
                for spec, kw in specs:
                    f, skip = fit_one(d, m, y, **kw)
                    rows.append(row_from(f, skip, {**meta, "measure": m, "outcome": y, "spec": spec}, d, m))
                    if kw.get("side") and f is not None and "zside" in f["terms"]:
                        rows.append(row_from(f, None, {**meta, "measure": kw["side"], "outcome": y,
                                                       "spec": spec + " [coef of the side term]"}, d, kw["side"], "zside"))
                for order in ["myth_game", "game_myth"]:
                    e = d[d["task_order"] == order]
                    if e.empty:
                        continue
                    f, skip = fit_one(e, m, y)
                    rows.append(row_from(f, skip, {**meta, "measure": m, "outcome": y, "spec": f"task order {order}"}, e, m))
                if size == 8:
                    for nm, e in (("first meetings", d[d["prior_meetings"] == 0]), ("repeat meetings", d[d["prior_meetings"] > 0])):
                        f, skip = fit_one(e, m, y)
                        rows.append(row_from(f, skip, {**meta, "measure": m, "outcome": y, "spec": nm}, e, m))
            # DeepSeek labels
            if have_ds:
                # September robustness (b): DeepSeek label levels as the level controls
                f, skip = fit_one(d.assign(same_label=d["same_label_ds"]), "same_label", y,
                                  cats=["inv_label_ds", "tru_label_ds", "inv_send_rule", "tru_send_rule"])
                rows.append(row_from(f, skip, {**meta, "measure": "same_label", "outcome": y, "spec": "DeepSeek labels"}, d, "same_label"))
            else:
                rows.append({**meta, "measure": "same_label", "outcome": y, "spec": "DeepSeek labels",
                             "skip_reason": "pending (DeepSeek label file not landed)"})
    out = pd.DataFrame(rows)
    out["holm_p"] = np.nan
    ok = out.get("skip_reason", pd.Series(np.nan, index=out.index)).isna() & out["p"].notna() & (out["g_eff"] >= MIN_G) \
        & ~out["spec"].str.contains("side term")
    for (st, spec), idx in out[ok].groupby(["stratum", "spec"]).groups.items():
        out.loc[idx, "holm_p"] = multipletests(out.loc[idx, "p"], method="holm")[1]
    out["status"] = [status_of(r) for r in out.to_dict("records")]
    return out


def founding_window(g: pd.DataFrame, per_stratum=True) -> pd.DataFrame:
    """R4: round-1 myth_game games; both myths written before any play."""
    rows = []
    d0 = g[(g["task_order"] == "myth_game") & (g["round"] == 1) & g["has_myths_before"]]
    groups = strata(d0) if per_stratum else [
        {"stratum": "8-agent all", "setting": "8-agent", "population": "all 8-agent runs (pooled)", "family": "pooled", "mask": d0["size"] == 8},
        {"stratum": "2-agent all", "setting": "2-agent", "population": "all 2-agent runs (pooled)", "family": "pooled", "mask": d0["size"] == 2}]
    for s in groups:
        d = d0[s["mask"]]
        meta = {k: v for k, v in s.items() if k != "mask"}
        if d.empty:
            continue
        size = int(d["size"].iloc[0])
        fes, cl = (["run_id"], "run_id") if size == 8 else (["composition"], None)
        for y in OUTCOMES:
            for m in MEASURES:
                e = d.dropna(subset=[m]).copy()
                e["zx"] = z(e[m])
                for spec, x in (("alone", ["zx"]), ("+ own send scores", ["zx"] + NUMS)):
                    base = {**meta, "measure": m, "outcome": y, "spec": spec}
                    if e["zx"].isna().all():
                        rows.append({**base, "n_games": len(e), "skip_reason": f"not estimable: ceiling (measure {m} constant)"})
                        continue
                    f = feols(e, y, x, [], fes, cluster=cl, key="zx")
                    if f is None:
                        rows.append({**base, "n_games": len(e), "n_runs": e["run_id"].nunique(),
                                     "skip_reason": f"not testable by design (n={len(e)} round-1 games < 30)"})
                        continue
                    r = row_from(f, None, base, e, m)
                    if cl is None:  # HC1: contributing units are games off their composition's modal value
                        mode = e.groupby("composition")[y].transform(lambda s: s.round(6).mode().iloc[0])
                        r["g_eff"] = int((e[y].round(6) != mode).sum())
                    if e[y].std() == 0 or "zx" not in f["terms"]:
                        r["skip_reason"] = "not estimable: ceiling (no round-1 variation)"
                    rows.append(r)
    out = pd.DataFrame(rows)
    out["holm_p"] = np.nan
    if "skip_reason" not in out:
        out["skip_reason"] = np.nan
    ok = out["skip_reason"].isna() & out["p"].notna() & (out["g_eff"] >= MIN_G)
    for st, idx in out[ok].groupby("stratum").groups.items():
        out.loc[idx, "holm_p"] = multipletests(out.loc[idx, "p"], method="holm")[1]
    out["status"] = [status_of(r) for r in out.to_dict("records")]
    return out


def reverse_models(g: pd.DataFrame, per_stratum=True) -> pd.DataFrame:
    """Does a cooperative game make the NEXT myths more aligned (controlling for alignment before)?"""
    rows = []
    groups = strata(g) if per_stratum else [
        {"stratum": f"{s}-agent all", "setting": f"{s}-agent", "population": f"all {s}-agent runs (pooled)",
         "family": "pooled", "mask": g["size"] == s} for s in (2, 8)]
    for s in groups:
        meta = {k: v for k, v in s.items() if k != "mask"}
        for order in ["myth_game", "game_myth", "both"]:
            d = g[s["mask"]]
            if order != "both":
                d = d[d["task_order"] == order]
            if d.empty:
                continue
            size = int(d["size"].iloc[0])
            for m in MEASURES:
                for x in ["sent_frac", "return_proportion"]:
                    base = {**meta, "task_order": order, "measure": f"{m} after the game", "predictor": x}
                    e = d.dropna(subset=[f"{m}_after", x]).copy()
                    e["before_missing"] = e[m].isna().astype(float)
                    e["before_f"] = e[m].fillna(e[m].mean())
                    f = feols(e, f"{m}_after", [x, "before_f", "before_missing"], [], fe_for(size, "cell"), key=x)
                    if f is None:
                        rows.append({**base, "n_games": len(e), "skip_reason": f"not testable by design (n={len(e)} < 30)"})
                        continue
                    r = {**base, "n_games": f["n_games"], "n_runs": f["n_runs"], "g_eff": f["g_eff"],
                         "sd_after": e[f"{m}_after"].std()}
                    if x in f["terms"]:
                        r.update(f["terms"][x])
                    else:
                        r["skip_reason"] = f"not estimable: ceiling ({x} locked within cells)"
                    rows.append(r)
    out = pd.DataFrame(rows)
    if "skip_reason" not in out:
        out["skip_reason"] = np.nan
    out["holm_p"] = np.nan
    ok = out["skip_reason"].isna() & out["p"].notna() & (out["g_eff"] >= MIN_G)
    for st, idx in out[ok].groupby("stratum").groups.items():
        out.loc[idx, "holm_p"] = multipletests(out.loc[idx, "p"], method="holm")[1]
    out["status"] = [status_of(r) for r in out.to_dict("records")]
    return out


def artefact_checks(g: pd.DataFrame, targets: list[tuple[str, str, str]]) -> pd.DataFrame:
    """For each (stratum, measure, outcome) hit: categorical own send scores; investor-above/trustee-above
    split for give_align; gap model with the pair's own send; filled lags."""
    rows = []
    base = g[g["has_myths_before"]].copy()
    for w in ["inv", "tru"]:
        base[f"{w}_give_c"] = base[f"{w}_give"].round().astype("Int64").astype(str)
        base[f"{w}_lag_any_missing"] = base[f"{w}_lag_any"].isna().astype(float)
        base[f"{w}_lag_any_f"] = base[f"{w}_lag_any"].fillna(0)
    base["inv_above"] = (base["inv_give"] - base["tru_give"]).clip(lower=0)
    base["tru_above"] = (base["tru_give"] - base["inv_give"]).clip(lower=0)
    lags_f = ["inv_lag_any_f", "inv_lag_any_missing", "tru_lag_any_f", "tru_lag_any_missing",
              "inv_lag_same_f", "inv_lag_same_missing", "tru_lag_same_f", "tru_lag_same_missing"]
    smap = {s["stratum"]: s for s in strata(base)}
    for st, m, y in targets:
        d = base[smap[st]["mask"]].dropna(subset=[m]).copy()
        d["zx"] = z(d[m])
        size = int(d["size"].iloc[0])
        specs = {"own send scores as 0-10 categories": (["zx"] + LAGS, CATS + ["inv_give_c", "tru_give_c"]),
                 "missing lags filled (0 + indicator)": (["zx"] + lags_f + NUMS, CATS)}
        if m == "give_align":
            specs["split: investor above / trustee above (raw points)"] = (["inv_above", "tru_above"] + LAGS + NUMS, CATS)
        if y == "giving_gap":
            specs["gap model + own sent_frac"] = (["zx", "sent_frac"] + LAGS + NUMS, CATS)
        for spec, (x, c) in specs.items():
            f = feols(d, y, x, c, fe_for(size, "cell"), key=x[0])
            terms = ["inv_above", "tru_above"] if spec.startswith("split") else ["zx"]
            for t in terms:
                if f is not None and t in f["terms"]:
                    rows.append({"stratum": st, "measure": m, "outcome": y, "spec": spec, "term": t, **f["terms"][t],
                                 "n_games": f["n_games"], "g_eff": f["g_eff"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    g = load_games()
    g = add_nonpartner_placebo(g)
    g.to_csv(BIG / "norm_alignment_games_v2.csv", index=False)
    na.descriptives(g).to_csv(OUT / "descriptives_by_stratum.csv", index=False)
    prim = primary_models(g)
    prim.to_csv(OUT / "primary_models.csv", index=False)
    secondary_models(g).to_csv(OUT / "secondary_models.csv", index=False)
    founding_window(g).to_csv(OUT / "founding_window_R4.csv", index=False)
    founding_window(g, per_stratum=False).to_csv(OUT / "founding_window_R4_pooled_septspec.csv", index=False)
    reverse_models(g).to_csv(OUT / "reverse_models.csv", index=False)
    reverse_models(g, per_stratum=False).to_csv(OUT / "reverse_models_pooled_septspec.csv", index=False)
    hits = prim[prim["p"] < 0.05][["stratum", "measure", "outcome"]].itertuples(index=False, name=None)
    artefact_checks(g, list(hits)).to_csv(OUT / "artefact_checks.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    cols = ["stratum", "measure", "outcome", "coef", "ci_low", "ci_high", "p", "holm_p", "n_games", "n_runs", "g_eff", "status"]
    print(prim[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
