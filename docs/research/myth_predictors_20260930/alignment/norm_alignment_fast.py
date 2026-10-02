#!/usr/bin/env python3
"""Hypothesis 1, fast re-implementation of the model stage of norm_alignment.py.

norm_alignment.py builds norm_alignment_games.csv (one row per game, both players'
latest myths before it). Its statsmodels dummy-variable models took >70 min and
crashed, so this script absorbs the fixed effects by alternating-projection demeaning
and computes CR1 run-clustered SEs directly (equivalent point estimates).

Adds the judges lens's finer norm measure (predictors/judges/measures.csv):
  give_align = 1 - |send score A - send score B| / 10   (GLM 0-10 "how much should
               a sender give" score of each player's latest myth)
and a non-partner placebo for 8-agent runs: the investor's mean alignment with the six
agents it is NOT playing that round (same myth round). A pair-specific alignment effect
should beat it; a level/drift artefact would not.

Primary family (Holm): 5 measures x 3 outcomes x 2 sizes = 30 tests.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import norm_alignment as na

OUT = Path(__file__).resolve().parent
JUDGES = OUT.parent / "judges" / "measures.csv"
MEASURES = ["same_label", "moral_cos", "rule_index", "give_align", "myth_cos"]
OUTCOMES = na.OUTCOMES
CATS = ["inv_label", "tru_label", "inv_send_rule", "tru_send_rule"]
NUMS = ["inv_give", "tru_give"]
LAGS = ["inv_lag_any", "tru_lag_any", "inv_lag_same_f", "inv_lag_same_missing", "tru_lag_same_f", "tru_lag_same_missing"]


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
          cluster: str | None = "run_id") -> dict | None:
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
    keep = np.abs(XX).max(axis=0) > 1e-9
    XX, names = XX[:, keep], X.columns[keep]
    B = np.linalg.pinv(XX.T @ XX)
    beta = B @ XX.T @ yy
    e = yy - XX @ beta
    n, k = XX.shape
    # count absorbed FE levels in K (conservative, matches the statsmodels dummy-variable scripts)
    k_df = k + sum(len(np.unique(gr)) - 1 for gr in groups) + (1 if groups else 0)
    if cluster:
        cl = pd.factorize(d[cluster])[0]
        G = cl.max() + 1
        S = np.zeros((G, k))
        np.add.at(S, cl, XX * e[:, None])
        V = B @ (S.T @ S) @ B * (G / (G - 1)) * ((n - 1) / (n - k_df))
        dof = G - 1
    else:  # HC1
        V = B @ ((XX * e[:, None] ** 2).T @ XX) @ B * n / (n - k_df)
        dof, G = n - k_df, np.nan
    se = np.sqrt(np.diag(V))
    res = {}
    for i, nm in enumerate(names):
        t = beta[i] / se[i] if se[i] > 0 else np.nan
        q = stats.t.ppf(0.975, dof)
        res[nm] = {"coef": beta[i], "se": se[i], "ci_low": beta[i] - q * se[i], "ci_high": beta[i] + q * se[i],
                   "p": 2 * stats.t.sf(abs(t), dof)}
    return {"terms": res, "n_games": n, "n_runs": d["run_id"].nunique(), "data": d, "G": G}


# --------------------------------------------------------------------------- data

def load_games() -> pd.DataFrame:
    g = pd.read_csv(OUT / "norm_alignment_games.csv")
    m = pd.read_csv(JUDGES)[na.KEY + ["give_send_glm"]]
    give = m.set_index(na.KEY)["give_send_glm"]
    rb = np.where(g["task_order"] == "myth_game", g["round"], g["round"] - 1)
    ra = np.where(g["task_order"] == "myth_game", g["round"] + 1, g["round"])
    for w, col in (("inv", "investor"), ("tru", "trustee")):
        g[f"{w}_give"] = [give.get(k, np.nan) for k in zip(g["run_id"], rb, g[col])]
        g[f"{w}_give_after"] = [give.get(k, np.nan) for k in zip(g["run_id"], ra, g[col])]
    g["give_align"] = 1 - (g["inv_give"] - g["tru_give"]).abs() / 10
    g["give_align_after"] = 1 - (g["inv_give_after"] - g["tru_give_after"]).abs() / 10
    g["pair_give"] = (g["inv_give"] + g["tru_give"]) / 2
    g["cell"] = g["run_id"] + "|" + g["pair_family"]
    g["runround"] = g["run_id"] + "|" + g["round"].astype(str)
    g["inv_id"] = g["run_id"] + "|" + g["investor"]
    g["tru_id"] = g["run_id"] + "|" + g["trustee"]
    g["composition_order"] = g["composition"] + "|" + g["task_order"]
    return g


def add_nonpartner_placebo(g: pd.DataFrame) -> pd.DataFrame:
    """8-agent: investor's mean alignment with the 6 agents it is not playing (same myth round)."""
    myths, emb, semb = na.load_myths()
    give = pd.read_csv(JUDGES).set_index(na.KEY)["give_send_glm"]
    myths["give"] = [give.get(k, np.nan) for k in zip(myths["run_id"], myths["round"], myths["agent"])]
    by_rr = {k: v.index.to_numpy() for k, v in myths.groupby(["run_id", "round"])}
    idx = {(r, t, a): k for k, (r, t, a) in enumerate(zip(myths["run_id"], myths["round"], myths["agent"]))}
    out = {m: np.full(len(g), np.nan) for m in ["same_label", "moral_cos", "rule_index", "give_align", "myth_cos"]}
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


def eff_n(d: pd.DataFrame, y: str) -> int:
    return int((d.groupby("cell")[y].transform("std") > 0).sum())


def fe_for(size: int, kind: str) -> list[str]:
    if kind == "cell":
        return ["cell", "runround"] if size == 8 else ["cell", "round"]
    if kind == "agents":
        return ["inv_id", "tru_id", "round"]
    raise ValueError(kind)


# --------------------------------------------------------------------------- analyses

def before_models(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = g[g["has_myths_before"]]
    strata = {"2-agent": base[base["size"] == 2], "8-agent": base[base["size"] == 8]}
    extra = {s: base[base["setting"] == s] for s in base["setting"].unique()}
    extra["8-agent first meetings"] = base[(base["size"] == 8) & (base["prior_meetings"] == 0)]
    extra["8-agent repeat meetings"] = base[(base["size"] == 8) & (base["prior_meetings"] > 0)]
    for s in ["2-agent", "8-agent"]:
        for o in ["myth_game", "game_myth"]:
            extra[f"{s} {o}"] = strata[s][strata[s]["task_order"] == o]
    for fam in ["Sonnet", "GPT", "Gemini"]:
        extra[f"{fam}-{fam} games"] = base[(base["inv_family"] == fam) & (base["tru_family"] == fam)]

    def one(name, d, m, y, spec, primary, fe="cell", controls=True, side=None, lags=True, levels=True):
        d = d.copy()
        d["zx"] = z(d[m])
        x = ["zx"]
        if side:
            d["zside"] = z(d[side])
            x.append("zside")
        if lags:
            x += LAGS
        if levels:
            x += NUMS
        size = int(d["size"].iloc[0])
        f = feols(d, y, x, cats=CATS if levels else [], fes=fe_for(size, fe) if not name.endswith("games") or True else [])
        if f is None:
            return
        t = f["terms"]["zx"]
        rows.append({"stratum": name, "measure": m, "outcome": y, "spec": spec, "fe": fe, "primary": primary,
                     **t, "n_games": f["n_games"], "n_runs": f["n_runs"], "eff_n_games": eff_n(f["data"], y),
                     "raw_mean": d[m].mean(), "raw_sd": d[m].std()})
        if side:
            ts = f["terms"]["zside"]
            rows.append({"stratum": name, "measure": side, "outcome": y, "spec": spec + f" [coef of {side}]",
                         "fe": fe, "primary": False, **ts, "n_games": f["n_games"], "n_runs": f["n_runs"]})

    for name, d in strata.items():
        for y in OUTCOMES:
            for m in MEASURES:
                one(name, d, m, y, "alone + levels + lags", True)
                one(name, d, m, y, "alone + levels + lags", False, fe="agents")
                one(name, d, m, y, "no controls", False, lags=False, levels=False)
                one(name, d, m, y, "lags, no levels", False, levels=False)
                if m != "myth_cos":
                    one(name, d, m, y, "side by side with whole-myth similarity", False, side="myth_cos")
                if name == "8-agent":
                    one(name, d, m, y, "vs non-partner placebo", False, side=f"np_{m}")
            one(name, d.assign(same_label=d["same_label_ds"]), "same_label", y, "DeepSeek labels", False)
            one(name, d[d["judges_agree"] == 1], "same_label", y, "both judges agree on both labels", False)
            for fld in na.FIELDS:
                one(name, d, fld, y, "rule field alone + levels + lags", False)
    for name, d in extra.items():
        for y in OUTCOMES:
            for m in MEASURES:
                one(name, d, m, y, "alone + levels + lags", False)
    out = pd.DataFrame(rows)
    prim = out["primary"] & out["p"].notna()
    out.loc[prim, "p_holm"] = multipletests(out.loc[prim, "p"], method="holm")[1]
    return out


def interaction_models(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = g[g["has_myths_before"]]
    for name in ["2-agent", "8-agent"]:
        d = base[base["size"] == int(name[0])].copy()
        for lab in ["generous", "fair", "cautious"]:
            d[f"both_{lab}"] = (d["label_pair"] == f"both {lab}").astype(float)
        for y in OUTCOMES:
            f = feols(d, y, ["both_generous", "both_fair", "both_cautious"] + LAGS + NUMS, CATS, fe_for(int(name[0]), "cell"))
            for t in ["both_generous", "both_fair", "both_cautious"]:
                rows.append({"stratum": name, "outcome": y, "test": f"{t.replace('_', ' ')} vs different labels, same own-label levels",
                             **f["terms"][t], "n_games": f["n_games"], "n_with": int(f["data"][t].sum())})
            for m, lvl in (("give_align", "pair_give"), ("moral_cos", "pair_label_level"),
                           ("rule_index", "pair_send_ord"), ("same_label", "pair_label_level")):
                e = d.dropna(subset=[m, lvl]).copy()
                e["za"], e["zl"] = z(e[m]), z(e[lvl])
                e["zaxl"] = e["za"] * e["zl"]
                f = feols(e, y, ["za", "zl", "zaxl"] + LAGS + NUMS, CATS, fe_for(int(name[0]), "cell"))
                rows.append({"stratum": name, "outcome": y, "test": f"{m} x {lvl} (+ = alignment matters more at generous level)",
                             **f["terms"]["zaxl"], "n_games": f["n_games"]})
    return pd.DataFrame(rows)


def founding_window(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d0 = g[(g["task_order"] == "myth_game") & (g["round"] == 1) & g["has_myths_before"]]
    for name, d, fes, cl in (("8-agent", d0[d0["size"] == 8], ["run_id"], "run_id"),
                             ("2-agent", d0[d0["size"] == 2], ["composition"], None)):
        for y in OUTCOMES:
            for m in MEASURES:
                e = d.dropna(subset=[m]).copy()
                e["zx"] = z(e[m])
                for spec, x in (("alone", ["zx"]), ("+ own send scores", ["zx"] + NUMS)):
                    f = feols(e, y, x, [], fes, cluster=cl)
                    if f is None:
                        continue
                    rows.append({"stratum": name, "measure": m, "outcome": y, "spec": spec, **f["terms"]["zx"],
                                 "n_games": f["n_games"], "n_runs": f["n_runs"], "eff_n_games": eff_n(f["data"].assign(cell=f["data"][fes[0]]), y),
                                 "raw_mean": e[m].mean()})
    return pd.DataFrame(rows)


def reverse_models(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for size in [2, 8]:
        for order in ["myth_game", "game_myth", "both"]:
            d = g[g["size"] == size]
            if order != "both":
                d = d[d["task_order"] == order]
            for m in MEASURES:
                for x in ["sent_frac", "return_proportion"]:
                    e = d.dropna(subset=[f"{m}_after", x]).copy()
                    e["before_missing"] = e[m].isna().astype(float)
                    e["before_f"] = e[m].fillna(e[m].mean())
                    f = feols(e, f"{m}_after", [x, "before_f", "before_missing"], [], fe_for(size, "cell"))
                    if f is None:
                        continue
                    rows.append({"stratum": f"{size}-agent", "task_order": order, "measure": f"{m} after the game",
                                 "predictor": x, **f["terms"][x], "n_games": f["n_games"], "n_runs": f["n_runs"],
                                 "sd_after": e[f"{m}_after"].std()})
    return pd.DataFrame(rows)


def main() -> None:
    g = load_games()
    g = add_nonpartner_placebo(g)
    g.to_csv(OUT / "norm_alignment_games_v2.csv", index=False)
    g[g["has_myths_before"]][MEASURES + ["pair_label_level", "pair_send_ord", "pair_give"]].corr(
        method="spearman").to_csv(OUT / "measure_correlations.csv")
    res = before_models(g)
    res.to_csv(OUT / "before_game_models.csv", index=False)
    interaction_models(g).to_csv(OUT / "interaction_models.csv", index=False)
    founding_window(g).to_csv(OUT / "founding_window_R4.csv", index=False)
    reverse_models(g).to_csv(OUT / "reverse_models.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["stratum", "measure", "outcome", "coef", "ci_low", "ci_high", "p", "p_holm", "n_games", "eff_n_games", "n_runs"]
    print(res[res["primary"]][cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
