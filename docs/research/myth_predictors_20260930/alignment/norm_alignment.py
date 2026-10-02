#!/usr/bin/env python3
"""Hypothesis 1: does NORM alignment between two players predict their cooperation?

Extends analyses/alignment_vs_cooperation.py (whole-myth embedding similarity, null
within runs) with norm-level alignment measures built from the judge outputs:

  same_label   both players' latest myths carry the same GLM-5.2 moral label
  moral_cos    cosine of the one-sentence moral summaries (mpnet)
  rule_index   mean agreement over the structured rule fields both players specified
               (send_rule, return_rule, after_letdown exact match; 1-|d send_amount|/5;
               consistency; test_first). unspecified is never counted as agreement.
  myth_cos     whole-myth cosine (the old measure), run side by side

Every game (investor decision) gets the two players' latest myths before it
(myth_game: same round; game_myth: previous round). Outcomes: sent/5, return
proportion, giving gap |sent/5 - return proportion|.

Primary model (R1+): outcome ~ z(alignment) + both players' own norm levels (label,
send_rule as categories) + each player's previous-game cooperation (any role) and
same-role lag + run x pair-family FE + round FE (8-agent: run x round FE too).
SE clustered by run. Holm over the declared primary family (4 measures x 3 outcomes
x 2 sizes = 24 tests). Everything else is secondary and labelled as such.

Reads the shared tables read-only; writes only to this folder.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

WT = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/moral-spread-viz")
DATA = WT / "data/analysis/linguistic_20260923"
OUT = Path(__file__).resolve().parent
KEY = ["run_id", "round", "agent"]
MIN_WORDS = 20
SEND_ORD = {"none": 0, "little": 1, "moderate": 2, "most": 3, "all": 4}
LABEL_LEVEL = {"be cautious": 0, "be fair": 1, "be generous": 2}
OUTCOMES = ["sent_frac", "return_proportion", "giving_gap"]
MEASURES = ["same_label", "moral_cos", "rule_index", "myth_cos"]
FIELDS = ["f_send_rule", "f_return_rule", "f_after_letdown", "f_send_amount", "f_consistency", "f_test_first"]


# --------------------------------------------------------------------------- data

def load_myths() -> pd.DataFrame:
    myths = pd.read_csv(DATA / "myths.csv").reset_index(drop=True)
    emb = np.load(DATA / "embeddings_mpnet.npy")
    semb = np.load(DATA / "embeddings_moral_summary_mpnet.npy")
    assert len(emb) == len(semb) == len(myths)
    glm = pd.read_csv(DATA / "moral_labels_z-ai__glm-5.2.csv")[KEY + ["label", "summary"]]
    ds = pd.read_csv(DATA / "moral_labels_deepseek__deepseek-v4-flash.csv")[KEY + ["label"]].rename(
        columns={"label": "label_ds"})
    rules = pd.read_csv(DATA / "myth_rules_september_z-ai__glm-5.2.csv")
    rules = rules[KEY + ["send_rule", "return_rule", "after_letdown", "send_amount", "test_first",
                         "consistency", "status"]]
    n = len(myths)
    myths = myths.merge(glm, on=KEY, how="left").merge(ds, on=KEY, how="left").merge(rules, on=KEY, how="left")
    assert len(myths) == n
    myths["row"] = np.arange(n)
    myths["valid"] = myths["n_words"] >= MIN_WORDS
    # rule fields: anything the judge could not parse or marked unspecified -> NaN
    ok = myths["status"] == "ok"
    for c in ["send_rule", "return_rule", "after_letdown"]:
        myths[c] = myths[c].where(ok & (myths[c] != "unspecified"))
    for c in ["test_first", "consistency"]:
        myths[c] = myths[c].where(ok).map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    myths["send_amount"] = myths["send_amount"].where(ok)
    myths["send_ord"] = myths["send_rule"].map(SEND_ORD)
    myths["label_level"] = myths["label"].map(LABEL_LEVEL)
    return myths, emb, semb


def decisions() -> pd.DataFrame:
    dec = pd.read_csv(DATA / "decisions.csv").sort_values(["run_id", "agent", "round"])
    # previous game of this agent (any role) and previous game in the same role
    dec["coop_lag_any"] = dec.groupby(["run_id", "agent"])["coop"].shift(1)
    dec["coop_lag_same"] = dec.groupby(["run_id", "agent", "role"])["coop"].shift(1)
    return dec


def agree(a, b):
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a == b)


def pair_measures(myths, emb, semb, i, j) -> dict:
    """Norm and text alignment between myth rows i and j (either may be None)."""
    if i is None or j is None:
        return {}
    A, B = myths.iloc[i], myths.iloc[j]
    out = {"same_label": agree(A.label, B.label), "same_label_ds": agree(A.label_ds, B.label_ds),
           "judges_agree": float(A.label == A.label_ds and B.label == B.label_ds)
           if isinstance(A.label, str) and isinstance(B.label, str) else np.nan,
           "moral_cos": float(semb[i] @ semb[j]) if isinstance(A.summary, str) and isinstance(B.summary, str) else np.nan,
           "myth_cos": float(emb[i] @ emb[j]) if A.valid and B.valid else np.nan,
           "f_send_rule": agree(A.send_rule, B.send_rule), "f_return_rule": agree(A.return_rule, B.return_rule),
           "f_after_letdown": agree(A.after_letdown, B.after_letdown),
           "f_send_amount": np.nan if pd.isna(A.send_amount) or pd.isna(B.send_amount)
           else 1 - abs(A.send_amount - B.send_amount) / 5,
           "f_consistency": agree(A.consistency, B.consistency), "f_test_first": agree(A.test_first, B.test_first),
           "send_ord_dist": np.nan if pd.isna(A.send_ord) or pd.isna(B.send_ord) else abs(A.send_ord - B.send_ord)}
    f = [out[k] for k in FIELDS if not pd.isna(out[k])]
    out["rule_index"] = float(np.mean(f)) if len(f) >= 2 else np.nan
    out["rule_n_fields"] = len(f)
    return out


def games_table() -> pd.DataFrame:
    myths, emb, semb = load_myths()
    idx = {(r, t, a): k for k, (r, t, a) in enumerate(zip(myths["run_id"], myths["round"], myths["agent"]))}
    dec = decisions()
    lag = dec.set_index(KEY)[["coop_lag_any", "coop_lag_same"]]
    inv = dec[dec["role"] == "investor"].rename(columns={"agent": "investor", "partner": "trustee",
                                                         "family": "inv_family", "partner_family": "tru_family"})
    g = inv[["run_id", "size", "mixed", "composition", "task_order", "replicate_id", "round", "investor", "trustee",
             "inv_family", "tru_family", "sent", "return_proportion"]].copy().reset_index(drop=True)
    g["sent_frac"] = g["sent"] / 5
    g["giving_gap"] = (g["sent_frac"] - g["return_proportion"]).abs()
    g["pair_family"] = ["-".join(sorted(p)) for p in zip(g["inv_family"], g["tru_family"])]
    g["pair_type"] = np.where(g["inv_family"] == g["tru_family"], "same family", "cross family")
    g["setting"] = [f"{s}-agent {'mixed' if m else 'homogeneous'}" for s, m in zip(g["size"], g["mixed"])]
    # how many times this pair met before this game (8-agent: first meetings share no history)
    g["pair"] = ["|".join(sorted(p)) for p in zip(g["investor"], g["trustee"])]
    g = g.sort_values(["run_id", "round"]).reset_index(drop=True)
    g["prior_meetings"] = g.groupby(["run_id", "pair"]).cumcount()
    for who in ["investor", "trustee"]:
        keys = list(zip(g["run_id"], g["round"], g[who]))
        g[f"{who[:3]}_lag_any"] = [lag["coop_lag_any"].get(k, np.nan) for k in keys]
        g[f"{who[:3]}_lag_same"] = [lag["coop_lag_same"].get(k, np.nan) for k in keys]

    rows_b, rows_a, lev = [], [], []
    for r in g.itertuples(index=False):
        rb = r.round if r.task_order == "myth_game" else r.round - 1
        ra = r.round + 1 if r.task_order == "myth_game" else r.round
        ib, jb = idx.get((r.run_id, rb, r.investor)), idx.get((r.run_id, rb, r.trustee))
        ia, ja = idx.get((r.run_id, ra, r.investor)), idx.get((r.run_id, ra, r.trustee))
        rows_b.append(pair_measures(myths, emb, semb, ib, jb))
        rows_a.append(pair_measures(myths, emb, semb, ia, ja))
        L = {}
        for who, k in (("inv", ib), ("tru", jb)):
            m = myths.iloc[k] if k is not None else None
            L[f"{who}_label"] = m.label if m is not None and isinstance(m.label, str) else np.nan
            L[f"{who}_label_level"] = m.label_level if m is not None else np.nan
            L[f"{who}_send_rule"] = (m.send_rule if isinstance(m.send_rule, str) else "unspecified") if m is not None else np.nan
            L[f"{who}_send_ord"] = m.send_ord if m is not None else np.nan
            L[f"{who}_return_rule"] = (m.return_rule if isinstance(m.return_rule, str) else "unspecified") if m is not None else np.nan
            L[f"{who}_myth_round"] = rb if k is not None else np.nan
        lev.append(L)
    before = pd.DataFrame(rows_b)
    after = pd.DataFrame(rows_a).add_suffix("_after")
    g = pd.concat([g, before, after, pd.DataFrame(lev)], axis=1)
    g["has_myths_before"] = g["inv_label"].notna() & g["tru_label"].notna()
    g["pair_label_level"] = (g["inv_label_level"] + g["tru_label_level"]) / 2
    g["pair_send_ord"] = (g["inv_send_ord"] + g["tru_send_ord"]) / 2
    g["label_pair"] = np.where(g["same_label"] == 1, "both " + g["inv_label"].astype(str).str.replace("be ", ""),
                               "different labels")
    # same-role lag is missing when a player has not yet held this role (dyads alternate roles)
    for w in ["inv", "tru"]:
        g[f"{w}_lag_same_missing"] = g[f"{w}_lag_same"].isna().astype(float)
        g[f"{w}_lag_same_f"] = g[f"{w}_lag_same"].fillna(0)
    return g


# --------------------------------------------------------------------------- models

LEVELS = "C(inv_label) + C(tru_label) + C(inv_send_rule) + C(tru_send_rule)"
LAGS = "inv_lag_any + tru_lag_any + inv_lag_same_f + inv_lag_same_missing + tru_lag_same_f + tru_lag_same_missing"


def fe_terms(d: pd.DataFrame, fe: str) -> str:
    if fe == "cell":  # primary: run x pair-family + round; 8-agent also run x round
        return "C(cell) + C(runround)" if d["size"].iloc[0] == 8 else "C(cell) + C(round)"
    if fe == "agents":  # R2: investor-within-run and trustee-within-run FE + round
        return "C(inv_id) + C(tru_id) + C(round)"
    if fe == "cell_only":
        return "C(cell)"
    if fe == "composition":
        return "C(composition)"
    raise ValueError(fe)


def prep(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["cell"] = d["run_id"] + "|" + d["pair_family"]
    d["runround"] = d["run_id"] + "|" + d["round"].astype(str)
    d["inv_id"] = d["run_id"] + "|" + d["investor"]
    d["tru_id"] = d["run_id"] + "|" + d["trustee"]
    return d


def fit(d: pd.DataFrame, y: str, terms: list[str], rhs: str, cluster: bool = True):
    import re
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", f"{y} {rhs}"))
    need = [y] + [c for c in terms] + [c for c in d.columns if c in tokens]
    d = d.dropna(subset=list(dict.fromkeys(need)))
    if len(d) < 30:
        return None, d
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cluster and d["run_id"].nunique() >= 5:
            f = smf.ols(f"{y} ~ {rhs}", data=d).fit(cov_type="cluster",
                                                   cov_kwds={"groups": pd.factorize(d["run_id"])[0]})
        else:
            f = smf.ols(f"{y} ~ {rhs}", data=d).fit(cov_type="HC1")
    return f, d


def effective_n(d: pd.DataFrame, y: str, group: str = "cell") -> int:
    sd = d.groupby(group)[y].transform("std")
    return int((sd > 0).sum())


def zscore(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = d.copy()
    for c in cols:
        s = d[c]
        d[f"z_{c}"] = (s - s.mean()) / s.std() if s.std() > 0 else np.nan
    return d


def coef_row(f, d, term, **meta) -> dict:
    if f is None or term not in f.params:
        return {**meta, "n_games": len(d)}
    ci = f.conf_int().loc[term]
    return {**meta, "coef": f.params[term], "ci_low": ci[0], "ci_high": ci[1], "p": f.pvalues[term],
            "n_games": int(f.nobs), "n_runs": d["run_id"].nunique(), "eff_n_games": effective_n(prep(d), meta.get("outcome", "sent_frac"))}


def run_models(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = prep(g[g["has_myths_before"]])
    strata = {"2-agent": base[base["size"] == 2], "8-agent": base[base["size"] == 8]}
    extra = {"2-agent homogeneous": base[base["setting"] == "2-agent homogeneous"],
             "2-agent mixed": base[base["setting"] == "2-agent mixed"],
             "8-agent homogeneous": base[base["setting"] == "8-agent homogeneous"],
             "8-agent mixed": base[base["setting"] == "8-agent mixed"],
             "8-agent first meetings": base[(base["size"] == 8) & (base["prior_meetings"] == 0)],
             "8-agent repeat meetings": base[(base["size"] == 8) & (base["prior_meetings"] > 0)]}
    for order in ["myth_game", "game_myth"]:
        for s in ["2-agent", "8-agent"]:
            extra[f"{s} {order}"] = strata[s][strata[s]["task_order"] == order]

    def one(name, d, measure, y, spec, fe, primary, extra_terms="", controls=True, cols=None):
        if d.empty:
            return
        d = zscore(d.dropna(subset=[measure]), [measure] + (cols or []))
        rhs = [f"z_{measure}"] + [f"z_{c}" for c in (cols or [])]
        form = " + ".join(rhs + ([LEVELS, LAGS] if controls else []) + ([extra_terms] if extra_terms else [])
                          + [fe_terms(d, fe)])
        need = rhs + (["inv_lag_any", "tru_lag_any", "inv_label", "tru_label"] if controls else [])
        f, dd = fit(d, y, need, form)
        rows.append(coef_row(f, dd, f"z_{measure}", stratum=name, measure=measure, outcome=y, spec=spec, fe=fe,
                             primary=primary, sd_raw=d[measure].std(), mean_raw=d[measure].mean()))
        for c in cols or []:
            rows.append(coef_row(f, dd, f"z_{c}", stratum=name, measure=c, outcome=y, spec=spec + f" [{c} coef]",
                                 fe=fe, primary=False, sd_raw=d[c].std(), mean_raw=d[c].mean()))

    for name, d in strata.items():
        for y in OUTCOMES:
            for m in MEASURES:
                one(name, d, m, y, "alone + levels + lags", "cell", True)
                one(name, d, m, y, "alone + levels + lags", "agents", False)
                one(name, d, m, y, "alone, no controls", "cell", False, controls=False)
                one(name, d, m, y, "alone + lags, no levels", "cell", False,
                    extra_terms=LAGS, controls=False)
                if m != "myth_cos":
                    one(name, d, m, y, "norm + whole-myth side by side", "cell", False, cols=["myth_cos"])
                if m == "same_label":
                    one(name, d.assign(same_label=d["same_label_ds"]), m, y, "DeepSeek labels", "cell", False)
                    one(name, d[d["judges_agree"] == 1], m, y, "both judges agree on both labels", "cell", False)
            for fld in FIELDS:
                one(name, d, fld, y, "field alone + levels + lags", "cell", False)
    for name, d in extra.items():
        for y in OUTCOMES:
            for m in MEASURES:
                one(name, d, m, y, "alone + levels + lags", "cell", False)
    out = pd.DataFrame(rows)
    prim = out["primary"] & out["p"].notna()
    out.loc[prim, "p_holm"] = multipletests(out.loc[prim, "p"], method="holm")[1]
    return out


def interaction_models(g: pd.DataFrame) -> pd.DataFrame:
    """Aligned on generous vs aligned on cautious: match terms per label, and alignment x level."""
    rows = []
    base = prep(g[g["has_myths_before"]])
    for name, d in (("2-agent", base[base["size"] == 2]), ("8-agent", base[base["size"] == 8])):
        for y in OUTCOMES:
            dd = d.copy()
            for lab in ["generous", "fair", "cautious"]:
                dd[f"both_{lab}"] = (dd["label_pair"] == f"both {lab}").astype(float)
            form = f"{y} ~ both_generous + both_fair + both_cautious + {LEVELS} + {LAGS} + {fe_terms(dd, 'cell')}"
            f, fd = fit(dd, y, ["inv_lag_any", "tru_lag_any", "inv_label", "tru_label"], form)
            for t in ["both_generous", "both_fair", "both_cautious"]:
                r = coef_row(f, fd, t, stratum=name, outcome=y, test=f"{t} (vs same levels, different labels)")
                r["n_cells_with_term"] = int(fd[t].sum()) if f is not None else np.nan
                rows.append(r)
            # joint test: full label x label interaction beyond additive levels
            form_full = f"{y} ~ C(inv_label):C(tru_label) + C(inv_send_rule) + C(tru_send_rule) + {LAGS} + {fe_terms(dd, 'cell')}"
            f2, fd2 = fit(dd, y, ["inv_lag_any", "tru_lag_any", "inv_label", "tru_label"], form_full)
            f1, _ = fit(dd, y, ["inv_lag_any", "tru_lag_any", "inv_label", "tru_label"], form.replace("both_generous + both_fair + both_cautious + ", ""))
            if f1 is not None and f2 is not None:
                # nested-model F test (homoskedastic; the cluster-robust version is the per-term tests above)
                df_diff = f1.df_resid - f2.df_resid
                F = ((f1.ssr - f2.ssr) / df_diff) / (f2.ssr / f2.df_resid) if df_diff > 0 else np.nan
                from scipy import stats
                rows.append({"stratum": name, "outcome": y, "test": "label x label interaction beyond additive levels (F)",
                             "coef": F, "p": stats.f.sf(F, df_diff, f2.df_resid) if df_diff > 0 else np.nan,
                             "n_games": int(f2.nobs), "df": df_diff})
            # continuous: alignment x pair level
            for m, lvl in (("moral_cos", "pair_label_level"), ("rule_index", "pair_send_ord"), ("same_label", "pair_label_level")):
                e = zscore(dd.dropna(subset=[m, lvl]), [m, lvl])
                form_i = f"{y} ~ z_{m} * z_{lvl} + {LEVELS} + {LAGS} + {fe_terms(e, 'cell')}"
                fi, fdi = fit(e, y, ["inv_lag_any", "tru_lag_any", "inv_label", "tru_label"], form_i)
                rows.append(coef_row(fi, fdi, f"z_{m}:z_{lvl}", stratum=name, outcome=y,
                                     test=f"{m} x {lvl} (positive = alignment matters more when norms are generous)"))
    return pd.DataFrame(rows)


def founding_window(g: pd.DataFrame) -> pd.DataFrame:
    """R4: round-1 myth_game games; both myths written before any play and before any exposure."""
    rows = []
    d0 = prep(g[(g["task_order"] == "myth_game") & (g["round"] == 1) & g["has_myths_before"]])
    for name, d, fe in (("8-agent", d0[d0["size"] == 8], "cell_only"), ("2-agent", d0[d0["size"] == 2], "composition")):
        for y in OUTCOMES:
            for m in MEASURES:
                e = zscore(d.dropna(subset=[m]), [m])
                for spec, rhs in (("alone", f"z_{m}"),
                                  ("+ own levels", f"z_{m} + C(inv_label) + C(tru_label) + inv_send_ord_f + tru_send_ord_f")):
                    e2 = e.assign(inv_send_ord_f=e["inv_send_ord"].fillna(e["inv_send_ord"].mean()),
                                  tru_send_ord_f=e["tru_send_ord"].fillna(e["tru_send_ord"].mean()))
                    f, fd = fit(e2, y, [f"z_{m}"], f"{rhs} + {fe_terms(e2, fe)}", cluster=(fe != "composition"))
                    rows.append(coef_row(f, fd, f"z_{m}", stratum=name, measure=m, outcome=y, spec=spec, fe=fe,
                                         mean_raw=e[m].mean(), sd_raw=e[m].std()))
    return pd.DataFrame(rows)


def reverse(g: pd.DataFrame) -> pd.DataFrame:
    """Does a cooperative game make the NEXT myths more aligned (controlling for alignment before)?"""
    rows = []
    base = prep(g)
    for size in [2, 8]:
        for order in ["myth_game", "game_myth", "both"]:
            d = base[base["size"] == size]
            if order != "both":
                d = d[d["task_order"] == order]
            for m in MEASURES:
                for x in ["sent_frac", "return_proportion"]:
                    e = d.dropna(subset=[f"{m}_after", x])
                    has_before = e[m].notna()
                    e = e.assign(before_f=e[m].fillna(e[m].mean()), before_missing=(~has_before).astype(float))
                    form = f"{m}_after ~ {x} + before_f + before_missing + {fe_terms(e, 'cell')}"
                    f, fd = fit(e, f"{m}_after", [x], form)
                    rows.append(coef_row(f, fd, x, stratum=f"{size}-agent", task_order=order, measure=m + " after the game",
                                         outcome=x, sd_after=e[f"{m}_after"].std()))
    out = pd.DataFrame(rows)
    return out


def descriptives(g: pd.DataFrame) -> pd.DataFrame:
    d = g[g["has_myths_before"]]
    per_run = d.groupby(["setting", "task_order", "run_id"])[MEASURES + FIELDS + ["sent_frac", "return_proportion", "giving_gap"]].mean()
    agg = per_run.groupby(["setting", "task_order"]).agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    n = d.groupby(["setting", "task_order"]).agg(n_games=("run_id", "size"), n_runs=("run_id", "nunique"))
    field_n = d.groupby(["setting", "task_order"])[FIELDS].count().add_suffix("_n_games")
    return n.join(agg).join(field_n).reset_index()


def main() -> None:
    g = games_table()
    g.to_csv(OUT / "norm_alignment_games.csv", index=False)
    desc = descriptives(g)
    desc.to_csv(OUT / "descriptives_by_setting.csv", index=False)
    corr = g[g["has_myths_before"]][MEASURES + ["pair_label_level", "pair_send_ord"]].corr(method="spearman")
    corr.to_csv(OUT / "measure_correlations.csv")
    res = run_models(g)
    res.to_csv(OUT / "before_game_models.csv", index=False)
    inter = interaction_models(g)
    inter.to_csv(OUT / "interaction_models.csv", index=False)
    fw = founding_window(g)
    fw.to_csv(OUT / "founding_window_R4.csv", index=False)
    rv = reverse(g)
    rv.to_csv(OUT / "reverse_models.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["stratum", "measure", "outcome", "coef", "ci_low", "ci_high", "p", "p_holm", "n_games", "eff_n_games", "n_runs"]
    print(res[res["primary"]][cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
