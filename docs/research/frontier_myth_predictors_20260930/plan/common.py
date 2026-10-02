"""Shared loaders for the frontier own-plan / amount / judges lens.

Ports the September amount lens (build_panel.py, common.py), search lens (r4_amount.py) and
judges lens (build_measures.py) into one dataset-parameterised panel. Read-only on the data dirs.

    load_myths(ds)  one row per myth with every myth measure (both judges)
    load_panel(ds)  one row per decision with own / own_prev / shown / unseen myth measures and
                    the move controls of the September H2 spec
ds is "september" or "frontier".
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as st

WT = Path(__file__).resolve().parents[4]  # frontier-myth worktree
OUT = Path(__file__).resolve().parent
KEY = ["run_id", "round", "agent"]
SEPT_LENS = WT / "docs/research/myth_predictors_20260930"
DATA = {"september": WT / "data/analysis/linguistic_20260923",
        "frontier": WT / "data/analysis/linguistic_frontier_20260930"}
RULES = {"september": ("myth_rules_september_z-ai__glm-5.2.csv", "myth_rules_september_deepseek__deepseek-v4-flash_sample1000.csv"),
         "frontier": ("myth_rules_frontier_z-ai__glm-5.2.csv", "myth_rules_frontier_deepseek__deepseek-v4-flash.csv")}
AMOUNT_CHECK = {"september": "myth_amount_check_september_z-ai__glm-5.2.csv",
                "frontier": "myth_amount_check_frontier_z-ai__glm-5.2.csv"}
GIVING_DIR = {"september": SEPT_LENS / "judges", "frontier": DATA["frontier"]}
# send-effect families (the ceiling family is reported separately as not estimable)
SEND_FAMS = {"september": ("Sonnet", "GPT"), "frontier": ("Sol", "Opus")}
CEILING_FAM = {"september": "Gemini", "frontier": "GeminiPro"}
MIDPOINT = {"all": 5.0, "most": 4.25, "moderate": 2.75, "little": 1.25, "none": 0.0}
SEND_ORD = {"none": 0, "little": 1, "moderate": 2, "most": 3, "all": 4}
LABEL_ORD = {"be cautious": 0, "be fair": 1, "be generous": 2}


def _read_rules(path: Path, suffix: str) -> pd.DataFrame:
    r = pd.read_csv(path)
    r = r[r["status"] == "ok"][KEY + ["send_rule", "send_amount", "return_rule", "consistency"]]
    return r.rename(columns={c: f"{c}{suffix}" for c in r.columns if c not in KEY})


def load_myths(ds: str) -> pd.DataFrame:
    d = DATA[ds]
    m = pd.read_csv(d / "myths.csv")
    glm_rules, ds_rules = RULES[ds]
    m = m.merge(_read_rules(d / glm_rules, ""), on=KEY, how="left")
    if (d / ds_rules).exists():
        m = m.merge(_read_rules(d / ds_rules, "_ds"), on=KEY, how="left")
    chk = pd.read_csv(d / AMOUNT_CHECK[ds])[KEY + ["amount_status"]]
    m = m.merge(chk, on=KEY, how="left")
    for j, tag in (("glm", "z-ai__glm-5.2"), ("ds", "deepseek__deepseek-v4-flash")):
        p = d / f"moral_labels_{tag}.csv"
        if p.exists():
            lab = pd.read_csv(p)
            lab = lab[lab["label_status"] == "ok"] if "label_status" in lab else lab
            m = m.merge(lab[KEY + ["label"]].rename(columns={"label": f"label_{j}"}), on=KEY, how="left")
        else:
            m[f"label_{j}"] = np.nan
        m[f"label_ord_{j}"] = m[f"label_{j}"].map(LABEL_ORD)
        m[f"gen_{j}"] = (m[f"label_{j}"] == "be generous").astype(float).where(m[f"label_{j}"].notna())
        g = GIVING_DIR[ds] / f"giving_scores_{tag}.csv"
        if g.exists():
            g = pd.read_csv(g)[KEY + ["g_send", "g_return"]]
            g.columns = KEY + [f"give_send_{j}", f"give_return_{j}"]
            m = m.merge(g, on=KEY, how="left")
    band = m["send_rule"].map(MIDPOINT)
    named = m["send_amount"].notna()
    m["judge_amount"] = m["send_amount"].where(named, band)                    # named, else band midpoint
    m["rule_prescribed"] = m["send_amount"].where(named & (m["amount_status"] == "endorsed"), band)
    m["rule_send_ord"] = m["send_rule"].map(SEND_ORD)
    m["setting"] = m["size"].astype(str) + "-agent " + np.where(m["mixed"], "mixed", "homogeneous")
    return m


FEATS = ["judge_amount", "send_amount", "rule_send_ord", "rule_prescribed", "give_send_glm", "give_send_ds",
         "label_ord_glm", "label_ord_ds", "gen_glm"]


def load_panel(ds: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = load_myths(ds)
    feats = [f for f in FEATS if f in m]
    d = pd.read_csv(DATA[ds] / "decisions.csv")
    d["setting"] = d["size"].astype(str) + "-agent " + np.where(d["mixed"], "mixed", "homogeneous")
    pair = d[["run_id", "round", "agent", "partner", "coop"]]
    contact_rounds = pair.groupby(["run_id", "agent", "partner"])["round"].apply(list).to_dict()
    d["own_round"] = np.where(d["task_order"] == "myth_game", d["round"], d["round"] - 1)
    mk = m.set_index(KEY)

    def attach(frame, rcol, acol, prefix, cols):
        idx = pd.MultiIndex.from_arrays([frame["run_id"], frame[rcol], frame[acol]])
        sub = mk.reindex(idx)[cols]
        sub.columns = [f"{prefix}{c}" for c in cols]
        return pd.concat([frame.reset_index(drop=True), sub.reset_index(drop=True)], axis=1)

    d = attach(d, "own_round", "agent", "own_", feats + ["exposed_author", "exposed_round", "exposed_family"])
    d["own_prev_round"] = d["own_round"] - 1
    d = attach(d, "own_prev_round", "agent", "own_prev_", feats)
    d = d.rename(columns={"own_exposed_author": "shown_author", "own_exposed_round": "shown_round",
                          "own_exposed_family": "shown_family"})
    d = attach(d, "shown_round", "shown_author", "shown_", feats)
    # the shown author's NEXT myth (same round as the reader's own myth; never shown to the reader)
    d["future_round"] = d["shown_round"] + 1
    d = attach(d, "future_round", "shown_author", "future_", ["judge_amount"])
    # unseen comparison: same run and round as the shown myth, same family as its author, neither reader nor author
    grp = {k: g for k, g in m.groupby(["run_id", "round"])}
    un = np.full(len(d), np.nan)
    for i, row in enumerate(d.itertuples()):
        if row.size == 8 and isinstance(row.shown_author, str):
            g = grp.get((row.run_id, row.shown_round))
            c = g[(g["family"] == row.shown_family) & (~g["agent"].isin([row.agent, row.shown_author]))]
            if len(c):
                un[i] = c["judge_amount"].mean()
    d["unseen_judge_amount"] = un
    pk = pair.set_index(["run_id", "agent", "partner", "round"])["coop"]

    def b_prev(row):
        if not isinstance(row.shown_author, str):
            return np.nan
        rounds = [r for r in contact_rounds.get((row.run_id, row.agent, row.shown_author), []) if r < row.round]
        return pk.get((row.run_id, row.shown_author, row.agent, max(rounds)), np.nan) if rounds else np.nan
    d["b_coop_prev"] = [b_prev(r) for r in d.itertuples()]
    d = d.sort_values(["run_id", "agent", "role", "round"])
    d["lag_sent"] = d.groupby(["run_id", "agent", "role"])["sent"].shift().where(d["role"] == "investor")
    d["coop_lag"] = d.groupby(["run_id", "agent", "role"])["coop"].shift()
    d = d.sort_values(["run_id", "agent", "round"])
    d["lag_coop_any"] = d.groupby(["run_id", "agent"])["coop"].shift()
    prevc = d.set_index(["run_id", "agent", "round"])["coop"]
    d["c_coop_prev"] = [prevc.get((r.run_id, r.partner, r.round - 1), np.nan) for r in d.itertuples()]
    d["send"] = d["sent"]
    d["runround"] = d["run_id"] + "|" + d["round"].astype(str)
    d["agentrun"] = d["run_id"] + "|" + d["agent"]
    d["cell"] = d["composition"] + "|" + d["task_order"]
    return m, d.reset_index(drop=True)


# ------------------------------------------------------------------ estimation
def demean(A: np.ndarray, groups: list, iters: int = 500, tol: float = 1e-10) -> np.ndarray:
    A = A - A.mean(0)
    codes = [pd.factorize(g)[0] for g in groups]
    for _ in range(iters if len(codes) > 1 else 1):
        prev = A.copy()
        for c in codes:
            s = np.zeros((c.max() + 1, A.shape[1]))
            np.add.at(s, c, A)
            A = A - (s / np.bincount(c)[:, None])[c]
        if np.abs(A - prev).max() < tol:
            break
    return A


def ceiling(y: pd.Series, min_off: int = 10, max_modal: float = 0.95) -> tuple[bool, str]:
    """An outcome is at the ceiling when fewer than min_off rows differ from its modal value or the
    modal value holds max_modal of the rows: a tight CI there is not an informative null."""
    y = y.dropna()
    if not len(y):
        return False, ""
    mode = y.round(6).mode().iloc[0]
    off = int((~np.isclose(y, mode)).sum())
    return (off < min_off or off / len(y) <= 1 - max_modal), f"{off} of {len(y)} rows off the modal value {mode:g}"


def t_inference(coef: float, se: float, n_clusters: int) -> tuple[float, float, float]:
    """CI and p from t(G-1), G = number of run clusters (frontier cells have 5-20 runs)."""
    q = st.t.ppf(0.975, n_clusters - 1)
    return coef - q * se, coef + q * se, 2 * st.t.sf(abs(coef / se), n_clusters - 1)


def fit(df: pd.DataFrame, formula: str, term: str, cluster: str = "run_id", min_resid_df: int = 10) -> dict:
    """OLS with fixed effects written C(x) absorbed by demeaning; SE clustered by run (statsmodels
    cluster-robust, as in the September amount lens), CI and p from t(G-1). Also returns the
    September-style normal-based CI/p (ci_low_z, ci_high_z, p_z) for the reproduction check."""
    lhs, rhs = [t.strip() for t in formula.split("~")]
    terms = [t.strip() for t in rhs.split("+")]
    fes = [re.match(r"C\((\w+)\)", t).group(1) for t in terms if t.startswith("C(")]
    xs = [t for t in terms if not t.startswith("C(")]
    df = df.dropna(subset=[lhs] + xs + fes)
    out = dict(n=len(df), n_runs=df[cluster].nunique() if len(df) else 0, coef=np.nan, ci_low=np.nan,
               ci_high=np.nan, p=np.nan, sd_y=df[lhs].std() if len(df) else np.nan,
               sd_x=df[term].std() if len(df) else np.nan, note="")
    if len(df) < 10 or out["n_runs"] < 4:
        out["note"] = "underpowered: too few rows or runs"
        return out
    if df[lhs].std() < 1e-9:
        out["note"] = "not estimable: outcome constant (ceiling)"
        return out
    at_ceiling, off = ceiling(df[lhs])
    out["outcome_spread"] = off
    A = demean(df[[lhs] + xs].to_numpy(float), [df[f].to_numpy() for f in fes])
    y, X = A[:, 0], A[:, 1:]
    j = xs.index(term)
    resid_df = len(df) - sum(df[f].nunique() for f in fes) - len(xs)
    if y.std() < 1e-9:
        out["note"] = "not estimable: no outcome variation within fixed effects"
        return out
    if X[:, j].std() < 1e-9:
        out["note"] = "not estimable: no predictor variation within fixed effects"
        return out
    if resid_df < min_resid_df:
        out["note"] = f"underpowered: {resid_df} residual df"
        return out
    r = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df[cluster])[0]})
    ci = r.conf_int()[j]
    lo, hi, p = t_inference(r.params[j], r.bse[j], out["n_runs"])
    out.update(coef=r.params[j], ci_low=lo, ci_high=hi, p=p, ci_low_z=ci[0], ci_high_z=ci[1], p_z=r.pvalues[j],
               resid_df=resid_df)
    if at_ceiling:
        out["note"] = f"not estimable: ceiling ({off})"
    return out


def status(row: dict, alpha: float = 0.05) -> str:
    note = row.get("note", "") or ""
    if "not testable" in note:
        return "not testable by design"
    if note.startswith("not estimable"):
        return "not estimable: ceiling"
    if note.startswith("underpowered") or (pd.notna(row.get("n_runs")) and row.get("n_runs", 99) < 10):
        return "underpowered"
    hp = row.get("holm_p", np.nan)
    p = row.get("p", np.nan)
    if pd.notna(hp) and hp < alpha:
        return "yes"
    if pd.notna(p) and p < alpha:
        return "suggestive"
    return "no detectable effect"


def spearman(a, b):
    ok = pd.notna(a) & pd.notna(b)
    return st.spearmanr(a[ok], b[ok]).statistic if ok.sum() > 2 else np.nan
