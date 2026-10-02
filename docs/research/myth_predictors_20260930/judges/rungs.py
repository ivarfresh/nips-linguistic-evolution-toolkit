#!/usr/bin/env python3
"""Decisive test: does a finer measure of 'how much the myth recommends giving' predict cooperation
where the 3-way label does not? Same rows, same rungs, predictors standardised (per pooled SD over myths).

Outcome coop: sent/5 (investor), return proportion (trustee). Gemini decisions dropped from the main
table (it sends 5 in almost every decision); a Gemini-included R2 row set is written separately.

R0  pooled, no controls (descriptive)
R1  cell (composition x task order) + family + round FE
R2  agent-within-run + round FE + own lagged coop in the same role
R3  8-agent myth->game: the SHOWN myth, written before its author had ever played the reader;
    agent + round FE, own lag, and the shown author's own coop toward the reader in their game
R4  myth->game round 1: own round-1 myth (written before any play) -> round-1 decision;
    family + composition FE (+ amount received for trustees)
SE clustered by run. Outputs: rung_estimates.csv, reliability_within.csv, noise_ceiling.csv
"""
from pathlib import Path
import sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
WT = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/moral-spread-viz")
D = WT / "data/analysis/linguistic_20260923"
KEY = ["run_id", "round", "agent"]
INV_MEASURES = ["label_ord_glm", "gen_glm", "label_ord_ds", "rule_prescribed", "rule_send_ord",
                "give_send_glm", "give_send_ds", "emb_axis_summary", "emb_axis_text"]
TRU_MEASURES = ["label_ord_glm", "gen_glm", "label_ord_ds", "give_return_glm", "give_return_ds",
                "give_send_glm", "emb_axis_summary", "emb_axis_text"]
PAIRS = {"label_ord": ("label_ord_glm", "label_ord_ds"), "gen": ("gen_glm", "gen_ds"),
         "give_send": ("give_send_glm", "give_send_ds"), "give_return": ("give_return_glm", "give_return_ds"),
         "give_cond": ("give_cond_glm", "give_cond_ds")}


def load():
    m = pd.read_csv(HERE / "measures.csv")
    zcols = sorted({*INV_MEASURES, *TRU_MEASURES, "gen_ds", "give_return_ds", "give_cond_glm", "give_cond_ds",
                    "give_send_ds"} & set(m.columns))
    for c in zcols:
        m[c] = (m[c] - m[c].mean()) / m[c].std()
    mi = m.set_index(KEY)
    dec = pd.read_csv(D / "decisions.csv").sort_values(["run_id", "agent", "round"]).reset_index(drop=True)
    dec["r_before"] = np.where(dec.task_order == "myth_game", dec["round"], dec["round"] - 1)
    own = mi.reindex(pd.MultiIndex.from_arrays([dec.run_id, dec.r_before, dec.agent]))
    for c in zcols:
        dec[c] = own[c].to_numpy()
    dec["shown_author"] = own["exposed_author"].to_numpy()
    dec["shown_round"] = own["exposed_round"].to_numpy()
    sh_idx = pd.MultiIndex.from_arrays([dec.run_id, dec.shown_round.fillna(-1).astype(int), dec.shown_author.fillna("")])
    sh = mi.reindex(sh_idx)
    for c in zcols:
        dec["shown_" + c] = sh[c].to_numpy()
    dec["coop_lag"] = dec.groupby(["run_id", "agent", "role"])["coop"].shift(1)
    dec["run_agent"] = dec.run_id + "|" + dec.agent
    dec["cell"] = dec.composition + "|" + dec.task_order
    # R3 cleanliness: had the shown author played the reader before the shown myth's round?
    games = dec[dec.role == "investor"][["run_id", "round", "agent", "partner", "coop"]]
    met = {}
    for r in games.itertuples(index=False):
        for a, b in ((r.agent, r.partner), (r.partner, r.agent)):
            met.setdefault((r.run_id, a, b), []).append(r.round)
    dec["shown_met_before_writing"] = [
        (isinstance(sa, str) and any(k < sr for k in met.get((run, ag, sa), [])))
        for run, ag, sa, sr in zip(dec.run_id, dec.agent, dec.shown_author, dec.shown_round.fillna(0))]
    # what the shown author did to the reader in the shown round (its coop in their game)
    both = dec.set_index(["run_id", "round", "agent", "partner"])["coop"]
    dec["shown_author_coop_to_reader"] = both.reindex(pd.MultiIndex.from_arrays(
        [dec.run_id, dec.shown_round.fillna(-1).astype(int), dec.shown_author.fillna(""), dec.agent])).to_numpy()
    # R4 trustee control: amount received (sent/5 by the partner in the same game)
    inv = dec[dec.role == "investor"].set_index(["run_id", "round", "agent"])["coop"]
    dec["received_frac"] = inv.reindex(pd.MultiIndex.from_arrays([dec.run_id, dec["round"], dec.partner])).to_numpy()
    return m, dec


def _demean(M, groups, iters=200, tol=1e-10):
    M = M - M.mean(0)
    if not groups:
        return M
    for _ in range(iters):
        old = M.copy()
        for g in groups:
            means = pd.DataFrame(M).groupby(g).transform("mean").to_numpy()
            M = M - means
        if np.abs(M - old).max() < tol:
            break
    return M


def fit(formula, data, term):
    """OLS with absorbed fixed effects (terms written C(x)) and run-clustered CR1 SEs."""
    from scipy import stats as st
    lhs, rhs = [t.strip() for t in formula.split("~")]
    terms = [t.strip() for t in rhs.split("+")]
    fes = [t[2:-1] for t in terms if t.startswith("C(")]
    xs = [t for t in terms if not t.startswith("C(")]
    data = data.dropna(subset=[lhs] + xs)
    groups = [pd.factorize(data[f])[0] for f in fes]
    M = _demean(data[[lhs] + xs].to_numpy(float), groups)
    y, X = M[:, 0], M[:, 1:]
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    cl = pd.factorize(data.run_id)[0]
    G = cl.max() + 1
    S = np.zeros((X.shape[1], X.shape[1]))
    sc = pd.DataFrame(X * e[:, None]).groupby(cl).sum().to_numpy()
    S = sc.T @ sc
    n = len(y)
    k = X.shape[1] + sum(len(np.unique(g)) for g in groups)
    adj = G / (G - 1) * (n - 1) / max(n - k, 1)
    V = adj * XtX_inv @ S @ XtX_inv
    i = xs.index(term)
    se = np.sqrt(V[i, i])
    t = b[i] / se
    p = 2 * st.t.sf(abs(t), G - 1)
    q = st.t.ppf(0.975, G - 1)
    return {"coef": b[i], "ci_low": b[i] - q * se, "ci_high": b[i] + q * se, "p": p, "n": n,
            "n_runs": data.run_id.nunique()}


def rung_specs(dec, role, fam_filter):
    d = dec[dec.role == role]
    d = d[fam_filter(d)]
    specs = {}
    specs["R0 pooled"] = (d, "coop ~ {x}")
    specs["R1 cell+family+round FE"] = (d, "coop ~ {x} + C(cell) + C(family) + C(round)")
    specs["R2 agent+round FE, own lag"] = (d.dropna(subset=["coop_lag"]), "coop ~ {x} + coop_lag + C(run_agent) + C(round)")
    r4 = d[(d.task_order == "myth_game") & (d["round"] == 1)]
    specs["R4 myth->game round 1"] = (r4, "coop ~ {x} + C(family) + C(composition)" +
                                      (" + received_frac" if role == "trustee" else ""))
    r3 = d[(d["size"] == 8) & (d.task_order == "myth_game") & d.shown_author.notna() & ~d.shown_met_before_writing]
    r3 = r3.dropna(subset=["coop_lag", "shown_author_coop_to_reader"])
    specs["R3 8-agent myth->game shown myth (clean)"] = (r3, "coop ~ {x} + coop_lag + shown_author_coop_to_reader + C(run_agent) + C(round)")
    return specs


def main():
    m, dec = load()
    dec.to_csv(HERE / "decision_table.csv", index=False)
    rows = []
    for role, measures in (("investor", INV_MEASURES), ("trustee", TRU_MEASURES)):
        for famset, flt in (("Sonnet+GPT", lambda d: d.family != "Gemini"), ("Sonnet", lambda d: d.family == "Sonnet"),
                            ("GPT", lambda d: d.family == "GPT")):
            for rung, (d, form) in rung_specs(dec, role, flt).items():
                shown = rung.startswith("R3")
                cols = [("shown_" if shown else "") + x for x in measures]
                inter = d.dropna(subset=cols)  # same rows for every measure
                for x, c in zip(measures, cols):
                    try:
                        r = fit(form.format(x=c), inter, c)
                    except Exception as e:  # noqa: BLE001
                        r = {"error": str(e)[:80]}
                    rows.append({"role": role, "families": famset, "rung": rung, "measure": x, **r})
    est = pd.DataFrame(rows)
    # Holm within the main family set, over every test in the table
    from statsmodels.stats.multitest import multipletests
    est["p_holm_all"] = np.nan
    ok = est.p.notna()
    est.loc[ok, "p_holm_all"] = multipletests(est.loc[ok, "p"], method="holm")[1]
    est.round(5).to_csv(HERE / "rung_estimates.csv", index=False)
    print(f"{ok.sum()} tests")

    # within-agent reliability of each measure on the R2 investor/trustee samples (Sonnet+GPT)
    rel = []
    for role in ("investor", "trustee"):
        d = dec[(dec.role == role) & (dec.family != "Gemini")].dropna(subset=["coop_lag"])
        cand = {**PAIRS, "rule_prescribed~give_send_ds": ("rule_prescribed", "give_send_ds"),
                "emb_axis_summary~give_send_ds": ("emb_axis_summary", "give_send_ds"),
                "emb_axis_text~give_send_ds": ("emb_axis_text", "give_send_ds"),
                "label_ord_glm~give_send_ds": ("label_ord_glm", "give_send_ds")}
        for name, (a, b) in cand.items():
            sub = d.dropna(subset=[a, b])
            res = {}
            grp = [pd.factorize(sub.run_agent)[0], pd.factorize(sub["round"])[0]]
            for c in (a, b):
                M = _demean(sub[[c, "coop_lag"]].to_numpy(float), grp)
                beta = np.linalg.lstsq(M[:, 1:], M[:, 0], rcond=None)[0]
                res[c] = M[:, 0] - M[:, 1:] @ beta
            ra, rb = res[a], res[b]
            tot_a = sub[a].var()
            rel.append({"role": role, "pair": name, "n": len(sub),
                        "within_share_of_var_a": ra.var() / tot_a,
                        "corr_within": np.corrcoef(ra, rb)[0, 1],
                        "lambda_a": np.cov(ra, rb)[0, 1] / ra.var(),
                        "corr_raw": sub[a].corr(sub[b]),
                        "agents_constant_a": float((sub.groupby("run_agent")[a].nunique() == 1).mean())})
    rel = pd.DataFrame(rel)
    rel.round(4).to_csv(HERE / "reliability_within.csv", index=False)
    print(rel.round(3).to_string())

    # noise ceiling for R2 (Sonnet+GPT): largest true within-agent effect per SD consistent with the CI
    e = est[(est.families == "Sonnet+GPT") & est.rung.str.startswith("R2")]
    nc = []
    lam_of = {"label_ord_glm": ("label_ord", "lambda_a"), "gen_glm": ("gen", "lambda_a"),
              "give_send_glm": ("give_send", "lambda_a"), "give_return_glm": ("give_return", "lambda_a")}
    for r in e.itertuples():
        if r.measure not in lam_of:
            continue
        pair, col = lam_of[r.measure]
        lam = rel[(rel.role == r.role) & (rel.pair == pair)][col].iloc[0]
        nc.append({"role": r.role, "measure": r.measure, "coef": r.coef, "ci_low": r.ci_low, "ci_high": r.ci_high,
                   "lambda_within": lam, "disattenuated_coef": r.coef / lam,
                   "max_true_effect_per_sd": max(abs(r.ci_low), abs(r.ci_high)) / lam})
    pd.DataFrame(nc).round(4).to_csv(HERE / "noise_ceiling.csv", index=False)
    print(pd.DataFrame(nc).round(4).to_string())


if __name__ == "__main__":
    main()
