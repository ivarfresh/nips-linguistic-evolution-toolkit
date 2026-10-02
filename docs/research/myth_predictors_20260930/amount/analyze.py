#!/usr/bin/env python3
"""H2: do myths that name a higher send amount raise cooperation?

Outputs (this folder):
  regex_validation.csv            regex vs judge agreement
  tests.csv                       every regression, rung, coef, CI, p, n, Holm-adjusted p within its family of tests
  anchoring_by_run.csv            per-run mean of |send - shown| - |send - unseen|
  run_level_R1.csv                run-level correlations within composition x task order
  cv_incremental.csv              held-out gain from adding the own myth's amount to the lagged send
  transplant_regex.csv            R5 donor amounts (judge + regex) vs host sends
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from common import OUT, regex_amount, MIDPOINT

warnings.filterwarnings("ignore")
P = pd.read_pickle(OUT / "panel.pkl")
P["runround"] = P["run_id"] + "|" + P["round"].astype(str)
P["agentrun"] = P["run_id"] + "|" + P["agent"]
P["cell"] = P["composition"] + "|" + P["task_order"]
P["send"] = P["sent"]  # $ out of 5
FAMS = ["Sonnet", "GPT", "Gemini"]
ROWS = []


def _demean(X: np.ndarray, groups: list, iters: int = 200, tol: float = 1e-10) -> np.ndarray:
    """Absorb one or more fixed effects by alternating projections."""
    X = X.copy()
    codes = [pd.factorize(g)[0] for g in groups]
    for _ in range(iters if len(codes) > 1 else 1):
        prev = X.copy()
        for c in codes:
            sums = np.zeros((c.max() + 1, X.shape[1]))
            np.add.at(sums, c, X)
            cnt = np.bincount(c).reshape(-1, 1)
            X = X - (sums / cnt)[c]
        if np.abs(X - prev).max() < tol:
            break
    return X


def fit(df, formula, term, rung, test, subset, fam="all", measure="judge_amount", extra=None, cluster="run_id", min_df=30):
    """OLS, fixed effects C(x) absorbed by demeaning (categorical non-FE terms like C(seed_type) too),
    SE clustered by run. Degrees of freedom are not corrected for absorbed FE (SEs slightly small)."""
    lhs, rhs = [t.strip() for t in formula.split("~")]
    terms = [t.strip() for t in rhs.split("+")]
    fes = [re.match(r"C\((\w+)\)", t).group(1) for t in terms if t.startswith("C(")]
    xs = [t for t in terms if not t.startswith("C(")]
    df = df.dropna(subset=[lhs] + xs + fes)
    n_runs = df[cluster].nunique() if len(df) else 0
    row = dict(rung=rung, test=test, subset=subset, family=fam, measure=measure, term=term,
               n=len(df), n_runs=n_runs, coef=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan,
               sd_x=np.nan, fe="+".join(fes))
    if len(df) < 15 or n_runs < 4 or df[term].std() == 0 or df[term].nunique() < 2:
        row["note"] = "too little variation"
        ROWS.append(row)
        return row
    n_absorbed = sum(df[f].nunique() for f in fes)
    row["resid_df"] = len(df) - n_absorbed - len(xs)
    if row["resid_df"] < min_df:
        row["note"] = "degenerate: too few residual df after absorbing fixed effects"
        ROWS.append(row)
        return row
    A = df[[lhs] + xs].to_numpy(float)
    if fes:
        A = _demean(A, [df[f].to_numpy() for f in fes])
    else:
        A = A - A.mean(0)
    y, X = A[:, 0], A[:, 1:]
    j = xs.index(term)
    if X[:, j].std() < 1e-9 or y.std() < 1e-9:
        row["note"] = "no within-FE variation (outcome or predictor constant, e.g. Gemini at the ceiling)"
        ROWS.append(row)
        return row
    try:
        import statsmodels.api as sm
        r = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df[cluster])[0]})
        ci = r.conf_int()[j]
        row.update(coef=r.params[j], ci_low=ci[0], ci_high=ci[1], p=r.pvalues[j], sd_x=df[term].std(),
                   sd_x_within=X[:, j].std())
    except Exception as e:
        row["note"] = f"fit failed: {type(e).__name__}"
    if extra:
        row.update(extra)
    ROWS.append(row)
    return row


def fam_loop(df, formula, term, rung, test, subset, measure="judge_amount", fams=("all", "Sonnet", "GPT", "Gemini")):
    for f in fams:
        sub = df if f == "all" else df[df["family"] == f]
        fit(sub, formula, term, rung, test, subset, fam=f, measure=measure)


inv = P[P["role"] == "investor"].copy()
tru = P[P["role"] == "trustee"].copy()
MEASURES = ["judge_amount", "judge_amount_strict", "rx_amount"]

# ================================================================ R3: 8-agent myth_game, shown vs unseen
r3 = inv[(inv["size"] == 8) & (inv["task_order"] == "myth_game") & inv["shown_judge_amount"].notna()].copy()
for meas in MEASURES:
    for x in ("shown", "unseen"):
        r3[f"{x}_amt"] = r3[f"{x}_{meas}"]
    r3["own_prev_amt"] = r3[f"own_prev_{meas}"]
    r3["own_amt"] = r3[f"own_{meas}"]
    base = "send ~ shown_amt + unseen_amt + lag_sent + own_prev_amt + b_coop_prev + c_coop_prev + C(runround)"
    fam_loop(r3, base, "shown_amt", "R3", "shown amount -> next send (run x round FE; unseen myth as placebo)",
             "8-agent myth_game", meas)
    fam_loop(r3, base, "unseen_amt", "R3-placebo", "UNSEEN comparison amount -> next send (placebo)",
             "8-agent myth_game", meas)
    if meas == "judge_amount":
        # with agent FE too
        fam_loop(r3, base + " + C(agentrun)", "shown_amt", "R3", "shown amount -> next send (+ agent-within-run FE)",
                 "8-agent myth_game", meas)
        # mediator: add own round-r myth (written after reading the shown myth)
        fam_loop(r3, base + " + own_amt", "shown_amt", "R3", "shown amount -> next send, own new myth held fixed",
                 "8-agent myth_game", meas)
        fam_loop(r3, base + " + own_amt", "own_amt", "R3*", "own new myth amount -> send (written after exposure; not clean)",
                 "8-agent myth_game", meas)
        # clean: shown author had never played the reader before writing
        nc = r3[r3["contact"] == 0]
        fam_loop(nc, base, "shown_amt", "R3", "shown amount -> next send, no prior contact",
                 "8-agent myth_game, no prior contact", meas)
        # founding window: round-2 decisions, shown myth written before any play
        r2 = r3[r3["round"] == 2]
        fam_loop(r2, "send ~ shown_amt + unseen_amt + lag_coop_any + own_prev_amt + b_coop_prev + C(runround)",
                 "shown_amt", "R3/R4", "round 2: shown round-1 myth (written before any play) -> round-2 send",
                 "8-agent myth_game round 2", meas)
        # mixed-run stratum and homogeneous stratum
        for mixed, lab in ((False, "homogeneous"), (True, "mixed")):
            fam_loop(r3[r3["mixed"] == mixed], base, "shown_amt", "R3", f"shown amount -> next send ({lab} only)",
                     f"8-agent myth_game {lab}", meas)
        # cross-family vs same-family exposure
        r3["same_fam"] = (r3["shown_family"] == r3["family"])
        for sf, lab in ((True, "same family"), (False, "other family")):
            fam_loop(r3[r3["same_fam"] == sf], base, "shown_amt", "R3", f"shown amount -> next send, shown {lab}",
                     f"8-agent myth_game, shown {lab}", meas)
        # game_myth, same design, lower rung (shown myth describes a game its author shared with the reader)
        gm = inv[(inv["size"] == 8) & (inv["task_order"] == "game_myth") & inv["shown_judge_amount"].notna()].copy()
        gm["shown_amt"], gm["unseen_amt"] = gm["shown_judge_amount"], gm["unseen_judge_amount"]
        gm["own_prev_amt"], gm["own_amt"] = gm["own_prev_judge_amount"], gm["own_judge_amount"]
        fam_loop(gm, base, "shown_amt", "R2+", "shown amount -> next send (game_myth: shown myth follows a shared game)",
                 "8-agent game_myth", meas)
        # beyond generic generosity: add shown myth's generous label
        fam_loop(r3, base + " + shown_generous + unseen_generous", "shown_amt", "R3",
                 "shown amount -> next send, shown 'be generous' label held fixed", "8-agent myth_game", meas)
        fam_loop(r3, base + " + shown_generous + unseen_generous", "shown_generous", "R3",
                 "shown 'be generous' label -> next send, amount held fixed", "8-agent myth_game", "generous")

    # ---- myth side: does the reader's new myth adopt the shown amount?
    mm = r3.copy()
    fam_loop(mm, "own_amt ~ shown_amt + unseen_amt + own_prev_amt + lag_sent + b_coop_prev + C(runround)",
             "shown_amt", "R3", "shown amount -> reader's next MYTH amount", "8-agent myth_game", meas)
    fam_loop(mm, "own_amt ~ shown_amt + unseen_amt + own_prev_amt + lag_sent + b_coop_prev + C(runround)",
             "unseen_amt", "R3-placebo", "UNSEEN amount -> reader's next MYTH amount (placebo)", "8-agent myth_game", meas)

# ---- in-window exposure: the game call's context holds the last 3 shown myths (rounds r-2..r in myth_game)
own = P.drop_duplicates(["run_id", "agent", "own_round"])[["run_id", "agent", "own_round", "shown_judge_amount",
                                                           "unseen_judge_amount", "shown_rx_amount", "unseen_rx_amount"]]
own = own.set_index(["run_id", "agent", "own_round"])
def window(row, col):
    vals = [own[col].get((row.run_id, row.agent, row.round - k), np.nan) for k in (0, 1, 2)]
    vals = [v for v in vals if pd.notna(v)]
    return np.mean(vals) if vals else np.nan
for meas in ("judge_amount", "rx_amount"):
    r3["shown_win"] = [window(r, f"shown_{meas}") for r in r3.itertuples()]
    r3["unseen_win"] = [window(r, f"unseen_{meas}") for r in r3.itertuples()]
    r3["own_prev_amt"] = r3[f"own_prev_{meas}"]
    wf = "send ~ shown_win + unseen_win + lag_sent + own_prev_amt + b_coop_prev + c_coop_prev + C(runround)"
    fam_loop(r3, wf, "shown_win", "R3", "mean of the <=3 in-window shown myths -> next send", "8-agent myth_game", meas)
    fam_loop(r3, wf, "unseen_win", "R3-placebo", "mean of in-window UNSEEN comparisons -> next send (placebo)",
             "8-agent myth_game", meas)
    fam_loop(r3[r3["round"] >= 4], wf, "shown_win", "R3", "mean of the 3 in-window shown myths -> next send, rounds 4-10",
             "8-agent myth_game rounds 4-10", meas)
r3.drop(columns=["shown_win", "unseen_win"]).head(0)
r3[["run_id", "agent", "round", "family", "send", "shown_judge_amount", "unseen_judge_amount"]].to_csv(OUT / "r3_rows.csv", index=False)

# myth side on ALL 8-agent myths (not just investor rows): each myth once
M = pd.read_pickle(OUT / "myths_features.pkl")
mt = P.drop_duplicates(["run_id", "agent", "own_round"])  # one row per own myth
mt = mt[(mt["size"] == 8) & (mt["task_order"] == "myth_game") & mt["shown_judge_amount"].notna()].copy()
mt["own_amt"], mt["shown_amt"], mt["unseen_amt"], mt["own_prev_amt"] = (
    mt["own_judge_amount"], mt["shown_judge_amount"], mt["unseen_judge_amount"], mt["own_prev_judge_amount"])
fam_loop(mt, "own_amt ~ shown_amt + unseen_amt + own_prev_amt + lag_coop_any + b_coop_prev + C(runround)",
         "shown_amt", "R3", "shown amount -> reader's next MYTH amount (all myths)", "8-agent myth_game")
fam_loop(mt, "own_amt ~ shown_amt + unseen_amt + own_prev_amt + lag_coop_any + b_coop_prev + C(runround)",
         "unseen_amt", "R3-placebo", "UNSEEN amount -> reader's next MYTH amount (all myths, placebo)", "8-agent myth_game")
mt["shown_minus_unseen"] = mt["shown_amt"] - mt["unseen_amt"]

# ================================================================ anchoring (distance to anchor)
anch = []
a = r3.dropna(subset=["shown_judge_amount", "unseen_judge_amount", "lag_sent"]).copy()
a["d_shown"] = (a["send"] - a["shown_judge_amount"]).abs()
cand = {k: g for k, g in M.groupby(["run_id", "round"])}
def d_unseen(row):
    g = cand[(row.run_id, row.shown_round)]
    c = g[(g["family"] == row.shown_family) & (~g["agent"].isin([row.agent, row.shown_author]))]["judge_amount"].dropna()
    return (row.send - c).abs().mean() if len(c) else np.nan
a["d_unseen"] = [d_unseen(r) for r in a.itertuples()]
a["closer"] = a["d_shown"] - a["d_unseen"]
# move toward: change in send projected on direction of anchor from the lagged send
a["gap_shown"] = a["shown_judge_amount"] - a["lag_sent"]
a["gap_unseen"] = a["unseen_judge_amount"] - a["lag_sent"]
a["change"] = a["send"] - a["lag_sent"]
a["disagree"] = (a["shown_judge_amount"] - a["unseen_judge_amount"]).abs() >= 0.5
for f in ["all", "Sonnet", "GPT", "Gemini"]:
    s = a if f == "all" else a[a["family"] == f]
    for only_dis in (False, True):
        s2 = s[s["disagree"]] if only_dis else s
        per_run = s2.groupby("run_id")["closer"].mean()
        if len(per_run) >= 5:
            w = stats.wilcoxon(per_run.round(10))
            anch.append(dict(family=f, only_when_shown_differs=only_dis, mean=per_run.mean(), sd=per_run.std(),
                             n_runs=len(per_run), n=len(s2), wilcoxon_p=w.pvalue, runs_closer_to_shown=(per_run < 0).sum()))
pd.DataFrame(anch).to_csv(OUT / "anchoring_by_run.csv", index=False)

# ================================================================ return side (trustees)
t3 = tru[(tru["size"] == 8) & (tru["task_order"] == "myth_game") & tru["shown_judge_return"].notna()].copy()
t3["shown_ret"], t3["unseen_ret"], t3["own_prev_ret"] = t3["shown_judge_return"], t3["unseen_judge_return"], t3["own_prev_judge_return"]
t3["lag_ret"] = t3["lag_coop_same_role"]
fam_loop(t3, "return_proportion ~ shown_ret + unseen_ret + lag_ret + own_prev_ret + received + b_coop_prev + C(runround)",
         "shown_ret", "R3", "shown return rule -> reader's next return proportion", "8-agent myth_game", "judge_return")
fam_loop(t3, "return_proportion ~ shown_ret + unseen_ret + lag_ret + own_prev_ret + received + b_coop_prev + C(runround)",
         "unseen_ret", "R3-placebo", "UNSEEN return rule -> next return proportion (placebo)", "8-agent myth_game", "judge_return")
# own return rule within agent (R2)
t2 = tru.dropna(subset=["own_judge_return", "lag_coop_same_role"]).copy()
fam_loop(t2, "return_proportion ~ own_judge_return + lag_coop_same_role + received + C(agentrun) + C(round)",
         "own_judge_return", "R2", "own latest return rule -> return proportion (agent FE)", "all settings", "judge_return")

# ================================================================ R4: round-1 myth_game, own myth before any play
r4 = inv[(inv["task_order"] == "myth_game") & (inv["round"] == 1)].copy()
for meas in MEASURES:
    r4["own_amt"] = r4[f"own_{meas}"]
    fam_loop(r4, "send ~ own_amt + C(composition)", "own_amt", "R4",
             "round-1 own myth amount -> round-1 send (composition FE)", "myth_game round 1, 2+8 agent", meas)
    for sz in (2, 8):
        fam_loop(r4[r4["size"] == sz], "send ~ own_amt + C(composition)", "own_amt", "R4",
                 f"round-1 own myth amount -> round-1 send ({sz}-agent)", f"myth_game round 1, {sz}-agent", meas,
                 fams=("all", "Sonnet", "GPT"))
# concrete vs abstract: does naming a concrete number matter beyond the band?
r4["has_number"] = r4["rx_concrete"].notna().astype(float) if "rx_concrete" in r4 else np.nan
r4 = r4.merge(M[["run_id", "round", "agent", "rx_concrete", "send_rule"]], left_on=["run_id", "own_round", "agent"],
              right_on=["run_id", "round", "agent"], how="left", suffixes=("", "_m"))
r4["band"] = r4["send_rule"].map(MIDPOINT)
r4["rx_c"] = r4["rx_concrete"]
fam_loop(r4.dropna(subset=["band", "rx_c"]), "send ~ rx_c + band + C(composition)", "rx_c", "R4",
         "round-1: concrete regex number beyond the judge's band", "myth_game round 1 with a number", "rx_concrete",
         fams=("all", "Sonnet", "GPT"))

# ================================================================ R2: within agent, all settings
r2 = inv[inv["round"] >= 2].copy()
for meas in ("judge_amount", "rx_amount"):
    r2["own_amt"], r2["shown_amt"] = r2[f"own_{meas}"], r2[f"shown_{meas}"]
    for setting in sorted(r2["setting"].unique()):
        for order in ("game_myth", "myth_game"):
            s = r2[(r2["setting"] == setting) & (r2["task_order"] == order)]
            fam_loop(s, "send ~ own_amt + shown_amt + lag_sent + C(agentrun) + C(round)", "own_amt", "R2",
                     "own latest myth amount -> send (agent-within-run + round FE)", f"{setting} {order}", meas,
                     fams=("Sonnet", "GPT"))
            fam_loop(s, "send ~ own_amt + shown_amt + lag_sent + C(agentrun) + C(round)", "shown_amt", "R2",
                     "latest shown myth amount -> send (agent-within-run + round FE)", f"{setting} {order}", meas,
                     fams=("Sonnet", "GPT"))
    fam_loop(r2, "send ~ own_amt + shown_amt + lag_sent + C(agentrun) + C(round)", "own_amt", "R2",
             "own latest myth amount -> send (agent FE), all settings", "all settings", meas)
    fam_loop(r2, "send ~ own_amt + shown_amt + lag_sent + C(agentrun) + C(round)", "shown_amt", "R2",
             "latest shown myth amount -> send (agent FE), all settings", "all settings", meas)
    # reverse: send -> next own myth amount
    rev = P[P["role"] == "investor"].copy()
    rev = rev.sort_values(["run_id", "agent", "round"])
rv = M.copy()
d_inv = P[P["role"] == "investor"][["run_id", "agent", "round", "send", "task_order"]]
# reverse direction: the send in round t -> the myth written next (game_myth: same round t; myth_game: round t+1)
d_inv = d_inv.assign(myth_round=np.where(d_inv["task_order"] == "game_myth", d_inv["round"], d_inv["round"] + 1))
rv = rv.merge(d_inv.drop(columns="task_order").rename(columns={"round": "send_round"}),
              left_on=["run_id", "agent", "round"], right_on=["run_id", "agent", "myth_round"], how="inner")
rv = rv.sort_values(["run_id", "agent", "round"])
rv["prev_amt"] = rv.groupby(["run_id", "agent"])["judge_amount"].shift()
rv["agentrun"] = rv["run_id"] + "|" + rv["agent"]
fam_loop(rv, "judge_amount ~ send + prev_amt + C(agentrun) + C(round)", "send", "R2-reverse",
         "own send -> next own myth amount (agent FE)", "all settings", "judge_amount")

# ================================================================ R1: run-level within composition x task order
r1rows = []
runlev = inv[inv["shown_judge_amount"].notna()].groupby(["run_id", "cell", "setting", "task_order"]).agg(
    send=("send", "mean"), shown=("shown_judge_amount", "mean"), own=("own_judge_amount", "mean")).reset_index()
for col in ("send", "shown", "own"):
    runlev[col + "_c"] = runlev[col] - runlev.groupby("cell")[col].transform("mean")
for (setting, order), g in runlev.groupby(["setting", "task_order"]):
    for x in ("shown", "own"):
        rho = stats.spearmanr(g[x + "_c"], g["send_c"], nan_policy="omit")
        r1rows.append(dict(setting=setting, task_order=order, predictor=f"run-mean {x} myth amount", n_runs=len(g),
                           spearman=rho.statistic, p=rho.pvalue))
    for x in ("shown", "own"):
        rho0 = stats.spearmanr(g[x], g["send"], nan_policy="omit")
        r1rows.append(dict(setting=setting, task_order=order, predictor=f"run-mean {x} myth amount (R0 pooled, uncentred)",
                           n_runs=len(g), spearman=rho0.statistic, p=rho0.pvalue))
pd.DataFrame(r1rows).to_csv(OUT / "run_level_R1.csv", index=False)

# ================================================================ own-myth incremental held-out gain
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression

cvrows = []
def cv_gain(df, base_cols, add_cols, label, reps=20):
    df = df.dropna(subset=base_cols + add_cols + ["send"]).reset_index(drop=True)
    X0 = pd.get_dummies(df[base_cols], drop_first=True, dtype=float)
    X1 = pd.get_dummies(df[base_cols + add_cols], drop_first=True, dtype=float)
    y = df["send"].to_numpy()
    runs = df["run_id"].to_numpy()
    uniq = np.unique(runs)
    gains, r2b, r2a = [], [], []
    rng = np.random.default_rng(0)
    for rep in range(reps):
        perm = {u: i for i, u in enumerate(rng.permutation(uniq))}
        grp_ids = np.array([perm[r] for r in runs])
        pred0, pred1 = np.zeros_like(y), np.zeros_like(y)
        for tr, te in GroupKFold(5).split(X0, y, grp_ids):
            pred0[te] = LinearRegression().fit(X0.iloc[tr], y[tr]).predict(X0.iloc[te])
            pred1[te] = LinearRegression().fit(X1.iloc[tr], y[tr]).predict(X1.iloc[te])
        ss = ((y - y.mean()) ** 2).sum()
        a0, a1 = 1 - ((y - pred0) ** 2).sum() / ss, 1 - ((y - pred1) ** 2).sum() / ss
        r2b.append(a0); r2a.append(a1); gains.append(a1 - a0)
    # run bootstrap for the gain's CI on one fixed split
    cvrows.append(dict(analysis=label, n=len(df), n_runs=len(uniq), base=" + ".join(base_cols), added=" + ".join(add_cols),
                       heldout_r2_base=np.mean(r2b), heldout_r2_with_myth=np.mean(r2a),
                       gain_mean=np.mean(gains), gain_min=np.min(gains), gain_max=np.max(gains)))

later = inv[(inv["round"] >= 2)].copy()
later["round_c"] = later["round"].astype(str)
for fam in ("all", "Sonnet", "GPT"):
    s = later if fam == "all" else later[later["family"] == fam]
    cv_gain(s, ["lag_sent", "family", "setting", "task_order", "round_c"], ["own_judge_amount"],
            f"rounds 2-10, {fam}: + own myth amount")
    cv_gain(s, ["lag_sent", "family", "setting", "task_order", "round_c"], ["own_rx_amount"],
            f"rounds 2-10, {fam}: + own myth amount (regex)")
    cv_gain(s, ["family", "setting", "task_order", "round_c"], ["own_judge_amount"],
            f"rounds 2-10, {fam}: own myth amount WITHOUT lagged send")
    cv_gain(s, ["own_judge_amount", "family", "setting", "task_order", "round_c"], ["lag_sent"],
            f"rounds 2-10, {fam}: + lagged send on top of own myth amount")
    cv_gain(s, ["lag_sent", "family", "setting", "task_order", "round_c"], ["shown_judge_amount"],
            f"rounds 2-10, {fam}: + shown myth amount")
    s8 = s[s["unseen_judge_amount"].notna()]
    cv_gain(s8, ["lag_sent", "family", "setting", "task_order", "round_c"], ["shown_judge_amount"],
            f"8-agent rounds 2-10, {fam}: + shown myth amount")
    cv_gain(s8, ["lag_sent", "family", "setting", "task_order", "round_c"], ["unseen_judge_amount"],
            f"8-agent rounds 2-10, {fam}: + UNSEEN comparison amount (placebo)")
r1 = inv[(inv["task_order"] == "myth_game") & (inv["round"] == 1)].copy()
for fam in ("all", "Sonnet", "GPT"):
    s = r1 if fam == "all" else r1[r1["family"] == fam]
    cv_gain(s, ["family", "composition"], ["own_judge_amount"], f"round 1 myth_game, {fam}: + own myth amount")
pd.DataFrame(cvrows).to_csv(OUT / "cv_incremental.csv", index=False)

# ================================================================ R5: transplant donors, regex check
TR = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/myth-rules/docs/figures/myth_rules_20260928/transplant_donor_rules.csv")
TOOL = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-linguistic-evolution-toolkit")
don = pd.read_csv(TR)
texts = []
for size, sub in (("8", "slide678_rerun_20260916"), ("2", "slide678_dyad_rerun_20260917")):
    pl = TOOL / "data/json/noise_experiments" / sub / "plan.json"
    if pl.exists():
        for c in json.loads(pl.read_text())["combos"]:
            if c.get("seed_text"):
                texts.append(dict(size=int(size), seed_type=c["seed_type"], rep=int(c["rep"]), text=c["seed_text"]))
if texts:
    tx = pd.DataFrame(texts)
    tx = pd.concat([tx, pd.DataFrame([regex_amount(t) for t in tx["text"]])], axis=1)
    don = don.merge(tx.drop(columns="text"), on=["size", "seed_type", "rep"], how="left")
don.to_csv(OUT / "transplant_regex.csv", index=False)
for size in (8, 2):
    s = don[don["size"] == size]
    for meas in ("prescribed", "rx_amount"):
        if meas not in s:
            continue
        q = s.dropna(subset=[meas, "host_send_mean"])
        fit(q.assign(amt=q[meas], row_id=np.arange(len(q))), "host_send_mean ~ amt + C(seed_type)", "amt", "R5",
            "donor text amount -> host mean send (donor-type FE, HC robust SE)", f"transplant {size}-agent",
            measure=meas, cluster="row_id", min_df=10)
        q2 = q[~((q["seed_type"] == "s_end_minus") & (q["rep"] == 1))]  # the "send nothing" donor (prescribed 0)
        fit(q2.assign(amt=q2[meas], row_id=np.arange(len(q2))), "host_send_mean ~ amt + C(seed_type)", "amt", "R5",
            "donor text amount -> host mean send, without the 'send nothing' donor", f"transplant {size}-agent",
            measure=meas, cluster="row_id", min_df=10)
        rho = stats.spearmanr(q[meas], q["host_send_mean"])
        ROWS.append(dict(rung="R5", test="donor amount vs host mean send, Spearman", subset=f"transplant {size}-agent",
                         family="Sonnet hosts", measure=meas, term="spearman", n=len(q), n_runs=len(q),
                         coef=rho.statistic, p=rho.pvalue))

# ================================================================ regex validation
vrows = []
b = M.dropna(subset=["rx_amount", "judge_amount"])
vrows.append(dict(comparison="regex amount vs judge amount (named else band), Spearman", n=len(b),
                  value=stats.spearmanr(b.rx_amount, b.judge_amount).statistic))
vrows.append(dict(comparison="regex amount within 0.5 of judge amount", n=len(b),
                  value=((b.rx_amount - b.judge_amount).abs() <= 0.5).mean()))
c = M.dropna(subset=["rx_concrete", "send_amount"])
vrows.append(dict(comparison="regex concrete number within 0.5 of judge's named amount", n=len(c),
                  value=((c.rx_concrete - c.send_amount).abs() <= 0.5).mean()))
vrows.append(dict(comparison="regex concrete vs judge named amount, Spearman", n=len(c),
                  value=stats.spearmanr(c.rx_concrete, c.send_amount).statistic))
vrows.append(dict(comparison="share of myths with a regex concrete number", n=len(M), value=M.rx_concrete.notna().mean()))
vrows.append(dict(comparison="share of myths with any regex amount", n=len(M), value=M.rx_amount.notna().mean()))
vrows.append(dict(comparison="share of myths with a judge named amount", n=len(M), value=M.send_amount.notna().mean()))
e = M[M["amount_status"].notna()]
for st in ("endorsed", "narrated"):
    s = e[e["amount_status"] == st].dropna(subset=["rx_concrete"])
    vrows.append(dict(comparison=f"regex concrete within 0.5 of named amount, judge-{st} myths", n=len(s),
                      value=((s.rx_concrete - s.send_amount).abs() <= 0.5).mean()))
for fam in FAMS:
    s = b[b.family == fam]
    vrows.append(dict(comparison=f"regex vs judge amount Spearman, {fam}", n=len(s),
                      value=stats.spearmanr(s.rx_amount, s.judge_amount).statistic))
pd.DataFrame(vrows).to_csv(OUT / "regex_validation.csv", index=False)

# ================================================================ Holm within rung
T = pd.DataFrame(ROWS)
T["few_clusters"] = T["n_runs"] < 10
T["p_holm_within_rung"] = np.nan
for rung, g in T.groupby("rung"):
    ok = g["p"].notna()
    if ok.sum():
        T.loc[g.index[ok], "p_holm_within_rung"] = multipletests(g.loc[ok, "p"], method="holm")[1]
ok = T["p"].notna()
T.loc[ok, "p_holm_all"] = multipletests(T.loc[ok, "p"], method="holm")[1]
PRIMARY = [
    ("R3", "shown amount -> next send (run x round FE; unseen myth as placebo)", "judge_amount"),
    ("R3", "mean of the <=3 in-window shown myths -> next send", "judge_amount"),
    ("R3", "shown amount -> reader's next MYTH amount (all myths)", "judge_amount"),
    ("R4", "round-1 own myth amount -> round-1 send (composition FE)", "judge_amount"),
    ("R2", "own latest myth amount -> send (agent FE), all settings", "judge_amount"),
    ("R2", "latest shown myth amount -> send (agent FE), all settings", "judge_amount"),
    ("R3", "shown return rule -> reader's next return proportion", "judge_return"),
]
T["primary"] = False
for rung, test, meas in PRIMARY:
    T.loc[(T.rung == rung) & (T.test == test) & (T.measure == meas) & T.family.isin(["all", "Sonnet", "GPT"]), "primary"] = True
T.loc[(T.rung == "R5") & (T.measure == "prescribed") & (T.term == "amt"), "primary"] = True
ok = T["primary"] & T["p"].notna()
T.loc[ok, "p_holm_primary"] = multipletests(T.loc[ok, "p"], method="holm")[1]
print(ok.sum(), "primary tests")
T.to_csv(OUT / "tests.csv", index=False)
print(f"{len(T)} rows, {ok.sum()} tests with p")
