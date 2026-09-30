#!/usr/bin/env python3
"""Part 2 (port of myth_predictors_20260930/consistency/predict.py; LINGUISTIC_DATASET=september|frontier):
does consistency language predict cooperation LEVEL or STABILITY?

Decision pairing follows moral_carryover.decision_table: the agent's own myth written
before the decision (myth_game: same round; game_myth: previous round), the myth it was
shown before writing that myth, and the co-player's myth from the same round (placebo).

Outcomes per decision:
  coop       investor sent/5, trustee return proportion
  absdelta   |coop - own previous coop in the same role|
  zero       investor sent 0
  cut_after_letdown  investor sends less than last time, only after a letdown (last time as investor
             got back less than a third, true amounts)
Rungs: R1 (cell FE + round FE), R2 (agent-within-run FE + round FE + own lagged coop),
R3 (8-agent myth_game, shown author != current partner, control for that author's last move toward
the reader; placebo = an unseen same-family myth from the same run and round), R4 (myth_game round-1
myth -> round-1 move and rounds 2-10 stability, per agent).

Changes from September: families from common.FAMILIES; each table is run twice, once pooled over
settings per family (the September layout, stratum 'all settings') and once per frontier stratum
(setting x family x task order; R3/R4 per setting x family). Every row carries the diagnostics the
scorecard uses to label cells (outcome sd, modal share, feature mean, agents whose feature varies)
and a Holm p within its stratum. R4 adds senders-only round-1 sends (the settled R4 spec).
Outputs: <dataset>/predict_*.csv, <dataset>/predict_r4_agents.csv, <dataset>/decision_table.csv
"""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from common import CEILING_SEND, DATA, FAMILIES, NAME, OUT, setting_of
from fe import fe_ols, rows_from

FEATS = ["cons_judge", "cons_lex", "cons_emb"]
ZSET = ["cons_judge", "cons_lex", "cons_emb_z"]


def load():
    m = pd.read_csv(OUT / "myth_features.csv")
    m.loc[~m["valid"], FEATS] = np.nan
    d = pd.read_csv(DATA / "decisions.csv").sort_values(["run_id", "agent", "round"]).reset_index(drop=True)
    d["setting"] = setting_of(d)
    d["cell"] = d["composition"] + "|" + d["task_order"]
    d["run_agent"] = d["run_id"] + "|" + d["agent"]
    key = {(r, t, a): i for i, (r, t, a) in enumerate(zip(m.run_id, m["round"], m.agent))}
    fv = m[FEATS].to_numpy()
    fam = m["family"].to_numpy()
    d["r_before"] = np.where(d["task_order"] == "myth_game", d["round"], d["round"] - 1)
    own_i, shown_i, cop_i = [], [], []
    for g in d.itertuples(index=False):
        i = key.get((g.run_id, g.r_before, g.agent))
        own_i.append(i)
        if i is not None and isinstance(m.at[i, "exposed_author"], str):
            shown_i.append(key.get((g.run_id, int(m.at[i, "exposed_round"]), m.at[i, "exposed_author"])))
        else:
            shown_i.append(None)
        cop_i.append(key.get((g.run_id, g.r_before, g.partner)))
    for name, idx in [("own", own_i), ("shown", shown_i), ("cop", cop_i)]:
        arr = np.array([fv[i] if i is not None else [np.nan] * len(FEATS) for i in idx])
        for j, f in enumerate(FEATS):
            d[f"{name}_{f}"] = arr[:, j]
    d["shown_idx"] = shown_i
    d["shown_family"] = [fam[i] if i is not None else None for i in shown_i]
    d["shown_author"] = [m.at[i, "agent"] if i is not None else None for i in shown_i]
    grp = d.groupby(["run_id", "agent", "role"])
    d["coop_lag"] = grp["coop"].shift(1)
    d["absdelta"] = (d["coop"] - d["coop_lag"]).abs()
    inv = d["role"] == "investor"
    d["zero"] = np.where(inv, (d["sent"] == 0).astype(float), np.nan)
    prev_rp = grp["return_proportion"].shift(1)
    prev_sent = grp["sent"].shift(1)
    letdown = inv & (prev_rp < 1 / 3) & (prev_sent > 0)
    d["cut_after_letdown"] = np.where(letdown, (d["sent"] < prev_sent).astype(float), np.nan)
    d["prev_zero"] = np.where(inv, (prev_sent == 0).astype(float), np.nan)
    ga = d.groupby(["run_id", "agent"])
    lr, lrp, ls = ga["role"].shift(1), ga["return_proportion"].shift(1), ga["sent"].shift(1)
    d["partner_prev_move"] = np.where(lr == "investor", lrp, np.where(lr == "trustee", ls / 5, np.nan))
    d.loc[~inv | prev_sent.isna(), "prev_zero"] = np.nan
    return m, d


def standardise(d):
    """cons_emb in sd units (pooled own_cons_emb sd) so coefficients are comparable; binary features stay 0/1."""
    sd = d["own_cons_emb"].std()
    for p in ["own", "shown", "cop", "null"]:
        c = f"{p}_cons_emb"
        if c in d:
            d[c + "_z"] = d[c] / sd
    return d


def diag(g, y, x, absorb=None):
    """Cell diagnostics for the scorecard labels."""
    s = g.dropna(subset=[y, x])
    out = {"y_mean": s[y].mean(), "y_sd": s[y].std(), "y_modal_share": s[y].round(4).value_counts(normalize=True).max() if len(s) else np.nan,
           "x_mean": s[x].mean(), "x_sd": s[x].std()}
    if absorb:
        out["x_varying_units"] = int((s.groupby(absorb)[x].nunique() > 1).sum())
    return out


def fit(rows, g, y, xs, absorb, dummies, report, **meta):
    try:
        res = fe_ols(g, y, xs, absorb=absorb, dummies=dummies)
    except Exception:
        return
    dg = diag(g, y, report[0], absorb)
    rows += [{**r, **dg} for r in rows_from(res, report, **meta)]


def strata(d, by_task_order=True):
    """(stratum, family, task_order, frame): pooled over settings per family (September layout) + frontier strata."""
    out = []
    for fam in FAMILIES + ["all"]:
        g = d if fam == "all" else d[d["family"] == fam]
        out.append(("all settings", fam, "both", g))
    for (st, fam), g in d.groupby(["setting", "family"]):
        out.append((st, fam, "both", g))
        if by_task_order:
            for to, h in g.groupby("task_order"):
                out.append((st, fam, to, h))
    return out


def r1_r2(d):
    rows = []
    outcomes = [("coop", "investor"), ("coop", "trustee"), ("absdelta", "investor"), ("absdelta", "trustee"),
                ("zero", "investor"), ("cut_after_letdown", "investor")]
    for st, fam, to, g0 in strata(d):
        for y, role in outcomes:
            if fam in CEILING_SEND and role == "investor":
                continue  # sends $5 almost always: nothing to predict
            g = g0[g0["role"] == role]
            meta = dict(stratum=st, family=fam, task_order=to, role=role, outcome=y)
            for f in ZSET:
                x = f"own_{f}"
                fit(rows, g, y, [x], "cell", ["round"], [x], rung="R1", feature=f, **meta)
                xs = [x] + ([] if y == "cut_after_letdown" else ["coop_lag"])
                fit(rows, g, y, xs, "run_agent", ["round"], [x], rung="R2", feature=f, **meta)
    return pd.DataFrame(rows)


def lockin(d):
    """(a) persistence: coop ~ lag x consistency (R1); (c) |change| ~ consistency split by previous level (R2);
    (b) zero-lock: after sending 0, does a consistency myth make a second 0 more likely? (the family with most
    repeat-zero opportunities; September GPT)."""
    rows = []
    d = d.copy()
    zfam = d[d["prev_zero"] == 1].groupby("family").size()
    for f in ZSET:
        x = f"own_{f}"
        d["lagXcons"] = d["coop_lag"] * d[x]
        for fam in [f_ for f_ in FAMILIES if f_ not in CEILING_SEND] + ["all"]:
            for role in ["investor", "trustee"]:
                g = d[(d["role"] == role) & ((d["family"] == fam) | (fam == "all"))]
                fit(rows, g, "coop", ["coop_lag", x, "lagXcons"], "cell", ["round", "family"], ["lagXcons"],
                    test="persistence (coop ~ lag*cons)", stratum="all settings", family=fam, role=role, feature=f, rung="R1")
                med = g["coop_lag"].median()
                for lab, sub in [("prev below median", g[g["coop_lag"] < med]), ("prev at/above median", g[g["coop_lag"] >= med])]:
                    fit(rows, sub, "absdelta", [x, "coop_lag"], "run_agent", ["round"], [x],
                        test=f"|change| ~ cons, {lab}", stratum="all settings", family=fam, role=role, feature=f, rung="R2")
        if len(zfam):
            zf = zfam.idxmax()
            g = d[(d["family"] == zf) & (d["prev_zero"] == 1)]
            fit(rows, g, "zero", [x], "cell", ["round"], [x], test=f"{zf}: send 0 again after sending 0",
                stratum="all settings", family=zf, role="investor", feature=f, rung="R1")
    t = pd.DataFrame(rows)
    t.attrs["zero_counts"] = zfam.to_dict()
    return t


def r3(d, m):
    """8-agent myth_game: shown myth's consistency -> reader's decision in the same round.
    The shown myth is always last round's partner's, so keep shown author != current partner and control
    for what that author did to the reader last round (partner_prev_move). Placebo: mean of unseen
    same-family myths from the same run and round (not the reader, author or partner)."""
    g = d[(d["size"] == 8) & (d["task_order"] == "myth_game") & d["shown_idx"].notna()].copy()
    g = g[g["shown_author"] != g["partner"]]
    mm = m[m["valid"]]
    pools = {k: v.index.to_list() for k, v in mm.groupby(["run_id", "round", "family"])}
    fv = m[FEATS]
    nulls = []
    for x in g.itertuples(index=False):
        sr = int(m.at[int(x.shown_idx), "round"])
        c = [j for j in pools.get((x.run_id, sr, x.shown_family), [])
             if m.at[j, "agent"] not in (x.agent, x.shown_author, x.partner)]
        nulls.append(fv.loc[c].mean().to_numpy() if c else [np.nan] * len(FEATS))
    nulls = np.array(nulls)
    for j, f in enumerate(FEATS):
        g[f"null_{f}"] = nulls[:, j]
    sd = d["own_cons_emb"].std()
    for p in ["null", "shown", "own"]:
        g[f"{p}_cons_emb_z"] = g[f"{p}_cons_emb"] / sd
    rows = []
    for st, fam, to, gf in strata(g, by_task_order=False):
        for y, role in [("coop", "investor"), ("coop", "trustee"), ("absdelta", "investor"), ("absdelta", "trustee"), ("zero", "investor")]:
            if fam in CEILING_SEND and role == "investor":
                continue
            s = gf[gf["role"] == role]
            for f in ZSET:
                xs = [f"shown_{f}", f"null_{f}", f"own_{f}", "coop_lag", "partner_prev_move"]
                fit(rows, s, y, xs, "run_agent", ["round"], [f"shown_{f}", f"null_{f}"],
                    rung="R3", stratum=st, family=fam, task_order="myth_game", role=role, outcome=y, feature=f)
    return pd.DataFrame(rows), len(g)


def r4(d):
    """myth_game round-1 myth (before any play) -> round-1 move and the agent's later stability.
    September spec: both roles, cell FE + role dummy. Settled spec for round-1 sends: senders only."""
    one = d[(d["task_order"] == "myth_game") & (d["round"] == 1)][["run_id", "agent", "run_agent", "family", "cell", "setting",
                                                                    "role", "coop", "own_cons_judge", "own_cons_lex", "own_cons_emb"]]
    later = d[(d["task_order"] == "myth_game") & (d["round"] >= 2)]
    agg = later.groupby("run_agent").agg(mean_absdelta=("absdelta", "mean"), zero_share=("zero", "mean"))
    agg["sd_sent"] = later[later["role"] == "investor"].groupby("run_agent")["sent"].std()
    agg["sd_retprop"] = later[later["role"] == "trustee"].groupby("run_agent")["return_proportion"].std()
    agg["mean_coop"] = later.groupby("run_agent")["coop"].mean()
    a = one.merge(agg.reset_index(), on="run_agent")
    a["own_cons_emb_z"] = a["own_cons_emb"] / d["own_cons_emb"].std()
    rows = []
    for st, fam, to, s in strata(a, by_task_order=False):
        for y in ["coop", "coop_senders", "mean_absdelta", "sd_sent", "sd_retprop", "zero_share", "mean_coop"]:
            ss = s[s["role"] == "investor"] if y == "coop_senders" else s
            yy = "coop" if y == "coop_senders" else y
            if fam in CEILING_SEND and y == "coop_senders":
                continue
            for f in ZSET:
                x = f"own_{f}"
                if ss[x].nunique() < 2:
                    continue
                dummies = ([] if y == "coop_senders" else ["role"]) + (["family"] if fam == "all" else [])
                lab = {"coop": "coop (round 1)", "coop_senders": "send (round 1, senders)"}.get(y, y + " (rounds 2-10)")
                fit(rows, ss, yy, [x], "cell", dummies, [x], rung="R4", stratum=st, family=fam, task_order="myth_game",
                    outcome=lab, feature=f, role="investor" if y == "coop_senders" else "both")
    return pd.DataFrame(rows), a


def holm(t):
    t = t.copy()
    t["p_holm_within_table"] = multipletests(t["p"].fillna(1), method="holm")[1]
    t["p_holm_stratum"] = np.nan
    keys = [k for k in ["stratum", "family", "task_order"] if k in t]
    for _, idx in t.groupby(keys).groups.items():
        t.loc[idx, "p_holm_stratum"] = multipletests(t.loc[idx, "p"].fillna(1), method="holm")[1]
    return t


def main():
    m, d = load()
    d = standardise(d)
    out = {}
    out["r12"] = r1_r2(d)
    lk = lockin(d)
    print("repeat-zero opportunities by family:", lk.attrs["zero_counts"])
    out["lockin"] = lk
    out["r3"], n3 = r3(d, m)
    out["r4"], a4 = r4(d)
    for k, t in out.items():
        t = holm(t)
        t.round(4).to_csv(OUT / f"predict_{k}.csv", index=False)
        out[k] = t
    a4.to_csv(OUT / "predict_r4_agents.csv", index=False)
    d.drop(columns=["path"]).to_csv(OUT / "decision_table.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(NAME, "R3 decisions:", n3)
    cols = ["rung", "stratum", "family", "role", "outcome", "feature", "term", "coef", "ci_low", "ci_high", "p", "n", "n_runs"]
    for k in ["r12", "r3", "r4"]:
        t = out[k]
        print(t[t["stratum"] == "all settings"][cols].round(3).to_string())


if __name__ == "__main__":
    main()
