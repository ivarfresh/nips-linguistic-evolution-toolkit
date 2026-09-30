#!/usr/bin/env python3
"""Own-myth-as-plan, stated amounts and judge quality: September vs frontier.

Run build.py first. Outputs (this folder):
  agreement.csv          GLM vs DeepSeek: label, send rule, stated amount, 0-10 send/return score
  r4.csv                 round-1 myth_game senders: own round-1 myth measure -> round-1 send (composition FE)
  persistence.csv        round-1 own myth -> mean send/5 over rounds 2-10 and 6-10 (cell FE)
  h2_send.csv            8-agent myth_game: shown myth's stated amount -> reader's next send
  h2_myth.csv            8-agent myth_game: shown myth's stated amount -> reader's next myth amount
  reverse.csv            own send -> next own myth amount (agent + round FE)
  label_vs_finer.csv     label vs finer measures (z per SD) at R1 / R2 / R4, investor send/5
  scorecard_rows.csv     one row per (dataset, setting, population, family, finding)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.multitest import multipletests

from build import CACHE
from common import CEILING_FAM, OUT, SEND_FAMS, SEND_ORD, ceiling, fit, status, t_inference

warnings.filterwarnings("ignore")
DS = ("september", "frontier")
M = {ds: pd.read_pickle(CACHE / f"myths_{ds}.pkl") for ds in DS}
P = {ds: pd.read_pickle(CACHE / f"panel_{ds}.pkl") for ds in DS}
SC = []  # scorecard rows


def fam_sets(ds):
    a, b = SEND_FAMS[ds]
    return [(f"{a}+{b}", [a, b]), (a, [a]), (b, [b]), (CEILING_FAM[ds], [CEILING_FAM[ds]])]


def sc(ds, setting, population, family, finding, r, note="", placebo=False, status_override=None):
    SC.append(dict(dataset=ds, placebo=placebo, status_override=status_override, setting=setting, population=population, family=family, finding=finding,
                   effect=r.get("coef"), ci_low=r.get("ci_low"), ci_high=r.get("ci_high"), p=r.get("p"),
                   n_runs=r.get("n_runs"), n_obs=r.get("n"), note=(r.get("note") or "") + (("; " + note) if note else "")))


def smfit(df, formula, term):
    """statsmodels formula with dummy FE (small-FE designs: R4, persistence), SE clustered by run."""
    lhs = formula.split("~")[0].strip()
    df = df.dropna(subset=[lhs, term])
    out = dict(n=len(df), n_runs=df.run_id.nunique(), coef=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, note="")
    if len(df) < 10 or out["n_runs"] < 4:
        out["note"] = "underpowered: too few rows or runs"
        return out
    if df[lhs].std() < 1e-9:
        out["note"] = "not estimable: outcome constant (ceiling)"
        return out
    if df[term].nunique() < 2:
        out["note"] = "not estimable: predictor constant"
        return out
    r = smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.run_id)[0]})
    ci = r.conf_int().loc[term]
    lo, hi, p = t_inference(r.params[term], r.bse[term], out["n_runs"])
    out.update(coef=r.params[term], ci_low=lo, ci_high=hi, p=p, ci_low_z=ci[0], ci_high_z=ci[1], p_z=r.pvalues[term])
    at_ceiling, off = ceiling(df[lhs])
    out["outcome_spread"] = off
    if at_ceiling:
        out["note"] = f"not estimable: ceiling ({off})"
    return out


# ============================================================ 1. judge agreement
def kappa(a, b, w=None):
    ok = a.notna() & b.notna()
    return (cohen_kappa_score(a[ok].astype(str), b[ok].astype(str), weights=w) if w is None
            else cohen_kappa_score(a[ok], b[ok], weights=w)), int(ok.sum())


ag = []
for ds in DS:
    m = M[ds]
    groups = [("all", m)] + list(m.groupby("family"))
    for fam, g in groups:
        base = dict(dataset=ds, family=fam)
        if g["label_ds"].notna().any():
            k, n = kappa(g.label_glm, g.label_ds)
            ag.append({**base, "measure": "moral label (3-way)", "stat": "kappa", "value": k, "n": n,
                       "pct_agree": (g.label_glm == g.label_ds)[g.label_ds.notna()].mean(),
                       "glm_share_generous": (g.label_glm == "be generous").mean(),
                       "ds_share_generous": (g.label_ds == "be generous")[g.label_ds.notna()].mean()})
        else:
            ag.append({**base, "measure": "moral label (3-way)", "stat": "kappa", "value": np.nan, "n": 0,
                       "note": "DeepSeek labels not available"})
        x = g.dropna(subset=["send_rule", "send_rule_ds"])
        if len(x):
            k, n = kappa(x.send_rule, x.send_rule_ds)
            ag.append({**base, "measure": "send rule (5 bands + unspecified)", "stat": "kappa", "value": k, "n": n,
                       "pct_agree": (x.send_rule == x.send_rule_ds).mean()})
            o = x[["send_rule", "send_rule_ds"]].apply(lambda s: s.map(SEND_ORD)).dropna()
            if o.send_rule.nunique() > 1:
                k, n = kappa(o.send_rule, o.send_rule_ds, "quadratic")
                ag.append({**base, "measure": "send rule ordinal", "stat": "quadratic kappa", "value": k, "n": n,
                           "pct_agree": (o.send_rule == o.send_rule_ds).mean()})
            b = x.dropna(subset=["send_amount", "send_amount_ds"])
            ag.append({**base, "measure": "stated send amount (both named)", "stat": "pearson r",
                       "value": b.send_amount.corr(b.send_amount_ds) if b.send_amount.nunique() > 1 else np.nan,
                       "n": len(b), "pct_agree": (b.send_amount == b.send_amount_ds).mean(),
                       "named_glm_only": int((x.send_amount.notna() & x.send_amount_ds.isna()).sum()),
                       "named_ds_only": int((x.send_amount.isna() & x.send_amount_ds.notna()).sum()),
                       "glm_share_named": x.send_amount.notna().mean(), "ds_share_named": x.send_amount_ds.notna().mean()})
            for f in ("consistency",):
                ag.append({**base, "measure": f"{f} flag", "stat": "kappa", "value": kappa(x[f], x[f + "_ds"])[0],
                           "n": len(x), "glm_rate": x[f].astype(float).mean(), "ds_rate": x[f + "_ds"].astype(float).mean()})
        for f in ("send", "return"):
            a, b = g[f"give_{f}_glm"], g[f"give_{f}_ds"]
            ok = a.notna() & b.notna()
            ag.append({**base, "measure": f"0-10 {f} score", "stat": "pearson r",
                       "value": a[ok].corr(b[ok]) if a[ok].std() > 0 and b[ok].std() > 0 else np.nan, "n": int(ok.sum()),
                       "spearman": stats.spearmanr(a[ok], b[ok]).statistic if a[ok].std() > 0 else np.nan,
                       "glm_mean": a.mean(), "glm_sd": a.std(), "ds_mean": b.mean(), "ds_sd": b.std()})
AG = pd.DataFrame(ag)
AG["rules_sample"] = np.where(AG.dataset == "september", "DeepSeek rules on a 1,000-myth sample", "all myths")
AG["deepseek_note"] = np.where(AG.dataset == "frontier", "frontier DeepSeek served with hidden reasoning; not strictly comparable to September DeepSeek", "")
AG.round(4).to_csv(OUT / "agreement.csv", index=False)

# ============================================================ 2. R4: own round-1 myth -> round-1 send
R4_MEAS = [("named amount only ($/$)", "own_send_amount"), ("named, else band midpoint ($/$)", "own_judge_amount"),
           ("send rule (per band level)", "own_rule_send_ord"), ("0-10 send score GLM (per point)", "own_give_send_glm"),
           ("0-10 send score DeepSeek (per point)", "own_give_send_ds"), ("label ordinal GLM (per step)", "own_label_ord_glm"),
           ("label ordinal DeepSeek (per step)", "own_label_ord_ds")]
r4rows = []
for ds in DS:
    p = P[ds]
    r4 = p[(p.role == "investor") & (p.task_order == "myth_game") & (p["round"] == 1)].copy()
    strata = [("2+8-agent", "all myth→game round-1 senders", r4)]
    for setting in sorted(r4.setting.unique()):
        strata.append((setting.split(" ")[0], setting, r4[r4.setting == setting]))
    for sname, spop, s0 in strata:
        for fname, fams in fam_sets(ds):
            s = s0[s0.family.isin(fams)]
            if not len(s):
                continue
            for lab, col in R4_MEAS:
                a = s.dropna(subset=[col])
                r = smfit(a, f"send ~ {col} + C(composition)", col) if len(a) else dict(n=0, note="underpowered: no sender has this measure")
                exact = np.isclose(a.send, a[col]).mean() if col == "own_send_amount" and len(a) else np.nan
                r4rows.append(dict(dataset=ds, setting=sname, population=spop, family=fname, measure=lab,
                                   n_senders=len(s), share_with_measure=len(a) / len(s), mean_send=s.send.mean(),
                                   sd_send=s.send.std(), mean_measure=a[col].mean() if len(a) else np.nan,
                                   exact_match=exact, **r))
R4 = pd.DataFrame(r4rows)
R4.round(4).to_csv(OUT / "r4.csv", index=False)

# exact-match counts and a non-degenerate test: does "myth names less than $5" go with "sends less than $5"?
ex = []
for ds in DS:
    p = P[ds]
    r4 = p[(p.role == "investor") & (p.task_order == "myth_game") & (p["round"] == 1)]
    for fname, fams in fam_sets(ds):
        s = r4[r4.family.isin(fams)]
        a = s.dropna(subset=["own_send_amount"])
        k = int(np.isclose(a.send, a.own_send_amount).sum())
        below_named, below_sent = s.own_send_amount < 5, s.send < 5   # unnamed counts as "not below $5"
        tab = pd.crosstab(below_named, below_sent).reindex(index=[False, True], columns=[False, True], fill_value=0)
        fp = stats.fisher_exact(tab.to_numpy())[1] if tab.to_numpy().min(0).sum() >= 0 and below_sent.any() and below_named.any() else np.nan
        ex.append(dict(dataset=ds, family=fname, n_senders=len(s), n_named=len(a), n_exact=k, exact_share=k / len(a) if len(a) else np.nan,
                       n_send_below5=int(below_sent.sum()), n_named_below5=int(below_named.sum()),
                       both_below5=int((below_named & below_sent).sum()), fisher_p_below5=fp,
                       named_values=", ".join(f"{v:g}->{w:g}" for v, w in zip(a.own_send_amount, a.send) if not np.isclose(v, w))))
EX = pd.DataFrame(ex)
EX.to_csv(OUT / "r4_exact.csv", index=False)
exk = EX.set_index(["dataset", "family"])
for r in R4[(R4.setting == "2+8-agent") & R4.measure.isin(["named amount only ($/$)", "named, else band midpoint ($/$)"])].to_dict("records"):
    e = exk.loc[(r["dataset"], r["family"])]
    note, ov = "", None
    if r["measure"].startswith("named amount only"):
        note = (f"{e.n_exact}/{e.n_named} send exactly the named amount; {e.n_send_below5}/{e.n_senders} senders send < $5, "
                f"{e.both_below5} of them named < $5 (Fisher p {e.fisher_p_below5:.2g})")
        if pd.notna(r["coef"]) and r["ci_high"] - r["ci_low"] < 1e-6:
            note += "; slope CI degenerate (every named sender matches), status from the exact-match count"
            ov = "yes"
        elif str(r.get("note", "")).startswith("not estimable: ceiling") and e.n_send_below5 > 0 \
                and e.both_below5 == e.n_send_below5 == e.n_named_below5:
            note += ("; near ceiling: every sub-$5 sender named a sub-$5 amount and vice versa, but only "
                     f"{e.n_send_below5} senders deviate from $5 (Fisher not clustered by run)")
            ov = "suggestive"
    sc(r["dataset"], "2+8-agent", "all myth→game round-1 senders", r["family"],
       f"R4 own round-1 myth → round-1 send, {r['measure']}", r, note=note, status_override=ov)
for r in R4[(R4.setting != "2+8-agent") & (R4.measure == "named amount only ($/$)")].to_dict("records"):
    sc(r["dataset"], r["setting"], r["population"], r["family"], "R4 own round-1 myth → round-1 send, named amount only ($/$)", r,
       status_override="yes" if pd.notna(r["coef"]) and r["ci_high"] - r["ci_low"] < 1e-6 and r["n_runs"] >= 10 else None,
       note=f"exact match {r['exact_match']:.0%}" if pd.notna(r.get("exact_match")) else "")

# ============================================================ 3. persistence of the round-1 plan
pers = []
for ds in DS:
    p = P[ds]
    mg = p[p.task_order == "myth_game"]
    r1 = M[ds][M[ds]["round"] == 1][["run_id", "agent", "rule_send_ord", "send_amount", "judge_amount"]]
    for win, (lo, hi) in (("rounds 2-10", (2, 10)), ("rounds 6-10", (6, 10))):
        g = (mg[(mg.role == "investor") & mg["round"].between(lo, hi)]
             .groupby(["run_id", "agent", "family", "size", "composition"])["coop"].mean().reset_index())
        g = g.merge(r1, on=["run_id", "agent"])
        g["cellf"] = g["size"].astype(str) + "|" + g.composition + "|" + g.family
        for fname, fams in fam_sets(ds):
            s = g[g.family.isin(fams)]
            for lab, col in (("send rule (per band level)", "rule_send_ord"), ("named, else band midpoint ($)", "judge_amount")):
                r = smfit(s.dropna(subset=[col]), f"coop ~ {col} + C(cellf)", col)
                pers.append(dict(dataset=ds, window=win, family=fname, measure=lab, outcome="mean send/5", **r))
PERS = pd.DataFrame(pers)
PERS.round(4).to_csv(OUT / "persistence.csv", index=False)
for r in PERS[PERS.measure == "send rule (per band level)"].to_dict("records"):
    sc(r["dataset"], "2+8-agent", "myth→game agents", r["family"],
       f"round-1 send rule → mean send/5 {r['window']} (per band level)", r)

# ============================================================ 4. H2: shown myth's stated amount
BASE = "send ~ shown_judge_amount + unseen_judge_amount + lag_sent + own_prev_judge_amount + b_coop_prev + c_coop_prev + C(runround)"
FUT = BASE.replace("unseen_judge_amount", "unseen_judge_amount + future_judge_amount")
MYTH = "own_judge_amount ~ shown_judge_amount + unseen_judge_amount + own_prev_judge_amount + lag_coop_any + b_coop_prev + C(runround)"
h2s, h2m = [], []
for ds in DS:
    p = P[ds]
    r3 = p[(p["size"] == 8) & (p.task_order == "myth_game") & p.shown_judge_amount.notna()]
    inv = r3[r3.role == "investor"]
    mt = r3.drop_duplicates(["run_id", "agent", "own_round"])
    pops = [("all 8-agent myth→game", lambda d: d)]
    pops += [("homogeneous", lambda d: d[~d.mixed]), ("mixed", lambda d: d[d.mixed])]
    all_fams = [("all", None)] + [(f, [f]) for f in sorted(p.family.unique())]
    for pop, sel in pops:
        for fname, fams in all_fams:
            i = sel(inv) if fams is None else sel(inv)[sel(inv).family.isin(fams)]
            specs = [("shown author ≠ current partner (headline)", i[i.shown_author != i.partner], BASE, "shown_judge_amount"),
                     ("shown author ≠ current partner: UNSEEN placebo", i[i.shown_author != i.partner], BASE, "unseen_judge_amount"),
                     ("all shown myths", i, BASE, "shown_judge_amount"),
                     ("≠ partner, future-myth placebo in model: shown", i[i.shown_author != i.partner], FUT, "shown_judge_amount"),
                     ("≠ partner, future-myth placebo in model: FUTURE", i[i.shown_author != i.partner], FUT, "future_judge_amount")]
            for lab, d, f, term in specs:
                h2s.append(dict(dataset=ds, population=pop, family=fname, spec=lab, term=term, **fit(d, f, term)))
            j = sel(mt) if fams is None else sel(mt)[sel(mt).family.isin(fams)]
            for lab, term in (("shown amount (headline, no future term)", "shown_judge_amount"),
                              ("UNSEEN placebo", "unseen_judge_amount")):
                h2m.append(dict(dataset=ds, population=pop, family=fname, spec=lab, term=term, **fit(j, MYTH, term)))
H2S, H2M = pd.DataFrame(h2s), pd.DataFrame(h2m)
H2S.round(4).to_csv(OUT / "h2_send.csv", index=False)
H2M.round(4).to_csv(OUT / "h2_myth.csv", index=False)


def h2note(r):
    if r["dataset"] == "frontier" and r["population"] == "mixed" and r["family"] == "GeminiPro":
        return "not testable by design: 2 GeminiPro members, so a GeminiPro-authored shown myth has no same-family unseen comparison"
    if r["dataset"] == "frontier" and r["population"] == "mixed":
        return "GeminiPro-authored shown myths drop (no same-family unseen comparison)"
    return ""


def popname(ds, pop, fam):
    if ds == "frontier":
        return {"all 8-agent myth→game": "8-agent myth→game (homog + mixed)", "homogeneous": f"8 {fam}" if fam != "all" else "8-agent homogeneous",
                "mixed": "2 GeminiPro + 3 Opus + 3 Sol"}[pop]
    return {"all 8-agent myth→game": "8-agent myth→game (homog + mixed)", "homogeneous": f"8 {fam}" if fam != "all" else "8-agent homogeneous",
            "mixed": "8-agent mixed"}[pop]


for r in H2S[H2S.spec.isin(["shown author ≠ current partner (headline)", "shown author ≠ current partner: UNSEEN placebo"])].to_dict("records"):
    sc(r["dataset"], "8-agent", popname(r["dataset"], r["population"], r["family"]), r["family"],
       "H2 shown myth stated amount → reader's next send ($/$)" + (" [unseen placebo]" if "UNSEEN" in r["spec"] else ""), r,
       placebo="UNSEEN" in r["spec"], note=h2note(r))
for r in H2M.to_dict("records"):
    sc(r["dataset"], "8-agent", popname(r["dataset"], r["population"], r["family"]), r["family"],
       "H2 shown myth stated amount → reader's next myth amount ($/$)" + (" [unseen placebo]" if "UNSEEN" in r["spec"] else ""), r,
       placebo="UNSEEN" in r["spec"], note=h2note(r))
for ds in DS:
    for fam in sorted(P[ds].family.unique()):
        for f in ("H2 shown myth stated amount → reader's next send ($/$)", "H2 shown myth stated amount → reader's next myth amount ($/$)"):
            sc(ds, "2-agent", "all dyads", fam, f, {"note": "not testable by design: in a dyad the shown myth is always the current partner's"})
for r in H2S[(H2S.spec == "shown author ≠ current partner (headline)") & (H2S.dataset == "september") & (H2S.population == "all 8-agent myth→game") & (H2S.family == "all")].to_dict("records"):
    sc("september", "8-agent", "8-agent myth→game (homog + mixed)", "all",
       "H2 note: README +0.04 (author ≠ partner) came from the model WITH the future-myth term; headline spec here", r,
       note="future-controlled version: see h2_send.csv", placebo=True)

# ============================================================ 5. reverse: send -> next own myth amount
rev = []
for ds in DS:
    p, m = P[ds], M[ds]
    di = p[p.role == "investor"][["run_id", "agent", "round", "send", "task_order"]]
    di = di.assign(myth_round=np.where(di.task_order == "game_myth", di["round"], di["round"] + 1))
    rv = m.merge(di.drop(columns="task_order").rename(columns={"round": "send_round"}),
                 left_on=["run_id", "agent", "round"], right_on=["run_id", "agent", "myth_round"], how="inner")
    rv = rv.sort_values(["run_id", "agent", "round"])
    rv["prev_amt"] = rv.groupby(["run_id", "agent"])["judge_amount"].shift()
    rv["agentrun"] = rv.run_id + "|" + rv.agent
    for fam in sorted(rv.family.unique()):
        for setting in ["all"] + sorted(rv.setting.unique()):
            for order in ("both", "game_myth", "myth_game"):
                s = rv[rv.family == fam]
                s = s if setting == "all" else s[s.setting == setting]
                s = s if order == "both" else s[s.task_order == order]
                r = fit(s, "judge_amount ~ send + prev_amt + C(agentrun) + C(round)", "send")
                rev.append(dict(dataset=ds, family=fam, setting=setting, task_order=order, **r))
REV = pd.DataFrame(rev)
REV.round(4).to_csv(OUT / "reverse.csv", index=False)
for r in REV[(REV.task_order == "both")].to_dict("records"):
    setting = "2+8-agent" if r["setting"] == "all" else r["setting"].split(" ")[0]
    pop = "all myth runs" if r["setting"] == "all" else r["setting"]
    sc(r["dataset"], setting, pop, r["family"], "reverse: own send → next own myth amount ($/$, agent FE)", r)

# ============================================================ 6. label vs finer measures by rung
LVF = ["label_ord_glm", "gen_glm", "label_ord_ds", "rule_prescribed", "rule_send_ord", "give_send_glm", "give_send_ds"]
lv = []
for ds in DS:
    m, p = M[ds], P[ds].copy()
    for c in LVF:
        if m[c].notna().any():
            p[f"z_{c}"] = (p[f"own_{c}"] - m[c].mean()) / m[c].std()
    meas = [c for c in LVF if f"z_{c}" in p]
    inv = p[p.role == "investor"]
    for fname, fams in fam_sets(ds)[:3]:
        d = inv[inv.family.isin(fams)]
        rungs = {"R1 cell+family+round FE": (d, "coop ~ {x} + C(cell) + C(family) + C(round)"),
                 "R2 agent+round FE, own lag": (d.dropna(subset=["coop_lag"]), "coop ~ {x} + coop_lag + C(agentrun) + C(round)"),
                 "R2 myth_game only": (d[d.task_order == "myth_game"].dropna(subset=["coop_lag"]), "coop ~ {x} + coop_lag + C(agentrun) + C(round)"),
                 "R2 game_myth only": (d[d.task_order == "game_myth"].dropna(subset=["coop_lag"]), "coop ~ {x} + coop_lag + C(agentrun) + C(round)"),
                 "R4 myth→game round 1": (d[(d.task_order == "myth_game") & (d["round"] == 1)], "coop ~ {x} + C(family) + C(composition)")}
        for rung, (dd, form) in rungs.items():
            inter = dd.dropna(subset=[f"z_{c}" for c in meas])
            for c in meas:
                r = fit(inter, form.format(x=f"z_{c}"), f"z_{c}")
                lv.append(dict(dataset=ds, family=fname, rung=rung, measure=c, **r))
LV = pd.DataFrame(lv)
LV.round(4).to_csv(OUT / "label_vs_finer.csv", index=False)
R2NOTE = ("low rung: the myth also records play (reverse slope), only the agent's own last move is controlled; "
          "in game_myth the own myth is written after the previous round's play")
for r in LV[LV.measure.isin(["label_ord_glm", "give_send_glm", "rule_prescribed"])].to_dict("records"):
    sc(r["dataset"], "2+8-agent", "all myth runs" if not r["rung"].startswith("R4") else "all myth→game round-1 senders",
       r["family"], f"{r['rung']}: own myth {r['measure']} → send/5 (per SD)", r,
       note=R2NOTE if r["rung"].startswith("R2") else "")

# ============================================================ scorecard: Holm within dataset x setting x population x family
S = pd.DataFrame(SC)
S["holm_p"] = np.nan
for _, g in S[~S.placebo].groupby(["dataset", "setting", "population", "family"]):
    ok = g.p.notna()
    if ok.sum():
        S.loc[g.index[ok], "holm_p"] = multipletests(g.loc[ok, "p"], method="holm")[1]
S["status"] = [r["status_override"] if isinstance(r["status_override"], str) else status({**r, "p": np.nan if r["placebo"] else r["p"]})
               for r in S.to_dict("records")]
S.loc[S.placebo & (S.status == "no detectable effect") & (S.p < 0.05), "status"] = "suggestive"
ceil = S.family.isin(["Gemini", "GeminiPro"]) & S.effect.isna() & ~S.status.str.startswith("not testable")
S.loc[ceil, "status"] = "not estimable: ceiling"
S["note"] = np.where(S.placebo, "placebo (raw p, not in Holm); " + S.note.fillna(""), S.note.fillna(""))
S["note"] = S.note.str.replace(r"(;\s*)+", "; ", regex=True).str.strip("; ")
cols = ["dataset", "setting", "population", "family", "finding", "effect", "ci_low", "ci_high", "p", "holm_p",
        "n_runs", "n_obs", "status", "note"]
S[cols].sort_values(["finding", "dataset", "setting", "population", "family"]).round(4).to_csv(OUT / "scorecard_rows.csv", index=False)
print(len(S), "scorecard rows")
