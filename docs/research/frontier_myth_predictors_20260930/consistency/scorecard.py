#!/usr/bin/env python3
"""Scorecard rows for the consistency lens (H3), September beside frontier.

Reads september/ and frontier/ outputs of features, drift, ratchet, ratchet_continuous, predict, spread and
spread_placebo_directions. Writes scorecard_rows.csv in this folder.

Status rules (in order):
  not estimable: ceiling   investor rows of the ceiling family (GeminiPro / Gemini), or outcome modal share >= 0.9,
                           or outcome sd 0
  underpowered             fewer than 10 runs, or the feature is (almost) never present (share < 0.03, or fewer
                           than 5 units where it varies), or fewer than 100 myths in the rarer ratchet arm
  yes                      Holm p < 0.05 (Holm within the source table's stratum = setting x family x task order)
  suggestive               raw p < 0.05
  no detectable effect     otherwise
Drift rows: judge-flag drift is 'yes' only when both judges (GLM, DeepSeek) rise Holm-significantly; the frontier
embedding difference score is not a drift toward consistency (the anchors-only projection falls).
holm_p_lens: Holm over all R1-R4 prediction rows of one dataset in this scorecard (comparable to September's
'none surviving Holm over ~540 tests').
"""
import numpy as np
import pandas as pd

from common import HERE

FEATLAB = {"cons_judge": "GLM judge flag", "cons_lex": "keyword", "cons_emb_z": "embedding score (sd)",
           "cons_emb": "embedding score", "cons_judge_ds": "DeepSeek judge flag"}
CEIL = {"september": ["Gemini"], "frontier": ["GeminiPro"]}
FRONTIER_MIX8 = "2 GeminiPro + 3 Opus + 3 Sol"


def population(ds, setting, fam):
    if setting == "all settings":
        return f"all {fam} runs" if fam != "all" else "all runs"
    if "-agent " not in setting or "," in setting:
        return setting  # spread strata are named by exposure
    size, kind = setting.split("-agent ")
    if kind == "homogeneous":
        return f"{fam} dyads" if size == "2" else f"8 {fam}"
    if size == "2":
        return f"mixed dyads, {fam} members"
    return FRONTIER_MIX8 if ds == "frontier" else f"8-agent mixed populations, {fam} members"


def status(ds, fam, role, p, holm, n_runs, y_modal=np.nan, y_sd=np.nan, x_mean=np.nan, x_var=np.nan, feature=""):
    if (fam in CEIL[ds] and role == "investor") or (y_modal == y_modal and y_modal >= 0.9) or y_sd == 0 or p != p:
        return "not estimable: ceiling"
    if n_runs < 10:
        return "underpowered"
    if feature in ("cons_lex", "cons_judge", "cons_judge_ds") and x_mean == x_mean and (x_mean < 0.03 or x_mean > 0.97):
        return "underpowered"
    if x_var == x_var and x_var < 5:
        return "underpowered"
    if holm == holm and holm < 0.05:
        return "yes"
    return "suggestive" if p < 0.05 else "no detectable effect"


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    b = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def row(ds, setting, fam, finding, effect, lo, hi, p, holm, n_runs, n_obs, st, note=""):
    return {"dataset": ds, "setting": setting, "population": population(ds, setting, fam), "family": fam, "finding": finding,
            "effect": effect, "ci_low": lo, "ci_high": hi, "p": p, "holm_p": holm, "n_runs": n_runs, "n_obs": n_obs,
            "status": st, "note": note}


def drift_rows(ds):
    out = []
    m = pd.read_csv(HERE / ds / "myth_features.csv")
    m = m[m.valid]
    m = m.assign(phase=np.where(m["round"] == 1, "r1", np.where(m["round"] >= 8, "late", None))).dropna(subset=["phase"])
    rp = pd.read_csv(HERE / ds / "drift_rise_pooled.csv")
    for meas in ["cons_lex", "cons_judge", "cons_judge_ds", "cons_emb"]:
        run = m.groupby(["setting", "family", "run_id", "phase"])[meas].mean().unstack("phase").dropna()
        groups = [(s, f, g) for (s, f), g in run.groupby(level=[0, 1])] + [("all settings", f, g) for f, g in run.groupby(level=1)]
        for s, f, g in groups:
            r = rp[(rp.setting == s) & (rp.family == f) & (rp.measure == meas)]
            if not len(r):
                continue
            r = r.iloc[0]
            d = g["late"] - g["r1"]
            lo, hi = boot_ci(d)
            base = (g["r1"].mean() + g["late"].mean()) / 2
            if meas == "cons_judge_ds" and ds == "september":
                note_extra = "; DeepSeek labels only on a 1,000-myth sample"
            else:
                note_extra = ""
            st = status(ds, f, "", r.p_wilcoxon, r.p_holm_stratum, len(d))
            if meas == "cons_lex" and max(g["r1"].mean(), g["late"].mean()) < 0.05:
                st = "underpowered"
                note_extra += "; keyword almost never used"
            fname = f"drift: {FEATLAB[meas]} round 1 -> rounds 8-10 (task orders pooled)"
            if meas == "cons_emb":
                fname = "drift: embedding difference score (consistency minus generosity anchors) round 1 -> rounds 8-10"
                if ds == "frontier":
                    note_extra += ("; the difference rises because myths move AWAY from the generosity anchors; the "
                                   "consistency-anchors projection falls (see anchors-only row): not a drift toward consistency")
                    if st == "yes":
                        st = "no detectable effect"
            if meas in ("cons_judge", "cons_judge_ds") and st == "yes":
                other = "cons_judge_ds" if meas == "cons_judge" else "cons_judge"
                o = rp[(rp.setting == s) & (rp.family == f) & (rp.measure == other)]
                if not len(o) or not (o.iloc[0].p_holm_stratum < 0.05 and o.iloc[0]["diff"] > 0):
                    st = "suggestive"
                    note_extra += "; only one judge rises Holm-significantly"
            out.append(row(ds, s, f, fname, d.mean(), lo, hi,
                           r.p_wilcoxon, r.p_holm_stratum, len(d), np.nan, st,
                           f"round 1 {g['r1'].mean():.2f} (±{g['r1'].std():.2f}) -> rounds 8-10 {g['late'].mean():.2f} "
                           f"(±{g['late'].std():.2f}); {int((d > 0).sum())} of {len(d)} runs rose; Wilcoxon, CI bootstrap over runs"
                           + note_extra))
    return out


def drift_placebo_rows(ds):
    t = pd.read_csv(HERE / ds / "drift_placebo_directions.csv")
    out = []
    for r in t.itertuples():
        st = "yes" if (r.cons_anchors_only_change_sd > 0 and r.share_random_abs_larger_anchors_only < 0.05) else \
             ("no detectable effect" if r.cons_anchors_only_change_sd <= 0 or r.share_random_abs_larger_anchors_only >= 0.2 else "suggestive")
        out.append(row(ds, r.setting, r.family, "drift: consistency-anchors-only embedding projection, round 1 -> rounds 8-10 (sd)",
                       r.cons_anchors_only_change_sd, np.nan, np.nan, np.nan, np.nan, r.n_runs, np.nan, st,
                       f"share of 40 random directions moving more: {r.share_random_abs_larger_anchors_only:.2f}; "
                       f"generosity-anchor projection {r.generosity_change_sd:+.2f} sd; the difference score moves "
                       f"{r.consistency_change_sd:+.2f} sd. Negative = myths move away from the consistency anchors"))
    return out


def ratchet_rows(ds):
    from fe import fe_ols
    m = pd.read_csv(HERE / ds / "myth_features.csv")
    m = m[m.valid].sort_values(["run_id", "agent", "round"])
    m["cell"] = m["composition"] + "|" + m["task_order"]
    t = pd.read_csv(HERE / ds / "ratchet.csv")
    out = []
    for meas in ["cons_lex", "cons_judge"]:
        x = m.assign(prev=m.groupby(["run_id", "agent"])[meas].shift(1)).dropna(subset=["prev", meas])
        for f, g in x.groupby("family"):
            tt = t[(t.measure == meas) & (t.family == f)]
            a, b = tt[tt.prev == 1].iloc[0], tt[tt.prev == 0].iloc[0]
            res = fe_ols(g, meas, ["prev"], absorb="cell", dummies=["round"])
            coef, lo, hi, p = res["prev"]
            st = "underpowered" if min(a.n_myths, b.n_myths) < 100 else status(ds, f, "", p, np.nan, res["n_runs"])
            out.append(row(ds, "all settings", f, f"self-copying ratchet: P({FEATLAB[meas]} | own previous had it) - P(... | not)",
                           coef, lo, hi, p, np.nan, res["n_runs"], res["n"], st,
                           f"raw run-level means {a['mean']:.2f} vs {b['mean']:.2f}; {int(a.n_myths)} myths follow a flagged myth, "
                           f"{int(b.n_myths)} an unflagged one; effect = LPM coefficient, cell + round FE"))
    from statsmodels.stats.multitest import multipletests
    for f in {r["family"] for r in out}:
        rs = [r for r in out if r["family"] == f]
        for r, h in zip(rs, multipletests([r["p"] for r in rs], method="holm")[1]):
            r["holm_p"] = h
            if r["status"] in ("yes", "suggestive"):
                r["status"] = "yes" if h < 0.05 else "suggestive"
    c = pd.read_csv(HERE / ds / "ratchet_continuous.csv")
    c = c[(c.measure == "cons_emb_z")]
    pl = pd.read_csv(HERE / ds / "ratchet_placebo_directions.csv").set_index("family")
    for (s, f), g in c.groupby(["stratum", "family"]):
        r = g[g.term == "own minus unseen"].iloc[0]
        own = g[g.term == "own_prev_cons_emb_z"].iloc[0]
        sh = g[g.term == "shown_prev_cons_emb_z"].iloc[0]
        st = status(ds, f, "", r.p, r.p_holm_stratum, r.n_runs)
        note = f"own previous {own.coef:+.2f}, shown previous {sh.coef:+.2f} (shared history, not causal), per sd; cell + round FE"
        if f in pl.index:
            q = pl.loc[f]
            note += (f"; NOT specific: 20 random embedding directions self-copy with own-previous coefficient median "
                     f"{q.random_own_prev_median:.2f} vs {q.consistency_own_prev:.2f} for consistency (all settings)")
            if st in ("yes", "suggestive") and q.share_random_larger >= 0.05:
                st = "no detectable effect"
        out.append(row(ds, s, f, "self-copying (embedding): own previous myth minus unseen previous myths", r.coef, r.ci_low, r.ci_high,
                       r.p, r.p_holm_stratum, int(r.n_runs), int(r.n), st, note))
    return out


def describes_rows(ds):
    t = pd.read_csv(HERE / ds / "drift_describes_play.csv")
    t = t[t.term == "instab"].copy()
    from statsmodels.stats.multitest import multipletests
    t["holm"] = t.groupby("family")["p"].transform(lambda p: multipletests(p, method="holm")[1])
    out = []
    for r in t.itertuples():
        st = status(ds, r.family, "", r.p, r.holm, r.n_runs)
        if ds == "frontier" and r.family == "GeminiPro":
            st, extra = "not estimable: ceiling", "; GeminiPro play barely varies"
        else:
            extra = ""
        out.append(row(ds, "all settings", r.family, f"describes play: own recent instability -> {FEATLAB[r.measure]} in next myth",
                       r.coef, r.ci_low, r.ci_high, r.p, r.holm, r.n_runs, r.n, st,
                       "negative = steadier play, more consistency language; agent FE + round FE; per unit mean |change| in coop" + extra))
    return out


def predict_rows(ds):
    out = []
    r12 = pd.read_csv(HERE / ds / "predict_r12.csv")
    r12 = r12[r12.task_order == "both"]
    spec = [("R2", "coop", "investor", "R2 own consistency -> send level (agent FE, lagged coop)"),
            ("R2", "coop", "trustee", "R2 own consistency -> return proportion (agent FE, lagged coop)"),
            ("R2", "absdelta", "investor", "R2 own consistency -> |change in send| (stability)"),
            ("R2", "absdelta", "trustee", "R2 own consistency -> |change in return| (stability)"),
            ("R2", "cut_after_letdown", "investor", "R2 own consistency -> cut after a letdown")]
    for rung, y, role, lab in spec:
        for f in ["cons_judge", "cons_lex", "cons_emb_z"]:
            t = r12[(r12.rung == rung) & (r12.outcome == y) & (r12.role == role) & (r12.feature == f)]
            for r in t.itertuples():
                out.append(row(ds, r.stratum, r.family, f"{lab}: {FEATLAB[f]}", r.coef, r.ci_low, r.ci_high, r.p, r.p_holm_stratum,
                               r.n_runs, r.n, status(ds, r.family, role, r.p, r.p_holm_stratum, r.n_runs,
                                                     r.y_modal_share if y in ("coop", "absdelta") else np.nan, r.y_sd,
                                                     r.x_mean, getattr(r, "x_varying_units", np.nan), f),
                               f"outcome sd {r.y_sd:.3f}, modal share {r.y_modal_share:.2f}; feature mean {r.x_mean:.2f}"))
            # ceiling family: explicit rows
            for s in r12.stratum.unique():
                for fam in CEIL[ds]:
                    if role == "investor" and not len(r12[(r12.stratum == s) & (r12.family == fam)]) == 0:
                        out.append(row(ds, s, fam, f"{lab}: {FEATLAB[f]}", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                       "not estimable: ceiling", "sends $5 almost always"))
    r3 = pd.read_csv(HERE / ds / "predict_r3.csv")
    for f in ["cons_judge", "cons_lex", "cons_emb_z"]:
        for role in ["investor", "trustee"]:
            t = r3[(r3.outcome == "coop") & (r3.role == role) & (r3.feature == f) & (r3.term == f"shown_{f}")]
            for r in t.itertuples():
                out.append(row(ds, r.stratum, r.family, f"R3 shown myth's consistency -> reader's {'send' if role == 'investor' else 'return'} "
                               f"(8-agent myth_game, author != partner): {FEATLAB[f]}", r.coef, r.ci_low, r.ci_high, r.p,
                               r.p_holm_stratum, r.n_runs, r.n,
                               status(ds, r.family, role, r.p, r.p_holm_stratum, r.n_runs, r.y_modal_share, r.y_sd, r.x_mean, np.nan, f),
                               "controls: unseen same-family placebo, own myth, lagged coop, author's last move toward reader"))
    if ds == "frontier":
        out.append(row(ds, "8-agent mixed", "GeminiPro", "R3 shown myth's consistency -> reader (within-family unseen placebo)", np.nan,
                       np.nan, np.nan, np.nan, np.nan, 10, np.nan, "not testable by design",
                       "only 2 GeminiPro members: no unseen same-family myth; GeminiPro sends at ceiling"))
    r4 = pd.read_csv(HERE / ds / "predict_r4.csv")
    for y, lab in [("send (round 1, senders)", "R4 own round-1 consistency -> round-1 send (senders)"),
                   ("mean_coop (rounds 2-10)", "R4 own round-1 consistency -> mean cooperation rounds 2-10")]:
        for f in ["cons_judge", "cons_lex", "cons_emb_z"]:
            for r in r4[(r4.outcome == y) & (r4.feature == f)].itertuples():
                out.append(row(ds, r.stratum, r.family, f"{lab}: {FEATLAB[f]}", r.coef, r.ci_low, r.ci_high, r.p, r.p_holm_stratum,
                               r.n_runs, r.n, status(ds, r.family, r.role, r.p, r.p_holm_stratum, r.n_runs, r.y_modal_share, r.y_sd,
                                                     r.x_mean, np.nan, f), "myth_game only; cell FE"))
    return out


def spread_rows(ds):
    t = pd.read_csv(HERE / ds / "spread_models.csv")
    pl = pd.read_csv(HERE / ds / "spread_placebo_summary.csv")
    pl = pl[pl.direction.str.startswith("consistency")].set_index("stratum")
    t = t[~t.stratum.str.contains(r"\| child") & t.term.isin(["shown (no future term)", "shown minus future"])]
    t = t[(t.measure != "cons_lex") | t["sample"].str.startswith("new")]
    out = []
    for r in t.itertuples():
        fam = "all"
        if r.stratum.startswith("cross-family"):
            fam = r.stratum.split(", ")[1].split(" parent")[0] + " parent"
        setting = r.stratum
        st = status(ds, fam, "", r.p, np.nan, r.n_runs)
        note = f"{r.sample}; future-myth placebo = shown author's next myth"
        if r.measure == "cons_emb_z" and r.stratum in pl.index:
            q = pl.loc[r.stratum]
            col = "share_random_larger_smf" if r.term == "shown minus future" else "share_random_larger_shown"
            note += f"; share of 40 random embedding directions with a larger coefficient: {q[col]:.2f}"
            if q[col] >= 0.05 and st in ("yes", "suggestive"):
                st = "no detectable effect"
                note += " (not specific to consistency: generic copying of the shown myth)"
        if r.measure == "cons_lex" and ds == "frontier":
            st = "underpowered"
            note += "; keyword rare in frontier myths"
        out.append({**row(ds, setting, fam, f"spread from the shown myth, {r.term}: {FEATLAB[r.measure]}", r.coef, r.ci_low, r.ci_high,
                          r.p, np.nan, r.n_runs, r.n, st, note), "population": r.stratum})
    return out


def main():
    rows = []
    for ds in ["september", "frontier"]:
        rows += drift_rows(ds) + drift_placebo_rows(ds) + ratchet_rows(ds) + describes_rows(ds) + predict_rows(ds) + spread_rows(ds)
    t = pd.DataFrame(rows)
    t = t.drop_duplicates(subset=["dataset", "setting", "population", "family", "finding", "status", "note"])
    for c in ["effect", "ci_low", "ci_high"]:
        t[c] = t[c].astype(float).round(4)
    for c in ["p", "holm_p"]:
        t[c] = t[c].astype(float).round(5)
    from statsmodels.stats.multitest import multipletests
    t["holm_p_lens"] = np.nan
    pred = t.finding.str.match(r"R[1-4] ")
    for ds in ["september", "frontier"]:
        ix = t.index[pred & (t.dataset == ds) & t.p.notna()]
        t.loc[ix, "holm_p_lens"] = multipletests(t.loc[ix, "p"], method="holm")[1]
        print(ds, "R1-R4 tests:", len(ix), "survive lens-wide Holm:", int((t.loc[ix, "holm_p_lens"] < 0.05).sum()),
              "estimable-status yes:", int((t.loc[ix, "status"] == "yes").sum()))
    t.to_csv(HERE / "scorecard_rows.csv", index=False)
    print(t.groupby(["dataset", "status"]).size())


if __name__ == "__main__":
    main()
