#!/usr/bin/env python3
"""Confirmatory tests of the frozen top-5 candidates (candidates_top5.csv) on rows the search
never saw.

R4 (founding window): myth_game round-1 decisions. The decision context is only the system prompt,
the myth prompt, the agent's own round-1 myth and the decision prompt (checked: system prompt,
initial_bias and population_role identical across agents in all 78 myth_game runs). Within a
family x composition cell the only input that differs between agents is the sampled myth, so an
association with the own-myth feature runs through the myth text.
   coop ~ z(own feature) + C(cell) [+ received_now for receivers], SE clustered by run.

R3 (clean exposure): 8-agent myth_game, rounds >= 2. The shown myth was written by the round r-1
co-player before their round r-1 game. Main sample: author had never played the reader before
round r-1 (so the myth was not shaped by the reader). Controls: own lagged moves, co-player's
last 3 games, received amount (receivers), and the author's move toward the reader in round r-1.
Placebo: mean feature of unseen same-family myths from the same run and round (not the reader's,
not the author's). Agent-within-run + round fixed effects, SE clustered by run.
   total effect  = shown coef (without own round-r myth)
   direct effect = shown coef with the reader's own round-r myth feature added (mediator)
   the test      = shown - unseen (Wald), Holm-corrected over all confirmatory tests.

Outputs: confirm_R3_R4.csv, confirm_R4_by_family.csv, confirm_R3_by_family.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
D = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/linguistic-mixed/data/analysis/linguistic_20260923")
OUT = Path(__file__).resolve().parent
CTRL = "lag_send_f + lag_return_f + lag_got_return_f + lag_got_send_f + cop_last3_send_f + cop_last3_return_f"


def fill(d, cols):
    for c in cols:
        d[c + "_miss"] = d[c].isna().astype(float)
        d[c + "_f"] = d[c].fillna(d[c].mean())
    return d


def fit(formula, data):
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data["run_id"])[0]})


def main():
    cands = pd.read_csv(OUT / "candidates_top5.csv")["feature"].tolist()
    myths = pd.read_csv(D / "myths.csv")
    feat = pd.read_csv(OUT / "myth_features.csv")
    # z within family over all myths: effects are per one within-family SD.
    for f in cands:
        g = feat.groupby("family")[f]
        feat[f"z_{f}"] = (feat[f] - g.transform("mean")) / g.transform("std")
    d = pd.read_csv(OUT / "decision_table.csv")
    dec_all = d
    for f in cands:
        zf = feat[f"z_{f}"].to_numpy()
        for src in ("own", "shown"):
            ii = d[f"{src}_i"].to_numpy()
            d[f"{src}_z_{f}"] = np.where(ii >= 0, zf[np.clip(ii, 0, None)], np.nan)
    d = fill(d, ["lag_send", "lag_return", "lag_got_return", "lag_got_send", "cop_last3_send", "cop_last3_return"])
    d["run_agent"] = d["run_id"] + "|" + d["agent"]
    d["cell"] = d["size"].astype(str) + "|" + d["composition"] + "|" + d["family"]
    rows, fam_rows = [], []

    # ---------------------------------------------------------------- R4
    r4 = d[d["split"] == "R4"]
    for f in cands:
        for oname, role in (("send", "investor"), ("return", "trustee")):
            s = r4[(r4["role"] == role)].dropna(subset=[f"own_z_{f}", "coop"])
            if role == "investor":
                s = s[s["family"] != "Gemini"]  # Gemini sends 5 in all 50 round-1 decisions
            form = f"coop ~ own_z_{f} + C(cell)" + (" + received_now" if role == "trustee" else "")
            m = fit(form, s)
            t = f"own_z_{f}"
            rows.append({"rung": "R4", "feature": f, "outcome": oname, "estimate": "own myth, per within-family SD",
                         "coef": m.params[t], "ci_low": m.conf_int().loc[t, 0], "ci_high": m.conf_int().loc[t, 1],
                         "p": m.pvalues[t], "n": int(m.nobs), "n_runs": s["run_id"].nunique(), "holm": True})
            for fam, sf in s.groupby("family"):
                if sf["run_id"].nunique() < 5 or sf["coop"].std() == 0:
                    continue
                try:
                    mf = fit(form, sf)
                    fam_rows.append({"rung": "R4", "feature": f, "outcome": oname, "family": fam, "coef": mf.params[t],
                                     "ci_low": mf.conf_int().loc[t, 0], "ci_high": mf.conf_int().loc[t, 1],
                                     "p": mf.pvalues[t], "n": int(mf.nobs)})
                except Exception:  # noqa: BLE001
                    pass

    # ---------------------------------------------------------------- R3
    r3 = d[d["split"] == "R3"].copy()
    # author's move toward the reader in round r-1, and whether they had played before r-1
    key = {(a, b, c): (p, co) for a, b, c, p, co in zip(dec_all.run_id, dec_all["round"], dec_all.agent, dec_all.partner, dec_all.coop)}
    partners = dec_all.groupby(["run_id", "agent"]).apply(lambda g: dict(zip(g["round"], g["partner"]))).to_dict()
    auth_move, before = [], []
    for g in r3.itertuples():
        if not isinstance(g.shown_author, str):
            auth_move.append(np.nan); before.append(np.nan); continue
        pr = int(g.shown_round)
        p, co = key.get((g.run_id, pr, g.shown_author), (None, np.nan))
        auth_move.append(co if p == g.agent else np.nan)
        hist = partners.get((g.run_id, g.agent), {})
        before.append(float(any(hist.get(k) == g.shown_author for k in range(1, pr))))
    r3["author_move_prev"], r3["played_before"] = auth_move, before
    # unseen placebo: same run, same round as the shown myth, same family as the author, not reader/author
    mi = myths.assign(i=np.arange(len(myths)))
    grp = {k: g for k, g in mi.groupby(["run_id", "round"])}
    for f in cands:
        zf = feat[f"z_{f}"].to_numpy()
        vals = []
        for g in r3.itertuples():
            if g.shown_i < 0:
                vals.append(np.nan); continue
            pool = grp.get((g.run_id, int(g.shown_round)))
            fam = myths.at[g.shown_i, "family"]
            c = pool[(pool["family"] == fam) & (~pool["agent"].isin([g.agent, g.shown_author]))]["i"].to_numpy()
            vals.append(np.nanmean(zf[c]) if len(c) else np.nan)
        r3[f"unseen_z_{f}"] = vals
    r3 = fill(r3, ["author_move_prev"])
    for f in cands:
        for oname, role, ycol in (("send", "investor", "coop"), ("return", "trustee", "coop"), ("dsend", "investor", "dsend")):
            for sample in ("never played before", "all"):
                s0 = r3[r3["role"] == role].dropna(subset=[ycol, f"shown_z_{f}", f"unseen_z_{f}"])
                if role == "investor":
                    s0 = s0[s0["family"] != "Gemini"]
                if sample == "never played before":
                    s0 = s0[s0["played_before"] == 0]
                ctrl = CTRL + " + author_move_prev_f + author_move_prev_miss" + (" + received_now" if role == "trustee" else "")
                for est, extra in (("total (shown - unseen)", ""), ("direct, own round-r myth held fixed", f" + own_z_{f}")):
                    s = s0.dropna(subset=[f"own_z_{f}"]) if extra else s0
                    form = f"{ycol} ~ shown_z_{f} + unseen_z_{f}{extra} + {ctrl} + C(run_agent) + C(round)"
                    m = fit(form, s)
                    a, b = f"shown_z_{f}", f"unseen_z_{f}"
                    w = m.t_test(f"{a} - {b} = 0")
                    ci = np.asarray(w.conf_int()).ravel()
                    rows.append({"rung": "R3", "feature": f, "outcome": oname, "sample": sample, "estimate": est,
                                 "coef": float(np.asarray(w.effect).ravel()[0]), "ci_low": ci[0], "ci_high": ci[1],
                                 "p": float(np.asarray(w.pvalue).ravel()[0]), "shown_coef": m.params[a],
                                 "shown_p": m.pvalues[a], "unseen_coef": m.params[b], "unseen_p": m.pvalues[b],
                                 "n": int(m.nobs), "n_runs": s["run_id"].nunique(),
                                 "holm": sample == "never played before" and est.startswith("total")})
                    if sample == "never played before" and est.startswith("total"):
                        for fam, sf in s.groupby("family"):
                            try:
                                mf = fit(form, sf)
                                wf = mf.t_test(f"{a} - {b} = 0")
                                cif = np.asarray(wf.conf_int()).ravel()
                                fam_rows.append({"rung": "R3", "feature": f, "outcome": oname, "family": fam,
                                                 "coef": float(np.asarray(wf.effect).ravel()[0]), "ci_low": cif[0],
                                                 "ci_high": cif[1], "p": float(np.asarray(wf.pvalue).ravel()[0]),
                                                 "n": int(mf.nobs)})
                            except Exception:  # noqa: BLE001
                                pass
    res = pd.DataFrame(rows)
    h = res["holm"]
    res.loc[h, "p_holm"] = multipletests(res.loc[h, "p"], method="holm")[1]
    res["n_holm_tests"] = int(h.sum())
    res.to_csv(OUT / "confirm_R3_R4.csv", index=False)
    pd.DataFrame(fam_rows).to_csv(OUT / "confirm_by_family.csv", index=False)
    print("Holm family size:", int(h.sum()))
    print(res[h][["rung", "feature", "outcome", "coef", "ci_low", "ci_high", "p", "p_holm", "n", "n_runs"]].round(4).to_string())


if __name__ == "__main__":
    main()
