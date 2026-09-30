#!/usr/bin/env python3
"""R3 robustness per the lead (2026-09-30): 8-agent myth_game, rounds >= 2, shown author != current
partner; control for the author's move toward the reader last round; placebo = the shown author's
NEXT myth (round r, written at the same time as the reader's, not visible to it). Test = shown -
future. Same 5 frozen candidates; reported alongside, not re-Holm'd with the main family (a
separate Holm over these 13 tests is given). Writes r3_future_placebo.csv."""
import warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")
D = "/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/linguistic-mixed/data/analysis/linguistic_20260923/"
cands = pd.read_csv("candidates_top5.csv")["feature"].tolist()
myths = pd.read_csv(D + "myths.csv"); feat = pd.read_csv("myth_features.csv")
idx = {(r, t, a): i for i, (r, t, a) in enumerate(zip(myths.run_id, myths["round"], myths.agent))}
d = pd.read_csv("decision_table.csv")
key = {(a, b, c): (p, co) for a, b, c, p, co in zip(d.run_id, d["round"], d.agent, d.partner, d.coop)}
r3 = d[(d.split == "R3") & (d.shown_i >= 0)].copy()
r3 = r3[r3.shown_author != r3.partner]
r3["author_move_prev"] = [key.get((g.run_id, int(g.shown_round), g.shown_author), (None, np.nan))[1] for g in r3.itertuples()]
r3["future_i"] = [idx.get((g.run_id, int(g.shown_round) + 1, g.shown_author), -1) for g in r3.itertuples()]
for c in ["lag_send", "lag_return", "lag_got_return", "lag_got_send", "cop_last3_send", "cop_last3_return", "author_move_prev"]:
    r3[c + "_m"] = r3[c].isna().astype(float); r3[c + "_f"] = r3[c].fillna(r3[c].mean())
r3["run_agent"] = r3.run_id + "|" + r3.agent
ctrl = " + ".join(c + "_f + " + c + "_m" for c in ["lag_send", "lag_return", "lag_got_return", "lag_got_send", "cop_last3_send", "cop_last3_return", "author_move_prev"])
rows = []
for f in cands:
    z = ((feat[f] - feat.groupby("family")[f].transform("mean")) / feat.groupby("family")[f].transform("std")).to_numpy()
    for src in ("shown", "future"):
        ii = r3[f"{src}_i"].to_numpy(); r3[f"{src}_z"] = np.where(ii >= 0, z[np.clip(ii, 0, None)], np.nan)
    for oname, role, y in (("send", "investor", "coop"), ("return", "trustee", "coop"), ("dsend", "investor", "dsend")):
        s = r3[r3.role == role].dropna(subset=[y, "shown_z", "future_z"])
        if role == "investor": s = s[s.family != "Gemini"]
        form = f"{y} ~ shown_z + future_z + {ctrl}" + (" + received_now" if role == "trustee" else "") + " + C(run_agent) + C(round)"
        m = smf.ols(form, s).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.run_id)[0]})
        w = m.t_test("shown_z - future_z = 0"); ci = np.asarray(w.conf_int()).ravel()
        rows.append({"feature": f, "outcome": oname, "shown_minus_future": float(np.asarray(w.effect).ravel()[0]),
                     "lo": ci[0], "hi": ci[1], "p": float(np.asarray(w.pvalue).ravel()[0]),
                     "shown_coef": m.params["shown_z"], "future_coef": m.params["future_z"], "n": int(m.nobs), "n_runs": s.run_id.nunique()})
r = pd.DataFrame(rows); r["p_holm_within_this_table"] = multipletests(r.p, method="holm")[1]
r.to_csv("r3_future_placebo.csv", index=False); print(r.round(4).to_string())
