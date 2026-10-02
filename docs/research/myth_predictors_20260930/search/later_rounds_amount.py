#!/usr/bin/env python3
"""Exploratory (not in the Holm family): after round 1, does the own myth's stated amount still
carry information beyond the agent's last send? myth_game only (myth written before the decision),
Sonnet + GPT investors, rounds >= 2. Writes later_rounds_amount.csv."""
import warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
d = pd.read_csv("decision_table.csv")
d = d[(d.task_order == "myth_game") & (d.role == "investor") & (d.family != "Gemini") & (d["round"] >= 2)].dropna(subset=["own_rule_send_amount", "lag_send"])
d["send"], d["lag_send_d"] = d["sent"], d["lag_send"] * 5
d["run_agent"] = d.run_id + "|" + d.agent
rows = []
for name, s in [("all", d)] + list(d.groupby("family")):
    rec = {"sample": name, "n": len(s), "n_runs": s.run_id.nunique(),
           "exact_match_amount": float(np.isclose(s.send, s.own_rule_send_amount).mean()),
           "exact_match_lag": float(np.isclose(s.send, s.lag_send_d).mean()),
           "amount_equals_lag": float(np.isclose(s.own_rule_send_amount, s.lag_send_d).mean())}
    for spec, f in [("amount only", "send ~ own_rule_send_amount + C(round)"),
                    ("amount + last send", "send ~ own_rule_send_amount + lag_send_d + C(round)"),
                    ("amount + last send, agent FE", "send ~ own_rule_send_amount + lag_send_d + C(run_agent) + C(round)")]:
        m = smf.ols(f, s).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.run_id)[0]})
        ci = m.conf_int().loc["own_rule_send_amount"]
        rows.append({**rec, "spec": spec, "slope": m.params["own_rule_send_amount"], "lo": ci[0], "hi": ci[1],
                     "p": m.pvalues["own_rule_send_amount"],
                     "lag_slope": m.params.get("lag_send_d", np.nan)})
r = pd.DataFrame(rows); r.to_csv("later_rounds_amount.csv", index=False); print(r.round(3).to_string())
