#!/usr/bin/env python3
"""Exploratory (overlaps R3 rows, not in the Holm family): does the round-1 myth's send rule, a
random draw within a cell, predict the agent's later sending (mean send over rounds 2-10 and over
rounds 6-10)? myth_game, Sonnet + GPT agents, cell FE, SE by run. Also mean return. Writes r4_persistence.csv."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
d = pd.read_csv("decision_table.csv"); d = d[d.task_order == "myth_game"]
f = pd.read_csv("myth_features.csv")
r1 = f[f["round"] == 1].merge(pd.read_csv("decision_table.csv")[["run_id"]].drop_duplicates(), on="run_id")
r1 = r1[["run_id", "agent", "rule_send_ord", "rule_send_amount"]]
rows = []
for lab, rr in (("rounds 2-10", (2, 10)), ("rounds 6-10", (6, 10))):
    for role, name in (("investor", "mean send/5"), ("trustee", "mean return")):
        g = d[(d.role == role) & d["round"].between(*rr)].groupby(["run_id", "agent", "family", "size", "composition"])["coop"].mean().reset_index()
        g = g.merge(r1, on=["run_id", "agent"]); g = g[g.family != "Gemini"]
        g["cell"] = g["size"].astype(str) + "|" + g.composition + "|" + g.family
        for fam, s in [("Sonnet+GPT", g)] + list(g.groupby("family")):
            s = s.dropna(subset=["rule_send_ord", "coop"])
            m = smf.ols("coop ~ rule_send_ord + C(cell)", s).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.run_id)[0]})
            ci = m.conf_int().loc["rule_send_ord"]
            rows.append({"window": lab, "outcome": name, "family": fam, "slope_per_rule_level": m.params["rule_send_ord"],
                         "lo": ci[0], "hi": ci[1], "p": m.pvalues["rule_send_ord"], "n_agents": len(s), "n_runs": s.run_id.nunique()})
r = pd.DataFrame(rows); r.to_csv("r4_persistence.csv", index=False); print(r.round(4).to_string())
