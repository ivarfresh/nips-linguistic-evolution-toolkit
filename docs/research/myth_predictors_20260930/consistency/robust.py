#!/usr/bin/env python3
"""Robustness checks on the few non-null associations in predict_r12 / predict_lockin / predict_r4.
  (1) stability results with the agent's own previous |change| as a control (volatile phases persist,
      and volatile agents write less about consistency: drift_describes_play.csv)
  (2) R4 round-1 keyword result with run FE instead of cell FE (only 7 Sonnet agents use the word at round 1)
Writes robust_checks.csv."""
import numpy as np
import pandas as pd
from fe import fe_ols, rows_from

d = pd.read_csv("decision_table.csv")
d["absdelta_lag"] = d.groupby(["run_id", "agent", "role"])["absdelta"].shift(1)
rows = []
for fam in ["Sonnet", "GPT", "all"]:
    for role in ["investor", "trustee"]:
        g = d[(d.role == role) & ((d.family == fam) | (fam == "all"))]
        for f in ["own_cons_judge", "own_cons_lex", "own_cons_emb_z"]:
            for ctrl, xs in [("lag coop", [f, "coop_lag"]), ("lag coop + lag |change|", [f, "coop_lag", "absdelta_lag"])]:
                s = g.dropna(subset=["absdelta_lag"])  # same sample for both specs
                res = fe_ols(s, "absdelta", xs, absorb="run_agent", dummies=["round"])
                rows += rows_from(res, [f], check="R2 |change| ~ own consistency", controls=ctrl, family=fam, role=role, feature=f)
            d2 = g.assign(lagX=g["coop_lag"] * g[f])
            s = d2.dropna(subset=["absdelta_lag"])
            res = fe_ols(s, "coop", ["coop_lag", f, "lagX", "absdelta_lag"], absorb="run_agent", dummies=["round"])
            rows += rows_from(res, ["lagX"], check="R2 persistence coop ~ lag x consistency", controls="agent FE + lag |change|",
                              family=fam, role=role, feature=f)
        # cut after letdown with previous-cut control is too sparse; report the raw counts instead
a = pd.read_csv("predict_r4_agents.csv")
for fam in ["Sonnet", "GPT", "all"]:
    s = a if fam == "all" else a[a.family == fam]
    for y in ["coop", "mean_coop"]:
        for f in ["own_cons_lex", "own_cons_judge"]:
            try:
                res = fe_ols(s, y, [f], absorb="run_id", dummies=["role"])
                rows += rows_from(res, [f], check=f"R4 {y} ~ round-1 consistency, run FE", controls="run FE", family=fam, role="both")
            except Exception as e:
                print(e)
t = pd.DataFrame(rows).round(4)
t.to_csv("robust_checks.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(t[["check", "controls", "family", "role", "term", "coef", "ci_low", "ci_high", "p", "n", "n_runs"]].to_string(index=False))
