#!/usr/bin/env python3
"""Self-persistence of the consistency keyword: P(use at round r | own myth at r-1 used it / did not).
The later-round myth prompt says 'Use the myth you wrote in the previous round as inspiration'.
Port; LINGUISTIC_DATASET=september|frontier. Writes <dataset>/ratchet.csv (by family) and
<dataset>/ratchet_by_stratum.csv (family x setting x task order), plus the same for the judge flag."""
import pandas as pd

from common import OUT

m = pd.read_csv(OUT / "myth_features.csv"); m = m[m.valid].sort_values(["run_id", "agent", "round"])
out, strat = [], []
for k in ["cons_lex", "cons_judge"]:
    x = m.assign(prev=m.groupby(["run_id", "agent"])[k].shift(1)).dropna(subset=["prev", k])
    for keys, dest in [(["family"], out), (["family", "setting", "task_order"], strat)]:
        run = x.groupby(keys + ["prev", "run_id"])[k].mean().reset_index()
        t = run.groupby(keys + ["prev"])[k].agg(mean="mean", sd="std", n_runs="size").reset_index()
        t = t.merge(x.groupby(keys + ["prev"]).size().rename("n_myths").reset_index(), on=keys + ["prev"])
        dest.append(t.assign(measure=k))
t = pd.concat(out).round(3); t.to_csv(OUT / "ratchet.csv", index=False); print(t.to_string(index=False))
pd.concat(strat).round(3).to_csv(OUT / "ratchet_by_stratum.csv", index=False)
