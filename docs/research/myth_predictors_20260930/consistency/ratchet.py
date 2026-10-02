#!/usr/bin/env python3
"""Self-persistence of the consistency keyword: P(use at round r | own myth at r-1 used it / did not).
The later-round myth prompt says 'Use the myth you wrote in the previous round as inspiration'. Writes ratchet.csv."""
import pandas as pd
m = pd.read_csv("myth_features.csv"); m = m[m.valid].sort_values(["run_id", "agent", "round"])
m["prev"] = m.groupby(["run_id", "agent"])["cons_lex"].shift(1)
m = m.dropna(subset=["prev"])
run = m.groupby(["family", "prev", "run_id"])["cons_lex"].mean().reset_index()
t = run.groupby(["family", "prev"])["cons_lex"].agg(mean="mean", sd="std", n_runs="size").reset_index()
t["n_myths"] = m.groupby(["family", "prev"]).size().to_numpy()
t.round(3).to_csv("ratchet.csv", index=False); print(t.round(3).to_string(index=False))
