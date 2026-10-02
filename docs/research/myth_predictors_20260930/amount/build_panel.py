#!/usr/bin/env python3
"""Build the decision-level panel for the send-amount lens.

One row per decision (investor or trustee). Attaches:
  own_*       the agent's latest own myth before the decision (myth_game: same round; game_myth: previous round)
  own_prev_*  the agent's own myth before that one (pre-exposure baseline for the latest own myth)
  shown_*     the myth the agent was shown just before writing its latest own myth
  unseen_*    mean over comparable UNSEEN myths: same run, same round as the shown myth, same family as the
              shown author, written by neither the reader nor the shown author (8-agent only)
  lag_*       the agent's own previous move in the same role
  b_coop_prev what the shown author did to the reader in their last game together before the decision
  c_coop_prev the current partner's move in its previous game (the reader sees this in its visible history)
  contact     1 if the shown author had played the reader before the shown myth was written
Writes panel.pkl / panel.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA, OUT, load_decisions

FEATS = ["judge_amount", "judge_amount_strict", "rx_amount", "rx_concrete", "judge_return", "generous"]

m = pd.read_pickle(OUT / "myths_features.pkl")
lab = pd.read_csv(DATA / "moral_labels_z-ai__glm-5.2.csv")[["run_id", "round", "agent", "label"]]
m = m.merge(lab, on=["run_id", "round", "agent"], how="left")
m["generous"] = (m["label"] == "be generous").astype(float).where(m["label"].notna())
d = load_decisions()

# ---- pairings and partner moves
pair = d[["run_id", "round", "agent", "partner", "coop", "role"]]
contact_rounds = pair.groupby(["run_id", "agent", "partner"])["round"].apply(list).to_dict()

# own latest myth before each decision
d["own_round"] = np.where(d["task_order"] == "myth_game", d["round"], d["round"] - 1)
mk = m.set_index(["run_id", "round", "agent"])
def attach(frame, rcol, acol, prefix, cols):
    idx = pd.MultiIndex.from_arrays([frame["run_id"], frame[rcol], frame[acol]])
    sub = mk.reindex(idx)[cols]
    sub.columns = [f"{prefix}{c}" for c in cols]
    return pd.concat([frame.reset_index(drop=True), sub.reset_index(drop=True)], axis=1)

d = attach(d, "own_round", "agent", "own_", FEATS + ["exposed_author", "exposed_round", "exposed_family"])
d["own_prev_round"] = d["own_round"] - 1
d = attach(d, "own_prev_round", "agent", "own_prev_", FEATS)
d = d.rename(columns={"own_exposed_author": "shown_author", "own_exposed_round": "shown_round",
                      "own_exposed_family": "shown_family"})
d = attach(d, "shown_round", "shown_author", "shown_", FEATS)

# unseen comparison myths (8-agent: same run/round/family, not reader, not shown author)
fam_of = m.set_index(["run_id", "agent"])["family"].to_dict()
grp = {k: g for k, g in m.groupby(["run_id", "round"])}
unseen = {f: [] for f in FEATS}
n_unseen = []
for row in d.itertuples():
    ok = False
    if row.size == 8 and isinstance(row.shown_author, str):
        g = grp.get((row.run_id, row.shown_round))
        c = g[(g["family"] == row.shown_family) & (~g["agent"].isin([row.agent, row.shown_author]))]
        if len(c):
            ok = True
            for f in FEATS:
                unseen[f].append(c[f].mean())
            n_unseen.append(len(c))
    if not ok:
        for f in FEATS:
            unseen[f].append(np.nan)
        n_unseen.append(0)
for f in FEATS:
    d[f"unseen_{f}"] = unseen[f]
d["n_unseen"] = n_unseen

# prior contact between shown author and reader before the shown myth was written
def contact(row):
    if not isinstance(row.shown_author, str):
        return np.nan
    rounds = contact_rounds.get((row.run_id, row.agent, row.shown_author), [])
    # myth_game: the author's round-t myth precedes its round-t game; game_myth: follows it
    lim = row.shown_round if row.task_order == "game_myth" else row.shown_round - 1
    return float(any(r <= lim for r in rounds))
d["contact"] = [contact(r) for r in d.itertuples()]

# what the shown author did to the reader in their latest game before the decision
pk = pair.set_index(["run_id", "agent", "partner", "round"])["coop"]
def b_prev(row):
    if not isinstance(row.shown_author, str):
        return np.nan
    rounds = [r for r in contact_rounds.get((row.run_id, row.agent, row.shown_author), []) if r < row.round]
    if not rounds:
        return np.nan
    return pk.get((row.run_id, row.shown_author, row.agent, max(rounds)), np.nan)
d["b_coop_prev"] = [b_prev(r) for r in d.itertuples()]
d["b_played_prev_round"] = [float(any(r == row.round - 1 for r in contact_rounds.get((row.run_id, row.agent, row.shown_author), [])))
                            if isinstance(row.shown_author, str) else np.nan for row in d.itertuples()]

# own lagged move in the same role, and partner's previous move (any role)
d = d.sort_values(["run_id", "agent", "role", "round"])
d["lag_coop_same_role"] = d.groupby(["run_id", "agent", "role"])["coop"].shift()
d["lag_sent"] = d.groupby(["run_id", "agent", "role"])["sent"].shift().where(d["role"] == "investor")
d = d.sort_values(["run_id", "agent", "round"])
d["lag_coop_any"] = d.groupby(["run_id", "agent"])["coop"].shift()
prevc = d.set_index(["run_id", "agent", "round"])["coop"]
d["c_coop_prev"] = [prevc.get((r.run_id, r.partner, r.round - 1), np.nan) for r in d.itertuples()]
d["send5"] = d["sent"].where(d["role"] == "investor")
d.to_pickle(OUT / "panel.pkl")
d.to_csv(OUT / "panel.csv", index=False)
inv = d[d.role == "investor"]
print(len(d), "decisions;", inv.shown_judge_amount.notna().sum(), "investor decisions with a shown myth;",
      inv.unseen_judge_amount.notna().sum(), "with an unseen comparison")
print(inv[(inv["size"] == 8) & (inv.task_order == "myth_game")].groupby("round")[["contact", "b_played_prev_round"]].mean())
