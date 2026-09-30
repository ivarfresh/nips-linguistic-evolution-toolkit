#!/usr/bin/env python3
"""Is embedding self-copying specific to consistency? Own-previous coefficient (cell + round FE, own/shown/unseen
previous as in ratchet_continuous.py) for the consistency direction vs 20 random unit directions (seed 0), per family
over all settings. Writes <dataset>/ratchet_placebo_directions.csv."""
import numpy as np
import pandas as pd

from common import DATA, EMB_LOCAL, OUT
from fe import fe_ols

m = pd.read_csv(OUT / "myth_features.csv")
emb = np.load(EMB_LOCAL) if EMB_LOCAL is not None else np.load(DATA / "embeddings_mpnet.npy")
rng = np.random.default_rng(0)
S = {"consistency": m["cons_emb"].to_numpy()}
for k in range(20):
    v = rng.standard_normal(emb.shape[1]); S[f"random_{k:02d}"] = emb @ (v / np.linalg.norm(v))
S = pd.DataFrame(S)
S = (S - S[m.valid].mean()) / S[m.valid].std()
S[~m.valid.to_numpy()] = np.nan
m["cell"] = m["composition"] + "|" + m["task_order"]
idx = {(r, t, a): i for i, (r, t, a) in enumerate(zip(m.run_id, m["round"], m.agent))}
v = m[m.valid]
brr = {k: g.index.to_list() for k, g in v.groupby(["run_id", "round", "family"])}
bcr = {k: g.index.to_list() for k, g in v.groupby(["cell", "round", "family"])}
rec = []
for i, r in m[(m["round"] >= 2) & m.valid].iterrows():
    own = idx.get((r.run_id, r["round"] - 1, r.agent))
    sh = idx.get((r.run_id, r.exposed_round, r.exposed_author)) if isinstance(r.exposed_author, str) else None
    c = ([j for j in brr.get((r.run_id, r["round"] - 1, r.family), []) if m.at[j, "agent"] not in (r.agent, r.exposed_author)]
         if r["size"] == 8 else [j for j in bcr.get((r.cell, r["round"] - 1, r.family), []) if m.at[j, "run_id"] != r.run_id])
    if own is None or sh is None or not c:
        continue
    rec.append((i, own, sh, c))
rows = []
for col in S.columns:
    x = S[col].to_numpy()
    d = pd.DataFrame({"y": [x[a] for a, *_ in rec], "own": [x[o] for _, o, _, _ in rec], "shown": [x[s] for _, _, s, _ in rec],
                      "unseen": [np.nanmean(x[c]) for *_, c in rec]})
    d = pd.concat([d, m.loc[[a for a, *_ in rec], ["run_id", "cell", "round", "family"]].reset_index(drop=True)], axis=1)
    for fam, g in d.groupby("family"):
        res = fe_ols(g, "y", ["own", "shown", "unseen"], absorb="cell", dummies=["round"])
        rows.append({"direction": col, "family": fam, "own_prev": res["own"][0], "shown_prev": res["shown"][0], "unseen_prev": res["unseen"][0]})
t = pd.DataFrame(rows)
summ = []
for fam, g in t.groupby("family"):
    c = g[g.direction == "consistency"].iloc[0]; r = g[g.direction != "consistency"]
    summ.append({"family": fam, "consistency_own_prev": c.own_prev, "random_own_prev_median": r.own_prev.median(),
                 "random_own_prev_min": r.own_prev.min(), "random_own_prev_max": r.own_prev.max(),
                 "share_random_larger": (r.own_prev >= c.own_prev).mean(),
                 "consistency_shown_prev": c.shown_prev, "random_shown_prev_median": r.shown_prev.median()})
u = pd.DataFrame(summ).round(3); u.to_csv(OUT / "ratchet_placebo_directions.csv", index=False); print(u.to_string(index=False))
