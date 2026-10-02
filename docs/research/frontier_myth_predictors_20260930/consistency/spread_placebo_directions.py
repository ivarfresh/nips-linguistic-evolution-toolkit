#!/usr/bin/env python3
"""Is the embedding-measured consistency spread specific to consistency, or does any direction in embedding
space 'spread' from the shown myth? (new check; the spread.py cons_emb_z model applied to placebo directions)

Directions: the consistency score (consistency anchors minus generosity anchors, as in features.py), the
generosity anchors alone, and 40 random unit directions (seed 0) in all-mpnet-base-v2 space. For each, the
myth's projection is z-scored and the spread.py model (child ~ shown + unseen + future + own previous, round FE
[+ family FE in mixed / cross-family strata], run-clustered) is fitted; we report shown (no future term) and
shown minus future. The consistency direction is compared with the random-direction distribution
(share of random directions with a larger shown-minus-future coefficient).
Writes <dataset>/spread_placebo_directions.csv and <dataset>/spread_placebo_summary.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

from common import DATA, EMB_LOCAL, NAME, OUT, WT
from fe import fe_ols

sys.path.insert(0, str(WT))
os.environ.setdefault("LINGUISTIC_DATASET", NAME)
from analyses.linguistic_uptake import load_myths, null_candidates  # noqa: E402
from features import CONS_ANCHORS, CTRL_ANCHORS  # noqa: E402

myths = load_myths()
emb = np.load(EMB_LOCAL) if EMB_LOCAL is not None else np.load(DATA / "embeddings_mpnet.npy")
from sentence_transformers import SentenceTransformer  # noqa: E402
st = SentenceTransformer("all-mpnet-base-v2")
a = st.encode(CONS_ANCHORS, normalize_embeddings=True).mean(0)
c = st.encode(CTRL_ANCHORS, normalize_embeddings=True).mean(0)
dirs = {"consistency (cons - gen anchors)": a / np.linalg.norm(a) - c / np.linalg.norm(c),
        "generosity anchors": c / np.linalg.norm(c)}
rng = np.random.default_rng(0)
for k in range(40):
    v = rng.standard_normal(emb.shape[1]); dirs[f"random_{k:02d}"] = v / np.linalg.norm(v)
S = pd.DataFrame({k: emb @ v for k, v in dirs.items()})
S = (S - S[myths.valid].mean()) / S[myths.valid].std()
S[~myths.valid.to_numpy()] = np.nan
cands = null_candidates(myths)
idx = {(r, t, a_): i for i, (r, t, a_) in enumerate(zip(myths.run_id, myths["round"], myths.agent))}
prev = {(r, t, a_): idx.get((r, t - 1, a_)) for (r, t, a_) in idx}
rows = []
for i, js in cands.items():
    p, nulls = js[0], js[1:]
    ci = myths.loc[i]
    fut = idx.get((ci.run_id, ci["round"], ci.exposed_author))
    own = prev[(ci.run_id, ci["round"], ci.agent)]
    rows.append((i, p, nulls, fut, own, ci["size"], ci.mixed, ci.family, myths.at[p, "family"], ci["round"], ci.run_id))
Sv = S.to_numpy()
child = np.array([Sv[r[0]] for r in rows]); shown = np.array([Sv[r[1]] for r in rows])
unseen = np.array([np.nanmean(Sv[r[2]], axis=0) for r in rows])
future = np.array([Sv[r[3]] if r[3] is not None else np.full(Sv.shape[1], np.nan) for r in rows])
ownp = np.array([Sv[r[4]] if r[4] is not None else np.full(Sv.shape[1], np.nan) for r in rows])
meta = pd.DataFrame([r[5:] for r in rows], columns=["size", "mixed", "family", "parent_family", "round", "run_id"])
meta["stratum"] = np.where(~meta["mixed"], meta["size"].astype(str) + "-agent homogeneous",
                           meta["size"].astype(str) + "-agent mixed, " + np.where(meta.family == meta.parent_family, "same family", "other family"))
strata = list(meta.groupby("stratum").groups.items())
for pf in sorted(meta.parent_family.unique()):
    strata.append((f"cross-family, {pf} parent -> other-family child", meta.index[(meta.parent_family == pf) & (meta.family != pf)]))
out = []
for j, name in enumerate(S.columns):
    base = meta.assign(child=child[:, j], shown=shown[:, j], unseen=unseen[:, j], future=future[:, j], own_prev=ownp[:, j])
    for sname, ix in strata:
        s = base.loc[ix].dropna(subset=["child", "shown", "unseen", "future", "own_prev"])
        dm = ["round", "family"] if ("mixed" in sname or "cross" in sname) else ["round"]
        r0 = fe_ols(s, "child", ["shown", "unseen", "own_prev"], absorb=None, dummies=dm)
        s = s.assign(half=(s.shown - s.future) / 2, sum_sf=(s.shown + s.future) / 2)
        r1 = fe_ols(s, "child", ["half", "sum_sf", "unseen", "own_prev"], absorb=None, dummies=dm)
        out.append({"direction": name, "stratum": sname, "shown_no_future": r0["shown"][0], "p_shown": r0["shown"][3],
                    "shown_minus_future": r1["half"][0], "smf_ci_low": r1["half"][1], "smf_ci_high": r1["half"][2],
                    "p_smf": r1["half"][3], "n": r1["n"], "n_runs": r1["n_runs"]})
t = pd.DataFrame(out)
t.round(4).to_csv(OUT / "spread_placebo_directions.csv", index=False)
summ = []
for sname, g in t.groupby("stratum"):
    rnd = g[g.direction.str.startswith("random")]
    for dname in ["consistency (cons - gen anchors)", "generosity anchors"]:
        x = g[g.direction == dname].iloc[0]
        summ.append({"stratum": sname, "direction": dname, "shown_minus_future": x.shown_minus_future, "p_smf": x.p_smf,
                     "random_median_smf": rnd.shown_minus_future.median(), "random_q90_smf": rnd.shown_minus_future.quantile(0.9),
                     "share_random_larger_smf": (rnd.shown_minus_future >= x.shown_minus_future).mean(),
                     "random_share_p_lt_05": (rnd.p_smf < 0.05).mean(),
                     "shown_no_future": x.shown_no_future, "random_median_shown": rnd.shown_no_future.median(),
                     "share_random_larger_shown": (rnd.shown_no_future >= x.shown_no_future).mean(), "n_runs": x.n_runs})
u = pd.DataFrame(summ).round(4)
u.to_csv(OUT / "spread_placebo_summary.csv", index=False)
pd.set_option("display.width", 250)
print(u.to_string(index=False))
