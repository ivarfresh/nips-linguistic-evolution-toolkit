#!/usr/bin/env python3
"""Is the embedding drift specific to consistency? Round-1 vs rounds 8-10 change (run-level mean, in sd units of the
projection) for the consistency direction vs the generosity anchors, the consistency anchors alone, and 40 random unit directions (seed 0).
Reports |change| of consistency against the distribution of |change| of random directions, per family x setting.
Writes <dataset>/drift_placebo_directions.csv."""
import numpy as np
import pandas as pd

from common import DATA, EMB_LOCAL, OUT
from features import CONS_ANCHORS, CTRL_ANCHORS

f = pd.read_csv(OUT / "myth_features.csv")
emb = np.load(EMB_LOCAL) if EMB_LOCAL is not None else np.load(DATA / "embeddings_mpnet.npy")
from sentence_transformers import SentenceTransformer  # noqa: E402
st = SentenceTransformer("all-mpnet-base-v2")
a = st.encode(CONS_ANCHORS, normalize_embeddings=True).mean(0)
c = st.encode(CTRL_ANCHORS, normalize_embeddings=True).mean(0)
dirs = {"consistency": a / np.linalg.norm(a) - c / np.linalg.norm(c), "generosity": c / np.linalg.norm(c),
        "consistency_anchors_only": a / np.linalg.norm(a)}
rng = np.random.default_rng(0)
for k in range(40):
    v = rng.standard_normal(emb.shape[1]); dirs[f"random_{k:02d}"] = v / np.linalg.norm(v)
S = pd.DataFrame({k: emb @ v for k, v in dirs.items()})
S = (S - S[f.valid].mean()) / S[f.valid].std()
d = pd.concat([f[["run_id", "setting", "family", "round", "valid"]], S], axis=1)
d = d[d.valid & ((d["round"] == 1) | (d["round"] >= 8))].assign(late=lambda x: x["round"] >= 8)
run = d.groupby(["setting", "family", "run_id", "late"])[list(dirs)].mean().unstack("late")
rows = []
for (st_, fam), g in list(run.groupby(level=[0, 1])) + [(("all settings", fam), g) for fam, g in run.groupby(level=1)]:
    diff = pd.DataFrame({k: g[(k, True)] - g[(k, False)] for k in dirs}).mean()
    rnd = diff[[k for k in dirs if k.startswith("random")]].abs()
    rows.append({"setting": st_, "family": fam, "consistency_change_sd": diff["consistency"], "generosity_change_sd": diff["generosity"],
                 "cons_anchors_only_change_sd": diff["consistency_anchors_only"],
                 "share_random_abs_larger_anchors_only": (rnd >= abs(diff["consistency_anchors_only"])).mean(),
                 "random_abs_change_median": rnd.median(), "random_abs_change_max": rnd.max(),
                 "share_random_abs_larger": (rnd >= abs(diff["consistency"])).mean(), "n_runs": len(g)})
t = pd.DataFrame(rows).round(3)
t.to_csv(OUT / "drift_placebo_directions.csv", index=False)
print(t.to_string(index=False))
