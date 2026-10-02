#!/usr/bin/env python3
"""Minimum detectable effect for the 'all myth features' upper bound, per stratum (ported from the
September search lens; MP_DATASET=september|frontier). A planted column (out-of-fold baseline
residual + noise, scaled so it alone would add a target R2) is put among the real myth columns
(incl. embedding PCs); we record whether the two-stage ridge recovers it and whether the bootstrap
CI excludes 0. Frontier has fewer rows, so the targets reach 0.10. Writes mde_calibration.csv."""
import sys

import numpy as np
import pandas as pd

import search as S
from common import NAME, OUT, ceiling

STRATA = {"september": [("pooled", None)],
          "frontier": [("pooled", None), ("Sol", "Sol"), ("Opus", "Opus")]}[NAME]
TARGETS = (0.005, 0.01, 0.02, 0.05) if NAME == "september" else (0.01, 0.02, 0.05, 0.10)
d = pd.read_csv(OUT / "decision_table.csv")
d = d[(d.split == "discovery") & (d.own_i >= 0) & (d.shown_i >= 0)].reset_index(drop=True)
rows = []
rng = np.random.default_rng(3)
for sname, fam in STRATA:
    for oname, role in (("send", "investor"), ("return", "trustee")):
        sub = d[d.role == role].dropna(subset=["coop"])
        if fam:
            sub = sub[sub.family == fam]
        sub = sub.reset_index(drop=True).copy()
        if fam and ceiling(sub["coop"]):
            continue
        cols, embs = S.feature_sets(sub)["ALL myth features"]
        for within in (False, True):
            F = S.Fitter(sub, "coop", within)
            p0, y, _ = F.oof([], [], "ridge")
            resid = y - p0
            for target in TARGETS:
                share = target / (1 - S.r2(y, p0))
                s = np.sqrt(np.var(resid) * (1 - share) / share)
                for rep in range(3):
                    sub["planted"] = resid + rng.normal(scale=s, size=len(sub))
                    F.df = sub
                    alone, _, _ = F.oof(["planted"], [], "ridge")
                    full, _, _ = F.oof(cols + ["planted"], embs, "ridge")
                    b = S.boot_gain(y, p0, full, F.groups, n=300)
                    rows.append({"dataset": NAME, "stratum": sname, "outcome": oname, "n": len(y),
                                 "n_runs": sub.run_id.nunique(), "version": "within-agent" if within else "raw",
                                 "target_gain": target, "rep": rep, "gain_alone": S.r2(y, alone) - S.r2(y, p0),
                                 "gain_in_full_set": b["gain"], "ci_lo": b["gain_lo"], "detected": b["gain_lo"] > 0})
                    print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(OUT / "mde_calibration.csv", index=False)
r = pd.DataFrame(rows)
r.to_csv(OUT / "mde_calibration.csv", index=False)
print(r.groupby(["stratum", "outcome", "version", "target_gain"])[["gain_alone", "gain_in_full_set", "detected"]]
      .mean().round(4).to_string())
