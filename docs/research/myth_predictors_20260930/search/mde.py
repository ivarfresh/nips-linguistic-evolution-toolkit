#!/usr/bin/env python3
"""Minimum detectable effect for the 'all myth features' upper bound. A planted column
(baseline residual + noise, scaled so it alone would add a target R2) is put among the real
~145 myth columns (incl. embedding PCs); we record whether the two-stage ridge recovers it and its
bootstrap CI excludes 0. Planted signal = residual of the out-of-fold baseline prediction plus noise,
so its incremental value over the baseline is known. Writes mde_calibration.csv."""
import numpy as np, pandas as pd
import search as S
d = pd.read_csv("decision_table.csv")
d = d[(d.split == "discovery") & (d.own_i >= 0) & (d.shown_i >= 0)].reset_index(drop=True)
rows = []
rng = np.random.default_rng(3)
for oname, role in (("send", "investor"), ("return", "trustee")):
    sub = d[d.role == role].dropna(subset=["coop"]).reset_index(drop=True).copy()
    cols, embs = S.feature_sets(sub)["ALL myth features"]
    for within in (False, True):
        F = S.Fitter(sub, "coop", within)
        p0, y, _ = F.oof([], [], "ridge")
        resid = y - p0
        for target in (0.005, 0.01, 0.02, 0.05):
            # corr^2 of planted column with the residual ~ share of residual variance explained
            share = target / (1 - S.r2(y, p0))
            s = np.sqrt(np.var(resid) * (1 - share) / share)
            for rep in range(3):
                sub["planted"] = resid + rng.normal(scale=s, size=len(sub)) if not within else 0
                if within:
                    # plant in raw units; the within transform demeans it like every other column
                    sub["planted"] = (sub["coop"] - sub.groupby("run_agent")["coop"].transform("mean")) * 0 + resid + rng.normal(scale=s, size=len(sub))
                F.df = sub
                alone, _, _ = F.oof(["planted"], [], "ridge")
                full, _, _ = F.oof(cols + ["planted"], embs, "ridge")
                b = S.boot_gain(y, p0, full, F.groups, n=300)
                rows.append({"outcome": oname, "version": "within-agent" if within else "raw", "target_gain": target,
                             "rep": rep, "gain_alone": S.r2(y, alone) - S.r2(y, p0), "gain_in_full_set": b["gain"],
                             "ci_lo": b["gain_lo"], "detected": b["gain_lo"] > 0})
                print(rows[-1], flush=True)
r = pd.DataFrame(rows); r.to_csv("mde_calibration.csv", index=False)
print(r.groupby(["outcome", "version", "target_gain"])[["gain_alone", "gain_in_full_set", "detected"]].mean().round(4))
