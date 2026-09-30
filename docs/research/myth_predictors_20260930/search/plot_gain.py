#!/usr/bin/env python3
"""Held-out gain from all myth features vs a same-width noise block (ridge), per outcome x family.
Reads gain_table.csv and r4_predictability.csv; writes gain_upper_bound.png."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np
g = pd.read_csv("gain_table.csv"); g = g[g.learner == "ridge"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, ver in zip(axes, ["raw", "within-agent"]):
    s = g[g.version == ver]
    cells = s[["outcome", "stratum"]].drop_duplicates().values.tolist()
    y = np.arange(len(cells))
    for k, (fs, col, off) in enumerate([("ALL myth features", "#0b5394", -0.15), ("NOISE placebo (same width)", "#999999", 0.15)]):
        r = [s[(s.outcome == o) & (s.stratum == st) & (s.feature_set == fs)].iloc[0] for o, st in cells]
        v = np.array([x.gain for x in r]) * 100; lo = np.array([x.gain_lo for x in r]) * 100; hi = np.array([x.gain_hi for x in r]) * 100
        ax.errorbar(v, y + off, xerr=[v - lo, hi - v], fmt="o", color=col, ms=4, capsize=2, label=fs if ver == "raw" else None)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([f"{o}, {st}" for o, st in cells], fontsize=8)
    ax.set_xlabel("held-out R² gain over behaviour baseline (percentage points)")
    ax.set_title(f"after round 1, {ver}", fontsize=10)
axes[0].legend(fontsize=8, loc="lower left")
r4 = pd.read_csv("r4_predictability.csv").iloc[0]
fig.suptitle(f"Myth text adds nothing once past behaviour is known (95% CI, bootstrap over runs).  "
             f"Round 1, before any play: stated send rule explains {r4.gain*100:.0f}% of send variance.", fontsize=9)
fig.tight_layout(); fig.savefig("gain_upper_bound.png", dpi=180)
