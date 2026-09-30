#!/usr/bin/env python3
"""September beside frontier: held-out R2 gain from all myth features vs a same-width noise block
(ridge, raw and within-agent), per outcome x stratum; ceiling strata listed as not estimable.
Right-most panel: round-1 (R4) gain from the stated amount + send rule. Writes gain_upper_bound.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import HERE, SEPT_SEARCH

SRC = {"September": (SEPT_SEARCH / "gain_table.csv", SEPT_SEARCH / "r4_predictability.csv"),
       "Frontier": (HERE / "frontier/gain_table.csv", HERE / "frontier/r4_predictability.csv")}
fig, axes = plt.subplots(2, 3, figsize=(14, 9), gridspec_kw={"width_ratios": [3, 3, 1.3]})
for row, (ds, (gp, rp)) in enumerate(SRC.items()):
    g = pd.read_csv(gp)
    fitted = g[g.get("status", pd.Series("fitted", index=g.index)).fillna("fitted") == "fitted"]
    fitted = fitted[fitted.learner == "ridge"]
    for ax, ver in zip(axes[row, :2], ["raw", "within-agent"]):
        s = fitted[fitted.version == ver]
        cells = s[["outcome", "stratum"]].drop_duplicates().values.tolist()
        y = np.arange(len(cells))
        for fs, col, off in [("ALL myth features", "#0b5394", -0.15), ("NOISE placebo (same width)", "#999999", 0.15)]:
            r = [s[(s.outcome == o) & (s.stratum == st) & (s.feature_set == fs)].iloc[0] for o, st in cells]
            v = np.array([x.gain for x in r]) * 100
            lo, hi = np.array([x.gain_lo for x in r]) * 100, np.array([x.gain_hi for x in r]) * 100
            ax.errorbar(v, y + off, xerr=[v - lo, hi - v], fmt="o", color=col, ms=4, capsize=2,
                        label=fs if (ver == "raw" and row == 0) else None)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{o}, {st}" for o, st in cells], fontsize=7)
        ax.set_xlabel("held-out R² gain over behaviour baseline (points)", fontsize=8)
        ax.set_title(f"{ds}: after round 1, {ver}", fontsize=10)
    if "status" in g:
        ne = g[g.status.fillna("fitted") != "fitted"][["outcome", "stratum", "status"]].drop_duplicates()
        if len(ne):
            axes[row, 1].text(1.02, 0.0, "not fitted:\n" + "\n".join(f"{a}, {b}: {c}" for a, b, c in ne.values),
                              transform=axes[row, 1].transAxes, fontsize=6, va="bottom")
    r4 = pd.read_csv(rp)
    r4 = r4[r4.model == "stated amount + send rule"].iloc[0]
    ax = axes[row, 2]
    ax.errorbar([0], [r4.gain * 100], yerr=[[r4.gain * 100 - r4.gain_lo * 100], [r4.gain_hi * 100 - r4.gain * 100]],
                fmt="o", color="#b45f06", capsize=3)
    ax.set_xticks([0])
    ax.set_xticklabels([f"round 1\n{r4.families if 'families' in r4 else 'Sonnet+GPT'}\n{int(r4.n)} senders"], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("held-out R² gain (points)", fontsize=8)
    ax.set_title(f"{ds}:\nown stated amount", fontsize=9)
axes[0, 0].legend(fontsize=8, loc="lower left")
fig.suptitle("Before play, an agent's own myth sets its first send (right). After round 1, myth features add nothing to "
             "predicting sends in either dataset;\nfrontier Opus return share shows a small uncorrected gain from the return rule its "
             "own myth states (between agents only, not confirmed at round 1). 95% CI, bootstrap over runs.", fontsize=10)
fig.tight_layout()
fig.savefig(HERE / "gain_upper_bound.png", dpi=160)
