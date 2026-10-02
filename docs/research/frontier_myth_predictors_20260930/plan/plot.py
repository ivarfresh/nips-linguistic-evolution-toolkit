#!/usr/bin/env python3
"""Figures: r4_plan_scatter.png (own round-1 myth's named amount vs round-1 send, September | frontier)
and key_estimates.png (the lens's main estimates, September beside frontier). Run after analyze.py."""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build import CACHE
from common import OUT

# family colours from analyses/linguistic_datasets.py (same slot per role across datasets)
COL = {"Sonnet": "#7570b3", "GPT": "#d95f02", "Gemini": "#1b9e77", "Opus": "#7570b3", "Sol": "#d95f02", "GeminiPro": "#1b9e77"}
MARK = {"Sonnet": "o", "Opus": "o", "GPT": "s", "Sol": "s", "Gemini": "^", "GeminiPro": "^"}

ex = pd.read_csv(OUT / "r4_exact.csv").set_index(["dataset", "family"])
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
rng = np.random.default_rng(0)
for ax, ds in zip(axes, ("september", "frontier")):
    p = pd.read_pickle(CACHE / f"panel_{ds}.pkl")
    r = p[(p.role == "investor") & (p.task_order == "myth_game") & (p["round"] == 1)].dropna(subset=["own_send_amount"])
    for fam, g in r.groupby("family"):
        e = ex.loc[(ds, fam)]
        j = rng.uniform(-0.13, 0.13, (2, len(g)))
        ax.scatter(g.own_send_amount + j[0], g.send + j[1], s=26, alpha=0.75, color=COL[fam], marker=MARK[fam],
                   edgecolor="white", linewidth=0.6, label=f"{fam}: {e.n_exact}/{e.n_named} exact")
    ax.plot([0, 5], [0, 5], color="#999999", lw=1, ls="--", zorder=0)
    ax.set_title({"september": "September (Sonnet 4.5, GPT-5 Nano, Gemini 3.7 Flash)",
                  "frontier": "Frontier (Opus 5, GPT-5.6 Sol, Gemini 3.1 Pro)"}[ds], fontsize=9.5)
    ax.set_xlabel("amount the agent's own round-1 myth names ($)")
    ax.set_xlim(-0.4, 5.4); ax.set_ylim(-0.4, 5.4)
    ax.grid(color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
axes[0].set_ylabel("amount it then sends in round 1 ($)")
fig.suptitle("Round 1, myth before game: senders send what their own myth names", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "r4_plan_scatter.png", dpi=180)

# ---- key estimates: September beside frontier, per family
S = pd.read_csv(OUT / "scorecard_rows.csv")
rows = [
    ("R4 own myth named $ → round-1 send", "R4 own round-1 myth → round-1 send, named amount only ($/$)", "2+8-agent", "all myth→game round-1 senders"),
    ("Round-1 rule → mean send/5, rounds 2-10", "round-1 send rule → mean send/5 rounds 2-10 (per band level)", "2+8-agent", "myth→game agents"),
    ("Shown myth $ → reader's next send", "H2 shown myth stated amount → reader's next send ($/$)", "8-agent", "8-agent myth→game (homog + mixed)"),
    ("Shown myth $ → reader's next myth $", "H2 shown myth stated amount → reader's next myth amount ($/$)", "8-agent", "8-agent myth→game (homog + mixed)"),
    ("Own send → next own myth $", "reverse: own send → next own myth amount ($/$, agent FE)", "2+8-agent", "all myth runs"),
]
fams = {"september": ["Sonnet", "GPT"], "frontier": ["Sol", "Opus"]}
fig, axes = plt.subplots(1, len(rows), figsize=(15, 3.6))
for ax, (title, finding, setting, pop) in zip(axes, rows):
    y, labels = 0, []
    for ds in ("september", "frontier"):
        for fam in fams[ds]:
            r = S[(S.finding == finding) & (S.dataset == ds) & (S.family == fam) & (S.setting == setting) & (S.population == pop)]
            labels.append(f"{fam}")
            if len(r) and pd.notna(r.effect.iloc[0]) and not r.status.iloc[0].startswith("not estimable"):
                r = r.iloc[0]
                ax.plot([r.ci_low, r.ci_high], [y, y], color=COL[fam], lw=2)
                ax.scatter([r.effect], [y], color=COL[fam], marker=MARK[fam], s=40, zorder=3, edgecolor="white")
            else:
                st = r.status.iloc[0] if len(r) else "n/a"
                ax.text(0, y, "ceiling" if st.startswith("not estimable") else st, fontsize=7.5, color="#666666", va="center")
            y -= 1
        y -= 0.6
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set_ylim(-4.1, 0.5)
    ax.set_yticks([0, -1, -2.6, -3.6]); ax.set_yticklabels(["Sept " + labels[0], "Sept " + labels[1], "Front. " + labels[2], "Front. " + labels[3]], fontsize=8)
    ax.set_title(title, fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8)
fig.suptitle("Own-plan and stated-amount estimates, 95% CI (t, clustered by run); 'ceiling' = too few non-$5 rows to estimate", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "key_estimates.png", dpi=160)
print("saved")
