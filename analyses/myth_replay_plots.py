#!/usr/bin/env python3
"""Distribution plots for the edit-and-replay probe.

  replay_boxes_by_amount.png   replayed send at each stated amount, per arm x model (rule edits);
                               one value per context and amount (its replays averaged)
  replay_rule_vs_story.png     Sonnet: rule sentence vs amount rewritten inside the story
  replay_ablation_style.png    bars per condition (unedited, rule $1/2/3/5) in the slide-678 ablation style
  opening_runs.png             existing September runs, one point per run: round-1 send and mean
                               send in rounds 2-10 by task order, and round-1 vs later send

Reads data/analysis/myth_replay_probe_20261001/main.jsonl and
data/analysis/linguistic_20260923/decisions.csv.

  python3 analyses/myth_replay_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses._shared import configure_matplotlib  # noqa: E402

DIR = ROOT / "docs/research/myth_replay_probe_20261001"
RAW = ROOT / "data/analysis/myth_replay_probe_20261001/main.jsonl"
FAM_COL = {"Sonnet": "#7570b3", "GPT": "#d95f02", "Gemini": "#1b9e77"}
ORDER_COL = {"myth_game": "#2b2b2b", "game_myth": "#9a9a9a"}
ORDER_NAME = {"myth_game": "story first", "game_myth": "game first"}
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"
ARMS = [("A", "Own first myth"), ("C", "Own later myth"), ("B", "Partner's myth, read")]
AMOUNTS = [1.0, 2.0, 3.0, 5.0]
RNG = np.random.default_rng(7)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def context_means() -> pd.DataFrame:
    rows = [json.loads(x) for x in RAW.read_text().splitlines() if x.strip()]
    d = pd.DataFrame(rows).drop_duplicates("key", keep="last")
    d = d[(d["made"] == True) & d["error"].isna() & d["send"].notna()]  # noqa: E712
    d = d[d["mode"].isin(["rule", "natural"])]
    return d.groupby(["arm", "family", "mode", "cid", "edited_amount"], as_index=False)["send"].mean()


def boxes(ax, data: list[np.ndarray], positions, color, width=0.6, hollow=False):
    bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True, showfliers=False,
                    medianprops={"color": "white" if not hollow else color, "lw": 2},
                    whiskerprops={"color": color, "lw": 1.2}, capprops={"color": color, "lw": 1.2},
                    boxprops={"edgecolor": color, "lw": 1.2})
    for patch in bp["boxes"]:
        patch.set_facecolor("white" if hollow else color)
        patch.set_alpha(1.0 if hollow else 0.85)
    for x, vals in zip(positions, data):
        ax.scatter(x + RNG.uniform(-width / 3, width / 3, len(vals)), vals, s=7, color=color,
                   alpha=0.45, lw=0, zorder=3)


def boxes_by_amount() -> Path:
    import matplotlib.pyplot as plt
    m = context_means()
    m = m[m["mode"] == "rule"]
    fams = ["Sonnet", "GPT", "Gemini"]
    fig, axes = plt.subplots(len(ARMS), len(fams), figsize=(9.5, 8.2), sharex=True, sharey=True)
    for i, (arm, arm_name) in enumerate(ARMS):
        for j, fam in enumerate(fams):
            ax = axes[i, j]
            g = m[(m["arm"] == arm) & (m["family"] == fam)]
            data = [g.loc[g["edited_amount"] == a, "send"].to_numpy() for a in AMOUNTS]
            boxes(ax, data, AMOUNTS, FAM_COL[fam])
            ax.plot([0.5, 5.5], [0.5, 5.5], color=MUTED, lw=1, ls=(0, (2, 2)), zorder=1)
            ax.set_xlim(0.3, 5.7)
            ax.set_ylim(-0.2, 5.3)
            ax.set_xticks(AMOUNTS, ["$1", "$2", "$3", "$5"])
            ax.set_yticks([0, 1, 2, 3, 4, 5])
            style(ax)
            if i == 0:
                ax.set_title(fam, fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(f"{arm_name}\nsend ($)", fontsize=9, color=INK)
            if i == len(ARMS) - 1:
                ax.set_xlabel("amount the rule says to send", fontsize=8.5, color=INK)
            ax.text(0.97, 0.04, f"n = {g['cid'].nunique()}", transform=ax.transAxes,
                    fontsize=7.5, color=MUTED, va="bottom", ha="right")
    fig.suptitle("Add \"whoever holds five should send X\" to a myth, replay the decision: what does the agent send?",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.01, "Each dot is one replayed decision situation (its replays averaged); boxes show the middle half, "
             "line the median. Dashed: sends exactly the stated amount.\nSeptember runs. Own first myth: written just "
             "before the first send. Own later myth: written after the last game.\nPartner's myth: read before the "
             "agent wrote its own myth, which is left unedited. n = decision situations.", fontsize=7.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    out = DIR / "replay_boxes_by_amount.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def rule_vs_story() -> Path:
    import matplotlib.pyplot as plt
    m = context_means()
    m = m[m["family"] == "Sonnet"]
    both = m.groupby("cid")["mode"].nunique()
    m = m[m["cid"].isin(both[both == 2].index)]  # same situations in both edit types
    fig, axes = plt.subplots(1, len(ARMS), figsize=(9.5, 3.4), sharey=True)
    col = FAM_COL["Sonnet"]
    for ax, (arm, arm_name) in zip(axes, ARMS):
        g = m[m["arm"] == arm]
        for mode, off, hollow in (("rule", -0.22, False), ("natural", 0.22, True)):
            data = [g.loc[(g["mode"] == mode) & (g["edited_amount"] == a), "send"].to_numpy() for a in AMOUNTS]
            boxes(ax, data, [a + off for a in AMOUNTS], col, width=0.38, hollow=hollow)
        ax.plot([0.5, 5.5], [0.5, 5.5], color=MUTED, lw=1, ls=(0, (2, 2)), zorder=1)
        ax.set_xlim(0.3, 5.7)
        ax.set_ylim(-0.2, 5.3)
        ax.set_xticks(AMOUNTS, ["$1", "$2", "$3", "$5"])
        style(ax)
        ax.set_title(f"{arm_name}  (n = {g['cid'].nunique()})", fontsize=9.5, color=INK)
        ax.set_xlabel("amount stated", fontsize=8.5, color=INK)
    axes[0].set_ylabel("Sonnet send ($)", fontsize=9, color=INK)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=col, edgecolor=col, label="rule sentence added"),
                        Patch(facecolor="white", edgecolor=col, label="amount rewritten inside the story")],
               loc="upper left", bbox_to_anchor=(0.02, 0.92), ncol=2, fontsize=8, frameon=False)
    fig.suptitle("A stated rule moves Sonnet more than the same amount told in the story",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.01, "Same decision situations in both edit types (myths that already named an amount). "
             "Dots: one situation each; dashed: sends exactly the stated amount.", fontsize=7.5, color=MUTED,
             ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 0.84))
    out = DIR / "replay_rule_vs_story.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def opening_runs() -> Path:
    import matplotlib.pyplot as plt
    d = pd.read_csv(ROOT / "data/analysis/linguistic_20260923/decisions.csv")
    inv = d[d["role"] == "investor"]
    x = d.drop_duplicates("run_id").set_index("run_id")[["mixed", "task_order", "composition"]].join(
        [inv[inv["round"] == 1].groupby("run_id")["sent"].mean().rename("r1"),
         inv[inv["round"] >= 2].groupby("run_id")["sent"].mean().rename("later")]).reset_index()
    fam_of = {"8 Sonnet": "Sonnet", "Sonnet+Sonnet": "Sonnet", "8 GPT": "GPT", "GPT+GPT": "GPT",
              "8 Gemini": "Gemini", "Gemini+Gemini": "Gemini"}
    x["model"] = [fam_of.get(c, "mixed") for c in x["composition"]]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8), gridspec_kw={"width_ratios": [1.4, 1.4, 1.2]})
    for j, (col, what) in enumerate((("r1", "Round-1 send ($)"), ("later", "Mean send, rounds 2–10 ($)"))):
        # row 1: single-model runs, split by model
        ax = axes[0, j]
        ticks, labels = [], []
        for k, fam in enumerate(("Sonnet", "GPT", "Gemini")):
            for o, to in enumerate(("game_myth", "myth_game")):
                pos = k * 2.6 + o
                vals = x.loc[(x["model"] == fam) & (x["task_order"] == to), col].to_numpy()
                boxes(ax, [vals], [pos], ORDER_COL[to], width=0.7)
            ticks.append(k * 2.6 + 0.5)
            labels.append(fam)
        ax.set_xticks(ticks, labels)
        # row 2: mixed runs, compositions pooled
        ax2 = axes[1, j]
        for o, to in enumerate(("game_myth", "myth_game")):
            boxes(ax2, [x.loc[(x["model"] == "mixed") & (x["task_order"] == to), col].to_numpy()], [o],
                  ORDER_COL[to], width=0.55)
        ax2.set_xticks([0, 1], [ORDER_NAME["game_myth"], ORDER_NAME["myth_game"]])
        for a in (ax, ax2):
            a.set_ylim(-0.2, 5.3)
            style(a)
            a.set_title(what, fontsize=9.5, color=INK)
    axes[0, 0].set_ylabel("Single-model populations\n(10 runs per model and order)", fontsize=9, color=INK)
    axes[1, 0].set_ylabel("Mixed-model populations\n(48 runs per order)", fontsize=9, color=INK)
    for i, mixed in enumerate((False, True)):
        ax = axes[i, 2]
        g = x[x["mixed"] == mixed]
        for to in ("game_myth", "myth_game"):
            h = g[g["task_order"] == to]
            ax.scatter(h["r1"], h["later"], s=18, color=ORDER_COL[to], alpha=0.75, lw=0, label=ORDER_NAME[to])
        ax.set_xlim(-0.2, 5.3)
        ax.set_ylim(-0.2, 5.3)
        style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_xlabel("round-1 send ($)", fontsize=8.5, color=INK)
        ax.set_ylabel("mean send, rounds 2–10 ($)", fontsize=8.5, color=INK)
        ax.set_title("Same opening, different later play?", fontsize=9.5, color=INK)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=ORDER_COL[t], label=ORDER_NAME[t]) for t in ("game_myth", "myth_game")],
               loc="upper left", bbox_to_anchor=(0.02, 0.945), ncol=2, fontsize=8.5, frameon=False)
    fig.suptitle("Story-first runs open higher and stay higher; in mixed populations the gap is not only the opening",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.01, "Existing September myth runs, one point per run. In game-first runs the round-1 send comes "
             "before any myth. Mixed row pools the model compositions.", fontsize=7.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    out = DIR / "opening_runs.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def ablation_style() -> Path:
    """Same layout as docs/figures/slide678_rerun_20260916/ablation_run.png: one bar per condition
    (mean, +-sd whisker), one dot per decision situation, mean (+-sd) and n printed above, dotted
    line at the unedited baseline mean, dashed line at the $5 ceiling."""
    import matplotlib.pyplot as plt
    rows = [json.loads(x) for x in RAW.read_text().splitlines() if x.strip()]
    d = pd.DataFrame(rows).drop_duplicates("key", keep="last")
    d = d[(d["made"] == True) & d["error"].isna() & d["send"].notna()]  # noqa: E712
    d = d[d["mode"].isin(["orig", "rule"])].copy()
    d["cond"] = np.where(d["mode"] == "orig", "unedited", "$" + d["edited_amount"].fillna(0).astype(int).astype(str))
    cm = d.groupby(["arm", "family", "cond", "cid"], as_index=False)["send"].mean()
    conds = [("unedited", "Unedited myth\n(baseline)", "#8c8c8c"), ("$1", "Rule: send $1", "#c6dbef"),
             ("$2", "Rule: send $2", "#6baed6"), ("$3", "Rule: send $3", "#2171b5"), ("$5", "Rule: send $5", "#08306b")]
    fams = ["Sonnet", "GPT"]
    fig, axes = plt.subplots(len(ARMS), len(fams), figsize=(13, 11), sharey=True)
    for i, (arm, arm_name) in enumerate(ARMS):
        for j, fam in enumerate(fams):
            ax = axes[i, j]
            g = cm[(cm["arm"] == arm) & (cm["family"] == fam)]
            base = g.loc[g["cond"] == "unedited", "send"].mean()
            for k, (c, label, color) in enumerate(conds):
                v = g.loc[g["cond"] == c, "send"].to_numpy()
                if not len(v):
                    continue
                mu, sd = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
                ax.bar(k, mu, width=0.62, color=color, edgecolor="black", linewidth=0.7, alpha=0.88)
                # +-sd whisker, clipped to the $0-$5 range a send can take
                ax.errorbar(k, mu, yerr=[[min(sd, mu)], [min(sd, 5 - mu)]], color="black", capsize=4, lw=1.2)
                ax.scatter(k + RNG.uniform(-0.2, 0.2, len(v)), v, s=22, color="black", edgecolor="white",
                           linewidth=0.6, zorder=3)
                ax.text(k, 5.55, f"${mu:.2f}\n(±${sd:.2f})\nn={len(v)}", ha="center",
                        va="bottom", fontsize=7.5)
            ax.axhline(5, color="red", ls="--", alpha=0.55, lw=1.2)
            ax.axhline(base, color="#7f7f7f", ls=":", alpha=0.8, lw=1.4)
            ax.set_xticks(range(len(conds)), [c[1] for c in conds], fontsize=8)
            ax.set_ylim(0, 6.9)
            ax.set_yticks(range(6))
            ax.grid(axis="y", alpha=0.35)
            ax.set_axisbelow(True)
            ax.set_title(f"{fam} · {arm_name}", fontsize=10)
            if j == 0:
                ax.set_ylabel("Replayed send ($)", fontsize=9)
    fig.suptitle("Myth edit-and-replay ablation: a send rule in the myth sets the next send\n"
                 "September runs · rule appended to the myth (\"whoever holds five should send X\") · same decision "
                 "situations in every condition · one dot per situation (replays averaged)\n"
                 "Own first and later myths: sends follow the rule; partner's myth (own reaction myth unedited): "
                 "sends barely move. Gemini sends $5 throughout and is not shown.", fontsize=10)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color="red", ls="--", alpha=0.55, label="Ceiling ($5)"),
                        Line2D([], [], color="#7f7f7f", ls=":", lw=1.4, label="Unedited (baseline) mean")],
               loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    out = DIR / "replay_ablation_style.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    configure_matplotlib()
    for f in ("replay_slopes.png", "replay_opening_check.png"):
        (DIR / f).unlink(missing_ok=True)
    print(boxes_by_amount())
    print(rule_vs_story())
    print(opening_runs())
    print(ablation_style())


if __name__ == "__main__":
    main()
