#!/usr/bin/env python3
"""Summary plots for the edit-and-replay probe, from the CSVs written by
analyses/myth_replay_analysis.py (run that first).

  replay_slopes.png            send moved per $1 stated, per arm x model, rule vs story edit
  replay_opening_check.png     story-first advantage with and without the round-1 send held fixed

  python3 analyses/myth_replay_plots.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses._shared import configure_matplotlib  # noqa: E402

DIR = ROOT / "docs/research/myth_replay_probe_20261001"
COL = {"Sonnet": "#7570b3", "GPT": "#d95f02", "Sonnet+GPT (primary)": "#2b2b2b"}
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"
ARMS = [("A", "Own first myth\n(written just before the send)"),
        ("C", "Own later myth\n(written after the last game)"),
        ("B", "Partner's myth, read\n(own reaction myth unedited)")]


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def slopes_plot() -> Path:
    import matplotlib.pyplot as plt
    s = pd.read_csv(DIR / "slopes.csv")
    rows = []  # (arm, label, family, mode)
    for arm, _ in ARMS:
        rows += [(arm, "Sonnet + GPT", "Sonnet+GPT (primary)", "rule"), (arm, "Sonnet", "Sonnet", "rule"),
                 (arm, "GPT", "GPT", "rule"), (arm, "Sonnet, story edit", "Sonnet", "natural")]
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    y, yt, yl = 0, [], []
    for k, (arm, title) in enumerate(ARMS):
        if k:
            y += 0.8
        ax.text(-0.27, y - 0.15, title, fontsize=9, color=INK, fontweight="bold", va="bottom", ha="left")
        y += 0.55
        for a, label, fam, mode in (r for r in rows if r[0] == arm):
            r = s[(s["arm"] == a) & (s["family"] == fam) & (s["mode"] == mode)].iloc[0]
            col = COL[fam]
            hollow = mode == "natural"
            ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot(r["slope"], y, "o", ms=8, color=col, mfc="white" if hollow else col, mew=2, zorder=3)
            ax.text(max(r["ci_high"], r["slope"]) + 0.03, y, f"{r['slope']:.2f}", va="center", fontsize=8.5, color=INK)
            yt.append(y)
            yl.append(label)
            y += 0.6
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(1, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(1.0, -0.05, "sends exactly\nwhat it says", fontsize=7.5, color=MUTED, ha="center", va="bottom")
    ax.set_yticks(yt, yl, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.3, 1.25)
    ax.set_ylim(y - 0.2, -0.75)
    ax.set_xlabel("dollars sent per $1 the myth says to send (95% CI)", color=INK, fontsize=9)
    style(ax)
    fig.suptitle("Add \"send X\" to a myth and replay the decision:\n"
                 "agents follow a rule in their own myth almost one for one", fontsize=11, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.01, "Filled: a rule sentence added to the myth (\"whoever holds five should send X\"). "
             "Hollow: the amount rewritten inside the story.\nSeptember runs; Gemini 3.7 Flash sends $5 whatever "
             "its myth says and is not shown. Context fixed effects, run-clustered CIs.",
             fontsize=7.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    out = DIR / "replay_slopes.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def opening_plot() -> Path:
    import matplotlib.pyplot as plt
    m = pd.read_csv(DIR / "opening_mediation.csv")
    groups = [("single-model", "Single-model populations (60 runs)"), ("mixed-model", "Mixed-model populations (96 runs)"),
              ("all", "All runs (156)")]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, (key, label) in enumerate(groups):
        for j, (spec, col, name) in enumerate((("task order only", "#2b2b2b", "as observed"),
                                               ("+ round-1 send", "#9a9a9a", "round-1 send held fixed"))):
            r = m[(m["runs"] == key) & (m["spec"] == spec)].iloc[0]
            y = i + (j - 0.5) * 0.32
            ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot(r["gap"], y, "o", ms=8, color=col, zorder=3, label=name if i == 0 else None)
            ax.text(r["ci_high"] + 0.03, y, f"+${r['gap']:.2f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(groups)), [g[1] for g in groups], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("extra $ sent per round (rounds 2–10), story-first vs game-first runs (95% CI)", color=INK, fontsize=9)
    style(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8, frameon=False)
    fig.suptitle("Does a good first move explain why story-first runs cooperate more?\n"
                 "Partly in single-model populations, not in mixed ones", fontsize=11, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.01, "Existing September runs; composition × size fixed effects. Observational: task order also "
             "affects the round-1 send itself.", fontsize=7.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out = DIR / "replay_opening_check.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    configure_matplotlib()
    print(slopes_plot())
    print(opening_plot())


if __name__ == "__main__":
    main()
