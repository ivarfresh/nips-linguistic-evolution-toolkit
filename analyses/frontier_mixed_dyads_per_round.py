#!/usr/bin/env python3
"""Per-round transfers in the mixed frontier dyads (paper Figure 4b candidates).

Companion to Figure 4a (frontier_mixed_dyads_resources_boxplots.png): same three
pairings, same runs, shown round by round. Send fraction per round = sent / $5 for
the round's single sender (roles alternate, so odd rounds are the first sender's).
Lines are the mean over runs, bands ± 1 sd. No plot titles (paper style).

Two versions, to compare:
  frontier_mixed_dyads_send_per_round.png        game only as one line
  frontier_mixed_dyads_send_per_round_split.png  game only split by which family sent first

Runs: the launcher-audited mixed dyad finals read by
scripts/analyze_frontier_main_mixed_20260928.py (54 = 3 pairings x 3 task orders x 6).
No API calls.
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
from scripts.analyze_frontier_main_mixed_20260928 import FAMILY_OF_MODEL, load  # noqa: E402

OUTPUT = ROOT / "docs/figures/frontier_mixed_dyads_per_round_20261006"
ENDOWMENT = 5.0
PANELS = ["Opus 5 + Sol", "Opus 5 + Gemini", "Gemini + Sol"]  # as Figure 4a
TASK_ORDERS = ["game", "game_myth", "myth_game"]
ORDER_LABELS = {"game": "Game only", "game_myth": "Game → Myth", "myth_game": "Myth → Game"}
LINE_COLORS = {"game": "#777777", "game_myth": "#fc8d62", "myth_game": "#66c2a5"}


def per_round(paths) -> pd.DataFrame:
    rows = []
    for path in paths:
        run = json.loads(Path(path).read_text())
        models = run["run_metadata"]["agent_models"]
        family = {a: FAMILY_OF_MODEL[m.split("/", 1)[-1]] for a, m in models.items()}
        panel = " + ".join(sorted(set(family.values()), key=["Opus 5", "Gemini", "Sol"].index))
        history = run["conversation_history"]
        first = family[history[0]["dyads"][0]["investor"]]
        for r in history:
            (dy,) = r["dyads"]
            rows.append({"path": str(path), "panel": panel, "task_order": "_".join(run["task_order"]),
                         "first_sender": first, "round": int(r["round"]),
                         "send_fraction": dy["sent_decision"] / ENDOWMENT})
    return pd.DataFrame(rows)


def line(ax, d: pd.DataFrame, color: str, style: str = "-") -> int:
    g = d.groupby("round")["send_fraction"]
    x, mean, sd = g.mean().index.to_numpy(), g.mean().to_numpy(), np.nan_to_num(g.std(ddof=1).to_numpy())
    ax.fill_between(x, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), color=color, alpha=0.13, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=2.2, linestyle=style, marker="o", ms=3.5)
    return d["path"].nunique()


def plot(df: pd.DataFrame, split: bool, filename: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), sharex=True, sharey=True)
    for ax, panel in zip(axes, PANELS):
        p = df[df["panel"] == panel]
        a, b = panel.split(" + ")
        for task_order in TASK_ORDERS:
            d = p[p["task_order"] == task_order]
            if task_order == "game" and split:
                for fam, style in ((a, "-"), (b, "--")):
                    line(ax, d[d["first_sender"] == fam], LINE_COLORS["game"], style)
            else:
                line(ax, d, LINE_COLORS[task_order])
        ax.set_title(panel, fontsize=12, fontweight="bold")
        if split:
            ax.legend(handles=[Line2D([], [], color=LINE_COLORS["game"], linewidth=2.2, linestyle=s, label=f"Game only, {f} first")
                               for f, s in ((a, "-"), (b, "--"))], loc="lower right", frameon=False, fontsize=9)
        ax.set_xticks(range(1, 11))
        ax.set_xlim(0.6, 10.4)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Round")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Send fraction (sent / $5)")
    handles = [Line2D([], [], color=LINE_COLORS[t], linewidth=2.2, marker="o", ms=4, label=ORDER_LABELS[t]) for t in TASK_ORDERS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.2)
    fig.savefig(OUTPUT / f"{filename}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _, _, mixed = load()
    dyad_paths = [p for p in mixed if json.loads(Path(p).read_text())["run_metadata"].get("num_agents") == 2]
    df = per_round(dyad_paths)
    counts = df.drop_duplicates("path").groupby(["panel", "task_order", "first_sender"]).size()
    if len(dyad_paths) != 54 or not (counts == 3).all():
        raise SystemExit(f"expected 54 dyads, 3 per first sender per cell; got {len(dyad_paths)}\n{counts}")
    df.to_csv(OUTPUT / "send_per_round.csv", index=False)
    plot(df, split=False, filename="frontier_mixed_dyads_send_per_round")
    plot(df, split=True, filename="frontier_mixed_dyads_send_per_round_split")


if __name__ == "__main__":
    main()
