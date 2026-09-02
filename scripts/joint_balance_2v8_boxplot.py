#!/usr/bin/env python3
"""Population-wide (JOINT) cumulative balance: 2-agent vs 8-agent, side by side.

Unlike the task-order boxplots (which plot MEAN balance per agent, normalizing out
headcount), this plots JOINT balance = sum of all agents' final balances per run --
i.e. total wealth the whole population accumulates. Grouped by task-order condition
with population size (2 vs 8 agents) as the hue, so the population-scale gap is the
headline. Informed-noise, memory-primary, Sonnet 4.5, uniform +-1.0, n=5/cell.

Sources:
  2-agent: noise2i_memprimary_{game,game_myth} / noise2i_memtest_memprimary
  8-agent: noise8i_memprimary_{game,game_myth,myth_game}

Usage: python scripts/joint_balance_2v8_boxplot.py [--out DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from analyses.trajectory_boxplot import CONDITION_LABELS, CONDITION_ORDER

BASE = "data/json/noise_experiments/v2"
CELLS = {
    "2-agent dyad": {
        "game": f"{BASE}/noise2i_memprimary_game",
        "game_myth": f"{BASE}/noise2i_memprimary_game_myth",
        "myth_game": f"{BASE}/noise2i_memtest_memprimary",
    },
    "8-agent population": {
        "game": f"{BASE}/noise8i_memprimary_game",
        "game_myth": f"{BASE}/noise8i_memprimary_game_myth",
        "myth_game": f"{BASE}/noise8i_memprimary_myth_game",
    },
}


def joint_balance(run):
    """Sum of all agents' final balances (last round that carries balances)."""
    for entry in reversed(run.get("conversation_history", [])):
        balances = entry.get("balances")
        if balances:
            return float(sum(balances.values()))
    return None


def load_runs(cond_dir):
    runs = []
    for path in sorted(Path(cond_dir).rglob("*.json")):
        if path.name.endswith((".results.json", ".checkpoint.json", ".error.json")):
            continue
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/plots/joint_balance_2v8")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    rows = []
    for pop, conds in CELLS.items():
        for cond, cond_dir in conds.items():
            for run in load_runs(cond_dir):
                jb = joint_balance(run)
                if jb is not None:
                    rows.append({
                        "Condition": CONDITION_LABELS[cond],
                        "Population": pop,
                        "Joint Balance": jb,
                    })
    df = pd.DataFrame(rows)

    condition_order = [
        CONDITION_LABELS[c] for c in CONDITION_ORDER
        if CONDITION_LABELS[c] in df["Condition"].unique()
    ]
    pop_order = ["2-agent dyad", "8-agent population"]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.boxplot(data=df, x="Condition", y="Joint Balance", hue="Population",
                order=condition_order, hue_order=pop_order, ax=ax, palette="Set2")
    ax.set_title("Population-Wide (Joint) Cumulative Balance: 2-Agent vs 8-Agent\n"
                 "(Informed Noise, memory-primary, sum of all agents per run)",
                 fontweight="bold", fontsize=13)
    ax.set_ylabel("Joint Cumulative Balance (sum of all agents)", fontsize=12)
    ax.set_xlabel("Condition", fontsize=12)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.legend(title="Population", loc="upper left")

    plt.tight_layout()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "joint_balance_2v8_by_condition.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

    print("\n=== Joint balance (sum of all agents) per cell ===")
    for pop in pop_order:
        for label in condition_order:
            v = df[(df["Population"] == pop) & (df["Condition"] == label)]["Joint Balance"]
            if len(v):
                print(f"{pop:20s} {label:14s} {v.mean():7.1f} (+/-{v.std(ddof=0):.1f})  "
                      f"n={len(v)}  runs: {[round(x, 1) for x in v]}")


if __name__ == "__main__":
    main()
