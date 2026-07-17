#!/usr/bin/env python3
"""Task-order condition comparison for the 8-agent memory-primary regime.

Same figure as analyses/trajectory_boxplot.py::plot_condition_comparison
(2x2: trust ratio, return ratio, sent, cooperation stability; Set2 palette),
with the standard Game Only / Game -> Myth / Myth -> Game conditions, computed
dyad-aware from the three memory-primary experiment sets.

Usage: python scripts/memprimary_taskorder_boxplot.py [--out DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from analyses.trajectory_boxplot import CONDITION_LABELS, CONDITION_ORDER
from memprimary_8agent_compare import load_runs, per_round_game

ENDOWMENT = 5.0

CONDITION_DIRS = {
    "game": "data/json/sonnet45_8agent_game_memprimary_r10_n5",
    "game_myth": "data/json/sonnet45_8agent_game_myth_memprimary_r10_n5",
    "myth_game": "data/json/sonnet45_8agent_myth_directive_history3_anon_memprimary_r10_n5",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/plots/memprimary_8agent")
    parser.add_argument("--game", default=None, help="override game-only runs dir")
    parser.add_argument("--game-myth", default=None, help="override game_myth runs dir")
    parser.add_argument("--myth-game", default=None, help="override myth_game runs dir")
    parser.add_argument("--title-suffix", default="", help="appended to figure titles")
    args = parser.parse_args()

    dirs = dict(CONDITION_DIRS)
    for cond, override in [("game", args.game), ("game_myth", args.game_myth),
                           ("myth_game", args.myth_game)]:
        if override:
            dirs[cond] = override

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    rows = []
    for cond, cond_dir in dirs.items():
        runs = load_runs(cond_dir)
        print(f"{CONDITION_LABELS[cond]}: {len(runs)} runs")
        for run in runs:
            sent, rr = per_round_game(run)
            if not len(sent):
                continue
            game_rounds = [r for r in run["conversation_history"] if r.get("balances")]
            final_balances = list(game_rounds[-1]["balances"].values()) if game_rounds else []
            final_balance = float(np.mean(final_balances)) if final_balances else 0.0
            rows.append({
                "Condition": CONDITION_LABELS[cond],
                "Mean Trust Ratio": float(np.mean(sent)) / ENDOWMENT,
                "Mean Return Ratio": float(np.mean(rr)),
                "Mean Sent": float(np.mean(sent)),
                "Cooperation Stability": float(np.std(rr)),
                "Final Cumulative Balance": final_balance,
            })

    df = pd.DataFrame(rows)

    # Set style
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cooperation Patterns Across Experimental Conditions' + args.title_suffix,
                 fontsize=16, fontweight='bold')

    condition_order = [
        CONDITION_LABELS[c] for c in CONDITION_ORDER
        if CONDITION_LABELS[c] in df["Condition"].unique()
    ]

    ax1 = axes[0, 0]
    sns.boxplot(data=df, x='Condition', y='Mean Trust Ratio',
                order=condition_order, ax=ax1, palette='Set2')
    ax1.set_title('Trust Ratio (Proportion Sent)', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Mean Trust Ratio (0-1)', fontsize=11)
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', rotation=15)
    ax1.set_ylim(0, 1)
    ax1.set_yticks(np.arange(0, 1.1, 0.1))

    ax2 = axes[0, 1]
    sns.boxplot(data=df, x='Condition', y='Mean Return Ratio',
                order=condition_order, ax=ax2, palette='Set2')
    ax2.set_title('Return Ratio (Proportion Returned)', fontweight='bold', fontsize=13)
    ax2.set_ylabel('Mean Return Ratio (0-1)', fontsize=11)
    ax2.set_xlabel('')
    ax2.tick_params(axis='x', rotation=15)
    ax2.set_ylim(0, 1)
    ax2.set_yticks(np.arange(0, 1.1, 0.1))

    ax3 = axes[1, 0]
    sns.boxplot(data=df, x='Condition', y='Mean Sent',
                order=condition_order, ax=ax3, palette='Set2')
    ax3.set_title('Amount Sent by Investors', fontweight='bold', fontsize=13)
    ax3.set_ylabel('Mean Amount Sent', fontsize=11)
    ax3.set_xlabel('Condition', fontsize=11)
    ax3.tick_params(axis='x', rotation=15)
    ax3.set_ylim(bottom=0)

    ax4 = axes[1, 1]
    sns.boxplot(data=df, x='Condition', y='Cooperation Stability',
                order=condition_order, ax=ax4, palette='Set2')
    ax4.set_title('Cooperation Stability (Lower = More Stable)', fontweight='bold', fontsize=13)
    ax4.set_ylabel('Std Dev of Return Ratio', fontsize=11)
    ax4.set_xlabel('Condition', fontsize=11)
    ax4.tick_params(axis='x', rotation=15)
    ax4.set_ylim(bottom=0)

    plt.tight_layout()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'condition_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {out_dir / 'condition_comparison.png'}")
    plt.close()

    # Final cumulative balance boxplot (layout of
    # trajectory_boxplot.py::plot_cumulative_balances_boxplot_by_condition)
    fig, ax = plt.subplots(figsize=(10, 7))

    sns.boxplot(data=df, x='Condition', y='Final Cumulative Balance',
                order=condition_order, ax=ax, palette='Set2')
    ax.set_title('Final Cumulative Balances by Condition' + args.title_suffix,
                 fontweight='bold', fontsize=14)
    ax.set_ylabel('Final Cumulative Balance (Average per Agent)', fontsize=12)
    ax.set_xlabel('Condition', fontsize=12)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(out_dir / 'cumulative_balance_boxplot_by_condition.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {out_dir / 'cumulative_balance_boxplot_by_condition.png'}")
    plt.close()

    # Cumulative balance over rounds (layout of
    # trajectory_boxplot.py::plot_cumulative_balances_by_condition)
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'game': '#66c2a5', 'myth': '#fc8d62', 'game_myth': '#8da0cb', 'myth_game': '#e78ac3'}

    for cond in CONDITION_ORDER:
        if cond not in dirs:
            continue
        runs = load_runs(dirs[cond])
        per_run = []
        for run in runs:
            balances = [
                float(np.mean(list(e["balances"].values())))
                for e in run.get("conversation_history", [])
                if e.get("balances")
            ]
            if balances:
                per_run.append(balances)
        if not per_run:
            continue
        n = min(len(b) for b in per_run)
        avg = np.stack([b[:n] for b in per_run]).mean(axis=0)
        rounds = list(range(1, n + 1))

        ax.plot(rounds, avg, linewidth=3, label=CONDITION_LABELS[cond],
                color=colors.get(cond, 'gray'), alpha=0.8)

    ax.set_xlabel('Round', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Cumulative Balance', fontsize=13, fontweight='bold')
    ax.set_title('Average Cumulative Balances Across Conditions' + args.title_suffix,
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(out_dir / 'cumulative_balances_by_condition.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {out_dir / 'cumulative_balances_by_condition.png'}")
    plt.close()

    print("\n=== Run-level means ===")
    for col in ["Mean Trust Ratio", "Mean Return Ratio", "Mean Sent", "Cooperation Stability",
                "Final Cumulative Balance"]:
        parts = []
        for label in condition_order:
            v = df[df["Condition"] == label][col]
            parts.append(f"{label}: {v.mean():.3f} (±{v.std(ddof=0):.3f})")
        print(f"{col:24s} " + "  ".join(parts))


if __name__ == "__main__":
    main()
