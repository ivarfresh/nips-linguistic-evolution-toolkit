#!/usr/bin/env python3
"""Final cumulative balance boxplot across task orders under INFORMED noise (dyads).

Conditions: game / game_myth / myth_game, 2-agent dyads, Sonnet 4.5, memory-primary,
bidirectional uniform +-1.0 noise with agents INFORMED of the noise channel (n=5).
Matched to the 8-agent informed triplet (noise8i_memprimary_*) on noise + memory
design; the only axis that differs is population size (2 vs 8 agents).

Figure follows trajectory_boxplot.py::plot_cumulative_balances_boxplot_by_condition
and its uninformed sibling scripts/noise_taskorder_balance_boxplot.py.

Usage: python scripts/noise2i_dyad_taskorder_boxplot.py [--out DIR]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from analyses.trajectory_boxplot import CONDITION_LABELS, CONDITION_ORDER
from memtest_compare import load_runs, final_balance

CONDITION_DIRS = {
    "game": "data/json/noise_experiments/v2/noise2i_memprimary_game",
    "game_myth": "data/json/noise_experiments/v2/noise2i_memprimary_game_myth",
    "myth_game": "data/json/noise_experiments/v2/noise2i_memtest_memprimary",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/plots/noise2i_dyad_triplet")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    rows = []
    for cond, cond_dir in CONDITION_DIRS.items():
        runs = load_runs(cond_dir)
        print(f"{CONDITION_LABELS[cond]}: {len(runs)} runs")
        for run in runs:
            rows.append({
                "Condition": CONDITION_LABELS[cond],
                "Final Cumulative Balance": final_balance(run),
            })

    df = pd.DataFrame(rows)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    condition_order = [
        CONDITION_LABELS[c] for c in CONDITION_ORDER
        if CONDITION_LABELS[c] in df["Condition"].unique()
    ]

    sns.boxplot(data=df, x='Condition', y='Final Cumulative Balance',
                order=condition_order, ax=ax, palette='Set2')
    ax.set_title('Final Cumulative Balances by Condition '
                 '(Informed Noise, 2-Agent Dyads)',
                 fontweight='bold', fontsize=14)
    ax.set_ylabel('Final Cumulative Balance (Average per Agent)', fontsize=12)
    ax.set_xlabel('Condition', fontsize=12)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'cumulative_balance_boxplot_by_condition.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

    print("\n=== Final balance per run ===")
    for label in condition_order:
        v = df[df["Condition"] == label]["Final Cumulative Balance"]
        print(f"{label:14s} {v.mean():.2f} (+/-{v.std(ddof=0):.2f})  "
              f"runs: {[round(x, 1) for x in v]}")


if __name__ == "__main__":
    main()
