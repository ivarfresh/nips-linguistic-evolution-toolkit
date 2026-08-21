#!/usr/bin/env python3
"""Analyze the frozen independent GPT-5 Nano history confirmation."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyses._shared import configure_matplotlib
from analyze_corrected_v2_confirmatory import load_runs, run_metrics


DEFAULT_INPUT = Path(
    "data/json/noise_experiments/history_visibility_confirm_gpt_n10_20260821"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/history_visibility_confirmation_gpt_n10_20260821"
)
HISTORY_ORDER = ["private", "dossier"]
HISTORY_LABELS = {
    "private": "Private interaction memory",
    "dossier": "Current-partner dossier",
}
HISTORY_COLORS = {"private": "#2a9d8f", "dossier": "#e76f51"}
TASK_ORDER = ["game", "myth_game"]
TASK_LABELS = {"game": "Game only", "myth_game": "Myth → Game"}
TASK_COLORS = {"game": "#66c2a5", "myth_game": "#e78ac3"}
CELL_DIRECTORIES = {
    ("private", "game"): "noise8_history_confirm_gpt_n10_private_game",
    ("private", "myth_game"): "noise8_history_confirm_gpt_n10_private_myth_game",
    ("dossier", "game"): "noise8_history_confirm_gpt_n10_dossier_game",
    ("dossier", "myth_game"): "noise8_history_confirm_gpt_n10_dossier_myth_game",
}


def ci(values):
    values = np.asarray(values, dtype=float)
    return stats.t.interval(
        0.95,
        len(values) - 1,
        loc=values.mean(),
        scale=stats.sem(values),
    )


def usage(run):
    totals = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    attempts = 0
    retries = 0
    for agent in (run.get("agents") or {}).values():
        for event in agent.get("interaction_history") or []:
            response = event.get("response") or {}
            call_usage = response.get("usage") or {}
            if response:
                attempts += 1
            if event.get("error"):
                retries += 1
            for key in totals:
                totals[key] += int(call_usage.get(key) or 0)
    return totals, attempts, retries


def load_confirmation(input_dir):
    rows = []
    trajectories = []
    token_totals = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    attempts = 0
    retries = 0
    expected_ids = set(range(5, 15))
    for (history, task), directory in CELL_DIRECTORIES.items():
        runs = load_runs(input_dir / directory)
        replicate_ids = {
            int((run.get("run_metadata") or {})["replicate_id"])
            for _, run in runs
        }
        if len(runs) != 10 or replicate_ids != expected_ids:
            raise RuntimeError(
                f"Cell {(history, task)} has IDs {sorted(replicate_ids)}"
            )
        for path, run in runs:
            metadata = run.get("run_metadata") or {}
            if metadata.get("code_dirty"):
                raise RuntimeError(f"Dirty execution provenance in {path}")
            metrics, points = run_metrics(path, run)
            replicate_id = int(metadata["replicate_id"])
            metrics["return_over_sent"] = (
                metrics["mean_returned"] / metrics["mean_sent"]
                if metrics["mean_sent"] > 0
                else math.nan
            )
            rows.append(
                {
                    "history": history,
                    "history_label": HISTORY_LABELS[history],
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "replicate_id": replicate_id,
                    **metrics,
                }
            )
            trajectories.extend(
                {
                    "history": history,
                    "history_label": HISTORY_LABELS[history],
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "replicate_id": replicate_id,
                    **point,
                }
                for point in points
            )
            run_usage, run_attempts, run_retries = usage(run)
            for key in token_totals:
                token_totals[key] += run_usage[key]
            attempts += run_attempts
            retries += run_retries
    return rows, trajectories, token_totals, attempts, retries


def summary_records(dataframe):
    records = []
    for history in HISTORY_ORDER:
        for task in TASK_ORDER:
            cell = dataframe[
                (dataframe["history"] == history) & (dataframe["task"] == task)
            ]
            for metric in (
                "final_balance",
                "mean_trust_ratio",
                "mean_return_ratio",
                "return_over_sent",
            ):
                values = cell[metric].to_numpy(dtype=float)
                low, high = ci(values)
                records.append(
                    {
                        "history": history,
                        "history_label": HISTORY_LABELS[history],
                        "task": task,
                        "task_label": TASK_LABELS[task],
                        "metric": metric,
                        "n": len(values),
                        "mean": values.mean(),
                        "sd": values.std(ddof=1),
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return records


def contrast_record(values, metric, contrast, primary=False):
    values = np.asarray(values, dtype=float)
    low, high = ci(values)
    test = stats.ttest_1samp(values, 0)
    return {
        "metric": metric,
        "contrast": contrast,
        "primary": primary,
        "n_pairs": len(values),
        "estimate": values.mean(),
        "ci_low": low,
        "ci_high": high,
        "p_value": test.pvalue,
        "cohens_dz": values.mean() / values.std(ddof=1),
    }


def contrast_records(dataframe):
    records = []
    for metric in ("final_balance", "mean_trust_ratio", "mean_return_ratio"):
        pivot = dataframe.pivot(
            index="replicate_id",
            columns=["history", "task"],
            values=metric,
        )
        dossier_game = pivot[("dossier", "game")] - pivot[("private", "game")]
        dossier_myth = (
            pivot[("dossier", "myth_game")] - pivot[("private", "myth_game")]
        )
        interaction = dossier_myth - dossier_game
        records.append(
            contrast_record(
                interaction,
                metric,
                "(Dossier − private | Myth→Game) − (Dossier − private | Game only)",
                primary=metric == "final_balance",
            )
        )
        records.append(
            contrast_record(
                dossier_game,
                metric,
                "Dossier − private | Game only",
            )
        )
        records.append(
            contrast_record(
                dossier_myth,
                metric,
                "Dossier − private | Myth→Game",
            )
        )
        records.append(
            contrast_record(
                pivot[("private", "myth_game")] - pivot[("private", "game")],
                metric,
                "Myth→Game − Game only | Private memory",
            )
        )
        records.append(
            contrast_record(
                pivot[("dossier", "myth_game")] - pivot[("dossier", "game")],
                metric,
                "Myth→Game − Game only | Partner dossier",
            )
        )
    return records


def plot_cells(dataframe, output_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    sns.boxplot(
        data=dataframe,
        x="task_label",
        y="final_balance",
        hue="history_label",
        order=[TASK_LABELS[value] for value in TASK_ORDER],
        hue_order=[HISTORY_LABELS[value] for value in HISTORY_ORDER],
        palette=[HISTORY_COLORS[value] for value in HISTORY_ORDER],
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=dataframe,
        x="task_label",
        y="final_balance",
        hue="history_label",
        order=[TASK_LABELS[value] for value in TASK_ORDER],
        hue_order=[HISTORY_LABELS[value] for value in HISTORY_ORDER],
        palette=["#263238", "#263238"],
        dodge=True,
        alpha=0.7,
        size=4.5,
        legend=False,
        ax=ax,
    )
    ax.set_title("Independent history-visibility confirmation (GPT-5 Nano)", fontweight="bold")
    ax.set_xlabel("Task order")
    ax.set_ylabel("Average final balance per agent")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Decision-time history")
    fig.tight_layout()
    fig.savefig(output_dir / "confirmatory_final_balance.png", dpi=300)
    plt.close(fig)


def plot_primary(dataframe, output_dir):
    import matplotlib.pyplot as plt

    pivot = dataframe.pivot(
        index="replicate_id",
        columns=["history", "task"],
        values="final_balance",
    )
    interactions = (
        pivot[("dossier", "myth_game")]
        - pivot[("private", "myth_game")]
        - pivot[("dossier", "game")]
        + pivot[("private", "game")]
    )
    low, high = ci(interactions)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.scatter(
        interactions.index,
        interactions,
        color="#607d8b",
        s=55,
        label="Matched replicate",
    )
    ax.errorbar(
        14.8,
        interactions.mean(),
        yerr=[[interactions.mean() - low], [high - interactions.mean()]],
        color="#9c2f1f",
        marker="o",
        markersize=8,
        capsize=5,
        linewidth=2.5,
        label="Mean and 95% CI",
    )
    ax.axhline(0, color="#263238", linestyle="--", linewidth=1)
    ax.set_xticks(list(interactions.index) + [15], [str(i) for i in interactions.index] + ["Mean"])
    ax.set_title("Preregistered history × Myth→Game interaction", fontweight="bold")
    ax.set_xlabel("Independent replicate ID")
    ax.set_ylabel("Difference-in-differences\n(final balance per agent)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "confirmatory_primary_interaction.png", dpi=300)
    plt.close(fig)


def plot_trajectories(trajectory_dataframe, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    for ax, history in zip(axes, HISTORY_ORDER):
        subset = trajectory_dataframe[trajectory_dataframe["history"] == history]
        for task in TASK_ORDER:
            means = (
                subset[subset["task"] == task]
                .groupby("round", as_index=False)["trust_ratio"]
                .mean()
            )
            ax.plot(
                means["round"],
                means["trust_ratio"],
                color=TASK_COLORS[task],
                marker="o",
                linewidth=2.5,
                label=TASK_LABELS[task],
            )
        ax.set_title(HISTORY_LABELS[history], fontweight="bold")
        ax.set_xlabel("Round")
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, linestyle="--")
    axes[0].set_ylabel("Mean proportion sent")
    axes[1].legend(title="Task order", loc="lower right")
    fig.suptitle("Confirmatory trust trajectories", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "confirmatory_trust_trajectories.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    configure_matplotlib()
    import pandas as pd
    import seaborn as sns

    sns.set_style("whitegrid")
    args.out.mkdir(parents=True, exist_ok=True)
    rows, trajectories, tokens, attempts, retries = load_confirmation(args.input)
    dataframe = pd.DataFrame(rows).sort_values(["history", "task", "replicate_id"])
    trajectory_dataframe = pd.DataFrame(trajectories)
    summary = pd.DataFrame(summary_records(dataframe))
    contrasts = pd.DataFrame(contrast_records(dataframe))
    estimated_cost = (
        tokens["input_tokens"] / 1_000_000 * 0.05
        + (tokens["output_tokens"] + tokens["reasoning_tokens"])
        / 1_000_000
        * 0.40
    )
    cost = pd.DataFrame(
        [
            {
                **tokens,
                "total_attempts": attempts,
                "recovered_retries": retries,
                "assumed_input_usd_per_million": 0.05,
                "assumed_output_usd_per_million": 0.40,
                "estimated_list_price_usd": estimated_cost,
            }
        ]
    )

    dataframe.to_csv(args.out / "run_metrics.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    contrasts.to_csv(args.out / "contrasts.csv", index=False)
    cost.to_csv(args.out / "token_cost.csv", index=False)
    plot_cells(dataframe, args.out)
    plot_primary(dataframe, args.out)
    plot_trajectories(trajectory_dataframe, args.out)

    print(summary[summary["metric"] == "final_balance"].to_string(index=False))
    print("\nFinal-balance tests (primary first):")
    print(contrasts[contrasts["metric"] == "final_balance"].to_string(index=False))
    print(f"\nAttempts: {attempts}; recovered retries: {retries}")
    print(f"Estimated list-price cost under recorded rates: ${estimated_cost:.4f}")
    print(f"Saved outputs to {args.out}")


if __name__ == "__main__":
    main()
