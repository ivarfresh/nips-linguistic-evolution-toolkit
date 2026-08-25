#!/usr/bin/env python3
"""Analyze the matched three-history by Game/Myth→Game GPT screen."""

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
from analyze_history_visibility_factorial_gpt_n5 import _selected_runs


DEFAULT_OUTPUT = Path(
    "docs/figures/population_ledger_myth_game_gpt_n5_20260821"
)
HISTORY_ORDER = ["private", "dossier", "public_ledger"]
HISTORY_LABELS = {
    "private": "Private memory",
    "dossier": "Current-partner dossier",
    "public_ledger": "Public population ledger",
}
HISTORY_COLORS = {
    "private": "#2a9d8f",
    "dossier": "#e76f51",
    "public_ledger": "#6c5ce7",
}
TASK_ORDER = ["game", "myth_game"]
TASK_LABELS = {"game": "Game only", "myth_game": "Myth → Game"}
TASK_COLORS = {"game": "#66c2a5", "myth_game": "#e78ac3"}
PUBLIC_GAME_PATHS = [
    Path(
        "data/json/noise_experiments/population_ledger_smoke_gpt_20260821/"
        "noise8_population_ledger_signed_gpt_smoke_game"
    ),
    Path(
        "data/json/noise_experiments/population_ledger_gpt_n5_extension_20260821/"
        "noise8_population_ledger_signed_gpt_n5_extension_game"
    ),
]
PUBLIC_MYTH_GAME_PATH = Path(
    "data/json/noise_experiments/population_ledger_myth_game_gpt_n5_20260821/"
    "noise8_population_ledger_signed_gpt_n5_myth_game"
)


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


def selected_cells():
    prior = _selected_runs()
    public_game = []
    for path in PUBLIC_GAME_PATHS:
        public_game.extend(load_runs(path))
    return {
        ("private", "game"): prior[("own_only", "game")],
        ("private", "myth_game"): prior[("own_only", "myth_game")],
        ("dossier", "game"): prior[("partner_dossier", "game")],
        ("dossier", "myth_game"): prior[("partner_dossier", "myth_game")],
        ("public_ledger", "game"): public_game,
        ("public_ledger", "myth_game"): load_runs(PUBLIC_MYTH_GAME_PATH),
    }


def load_factorial():
    rows = []
    trajectories = []
    tokens = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    attempts = 0
    retries = 0
    expected_ids = set(range(5))
    for (history, task), runs in selected_cells().items():
        replicate_ids = [
            int((run.get("run_metadata") or {})["replicate_id"])
            for _, run in runs
        ]
        if len(runs) != 5 or set(replicate_ids) != expected_ids:
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
            for key in tokens:
                tokens[key] += run_usage[key]
            attempts += run_attempts
            retries += run_retries
    return rows, trajectories, tokens, attempts, retries


def summary_records(dataframe):
    records = []
    for history in HISTORY_ORDER:
        for task in TASK_ORDER:
            cell = dataframe[
                (dataframe["history"] == history)
                & (dataframe["task"] == task)
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


def difference_record(values, metric, contrast, primary=False):
    values = np.asarray(values, dtype=float)
    low, high = ci(values)
    sd = values.std(ddof=1)
    return {
        "metric": metric,
        "contrast": contrast,
        "primary": primary,
        "n_pairs": len(values),
        "estimate": values.mean(),
        "ci_low": low,
        "ci_high": high,
        "p_value": stats.ttest_1samp(values, 0).pvalue,
        "cohens_dz": values.mean() / sd if sd else math.nan,
    }


def contrast_records(dataframe):
    records = []
    for metric in ("final_balance", "mean_trust_ratio", "mean_return_ratio"):
        pivot = dataframe.pivot(
            index="replicate_id", columns=["history", "task"], values=metric
        )
        task_effects = {
            history: pivot[(history, "myth_game")] - pivot[(history, "game")]
            for history in HISTORY_ORDER
        }
        for history in HISTORY_ORDER:
            records.append(
                difference_record(
                    task_effects[history],
                    metric,
                    f"Myth→Game − Game only | {HISTORY_LABELS[history]}",
                    primary=history == "public_ledger" and metric == "final_balance",
                )
            )
        records.append(
            difference_record(
                task_effects["public_ledger"] - task_effects["private"],
                metric,
                "Task interaction: public ledger − private memory",
            )
        )
        records.append(
            difference_record(
                task_effects["public_ledger"] - task_effects["dossier"],
                metric,
                "Task interaction: public ledger − partner dossier",
            )
        )
    return records


def plot_cells(dataframe, output_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    sns.boxplot(
        data=dataframe,
        x="history_label",
        y="final_balance",
        hue="task_label",
        order=[HISTORY_LABELS[value] for value in HISTORY_ORDER],
        hue_order=[TASK_LABELS[value] for value in TASK_ORDER],
        palette=[TASK_COLORS[value] for value in TASK_ORDER],
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=dataframe,
        x="history_label",
        y="final_balance",
        hue="task_label",
        order=[HISTORY_LABELS[value] for value in HISTORY_ORDER],
        hue_order=[TASK_LABELS[value] for value in TASK_ORDER],
        palette=["#263238", "#263238"],
        dodge=True,
        alpha=0.75,
        size=5,
        legend=False,
        ax=ax,
    )
    ax.set_title("History visibility × Myth→Game (GPT-5 Nano)", fontweight="bold")
    ax.set_xlabel("Decision-time history")
    ax.set_ylabel("Average final balance per agent")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Task order")
    fig.tight_layout()
    fig.savefig(output_dir / "final_balance_history_by_task.png", dpi=300)
    plt.close(fig)


def plot_task_effects(dataframe, output_dir):
    import matplotlib.pyplot as plt

    pivot = dataframe.pivot(
        index="replicate_id", columns=["history", "task"], values="final_balance"
    )
    x = np.arange(len(HISTORY_ORDER))
    effects = {
        history: pivot[(history, "myth_game")] - pivot[(history, "game")]
        for history in HISTORY_ORDER
    }
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for replicate_id in pivot.index:
        ax.plot(
            x,
            [effects[history].loc[replicate_id] for history in HISTORY_ORDER],
            color="#90a4ae",
            alpha=0.5,
            marker="o",
        )
    means = [effects[history].mean() for history in HISTORY_ORDER]
    intervals = [ci(effects[history]) for history in HISTORY_ORDER]
    yerr = np.asarray(
        [
            [mean - interval[0] for mean, interval in zip(means, intervals)],
            [interval[1] - mean for mean, interval in zip(means, intervals)],
        ]
    )
    ax.errorbar(
        x,
        means,
        yerr=yerr,
        color="#263238",
        marker="o",
        markersize=8,
        linewidth=2.5,
        capsize=5,
        label="Mean and 95% CI",
    )
    ax.axhline(0, color="#263238", linestyle="--", linewidth=1)
    ax.set_xticks(x, [HISTORY_LABELS[value] for value in HISTORY_ORDER])
    ax.set_title("Matched Myth→Game effect by history visibility", fontweight="bold")
    ax.set_xlabel("Decision-time history")
    ax.set_ylabel("Myth→Game − Game only\n(final balance per agent)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "myth_game_effect_by_history.png", dpi=300)
    plt.close(fig)


def plot_trajectories(trajectory_dataframe, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
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
    axes[2].legend(title="Task order", loc="lower right")
    fig.suptitle("Trust trajectories by history and task order", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "trust_trajectories_history_by_task.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    configure_matplotlib()
    import pandas as pd
    import seaborn as sns

    sns.set_style("whitegrid")
    args.out.mkdir(parents=True, exist_ok=True)
    rows, trajectories, tokens, attempts, retries = load_factorial()
    dataframe = pd.DataFrame(rows).sort_values(
        ["history", "task", "replicate_id"]
    )
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
    trajectory_dataframe.to_csv(args.out / "round_metrics.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    contrasts.to_csv(args.out / "paired_contrasts.csv", index=False)
    cost.to_csv(args.out / "token_cost.csv", index=False)
    plot_cells(dataframe, args.out)
    plot_task_effects(dataframe, args.out)
    plot_trajectories(trajectory_dataframe, args.out)

    print(summary[summary["metric"] == "final_balance"].to_string(index=False))
    print("\nFinal-balance contrasts:")
    print(contrasts[contrasts["metric"] == "final_balance"].to_string(index=False))
    print(f"\nAttempts: {attempts}; recovered retries: {retries}")
    print(f"Estimated list-price cost under recorded rates: ${estimated_cost:.4f}")
    print(f"Saved outputs to {args.out}")


if __name__ == "__main__":
    main()
