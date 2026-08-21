#!/usr/bin/env python3
"""Analyze the matched GPT-5 Nano game-only history-visibility screen."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyses._shared import configure_matplotlib
from analyze_corrected_v2_confirmatory import holm_adjust, load_runs, run_metrics


DEFAULT_OUTPUT = Path("docs/figures/population_ledger_gpt_n5_20260821")
ARM_ORDER = ["private", "dossier", "public_ledger"]
ARM_LABELS = {
    "private": "Private memory",
    "dossier": "Current-partner dossier",
    "public_ledger": "Public population ledger",
}
ARM_COLORS = {
    "private": "#2a9d8f",
    "dossier": "#e76f51",
    "public_ledger": "#6c5ce7",
}
ARM_PATHS = {
    "private": [
        Path(
            "data/json/noise_experiments/history_gate_signed_gpt_n5_20260821/"
            "noise8_history_gate_signed_gpt_n5_ownonly_game"
        )
    ],
    "dossier": [
        Path(
            "data/json/noise_experiments/crossmodel_signed_gpt_n5_20260821/"
            "noise8_crossmodel_signed_gpt_n5_game"
        )
    ],
    "public_ledger": [
        Path(
            "data/json/noise_experiments/population_ledger_smoke_gpt_20260821/"
            "noise8_population_ledger_signed_gpt_smoke_game"
        ),
        Path(
            "data/json/noise_experiments/population_ledger_gpt_n5_extension_20260821/"
            "noise8_population_ledger_signed_gpt_n5_extension_game"
        ),
    ],
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


def load_screen():
    rows = []
    trajectories = []
    token_totals = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    attempts = 0
    retries = 0
    expected_ids = set(range(5))

    for arm in ARM_ORDER:
        runs = []
        for path in ARM_PATHS[arm]:
            runs.extend(load_runs(path))
        replicate_ids = [
            int((run.get("run_metadata") or {})["replicate_id"])
            for _, run in runs
        ]
        if len(runs) != 5 or set(replicate_ids) != expected_ids:
            raise RuntimeError(f"Arm {arm} has IDs {sorted(replicate_ids)}")

        for path, run in runs:
            metadata = run.get("run_metadata") or {}
            if metadata.get("code_dirty"):
                raise RuntimeError(f"Dirty execution provenance in {path}")
            if metadata.get("model") != "openai/gpt-5-nano":
                raise RuntimeError(f"Unexpected model in {path}")
            metrics, points = run_metrics(path, run)
            replicate_id = int(metadata["replicate_id"])
            metrics["return_over_sent"] = (
                metrics["mean_returned"] / metrics["mean_sent"]
                if metrics["mean_sent"] > 0
                else math.nan
            )
            rows.append(
                {
                    "history_arm": arm,
                    "history_label": ARM_LABELS[arm],
                    "replicate_id": replicate_id,
                    **metrics,
                }
            )
            trajectories.extend(
                {
                    "history_arm": arm,
                    "history_label": ARM_LABELS[arm],
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
    for arm in ARM_ORDER:
        cell = dataframe[dataframe["history_arm"] == arm]
        for metric in (
            "final_balance",
            "mean_trust_ratio",
            "mean_return_ratio",
            "return_over_sent",
            "mean_sent",
            "mean_returned",
        ):
            values = cell[metric].to_numpy(dtype=float)
            low, high = ci(values)
            records.append(
                {
                    "history_arm": arm,
                    "history_label": ARM_LABELS[arm],
                    "metric": metric,
                    "n": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return records


def contrast_records(dataframe):
    records = []
    pairs = [
        ("dossier", "private"),
        ("public_ledger", "private"),
        ("public_ledger", "dossier"),
    ]
    for metric in ("final_balance", "mean_trust_ratio", "mean_return_ratio"):
        pivot = dataframe.pivot(
            index="replicate_id", columns="history_arm", values=metric
        )
        metric_records = []
        for left, right in pairs:
            differences = (pivot[left] - pivot[right]).to_numpy(dtype=float)
            low, high = ci(differences)
            sd = differences.std(ddof=1)
            metric_records.append(
                {
                    "metric": metric,
                    "contrast": f"{ARM_LABELS[left]} − {ARM_LABELS[right]}",
                    "n_pairs": len(differences),
                    "estimate": differences.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "p_value": stats.ttest_1samp(differences, 0).pvalue,
                    "cohens_dz": differences.mean() / sd if sd else math.nan,
                }
            )
        adjusted = holm_adjust([record["p_value"] for record in metric_records])
        for record, adjusted_p in zip(metric_records, adjusted):
            record["holm_p_within_metric"] = adjusted_p
            records.append(record)
    return records


def plot_final_balance(dataframe, output_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns

    order = [ARM_LABELS[arm] for arm in ARM_ORDER]
    palette = [ARM_COLORS[arm] for arm in ARM_ORDER]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sns.boxplot(
        data=dataframe,
        x="history_label",
        y="final_balance",
        order=order,
        palette=palette,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=dataframe,
        x="history_label",
        y="final_balance",
        order=order,
        color="#263238",
        size=6,
        alpha=0.8,
        ax=ax,
    )
    ax.set_title("Population-wide history visibility (GPT-5 Nano)", fontweight="bold")
    ax.set_xlabel("Decision-time history")
    ax.set_ylabel("Average final balance per agent")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "final_balance_by_history_arm.png", dpi=300)
    plt.close(fig)


def plot_paired_balances(dataframe, output_dir):
    import matplotlib.pyplot as plt

    pivot = dataframe.pivot(
        index="replicate_id", columns="history_arm", values="final_balance"
    )
    x = np.arange(len(ARM_ORDER))
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for replicate_id, row in pivot.iterrows():
        ax.plot(
            x,
            [row[arm] for arm in ARM_ORDER],
            color="#78909c",
            alpha=0.55,
            marker="o",
            label="Matched replicate" if replicate_id == pivot.index[0] else None,
        )
    means = [pivot[arm].mean() for arm in ARM_ORDER]
    intervals = [ci(pivot[arm]) for arm in ARM_ORDER]
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
    ax.set_xticks(x, [ARM_LABELS[arm] for arm in ARM_ORDER])
    ax.set_title("Matched population-level outcomes", fontweight="bold")
    ax.set_xlabel("Decision-time history")
    ax.set_ylabel("Average final balance per agent")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "paired_final_balance_by_history_arm.png", dpi=300)
    plt.close(fig)


def plot_trajectories(trajectory_dataframe, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, metric, ylabel in (
        (axes[0], "trust_ratio", "Mean proportion sent"),
        (axes[1], "return_ratio", "Mean proportion returned"),
    ):
        for arm in ARM_ORDER:
            subset = trajectory_dataframe[
                trajectory_dataframe["history_arm"] == arm
            ]
            summary = subset.groupby("round")[metric].agg(["mean", "sem"])
            rounds = summary.index.to_numpy(dtype=float)
            means = summary["mean"].to_numpy(dtype=float)
            errors = summary["sem"].to_numpy(dtype=float) * stats.t.ppf(0.975, 4)
            ax.plot(
                rounds,
                means,
                color=ARM_COLORS[arm],
                marker="o",
                linewidth=2.5,
                label=ARM_LABELS[arm],
            )
            ax.fill_between(
                rounds,
                means - errors,
                means + errors,
                color=ARM_COLORS[arm],
                alpha=0.12,
            )
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, linestyle="--")
    axes[1].legend(title="Decision-time history", loc="lower right")
    fig.suptitle("Cooperation trajectories under different history visibility", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "cooperation_trajectories_by_history_arm.png", dpi=300)
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
    rows, trajectories, tokens, attempts, retries = load_screen()
    dataframe = pd.DataFrame(rows).sort_values(
        ["history_arm", "replicate_id"]
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
    summary.to_csv(args.out / "summary.csv", index=False)
    contrasts.to_csv(args.out / "paired_contrasts.csv", index=False)
    trajectory_dataframe.to_csv(args.out / "round_metrics.csv", index=False)
    cost.to_csv(args.out / "token_cost.csv", index=False)
    plot_final_balance(dataframe, args.out)
    plot_paired_balances(dataframe, args.out)
    plot_trajectories(trajectory_dataframe, args.out)

    print(summary[summary["metric"] == "final_balance"].to_string(index=False))
    print("\nFinal-balance paired contrasts:")
    print(contrasts[contrasts["metric"] == "final_balance"].to_string(index=False))
    print(f"\nAttempts: {attempts}; recovered retries: {retries}")
    print(f"Estimated list-price cost under recorded rates: ${estimated_cost:.4f}")
    print(f"Saved outputs to {args.out}")


if __name__ == "__main__":
    main()
