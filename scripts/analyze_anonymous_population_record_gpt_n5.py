#!/usr/bin/env python3
"""Analyze anonymous social information versus persistent reputation tracking."""

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
from analyze_stable_ids_control_gpt_n5 import selected_runs as prior_selected_runs


DEFAULT_OUTPUT = Path(
    "docs/figures/anonymous_population_record_gpt_n5_20260821"
)
ARM_ORDER = ["private", "anonymous_record", "stable_ids", "public_ledger"]
ARM_LABELS = {
    "private": "Anonymous\nprivate memory",
    "anonymous_record": "Anonymous\npopulation record",
    "stable_ids": "Stable IDs,\nno ledger",
    "public_ledger": "Stable IDs +\npublic ledger",
}
ARM_COLORS = {
    "private": "#2a9d8f",
    "anonymous_record": "#457b9d",
    "stable_ids": "#e9c46a",
    "public_ledger": "#6c5ce7",
}
ANONYMOUS_RECORD_PATH = Path(
    "data/json/noise_experiments/anonymous_record_gpt_n5_20260821/"
    "noise8_anonymous_record_signed_gpt_n5_game"
)


def ci(values):
    values = np.asarray(values, dtype=float)
    return stats.t.interval(
        0.95,
        len(values) - 1,
        loc=values.mean(),
        scale=stats.sem(values),
    )


def selected_runs():
    cells = prior_selected_runs()
    cells["anonymous_record"] = load_runs(ANONYMOUS_RECORD_PATH)
    return cells


def load_screen():
    rows = []
    trajectories = []
    expected_ids = set(range(5))
    for arm, runs in selected_runs().items():
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
            metrics, points = run_metrics(path, run)
            replicate_id = int(metadata["replicate_id"])
            metrics["return_over_sent"] = (
                metrics["mean_returned"] / metrics["mean_sent"]
                if metrics["mean_sent"] > 0
                else math.nan
            )
            rows.append(
                {
                    "information_arm": arm,
                    "information_label": ARM_LABELS[arm].replace("\n", " "),
                    "replicate_id": replicate_id,
                    **metrics,
                }
            )
            trajectories.extend(
                {
                    "information_arm": arm,
                    "information_label": ARM_LABELS[arm].replace("\n", " "),
                    "replicate_id": replicate_id,
                    **point,
                }
                for point in points
            )
    return rows, trajectories


def summary_records(dataframe, trajectory_dataframe):
    records = []
    for arm in ARM_ORDER:
        cell = dataframe[dataframe["information_arm"] == arm]
        round_one = trajectory_dataframe[
            (trajectory_dataframe["information_arm"] == arm)
            & (trajectory_dataframe["round"] == 1)
        ]["trust_ratio"].to_numpy(dtype=float)
        metrics = {
            "round1_trust_ratio": round_one,
            "final_balance": cell["final_balance"].to_numpy(dtype=float),
            "mean_trust_ratio": cell["mean_trust_ratio"].to_numpy(dtype=float),
            "mean_return_ratio": cell["mean_return_ratio"].to_numpy(dtype=float),
        }
        for metric, values in metrics.items():
            low, high = ci(values)
            records.append(
                {
                    "information_arm": arm,
                    "information_label": ARM_LABELS[arm].replace("\n", " "),
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


def contrast_records(dataframe, trajectory_dataframe):
    records = []
    round_one = trajectory_dataframe[trajectory_dataframe["round"] == 1].pivot(
        index="replicate_id", columns="information_arm", values="trust_ratio"
    )
    metric_values = {
        "round1_trust_ratio": round_one,
        "final_balance": dataframe.pivot(
            index="replicate_id",
            columns="information_arm",
            values="final_balance",
        ),
        "mean_trust_ratio": dataframe.pivot(
            index="replicate_id",
            columns="information_arm",
            values="mean_trust_ratio",
        ),
        "mean_return_ratio": dataframe.pivot(
            index="replicate_id",
            columns="information_arm",
            values="mean_return_ratio",
        ),
    }
    pairs = [
        ("anonymous_record", "private", "Anonymous record − private memory"),
        ("anonymous_record", "stable_ids", "Anonymous record − stable IDs"),
        ("public_ledger", "anonymous_record", "Public ledger − anonymous record"),
    ]
    for metric, pivot in metric_values.items():
        for left, right, label in pairs:
            records.append(
                difference_record(
                    pivot[left] - pivot[right],
                    metric,
                    label,
                    primary=(
                        metric == "final_balance"
                        and left == "anonymous_record"
                        and right == "private"
                    ),
                )
            )
    return records


def paired_plot(dataframe, value, ylabel, title, filename, output_dir):
    import matplotlib.pyplot as plt

    pivot = dataframe.pivot(
        index="replicate_id", columns="information_arm", values=value
    )
    x = np.arange(len(ARM_ORDER))
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    for _, row in pivot.iterrows():
        ax.plot(
            x,
            [row[arm] for arm in ARM_ORDER],
            color="#90a4ae",
            alpha=0.5,
            marker="o",
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
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Population information and identity condition")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)


def plot_trajectories(trajectory_dataframe, output_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    for arm in ARM_ORDER:
        subset = trajectory_dataframe[trajectory_dataframe["information_arm"] == arm]
        means = subset.groupby("round", as_index=False)["trust_ratio"].mean()
        ax.plot(
            means["round"],
            means["trust_ratio"],
            color=ARM_COLORS[arm],
            marker="o",
            linewidth=2.5,
            label=ARM_LABELS[arm].replace("\n", " "),
        )
    ax.set_title("Social information versus persistent reputation tracking", fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean proportion sent")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(title="Condition", loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "trust_trajectories.png", dpi=300)
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
    rows, trajectories = load_screen()
    dataframe = pd.DataFrame(rows).sort_values(
        ["information_arm", "replicate_id"]
    )
    trajectory_dataframe = pd.DataFrame(trajectories)
    summary = pd.DataFrame(summary_records(dataframe, trajectory_dataframe))
    contrasts = pd.DataFrame(contrast_records(dataframe, trajectory_dataframe))

    dataframe.to_csv(args.out / "run_metrics.csv", index=False)
    trajectory_dataframe.to_csv(args.out / "round_metrics.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    contrasts.to_csv(args.out / "paired_contrasts.csv", index=False)

    paired_plot(
        trajectory_dataframe[trajectory_dataframe["round"] == 1],
        "trust_ratio",
        "Mean proportion sent in round 1",
        "Round-one framing diagnostic",
        "paired_round1_trust.png",
        args.out,
    )
    paired_plot(
        dataframe,
        "final_balance",
        "Average final balance per agent",
        "Matched population-level outcomes",
        "paired_final_balance.png",
        args.out,
    )
    paired_plot(
        dataframe,
        "mean_return_ratio",
        "Mean proportion returned",
        "Receiver behavior across information conditions",
        "paired_return_ratio.png",
        args.out,
    )
    plot_trajectories(trajectory_dataframe, args.out)

    print(summary.to_string(index=False))
    print("\nPlanned contrasts:")
    print(contrasts.to_string(index=False))
    print(f"Saved outputs to {args.out}")


if __name__ == "__main__":
    main()
