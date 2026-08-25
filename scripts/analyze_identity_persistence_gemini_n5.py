#!/usr/bin/env python3
"""Analyze the frozen Gemini identity-persistence replication screen."""

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
    "data/json/noise_experiments/identity_persistence_gemini_screen_20260821"
)
DEFAULT_GPT_CONTRASTS = Path(
    "docs/figures/identity_persistence_confirmation_gpt_n10_20260821/"
    "paired_contrasts.csv"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/identity_persistence_gemini_n5_20260821"
)
ARM_ORDER = ["relative", "stable"]
ARM_LABELS = {
    "relative": "Round-local pair IDs",
    "stable": "Persistent stable IDs",
}
ARM_COLORS = {"relative": "#457b9d", "stable": "#e76f51"}
CELL_DIRS = {
    "relative": "noise8_identity_persistence_gemini_n5_relative",
    "stable": "noise8_identity_persistence_gemini_n5_stable",
}


def ci(values):
    values = np.asarray(values, dtype=float)
    if np.allclose(values, values[0]):
        return values.mean(), values.mean()
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


def load_screen(input_dir):
    rows = []
    trajectories = []
    tokens = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    attempts = 0
    retries = 0
    expected_ids = set(range(25, 30))
    for arm in ARM_ORDER:
        runs = load_runs(input_dir / CELL_DIRS[arm])
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
            if metadata.get("model") != "google/gemini-3.1-flash-lite":
                raise RuntimeError(f"Unexpected model in {path}: {metadata.get('model')}")
            metrics, points = run_metrics(path, run)
            replicate_id = int(metadata["replicate_id"])
            metrics["return_over_sent"] = (
                metrics["mean_returned"] / metrics["mean_sent"]
                if metrics["mean_sent"] > 0
                else math.nan
            )
            rows.append(
                {
                    "identity_arm": arm,
                    "identity_label": ARM_LABELS[arm],
                    "replicate_id": replicate_id,
                    **metrics,
                }
            )
            trajectories.extend(
                {
                    "identity_arm": arm,
                    "identity_label": ARM_LABELS[arm],
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


def summary_records(dataframe, trajectory_dataframe):
    records = []
    for arm in ARM_ORDER:
        cell = dataframe[dataframe["identity_arm"] == arm]
        round_one = trajectory_dataframe[
            (trajectory_dataframe["identity_arm"] == arm)
            & (trajectory_dataframe["round"] == 1)
        ]["trust_ratio"].to_numpy(dtype=float)
        metrics = {
            "round1_trust_ratio": round_one,
            "final_balance": cell["final_balance"].to_numpy(dtype=float),
            "mean_trust_ratio": cell["mean_trust_ratio"].to_numpy(dtype=float),
            "mean_return_ratio": cell["mean_return_ratio"].to_numpy(dtype=float),
            "return_over_sent": cell["return_over_sent"].to_numpy(dtype=float),
        }
        for metric, values in metrics.items():
            low, high = ci(values)
            records.append(
                {
                    "identity_arm": arm,
                    "identity_label": ARM_LABELS[arm],
                    "metric": metric,
                    "n": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return records


def contrast_record(values, metric, primary=False):
    values = np.asarray(values, dtype=float)
    low, high = ci(values)
    sd = values.std(ddof=1)
    exact_zero = np.allclose(values, 0)
    return {
        "metric": metric,
        "contrast": "Persistent stable IDs − round-local pair IDs",
        "primary": primary,
        "n_pairs": len(values),
        "estimate": values.mean(),
        "ci_low": low,
        "ci_high": high,
        "p_value": 1.0 if exact_zero else stats.ttest_1samp(values, 0).pvalue,
        "cohens_dz": 0.0 if exact_zero else values.mean() / sd if sd else math.nan,
    }


def contrast_records(dataframe, trajectory_dataframe):
    round_one = trajectory_dataframe[trajectory_dataframe["round"] == 1].pivot(
        index="replicate_id", columns="identity_arm", values="trust_ratio"
    )
    metrics = {
        "round1_trust_ratio": round_one,
        "final_balance": dataframe.pivot(
            index="replicate_id", columns="identity_arm", values="final_balance"
        ),
        "mean_trust_ratio": dataframe.pivot(
            index="replicate_id", columns="identity_arm", values="mean_trust_ratio"
        ),
        "mean_return_ratio": dataframe.pivot(
            index="replicate_id", columns="identity_arm", values="mean_return_ratio"
        ),
        "return_over_sent": dataframe.pivot(
            index="replicate_id", columns="identity_arm", values="return_over_sent"
        ),
    }
    return [
        contrast_record(
            pivot["stable"] - pivot["relative"],
            metric,
            primary=metric == "final_balance",
        )
        for metric, pivot in metrics.items()
    ]


def plot_cells(dataframe, output_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    sns.boxplot(
        data=dataframe,
        x="identity_label",
        y="final_balance",
        hue="identity_label",
        order=[ARM_LABELS[arm] for arm in ARM_ORDER],
        hue_order=[ARM_LABELS[arm] for arm in ARM_ORDER],
        palette=[ARM_COLORS[arm] for arm in ARM_ORDER],
        legend=False,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=dataframe,
        x="identity_label",
        y="final_balance",
        order=[ARM_LABELS[arm] for arm in ARM_ORDER],
        color="#263238",
        size=6,
        alpha=0.8,
        ax=ax,
    )
    ax.set_title("Gemini identity-persistence replication screen", fontweight="bold")
    ax.set_xlabel("Identity context")
    ax.set_ylabel("Average final balance per agent")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "gemini_final_balance.png", dpi=300)
    plt.close(fig)


def plot_model_contrasts(contrasts, gpt_contrasts_path, output_dir):
    import matplotlib.pyplot as plt
    import pandas as pd

    gemini = contrasts[contrasts["metric"] == "final_balance"].iloc[0]
    gpt_all = pd.read_csv(gpt_contrasts_path)
    gpt = gpt_all[gpt_all["metric"] == "final_balance"].iloc[0]
    labels = ["GPT-5 Nano\n(n=10 pairs)", "Gemini 3.1 Flash-Lite\n(n=5 pairs)"]
    estimates = np.array([gpt["estimate"], gemini["estimate"]], dtype=float)
    lows = np.array([gpt["ci_low"], gemini["ci_low"]], dtype=float)
    highs = np.array([gpt["ci_high"], gemini["ci_high"]], dtype=float)
    positions = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.errorbar(
        positions,
        estimates,
        yerr=np.vstack([estimates - lows, highs - estimates]),
        fmt="o",
        markersize=9,
        capsize=6,
        linewidth=2.5,
        color="#9c2f1f",
    )
    ax.axhline(0, color="#263238", linestyle="--", linewidth=1)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Persistent − round-local IDs\n(final balance per agent)")
    ax.set_title("Identity-persistence effect by model", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "model_contrast_comparison.png", dpi=300)
    plt.close(fig)


def plot_trajectories(trajectory_dataframe, output_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for arm in ARM_ORDER:
        subset = trajectory_dataframe[trajectory_dataframe["identity_arm"] == arm]
        grouped = subset.groupby("round")["trust_ratio"].agg(["mean", "sem"])
        rounds = grouped.index.to_numpy(dtype=float)
        means = grouped["mean"].to_numpy(dtype=float)
        errors = grouped["sem"].to_numpy(dtype=float) * stats.t.ppf(0.975, 4)
        ax.plot(
            rounds,
            means,
            marker="o",
            linewidth=2.5,
            color=ARM_COLORS[arm],
            label=ARM_LABELS[arm],
        )
        ax.fill_between(
            rounds,
            means - errors,
            means + errors,
            color=ARM_COLORS[arm],
            alpha=0.14,
        )
    ax.set_ylim(0, 1)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean proportion sent")
    ax.set_title("Gemini sending trajectories by identity persistence", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Identity context")
    fig.tight_layout()
    fig.savefig(output_dir / "gemini_trust_trajectories.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--gpt-contrasts", type=Path, default=DEFAULT_GPT_CONTRASTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    import pandas as pd

    rows, trajectories, tokens, attempts, retries = load_screen(args.input)
    dataframe = pd.DataFrame(rows).sort_values(["identity_arm", "replicate_id"])
    trajectory_dataframe = pd.DataFrame(trajectories).sort_values(
        ["identity_arm", "replicate_id", "round"]
    )
    summary = pd.DataFrame(summary_records(dataframe, trajectory_dataframe))
    contrasts = pd.DataFrame(contrast_records(dataframe, trajectory_dataframe))
    token_usage = pd.DataFrame(
        [{**tokens, "total_attempts": attempts, "recovered_retries": retries}]
    )

    dataframe.to_csv(args.out / "run_metrics.csv", index=False)
    trajectory_dataframe.to_csv(args.out / "round_metrics.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    contrasts.to_csv(args.out / "paired_contrasts.csv", index=False)
    token_usage.to_csv(args.out / "token_usage.csv", index=False)
    plot_cells(dataframe, args.out)
    plot_model_contrasts(contrasts, args.gpt_contrasts, args.out)
    plot_trajectories(trajectory_dataframe, args.out)

    print(summary.to_string(index=False))
    print("\nFrozen Gemini contrasts:")
    print(contrasts.to_string(index=False))
    print(f"\nAttempts: {attempts}; recovered retries: {retries}")
    print(f"Recorded usage: {tokens}")
    print(f"Saved outputs to {args.out}")


if __name__ == "__main__":
    main()
