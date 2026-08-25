#!/usr/bin/env python3
"""Analyze the clean paired GPT-5 Nano 8-agent replication batch.

Outputs reproducible run metrics, paired contrasts, token-cost accounting, and
three compact figures. The analysis unit is the independent eight-agent run;
the five protocol seeds are matched across task orders.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyze_corrected_v2_confirmatory import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_ORDER,
    holm_adjust,
    load_runs,
    run_metrics,
)

from analyses._shared import configure_matplotlib


DEFAULT_INPUT = Path(
    "data/json/noise_experiments/crossmodel_signed_gpt_n5_20260821"
)
DEFAULT_REPAIR_INPUT = Path(
    "data/json/noise_experiments/history_factorial_mythrepairs_20260821"
)
DEFAULT_OUTPUT = Path("docs/figures/crossmodel_signed_gpt_n5_20260821")
CELL_DIRECTORIES = {
    "game": "noise8_crossmodel_signed_gpt_n5_game",
    "game_myth": "noise8_crossmodel_signed_gpt_n5_game_myth",
    "myth_game": "noise8_crossmodel_signed_gpt_n5_myth_game",
}


def usage_records(run):
    """Yield one copy of each accepted or rejected provider attempt's usage."""
    for agent in (run.get("agents") or {}).values():
        for event in agent.get("interaction_history") or []:
            usage = ((event.get("response") or {}).get("usage") or {})
            if usage:
                yield usage


def load_batch(input_dir: Path, repair_dir: Path):
    rows = []
    trajectories = []
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

    for condition, subdirectory in CELL_DIRECTORIES.items():
        runs = load_runs(input_dir / subdirectory)
        if condition == "game_myth":
            runs = [
                item
                for item in runs
                if int((item[1].get("run_metadata") or {})["replicate_id"]) != 1
            ]
            runs.extend(
                load_runs(
                    repair_dir
                    / "noise8_crossmodel_signed_gpt_n5_game_myth_mythrepair"
                )
            )
        elif condition == "myth_game":
            runs = [
                item
                for item in runs
                if int((item[1].get("run_metadata") or {})["replicate_id"]) != 0
            ]
            runs.extend(
                load_runs(
                    repair_dir
                    / "noise8_crossmodel_signed_gpt_n5_myth_game_mythrepair"
                )
            )
        if len(runs) != 5:
            raise RuntimeError(
                f"Expected five final runs in {subdirectory}; found {len(runs)}"
            )
        for path, run in runs:
            metadata = run.get("run_metadata") or {}
            if metadata.get("code_dirty"):
                raise RuntimeError(f"Dirty execution provenance in {path}")
            if metadata.get("model") != "openai/gpt-5-nano":
                raise RuntimeError(f"Unexpected model in {path}: {metadata.get('model')}")

            metrics, run_trajectory = run_metrics(path, run)
            replicate_id = int(metadata["replicate_id"])
            metrics["return_over_sent"] = (
                metrics["mean_returned"] / metrics["mean_sent"]
                if metrics["mean_sent"] > 0
                else math.nan
            )
            rows.append(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "replicate_id": replicate_id,
                    "pairing_seed": metadata.get("pairing_seed"),
                    "noise_seed": metadata.get("noise_seed"),
                    **metrics,
                }
            )
            trajectories.extend(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "replicate_id": replicate_id,
                    **point,
                }
                for point in run_trajectory
            )
            for usage in usage_records(run):
                for key in usage_totals:
                    usage_totals[key] += int(usage.get(key) or 0)

    expected_replicates = set(range(5))
    for condition in CONDITION_ORDER:
        observed = {
            row["replicate_id"] for row in rows if row["condition"] == condition
        }
        if observed != expected_replicates:
            raise RuntimeError(
                f"Replicate mismatch for {condition}: {sorted(observed)}"
            )
    return rows, trajectories, usage_totals


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    sem = stats.sem(values)
    return stats.t.interval(0.95, len(values) - 1, loc=values.mean(), scale=sem)


def build_summary(dataframe):
    records = []
    metrics = [
        "final_balance",
        "mean_trust_ratio",
        "mean_return_ratio",
        "return_over_sent",
        "mean_sent",
        "mean_returned",
    ]
    for condition in CONDITION_ORDER:
        cell = dataframe[dataframe["condition"] == condition]
        for metric in metrics:
            values = cell[metric].to_numpy(dtype=float)
            ci_low, ci_high = confidence_interval(values)
            records.append(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "metric": metric,
                    "n": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return records


def build_paired_contrasts(dataframe):
    comparisons = [
        ("game_myth", "game"),
        ("myth_game", "game"),
        ("myth_game", "game_myth"),
    ]
    records = []
    for metric in ["final_balance", "mean_trust_ratio", "mean_return_ratio"]:
        for left, right in comparisons:
            left_values = dataframe[dataframe["condition"] == left].set_index(
                "replicate_id"
            )[metric]
            right_values = dataframe[dataframe["condition"] == right].set_index(
                "replicate_id"
            )[metric]
            differences = (left_values - right_values).sort_index().to_numpy()
            ci_low, ci_high = confidence_interval(differences)
            test = stats.ttest_1samp(differences, 0)
            records.append(
                {
                    "metric": metric,
                    "left": left,
                    "right": right,
                    "contrast": (
                        f"{CONDITION_LABELS[left]} − {CONDITION_LABELS[right]}"
                    ),
                    "n_pairs": len(differences),
                    "estimate": differences.mean(),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": test.pvalue,
                }
            )

    final_indices = [
        index
        for index, record in enumerate(records)
        if record["metric"] == "final_balance"
    ]
    adjusted = holm_adjust([records[index]["p_value"] for index in final_indices])
    for record in records:
        record["holm_p_final_balance_family"] = math.nan
    for index, adjusted_p in zip(final_indices, adjusted):
        records[index]["holm_p_final_balance_family"] = adjusted_p
    return records


def plot_final_balance(dataframe, output_dir: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    order = [CONDITION_LABELS[condition] for condition in CONDITION_ORDER]
    palette = [CONDITION_COLORS[condition] for condition in CONDITION_ORDER]
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    sns.boxplot(
        data=dataframe,
        x="condition_label",
        y="final_balance",
        hue="condition_label",
        order=order,
        palette=palette,
        legend=False,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=dataframe,
        x="condition_label",
        y="final_balance",
        order=order,
        color="#263238",
        jitter=0.08,
        size=6,
        ax=ax,
    )
    for replicate_id in range(5):
        paired = dataframe[dataframe["replicate_id"] == replicate_id].set_index(
            "condition"
        )
        ax.plot(
            range(3),
            [paired.loc[condition, "final_balance"] for condition in CONDITION_ORDER],
            color="#607d8b",
            alpha=0.35,
            linewidth=1,
            zorder=0,
        )
    ax.set_title(
        "GPT-5 Nano: Myth→Game raises 8-agent cooperation",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Task order")
    ax.set_ylabel("Average final balance per agent")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "final_balance_gpt_n5.png", dpi=300)
    plt.close(fig)


def plot_behavior(dataframe, output_dir: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    order = [CONDITION_LABELS[condition] for condition in CONDITION_ORDER]
    palette = [CONDITION_COLORS[condition] for condition in CONDITION_ORDER]
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.5))
    panels = [
        ("mean_trust_ratio", "Proportion sent", (0, 1)),
        ("mean_return_ratio", "Proportion of tripled amount returned", (0, 1)),
        ("return_over_sent", "Dollars returned / dollars sent", (0, None)),
    ]
    for ax, (metric, title, limits) in zip(axes, panels):
        sns.boxplot(
            data=dataframe,
            x="condition_label",
            y=metric,
            hue="condition_label",
            order=order,
            palette=palette,
            legend=False,
            fliersize=0,
            ax=ax,
        )
        sns.stripplot(
            data=dataframe,
            x="condition_label",
            y=metric,
            order=order,
            color="#263238",
            jitter=0.08,
            size=5,
            ax=ax,
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=12)
        ax.set_ylim(*limits)
        ax.grid(True, axis="y", alpha=0.3)
    axes[2].axhline(1, color="#546e7a", linestyle="--", linewidth=1)
    fig.suptitle("GPT-5 Nano cooperation metrics (n=5 paired runs)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "behavior_metrics_gpt_n5.png", dpi=300)
    plt.close(fig)


def plot_trajectories(trajectory_dataframe, output_dir: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for condition in CONDITION_ORDER:
        condition_data = trajectory_dataframe[
            trajectory_dataframe["condition"] == condition
        ]
        for _, run_data in condition_data.groupby("replicate_id"):
            run_data = run_data.sort_values("round")
            ax.plot(
                run_data["round"],
                run_data["trust_ratio"],
                color=CONDITION_COLORS[condition],
                alpha=0.18,
                linewidth=1,
            )
        mean_data = condition_data.groupby("round", as_index=False)[
            "trust_ratio"
        ].mean()
        ax.plot(
            mean_data["round"],
            mean_data["trust_ratio"],
            color=CONDITION_COLORS[condition],
            linewidth=3,
            marker="o",
            label=CONDITION_LABELS[condition],
        )
    ax.set_title("GPT-5 Nano trust trajectories", fontsize=15, fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean proportion sent")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(title="Task order", loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "trust_trajectories_gpt_n5.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--repairs", type=Path, default=DEFAULT_REPAIR_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    configure_matplotlib()
    import pandas as pd
    import seaborn as sns

    sns.set_style("whitegrid")
    args.out.mkdir(parents=True, exist_ok=True)
    rows, trajectories, usage_totals = load_batch(args.input, args.repairs)
    dataframe = pd.DataFrame(rows).sort_values(["condition", "replicate_id"])
    trajectory_dataframe = pd.DataFrame(trajectories)
    summary = pd.DataFrame(build_summary(dataframe))
    contrasts = pd.DataFrame(build_paired_contrasts(dataframe))

    estimated_cost = (
        usage_totals["input_tokens"] / 1_000_000 * 0.05
        + (usage_totals["output_tokens"] + usage_totals["reasoning_tokens"])
        / 1_000_000
        * 0.40
    )
    cost = pd.DataFrame([{**usage_totals, "estimated_list_price_usd": estimated_cost}])

    dataframe.to_csv(args.out / "run_metrics.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    contrasts.to_csv(args.out / "paired_contrasts.csv", index=False)
    cost.to_csv(args.out / "token_cost.csv", index=False)
    plot_final_balance(dataframe, args.out)
    plot_behavior(dataframe, args.out)
    plot_trajectories(trajectory_dataframe, args.out)

    print(summary[summary["metric"].isin(["final_balance", "mean_trust_ratio", "mean_return_ratio"])].to_string(index=False))
    print("\nPaired final-balance contrasts:")
    print(contrasts[contrasts["metric"] == "final_balance"].to_string(index=False))
    print(f"\nEstimated list-price cost: ${estimated_cost:.4f}")
    print(f"Saved outputs to {args.out}")


if __name__ == "__main__":
    main()
