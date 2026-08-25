#!/usr/bin/env python3
"""Analyze the frozen independent cross-model defector confirmation."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyses._shared import configure_matplotlib
import analyze_defector_myth_game_crossmodel_n5 as pilot


DEFAULT_INPUT = Path(
    "data/json/noise_experiments/defector_myth_game_crossmodel_confirm_20260821"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/defector_myth_game_crossmodel_confirmation_n10_20260821"
)
CONFIRM_SPECS = {
    "gpt": {
        **pilot.MODEL_SPECS["gpt"],
        "directory": "noise8i_defector_myth_game_gpt_confirm_n10",
    },
    "gemini": {
        **pilot.MODEL_SPECS["gemini"],
        "directory": "noise8i_defector_myth_game_gemini_confirm_n10",
    },
}


def model_differences(run_dataframe, myth_aggregates):
    differences = {}
    for model_id in CONFIRM_SPECS:
        model_runs = run_dataframe[run_dataframe["model_id"] == model_id]
        behavior = model_runs.pivot(
            index="replicate_id",
            columns="condition",
            values="standard_send_ratio",
        )
        standard_myths = myth_aggregates[
            (myth_aggregates["model_id"] == model_id)
            & (myth_aggregates["author_type"] == "standard")
        ]
        culture = standard_myths.pivot(
            index="replicate_id",
            columns="condition",
            values="coop_density",
        )
        differences[model_id] = {
            "standard_send_ratio": (
                behavior["defectors25"] - behavior["control"]
            ).to_numpy(dtype=float),
            "standard_myth_coop_density": (
                culture["defectors25"] - culture["control"]
            ).to_numpy(dtype=float),
        }
    return differences


def stratified_test(gpt_values, gemini_values, outcome):
    gpt_values = np.asarray(gpt_values, dtype=float)
    gemini_values = np.asarray(gemini_values, dtype=float)
    means = np.array([gpt_values.mean(), gemini_values.mean()])
    variances = np.array(
        [gpt_values.var(ddof=1), gemini_values.var(ddof=1)], dtype=float
    )
    ns = np.array([len(gpt_values), len(gemini_values)], dtype=float)
    estimate = means.mean()
    variance_terms = variances / ns
    standard_error = math.sqrt(variance_terms.sum()) / 2
    numerator = variance_terms.sum() ** 2
    denominator = np.sum((variance_terms**2) / (ns - 1))
    degrees_freedom = numerator / denominator if denominator else math.inf
    statistic = estimate / standard_error if standard_error else math.nan
    p_value = (
        2 * stats.t.sf(abs(statistic), degrees_freedom)
        if standard_error
        else 1.0
    )
    critical = stats.t.ppf(0.975, degrees_freedom)
    return {
        "outcome": outcome,
        "contrast": "2 of 8 defectors − no defectors",
        "estimate": estimate,
        "standard_error": standard_error,
        "degrees_freedom": degrees_freedom,
        "ci_low": estimate - critical * standard_error,
        "ci_high": estimate + critical * standard_error,
        "raw_p_value": p_value,
        "gpt_estimate": means[0],
        "gemini_estimate": means[1],
        "n_pairs_per_model": int(ns[0]),
    }


def holm_adjust(records):
    ordered = sorted(range(len(records)), key=lambda index: records[index]["raw_p_value"])
    running = 0.0
    total = len(records)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * records[index]["raw_p_value"])
        running = max(running, adjusted)
        records[index]["holm_p_value"] = running
    for record in records:
        record["confirmed"] = bool(
            record["estimate"] < 0 and record["holm_p_value"] < 0.05
        )
    return records


def plot_confirmatory_tests(confirmatory, model_contrasts, output_dir):
    import matplotlib.pyplot as plt

    outcome_specs = [
        ("standard_send_ratio", "Ordinary-agent sending"),
        ("standard_myth_coop_density", "Ordinary-agent myth language"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    for ax, (outcome, title) in zip(axes, outcome_specs):
        combined = next(row for row in confirmatory if row["outcome"] == outcome)
        if outcome == "standard_send_ratio":
            metric_name = "standard_send_ratio"
        else:
            metric_name = "standard_myth_coop_density"
        selected = model_contrasts[model_contrasts["metric"] == metric_name]
        labels = ["GPT-5 Nano", "Gemini Flash-Lite", "Equal-model average"]
        estimates = list(selected["estimate"].to_numpy(dtype=float)) + [
            combined["estimate"]
        ]
        lows = list(selected["ci_low"].to_numpy(dtype=float)) + [combined["ci_low"]]
        highs = list(selected["ci_high"].to_numpy(dtype=float)) + [combined["ci_high"]]
        colors = [pilot.MODEL_SPECS[row]["color"] for row in selected["model_id"]]
        colors.append("#263238")
        positions = np.arange(3)
        for position, estimate, low, high, color in zip(
            positions, estimates, lows, highs, colors
        ):
            ax.errorbar(
                position,
                estimate,
                yerr=[[estimate - low], [high - estimate]],
                fmt="o",
                markersize=8,
                capsize=5,
                linewidth=2.3,
                color=color,
            )
        ax.axhline(0, color="#6c757d", linestyle="--", linewidth=1)
        ax.set_xticks(positions, labels, rotation=12)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Treatment − control proportion sent")
    axes[1].set_ylabel("Treatment − control matches per 100 words")
    fig.suptitle("Independent defector confirmation", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "confirmatory_effects.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    import pandas as pd

    run_rows, round_rows, myth_rows, usage_rows = pilot.load_data(
        args.input,
        model_specs=CONFIRM_SPECS,
        expected_ids=set(range(35, 45)),
    )
    runs = pd.DataFrame(run_rows).sort_values(
        ["model_id", "condition", "replicate_id"]
    )
    rounds = pd.DataFrame(round_rows).sort_values(
        ["model_id", "condition", "replicate_id", "round"]
    )
    myths = pd.DataFrame(myth_rows).sort_values(
        ["model_id", "condition", "replicate_id", "round", "agent_id"]
    )
    myth_aggregates = pilot.aggregate_myths(myths)
    summaries = pd.DataFrame(pilot.summary_records(runs, myth_aggregates))
    model_contrasts = pd.DataFrame(
        pilot.contrast_records(runs, myth_aggregates, myths)
    )
    usage_dataframe = pd.DataFrame(usage_rows).sort_values(
        ["model_id", "condition", "replicate_id"]
    )

    differences = model_differences(runs, myth_aggregates)
    confirmatory = holm_adjust(
        [
            stratified_test(
                differences["gpt"]["standard_send_ratio"],
                differences["gemini"]["standard_send_ratio"],
                "standard_send_ratio",
            ),
            stratified_test(
                differences["gpt"]["standard_myth_coop_density"],
                differences["gemini"]["standard_myth_coop_density"],
                "standard_myth_coop_density",
            ),
        ]
    )
    confirmatory_dataframe = pd.DataFrame(confirmatory)

    runs.to_csv(args.out / "run_metrics.csv", index=False)
    rounds.to_csv(args.out / "round_metrics.csv", index=False)
    myths.to_csv(args.out / "myth_metrics.csv", index=False)
    myth_aggregates.to_csv(args.out / "myth_population_metrics.csv", index=False)
    summaries.to_csv(args.out / "summary.csv", index=False)
    model_contrasts.to_csv(args.out / "model_paired_contrasts.csv", index=False)
    confirmatory_dataframe.to_csv(args.out / "confirmatory_tests.csv", index=False)
    usage_dataframe.to_csv(args.out / "token_usage.csv", index=False)

    pilot.plot_behavior_trajectories(rounds, args.out)
    pilot.plot_behavior_contrasts(model_contrasts, args.out)
    pilot.plot_myth_contrasts(model_contrasts, args.out)
    pilot.plot_myth_author_trajectories(myths, args.out)
    plot_confirmatory_tests(confirmatory, model_contrasts, args.out)

    print("Confirmatory tests:")
    print(confirmatory_dataframe.to_string(index=False))
    print("\nModel-specific and secondary contrasts:")
    print(model_contrasts.to_string(index=False))
    print("\nToken/attempt totals by model:")
    print(
        usage_dataframe.groupby("model_id")[[
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "attempts",
            "recovered_retries",
            "forced_responses",
        ]]
        .sum()
        .to_string()
    )
    print(f"\nSaved outputs to {args.out}")


if __name__ == "__main__":
    main()
