#!/usr/bin/env python3
"""Analyze the frozen GPT-5 Nano deduction-stage calibration."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyses._shared import configure_matplotlib


DEFAULT_INPUT = Path(
    "data/json/noise_experiments/"
    "punishment_comprehension_gpt_20260821/results.json"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/punishment_comprehension_gpt_20260821"
)
VARIANT_ORDER = ["current", "cost_salient"]
VARIANT_LABELS = {
    "current": "Current wording",
    "cost_salient": "Cost-salient clarification",
}
COLORS = {"current": "#457b9d", "cost_salient": "#c14953"}


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    if len(values) == 1 or np.allclose(values, values[0]):
        return float(values.mean()), float(values.mean())
    low, high = stats.t.interval(
        0.95,
        len(values) - 1,
        loc=values.mean(),
        scale=stats.sem(values),
    )
    return float(low), float(high)


def load_trials(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata") or {}
    trials = payload.get("trials") or []
    issues = []
    if len(trials) != 100:
        issues.append(f"found {len(trials)} trials; expected 100")
    if metadata.get("execution_provenance_version") != 1:
        issues.append("missing execution provenance")
    if not metadata.get("code_commit") or metadata.get("code_dirty"):
        issues.append("dirty or incomplete Git provenance")
    if metadata.get("llm_provider") != "openai":
        issues.append("calibration did not use direct OpenAI")
    if metadata.get("provider_model") != "gpt-5-nano":
        issues.append("unexpected provider model")

    rows = []
    cell_messages = {}
    for trial in trials:
        variant = trial.get("variant")
        return_ratio = float(trial.get("return_ratio"))
        spent = trial.get("deduction_spent")
        if not trial.get("success") or spent not in (0, 1, 2):
            issues.append(f"invalid accepted trial {trial.get('trial_id')}")
            continue
        messages = json.dumps(trial.get("messages"), sort_keys=True)
        cell_messages.setdefault((variant, return_ratio), set()).add(messages)
        attempts = trial.get("attempts") or []
        rows.append(
            {
                "trial_id": trial["trial_id"],
                "call_order": int(trial["call_order"]),
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "return_ratio": return_ratio,
                "replicate": int(trial["replicate"]),
                "deduction_spent": int(spent),
                "any_deduction": float(spent > 0),
                "max_deduction": float(spent == 2),
                "attempts": len(attempts),
                "input_tokens": sum(
                    int(((item.get("response") or {}).get("usage") or {}).get("input_tokens") or 0)
                    for item in attempts
                ),
                "output_tokens": sum(
                    int(((item.get("response") or {}).get("usage") or {}).get("output_tokens") or 0)
                    for item in attempts
                ),
                "reasoning_tokens": sum(
                    int(((item.get("response") or {}).get("usage") or {}).get("reasoning_tokens") or 0)
                    for item in attempts
                ),
            }
        )
    expected_cells = {
        (variant, ratio)
        for variant in VARIANT_ORDER
        for ratio in (0.0, 0.1, 0.25, 0.5, 0.75)
    }
    observed_cells = {(row["variant"], row["return_ratio"]) for row in rows}
    if observed_cells != expected_cells:
        issues.append(f"wrong cells: {observed_cells}")
    for cell in expected_cells:
        if sum(
            row["variant"] == cell[0] and row["return_ratio"] == cell[1]
            for row in rows
        ) != 10:
            issues.append(f"cell {cell} does not have 10 trials")
        if len(cell_messages.get(cell, set())) != 1:
            issues.append(f"cell {cell} does not have identical messages")
    if issues:
        raise RuntimeError("; ".join(issues))
    return metadata, rows


def cell_summary(dataframe):
    records = []
    for variant in VARIANT_ORDER:
        for return_ratio in sorted(dataframe["return_ratio"].unique()):
            cell = dataframe[
                (dataframe["variant"] == variant)
                & (dataframe["return_ratio"] == return_ratio)
            ]
            values = cell["deduction_spent"].to_numpy(dtype=float)
            low, high = confidence_interval(values)
            records.append(
                {
                    "variant": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "return_ratio": return_ratio,
                    "n": len(values),
                    "mean_deduction": values.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "any_deduction": cell["any_deduction"].mean(),
                    "max_deduction": cell["max_deduction"].mean(),
                }
            )
    return records


def independent_difference(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    estimate = left.mean() - right.mean()
    se = math.sqrt(left.var(ddof=1) / len(left) + right.var(ddof=1) / len(right))
    numerator = (left.var(ddof=1) / len(left) + right.var(ddof=1) / len(right)) ** 2
    denominator = (
        (left.var(ddof=1) / len(left)) ** 2 / (len(left) - 1)
        + (right.var(ddof=1) / len(right)) ** 2 / (len(right) - 1)
    )
    df = numerator / denominator if denominator else math.inf
    critical = stats.t.ppf(0.975, df) if math.isfinite(df) else stats.norm.ppf(0.975)
    return {
        "estimate": estimate,
        "ci_low": estimate - critical * se,
        "ci_high": estimate + critical * se,
        "p_value": stats.ttest_ind(left, right, equal_var=False).pvalue,
    }


def interaction_ols(dataframe):
    ratio = dataframe["return_ratio"].to_numpy(dtype=float)
    salient = (dataframe["variant"] == "cost_salient").to_numpy(dtype=float)
    y = dataframe["deduction_spent"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(y)), ratio, salient, ratio * salient])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ beta
    df = len(y) - design.shape[1]
    covariance = (residuals @ residuals / df) * np.linalg.inv(design.T @ design)
    se = math.sqrt(covariance[3, 3])
    critical = stats.t.ppf(0.975, df)
    t_value = beta[3] / se
    return {
        "estimate": beta[3],
        "ci_low": beta[3] - critical * se,
        "ci_high": beta[3] + critical * se,
        "p_value": 2 * stats.t.sf(abs(t_value), df),
    }


def diagnostics(dataframe, summary_dataframe):
    records = []
    separation = {}
    gate = {}
    for variant in VARIANT_ORDER:
        cell = dataframe[dataframe["variant"] == variant]
        slope = stats.linregress(cell["return_ratio"], cell["deduction_spent"])
        spearman = stats.spearmanr(cell["return_ratio"], cell["deduction_spent"])
        low = cell[cell["return_ratio"].isin([0.0, 0.1])][
            "deduction_spent"
        ].to_numpy(dtype=float)
        high = cell[cell["return_ratio"].isin([0.5, 0.75])][
            "deduction_spent"
        ].to_numpy(dtype=float)
        low_high = independent_difference(low, high)
        separation[variant] = low_high["estimate"]
        high_any = cell[cell["return_ratio"].isin([0.5, 0.75])][
            "any_deduction"
        ].mean()
        means = summary_dataframe[summary_dataframe["variant"] == variant].sort_values(
            "return_ratio"
        )["mean_deduction"].to_numpy(dtype=float)
        reversals = np.diff(means)
        positive_reversals = reversals[reversals > 0]
        monotonic_gate = len(positive_reversals) <= 1 and (
            not len(positive_reversals) or positive_reversals.max() <= 0.2
        )
        passed = (
            low_high["estimate"] >= 0.5
            and high_any <= 0.25
            and monotonic_gate
        )
        gate[variant] = passed
        records.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "linear_slope": slope.slope,
                "slope_p_value": slope.pvalue,
                "spearman_rho": spearman.statistic,
                "spearman_p_value": spearman.pvalue,
                "low_minus_high": low_high["estimate"],
                "low_minus_high_ci_low": low_high["ci_low"],
                "low_minus_high_ci_high": low_high["ci_high"],
                "low_minus_high_p_value": low_high["p_value"],
                "high_return_any_deduction": high_any,
                "positive_adjacent_reversals": len(positive_reversals),
                "largest_positive_reversal": (
                    positive_reversals.max() if len(positive_reversals) else 0.0
                ),
                "passes_selectivity_gate": passed,
            }
        )
    eligible = (
        gate["cost_salient"]
        and separation["cost_salient"] - separation["current"] >= 0.5
    )
    return records, eligible


def plot_calibration(summary_dataframe, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    for variant in VARIANT_ORDER:
        cell = summary_dataframe[summary_dataframe["variant"] == variant].sort_values(
            "return_ratio"
        )
        axes[0].errorbar(
            cell["return_ratio"] * 100,
            cell["mean_deduction"],
            yerr=np.vstack(
                [
                    cell["mean_deduction"] - cell["ci_low"],
                    cell["ci_high"] - cell["mean_deduction"],
                ]
            ),
            marker="o",
            linewidth=2.5,
            capsize=5,
            color=COLORS[variant],
            label=VARIANT_LABELS[variant],
        )
        axes[1].plot(
            cell["return_ratio"] * 100,
            cell["any_deduction"],
            marker="o",
            linewidth=2.5,
            color=COLORS[variant],
            label=VARIANT_LABELS[variant],
        )
    axes[0].set_ylabel("Mean deduction points")
    axes[0].set_ylim(-0.1, 2.1)
    axes[0].set_title("Deduction intensity", fontweight="bold")
    axes[1].set_ylabel("Probability of any deduction")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Deduction frequency", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Visible return (% of amount received)")
        ax.set_xticks([0, 10, 25, 50, 75])
        ax.grid(True, alpha=0.3)
    axes[1].legend(title="Prompt")
    fig.suptitle("Controlled deduction-stage calibration (GPT-5 Nano)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "deduction_by_controlled_return.png", dpi=300)
    plt.close(fig)


def plot_decision_distribution(dataframe, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharey=True)
    colors = ["#d9e8f0", "#e9c46a", "#c14953"]
    for ax, variant in zip(axes, VARIANT_ORDER):
        cell = dataframe[dataframe["variant"] == variant]
        bottom = np.zeros(5)
        ratios = sorted(cell["return_ratio"].unique())
        for spent, color in zip((0, 1, 2), colors):
            proportions = np.array(
                [
                    (cell[cell["return_ratio"] == ratio]["deduction_spent"] == spent).mean()
                    for ratio in ratios
                ]
            )
            ax.bar(
                np.arange(5),
                proportions,
                bottom=bottom,
                color=color,
                label=f"Spend {spent}",
            )
            bottom += proportions
        ax.set_xticks(np.arange(5), ["0%", "10%", "25%", "50%", "75%"])
        ax.set_xlabel("Visible return")
        ax.set_title(VARIANT_LABELS[variant], fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Decision proportion")
    fig.suptitle("Distribution of controlled deduction choices", fontweight="bold")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Decision",
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    fig.savefig(output_dir / "deduction_decision_distribution.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    configure_matplotlib()
    import pandas as pd
    import seaborn as sns

    sns.set_style("whitegrid")
    args.out.mkdir(parents=True, exist_ok=True)
    metadata, rows = load_trials(args.input)
    dataframe = pd.DataFrame(rows).sort_values("call_order")
    summary_dataframe = pd.DataFrame(cell_summary(dataframe))
    diagnostic_rows, eligible = diagnostics(dataframe, summary_dataframe)
    diagnostic_dataframe = pd.DataFrame(diagnostic_rows)
    interaction = interaction_ols(dataframe)
    interaction_dataframe = pd.DataFrame(
        [{"contrast": "Cost-salient × return-ratio slope", **interaction}]
    )
    wording_effect = independent_difference(
        dataframe[dataframe["variant"] == "cost_salient"]["deduction_spent"],
        dataframe[dataframe["variant"] == "current"]["deduction_spent"],
    )
    wording_effect_dataframe = pd.DataFrame(
        [
            {
                "contrast": "Cost-salient − current mean deduction",
                **wording_effect,
            }
        ]
    )
    provenance_dataframe = pd.DataFrame(
        [
            {
                key: metadata.get(key)
                for key in (
                    "code_commit",
                    "code_dirty",
                    "config_sha256",
                    "llm_provider",
                    "provider_model",
                    "temperature",
                )
            }
        ]
    )

    dataframe.to_csv(args.out / "trials.csv", index=False)
    summary_dataframe.to_csv(args.out / "cell_summary.csv", index=False)
    diagnostic_dataframe.to_csv(args.out / "selectivity_diagnostics.csv", index=False)
    interaction_dataframe.to_csv(args.out / "wording_interaction.csv", index=False)
    wording_effect_dataframe.to_csv(args.out / "wording_main_effect.csv", index=False)
    provenance_dataframe.to_csv(args.out / "provenance.csv", index=False)
    (args.out / "decision.txt").write_text(
        (
            "ELIGIBLE: cost-salient wording may proceed to a population pilot.\n"
            if eligible
            else "NOT ELIGIBLE: do not proceed to a population pilot with either wording.\n"
        ),
        encoding="utf-8",
    )
    plot_calibration(summary_dataframe, args.out)
    plot_decision_distribution(dataframe, args.out)

    print("Cell summaries:")
    print(summary_dataframe.to_string(index=False))
    print("\nSelectivity diagnostics:")
    print(diagnostic_dataframe.to_string(index=False))
    print("\nWording × return interaction:")
    print(interaction_dataframe.to_string(index=False))
    print("\nDescriptive wording main effect:")
    print(wording_effect_dataframe.to_string(index=False))
    print(f"\nCost-salient eligible for population pilot: {eligible}")
    print("\nToken totals:")
    print(
        dataframe[["input_tokens", "output_tokens", "reasoning_tokens"]]
        .sum()
        .to_string()
    )
    print(f"Saved outputs to {args.out}")


if __name__ == "__main__":
    main()
