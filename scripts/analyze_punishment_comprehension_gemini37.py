#!/usr/bin/env python3
"""Analyze the frozen Gemini 3.7 Flash deduction calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyses._shared import configure_matplotlib
from scripts.analyze_punishment_comprehension_calibration import (
    confidence_interval,
    independent_difference,
)


DEFAULT_INPUT = Path(
    "data/json/noise_experiments/"
    "punishment_comprehension_gemini37_20260823/results.json"
)
DEFAULT_LITE = Path(
    "data/json/noise_experiments/"
    "punishment_comprehension_gemini_20260821/results.json"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/punishment_comprehension_gemini37_20260823"
)
RETURN_RATIOS = [0.0, 0.1, 0.25, 0.5, 0.75]


def load_trials(path, expected_provider_model, label):
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata") or {}
    if metadata.get("execution_provenance_version") != 1:
        raise RuntimeError(f"{label}: missing execution provenance")
    if not metadata.get("code_commit") or metadata.get("code_dirty"):
        raise RuntimeError(f"{label}: dirty or incomplete provenance")
    if metadata.get("llm_provider") != "google":
        raise RuntimeError(f"{label}: expected direct Google provider")
    if metadata.get("provider_model") != expected_provider_model:
        raise RuntimeError(f"{label}: unexpected provider model")
    if expected_provider_model == "gemini-3.7-flash":
        expected = {
            "thinking_level": "medium",
            "thinking_level_source": "GEMINI_THINKING_LEVEL",
            "temperature_sent": False,
            "request_timeout_seconds": 300.0,
            "request_timeout_source": "GEMINI_REQUEST_TIMEOUT_SECONDS",
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(
                    f"{label}: expected {key}={value!r}; got {metadata.get(key)!r}"
                )

    rows = []
    messages_by_cell = {}
    usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    for trial in payload.get("trials") or []:
        if trial.get("variant") != "current":
            continue
        ratio = float(trial["return_ratio"])
        spent = trial.get("deduction_spent")
        if not trial.get("success") or spent not in (0, 1, 2):
            raise RuntimeError(f"{label}: invalid trial {trial.get('trial_id')}")
        messages_by_cell.setdefault(ratio, set()).add(
            json.dumps(trial.get("messages"), sort_keys=True)
        )
        attempts = trial.get("attempts") or []
        for attempt in attempts:
            record = ((attempt.get("response") or {}).get("usage") or {})
            for key in usage:
                usage[key] += int(record.get(key) or 0)
        rows.append(
            {
                "model": label,
                "return_ratio": ratio,
                "replicate": int(trial["replicate"]),
                "deduction_spent": int(spent),
                "any_deduction": float(spent > 0),
                "attempts": len(attempts),
            }
        )
    if len(rows) != 50:
        raise RuntimeError(f"{label}: expected 50 current trials; found {len(rows)}")
    for ratio in RETURN_RATIOS:
        if sum(row["return_ratio"] == ratio for row in rows) != 10:
            raise RuntimeError(f"{label}: wrong count at return ratio {ratio}")
        if len(messages_by_cell.get(ratio, set())) != 1:
            raise RuntimeError(f"{label}: messages differ within {ratio} cell")
    return metadata, rows, usage


def summarize(dataframe):
    records = []
    for model in dataframe["model"].drop_duplicates():
        for ratio in RETURN_RATIOS:
            cell = dataframe[
                (dataframe["model"] == model)
                & (dataframe["return_ratio"] == ratio)
            ]
            values = cell["deduction_spent"].to_numpy(dtype=float)
            low, high = confidence_interval(values)
            records.append(
                {
                    "model": model,
                    "return_ratio": ratio,
                    "n": len(values),
                    "mean_deduction": values.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "any_deduction": cell["any_deduction"].mean(),
                }
            )
    return records


def selectivity(dataframe, summary_dataframe):
    cell = dataframe[dataframe["model"] == "Gemini 3.7 Flash"]
    low_values = cell[cell["return_ratio"].isin([0.0, 0.1])][
        "deduction_spent"
    ].to_numpy(dtype=float)
    high_values = cell[cell["return_ratio"].isin([0.5, 0.75])][
        "deduction_spent"
    ].to_numpy(dtype=float)
    difference = independent_difference(low_values, high_values)
    high_any = cell[cell["return_ratio"].isin([0.5, 0.75])][
        "any_deduction"
    ].mean()
    means = summary_dataframe[
        summary_dataframe["model"] == "Gemini 3.7 Flash"
    ].sort_values("return_ratio")["mean_deduction"].to_numpy(dtype=float)
    positive_reversals = np.diff(means)[np.diff(means) > 0]
    monotonic = len(positive_reversals) <= 1 and (
        not len(positive_reversals) or positive_reversals.max() <= .2
    )
    slope = stats.linregress(cell["return_ratio"], cell["deduction_spent"])
    spearman = stats.spearmanr(cell["return_ratio"], cell["deduction_spent"])
    return {
        "linear_slope": slope.slope,
        "slope_p_value": slope.pvalue,
        "spearman_rho": spearman.statistic,
        "spearman_p_value": spearman.pvalue,
        "low_minus_high": difference["estimate"],
        "low_minus_high_ci_low": difference["ci_low"],
        "low_minus_high_ci_high": difference["ci_high"],
        "low_minus_high_p_value": difference["p_value"],
        "high_return_any_deduction": high_any,
        "positive_adjacent_reversals": len(positive_reversals),
        "largest_positive_reversal": (
            positive_reversals.max() if len(positive_reversals) else 0.0
        ),
        "low_minus_high_gate": difference["estimate"] >= .5,
        "high_return_restraint_gate": high_any <= .25,
        "monotonic_gate": monotonic,
        "passes_selectivity_gate": (
            difference["estimate"] >= .5 and high_any <= .25 and monotonic
        ),
    }


def plot_summary(summary_dataframe, output_dir):
    import matplotlib.pyplot as plt

    colors = {"Gemini 3.1 Flash-Lite": "#90a4ae", "Gemini 3.7 Flash": "#c14953"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for model in ["Gemini 3.1 Flash-Lite", "Gemini 3.7 Flash"]:
        cell = summary_dataframe[summary_dataframe["model"] == model]
        axes[0].plot(
            cell["return_ratio"] * 100, cell["mean_deduction"], marker="o",
            linewidth=2.5, color=colors[model], label=model
        )
        axes[1].plot(
            cell["return_ratio"] * 100, cell["any_deduction"], marker="o",
            linewidth=2.5, color=colors[model], label=model
        )
    axes[0].set_title("Deduction intensity", fontweight="bold")
    axes[0].set_ylabel("Mean points spent")
    axes[0].set_ylim(-.1, 2.1)
    axes[1].set_title("Deduction frequency", fontweight="bold")
    axes[1].set_ylabel("Probability of any deduction")
    axes[1].set_ylim(-.05, 1.05)
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("Visible return (% of amount received)")
        axis.set_xticks([0, 10, 25, 50, 75])
        axis.grid(alpha=.3)
    fig.suptitle("Controlled punishment policy across Gemini Flash models", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "gemini_flash_deduction_calibration.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--lite", type=Path, default=DEFAULT_LITE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    configure_matplotlib()
    import pandas as pd

    args.out.mkdir(parents=True, exist_ok=True)
    metadata, rows, usage = load_trials(
        args.input, "gemini-3.7-flash", "Gemini 3.7 Flash"
    )
    _, lite_rows, _ = load_trials(
        args.lite, "gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite"
    )
    dataframe = pd.DataFrame(rows + lite_rows)
    summary = pd.DataFrame(summarize(dataframe))
    decision = pd.DataFrame([selectivity(dataframe, summary)])
    retries = sum(max(0, row["attempts"] - 1) for row in rows)
    cost_value = (
        usage["input_tokens"] * .75
        + (usage["output_tokens"] + usage["reasoning_tokens"]) * 3.75
    ) / 1_000_000
    cost = pd.DataFrame(
        [{**usage, "retries": retries, "estimated_list_price_usd": cost_value}]
    )

    pd.DataFrame(rows).to_csv(args.out / "trials.csv", index=False)
    summary.to_csv(args.out / "cell_summary.csv", index=False)
    decision.to_csv(args.out / "selectivity_decision.csv", index=False)
    cost.to_csv(args.out / "token_cost.csv", index=False)
    (args.out / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    plot_summary(summary, args.out)

    print(summary.to_string(index=False))
    print("\nFrozen decision:\n", decision.to_string(index=False))
    print(f"\nRetries: {retries}; estimated list-price cost: ${cost_value:.4f}")


if __name__ == "__main__":
    main()
