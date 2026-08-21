#!/usr/bin/env python3
"""Analyze Gemini deduction choices inside the live true-zero noise band."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyses._shared import configure_matplotlib
from analyze_punishment_comprehension_calibration import confidence_interval


DEFAULT_INPUT = Path(
    "data/json/noise_experiments/"
    "punishment_comprehension_gemini_noise_boundary_20260821/results.json"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/punishment_comprehension_gemini_noise_boundary_20260821"
)
EXPECTED_AMOUNTS = [0.0, 0.25, 0.50, 0.75, 1.0]


def load_data(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata") or {}
    if metadata.get("execution_provenance_version") != 1:
        raise RuntimeError("Missing execution provenance")
    if not metadata.get("code_commit") or metadata.get("code_dirty"):
        raise RuntimeError("Dirty or incomplete execution provenance")
    if metadata.get("llm_provider") != "google":
        raise RuntimeError("Expected direct Google provider")
    if metadata.get("provider_model") != "gemini-3.1-flash-lite":
        raise RuntimeError("Unexpected model")
    rows = []
    messages = {}
    for trial in payload.get("trials") or []:
        amount = round(float(trial["state"]["returned_communicated"]), 2)
        spent = trial.get("deduction_spent")
        if not trial.get("success") or spent not in (0, 1, 2):
            raise RuntimeError(f"Invalid trial {trial.get('trial_id')}")
        messages.setdefault(amount, set()).add(
            json.dumps(trial.get("messages"), sort_keys=True)
        )
        rows.append(
            {
                "trial_id": trial["trial_id"],
                "call_order": int(trial["call_order"]),
                "visible_return": amount,
                "return_ratio": float(trial["return_ratio"]),
                "replicate": int(trial["replicate"]),
                "deduction_spent": int(spent),
                "any_deduction": float(spent > 0),
                "max_deduction": float(spent == 2),
                "attempts": len(trial.get("attempts") or []),
            }
        )
    if len(rows) != 50:
        raise RuntimeError(f"Found {len(rows)} trials; expected 50")
    for amount in EXPECTED_AMOUNTS:
        if sum(row["visible_return"] == amount for row in rows) != 10:
            raise RuntimeError(f"Wrong count at ${amount:.2f}")
        if len(messages.get(amount, set())) != 1:
            raise RuntimeError(f"Nonidentical messages at ${amount:.2f}")
    return metadata, rows


def summarize(dataframe):
    rows = []
    for amount in EXPECTED_AMOUNTS:
        cell = dataframe[dataframe["visible_return"] == amount]
        values = cell["deduction_spent"].to_numpy(dtype=float)
        low, high = confidence_interval(values)
        rows.append(
            {
                "visible_return": amount,
                "return_ratio": cell["return_ratio"].iloc[0],
                "n": len(values),
                "mean_deduction": values.mean(),
                "ci_low": low,
                "ci_high": high,
                "any_deduction": cell["any_deduction"].mean(),
                "max_deduction": cell["max_deduction"].mean(),
            }
        )
    return rows


def boundary_metrics(summary_dataframe):
    amounts = summary_dataframe["visible_return"].to_numpy(dtype=float)
    probabilities = summary_dataframe["any_deduction"].to_numpy(dtype=float)
    mean_points = summary_dataframe["mean_deduction"].to_numpy(dtype=float)
    majority = probabilities > 0.5
    switch_low = None
    switch_high = None
    for index in range(len(amounts) - 1):
        if majority[index] and probabilities[index + 1] < 0.5:
            switch_low = amounts[index]
            switch_high = amounts[index + 1]
            break
    uniform_positive_probability = np.trapz(probabilities, amounts)
    uniform_positive_points = np.trapz(mean_points, amounts)
    implied_probability = 0.5 * probabilities[0] + 0.5 * uniform_positive_probability
    implied_points = 0.5 * mean_points[0] + 0.5 * uniform_positive_points
    return {
        "switch_low": switch_low,
        "switch_high": switch_high,
        "uniform_positive_any_probability": uniform_positive_probability,
        "uniform_positive_mean_points": uniform_positive_points,
        "implied_true_zero_punishment_probability": implied_probability,
        "implied_true_zero_mean_points": implied_points,
        "passes_population_pilot_gate": bool(implied_probability >= 0.5),
    }


def plot_boundary(summary_dataframe, metrics, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    axes[0].plot(
        summary_dataframe["visible_return"],
        summary_dataframe["mean_deduction"],
        marker="o",
        linewidth=2.7,
        color="#c14953",
    )
    axes[0].set_ylabel("Mean deduction points")
    axes[0].set_ylim(-0.1, 2.1)
    axes[0].set_title("Deduction intensity", fontweight="bold")
    axes[1].plot(
        summary_dataframe["visible_return"],
        summary_dataframe["any_deduction"],
        marker="o",
        linewidth=2.7,
        color="#c14953",
    )
    axes[1].fill_between(
        summary_dataframe["visible_return"],
        0,
        summary_dataframe["any_deduction"],
        color="#c14953",
        alpha=0.15,
        label="Positive-noise integral",
    )
    axes[1].axhline(0.5, color="#263238", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Probability of any deduction")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Deduction frequency", fontweight="bold")
    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.axvspan(
            metrics["switch_low"],
            metrics["switch_high"],
            color="#e9c46a",
            alpha=0.22,
        )
        ax.set_xlabel("Visible return from a true-zero defector ($)")
        ax.set_xticks(EXPECTED_AMOUNTS)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Gemini punishment threshold inside the ±$1 noise band", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "gemini_near_zero_boundary.png", dpi=300)
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
    metadata, rows = load_data(args.input)
    dataframe = pd.DataFrame(rows).sort_values("call_order")
    summary_dataframe = pd.DataFrame(summarize(dataframe))
    metrics = boundary_metrics(summary_dataframe)
    provenance = {
        key: metadata.get(key)
        for key in (
            "code_commit",
            "code_dirty",
            "config_sha256",
            "llm_provider",
            "provider_model",
        )
    }

    dataframe.to_csv(args.out / "trials.csv", index=False)
    summary_dataframe.to_csv(args.out / "cell_summary.csv", index=False)
    pd.DataFrame([metrics]).to_csv(args.out / "boundary_metrics.csv", index=False)
    pd.DataFrame([provenance]).to_csv(args.out / "provenance.csv", index=False)
    plot_boundary(summary_dataframe, metrics, args.out)

    print("Cell summaries:")
    print(summary_dataframe.to_string(index=False))
    print("\nBoundary metrics:")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"\nSaved outputs to {args.out}")


if __name__ == "__main__":
    main()
