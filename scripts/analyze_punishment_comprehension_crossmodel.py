#!/usr/bin/env python3
"""Compare current-wording deduction calibration in GPT-5 Nano and Gemini."""

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
from analyze_punishment_comprehension_calibration import (
    confidence_interval,
    independent_difference,
)


DEFAULT_GPT = Path(
    "data/json/noise_experiments/"
    "punishment_comprehension_gpt_20260821/results.json"
)
DEFAULT_GEMINI = Path(
    "data/json/noise_experiments/"
    "punishment_comprehension_gemini_20260821/results.json"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/punishment_comprehension_crossmodel_20260821"
)
MODEL_SPECS = {
    "gpt": {
        "label": "GPT-5 Nano",
        "provider": "openai",
        "provider_model": "gpt-5-nano",
        "color": "#457b9d",
    },
    "gemini": {
        "label": "Gemini 3.1 Flash-Lite",
        "provider": "google",
        "provider_model": "gemini-3.1-flash-lite",
        "color": "#c14953",
    },
}
MODEL_ORDER = ["gpt", "gemini"]
RETURN_RATIOS = [0.0, 0.1, 0.25, 0.5, 0.75]


def load_model(path, model_id):
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata") or {}
    spec = MODEL_SPECS[model_id]
    if metadata.get("execution_provenance_version") != 1:
        raise RuntimeError(f"{model_id}: missing execution provenance")
    if not metadata.get("code_commit") or metadata.get("code_dirty"):
        raise RuntimeError(f"{model_id}: dirty or incomplete provenance")
    if metadata.get("llm_provider") != spec["provider"]:
        raise RuntimeError(f"{model_id}: wrong provider")
    if metadata.get("provider_model") != spec["provider_model"]:
        raise RuntimeError(f"{model_id}: wrong provider model")

    rows = []
    messages_by_cell = {}
    for trial in payload.get("trials") or []:
        if trial.get("variant") != "current":
            continue
        ratio = float(trial["return_ratio"])
        spent = trial.get("deduction_spent")
        if not trial.get("success") or spent not in (0, 1, 2):
            raise RuntimeError(f"{model_id}: invalid trial {trial.get('trial_id')}")
        messages_by_cell.setdefault(ratio, set()).add(
            json.dumps(trial.get("messages"), sort_keys=True)
        )
        rows.append(
            {
                "model_id": model_id,
                "model_label": spec["label"],
                "return_ratio": ratio,
                "replicate": int(trial["replicate"]),
                "deduction_spent": int(spent),
                "any_deduction": float(spent > 0),
                "max_deduction": float(spent == 2),
                "attempts": len(trial.get("attempts") or []),
            }
        )
    if len(rows) != 50:
        raise RuntimeError(f"{model_id}: found {len(rows)} current trials; expected 50")
    for ratio in RETURN_RATIOS:
        if sum(row["return_ratio"] == ratio for row in rows) != 10:
            raise RuntimeError(f"{model_id}: wrong count at return {ratio}")
        if len(messages_by_cell.get(ratio, set())) != 1:
            raise RuntimeError(f"{model_id}: nonidentical messages at return {ratio}")
    return metadata, rows


def summarize(dataframe):
    rows = []
    for model_id in MODEL_ORDER:
        for ratio in RETURN_RATIOS:
            cell = dataframe[
                (dataframe["model_id"] == model_id)
                & (dataframe["return_ratio"] == ratio)
            ]
            values = cell["deduction_spent"].to_numpy(dtype=float)
            low, high = confidence_interval(values)
            rows.append(
                {
                    "model_id": model_id,
                    "model_label": MODEL_SPECS[model_id]["label"],
                    "return_ratio": ratio,
                    "n": len(values),
                    "mean_deduction": values.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "any_deduction": cell["any_deduction"].mean(),
                    "max_deduction": cell["max_deduction"].mean(),
                }
            )
    return rows


def selectivity(dataframe, summary_dataframe):
    rows = []
    for model_id in MODEL_ORDER:
        cell = dataframe[dataframe["model_id"] == model_id]
        slope = stats.linregress(cell["return_ratio"], cell["deduction_spent"])
        spearman = stats.spearmanr(cell["return_ratio"], cell["deduction_spent"])
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
            summary_dataframe["model_id"] == model_id
        ].sort_values("return_ratio")["mean_deduction"].to_numpy(dtype=float)
        positive_reversals = np.diff(means)[np.diff(means) > 0]
        monotonic_gate = len(positive_reversals) <= 1 and (
            not len(positive_reversals) or positive_reversals.max() <= 0.2
        )
        passes = (
            difference["estimate"] >= 0.5
            and high_any <= 0.25
            and monotonic_gate
        )
        rows.append(
            {
                "model_id": model_id,
                "model_label": MODEL_SPECS[model_id]["label"],
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
                "passes_selectivity_gate": passes,
            }
        )
    return rows


def model_interaction(dataframe):
    ratio = dataframe["return_ratio"].to_numpy(dtype=float)
    gemini = (dataframe["model_id"] == "gemini").to_numpy(dtype=float)
    y = dataframe["deduction_spent"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(y)), ratio, gemini, ratio * gemini])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ beta
    df = len(y) - design.shape[1]
    covariance = (residuals @ residuals / df) * np.linalg.inv(design.T @ design)
    se = math.sqrt(covariance[3, 3])
    critical = stats.t.ppf(0.975, df)
    t_value = beta[3] / se
    return {
        "contrast": "Gemini × return-ratio slope",
        "estimate": beta[3],
        "ci_low": beta[3] - critical * se,
        "ci_high": beta[3] + critical * se,
        "p_value": 2 * stats.t.sf(abs(t_value), df),
    }


def plot_crossmodel(summary_dataframe, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    for model_id in MODEL_ORDER:
        cell = summary_dataframe[summary_dataframe["model_id"] == model_id]
        axes[0].plot(
            cell["return_ratio"] * 100,
            cell["mean_deduction"],
            marker="o",
            linewidth=2.5,
            color=MODEL_SPECS[model_id]["color"],
            label=MODEL_SPECS[model_id]["label"],
        )
        axes[1].plot(
            cell["return_ratio"] * 100,
            cell["any_deduction"],
            marker="o",
            linewidth=2.5,
            color=MODEL_SPECS[model_id]["color"],
            label=MODEL_SPECS[model_id]["label"],
        )
    axes[0].set_title("Deduction intensity", fontweight="bold")
    axes[0].set_ylabel("Mean deduction points")
    axes[0].set_ylim(-0.1, 2.1)
    axes[1].set_title("Deduction frequency", fontweight="bold")
    axes[1].set_ylabel("Probability of any deduction")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(title="Model")
    for ax in axes:
        ax.set_xlabel("Visible return (% of amount received)")
        ax.set_xticks([0, 10, 25, 50, 75])
        ax.grid(True, alpha=0.3)
    fig.suptitle("Current deduction wording behaves differently across models", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "crossmodel_deduction_calibration.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt", type=Path, default=DEFAULT_GPT)
    parser.add_argument("--gemini", type=Path, default=DEFAULT_GEMINI)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    configure_matplotlib()
    import pandas as pd
    import seaborn as sns

    sns.set_style("whitegrid")
    args.out.mkdir(parents=True, exist_ok=True)
    provenance_rows = []
    rows = []
    for model_id, path in (("gpt", args.gpt), ("gemini", args.gemini)):
        metadata, model_rows = load_model(path, model_id)
        rows.extend(model_rows)
        provenance_rows.append(
            {
                "model_id": model_id,
                **{
                    key: metadata.get(key)
                    for key in (
                        "code_commit",
                        "code_dirty",
                        "config_sha256",
                        "llm_provider",
                        "provider_model",
                    )
                },
            }
        )
    dataframe = pd.DataFrame(rows)
    summary_dataframe = pd.DataFrame(summarize(dataframe))
    selectivity_dataframe = pd.DataFrame(selectivity(dataframe, summary_dataframe))
    interaction_dataframe = pd.DataFrame([model_interaction(dataframe)])

    dataframe.to_csv(args.out / "trials.csv", index=False)
    summary_dataframe.to_csv(args.out / "cell_summary.csv", index=False)
    selectivity_dataframe.to_csv(args.out / "selectivity_diagnostics.csv", index=False)
    interaction_dataframe.to_csv(args.out / "model_interaction.csv", index=False)
    pd.DataFrame(provenance_rows).to_csv(args.out / "provenance.csv", index=False)
    plot_crossmodel(summary_dataframe, args.out)

    print("Cell summaries:")
    print(summary_dataframe.to_string(index=False))
    print("\nSelectivity diagnostics:")
    print(selectivity_dataframe.to_string(index=False))
    print("\nCross-model interaction:")
    print(interaction_dataframe.to_string(index=False))
    print(f"\nSaved outputs to {args.out}")


if __name__ == "__main__":
    main()
