#!/usr/bin/env python3
"""Analyze the exploratory GPT-5 Nano punishment-availability factorial."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyses._shared import configure_matplotlib
from analyze_corrected_v2_confirmatory import load_runs
from analyze_defector_myth_game_crossmodel_n5 import myth_metrics


DEFAULT_OFF_INPUT = Path(
    "data/json/noise_experiments/"
    "defector_punishment_factorial_off_provenance_gpt_n5_20260821"
)
DEFAULT_ON_INPUT = Path(
    "data/json/noise_experiments/defector_punishment_gpt_n5_20260821"
)
DEFAULT_EXCLUDED_OFF_INPUT = Path(
    "data/json/noise_experiments/"
    "defector_punishment_factorial_off_gpt_n5_20260821"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/defector_punishment_factorial_gpt_n5_20260821"
)
EXPERIMENTS = {
    "off": "noise8i_defector_punishment_factorial_off_gpt_n5",
    "on": "noise8i_defector_punishment_gpt_n5",
}
MODEL = "openai/gpt-5-nano"
EXPECTED_IDS = set(range(61, 66))
AVAILABILITY_ORDER = ["off", "on"]
DEFECTOR_ORDER = ["control", "defectors25"]
AVAILABILITY_LABELS = {"off": "Unavailable", "on": "Available"}
DEFECTOR_LABELS = {"control": "0% defectors", "defectors25": "25% defectors"}
COLORS = {"off": "#457b9d", "on": "#c14953"}
METRIC_LABELS = {
    "standard_send_ratio": "Ordinary-agent proportion sent",
    "standard_return_ratio": "Ordinary receiver return ratio",
    "standard_myth_coop_density": "Ordinary myth cooperation density",
    "standard_myth_threat_density": "Ordinary myth threat density",
    "standard_myth_half_rule": "Ordinary myth explicit half-rule rate",
}


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


def mean_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else math.nan


def usage_metrics(run):
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "attempts": 0,
        "recovered_retries": 0,
        "forced_responses": 0,
        "notifications": 0,
    }
    for agent in (run.get("agents") or {}).values():
        for event in agent.get("interaction_history") or []:
            response = event.get("response") or {}
            source = response.get("response_source", "llm")
            if event.get("error"):
                totals["recovered_retries"] += 1
            if source == "forced_zero":
                totals["forced_responses"] += 1
            elif source == "deduction_notification":
                totals["notifications"] += 1
            elif response:
                totals["attempts"] += 1
            call_usage = response.get("usage") or {}
            for key in ("input_tokens", "output_tokens", "reasoning_tokens"):
                totals[key] += int(call_usage.get(key) or 0)
    return totals


def load_data(off_input, on_input):
    run_rows = []
    round_rows = []
    myth_rows = []
    usage_rows = []
    seen = {
        (availability, condition): set()
        for availability in AVAILABILITY_ORDER
        for condition in DEFECTOR_ORDER
    }

    for availability, input_dir in (("off", off_input), ("on", on_input)):
        runs = load_runs(input_dir / EXPERIMENTS[availability])
        if len(runs) != 10:
            raise RuntimeError(
                f"{availability} has {len(runs)} final runs; expected 10"
            )
        for path, run in runs:
            metadata = run.get("run_metadata") or {}
            if metadata.get("model") != MODEL:
                raise RuntimeError(f"Unexpected model in {path}")
            if metadata.get("execution_provenance_version") != 1:
                raise RuntimeError(f"Missing execution provenance in {path}")
            if not metadata.get("code_commit") or metadata.get("code_dirty"):
                raise RuntimeError(f"Dirty or incomplete provenance in {path}")
            if bool(metadata.get("punishment_enabled")) != (availability == "on"):
                raise RuntimeError(f"Wrong punishment availability in {path}")
            expected_capacity = 9 if availability == "on" else 6
            if int(metadata.get("memory_capacity")) != expected_capacity:
                raise RuntimeError(f"Wrong memory capacity in {path}")

            replicate_id = int(metadata["replicate_id"])
            defector_ids = set(metadata.get("defector_agent_ids") or [])
            condition = "defectors25" if defector_ids else "control"
            seen[(availability, condition)].add(replicate_id)
            standard_ids = set(run.get("agents") or {}) - defector_ids

            sends = []
            returns = []
            run_myths = []
            for entry in run.get("conversation_history") or []:
                round_number = int(entry["round"])
                round_sends = []
                round_returns = []
                round_myths = []
                for dyad in entry.get("dyads") or []:
                    investor = dyad["investor"]
                    trustee = dyad["trustee"]
                    if investor in standard_ids:
                        ratio = float(dyad["sent"]) / 5.0
                        sends.append(ratio)
                        round_sends.append(ratio)
                    if trustee in standard_ids:
                        received = float(dyad.get("received") or 0)
                        if received > 0:
                            ratio = float(dyad.get("returned") or 0) / received
                            returns.append(ratio)
                            round_returns.append(ratio)

                for agent_id, text in (entry.get("myths") or {}).items():
                    if agent_id not in standard_ids:
                        continue
                    metrics = myth_metrics(text)
                    row = {
                        "availability": availability,
                        "availability_label": AVAILABILITY_LABELS[availability],
                        "condition": condition,
                        "condition_label": DEFECTOR_LABELS[condition],
                        "replicate_id": replicate_id,
                        "round": round_number,
                        "agent_id": agent_id,
                        **metrics,
                    }
                    myth_rows.append(row)
                    run_myths.append(row)
                    round_myths.append(row)

                round_rows.append(
                    {
                        "availability": availability,
                        "availability_label": AVAILABILITY_LABELS[availability],
                        "condition": condition,
                        "condition_label": DEFECTOR_LABELS[condition],
                        "replicate_id": replicate_id,
                        "round": round_number,
                        "standard_send_ratio": mean_or_nan(round_sends),
                        "standard_return_ratio": mean_or_nan(round_returns),
                        "standard_myth_coop_density": mean_or_nan(
                            [item["coop_density"] for item in round_myths]
                        ),
                    }
                )

            run_rows.append(
                {
                    "availability": availability,
                    "availability_label": AVAILABILITY_LABELS[availability],
                    "condition": condition,
                    "condition_label": DEFECTOR_LABELS[condition],
                    "replicate_id": replicate_id,
                    "defector_ids": ",".join(sorted(defector_ids)),
                    "code_commit": metadata["code_commit"],
                    "standard_send_ratio": mean_or_nan(sends),
                    "standard_return_ratio": mean_or_nan(returns),
                    "standard_myth_coop_density": mean_or_nan(
                        [item["coop_density"] for item in run_myths]
                    ),
                    "standard_myth_threat_density": mean_or_nan(
                        [item["threat_density"] for item in run_myths]
                    ),
                    "standard_myth_half_rule": mean_or_nan(
                        [item["half_rule"] for item in run_myths]
                    ),
                }
            )
            usage_rows.append(
                {
                    "availability": availability,
                    "condition": condition,
                    "replicate_id": replicate_id,
                    **usage_metrics(run),
                }
            )

    for cell, replicate_ids in seen.items():
        if replicate_ids != EXPECTED_IDS:
            raise RuntimeError(f"Cell {cell} has replicate IDs {replicate_ids}")

    return run_rows, round_rows, myth_rows, usage_rows


def load_excluded_off_behavior(input_dir):
    """Load the pre-provenance off batch only for a labeled sensitivity check."""
    rows = []
    seen = {condition: set() for condition in DEFECTOR_ORDER}
    runs = load_runs(input_dir / EXPERIMENTS["off"])
    if len(runs) != 10:
        raise RuntimeError(f"Excluded off batch has {len(runs)} runs; expected 10")
    for path, run in runs:
        metadata = run.get("run_metadata") or {}
        if metadata.get("execution_provenance_version"):
            raise RuntimeError(f"Excluded batch unexpectedly has provenance: {path}")
        if metadata.get("model") != MODEL or metadata.get("punishment_enabled"):
            raise RuntimeError(f"Wrong excluded sensitivity cell in {path}")
        replicate_id = int(metadata["replicate_id"])
        defector_ids = set(metadata.get("defector_agent_ids") or [])
        condition = "defectors25" if defector_ids else "control"
        seen[condition].add(replicate_id)
        standard_ids = set(run.get("agents") or {}) - defector_ids
        sends = []
        returns = []
        for entry in run.get("conversation_history") or []:
            for dyad in entry.get("dyads") or []:
                if dyad["investor"] in standard_ids:
                    sends.append(float(dyad["sent"]) / 5.0)
                received = float(dyad.get("received") or 0)
                if dyad["trustee"] in standard_ids and received > 0:
                    returns.append(float(dyad.get("returned") or 0) / received)
        rows.append(
            {
                "availability": "off",
                "condition": condition,
                "replicate_id": replicate_id,
                "standard_send_ratio": mean_or_nan(sends),
                "standard_return_ratio": mean_or_nan(returns),
            }
        )
    for condition, replicate_ids in seen.items():
        if replicate_ids != EXPECTED_IDS:
            raise RuntimeError(
                f"Excluded sensitivity cell {condition} has IDs {replicate_ids}"
            )
    return rows


def paired_result(values, metric, contrast, contrast_type):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    low, high = confidence_interval(values)
    exact_zero = len(values) and np.allclose(values, 0)
    sd = values.std(ddof=1) if len(values) > 1 else math.nan
    p_value = (
        1.0
        if exact_zero
        else stats.ttest_1samp(values, 0).pvalue
        if len(values) > 1 and not np.isclose(sd, 0)
        else math.nan
    )
    return {
        "metric": metric,
        "metric_label": METRIC_LABELS[metric],
        "contrast": contrast,
        "contrast_type": contrast_type,
        "n_pairs": len(values),
        "estimate": float(values.mean()) if len(values) else math.nan,
        "ci_low": low,
        "ci_high": high,
        "p_value": p_value,
        "cohens_dz": 0.0 if exact_zero else values.mean() / sd if sd else math.nan,
    }


def analyze_contrasts(dataframe):
    records = []
    replicate_records = []
    for metric in METRIC_LABELS:
        pivot = dataframe.pivot(
            index="replicate_id",
            columns=["availability", "condition"],
            values=metric,
        )
        differences = {
            "availability_control": pivot[("on", "control")]
            - pivot[("off", "control")],
            "availability_defectors25": pivot[("on", "defectors25")]
            - pivot[("off", "defectors25")],
            "defectors_off": pivot[("off", "defectors25")]
            - pivot[("off", "control")],
            "defectors_on": pivot[("on", "defectors25")]
            - pivot[("on", "control")],
        }
        differences["interaction"] = (
            differences["availability_defectors25"]
            - differences["availability_control"]
        )
        labels = {
            "availability_control": "Available − unavailable | 0% defectors",
            "availability_defectors25": "Available − unavailable | 25% defectors",
            "defectors_off": "25% − 0% defectors | unavailable",
            "defectors_on": "25% − 0% defectors | available",
            "interaction": "Availability × defector difference-in-differences",
        }
        for contrast_type, values in differences.items():
            records.append(
                paired_result(
                    values.to_numpy(dtype=float),
                    metric,
                    labels[contrast_type],
                    contrast_type,
                )
            )
            for replicate_id, value in values.items():
                replicate_records.append(
                    {
                        "metric": metric,
                        "contrast_type": contrast_type,
                        "replicate_id": replicate_id,
                        "difference": value,
                    }
                )
    return records, replicate_records


def sensitivity_contrasts(excluded_dataframe, accepted_dataframe):
    combined = accepted_dataframe[accepted_dataframe["availability"] == "on"]
    combined = combined[
        [
            "availability",
            "condition",
            "replicate_id",
            "standard_send_ratio",
            "standard_return_ratio",
        ]
    ]
    import pandas as pd

    combined = pd.concat([excluded_dataframe, combined], ignore_index=True)
    records = []
    for metric in ("standard_send_ratio", "standard_return_ratio"):
        pivot = combined.pivot(
            index="replicate_id",
            columns=["availability", "condition"],
            values=metric,
        )
        differences = {
            "availability_control": pivot[("on", "control")]
            - pivot[("off", "control")],
            "availability_defectors25": pivot[("on", "defectors25")]
            - pivot[("off", "defectors25")],
        }
        differences["interaction"] = (
            differences["availability_defectors25"]
            - differences["availability_control"]
        )
        labels = {
            "availability_control": "Available − excluded unavailable | 0% defectors",
            "availability_defectors25": "Available − excluded unavailable | 25% defectors",
            "interaction": "Excluded-batch availability × defector interaction",
        }
        for contrast_type, values in differences.items():
            record = paired_result(
                values.to_numpy(dtype=float),
                metric,
                labels[contrast_type],
                contrast_type,
            )
            record["analysis_status"] = (
                "post_hoc sensitivity; unavailable batch lacks embedded provenance"
            )
            records.append(record)
    return records


def summarize(dataframe):
    rows = []
    for availability in AVAILABILITY_ORDER:
        for condition in DEFECTOR_ORDER:
            cell = dataframe[
                (dataframe["availability"] == availability)
                & (dataframe["condition"] == condition)
            ]
            for metric in METRIC_LABELS:
                values = cell[metric].to_numpy(dtype=float)
                low, high = confidence_interval(values)
                rows.append(
                    {
                        "availability": availability,
                        "availability_label": AVAILABILITY_LABELS[availability],
                        "condition": condition,
                        "condition_label": DEFECTOR_LABELS[condition],
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "n": len(values),
                        "mean": values.mean(),
                        "sd": values.std(ddof=1),
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return rows


def plot_factorial_behavior(dataframe, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    for ax, metric in zip(
        axes,
        ("standard_send_ratio", "standard_return_ratio"),
    ):
        x = np.arange(2)
        offsets = {"off": -0.08, "on": 0.08}
        for availability in AVAILABILITY_ORDER:
            means = []
            errors = []
            for condition in DEFECTOR_ORDER:
                values = dataframe[
                    (dataframe["availability"] == availability)
                    & (dataframe["condition"] == condition)
                ][metric].to_numpy(dtype=float)
                mean = values.mean()
                low, high = confidence_interval(values)
                means.append(mean)
                errors.append((mean - low, high - mean))
                jitter = np.linspace(-0.025, 0.025, len(values))
                ax.scatter(
                    np.full(len(values), x[DEFECTOR_ORDER.index(condition)] + offsets[availability])
                    + jitter,
                    values,
                    color=COLORS[availability],
                    alpha=0.45,
                    s=28,
                    zorder=2,
                )
            ax.errorbar(
                x + offsets[availability],
                means,
                yerr=np.asarray(errors).T,
                marker="o",
                linewidth=2.5,
                capsize=5,
                color=COLORS[availability],
                label=AVAILABILITY_LABELS[availability],
                zorder=3,
            )
        ax.set_xticks(x, [DEFECTOR_LABELS[item] for item in DEFECTOR_ORDER])
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_title("Sending", fontweight="bold")
    axes[1].set_title("Returning", fontweight="bold")
    axes[1].legend(title="Deduction stage")
    fig.suptitle(
        "Punishment availability × hidden defectors (GPT-5 Nano)",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "factorial_behavior.png", dpi=300)
    plt.close(fig)


def plot_availability_effects(replicate_dataframe, contrast_dataframe, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharex=True)
    for ax, metric in zip(
        axes,
        ("standard_send_ratio", "standard_return_ratio"),
    ):
        for x, (condition, contrast_type) in enumerate(
            zip(
                DEFECTOR_ORDER,
                ("availability_control", "availability_defectors25"),
            )
        ):
            values = replicate_dataframe[
                (replicate_dataframe["metric"] == metric)
                & (replicate_dataframe["contrast_type"] == contrast_type)
            ]["difference"].to_numpy(dtype=float)
            ax.scatter(
                np.full(len(values), x) + np.linspace(-0.04, 0.04, len(values)),
                values,
                color="#607d8b",
                alpha=0.65,
                s=38,
            )
            result = contrast_dataframe[
                (contrast_dataframe["metric"] == metric)
                & (contrast_dataframe["contrast_type"] == contrast_type)
            ].iloc[0]
            ax.errorbar(
                x,
                result["estimate"],
                yerr=[
                    [result["estimate"] - result["ci_low"]],
                    [result["ci_high"] - result["estimate"]],
                ],
                color=COLORS["on"],
                marker="o",
                capsize=6,
                linewidth=2.5,
                zorder=3,
            )
        ax.axhline(0, color="#263238", linestyle="--", linewidth=1)
        ax.set_xticks(range(2), [DEFECTOR_LABELS[item] for item in DEFECTOR_ORDER])
        ax.set_ylabel(f"Available − unavailable\n({METRIC_LABELS[metric].lower()})")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_title("Effect on sending", fontweight="bold")
    axes[1].set_title("Effect on returning", fontweight="bold")
    fig.suptitle("Matched punishment-availability effects", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "availability_effects.png", dpi=300)
    plt.close(fig)


def plot_trajectories(round_dataframe, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey="row")
    for column, condition in enumerate(DEFECTOR_ORDER):
        cell = round_dataframe[round_dataframe["condition"] == condition]
        for row, metric in enumerate(
            ("standard_send_ratio", "standard_return_ratio")
        ):
            ax = axes[row, column]
            for availability in AVAILABILITY_ORDER:
                values = cell[cell["availability"] == availability]
                means = values.groupby("round", as_index=False)[metric].mean()
                ax.plot(
                    means["round"],
                    means[metric],
                    marker="o",
                    linewidth=2.2,
                    color=COLORS[availability],
                    label=AVAILABILITY_LABELS[availability],
                )
            ax.set_ylim(0, 1)
            ax.set_xticks(range(1, 11))
            ax.grid(True, alpha=0.3, linestyle="--")
            if row == 0:
                ax.set_title(DEFECTOR_LABELS[condition], fontweight="bold")
            if column == 0:
                ax.set_ylabel(
                    "Ordinary proportion sent"
                    if row == 0
                    else "Ordinary return ratio"
                )
            if row == 1:
                ax.set_xlabel("Round")
    axes[0, 1].legend(title="Deduction stage")
    fig.suptitle("Behavior across rounds", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "behavior_trajectories.png", dpi=300)
    plt.close(fig)


def plot_myth_language(dataframe, output_dir):
    import matplotlib.pyplot as plt

    metrics = [
        "standard_myth_coop_density",
        "standard_myth_threat_density",
        "standard_myth_half_rule",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, metric in zip(axes, metrics):
        x = np.arange(2)
        for availability in AVAILABILITY_ORDER:
            means = []
            for condition in DEFECTOR_ORDER:
                means.append(
                    dataframe[
                        (dataframe["availability"] == availability)
                        & (dataframe["condition"] == condition)
                    ][metric].mean()
                )
            ax.plot(
                x,
                means,
                marker="o",
                linewidth=2.4,
                color=COLORS[availability],
                label=AVAILABILITY_LABELS[availability],
            )
        ax.set_xticks(x, ["0%", "25%"])
        ax.set_xlabel("Hidden defectors")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(True, axis="y", alpha=0.3)
    axes[-1].legend(title="Deduction stage")
    fig.suptitle("Ordinary-authored myth language", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "myth_language_factorial.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off-input", type=Path, default=DEFAULT_OFF_INPUT)
    parser.add_argument("--on-input", type=Path, default=DEFAULT_ON_INPUT)
    parser.add_argument(
        "--excluded-off-input",
        type=Path,
        default=DEFAULT_EXCLUDED_OFF_INPUT,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    configure_matplotlib()
    import pandas as pd
    import seaborn as sns

    sns.set_style("whitegrid")
    args.out.mkdir(parents=True, exist_ok=True)
    run_rows, round_rows, myth_rows, usage_rows = load_data(
        args.off_input,
        args.on_input,
    )
    dataframe = pd.DataFrame(run_rows).sort_values(
        ["availability", "condition", "replicate_id"]
    )
    round_dataframe = pd.DataFrame(round_rows)
    myth_dataframe = pd.DataFrame(myth_rows)
    usage_dataframe = pd.DataFrame(usage_rows)
    summary_dataframe = pd.DataFrame(summarize(dataframe))
    contrasts, replicate_contrasts = analyze_contrasts(dataframe)
    contrast_dataframe = pd.DataFrame(contrasts)
    replicate_dataframe = pd.DataFrame(replicate_contrasts)
    excluded_dataframe = pd.DataFrame(
        load_excluded_off_behavior(args.excluded_off_input)
    )
    sensitivity_dataframe = pd.DataFrame(
        sensitivity_contrasts(excluded_dataframe, dataframe)
    )

    dataframe.to_csv(args.out / "run_metrics.csv", index=False)
    round_dataframe.to_csv(args.out / "round_metrics.csv", index=False)
    myth_dataframe.to_csv(args.out / "myth_metrics.csv", index=False)
    usage_dataframe.to_csv(args.out / "token_usage.csv", index=False)
    summary_dataframe.to_csv(args.out / "summary.csv", index=False)
    contrast_dataframe.to_csv(args.out / "paired_contrasts.csv", index=False)
    replicate_dataframe.to_csv(args.out / "replicate_contrasts.csv", index=False)
    excluded_dataframe.to_csv(
        args.out / "excluded_off_run_metrics.csv",
        index=False,
    )
    sensitivity_dataframe.to_csv(
        args.out / "excluded_batch_sensitivity.csv",
        index=False,
    )

    plot_factorial_behavior(dataframe, args.out)
    plot_availability_effects(
        replicate_dataframe,
        contrast_dataframe,
        args.out,
    )
    plot_trajectories(round_dataframe, args.out)
    plot_myth_language(dataframe, args.out)

    print("Behavioral cell means:")
    print(
        summary_dataframe[
            summary_dataframe["metric"].isin(
                ["standard_send_ratio", "standard_return_ratio"]
            )
        ].to_string(index=False)
    )
    print("\nFrozen behavioral contrasts:")
    print(
        contrast_dataframe[
            contrast_dataframe["metric"].isin(
                ["standard_send_ratio", "standard_return_ratio"]
            )
        ].to_string(index=False)
    )
    print("\nToken/attempt totals by availability:")
    print(
        usage_dataframe.groupby("availability").sum(numeric_only=True).to_string()
    )
    print("\nPost-hoc sensitivity using the excluded first off batch:")
    print(sensitivity_dataframe.to_string(index=False))
    print(f"\nSaved outputs to {args.out}")


if __name__ == "__main__":
    main()
