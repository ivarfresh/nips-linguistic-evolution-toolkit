#!/usr/bin/env python3
"""Analyze the frozen GPT-5 Nano hidden-defector deduction-point screen."""

from __future__ import annotations

import argparse
from collections import Counter
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyses._shared import configure_matplotlib
from analyze_corrected_v2_confirmatory import load_runs
from analyze_defector_myth_game_crossmodel_n5 import myth_metrics


DEFAULT_INPUT = Path(
    "data/json/noise_experiments/defector_punishment_gpt_n5_20260821"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/defector_punishment_gpt_n5_20260821"
)
EXPERIMENT_DIR = "noise8i_defector_punishment_gpt_n5"
EXPECTED_IDS = set(range(61, 66))
MODEL = "openai/gpt-5-nano"
CONDITION_ORDER = ["control", "defectors25"]
CONDITION_LABELS = {
    "control": "0% defectors",
    "defectors25": "25% defectors",
}
COLORS = {
    "control": "#457b9d",
    "defectors25": "#c14953",
}
RETURN_PLOT_TITLE = "Deduction was not reserved for low returns"
TRAJECTORY_PLOT_TITLE = "Behavior across the ten-round punishment screen"
PUNISH_PATTERN = re.compile(
    r"\b(?:punish\w*|sanction\w*|retaliat\w*|revenge\w*|"
    r"deduct\w*|penalt\w*|consequence\w*)\b",
    re.IGNORECASE,
)


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


def paired_result(values, metric, contrast, label):
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
        "contrast": contrast,
        "label": label,
        "n_pairs": len(values),
        "estimate": values.mean() if len(values) else math.nan,
        "ci_low": low,
        "ci_high": high,
        "p_value": p_value,
        "cohens_dz": 0.0 if exact_zero else values.mean() / sd if sd else math.nan,
    }


def mean_or_nan(values):
    values = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else math.nan


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


def load_data(input_dir):
    runs = load_runs(input_dir / EXPERIMENT_DIR)
    expected_run_count = len(CONDITION_ORDER) * len(EXPECTED_IDS)
    if len(runs) != expected_run_count:
        raise RuntimeError(
            f"Found {len(runs)} runs; expected {expected_run_count}"
        )

    run_rows = []
    decision_rows = []
    myth_rows = []
    next_rows = []
    round_rows = []
    usage_rows = []
    seen = {condition: set() for condition in CONDITION_ORDER}

    for path, run in runs:
        metadata = run.get("run_metadata") or {}
        if metadata.get("model") != MODEL:
            raise RuntimeError(f"Unexpected model in {path}: {metadata.get('model')}")
        if metadata.get("code_dirty"):
            raise RuntimeError(f"Dirty execution provenance in {path}")
        if not metadata.get("punishment_enabled"):
            raise RuntimeError(f"Deduction stage absent in {path}")
        if int(metadata.get("punishment_budget")) != 2:
            raise RuntimeError(f"Unexpected deduction budget in {path}")
        if float(metadata.get("punishment_effect_multiplier")) != 3:
            raise RuntimeError(f"Unexpected deduction multiplier in {path}")

        replicate_id = int(metadata["replicate_id"])
        condition = (
            "defectors25"
            if float(metadata.get("defector_ratio_actual") or 0) > 0
            else "control"
        )
        seen[condition].add(replicate_id)
        defector_ids = set(metadata.get("defector_agent_ids") or [])
        standard_ids = set(run.get("agents") or {}) - defector_ids
        history = run.get("conversation_history") or []

        run_decisions = []
        run_myths = []
        for entry in history:
            round_number = int(entry["round"])
            round_decisions = []
            for dyad in entry.get("dyads") or []:
                investor_id = dyad["investor"]
                trustee_id = dyad["trustee"]
                if investor_id not in standard_ids:
                    continue
                received_visible = float(dyad.get("received_communicated") or 0)
                returned_visible = float(dyad.get("returned_communicated") or 0)
                visible_return_ratio = (
                    returned_visible / received_visible
                    if received_visible > 0
                    else math.nan
                )
                received_actual = float(dyad.get("received") or 0)
                returned_actual = float(dyad.get("returned") or 0)
                actual_return_ratio = (
                    returned_actual / received_actual
                    if received_actual > 0
                    else math.nan
                )
                spent = int(dyad["deduction_spent"])
                row = {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "replicate_id": replicate_id,
                    "round": round_number,
                    "investor_id": investor_id,
                    "trustee_id": trustee_id,
                    "target_type": (
                        "defector" if trustee_id in defector_ids else "standard"
                    ),
                    "send_ratio": float(dyad["sent"]) / 5.0,
                    "actual_return_ratio": actual_return_ratio,
                    "visible_return_ratio": visible_return_ratio,
                    "deduction_spent": spent,
                    "any_deduction": float(spent > 0),
                    "max_deduction": float(spent == 2),
                    "adequate_half_return": float(
                        np.isfinite(visible_return_ratio)
                        and visible_return_ratio >= 0.5
                    ),
                    "deduction_after_half_return": float(
                        spent > 0
                        and np.isfinite(visible_return_ratio)
                        and visible_return_ratio >= 0.5
                    ),
                    "deduction_actual_loss": float(
                        dyad["deduction_actual_loss"]
                    ),
                }
                decision_rows.append(row)
                run_decisions.append(row)
                round_decisions.append(row)

                # Observational next-round response of the sanctioned receiver.
                next_entry = next(
                    (
                        later
                        for later in history
                        if int(later["round"]) == round_number + 1
                    ),
                    None,
                )
                if next_entry is not None and trustee_id in standard_ids:
                    next_dyad = next(
                        (
                            candidate
                            for candidate in next_entry.get("dyads") or []
                            if trustee_id in (candidate.get("agents") or [])
                        ),
                        None,
                    )
                    if next_dyad is not None:
                        next_role = next_dyad["roles"][trustee_id]
                        next_send = math.nan
                        next_return = math.nan
                        if next_role == "investor":
                            next_send = float(next_dyad["sent"]) / 5.0
                        else:
                            received = float(next_dyad.get("received") or 0)
                            if received > 0:
                                next_return = float(next_dyad["returned"]) / received
                        next_rows.append(
                            {
                                "condition": condition,
                                "replicate_id": replicate_id,
                                "source_round": round_number,
                                "agent_id": trustee_id,
                                "previous_deduction_spent": spent,
                                "previous_any_deduction": float(spent > 0),
                                "next_role": next_role,
                                "next_send_ratio": next_send,
                                "next_return_ratio": next_return,
                            }
                        )

            round_rows.append(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "replicate_id": replicate_id,
                    "round": round_number,
                    "standard_send_ratio": mean_or_nan(
                        [row["send_ratio"] for row in round_decisions]
                    ),
                    "standard_deduction_spent": mean_or_nan(
                        [row["deduction_spent"] for row in round_decisions]
                    ),
                    "standard_any_deduction": mean_or_nan(
                        [row["any_deduction"] for row in round_decisions]
                    ),
                }
            )

            for agent_id, text in (entry.get("myths") or {}).items():
                metrics = myth_metrics(text)
                words = re.findall(r"\b[\w'-]+\b", text.lower())
                punishment_matches = len(PUNISH_PATTERN.findall(text))
                myth_row = {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "replicate_id": replicate_id,
                    "round": round_number,
                    "agent_id": agent_id,
                    "author_type": (
                        "defector" if agent_id in defector_ids else "standard"
                    ),
                    "text": text,
                    **metrics,
                    "punishment_density": (
                        100 * punishment_matches / len(words) if words else 0.0
                    ),
                    "punishment_presence": float(punishment_matches > 0),
                }
                myth_rows.append(myth_row)
                run_myths.append(myth_row)

        standard_returns = []
        for entry in history:
            for dyad in entry.get("dyads") or []:
                if dyad["trustee"] not in standard_ids:
                    continue
                received = float(dyad.get("received") or 0)
                if received > 0:
                    standard_returns.append(float(dyad["returned"]) / received)

        defector_targets = [
            row for row in run_decisions if row["target_type"] == "defector"
        ]
        standard_targets = [
            row for row in run_decisions if row["target_type"] == "standard"
        ]
        standard_myths = [row for row in run_myths if row["author_type"] == "standard"]
        finite_rr = [
            row for row in run_decisions if np.isfinite(row["visible_return_ratio"])
        ]
        slope = math.nan
        if len(finite_rr) > 1 and len(
            {row["visible_return_ratio"] for row in finite_rr}
        ) > 1:
            slope = stats.linregress(
                [row["visible_return_ratio"] for row in finite_rr],
                [row["deduction_spent"] for row in finite_rr],
            ).slope

        run_rows.append(
            {
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "replicate_id": replicate_id,
                "defector_ids": ",".join(sorted(defector_ids)),
                "n_standard_sender_decisions": len(run_decisions),
                "n_defector_targets": len(defector_targets),
                "standard_send_ratio": mean_or_nan(
                    [row["send_ratio"] for row in run_decisions]
                ),
                "standard_return_ratio": mean_or_nan(standard_returns),
                "standard_deduction_spent": mean_or_nan(
                    [row["deduction_spent"] for row in run_decisions]
                ),
                "standard_any_deduction": mean_or_nan(
                    [row["any_deduction"] for row in run_decisions]
                ),
                "deduct_to_defector_spent": mean_or_nan(
                    [row["deduction_spent"] for row in defector_targets]
                ),
                "deduct_to_defector_any": mean_or_nan(
                    [row["any_deduction"] for row in defector_targets]
                ),
                "deduct_to_standard_spent": mean_or_nan(
                    [row["deduction_spent"] for row in standard_targets]
                ),
                "deduct_to_standard_any": mean_or_nan(
                    [row["any_deduction"] for row in standard_targets]
                ),
                "deduction_after_half_return_rate": mean_or_nan(
                    [
                        row["any_deduction"]
                        for row in run_decisions
                        if row["adequate_half_return"]
                    ]
                ),
                "deduction_return_slope": slope,
                "standard_myth_coop_density": mean_or_nan(
                    [row["coop_density"] for row in standard_myths]
                ),
                "standard_myth_threat_density": mean_or_nan(
                    [row["threat_density"] for row in standard_myths]
                ),
                "standard_myth_punishment_density": mean_or_nan(
                    [row["punishment_density"] for row in standard_myths]
                ),
                "standard_myth_punishment_presence": mean_or_nan(
                    [row["punishment_presence"] for row in standard_myths]
                ),
            }
        )
        usage_rows.append(
            {
                "condition": condition,
                "replicate_id": replicate_id,
                **usage_metrics(run),
            }
        )

    for condition in CONDITION_ORDER:
        if seen[condition] != EXPECTED_IDS:
            raise RuntimeError(
                f"{condition}: replicate IDs {seen[condition]}; expected {EXPECTED_IDS}"
            )
    return run_rows, decision_rows, myth_rows, next_rows, round_rows, usage_rows


def make_contrasts(runs):
    rows = []
    by_condition = runs.set_index(["condition", "replicate_id"])
    for metric, label in (
        ("standard_send_ratio", "Ordinary-agent proportion sent"),
        ("standard_return_ratio", "Ordinary receiver return ratio"),
        ("standard_deduction_spent", "Ordinary sender deduction points"),
        ("standard_any_deduction", "Probability of any deduction"),
        ("deduct_to_standard_spent", "Deduction points toward ordinary receivers"),
        ("deduct_to_standard_any", "Any deduction toward ordinary receivers"),
        ("standard_myth_coop_density", "Ordinary myth cooperation density"),
        ("standard_myth_threat_density", "Ordinary myth threat density"),
        ("standard_myth_punishment_density", "Ordinary myth punishment density"),
        ("standard_myth_punishment_presence", "Ordinary myth punishment presence"),
    ):
        differences = [
            by_condition.loc[("defectors25", replicate_id), metric]
            - by_condition.loc[("control", replicate_id), metric]
            for replicate_id in sorted(EXPECTED_IDS)
        ]
        rows.append(
            paired_result(
                differences,
                metric,
                "25% defectors − 0% defectors",
                label,
            )
        )

    treatment = runs[runs["condition"] == "defectors25"].set_index("replicate_id")
    for suffix, label in (
        ("spent", "Deduction points: defector − ordinary target"),
        ("any", "Any deduction: defector − ordinary target"),
    ):
        differences = [
            treatment.loc[replicate_id, f"deduct_to_defector_{suffix}"]
            - treatment.loc[replicate_id, f"deduct_to_standard_{suffix}"]
            for replicate_id in sorted(EXPECTED_IDS)
        ]
        rows.append(
            paired_result(
                differences,
                f"target_contrast_{suffix}",
                "defector receiver − ordinary receiver | 25% defectors",
                label,
            )
        )
    return rows


def make_summaries(runs):
    rows = []
    metrics = [
        "standard_send_ratio",
        "standard_return_ratio",
        "standard_deduction_spent",
        "standard_any_deduction",
        "deduct_to_defector_spent",
        "deduct_to_defector_any",
        "deduct_to_standard_spent",
        "deduct_to_standard_any",
        "deduction_after_half_return_rate",
        "deduction_return_slope",
        "standard_myth_coop_density",
        "standard_myth_threat_density",
        "standard_myth_punishment_density",
        "standard_myth_punishment_presence",
    ]
    for condition in CONDITION_ORDER:
        subset = runs[runs["condition"] == condition]
        for metric in metrics:
            values = subset[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            low, high = confidence_interval(values)
            rows.append(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "metric": metric,
                    "n": len(values),
                    "mean": values.mean() if len(values) else math.nan,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return rows


def make_return_bins(decisions):
    import pandas as pd

    subset = decisions[np.isfinite(decisions["visible_return_ratio"])].copy()
    subset["return_bin"] = pd.cut(
        subset["visible_return_ratio"],
        bins=[-np.inf, 0.001, 0.25, 0.5, np.inf],
        labels=["≈0", "(0,.25)", "[.25,.50)", "≥.50"],
        right=False,
    )
    return (
        subset.groupby(["condition", "return_bin"], observed=False)
        .agg(
            n=("deduction_spent", "size"),
            mean_deduction=("deduction_spent", "mean"),
            any_deduction=("any_deduction", "mean"),
        )
        .reset_index()
    )


def make_next_action_summary(next_actions):
    rows = []
    for condition in CONDITION_ORDER:
        subset = next_actions[next_actions["condition"] == condition]
        for previous_any in (0.0, 1.0):
            group = subset[subset["previous_any_deduction"] == previous_any]
            for metric in ("next_send_ratio", "next_return_ratio"):
                values = group[metric].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                rows.append(
                    {
                        "condition": condition,
                        "previous_any_deduction": previous_any,
                        "metric": metric,
                        "n": len(values),
                        "mean": values.mean() if len(values) else math.nan,
                    }
                )
    return rows


def make_term_counts(myths):
    rows = []
    for (condition, author_type), group in myths.groupby(
        ["condition", "author_type"]
    ):
        counts = Counter()
        for text in group["text"]:
            counts.update(match.group(0).lower() for match in PUNISH_PATTERN.finditer(text))
        for term, count in counts.most_common():
            rows.append(
                {
                    "condition": condition,
                    "author_type": author_type,
                    "term": term,
                    "count": count,
                }
            )
    return rows


def plot_targeting(runs, output_dir):
    import matplotlib.pyplot as plt

    treatment = runs[runs["condition"] == "defectors25"].sort_values("replicate_id")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7))
    specs = [
        ("spent", "Mean deduction points"),
        ("any", "Probability of any deduction"),
    ]
    for ax, (suffix, ylabel) in zip(axes, specs):
        standard = treatment[f"deduct_to_standard_{suffix}"].to_numpy()
        defector = treatment[f"deduct_to_defector_{suffix}"].to_numpy()
        for left, right in zip(standard, defector):
            ax.plot([0, 1], [left, right], marker="o", alpha=0.72, color="#6c757d")
        ax.scatter([0, 1], [standard.mean(), defector.mean()], s=125, color="#c14953", zorder=4)
        ax.set_xticks([0, 1], ["Ordinary receiver", "Defector receiver"])
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Do ordinary senders target hidden defectors?", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "deduction_targeting.png", dpi=300)
    plt.close(fig)


def plot_return_bins(bins, output_dir):
    import matplotlib.pyplot as plt

    labels = ["≈0", "(0,.25)", "[.25,.50)", "≥.50"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    for index, condition in enumerate(CONDITION_ORDER):
        subset = bins[bins["condition"] == condition].set_index("return_bin")
        raw_values = [subset.loc[label, "any_deduction"] for label in labels]
        ns = [int(subset.loc[label, "n"]) for label in labels]
        values = [value if np.isfinite(value) else 0.0 for value in raw_values]
        positions = x + (index - 0.5) * width
        bars = ax.bar(
            positions,
            values,
            width,
            label=CONDITION_LABELS[condition],
            color=COLORS[condition],
        )
        for bar, n in zip(bars, ns):
            if n == 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x, labels)
    ax.set_xlabel("Visible return / visible amount received")
    ax.set_ylabel("Probability sender spent any deduction point")
    ax.set_ylim(0, 1.15)
    ax.set_title(RETURN_PLOT_TITLE, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "deduction_by_visible_return.png", dpi=300)
    plt.close(fig)


def plot_trajectories(rounds, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for ax, metric, ylabel in (
        (axes[0], "standard_send_ratio", "Ordinary-agent proportion sent"),
        (axes[1], "standard_deduction_spent", "Mean deduction points"),
    ):
        lower_bound, upper_bound = (
            (0.0, 1.0)
            if metric == "standard_send_ratio"
            else (0.0, 2.0)
        )
        for condition in CONDITION_ORDER:
            subset = rounds[rounds["condition"] == condition]
            grouped = subset.groupby("round")[metric].agg(["mean", "sem", "count"])
            x = grouped.index.to_numpy(dtype=float)
            mean = grouped["mean"].to_numpy(dtype=float)
            critical = grouped["count"].map(
                lambda count: stats.t.ppf(0.975, count - 1) if count > 1 else 0
            )
            error = (
                grouped["sem"].fillna(0).to_numpy(dtype=float)
                * critical.to_numpy(dtype=float)
            )
            ax.plot(
                x,
                mean,
                marker="o",
                linewidth=2.2,
                color=COLORS[condition],
                label=CONDITION_LABELS[condition],
            )
            ax.fill_between(
                x,
                np.clip(mean - error, lower_bound, upper_bound),
                np.clip(mean + error, lower_bound, upper_bound),
                color=COLORS[condition],
                alpha=0.14,
            )
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_ylim(lower_bound - 0.05, upper_bound + 0.05)
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.25)
    axes[0].legend()
    fig.suptitle(TRAJECTORY_PLOT_TITLE, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "sending_and_deduction_trajectories.png", dpi=300)
    plt.close(fig)


def plot_myth_language(runs, output_dir):
    import matplotlib.pyplot as plt

    metrics = [
        ("standard_myth_coop_density", "Cooperation/fairness\nmatches per 100 words"),
        ("standard_myth_punishment_density", "Punishment/deduction\nmatches per 100 words"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
    x = np.arange(2)
    for ax, (metric, ylabel) in zip(axes, metrics):
        for index, condition in enumerate(CONDITION_ORDER):
            values = runs[runs["condition"] == condition].sort_values("replicate_id")[metric].to_numpy()
            ax.scatter(
                np.full(len(values), index) + np.linspace(-0.06, 0.06, len(values)),
                values,
                color=COLORS[condition],
                alpha=0.8,
                s=52,
            )
            low, high = confidence_interval(values)
            ax.errorbar(
                index,
                values.mean(),
                yerr=[[values.mean() - low], [high - values.mean()]],
                fmt="o",
                color="#263238",
                capsize=6,
                markersize=8,
                linewidth=2,
            )
        ax.set_xticks(x, [CONDITION_LABELS[c] for c in CONDITION_ORDER])
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Language in ordinary-authored myths", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "ordinary_myth_language.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    import pandas as pd

    run_rows, decision_rows, myth_rows, next_rows, round_rows, usage_rows = load_data(args.input)
    runs = pd.DataFrame(run_rows).sort_values(["condition", "replicate_id"])
    decisions = pd.DataFrame(decision_rows).sort_values(["condition", "replicate_id", "round", "investor_id"])
    myths = pd.DataFrame(myth_rows).sort_values(["condition", "replicate_id", "round", "agent_id"])
    next_actions = pd.DataFrame(next_rows).sort_values(["condition", "replicate_id", "source_round", "agent_id"])
    rounds = pd.DataFrame(round_rows).sort_values(["condition", "replicate_id", "round"])
    token_usage = pd.DataFrame(usage_rows).sort_values(["condition", "replicate_id"])
    contrasts = pd.DataFrame(make_contrasts(runs))
    summaries = pd.DataFrame(make_summaries(runs))
    return_bins = make_return_bins(decisions)
    next_summary = pd.DataFrame(make_next_action_summary(next_actions))
    term_counts = pd.DataFrame(make_term_counts(myths))

    runs.to_csv(args.out / "run_metrics.csv", index=False)
    decisions.to_csv(args.out / "deduction_decisions.csv", index=False)
    myths.to_csv(args.out / "myth_metrics.csv", index=False)
    next_actions.to_csv(args.out / "next_action_observations.csv", index=False)
    rounds.to_csv(args.out / "round_metrics.csv", index=False)
    token_usage.to_csv(args.out / "token_usage.csv", index=False)
    contrasts.to_csv(args.out / "paired_contrasts.csv", index=False)
    summaries.to_csv(args.out / "summary.csv", index=False)
    return_bins.to_csv(args.out / "deduction_by_return_bin.csv", index=False)
    next_summary.to_csv(args.out / "next_action_summary.csv", index=False)
    term_counts.to_csv(args.out / "punishment_term_counts.csv", index=False)

    plot_targeting(runs, args.out)
    plot_return_bins(return_bins, args.out)
    plot_trajectories(rounds, args.out)
    plot_myth_language(runs, args.out)

    print("Run metrics:")
    print(runs.to_string(index=False))
    print("\nContrasts:")
    print(contrasts.to_string(index=False))
    print("\nReturn bins:")
    print(return_bins.to_string(index=False))
    print("\nNext-action diagnostic:")
    print(next_summary.to_string(index=False))
    print("\nToken totals:")
    print(token_usage.groupby("condition")[["input_tokens", "output_tokens", "reasoning_tokens", "attempts", "recovered_retries", "forced_responses"]].sum().to_string())
    print(f"\nSaved outputs to {args.out}")


if __name__ == "__main__":
    main()
