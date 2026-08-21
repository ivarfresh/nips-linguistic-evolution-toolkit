#!/usr/bin/env python3
"""Analyze the frozen GPT/Gemini Myth→Game defector stress test."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyses._shared import configure_matplotlib
from analyze_corrected_v2_confirmatory import load_runs


DEFAULT_INPUT = Path(
    "data/json/noise_experiments/defector_myth_game_crossmodel_20260821"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/defector_myth_game_crossmodel_n5_20260821"
)
MODEL_SPECS = {
    "gpt": {
        "label": "GPT-5 Nano",
        "slug": "openai/gpt-5-nano",
        "directory": "noise8i_defector_myth_game_gpt_n5",
        "color": "#457b9d",
    },
    "gemini": {
        "label": "Gemini 3.1 Flash-Lite",
        "slug": "google/gemini-3.1-flash-lite",
        "directory": "noise8i_defector_myth_game_gemini_n5",
        "color": "#e76f51",
    },
}
CONDITION_ORDER = ["control", "defectors25"]
CONDITION_LABELS = {
    "control": "No defectors",
    "defectors25": "2 of 8 defectors",
}
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
COOP_PATTERN = re.compile(
    r"\b(?:cooperat\w*|trust\w*|reciproc\w*|shar\w*|generos\w*|fair\w*|"
    r"return\w*|mutual\w*|together\w*|gift\w*|help\w*)\b",
    re.IGNORECASE,
)
THREAT_PATTERN = re.compile(
    r"\b(?:defect\w*|betray\w*|exploit\w*|withhold\w*|punish\w*|"
    r"retaliat\w*|revenge\w*|threat\w*|distrust\w*|selfish\w*|zero\w*)\b",
    re.IGNORECASE,
)
HALF_PATTERN = re.compile(
    r"(?:\b(?:return\w*|giv\w*|shar\w*|split\w*)\b[^.!?\n]{0,80}"
    r"\b(?:half|equal\w*|fifty\s*percent|50\s*%)\b|"
    r"\b(?:half|equal\w*|fifty\s*percent|50\s*%)\b[^.!?\n]{0,80}"
    r"\b(?:return\w*|giv\w*|shar\w*|split\w*)\b)",
    re.IGNORECASE,
)


def ci(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if not len(values):
        return math.nan, math.nan
    if len(values) == 1 or np.allclose(values, values[0]):
        return values.mean(), values.mean()
    return stats.t.interval(
        0.95,
        len(values) - 1,
        loc=values.mean(),
        scale=stats.sem(values),
    )


def paired_record(values, model_id, metric, contrast, primary=False):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    low, high = ci(values)
    sd = values.std(ddof=1) if len(values) > 1 else math.nan
    exact_zero = len(values) and np.allclose(values, 0)
    p_value = (
        1.0
        if exact_zero
        else stats.ttest_1samp(values, 0).pvalue
        if len(values) > 1 and not np.isclose(sd, 0)
        else math.nan
    )
    dz = 0.0 if exact_zero else values.mean() / sd if sd else math.nan
    return {
        "model_id": model_id,
        "model_label": MODEL_SPECS[model_id]["label"],
        "metric": metric,
        "contrast": contrast,
        "primary": primary,
        "n_pairs": len(values),
        "estimate": values.mean() if len(values) else math.nan,
        "ci_low": low,
        "ci_high": high,
        "p_value": p_value,
        "cohens_dz": dz,
    }


def myth_metrics(text):
    words = WORD_PATTERN.findall(text or "")
    denominator = len(words)
    coop_count = len(COOP_PATTERN.findall(text or ""))
    threat_count = len(THREAT_PATTERN.findall(text or ""))
    return {
        "word_count": denominator,
        "coop_density": coop_count / denominator * 100 if denominator else math.nan,
        "coop_present": float(coop_count > 0),
        "threat_density": threat_count / denominator * 100 if denominator else math.nan,
        "threat_present": float(threat_count > 0),
        "half_rule": float(bool(HALF_PATTERN.search(text or ""))),
    }


def usage(run):
    totals = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    attempts = 0
    retries = 0
    forced = 0
    for agent in (run.get("agents") or {}).values():
        for event in agent.get("interaction_history") or []:
            response = event.get("response") or {}
            call_usage = response.get("usage") or {}
            if response:
                attempts += 1
            if event.get("error"):
                retries += 1
            if (event.get("metadata") or {}).get("response_source") == "forced_zero":
                forced += 1
            for key in totals:
                totals[key] += int(call_usage.get(key) or 0)
    return totals, attempts, retries, forced


def load_data(input_dir):
    run_rows = []
    behavior_round_rows = []
    myth_rows = []
    usage_rows = []
    expected_ids = set(range(30, 35))

    for model_id, spec in MODEL_SPECS.items():
        runs = load_runs(input_dir / spec["directory"])
        if len(runs) != 10:
            raise RuntimeError(f"{model_id} has {len(runs)} runs, expected 10")
        seen = {condition: set() for condition in CONDITION_ORDER}
        for path, run in runs:
            metadata = run.get("run_metadata") or {}
            game_data = run.get("game_data") or {}
            if metadata.get("model") != spec["slug"]:
                raise RuntimeError(f"Unexpected model in {path}: {metadata.get('model')}")
            if metadata.get("code_dirty"):
                raise RuntimeError(f"Dirty execution provenance in {path}")
            replicate_id = int(metadata["replicate_id"])
            defector_ids = set(
                metadata.get("defector_agent_ids")
                or game_data.get("defector_agent_ids")
                or []
            )
            condition = "defectors25" if defector_ids else "control"
            seen[condition].add(replicate_id)
            agent_types = game_data.get("agent_types") or {}
            standard_ids = {
                agent_id
                for agent_id in (run.get("agents") or {})
                if agent_id not in defector_ids
            }
            sent_all = []
            sent_to_standard = []
            sent_to_defector = []
            return_ratios = []
            return_from_standard_sender = []
            zero_return_opportunities_from_defectors = 0
            by_round = []

            history = run.get("conversation_history") or []
            for entry in history:
                round_sent = []
                for dyad in entry.get("dyads") or []:
                    investor = dyad.get("investor")
                    trustee = dyad.get("trustee")
                    sent = float(dyad.get("sent") or 0)
                    received = float(dyad.get("received") or 0)
                    returned = float(dyad.get("returned") or 0)
                    if investor in standard_ids:
                        ratio = sent / 5.0
                        sent_all.append(ratio)
                        round_sent.append(ratio)
                        if trustee in defector_ids:
                            sent_to_defector.append(ratio)
                        else:
                            sent_to_standard.append(ratio)
                    if trustee in standard_ids:
                        if received > 0:
                            ratio = returned / received
                            return_ratios.append(ratio)
                            if investor not in defector_ids:
                                return_from_standard_sender.append(ratio)
                        elif investor in defector_ids:
                            zero_return_opportunities_from_defectors += 1
                by_round.append(
                    {
                        "round": int(entry["round"]),
                        "standard_send_ratio": np.mean(round_sent)
                        if round_sent
                        else math.nan,
                    }
                )

                for agent_id, text in (entry.get("myths") or {}).items():
                    author_type = (
                        "defector" if agent_id in defector_ids else "standard"
                    )
                    myth_rows.append(
                        {
                            "model_id": model_id,
                            "model_label": spec["label"],
                            "condition": condition,
                            "condition_label": CONDITION_LABELS[condition],
                            "replicate_id": replicate_id,
                            "round": int(entry["round"]),
                            "agent_id": agent_id,
                            "author_type": author_type,
                            "text": text,
                            **myth_metrics(text),
                        }
                    )

            final_balances = (history[-1].get("balances") or {}) if history else {}
            standard_final = [
                float(final_balances[agent_id])
                for agent_id in standard_ids
                if agent_id in final_balances
            ]
            population_final = [float(value) for value in final_balances.values()]
            run_rows.append(
                {
                    "model_id": model_id,
                    "model_label": spec["label"],
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "replicate_id": replicate_id,
                    "defector_ids": ",".join(sorted(defector_ids)),
                    "n_standard": len(standard_ids),
                    "standard_send_ratio": np.mean(sent_all),
                    "round1_standard_send": by_round[0]["standard_send_ratio"],
                    "standard_send_to_standard": np.mean(sent_to_standard),
                    "standard_send_to_defector": np.mean(sent_to_defector)
                    if sent_to_defector
                    else math.nan,
                    "standard_return_ratio": np.mean(return_ratios)
                    if return_ratios
                    else math.nan,
                    "standard_return_from_standard_sender": np.mean(
                        return_from_standard_sender
                    )
                    if return_from_standard_sender
                    else math.nan,
                    "zero_return_opportunities_from_defectors": (
                        zero_return_opportunities_from_defectors
                    ),
                    "standard_final_balance": np.mean(standard_final),
                    "population_final_balance": np.mean(population_final),
                }
            )
            behavior_round_rows.extend(
                {
                    "model_id": model_id,
                    "model_label": spec["label"],
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "replicate_id": replicate_id,
                    **point,
                }
                for point in by_round
            )
            totals, attempts, retries, forced = usage(run)
            usage_rows.append(
                {
                    "model_id": model_id,
                    "condition": condition,
                    "replicate_id": replicate_id,
                    **totals,
                    "attempts": attempts,
                    "recovered_retries": retries,
                    "forced_responses": forced,
                }
            )

        for condition in CONDITION_ORDER:
            if seen[condition] != expected_ids:
                raise RuntimeError(
                    f"{model_id}/{condition} has IDs {sorted(seen[condition])}"
                )
    return run_rows, behavior_round_rows, myth_rows, usage_rows


def aggregate_myths(myth_dataframe):
    metrics = [
        "coop_density",
        "coop_present",
        "threat_density",
        "threat_present",
        "half_rule",
        "word_count",
    ]
    return (
        myth_dataframe.groupby(
            [
                "model_id",
                "model_label",
                "condition",
                "condition_label",
                "replicate_id",
                "author_type",
            ],
            as_index=False,
        )[metrics]
        .mean()
        .sort_values(["model_id", "condition", "replicate_id", "author_type"])
    )


def summary_records(run_dataframe, myth_aggregates):
    records = []
    run_metrics = [
        "standard_send_ratio",
        "round1_standard_send",
        "standard_return_ratio",
        "standard_final_balance",
        "population_final_balance",
    ]
    for model_id in MODEL_SPECS:
        for condition in CONDITION_ORDER:
            subset = run_dataframe[
                (run_dataframe["model_id"] == model_id)
                & (run_dataframe["condition"] == condition)
            ]
            for metric in run_metrics:
                values = subset[metric].to_numpy(dtype=float)
                low, high = ci(values)
                records.append(
                    {
                        "domain": "behavior",
                        "model_id": model_id,
                        "condition": condition,
                        "author_type": "standard",
                        "metric": metric,
                        "n": len(values),
                        "mean": np.nanmean(values),
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
            myths = myth_aggregates[
                (myth_aggregates["model_id"] == model_id)
                & (myth_aggregates["condition"] == condition)
                & (myth_aggregates["author_type"] == "standard")
            ]
            for metric in [
                "coop_density",
                "threat_density",
                "half_rule",
                "word_count",
            ]:
                values = myths[metric].to_numpy(dtype=float)
                low, high = ci(values)
                records.append(
                    {
                        "domain": "myth",
                        "model_id": model_id,
                        "condition": condition,
                        "author_type": "standard",
                        "metric": metric,
                        "n": len(values),
                        "mean": np.nanmean(values),
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return records


def contrast_records(run_dataframe, myth_aggregates, myth_dataframe):
    records = []
    for model_id in MODEL_SPECS:
        model_runs = run_dataframe[run_dataframe["model_id"] == model_id]
        for metric in [
            "standard_send_ratio",
            "round1_standard_send",
            "standard_return_ratio",
            "standard_final_balance",
            "population_final_balance",
        ]:
            pivot = model_runs.pivot(
                index="replicate_id", columns="condition", values=metric
            )
            records.append(
                paired_record(
                    pivot["defectors25"] - pivot["control"],
                    model_id,
                    metric,
                    "2 of 8 defectors − no defectors",
                    primary=metric == "standard_send_ratio",
                )
            )

        treatment = model_runs[model_runs["condition"] == "defectors25"]
        records.append(
            paired_record(
                treatment["standard_send_to_defector"].to_numpy(dtype=float)
                - treatment["standard_send_to_standard"].to_numpy(dtype=float),
                model_id,
                "standard_send_partner_contrast",
                "Send to defector − send to standard (treatment)",
            )
        )

        standard_myths = myth_aggregates[
            (myth_aggregates["model_id"] == model_id)
            & (myth_aggregates["author_type"] == "standard")
        ]
        for metric in [
            "coop_density",
            "coop_present",
            "threat_density",
            "threat_present",
            "half_rule",
            "word_count",
        ]:
            pivot = standard_myths.pivot(
                index="replicate_id", columns="condition", values=metric
            )
            records.append(
                paired_record(
                    pivot["defectors25"] - pivot["control"],
                    model_id,
                    f"standard_myth_{metric}",
                    "Treatment-standard − control-standard",
                    primary=metric == "coop_density",
                )
            )

        treatment_myths = myth_aggregates[
            (myth_aggregates["model_id"] == model_id)
            & (myth_aggregates["condition"] == "defectors25")
        ]
        for metric in ["coop_density", "threat_density", "half_rule", "word_count"]:
            pivot = treatment_myths.pivot(
                index="replicate_id", columns="author_type", values=metric
            )
            records.append(
                paired_record(
                    pivot["defector"] - pivot["standard"],
                    model_id,
                    f"defector_myth_{metric}",
                    "Defector-authored − standard-authored (treatment)",
                )
            )

        treatment_raw = myth_dataframe[
            (myth_dataframe["model_id"] == model_id)
            & (myth_dataframe["condition"] == "defectors25")
        ]
        for period, period_subset in (
            ("round1", treatment_raw[treatment_raw["round"] == 1]),
            ("rounds2_10", treatment_raw[treatment_raw["round"] >= 2]),
        ):
            by_population_author = (
                period_subset.groupby(
                    ["replicate_id", "author_type"], as_index=False
                )[["coop_density", "threat_density"]]
                .mean()
            )
            for metric in ["coop_density", "threat_density"]:
                pivot = by_population_author.pivot(
                    index="replicate_id", columns="author_type", values=metric
                )
                records.append(
                    paired_record(
                        pivot["defector"] - pivot["standard"],
                        model_id,
                        f"defector_myth_{metric}_{period}",
                        f"Defector-authored − standard-authored ({period})",
                    )
                )
    return records


def plot_behavior_trajectories(round_dataframe, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    colors = {"control": "#2a9d8f", "defectors25": "#e76f51"}
    for ax, model_id in zip(axes, MODEL_SPECS):
        model = round_dataframe[round_dataframe["model_id"] == model_id]
        for condition in CONDITION_ORDER:
            subset = model[model["condition"] == condition]
            grouped = subset.groupby("round")["standard_send_ratio"].agg(
                ["mean", "sem"]
            )
            rounds = grouped.index.to_numpy(dtype=float)
            means = grouped["mean"].to_numpy(dtype=float)
            errors = grouped["sem"].to_numpy(dtype=float) * stats.t.ppf(0.975, 4)
            ax.plot(
                rounds,
                means,
                marker="o",
                linewidth=2.4,
                color=colors[condition],
                label=CONDITION_LABELS[condition],
            )
            ax.fill_between(
                rounds,
                means - errors,
                means + errors,
                color=colors[condition],
                alpha=0.14,
            )
        ax.set_title(MODEL_SPECS[model_id]["label"], fontweight="bold")
        ax.set_xlabel("Round")
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Mean proportion sent by standard agents")
    axes[0].set_ylim(0, 1.03)
    axes[1].legend(title="Population")
    fig.suptitle("Do mechanical defectors change ordinary-agent sending?", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "standard_sending_trajectories.png", dpi=300)
    plt.close(fig)


def plot_behavior_contrasts(contrasts, output_dir):
    import matplotlib.pyplot as plt

    selected = contrasts[contrasts["metric"] == "standard_send_ratio"].copy()
    fig, ax = plt.subplots(figsize=(9, 5.8))
    positions = np.arange(len(selected))
    estimates = selected["estimate"].to_numpy(dtype=float)
    lows = selected["ci_low"].to_numpy(dtype=float)
    highs = selected["ci_high"].to_numpy(dtype=float)
    colors = [MODEL_SPECS[row]["color"] for row in selected["model_id"]]
    for position, estimate, low, high, color in zip(
        positions, estimates, lows, highs, colors
    ):
        ax.errorbar(
            position,
            estimate,
            yerr=[[estimate - low], [high - estimate]],
            fmt="o",
            markersize=9,
            capsize=6,
            linewidth=2.5,
            color=color,
        )
    ax.axhline(0, color="#263238", linestyle="--", linewidth=1)
    ax.set_xticks(positions, selected["model_label"])
    ax.set_ylabel("2 defectors − control\n(proportion sent by standard agents)")
    ax.set_title("Frozen ordinary-agent spillover contrast", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "standard_sending_contrasts.png", dpi=300)
    plt.close(fig)


def plot_myth_contrasts(contrasts, output_dir):
    import matplotlib.pyplot as plt

    metric_specs = [
        ("standard_myth_coop_density", "Cooperation/fairness"),
        ("standard_myth_threat_density", "Defection/threat"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (metric, title) in zip(axes, metric_specs):
        selected = contrasts[contrasts["metric"] == metric]
        positions = np.arange(len(selected))
        for position, (_, row) in zip(positions, selected.iterrows()):
            ax.errorbar(
                position,
                row["estimate"],
                yerr=[
                    [row["estimate"] - row["ci_low"]],
                    [row["ci_high"] - row["estimate"]],
                ],
                fmt="o",
                markersize=9,
                capsize=6,
                linewidth=2.5,
                color=MODEL_SPECS[row["model_id"]]["color"],
            )
        ax.axhline(0, color="#263238", linestyle="--", linewidth=1)
        ax.set_xticks(positions, selected["model_label"], rotation=8)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Treatment − control density\n(matches per 100 myth words)")
    fig.suptitle("Frozen changes in standard-agent myth language", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "standard_myth_language_contrasts.png", dpi=300)
    plt.close(fig)


def plot_myth_author_trajectories(myth_dataframe, output_dir):
    import matplotlib.pyplot as plt

    treatment = myth_dataframe[myth_dataframe["condition"] == "defectors25"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    colors = {"standard": "#2a9d8f", "defector": "#9c2f1f"}
    labels = {"standard": "Standard authors", "defector": "Defector authors"}
    for ax, model_id in zip(axes, MODEL_SPECS):
        model = treatment[treatment["model_id"] == model_id]
        for author_type in ["standard", "defector"]:
            author = model[model["author_type"] == author_type]
            by_population_round = (
                author.groupby(["replicate_id", "round"], as_index=False)[
                    "coop_density"
                ]
                .mean()
            )
            grouped = by_population_round.groupby("round")["coop_density"].agg(
                ["mean", "sem"]
            )
            rounds = grouped.index.to_numpy(dtype=float)
            means = grouped["mean"].to_numpy(dtype=float)
            errors = grouped["sem"].to_numpy(dtype=float) * stats.t.ppf(0.975, 4)
            ax.plot(
                rounds,
                means,
                marker="o",
                linewidth=2.4,
                color=colors[author_type],
                label=labels[author_type],
            )
            ax.fill_between(
                rounds,
                means - errors,
                means + errors,
                color=colors[author_type],
                alpha=0.14,
            )
        ax.axvline(1.5, color="#6c757d", linestyle=":", linewidth=1.3)
        ax.set_title(MODEL_SPECS[model_id]["label"], fontweight="bold")
        ax.set_xlabel("Round")
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Cooperation/fairness matches per 100 myth words")
    axes[1].legend(title="Treatment myths")
    fig.suptitle(
        "Does forced defection become visible in the myths?", fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "treatment_myth_author_trajectories.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    import pandas as pd

    run_rows, round_rows, myth_rows, usage_rows = load_data(args.input)
    runs = pd.DataFrame(run_rows).sort_values(
        ["model_id", "condition", "replicate_id"]
    )
    rounds = pd.DataFrame(round_rows).sort_values(
        ["model_id", "condition", "replicate_id", "round"]
    )
    myths = pd.DataFrame(myth_rows).sort_values(
        ["model_id", "condition", "replicate_id", "round", "agent_id"]
    )
    myth_aggregates = aggregate_myths(myths)
    summaries = pd.DataFrame(summary_records(runs, myth_aggregates))
    contrasts = pd.DataFrame(contrast_records(runs, myth_aggregates, myths))
    usage_dataframe = pd.DataFrame(usage_rows).sort_values(
        ["model_id", "condition", "replicate_id"]
    )

    runs.to_csv(args.out / "run_metrics.csv", index=False)
    rounds.to_csv(args.out / "round_metrics.csv", index=False)
    myths.to_csv(args.out / "myth_metrics.csv", index=False)
    myth_aggregates.to_csv(args.out / "myth_population_metrics.csv", index=False)
    summaries.to_csv(args.out / "summary.csv", index=False)
    contrasts.to_csv(args.out / "paired_contrasts.csv", index=False)
    usage_dataframe.to_csv(args.out / "token_usage.csv", index=False)
    plot_behavior_trajectories(rounds, args.out)
    plot_behavior_contrasts(contrasts, args.out)
    plot_myth_contrasts(contrasts, args.out)
    plot_myth_author_trajectories(myths, args.out)

    print("Run-level behavior:")
    print(runs.to_string(index=False))
    print("\nFrozen and secondary contrasts:")
    print(contrasts.to_string(index=False))
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
