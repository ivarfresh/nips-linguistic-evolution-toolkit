#!/usr/bin/env python3
"""Analyze the frozen Gemini 3.7 defector/punishment population screen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyses._shared import configure_matplotlib
from scripts import analyze_defector_punishment_gemini_factorial_confirmation_n10 as base


DEFAULT_INPUT = Path(
    "data/json/noise_experiments/defector_punishment_gemini37_n3_20260823"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/defector_punishment_gemini37_n3_20260823"
)


def configure_base():
    base.EXPERIMENT = "noise8i_defector_punishment_gemini37_n3"
    base.EXPECTED_IDS = {93, 94, 95}
    base.MODEL = "google/gemini-3.7-flash"
    base.PROVIDER_MODEL = "gemini-3.7-flash"
    base.EXPECTED_RUNTIME_METADATA = {
        "thinking_level": "medium",
        "thinking_level_source": "GEMINI_THINKING_LEVEL",
        "temperature_sent": False,
        "request_timeout_seconds": 300.0,
        "request_timeout_source": "GEMINI_REQUEST_TIMEOUT_SECONDS",
    }


def paired_targeting(deductions):
    rows = []
    treatment = deductions[deductions["condition"] == "defectors25"]
    for replicate_id in sorted(base.EXPECTED_IDS):
        cell = treatment[treatment["replicate_id"] == replicate_id]
        target = cell.groupby("target_type")[[
            "deduction_spent", "any_deduction"
        ]].mean()
        rows.append(
            {
                "replicate_id": replicate_id,
                "intensity_difference": (
                    target.loc["defector", "deduction_spent"]
                    - target.loc["standard", "deduction_spent"]
                ),
                "probability_difference": (
                    target.loc["defector", "any_deduction"]
                    - target.loc["standard", "any_deduction"]
                ),
            }
        )
    return rows


def decision_table(contrasts, replicate_contrasts, deductions, targeting):
    indexed = contrasts.set_index(["metric", "contrast_type"])
    direct = indexed.loc[("standard_return_ratio", "availability_defectors25")]
    interaction = indexed.loc[("standard_return_ratio", "interaction")]
    target_intensity = float(np.mean([row["intensity_difference"] for row in targeting]))
    target_probability = float(np.mean([row["probability_difference"] for row in targeting]))
    all_target_positive = all(row["intensity_difference"] > 0 for row in targeting)
    fair = deductions[
        np.isfinite(deductions["visible_return_ratio"])
        & (deductions["visible_return_ratio"] >= .5)
    ]
    fair_deductions = int((fair["deduction_spent"] > 0).sum())
    targeting_pass = (
        target_intensity >= 1
        and target_probability >= .5
        and all_target_positive
        and fair_deductions == 0
    )
    direct_negative = int((replicate_contrasts[
        (replicate_contrasts["metric"] == "standard_return_ratio")
        & (replicate_contrasts["contrast_type"] == "availability_defectors25")
    ]["difference"] < 0).sum())
    interaction_negative = int((replicate_contrasts[
        (replicate_contrasts["metric"] == "standard_return_ratio")
        & (replicate_contrasts["contrast_type"] == "interaction")
    ]["difference"] < 0).sum())
    direct_pass = direct["estimate"] <= -.025 and direct_negative >= 2
    interaction_pass = interaction["estimate"] <= -.025 and interaction_negative >= 2
    if targeting_pass and direct_pass and interaction_pass:
        action = "expand full 2x2"
    elif targeting_pass and direct_pass:
        action = "expand only matched defector availability cells"
    else:
        action = "do not scale population design"
    return {
        "target_intensity_difference": target_intensity,
        "target_probability_difference": target_probability,
        "all_three_targeting_differences_positive": all_target_positive,
        "fair_return_deductions": fair_deductions,
        "targeting_gate": targeting_pass,
        "defector_return_availability_effect": direct["estimate"],
        "defector_return_negative_pairs": direct_negative,
        "defector_return_gate": direct_pass,
        "return_interaction": interaction["estimate"],
        "return_interaction_same_direction_pairs": interaction_negative,
        "moderation_gate": interaction_pass,
        "next_action": action,
    }


def author_myth_rows(input_root):
    rows = []
    for _path, run in base.load_runs(input_root / base.EXPERIMENT):
        metadata = run.get("run_metadata") or {}
        availability = "on" if metadata.get("punishment_enabled") else "off"
        defector_ids = set(metadata.get("defector_agent_ids") or [])
        condition = "defectors25" if defector_ids else "control"
        replicate_id = int(metadata["replicate_id"])
        for entry in run.get("conversation_history") or []:
            for agent_id, text in (entry.get("myths") or {}).items():
                metrics = base.myth_metrics(text)
                words = re.findall(r"\b[\w'-]+\b", text.lower())
                punishment_matches = len(base.PUNISH_PATTERN.findall(text))
                rows.append(
                    {
                        "availability": availability,
                        "condition": condition,
                        "replicate_id": replicate_id,
                        "round": int(entry["round"]),
                        "agent_id": agent_id,
                        "author_type": (
                            "defector" if agent_id in defector_ids else "standard"
                        ),
                        "coop_density": metrics["coop_density"],
                        "threat_density": metrics["threat_density"],
                        "half_rule": metrics["half_rule"],
                        "punishment_density": (
                            100 * punishment_matches / len(words) if words else 0
                        ),
                    }
                )
    return rows


def return_rule_rows(input_root):
    """Extract ordinary-receiver choices against the amount shown in the prompt."""
    rows = []
    for _path, run in base.load_runs(input_root / base.EXPERIMENT):
        metadata = run.get("run_metadata") or {}
        availability = "on" if metadata.get("punishment_enabled") else "off"
        defector_ids = set(metadata.get("defector_agent_ids") or [])
        condition = "defectors25" if defector_ids else "control"
        replicate_id = int(metadata["replicate_id"])
        for entry in run.get("conversation_history") or []:
            for dyad in entry.get("dyads") or []:
                if dyad["trustee"] in defector_ids or dyad["received"] <= 0:
                    continue
                visible_receipt = float(dyad["received_communicated"])
                returned = float(dyad["returned"])
                error = returned - visible_receipt / 2
                rows.append(
                    {
                        "availability": availability,
                        "condition": condition,
                        "replicate_id": replicate_id,
                        "round": int(entry["round"]),
                        "visible_receipt": visible_receipt,
                        "returned": returned,
                        "half_visible_error": error,
                        "exact_half_visible": bool(np.isclose(error, 0)),
                        "within_half_cent": abs(error) <= .0051,
                    }
                )
    return rows


def author_postround_contrasts(author_myths):
    """Compare defector and ordinary authors after shared round-one priors."""
    metrics = ["coop_density", "threat_density", "half_rule", "punishment_density"]
    post = author_myths[
        (author_myths["condition"] == "defectors25")
        & (author_myths["round"] >= 2)
    ]
    grouped = post.groupby(
        ["availability", "replicate_id", "author_type"], as_index=False
    )[metrics].mean()
    rows = []
    for availability in base.AVAILABILITY_ORDER:
        cell = grouped[grouped["availability"] == availability]
        for metric in metrics:
            wide = cell.pivot(
                index="replicate_id", columns="author_type", values=metric
            )
            differences = wide["defector"] - wide["standard"]
            low, high = base.ci(differences)
            rows.append(
                {
                    "availability": availability,
                    "metric": metric,
                    "contrast": "defector author - ordinary author, rounds 2-10",
                    "n_populations": len(differences),
                    "estimate": float(differences.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "positive_pairs": int((differences > 0).sum()),
                    "negative_pairs": int((differences < 0).sum()),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_base()
    configure_matplotlib()

    import pandas as pd

    extracted = base.extract(args.input)
    runs = pd.DataFrame(extracted[0]).sort_values(
        ["availability", "condition", "replicate_id"]
    )
    rounds = pd.DataFrame(extracted[1]).sort_values(
        ["availability", "condition", "replicate_id", "round"]
    )
    myths = pd.DataFrame(extracted[2]).sort_values(
        ["availability", "condition", "replicate_id", "round", "agent_id"]
    )
    myths["text"] = myths["text"].str.replace(
        r"[ \t]+(?=\n|$)", "", regex=True
    )
    returns = pd.DataFrame(extracted[3]).sort_values(
        ["availability", "condition", "replicate_id", "round", "trustee_id"]
    )
    deductions = pd.DataFrame(extracted[4]).sort_values(
        ["condition", "replicate_id", "round", "target_type"]
    )
    token_usage = pd.DataFrame(extracted[5]).sort_values(
        ["availability", "condition", "replicate_id"]
    )
    audits = extracted[6]
    contrast_rows, replicate_rows = base.make_contrasts(runs)
    contrasts = pd.DataFrame(contrast_rows)
    replicate_contrasts = pd.DataFrame(replicate_rows)
    targeting_rows = paired_targeting(deductions)
    decision = pd.DataFrame(
        [decision_table(contrasts, replicate_contrasts, deductions, targeting_rows)]
    )
    term_counts = pd.DataFrame(base.make_term_counts(myths))
    author_myths = pd.DataFrame(author_myth_rows(args.input)).sort_values(
        ["availability", "condition", "replicate_id", "round", "agent_id"]
    )
    author_myth_summary = author_myths.groupby(
        ["availability", "condition", "author_type"], as_index=False
    )[["coop_density", "threat_density", "half_rule", "punishment_density"]].mean()
    author_postround = pd.DataFrame(author_postround_contrasts(author_myths))
    return_rules = pd.DataFrame(return_rule_rows(args.input)).sort_values(
        ["availability", "condition", "replicate_id", "round"]
    )
    return_rule_summary = return_rules.groupby(
        ["availability", "condition"], as_index=False
    ).agg(
        n=("returned", "size"),
        exact_half_visible=("exact_half_visible", "sum"),
        within_half_cent=("within_half_cent", "sum"),
        mean_absolute_error=("half_visible_error", lambda values: np.abs(values).mean()),
        max_absolute_error=("half_visible_error", lambda values: np.abs(values).max()),
    )
    audit_table = pd.DataFrame(
        [
            {
                key: value
                for key, value in audit.items()
                if key not in {"issues", "pairing_signature"}
            }
            for audit in audits
        ]
    )
    input_tokens = int(token_usage["input_tokens"].sum())
    output_tokens = int(token_usage["output_tokens"].sum())
    reasoning_tokens = int(token_usage["reasoning_tokens"].sum())
    cost = (
        input_tokens * .75 + (output_tokens + reasoning_tokens) * 3.75
    ) / 1_000_000
    cost_frame = pd.DataFrame(
        [{
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "attempts": int(token_usage["attempts"].sum()),
            "recovered_retries": int(token_usage["recovered_retries"].sum()),
            "estimated_list_price_usd": cost,
        }]
    )

    runs.to_csv(args.out / "run_metrics.csv", index=False)
    rounds.to_csv(args.out / "round_metrics.csv", index=False)
    myths.to_csv(args.out / "myth_metrics.csv", index=False)
    returns.to_csv(args.out / "return_decisions.csv", index=False)
    deductions.to_csv(args.out / "deduction_decisions.csv", index=False)
    token_usage.to_csv(args.out / "token_usage_by_run.csv", index=False)
    cost_frame.to_csv(args.out / "token_cost.csv", index=False)
    contrasts.to_csv(args.out / "contrasts.csv", index=False)
    replicate_contrasts.to_csv(args.out / "replicate_contrasts.csv", index=False)
    pd.DataFrame(targeting_rows).to_csv(args.out / "targeting_by_replicate.csv", index=False)
    decision.to_csv(args.out / "screen_decision.csv", index=False)
    audit_table.to_csv(args.out / "audit.csv", index=False)
    term_counts.to_csv(args.out / "myth_term_counts.csv", index=False)
    author_myths.to_csv(args.out / "author_myth_metrics.csv", index=False)
    author_myth_summary.to_csv(args.out / "author_myth_summary.csv", index=False)
    author_postround.to_csv(args.out / "author_myth_postround_contrasts.csv", index=False)
    return_rules.to_csv(args.out / "return_rule_decisions.csv", index=False)
    return_rule_summary.to_csv(args.out / "return_rule_summary.csv", index=False)
    base.plot_behavior(runs, args.out)
    base.plot_return_trajectories(rounds, args.out)
    base.plot_myth_interactions(contrasts, args.out)

    print("Frozen screen decision:\n", decision.to_string(index=False))
    print("\nBehavior contrasts:\n", contrasts[
        contrasts["metric"].isin(["standard_send_ratio", "standard_return_ratio"])
    ].to_string(index=False))
    print(f"\nEstimated list-price cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
