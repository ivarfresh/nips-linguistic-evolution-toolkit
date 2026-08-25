#!/usr/bin/env python3
"""Analyze the frozen Gemini hidden-defector punishment mechanism pilot."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_defector_punishment_gpt_n5 as base
from analyze_corrected_v2_confirmatory import load_runs


INPUT_ROOT = Path(
    "data/json/noise_experiments/defector_punishment_gemini_n3_20260821"
)
EXPERIMENT = "noise8i_defector_punishment_gemini_n3"
EXPECTED_IDS = {66, 67, 68}
EXPECTED_COMMIT = "4e5a47d4f19d3869d16dba1f3abcb48df01dbfe1"
EXPECTED_CONFIG_SHA256 = (
    "e5196e510beb6cc4dc18305c2da828585a0fa14c28521091ccfcd1d84d813ff9"
)
OUTPUT_ROOT = Path(
    "docs/figures/defector_punishment_gemini_n3_20260821"
)


def validate_frozen_inputs() -> None:
    runs = load_runs(INPUT_ROOT / EXPERIMENT)
    if len(runs) != 6:
        raise RuntimeError(f"Found {len(runs)} runs; expected 6")
    seen = {"control": set(), "defectors25": set()}
    for path, run in runs:
        metadata = run.get("run_metadata") or {}
        condition = (
            "defectors25"
            if float(metadata.get("defector_ratio_actual") or 0) > 0
            else "control"
        )
        seen[condition].add(int(metadata["replicate_id"]))
        expected = {
            "model": "google/gemini-3.1-flash-lite",
            "llm_provider": "google",
            "provider_model": "gemini-3.1-flash-lite",
            "code_commit": EXPECTED_COMMIT,
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "punishment_prompt_variant": "current",
            "execution_provenance_version": 1,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(
                    f"{path}: {key}={metadata.get(key)!r}; expected {value!r}"
                )
        if metadata.get("code_dirty") is not False:
            raise RuntimeError(f"{path}: dirty or missing execution provenance")
    if any(ids != EXPECTED_IDS for ids in seen.values()):
        raise RuntimeError(f"Unexpected replicate sets: {seen}")


def main() -> None:
    validate_frozen_inputs()
    base.DEFAULT_INPUT = INPUT_ROOT
    base.DEFAULT_OUTPUT = OUTPUT_ROOT
    base.EXPERIMENT_DIR = EXPERIMENT
    base.EXPECTED_IDS = EXPECTED_IDS
    base.MODEL = "google/gemini-3.1-flash-lite"
    base.RETURN_PLOT_TITLE = "Gemini reserved punishment for visibly low returns"
    base.TRAJECTORY_PLOT_TITLE = (
        "Behavior across the ten-round Gemini punishment pilot"
    )
    base.main()

    import pandas as pd

    decisions = pd.read_csv(OUTPUT_ROOT / "deduction_decisions.csv")
    treatment = decisions[decisions["condition"] == "defectors25"]
    defector = treatment[treatment["target_type"] == "defector"]
    standard = treatment[treatment["target_type"] == "standard"]
    adequate = treatment[treatment["adequate_half_return"] == 1]
    gate = pd.DataFrame(
        [
            {
                "criterion": "defector_minus_standard_mean_spending",
                "observed": defector["deduction_spent"].mean()
                - standard["deduction_spent"].mean(),
                "threshold": 0.5,
                "comparison": ">=",
            },
            {
                "criterion": "defector_minus_standard_any_probability",
                "observed": defector["any_deduction"].mean()
                - standard["any_deduction"].mean(),
                "threshold": 0.25,
                "comparison": ">=",
            },
            {
                "criterion": "defector_target_any_probability",
                "observed": defector["any_deduction"].mean(),
                "threshold": 0.5,
                "comparison": ">=",
            },
            {
                "criterion": "any_after_visible_half_return",
                "observed": adequate["any_deduction"].mean(),
                "threshold": 0.25,
                "comparison": "<=",
            },
        ]
    )
    gate["passed"] = [
        value >= threshold if comparison == ">=" else value <= threshold
        for value, threshold, comparison in zip(
            gate["observed"], gate["threshold"], gate["comparison"]
        )
    ]
    gate.to_csv(OUTPUT_ROOT / "mechanism_gate.csv", index=False)
    if not gate["passed"].all():
        raise RuntimeError(f"Frozen mechanism gate failed:\n{gate}")


if __name__ == "__main__":
    main()
