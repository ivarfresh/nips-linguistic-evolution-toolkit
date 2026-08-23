#!/usr/bin/env python3
"""Audit and analyze the frozen Gemini 3.7 Flash task-order screen."""

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
from scripts.analyze_corrected_v2_confirmatory import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_ORDER,
    load_runs,
    run_metrics,
)
from scripts.audit_v2_protocol import audit_paired_schedules, audit_run


DEFAULT_INPUT = Path(
    "data/json/noise_experiments/gemini37_flash_task_order_n3_20260823"
)
DEFAULT_OUTPUT = Path("docs/figures/gemini37_flash_task_order_n3_20260823")
CELL_DIRECTORIES = {
    "game": "noise8_crossmodel_gemini37_flash_n3_game",
    "game_myth": "noise8_crossmodel_gemini37_flash_n3_game_myth",
    "myth_game": "noise8_crossmodel_gemini37_flash_n3_myth_game",
}
EXPECTED_REPLICATES = {90, 91, 92}
EXPECTED_MODEL = "google/gemini-3.7-flash"


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.allclose(values, values[0]):
        return float(values.mean()), float(values.mean())
    sem = stats.sem(values)
    return stats.t.interval(0.95, len(values) - 1, loc=values.mean(), scale=sem)


def usage_records(run):
    for agent in (run.get("agents") or {}).values():
        for event in agent.get("interaction_history") or []:
            usage = ((event.get("response") or {}).get("usage") or {})
            if usage:
                yield usage


def sender_counts(run):
    total = 0
    maximum = 0
    for round_data in run.get("conversation_history") or []:
        for dyad in round_data.get("dyads") or []:
            total += 1
            maximum += int(math.isclose(float(dyad["sent"]), 5.0, abs_tol=1e-9))
    return total, maximum


def load_batch(input_dir):
    rows = []
    trajectories = []
    audit_results = []
    audit_rows = []
    usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

    for condition, experiment in CELL_DIRECTORIES.items():
        runs = load_runs(input_dir / experiment)
        if len(runs) != 3:
            raise RuntimeError(f"Expected three runs in {experiment}; found {len(runs)}")
        observed = set()
        for path, run in runs:
            metadata = run.get("run_metadata") or {}
            replicate_id = int(metadata.get("replicate_id"))
            observed.add(replicate_id)
            expected = {
                "model": EXPECTED_MODEL,
                "llm_provider": "google",
                "provider_model": "gemini-3.7-flash",
                "thinking_level": "medium",
                "thinking_level_source": "GEMINI_THINKING_LEVEL",
                "temperature_sent": False,
                "request_timeout_seconds": 300.0,
                "request_timeout_source": "GEMINI_REQUEST_TIMEOUT_SECONDS",
                "code_dirty": False,
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise RuntimeError(
                        f"{path}: expected {key}={value!r}; got {metadata.get(key)!r}"
                    )
            if not metadata.get("code_commit") or not metadata.get("config_sha256"):
                raise RuntimeError(f"{path}: missing immutable provenance")

            audited = audit_run(path)
            audit_results.append(audited)
            audit_rows.append(
                {
                    "path": str(path),
                    "condition": condition,
                    "replicate_id": replicate_id,
                    "issues": len(audited["issues"]),
                    "accepted_calls": audited["calls"],
                    "attempts": audited["attempts"],
                    "retries": audited["retry_attempts"],
                    "noise_checks": audited["noise_checks"],
                }
            )

            metrics, trajectory = run_metrics(path, run)
            total_sends, maximum_sends = sender_counts(run)
            rows.append(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "replicate_id": replicate_id,
                    "pairing_seed": metadata.get("pairing_seed"),
                    "noise_seed": metadata.get("noise_seed"),
                    "total_sends": total_sends,
                    "maximum_sends": maximum_sends,
                    "maximum_send_rate": maximum_sends / total_sends,
                    **metrics,
                }
            )
            trajectories.extend(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "replicate_id": replicate_id,
                    **point,
                }
                for point in trajectory
            )
            for record in usage_records(run):
                for key in usage:
                    usage[key] += int(record.get(key) or 0)
        if observed != EXPECTED_REPLICATES:
            raise RuntimeError(f"{experiment}: replicate IDs {sorted(observed)}")

    audit_paired_schedules(audit_results)
    issues = [issue for result in audit_results for issue in result["issues"]]
    if issues:
        raise RuntimeError("Joint audit failed:\n" + "\n".join(issues))
    if sum(row["accepted_calls"] for row in audit_rows) != 1200:
        raise RuntimeError("Acceptance gate failed: expected 1,200 accepted calls")
    if sum(row["noise_checks"] for row in audit_rows) != 720:
        raise RuntimeError("Acceptance gate failed: expected 720 noise checks")
    return rows, trajectories, audit_rows, usage


def summarize(dataframe):
    records = []
    for condition in CONDITION_ORDER:
        cell = dataframe[dataframe["condition"] == condition]
        for metric in [
            "final_balance",
            "mean_trust_ratio",
            "mean_return_ratio",
            "maximum_send_rate",
        ]:
            values = cell[metric].to_numpy(dtype=float)
            low, high = confidence_interval(values)
            records.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "n": len(values),
                    "mean": values.mean(),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return records


def contrasts(dataframe):
    records = []
    for left, right in [
        ("game_myth", "game"),
        ("myth_game", "game"),
        ("myth_game", "game_myth"),
    ]:
        for metric in ["final_balance", "mean_trust_ratio", "mean_return_ratio"]:
            a = dataframe[dataframe["condition"] == left].set_index("replicate_id")[metric]
            b = dataframe[dataframe["condition"] == right].set_index("replicate_id")[metric]
            differences = (a - b).sort_index().to_numpy(dtype=float)
            low, high = confidence_interval(differences)
            p_value = (
                float(stats.ttest_1samp(differences, 0).pvalue)
                if not np.allclose(differences, 0)
                else 1.0
            )
            records.append(
                {
                    "metric": metric,
                    "contrast": f"{left} - {right}",
                    "estimate": differences.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "p_value_descriptive": p_value,
                    "positive_pairs": int((differences > 0).sum()),
                    "zero_pairs": int((differences == 0).sum()),
                }
            )
    return records


def plot_results(dataframe, trajectories, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for replicate_id in sorted(EXPECTED_REPLICATES):
        paired = dataframe[dataframe["replicate_id"] == replicate_id].set_index("condition")
        axes[0].plot(
            range(3),
            [paired.loc[c, "final_balance"] for c in CONDITION_ORDER],
            marker="o",
            color="#90a4ae",
            alpha=.75,
        )
    means = dataframe.groupby("condition")["final_balance"].mean()
    axes[0].scatter(
        range(3), [means[c] for c in CONDITION_ORDER], s=130, color="#263238", zorder=3
    )
    axes[0].set_xticks(range(3), [CONDITION_LABELS[c] for c in CONDITION_ORDER])
    axes[0].set_ylabel("Final balance per agent")
    axes[0].set_title("Matched populations")
    axes[0].grid(axis="y", alpha=.25)

    trajectory_dataframe = trajectories
    for condition in CONDITION_ORDER:
        cell = trajectory_dataframe[trajectory_dataframe["condition"] == condition]
        mean = cell.groupby("round", as_index=False)["trust_ratio"].mean()
        axes[1].plot(
            mean["round"], mean["trust_ratio"], marker="o",
            color=CONDITION_COLORS[condition], label=CONDITION_LABELS[condition]
        )
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xticks(range(1, 11))
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Proportion sent")
    axes[1].set_title("Sending trajectories")
    axes[1].grid(alpha=.25)
    axes[1].legend()
    fig.suptitle("Gemini 3.7 Flash task-order screen", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "task_order_screen.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    configure_matplotlib()
    import pandas as pd

    args.out.mkdir(parents=True, exist_ok=True)
    rows, trajectory_rows, audit_rows, usage = load_batch(args.input)
    dataframe = pd.DataFrame(rows).sort_values(["condition", "replicate_id"])
    trajectory_dataframe = pd.DataFrame(trajectory_rows)
    summary = pd.DataFrame(summarize(dataframe))
    contrast_frame = pd.DataFrame(contrasts(dataframe))

    maximum_sends = int(dataframe["maximum_sends"].sum())
    total_sends = int(dataframe["total_sends"].sum())
    myth_game = contrast_frame[
        (contrast_frame["metric"] == "final_balance")
        & (contrast_frame["contrast"] == "myth_game - game")
    ].iloc[0]
    headroom = maximum_sends < total_sends
    escalate = (
        headroom
        and myth_game["estimate"] >= 1.5
        and int(myth_game["positive_pairs"]) == 3
    )
    decision = pd.DataFrame(
        [
            {
                "headroom": headroom,
                "maximum_sends": maximum_sends,
                "total_sends": total_sends,
                "myth_game_minus_game": myth_game["estimate"],
                "positive_pairs": int(myth_game["positive_pairs"]),
                "task_order_expansion": bool(escalate),
                "decision": (
                    "expand task-order comparison"
                    if escalate
                    else "do not expand task-order comparison"
                ),
            }
        ]
    )
    estimated_cost = (
        usage["input_tokens"] * .75
        + (usage["output_tokens"] + usage["reasoning_tokens"]) * 3.75
    ) / 1_000_000
    cost = pd.DataFrame([{**usage, "estimated_list_price_usd": estimated_cost}])

    dataframe.to_csv(args.out / "run_metrics.csv", index=False)
    trajectory_dataframe.to_csv(args.out / "round_metrics.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(args.out / "audit.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    contrast_frame.to_csv(args.out / "paired_contrasts.csv", index=False)
    decision.to_csv(args.out / "screen_decision.csv", index=False)
    cost.to_csv(args.out / "token_cost.csv", index=False)
    plot_results(dataframe, trajectory_dataframe, args.out)

    print(summary.to_string(index=False))
    print("\nPaired contrasts:\n", contrast_frame.to_string(index=False))
    print("\nFrozen decision:\n", decision.to_string(index=False))
    print(f"\nEstimated list-price cost: ${estimated_cost:.4f}")


if __name__ == "__main__":
    main()
