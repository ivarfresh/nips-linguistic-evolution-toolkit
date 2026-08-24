#!/usr/bin/env python3
"""Audit and plot corrected no-defector population-size results by model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyses._shared import configure_matplotlib
from scripts.analyze_corrected_v2_confirmatory import (
    CONDITION_LABELS,
    CONDITION_ORDER,
    load_runs,
    run_metrics,
)
from scripts.audit_v2_protocol import audit_paired_schedules, audit_run


DEFAULT_NEW_INPUT = Path(
    "data/json/noise_experiments/crossmodel_population_size_dyads_20260824"
)
DEFAULT_OUTPUT = Path("docs/figures/crossmodel_population_size_20260824")
CLAUDE_METRICS = Path(
    "docs/figures/corrected_v2_confirmatory_20260812/run_metrics.csv"
)
GPT8_METRICS = Path(
    "docs/figures/crossmodel_signed_gpt_n5_20260821/run_metrics.csv"
)
GEMINI8_METRICS = Path(
    "docs/figures/gemini37_flash_task_order_n3_20260823/run_metrics.csv"
)
FROZEN_PROTOCOL_COMMIT = "e5e3a7482479530125ace6d17ae8be6cfde105c6"

MODEL_SPECS = {
    "gpt": {
        "label": "GPT-5 Nano",
        "model": "openai/gpt-5-nano",
        "provider": "openai",
        "replicates": set(range(5)),
        "cells": {
            "game": "noise2_crossmodel_signed_gpt_n5_game",
            "game_myth": "noise2_crossmodel_signed_gpt_n5_game_myth",
            "myth_game": "noise2_crossmodel_signed_gpt_n5_myth_game",
        },
    },
    "gemini": {
        "label": "Gemini 3.7 Flash",
        "model": "google/gemini-3.7-flash",
        "provider": "google",
        "replicates": {90, 91, 92},
        "cells": {
            "game": "noise2_crossmodel_gemini37_flash_n3_game",
            "game_myth": "noise2_crossmodel_gemini37_flash_n3_game_myth",
            "myth_game": "noise2_crossmodel_gemini37_flash_n3_myth_game",
        },
    },
}

POPULATION_LABELS = {
    2: "2-agent repeated dyad",
    8: "8-agent rotating population",
}
POPULATION_COLORS = {
    2: "#66c2a5",
    8: "#fc8d62",
}


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.allclose(values, values[0]):
        return float(values.mean()), float(values.mean())
    sem = stats.sem(values)
    return stats.t.interval(0.95, len(values) - 1, loc=values.mean(), scale=sem)


def load_new_dyads(input_dir: Path):
    rows = []
    audit_rows = []
    config_hashes = set()
    code_commits = set()

    for family, spec in MODEL_SPECS.items():
        family_audits = []
        for condition, experiment in spec["cells"].items():
            runs = load_runs(input_dir / experiment)
            if len(runs) != len(spec["replicates"]):
                raise RuntimeError(
                    f"{experiment}: expected {len(spec['replicates'])} runs; "
                    f"found {len(runs)}"
                )
            observed = set()
            for path, run in runs:
                metadata = run.get("run_metadata") or {}
                replicate_id = int(metadata.get("replicate_id"))
                observed.add(replicate_id)
                expected = {
                    "model": spec["model"],
                    "llm_provider": spec["provider"],
                    "num_agents": 2,
                    "defector_count": 0,
                    "punishment_enabled": False,
                    "code_dirty": False,
                }
                for key, value in expected.items():
                    actual = metadata.get(key, False if key == "punishment_enabled" else None)
                    if actual != value:
                        raise RuntimeError(
                            f"{path}: expected {key}={value!r}; got {actual!r}"
                        )
                if family == "gemini":
                    gemini_expected = {
                        "provider_model": "gemini-3.7-flash",
                        "thinking_level": "medium",
                        "thinking_level_source": "GEMINI_THINKING_LEVEL",
                        "temperature_sent": False,
                    }
                    for key, value in gemini_expected.items():
                        if metadata.get(key) != value:
                            raise RuntimeError(
                                f"{path}: expected {key}={value!r}; "
                                f"got {metadata.get(key)!r}"
                            )
                config_hash = metadata.get("config_sha256")
                if not config_hash:
                    raise RuntimeError(f"{path}: missing config hash")
                config_hashes.add(config_hash)
                code_commit = metadata.get("code_commit")
                if not code_commit:
                    raise RuntimeError(f"{path}: missing code commit")
                code_commits.add(code_commit)

                audited = audit_run(path)
                family_audits.append(audited)
                expected_calls = 20 if condition == "game" else 40
                if audited["calls"] != expected_calls:
                    raise RuntimeError(
                        f"{path}: expected {expected_calls} accepted calls; "
                        f"found {audited['calls']}"
                    )
                if audited["noise_checks"] != 20:
                    raise RuntimeError(
                        f"{path}: expected 20 noise checks; "
                        f"found {audited['noise_checks']}"
                    )
                audit_rows.append(
                    {
                        "model": spec["label"],
                        "condition": condition,
                        "replicate_id": replicate_id,
                        "path": str(path),
                        "issues": len(audited["issues"]),
                        "accepted_calls": audited["calls"],
                        "attempts": audited["attempts"],
                        "retries": audited["retry_attempts"],
                        "noise_checks": audited["noise_checks"],
                    }
                )
                metrics, _ = run_metrics(path, run)
                rows.append(
                    {
                        "model": spec["label"],
                        "model_family": family,
                        "population": POPULATION_LABELS[2],
                        "num_agents": 2,
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "replicate_id": replicate_id,
                        "path": str(path),
                        **metrics,
                    }
                )
            if observed != spec["replicates"]:
                raise RuntimeError(
                    f"{experiment}: replicate IDs {sorted(observed)}"
                )
        audit_paired_schedules(family_audits)
        issues = [issue for result in family_audits for issue in result["issues"]]
        if issues:
            raise RuntimeError(
                f"{spec['label']} audit failed:\n" + "\n".join(issues)
            )

    if len(config_hashes) != 1:
        raise RuntimeError(f"Expected one frozen config hash; got {config_hashes}")
    return (
        pd.DataFrame(rows),
        pd.DataFrame(audit_rows),
        config_hashes.pop(),
        sorted(code_commits),
    )


def load_existing_runs():
    claude = pd.read_csv(CLAUDE_METRICS)
    claude = claude.assign(
        model="Claude Sonnet 4.5",
        model_family="claude",
        population=claude["num_agents"].map(POPULATION_LABELS),
    )

    gpt = pd.read_csv(GPT8_METRICS).assign(
        model="GPT-5 Nano",
        model_family="gpt",
        population=POPULATION_LABELS[8],
    )
    gemini = pd.read_csv(GEMINI8_METRICS).assign(
        model="Gemini 3.7 Flash",
        model_family="gemini",
        population=POPULATION_LABELS[8],
    )
    keep = [
        "model",
        "model_family",
        "population",
        "num_agents",
        "condition",
        "condition_label",
        "replicate_id",
        "path",
        "final_balance",
        "joint_balance",
        "mean_trust_ratio",
        "mean_return_ratio",
        "mean_sent",
        "mean_returned",
        "mean_later_communicated_sent",
        "num_interactions",
    ]
    return pd.concat([claude[keep], gpt[keep], gemini[keep]], ignore_index=True)


def summarize(dataframe):
    rows = []
    for model in ["Claude Sonnet 4.5", "GPT-5 Nano", "Gemini 3.7 Flash"]:
        for num_agents in [2, 8]:
            for condition in CONDITION_ORDER:
                cell = dataframe[
                    (dataframe["model"] == model)
                    & (dataframe["num_agents"] == num_agents)
                    & (dataframe["condition"] == condition)
                ]
                values = cell["final_balance"].to_numpy(dtype=float)
                if not len(values):
                    raise RuntimeError(
                        f"Missing cell: {model}, {num_agents} agents, {condition}"
                    )
                low, high = confidence_interval(values)
                rows.append(
                    {
                        "model": model,
                        "population": POPULATION_LABELS[num_agents],
                        "num_agents": num_agents,
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "n": len(values),
                        "mean": values.mean(),
                        "sd": values.std(ddof=1) if len(values) > 1 else 0.0,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return pd.DataFrame(rows)


def plot_final_balance(run_dataframe, summary_dataframe, output_dir: Path):
    import matplotlib.pyplot as plt

    models = ["Claude Sonnet 4.5", "GPT-5 Nano", "Gemini 3.7 Flash"]
    x = np.arange(len(CONDITION_ORDER), dtype=float)
    offsets = {2: -0.10, 8: 0.10}
    rng = np.random.default_rng(20260824)

    fig, axes = plt.subplots(1, 3, figsize=(11, 7), sharey=True)
    fig.suptitle(
        "Average final balance by model and population regime",
        fontsize=15,
        fontweight="bold",
    )
    for ax, model in zip(axes, models):
        for num_agents in [2, 8]:
            color = POPULATION_COLORS[num_agents]
            pop_label = POPULATION_LABELS[num_agents]
            cell_summary = summary_dataframe[
                (summary_dataframe["model"] == model)
                & (summary_dataframe["num_agents"] == num_agents)
            ].set_index("condition").loc[CONDITION_ORDER]
            mean = cell_summary["mean"].to_numpy(dtype=float)
            low = cell_summary["ci_low"].to_numpy(dtype=float)
            high = cell_summary["ci_high"].to_numpy(dtype=float)
            ax.errorbar(
                x + offsets[num_agents],
                mean,
                yerr=np.vstack([mean - low, high - mean]),
                color=color,
                marker="o" if num_agents == 2 else "s",
                markersize=7,
                linewidth=2.2,
                capsize=5,
                label=(
                    f"{pop_label} "
                    f"(n={int(cell_summary['n'].iloc[0])}/cell)"
                ),
                zorder=3,
            )
            raw = run_dataframe[
                (run_dataframe["model"] == model)
                & (run_dataframe["num_agents"] == num_agents)
            ]
            for condition_index, condition in enumerate(CONDITION_ORDER):
                values = raw[raw["condition"] == condition]["final_balance"].to_numpy()
                jitter = rng.uniform(-0.035, 0.035, size=len(values))
                ax.scatter(
                    condition_index + offsets[num_agents] + jitter,
                    values,
                    s=18,
                    color=color,
                    alpha=0.30,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=2,
                )
        ax.set_title(model, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [CONDITION_LABELS[condition] for condition in CONDITION_ORDER],
            rotation=15,
        )
        ax.set_xlabel("Task order")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(45, 78)
    axes[0].set_ylabel("Final cumulative balance per agent")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.text(
        0.5,
        0.015,
        "Corrected informed signed noise; no defectors or punishment. "
        "Points are run means; bars are 95% t intervals.",
        ha="center",
        fontsize=9,
        color="#37474f",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    fig.patch.set_edgecolor("#111111")
    fig.patch.set_linewidth(1.4)
    fig.savefig(
        output_dir / "final_balance_by_model_and_population.png",
        dpi=300,
        facecolor="white",
        edgecolor="#111111",
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_NEW_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    configure_matplotlib()
    args.out.mkdir(parents=True, exist_ok=True)
    dyads, audits, config_hash, code_commits = load_new_dyads(args.input)
    all_runs = pd.concat([load_existing_runs(), dyads], ignore_index=True)
    all_runs = all_runs.sort_values(
        ["model", "num_agents", "condition", "replicate_id"]
    )
    summary = summarize(all_runs)
    all_runs.to_csv(args.out / "run_metrics.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    audits.to_csv(args.out / "new_dyad_audit.csv", index=False)
    (args.out / "provenance.json").write_text(
        json.dumps(
            {
                "frozen_protocol_commit": FROZEN_PROTOCOL_COMMIT,
                "accepted_run_commits": code_commits,
                "new_dyad_config_sha256": config_hash,
                "new_population_count": int(len(dyads)),
                "new_accepted_calls": int(audits["accepted_calls"].sum()),
                "new_noise_checks": int(audits["noise_checks"].sum()),
                "new_retry_attempts": int(audits["retries"].sum()),
                "new_audit_issues": int(audits["issues"].sum()),
            },
            indent=2,
        )
        + "\n"
    )
    plot_final_balance(all_runs, summary, args.out)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
