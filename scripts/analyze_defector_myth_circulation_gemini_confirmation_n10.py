#!/usr/bin/env python3
"""Analyze the independent Gemini threat-language transmission confirmation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyses._shared import configure_matplotlib
import analyze_defector_myth_circulation_gemini_n5 as screen


DEFAULT_INPUT = Path(
    "data/json/noise_experiments/defector_myth_circulation_confirm_20260821"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/defector_myth_circulation_gemini_confirmation_n10_20260821"
)
EXPERIMENT_DIR = "noise8i_defector_myth_circulation_gemini_confirm_n10"
EXPECTED_IDS = set(range(50, 60))


def plot_confirmatory_effect(contrasts, output_dir):
    import matplotlib.pyplot as plt

    row = contrasts[
        contrasts["metric"] == "direct_target_threat_density"
    ].iloc[0]
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    ax.errorbar(
        0,
        row["estimate"],
        yerr=[
            [row["estimate"] - row["ci_low"]],
            [row["ci_high"] - row["estimate"]],
        ],
        fmt="o",
        markersize=11,
        capsize=8,
        linewidth=2.8,
        color="#2a9d8f",
    )
    ax.axhline(0, color="#263238", linestyle="--", linewidth=1.2)
    ax.set_xticks([0], ["Ordinary substitute − defector myth"])
    ax.set_ylabel("Difference in threat/defection matches\nper 100 target-myth words")
    ax.set_title("Independent one-step transmission confirmation", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.28)
    fig.tight_layout()
    fig.savefig(output_dir / "confirmatory_threat_effect.png", dpi=300)
    plt.close(fig)


def plot_paired_threat(runs, output_dir):
    import matplotlib.pyplot as plt

    pivot = runs.pivot(
        index="replicate_id",
        columns="policy",
        values="direct_target_threat_density",
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    x = np.arange(2)
    for replicate_id, row in pivot.iterrows():
        ax.plot(
            x,
            [row["normal"], row["standard_substitute"]],
            marker="o",
            alpha=0.68,
            linewidth=1.6,
            label=str(replicate_id),
        )
    ax.set_xticks(x, ["Defector myth", "Ordinary substitute"])
    ax.set_ylabel("Threat/defection matches per 100 target-myth words")
    ax.set_title("Ten independent matched population pairs", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.28)
    fig.tight_layout()
    fig.savefig(output_dir / "paired_target_threat.png", dpi=300)
    plt.close(fig)


def plot_threat_trajectory(exposures, output_dir):
    import matplotlib.pyplot as plt

    by_population_round = (
        exposures.groupby(["policy", "replicate_id", "round"], as_index=False)[
            "target_threat_density"
        ]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for policy in screen.POLICY_ORDER:
        subset = by_population_round[by_population_round["policy"] == policy]
        grouped = subset.groupby("round")["target_threat_density"].agg(
            ["mean", "sem", "count"]
        )
        rounds = grouped.index.to_numpy(dtype=float)
        means = grouped["mean"].to_numpy(dtype=float)
        errors = []
        for _, row in grouped.iterrows():
            if row["count"] <= 1 or not np.isfinite(row["sem"]):
                errors.append(0.0)
            else:
                errors.append(
                    row["sem"] * stats.t.ppf(0.975, int(row["count"]) - 1)
                )
        errors = np.asarray(errors, dtype=float)
        ax.plot(
            rounds,
            means,
            marker="o",
            linewidth=2.4,
            color=screen.COLORS[policy],
            label=screen.POLICY_LABELS[policy],
        )
        ax.fill_between(
            rounds,
            means - errors,
            means + errors,
            color=screen.COLORS[policy],
            alpha=0.14,
        )
    ax.set_xlabel("Round of target response")
    ax.set_ylabel("Threat/defection matches per 100 target-myth words")
    ax.set_xticks(range(2, 11))
    ax.set_title("Threat language after direct cultural exposure", fontweight="bold")
    ax.legend(title="Prior myth shown")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "target_threat_trajectory.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    import pandas as pd

    run_rows, exposure_rows, round_rows, myth_rows, usage_rows = screen.load_data(
        args.input,
        experiment_dir=EXPERIMENT_DIR,
        expected_ids=EXPECTED_IDS,
    )
    runs = pd.DataFrame(run_rows).sort_values(["policy", "replicate_id"])
    exposures = pd.DataFrame(exposure_rows).sort_values(
        ["policy", "replicate_id", "round", "agent_id"]
    )
    rounds = pd.DataFrame(round_rows).sort_values(
        ["policy", "replicate_id", "round"]
    )
    myths = pd.DataFrame(myth_rows).sort_values(
        ["policy", "replicate_id", "round", "agent_id"]
    )
    token_usage = pd.DataFrame(usage_rows).sort_values(["policy", "replicate_id"])
    contrasts = pd.DataFrame(screen.make_contrasts(runs))
    # The exploratory screen had two different primary outcomes.  This
    # independent run was frozen with threat density as its sole primary, so
    # replace those inherited labels before saving confirmation outputs.
    contrasts["primary"] = (
        contrasts["metric"] == "direct_target_threat_density"
    )
    contrasts["holm_p_value"] = np.nan
    summaries = pd.DataFrame(screen.make_summaries(runs))
    term_counts = pd.DataFrame(screen.make_term_counts(exposures))

    primary = contrasts[
        contrasts["metric"] == "direct_target_threat_density"
    ].copy()
    primary["confirmed"] = (
        (primary["estimate"] < 0)
        & (primary["ci_high"] < 0)
        & (primary["p_value"] < 0.05)
    )

    runs.to_csv(args.out / "run_metrics.csv", index=False)
    exposures.to_csv(args.out / "direct_exposure_metrics.csv", index=False)
    rounds.to_csv(args.out / "round_metrics.csv", index=False)
    myths.to_csv(args.out / "myth_metrics.csv", index=False)
    token_usage.to_csv(args.out / "token_usage.csv", index=False)
    contrasts.to_csv(args.out / "paired_contrasts.csv", index=False)
    primary.to_csv(args.out / "confirmatory_test.csv", index=False)
    summaries.to_csv(args.out / "summary.csv", index=False)
    term_counts.to_csv(args.out / "lexical_term_counts.csv", index=False)

    plot_confirmatory_effect(contrasts, args.out)
    plot_paired_threat(runs, args.out)
    plot_threat_trajectory(exposures, args.out)
    screen.plot_lexical_transmission(contrasts, args.out)
    screen.plot_manipulation_check(runs, args.out)

    print("Confirmatory test:")
    print(primary.to_string(index=False))
    print("\nAll frozen and secondary contrasts:")
    print(contrasts.to_string(index=False))
    print("\nRun-level metrics:")
    print(runs.to_string(index=False))
    print("\nToken/attempt totals:")
    print(
        token_usage.groupby("policy")[[
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
