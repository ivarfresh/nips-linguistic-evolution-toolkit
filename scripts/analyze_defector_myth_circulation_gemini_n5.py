#!/usr/bin/env python3
"""Analyze the frozen Gemini defector-myth circulation mechanism screen."""

from __future__ import annotations

import argparse
from collections import Counter
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyses._shared import configure_matplotlib
from analyze_corrected_v2_confirmatory import load_runs
from analyze_defector_myth_game_crossmodel_n5 import (
    COOP_PATTERN,
    THREAT_PATTERN,
    myth_metrics,
    usage,
)


DEFAULT_INPUT = Path(
    "data/json/noise_experiments/defector_myth_circulation_20260821"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/defector_myth_circulation_gemini_n5_20260821"
)
EXPERIMENT_DIR = "noise8i_defector_myth_circulation_gemini_n5"
MODEL = "google/gemini-3.1-flash-lite"
EXPECTED_IDS = set(range(45, 50))
POLICY_ORDER = ["normal", "standard_substitute"]
POLICY_LABELS = {
    "normal": "Defector myth",
    "standard_substitute": "Ordinary substitute",
}
COLORS = {
    "normal": "#9c2f1f",
    "standard_substitute": "#2a9d8f",
}


def confidence_interval(values):
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


def paired_result(values, metric, label, primary=False):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
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
        "contrast": "ordinary substitute − defector myth",
        "label": label,
        "primary": primary,
        "n_pairs": len(values),
        "estimate": values.mean() if len(values) else math.nan,
        "ci_low": low,
        "ci_high": high,
        "p_value": p_value,
        "cohens_dz": 0.0 if exact_zero else values.mean() / sd if sd else math.nan,
    }


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    finite = np.where(np.isfinite(p_values))[0]
    if not len(finite):
        return adjusted
    order = finite[np.argsort(p_values[finite])]
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def dyad_for_agent(entry, agent_id):
    for dyad in entry.get("dyads") or []:
        if agent_id in (dyad.get("agents") or []):
            return dyad
    return None


def load_data(input_dir, experiment_dir=EXPERIMENT_DIR, expected_ids=None):
    expected_ids = expected_ids or EXPECTED_IDS
    runs = load_runs(input_dir / experiment_dir)
    expected_run_count = 2 * len(expected_ids)
    if len(runs) != expected_run_count:
        raise RuntimeError(
            f"Found {len(runs)} runs; expected {expected_run_count}"
        )

    run_rows = []
    exposure_rows = []
    standard_round_rows = []
    myth_rows = []
    usage_rows = []
    seen = {policy: set() for policy in POLICY_ORDER}

    for path, run in runs:
        metadata = run.get("run_metadata") or {}
        game_data = run.get("game_data") or {}
        if metadata.get("model") != MODEL:
            raise RuntimeError(f"Unexpected model in {path}: {metadata.get('model')}")
        if metadata.get("code_dirty"):
            raise RuntimeError(f"Dirty execution provenance in {path}")
        replicate_id = int(metadata["replicate_id"])
        policy = metadata.get("defector_myth_policy")
        if policy not in POLICY_ORDER:
            raise RuntimeError(f"Unexpected circulation policy in {path}: {policy}")
        seen[policy].add(replicate_id)

        defector_ids = set(
            metadata.get("defector_agent_ids")
            or game_data.get("defector_agent_ids")
            or []
        )
        standard_ids = set(run.get("agents") or {}) - defector_ids
        all_standard_sends = []
        all_standard_myth_metrics = []
        direct_rows = []
        history = run.get("conversation_history") or []

        for entry in history:
            round_number = int(entry["round"])
            round_sends = []
            for dyad in entry.get("dyads") or []:
                investor = dyad.get("investor")
                if investor in standard_ids:
                    send_ratio = float(dyad.get("sent") or 0) / 5.0
                    all_standard_sends.append(send_ratio)
                    round_sends.append(send_ratio)
            standard_round_rows.append(
                {
                    "policy": policy,
                    "policy_label": POLICY_LABELS[policy],
                    "replicate_id": replicate_id,
                    "round": round_number,
                    "standard_send_ratio": np.mean(round_sends),
                }
            )

            for agent_id, text in (entry.get("myths") or {}).items():
                author_type = "defector" if agent_id in defector_ids else "standard"
                metrics = myth_metrics(text)
                myth_rows.append(
                    {
                        "policy": policy,
                        "policy_label": POLICY_LABELS[policy],
                        "replicate_id": replicate_id,
                        "round": round_number,
                        "agent_id": agent_id,
                        "author_type": author_type,
                        "text": text,
                        **metrics,
                    }
                )
                if author_type == "standard":
                    all_standard_myth_metrics.append(metrics)

            if round_number == 1:
                continue
            previous = history[round_number - 2]
            previous_myths = previous.get("myths") or {}
            exposures = entry.get("myth_exposures") or {}
            for agent_id in sorted(standard_ids):
                exposure = exposures.get(agent_id) or {}
                if exposure.get("original_author_type") != "defector":
                    continue
                target_text = (entry.get("myths") or {}).get(agent_id, "")
                target_metrics = myth_metrics(target_text)
                presented_text = previous_myths.get(
                    exposure.get("presented_author_id"), ""
                )
                presented_metrics = myth_metrics(presented_text)
                dyad = dyad_for_agent(entry, agent_id) or {}
                role = (dyad.get("roles") or {}).get(agent_id)
                send_ratio = math.nan
                return_ratio = math.nan
                if role == "investor":
                    send_ratio = float(dyad.get("sent") or 0) / 5.0
                elif role == "trustee":
                    received = float(dyad.get("received") or 0)
                    if received > 0:
                        return_ratio = float(dyad.get("returned") or 0) / received
                row = {
                    "policy": policy,
                    "policy_label": POLICY_LABELS[policy],
                    "replicate_id": replicate_id,
                    "round": round_number,
                    "agent_id": agent_id,
                    "current_role": role,
                    "original_author_id": exposure.get("original_author_id"),
                    "presented_author_id": exposure.get("presented_author_id"),
                    "substitution_applied": bool(
                        exposure.get("substitution_applied")
                    ),
                    "presented_myth": presented_text,
                    "target_myth": target_text,
                    "target_coop_density": target_metrics["coop_density"],
                    "target_threat_density": target_metrics["threat_density"],
                    "target_half_rule": target_metrics["half_rule"],
                    "presented_coop_density": presented_metrics["coop_density"],
                    "presented_threat_density": presented_metrics["threat_density"],
                    "current_send_ratio": send_ratio,
                    "current_return_ratio": return_ratio,
                }
                exposure_rows.append(row)
                direct_rows.append(row)

        defector_post = [
            row
            for row in myth_rows
            if row["policy"] == policy
            and row["replicate_id"] == replicate_id
            and row["author_type"] == "defector"
            and row["round"] >= 2
        ]
        run_rows.append(
            {
                "policy": policy,
                "policy_label": POLICY_LABELS[policy],
                "replicate_id": replicate_id,
                "defector_ids": ",".join(sorted(defector_ids)),
                "n_direct_exposures": len(direct_rows),
                "n_direct_sender_exposures": sum(
                    np.isfinite(row["current_send_ratio"]) for row in direct_rows
                ),
                "n_substitutions": sum(
                    row["substitution_applied"] for row in direct_rows
                ),
                "direct_target_coop_density": np.mean(
                    [row["target_coop_density"] for row in direct_rows]
                ),
                "direct_target_threat_density": np.mean(
                    [row["target_threat_density"] for row in direct_rows]
                ),
                "direct_target_half_rule": np.mean(
                    [row["target_half_rule"] for row in direct_rows]
                ),
                "direct_sender_send_ratio": np.nanmean(
                    [row["current_send_ratio"] for row in direct_rows]
                ),
                "direct_receiver_return_ratio": np.nanmean(
                    [row["current_return_ratio"] for row in direct_rows]
                ),
                "presented_coop_density": np.mean(
                    [row["presented_coop_density"] for row in direct_rows]
                ),
                "presented_threat_density": np.mean(
                    [row["presented_threat_density"] for row in direct_rows]
                ),
                "population_standard_send_ratio": np.mean(all_standard_sends),
                "population_standard_myth_coop_density": np.mean(
                    [row["coop_density"] for row in all_standard_myth_metrics]
                ),
                "population_standard_myth_threat_density": np.mean(
                    [row["threat_density"] for row in all_standard_myth_metrics]
                ),
                "defector_post_coop_density": np.mean(
                    [row["coop_density"] for row in defector_post]
                ),
                "defector_post_threat_density": np.mean(
                    [row["threat_density"] for row in defector_post]
                ),
            }
        )
        totals, attempts, retries, forced = usage(run)
        usage_rows.append(
            {
                "policy": policy,
                "replicate_id": replicate_id,
                **totals,
                "attempts": attempts,
                "recovered_retries": retries,
                "forced_responses": forced,
            }
        )

    for policy in POLICY_ORDER:
        if seen[policy] != expected_ids:
            raise RuntimeError(f"{policy} has replicate IDs {sorted(seen[policy])}")

    return run_rows, exposure_rows, standard_round_rows, myth_rows, usage_rows


def make_contrasts(runs):
    metrics = [
        ("direct_target_coop_density", "Directly exposed myth: cooperation", True),
        ("direct_sender_send_ratio", "Directly exposed sender: proportion sent", True),
        ("direct_target_threat_density", "Directly exposed myth: threat", False),
        ("direct_target_half_rule", "Directly exposed myth: half rule", False),
        ("direct_receiver_return_ratio", "Directly exposed receiver: return ratio", False),
        ("presented_coop_density", "Presented myth: cooperation", False),
        ("presented_threat_density", "Presented myth: threat", False),
        ("population_standard_send_ratio", "All ordinary senders", False),
        (
            "population_standard_myth_coop_density",
            "All ordinary-authored myths: cooperation",
            False,
        ),
        (
            "population_standard_myth_threat_density",
            "All ordinary-authored myths: threat",
            False,
        ),
        ("defector_post_coop_density", "Defector myths rounds 2–10: cooperation", False),
        ("defector_post_threat_density", "Defector myths rounds 2–10: threat", False),
    ]
    records = []
    for metric, label, primary in metrics:
        pivot = runs.pivot(index="replicate_id", columns="policy", values=metric)
        records.append(
            paired_result(
                pivot["standard_substitute"] - pivot["normal"],
                metric,
                label,
                primary,
            )
        )
    primary_indices = [index for index, row in enumerate(records) if row["primary"]]
    adjusted = holm_adjust([records[index]["p_value"] for index in primary_indices])
    for index, p_value in zip(primary_indices, adjusted):
        records[index]["holm_p_value"] = p_value
    for index, row in enumerate(records):
        if index not in primary_indices:
            row["holm_p_value"] = math.nan
    return records


def make_summaries(runs):
    metrics = [
        "n_direct_exposures",
        "n_direct_sender_exposures",
        "n_substitutions",
        "direct_target_coop_density",
        "direct_target_threat_density",
        "direct_target_half_rule",
        "direct_sender_send_ratio",
        "direct_receiver_return_ratio",
        "presented_coop_density",
        "presented_threat_density",
        "population_standard_send_ratio",
        "population_standard_myth_coop_density",
        "population_standard_myth_threat_density",
        "defector_post_coop_density",
        "defector_post_threat_density",
    ]
    records = []
    for policy in POLICY_ORDER:
        subset = runs[runs["policy"] == policy]
        for metric in metrics:
            values = subset[metric].to_numpy(dtype=float)
            low, high = confidence_interval(values)
            records.append(
                {
                    "policy": policy,
                    "policy_label": POLICY_LABELS[policy],
                    "metric": metric,
                    "n": np.isfinite(values).sum(),
                    "mean": np.nanmean(values),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return records


def make_term_counts(exposures):
    records = []
    for policy in POLICY_ORDER:
        subset = exposures[exposures["policy"] == policy]
        for scope, column in (
            ("presented", "presented_myth"),
            ("target_response", "target_myth"),
        ):
            for lexicon, pattern in (
                ("cooperation_fairness", COOP_PATTERN),
                ("defection_threat", THREAT_PATTERN),
            ):
                counts = Counter()
                for text in subset[column]:
                    counts.update(match.lower() for match in pattern.findall(text or ""))
                for term, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                    records.append(
                        {
                            "policy": policy,
                            "policy_label": POLICY_LABELS[policy],
                            "scope": scope,
                            "lexicon": lexicon,
                            "term": term,
                            "count": count,
                        }
                    )
    return records


def plot_primary_effects(contrasts, output_dir):
    import matplotlib.pyplot as plt

    specs = [
        ("direct_target_coop_density", "Target myth\ncooperation/fairness"),
        ("direct_sender_send_ratio", "Target's same-round\nproportion sent"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, (metric, title) in zip(axes, specs):
        row = contrasts[contrasts["metric"] == metric].iloc[0]
        ax.errorbar(
            0,
            row["estimate"],
            yerr=[
                [row["estimate"] - row["ci_low"]],
                [row["ci_high"] - row["estimate"]],
            ],
            fmt="o",
            markersize=10,
            capsize=7,
            linewidth=2.7,
            color="#2a9d8f",
        )
        ax.axhline(0, color="#263238", linestyle="--", linewidth=1.2)
        ax.set_xticks([0], ["Ordinary substitute −\ndefector myth"])
        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.28)
    axes[0].set_ylabel("Difference in matches per 100 myth words")
    axes[1].set_ylabel("Difference in proportion sent")
    fig.suptitle("Immediate effects of replacing a defector-authored myth", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "primary_effects.png", dpi=300)
    plt.close(fig)


def plot_paired_outcomes(runs, output_dir):
    import matplotlib.pyplot as plt

    specs = [
        ("direct_target_coop_density", "Directly exposed myth: cooperation/fairness"),
        ("direct_sender_send_ratio", "Directly exposed sender: proportion sent"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    x = np.arange(2)
    for ax, (metric, title) in zip(axes, specs):
        pivot = runs.pivot(index="replicate_id", columns="policy", values=metric)
        for replicate_id, row in pivot.iterrows():
            values = [row["normal"], row["standard_substitute"]]
            ax.plot(x, values, marker="o", alpha=0.65, linewidth=1.5, label=replicate_id)
        ax.set_xticks(x, ["Defector myth", "Ordinary substitute"])
        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.28)
    axes[0].set_ylabel("Matches per 100 myth words")
    axes[1].set_ylabel("Proportion sent")
    fig.suptitle("Five matched population pairs", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "paired_direct_outcomes.png", dpi=300)
    plt.close(fig)


def plot_direct_myth_trajectory(exposures, output_dir):
    import matplotlib.pyplot as plt

    by_population_round = (
        exposures.groupby(["policy", "replicate_id", "round"], as_index=False)[
            "target_coop_density"
        ]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for policy in POLICY_ORDER:
        subset = by_population_round[by_population_round["policy"] == policy]
        grouped = subset.groupby("round")["target_coop_density"].agg(["mean", "sem"])
        rounds = grouped.index.to_numpy(dtype=float)
        means = grouped["mean"].to_numpy(dtype=float)
        errors = grouped["sem"].fillna(0).to_numpy(dtype=float) * stats.t.ppf(0.975, 4)
        ax.plot(
            rounds,
            means,
            marker="o",
            linewidth=2.4,
            color=COLORS[policy],
            label=POLICY_LABELS[policy],
        )
        ax.fill_between(
            rounds,
            means - errors,
            means + errors,
            color=COLORS[policy],
            alpha=0.14,
        )
    ax.set_xlabel("Round of target response")
    ax.set_ylabel("Cooperation/fairness matches per 100 myth words")
    ax.set_xticks(range(2, 11))
    ax.set_title("Myths written immediately after meeting a defector", fontweight="bold")
    ax.legend(title="Prior myth shown")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "direct_target_myth_trajectory.png", dpi=300)
    plt.close(fig)


def plot_manipulation_check(runs, output_dir):
    import matplotlib.pyplot as plt

    specs = [
        ("presented_coop_density", "Cooperation/fairness"),
        ("presented_threat_density", "Defection/threat"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, (metric, title) in zip(axes, specs):
        for position, policy in enumerate(POLICY_ORDER):
            values = runs[runs["policy"] == policy][metric].to_numpy(dtype=float)
            low, high = confidence_interval(values)
            mean = np.mean(values)
            ax.errorbar(
                position,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="o",
                markersize=9,
                capsize=6,
                linewidth=2.4,
                color=COLORS[policy],
            )
        ax.set_xticks(range(2), [POLICY_LABELS[p] for p in POLICY_ORDER])
        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.28)
    axes[0].set_ylabel("Matches in presented text per 100 words")
    fig.suptitle("Manipulation check: myths presented at direct exposures", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "presented_myth_manipulation_check.png", dpi=300)
    plt.close(fig)


def plot_lexical_transmission(contrasts, output_dir):
    import matplotlib.pyplot as plt

    specs = [
        (
            ["presented_coop_density", "direct_target_coop_density"],
            "Cooperation/fairness",
        ),
        (
            ["presented_threat_density", "direct_target_threat_density"],
            "Defection/threat",
        ),
    ]
    labels = ["Presented source text", "Target's next myth"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3))
    for ax, (metrics, title) in zip(axes, specs):
        selected = contrasts.set_index("metric").loc[metrics]
        for position, (_, row) in enumerate(selected.iterrows()):
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
                color=["#457b9d", "#2a9d8f"][position],
            )
        ax.axhline(0, color="#263238", linestyle="--", linewidth=1.1)
        ax.set_xticks(range(2), labels, rotation=5)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.28)
    axes[0].set_ylabel(
        "Ordinary substitute − defector myth\n(matches per 100 words)"
    )
    fig.suptitle("The lexical signal after one transmission step", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "lexical_transmission_effects.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    import pandas as pd

    run_rows, exposure_rows, round_rows, myth_rows, usage_rows = load_data(args.input)
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
    contrasts = pd.DataFrame(make_contrasts(runs))
    summaries = pd.DataFrame(make_summaries(runs))
    term_counts = pd.DataFrame(make_term_counts(exposures))

    runs.to_csv(args.out / "run_metrics.csv", index=False)
    exposures.to_csv(args.out / "direct_exposure_metrics.csv", index=False)
    rounds.to_csv(args.out / "round_metrics.csv", index=False)
    myths.to_csv(args.out / "myth_metrics.csv", index=False)
    token_usage.to_csv(args.out / "token_usage.csv", index=False)
    contrasts.to_csv(args.out / "paired_contrasts.csv", index=False)
    summaries.to_csv(args.out / "summary.csv", index=False)
    term_counts.to_csv(args.out / "lexical_term_counts.csv", index=False)
    plot_primary_effects(contrasts, args.out)
    plot_paired_outcomes(runs, args.out)
    plot_direct_myth_trajectory(exposures, args.out)
    plot_manipulation_check(runs, args.out)
    plot_lexical_transmission(contrasts, args.out)

    print("Run-level metrics:")
    print(runs.to_string(index=False))
    print("\nPaired contrasts:")
    print(contrasts.to_string(index=False))
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
