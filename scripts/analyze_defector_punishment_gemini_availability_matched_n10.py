#!/usr/bin/env python3
"""Analyze the exploratory matched Gemini punishment-availability follow-up."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyses._shared import configure_matplotlib
from analyze_corrected_v2_confirmatory import holm_adjust, load_runs
from analyze_defector_myth_game_crossmodel_n5 import myth_metrics
from analyze_defector_punishment_gpt_n5 import PUNISH_PATTERN, usage_metrics
from audit_v2_protocol import audit_paired_schedules, audit_run


ON_ROOT = Path(
    "data/json/noise_experiments/defector_punishment_gemini_confirmation_n10_20260821"
)
ON_EXPERIMENT = "noise8i_defector_punishment_gemini_confirmation_n10"
OFF_ROOT = Path(
    "data/json/noise_experiments/defector_punishment_gemini_availability_matched_n10_20260822"
)
OFF_EXPERIMENT = (
    "noise8i_defector_punishment_gemini_availability_off_matched_n10"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/defector_punishment_gemini_availability_matched_n10_20260822"
)
EXPECTED_IDS = set(range(70, 80))
MODEL = "google/gemini-3.1-flash-lite"
ARM_ORDER = ["off", "on"]
ARM_LABELS = {"off": "Deduction unavailable", "on": "Deduction available"}
COLORS = {"off": "#457b9d", "on": "#c14953"}


def ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    if len(values) == 1 or np.allclose(values, values[0]):
        return float(values.mean()), float(values.mean())
    low, high = stats.t.interval(
        .95,
        len(values) - 1,
        loc=values.mean(),
        scale=stats.sem(values),
    )
    return float(low), float(high)


def mean(values):
    values = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else math.nan


def paired_record(values, metric, label, primary):
    values = np.asarray(values, dtype=float)
    low, high = ci(values)
    sd = values.std(ddof=1)
    if np.isclose(sd, 0):
        exact_zero = np.isclose(values.mean(), 0)
        p_value = 1.0 if exact_zero else 0.0
        statistic = 0.0 if exact_zero else math.inf
        dz = 0.0 if exact_zero else math.inf
    else:
        test = stats.ttest_1samp(values, 0)
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
        dz = float(values.mean() / sd)
    direction = np.sign(values.mean())
    same_direction = int(np.sum(np.sign(values) == direction))
    return {
        "metric": metric,
        "label": label,
        "contrast": "deduction available − unavailable",
        "primary": primary,
        "n_pairs": len(values),
        "estimate": float(values.mean()),
        "ci_low": low,
        "ci_high": high,
        "t_statistic": statistic,
        "p_value": p_value,
        "cohens_dz": dz,
        "pairs_same_direction": same_direction,
        "pairs_zero": int(np.sum(np.isclose(values, 0))),
    }


def load_arm(root, experiment, arm):
    runs = load_runs(root / experiment)
    if len(runs) != 10:
        raise RuntimeError(f"{arm}: found {len(runs)} runs; expected 10")

    run_rows = []
    round_rows = []
    myth_rows = []
    usage_rows = []
    audits = []
    seen_ids = set()
    commits = set()
    config_hashes = set()

    for path, run in runs:
        audit = audit_run(path)
        audits.append(audit)
        if audit["issues"]:
            raise RuntimeError("\n".join(audit["issues"]))
        metadata = run.get("run_metadata") or {}
        expected = {
            "model": MODEL,
            "llm_provider": "google",
            "provider_model": "gemini-3.1-flash-lite",
            "execution_provenance_version": 1,
            "defector_ratio_actual": .25,
            "defector_action_policy": "forced_zero",
            "defector_myth_policy": "normal",
            "history_policy": "none",
            "self_history_window": 0,
            "coplayer_history_window": 0,
            "population_history_window": 0,
            "punishment_enabled": arm == "on",
            "memory_capacity": 9 if arm == "on" else 6,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(
                    f"{path}: {key}={metadata.get(key)!r}; expected {value!r}"
                )
        if metadata.get("code_dirty") is not False:
            raise RuntimeError(f"{path}: dirty or missing provenance")
        if arm == "on" and metadata.get("punishment_prompt_variant") != "current":
            raise RuntimeError(f"{path}: on arm did not use current prompt")
        replicate_id = int(metadata["replicate_id"])
        seen_ids.add(replicate_id)
        commits.add(metadata.get("code_commit"))
        config_hashes.add(metadata.get("config_sha256"))
        defector_ids = set(metadata.get("defector_agent_ids") or [])
        standard_ids = set(run.get("agents") or {}) - defector_ids
        if len(defector_ids) != 2:
            raise RuntimeError(f"{path}: expected two defectors")

        sends = []
        returns = []
        myths = []
        for entry in run.get("conversation_history") or []:
            round_number = int(entry["round"])
            round_sends = []
            round_returns = []
            for dyad in entry.get("dyads") or []:
                investor = dyad["investor"]
                trustee = dyad["trustee"]
                if investor in standard_ids:
                    value = float(dyad["sent"]) / 5.0
                    sends.append(value)
                    round_sends.append(value)
                received = float(dyad.get("received") or 0)
                if trustee in standard_ids and received > 0:
                    value = float(dyad["returned"]) / received
                    returns.append(value)
                    round_returns.append(value)
            round_rows.append(
                {
                    "arm": arm,
                    "replicate_id": replicate_id,
                    "round": round_number,
                    "standard_send_ratio": mean(round_sends),
                    "standard_return_ratio": mean(round_returns),
                }
            )
            for agent_id, text in (entry.get("myths") or {}).items():
                if agent_id not in standard_ids:
                    continue
                metrics = myth_metrics(text)
                words = re.findall(r"\b[\w'-]+\b", text.lower())
                punishment_matches = len(PUNISH_PATTERN.findall(text))
                row = {
                    "arm": arm,
                    "replicate_id": replicate_id,
                    "round": round_number,
                    "agent_id": agent_id,
                    "text": text,
                    **metrics,
                    "punishment_density": (
                        100 * punishment_matches / len(words) if words else 0
                    ),
                    "punishment_presence": float(punishment_matches > 0),
                }
                myth_rows.append(row)
                myths.append(row)

        if len(sends) != 30:
            raise RuntimeError(f"{path}: {len(sends)} ordinary sends; expected 30")
        run_rows.append(
            {
                "arm": arm,
                "arm_label": ARM_LABELS[arm],
                "replicate_id": replicate_id,
                "defector_ids": ",".join(sorted(defector_ids)),
                "pairing_seed": metadata["pairing_seed"],
                "noise_seed": metadata["noise_seed"],
                "standard_send_ratio": mean(sends),
                "standard_return_ratio": mean(returns),
                "myth_coop_density": mean(
                    [row["coop_density"] for row in myths]
                ),
                "myth_threat_density": mean(
                    [row["threat_density"] for row in myths]
                ),
                "myth_half_rule": mean([row["half_rule"] for row in myths]),
                "myth_punishment_density": mean(
                    [row["punishment_density"] for row in myths]
                ),
                "myth_punishment_presence": mean(
                    [row["punishment_presence"] for row in myths]
                ),
            }
        )
        usage_rows.append(
            {"arm": arm, "replicate_id": replicate_id, **usage_metrics(run)}
        )

    if seen_ids != EXPECTED_IDS:
        raise RuntimeError(f"{arm}: replicate IDs {seen_ids}; expected {EXPECTED_IDS}")
    if len(commits) != 1 or None in commits:
        raise RuntimeError(f"{arm}: expected one clean code commit; got {commits}")
    if len(config_hashes) != 1 or None in config_hashes:
        raise RuntimeError(f"{arm}: expected one config hash; got {config_hashes}")
    return run_rows, round_rows, myth_rows, usage_rows, audits


def validate_joint(runs, audits):
    audit_paired_schedules(audits)
    issues = [issue for audit in audits for issue in audit["issues"]]
    if issues:
        raise RuntimeError("Joint schedule audit failed:\n" + "\n".join(issues))
    expected = {
        "off": {
            "rounds": 100,
            "dyads": 400,
            "calls": 1600,
            "llm_calls": 1400,
            "forced_responses": 200,
            "notifications": 0,
            "noise_checks": 800,
        },
        "on": {
            "rounds": 100,
            "dyads": 400,
            "calls": 2400,
            "llm_calls": 1700,
            "forced_responses": 300,
            "notifications": 400,
            "noise_checks": 800,
        },
    }
    for arm in ARM_ORDER:
        arm_audits = [audit for audit in audits if f"/{ON_EXPERIMENT}/" in str(audit["path"]) ] if arm == "on" else [audit for audit in audits if f"/{OFF_EXPERIMENT}/" in str(audit["path"])]
        for key, target in expected[arm].items():
            observed = sum(int(audit[key]) for audit in arm_audits)
            if observed != target:
                raise RuntimeError(f"{arm}: {key}={observed}; expected {target}")

    by_key = runs.set_index(["arm", "replicate_id"])
    for replicate_id in sorted(EXPECTED_IDS):
        on = by_key.loc[("on", replicate_id)]
        off = by_key.loc[("off", replicate_id)]
        for key in ("defector_ids", "pairing_seed", "noise_seed"):
            if on[key] != off[key]:
                raise RuntimeError(
                    f"replicate {replicate_id}: unmatched {key}: "
                    f"{on[key]!r} vs {off[key]!r}"
                )


def make_contrasts(runs):
    by_key = runs.set_index(["arm", "replicate_id"])
    specs = (
        ("standard_send_ratio", "Ordinary-agent proportion sent", True),
        ("standard_return_ratio", "Ordinary receiver return ratio", True),
        ("myth_coop_density", "Ordinary myth cooperation density", False),
        ("myth_threat_density", "Ordinary myth threat density", False),
        ("myth_half_rule", "Ordinary myth explicit half-rule", False),
        ("myth_punishment_density", "Ordinary myth punishment density", False),
        ("myth_punishment_presence", "Ordinary myth punishment presence", False),
    )
    rows = []
    for metric, label, primary in specs:
        differences = [
            by_key.loc[("on", replicate_id), metric]
            - by_key.loc[("off", replicate_id), metric]
            for replicate_id in sorted(EXPECTED_IDS)
        ]
        rows.append(paired_record(differences, metric, label, primary))
    primary_indexes = [index for index, row in enumerate(rows) if row["primary"]]
    adjusted = holm_adjust([rows[index]["p_value"] for index in primary_indexes])
    for index, p_holm in zip(primary_indexes, adjusted):
        rows[index]["p_holm"] = p_holm
    for index, row in enumerate(rows):
        if index not in primary_indexes:
            row["p_holm"] = math.nan
    return rows


def plot_paired_outcomes(runs, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, metric, ylabel in (
        (axes[0], "standard_send_ratio", "Ordinary-agent proportion sent"),
        (axes[1], "standard_return_ratio", "Ordinary receiver return ratio"),
    ):
        wide = runs.pivot(index="replicate_id", columns="arm", values=metric)
        for _, row in wide.iterrows():
            ax.plot([0, 1], [row["off"], row["on"]], color="#7b8794", marker="o", alpha=.7)
        ax.scatter(
            [0, 1], [wide["off"].mean(), wide["on"].mean()],
            color="#263238", s=125, zorder=5,
        )
        ax.set_xticks([0, 1], [ARM_LABELS["off"], ARM_LABELS["on"]])
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=.25)
    fig.suptitle("Matched punishment-availability outcomes", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "paired_behavior.png", dpi=300)
    plt.close(fig)


def plot_trajectories(rounds, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, metric, ylabel in (
        (axes[0], "standard_send_ratio", "Ordinary-agent proportion sent"),
        (axes[1], "standard_return_ratio", "Ordinary receiver return ratio"),
    ):
        for arm in ARM_ORDER:
            group = rounds[rounds["arm"] == arm].groupby("round")[metric].agg(["mean", "sem"])
            x = group.index.to_numpy()
            center = group["mean"].to_numpy()
            error = stats.t.ppf(.975, 9) * group["sem"].fillna(0).to_numpy()
            ax.plot(x, center, marker="o", color=COLORS[arm], label=ARM_LABELS[arm])
            ax.fill_between(x, center-error, center+error, color=COLORS[arm], alpha=.13)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=.25)
    axes[0].legend()
    fig.suptitle("Behavior across matched defector populations", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "behavior_trajectories.png", dpi=300)
    plt.close(fig)


def plot_myths(runs, output_dir):
    import matplotlib.pyplot as plt

    specs = (
        ("myth_coop_density", "Cooperation/fairness\nper 100 words"),
        ("myth_threat_density", "Threat/defection\nper 100 words"),
        ("myth_half_rule", "Explicit half-rule\nproportion"),
        ("myth_punishment_density", "Punishment/deduction\nper 100 words"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))
    for ax, (metric, ylabel) in zip(axes.flat, specs):
        wide = runs.pivot(index="replicate_id", columns="arm", values=metric)
        for _, row in wide.iterrows():
            ax.plot([0, 1], [row["off"], row["on"]], color="#7b8794", alpha=.65)
        for index, arm in enumerate(ARM_ORDER):
            ax.scatter(
                np.full(len(wide), index), wide[arm], color=COLORS[arm], s=45, alpha=.75
            )
            low, high = ci(wide[arm])
            ax.errorbar(
                index, wide[arm].mean(),
                yerr=[[wide[arm].mean()-low], [high-wide[arm].mean()]],
                fmt="o", color="#263238", capsize=6, linewidth=2,
            )
        ax.set_xticks([0, 1], ["Unavailable", "Available"])
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=.25)
    fig.suptitle("Ordinary-authored myth language", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "paired_myth_language.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on-root", type=Path, default=ON_ROOT)
    parser.add_argument("--off-root", type=Path, default=OFF_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    import pandas as pd

    on = load_arm(args.on_root, ON_EXPERIMENT, "on")
    off = load_arm(args.off_root, OFF_EXPERIMENT, "off")
    runs = pd.DataFrame(on[0] + off[0]).sort_values(["arm", "replicate_id"])
    rounds = pd.DataFrame(on[1] + off[1]).sort_values(["arm", "replicate_id", "round"])
    myths = pd.DataFrame(on[2] + off[2]).sort_values(["arm", "replicate_id", "round", "agent_id"])
    token_usage = pd.DataFrame(on[3] + off[3]).sort_values(["arm", "replicate_id"])
    audits = on[4] + off[4]
    validate_joint(runs, audits)
    contrasts = pd.DataFrame(make_contrasts(runs))

    primaries = contrasts[contrasts["primary"]]
    send = primaries[primaries["metric"] == "standard_send_ratio"].iloc[0]
    escalate = bool(
        (primaries["p_holm"] < .05).any()
        or (
            abs(send["estimate"]) >= .03
            and send["pairs_same_direction"] >= 8
        )
    )
    decision = pd.DataFrame(
        [{
            "decision": "escalate_to_independent_factorial" if escalate else "do_not_escalate",
            "any_primary_holm_below_05": bool((primaries["p_holm"] < .05).any()),
            "send_abs_at_least_03": bool(abs(send["estimate"]) >= .03),
            "send_pairs_same_direction_at_least_8": bool(send["pairs_same_direction"] >= 8),
        }]
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
    runs.to_csv(args.out / "run_metrics.csv", index=False)
    rounds.to_csv(args.out / "round_metrics.csv", index=False)
    myths.to_csv(args.out / "myth_metrics.csv", index=False)
    token_usage.to_csv(args.out / "token_usage.csv", index=False)
    contrasts.to_csv(args.out / "paired_contrasts.csv", index=False)
    decision.to_csv(args.out / "escalation_decision.csv", index=False)
    audit_table.to_csv(args.out / "audit.csv", index=False)
    plot_paired_outcomes(runs, args.out)
    plot_trajectories(rounds, args.out)
    plot_myths(runs, args.out)

    print("Paired contrasts:")
    print(contrasts.to_string(index=False))
    print("\nEscalation decision:")
    print(decision.to_string(index=False))
    print(f"\nSaved outputs to {args.out}")


if __name__ == "__main__":
    main()
