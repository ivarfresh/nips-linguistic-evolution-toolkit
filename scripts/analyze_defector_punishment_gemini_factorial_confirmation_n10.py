#!/usr/bin/env python3
"""Analyze the frozen Gemini punishment-availability 2x2 confirmation."""

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
from analyze_corrected_v2_confirmatory import holm_adjust, load_runs
from analyze_defector_myth_game_crossmodel_n5 import THREAT_PATTERN, myth_metrics
from analyze_defector_punishment_gpt_n5 import PUNISH_PATTERN, usage_metrics
from audit_v2_protocol import audit_paired_schedules, audit_run


DEFAULT_INPUT = Path(
    "data/json/noise_experiments/"
    "defector_punishment_gemini_factorial_confirmation_n10_20260822"
)
EXPERIMENT = "noise8i_defector_punishment_gemini_factorial_confirmation_n10"
DEFAULT_OUTPUT = Path(
    "docs/figures/"
    "defector_punishment_gemini_factorial_confirmation_n10_20260822"
)
EXPECTED_IDS = set(range(80, 90))
MODEL = "google/gemini-3.1-flash-lite"
PROVIDER_MODEL = "gemini-3.1-flash-lite"
EXPECTED_RUNTIME_METADATA = {}
AVAILABILITY_ORDER = ["off", "on"]
DEFECTOR_ORDER = ["control", "defectors25"]
AVAILABILITY_LABELS = {"off": "Deduction unavailable", "on": "Deduction available"}
DEFECTOR_LABELS = {"control": "0% defectors", "defectors25": "25% defectors"}
COLORS = {"off": "#457b9d", "on": "#c14953"}
METRICS = {
    "standard_return_ratio": "Ordinary receiver return ratio",
    "standard_send_ratio": "Ordinary-agent proportion sent",
    "myth_coop_density": "Ordinary myth cooperation density",
    "myth_threat_density": "Ordinary myth threat density",
    "myth_half_rule": "Ordinary myth explicit half-rule",
    "myth_punishment_density": "Ordinary myth punishment density",
    "myth_punishment_presence": "Ordinary myth punishment presence",
}


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


def contrast_record(values, metric, contrast_type, label, primary=False):
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
    return {
        "metric": metric,
        "metric_label": METRICS[metric],
        "contrast_type": contrast_type,
        "contrast": label,
        "primary": primary,
        "n_quadruplets": len(values),
        "estimate": float(values.mean()),
        "ci_low": low,
        "ci_high": high,
        "t_statistic": statistic,
        "p_value": p_value,
        "cohens_dz": dz,
        "same_direction": int(np.sum(np.sign(values) == direction)),
        "zero_differences": int(np.sum(np.isclose(values, 0))),
    }


def extract(input_root):
    runs = load_runs(input_root / EXPERIMENT)
    expected_run_count = 4 * len(EXPECTED_IDS)
    if len(runs) != expected_run_count:
        raise RuntimeError(
            f"Found {len(runs)} runs; expected {expected_run_count}"
        )

    run_rows = []
    round_rows = []
    myth_rows = []
    return_rows = []
    deduction_rows = []
    usage_rows = []
    audits = []
    seen = {
        (availability, condition): set()
        for availability in AVAILABILITY_ORDER
        for condition in DEFECTOR_ORDER
    }
    commits = set()
    config_hashes = set()

    for path, run in runs:
        audit = audit_run(path)
        if audit["issues"]:
            raise RuntimeError("\n".join(audit["issues"]))
        metadata = run.get("run_metadata") or {}
        availability = "on" if metadata.get("punishment_enabled") else "off"
        defector_ids = set(metadata.get("defector_agent_ids") or [])
        condition = "defectors25" if defector_ids else "control"
        audit["availability"] = availability
        audit["condition"] = condition
        audits.append(audit)

        expected = {
            "model": MODEL,
            "llm_provider": "google",
            "provider_model": PROVIDER_MODEL,
            "execution_provenance_version": 1,
            "defector_ratio_actual": .25 if condition == "defectors25" else 0.0,
            "defector_action_policy": "forced_zero",
            "defector_myth_policy": "normal",
            "history_policy": "none",
            "self_history_window": 0,
            "coplayer_history_window": 0,
            "population_history_window": 0,
            "memory_capacity": 9 if availability == "on" else 6,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(
                    f"{path}: {key}={metadata.get(key)!r}; expected {value!r}"
                )
        for key, value in EXPECTED_RUNTIME_METADATA.items():
            if metadata.get(key) != value:
                raise RuntimeError(
                    f"{path}: {key}={metadata.get(key)!r}; expected {value!r}"
                )
        if metadata.get("code_dirty") is not False:
            raise RuntimeError(f"{path}: dirty or missing provenance")
        if availability == "on" and metadata.get("punishment_prompt_variant") != "current":
            raise RuntimeError(f"{path}: available arm did not use current wording")
        if len(defector_ids) != (2 if condition == "defectors25" else 0):
            raise RuntimeError(f"{path}: wrong defector count")

        replicate_id = int(metadata["replicate_id"])
        seen[(availability, condition)].add(replicate_id)
        commits.add(metadata.get("code_commit"))
        config_hashes.add(metadata.get("config_sha256"))
        standard_ids = set(run.get("agents") or {}) - defector_ids
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
                    send_ratio = float(dyad["sent"]) / 5.0
                    sends.append(send_ratio)
                    round_sends.append(send_ratio)
                received = float(dyad.get("received") or 0)
                if trustee in standard_ids and received > 0:
                    returned = float(dyad.get("returned") or 0)
                    return_ratio = returned / received
                    returns.append(return_ratio)
                    round_returns.append(return_ratio)
                    return_rows.append(
                        {
                            "availability": availability,
                            "condition": condition,
                            "replicate_id": replicate_id,
                            "round": round_number,
                            "trustee_id": trustee,
                            "received": received,
                            "returned": returned,
                            "return_ratio": return_ratio,
                        }
                    )
                if availability == "on" and investor in standard_ids:
                    received_visible = float(dyad.get("received_communicated") or 0)
                    returned_visible = float(dyad.get("returned_communicated") or 0)
                    visible_ratio = (
                        returned_visible / received_visible
                        if received_visible > 0
                        else math.nan
                    )
                    spent = int(dyad["deduction_spent"])
                    deduction_rows.append(
                        {
                            "condition": condition,
                            "replicate_id": replicate_id,
                            "round": round_number,
                            "target_type": (
                                "defector" if trustee in defector_ids else "standard"
                            ),
                            "visible_return_ratio": visible_ratio,
                            "deduction_spent": spent,
                            "any_deduction": float(spent > 0),
                        }
                    )

            round_rows.append(
                {
                    "availability": availability,
                    "condition": condition,
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
                    "availability": availability,
                    "condition": condition,
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

        expected_sends = 30 if condition == "defectors25" else 40
        if len(sends) != expected_sends:
            raise RuntimeError(
                f"{path}: {len(sends)} ordinary sends; expected {expected_sends}"
            )
        run_rows.append(
            {
                "availability": availability,
                "availability_label": AVAILABILITY_LABELS[availability],
                "condition": condition,
                "condition_label": DEFECTOR_LABELS[condition],
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
    if len(commits) != 1 or None in commits:
        raise RuntimeError(f"Expected one code commit; got {commits}")
    if len(config_hashes) != 1 or None in config_hashes:
        raise RuntimeError(f"Expected one config hash; got {config_hashes}")

    audit_paired_schedules(audits)
    issues = [issue for audit in audits for issue in audit["issues"]]
    if issues:
        raise RuntimeError("Joint schedule audit failed:\n" + "\n".join(issues))
    number_of_replicates = len(EXPECTED_IDS)
    expected_cell_totals = {
        ("off", "control"): (160, 160, 0, 0),
        ("off", "defectors25"): (160, 140, 20, 0),
        ("on", "control"): (240, 200, 0, 40),
        ("on", "defectors25"): (240, 170, 30, 40),
    }
    for cell, per_run_targets in expected_cell_totals.items():
        targets = tuple(
            number_of_replicates * target for target in per_run_targets
        )
        subset = [
            audit
            for audit in audits
            if (audit["availability"], audit["condition"]) == cell
        ]
        for key, target in zip(
            ("calls", "llm_calls", "forced_responses", "notifications"), targets
        ):
            observed = sum(int(audit[key]) for audit in subset)
            if observed != target:
                raise RuntimeError(f"{cell}: {key}={observed}; expected {target}")
        noise_checks = sum(int(audit["noise_checks"]) for audit in subset)
        expected_noise_checks = 80 * number_of_replicates
        if noise_checks != expected_noise_checks:
            raise RuntimeError(
                f"{cell}: noise_checks={noise_checks}; "
                f"expected {expected_noise_checks}"
            )

    import pandas as pd

    run_frame = pd.DataFrame(run_rows).set_index(
        ["availability", "condition", "replicate_id"]
    )
    for replicate_id in sorted(EXPECTED_IDS):
        rows = {
            (availability, condition): run_frame.loc[
                (availability, condition, replicate_id)
            ]
            for availability in AVAILABILITY_ORDER
            for condition in DEFECTOR_ORDER
        }
        seeds = {
            (row["pairing_seed"], row["noise_seed"]) for row in rows.values()
        }
        if len(seeds) != 1:
            raise RuntimeError(f"replicate {replicate_id}: unmatched seeds {seeds}")
        if (
            rows[("off", "defectors25")]["defector_ids"]
            != rows[("on", "defectors25")]["defector_ids"]
        ):
            raise RuntimeError(f"replicate {replicate_id}: unmatched defectors")

    return (
        run_rows,
        round_rows,
        myth_rows,
        return_rows,
        deduction_rows,
        usage_rows,
        audits,
    )


def make_contrasts(runs):
    rows = []
    replicate_rows = []
    pivoted = runs.set_index(["availability", "condition", "replicate_id"])
    for metric in METRICS:
        values = {
            cell: np.asarray(
                [
                    pivoted.loc[(cell[0], cell[1], replicate_id), metric]
                    for replicate_id in sorted(EXPECTED_IDS)
                ],
                dtype=float,
            )
            for cell in (
                ("off", "control"),
                ("on", "control"),
                ("off", "defectors25"),
                ("on", "defectors25"),
            )
        }
        availability_control = values[("on", "control")] - values[("off", "control")]
        availability_defectors = (
            values[("on", "defectors25")] - values[("off", "defectors25")]
        )
        contrasts = {
            "availability_defectors25": (
                availability_defectors,
                "Available − unavailable | 25% defectors",
            ),
            "availability_control": (
                availability_control,
                "Available − unavailable | 0% defectors",
            ),
            "availability_main": (
                (availability_defectors + availability_control) / 2,
                "Equal-weight availability main effect",
            ),
            "interaction": (
                availability_defectors - availability_control,
                "Availability × defector difference-in-differences",
            ),
            "defector_effect_off": (
                values[("off", "defectors25")] - values[("off", "control")],
                "25% − 0% defectors | unavailable",
            ),
            "defector_effect_on": (
                values[("on", "defectors25")] - values[("on", "control")],
                "25% − 0% defectors | available",
            ),
        }
        for contrast_type, (contrast_values, label) in contrasts.items():
            primary = metric == "standard_return_ratio" and contrast_type in {
                "availability_defectors25",
                "interaction",
            }
            rows.append(
                contrast_record(
                    contrast_values,
                    metric,
                    contrast_type,
                    label,
                    primary,
                )
            )
            for replicate_id, value in zip(sorted(EXPECTED_IDS), contrast_values):
                replicate_rows.append(
                    {
                        "metric": metric,
                        "contrast_type": contrast_type,
                        "replicate_id": replicate_id,
                        "difference": value,
                    }
                )
    primary_indexes = [index for index, row in enumerate(rows) if row["primary"]]
    adjusted = holm_adjust([rows[index]["p_value"] for index in primary_indexes])
    for index, p_holm in zip(primary_indexes, adjusted):
        rows[index]["p_holm"] = p_holm
    for index, row in enumerate(rows):
        if index not in primary_indexes:
            row["p_holm"] = math.nan
    return rows, replicate_rows


def make_term_counts(myths):
    rows = []
    for (availability, condition), group in myths.groupby(
        ["availability", "condition"]
    ):
        for lexicon, pattern in (
            ("threat_defection", THREAT_PATTERN),
            ("punishment_deduction", PUNISH_PATTERN),
        ):
            counts = Counter()
            for text in group["text"]:
                counts.update(match.group(0).lower() for match in pattern.finditer(text))
            for term, count in counts.most_common():
                rows.append(
                    {
                        "availability": availability,
                        "condition": condition,
                        "lexicon": lexicon,
                        "term": term,
                        "count": count,
                    }
                )
    return rows


def plot_behavior(runs, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for row_index, (metric, ylabel) in enumerate(
        (
            ("standard_return_ratio", "Ordinary receiver return ratio"),
            ("standard_send_ratio", "Ordinary-agent proportion sent"),
        )
    ):
        for col_index, condition in enumerate(DEFECTOR_ORDER):
            ax = axes[row_index, col_index]
            subset = runs[runs["condition"] == condition]
            wide = subset.pivot(
                index="replicate_id", columns="availability", values=metric
            )
            for _, values in wide.iterrows():
                ax.plot(
                    [0, 1], [values["off"], values["on"]],
                    marker="o", color="#7b8794", alpha=.7,
                )
            ax.scatter(
                [0, 1], [wide["off"].mean(), wide["on"].mean()],
                color="#263238", s=125, zorder=5,
            )
            ax.set_xticks([0, 1], ["Unavailable", "Available"])
            ax.set_ylabel(ylabel)
            ax.set_title(DEFECTOR_LABELS[condition])
            ax.grid(True, axis="y", alpha=.25)
    fig.suptitle("Independent punishment-availability factorial", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "paired_behavior_factorial.png", dpi=300)
    plt.close(fig)


def plot_return_trajectories(rounds, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, condition in zip(axes, DEFECTOR_ORDER):
        for availability in AVAILABILITY_ORDER:
            subset = rounds[
                (rounds["condition"] == condition)
                & (rounds["availability"] == availability)
            ]
            group = subset.groupby("round")["standard_return_ratio"].agg(
                ["mean", "sem"]
            )
            x = group.index.to_numpy()
            center = group["mean"].to_numpy()
            error = stats.t.ppf(.975, len(EXPECTED_IDS) - 1) * group[
                "sem"
            ].fillna(0).to_numpy()
            ax.plot(
                x, center, marker="o", color=COLORS[availability],
                label=AVAILABILITY_LABELS[availability],
            )
            ax.fill_between(
                x,
                np.clip(center - error, 0, 1),
                np.clip(center + error, 0, 1),
                color=COLORS[availability],
                alpha=.13,
            )
        ax.set_title(DEFECTOR_LABELS[condition])
        ax.set_xlabel("Round")
        ax.set_ylabel("Ordinary receiver return ratio")
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=.25)
    axes[0].legend()
    fig.suptitle("Return trajectories in the independent factorial", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "return_trajectories.png", dpi=300)
    plt.close(fig)


def plot_myth_interactions(contrasts, output_dir):
    import matplotlib.pyplot as plt

    metrics = [
        "myth_coop_density",
        "myth_threat_density",
        "myth_half_rule",
        "myth_punishment_density",
    ]
    labels = [
        "Cooperation\ndensity",
        "Threat\ndensity",
        "Explicit\nhalf rule",
        "Punishment\ndensity",
    ]
    subset = contrasts[
        (contrasts["contrast_type"] == "interaction")
        & contrasts["metric"].isin(metrics)
    ].set_index("metric")
    estimates = [subset.loc[metric, "estimate"] for metric in metrics]
    lows = [subset.loc[metric, "ci_low"] for metric in metrics]
    highs = [subset.loc[metric, "ci_high"] for metric in metrics]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.errorbar(
        x,
        estimates,
        yerr=[
            np.asarray(estimates) - np.asarray(lows),
            np.asarray(highs) - np.asarray(estimates),
        ],
        fmt="o",
        color="#6a4c93",
        capsize=7,
        markersize=9,
        linewidth=2,
    )
    ax.axhline(0, color="#263238", linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Availability × defector interaction")
    ax.set_title("Ordinary-authored myth interactions", fontweight="bold")
    ax.grid(True, axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(output_dir / "myth_interactions.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    import pandas as pd

    extracted = extract(args.input)
    runs = pd.DataFrame(extracted[0]).sort_values(
        ["availability", "condition", "replicate_id"]
    )
    rounds = pd.DataFrame(extracted[1]).sort_values(
        ["availability", "condition", "replicate_id", "round"]
    )
    myths = pd.DataFrame(extracted[2]).sort_values(
        ["availability", "condition", "replicate_id", "round", "agent_id"]
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
    contrast_rows, replicate_rows = make_contrasts(runs)
    contrasts = pd.DataFrame(contrast_rows)
    replicate_contrasts = pd.DataFrame(replicate_rows)
    primaries = contrasts[contrasts["primary"]].set_index("contrast_type")
    replication = primaries.loc["availability_defectors25"]
    interaction = primaries.loc["interaction"]
    decisions = pd.DataFrame(
        [
            {
                "claim": "negative availability effect within 25% defectors",
                "estimate": replication["estimate"],
                "minimum_effect_rule": "estimate <= -0.025",
                "minimum_effect_passed": bool(replication["estimate"] <= -.025),
                "p_holm": replication["p_holm"],
                "testing_passed": bool(replication["p_holm"] < .05),
                "decision": (
                    "confirmed"
                    if replication["estimate"] <= -.025
                    and replication["p_holm"] < .05
                    else "not_confirmed"
                ),
            },
            {
                "claim": "availability by defector interaction",
                "estimate": interaction["estimate"],
                "minimum_effect_rule": "abs(estimate) >= 0.025",
                "minimum_effect_passed": bool(abs(interaction["estimate"]) >= .025),
                "p_holm": interaction["p_holm"],
                "testing_passed": bool(interaction["p_holm"] < .05),
                "decision": (
                    "confirmed"
                    if abs(interaction["estimate"]) >= .025
                    and interaction["p_holm"] < .05
                    else "not_confirmed"
                ),
            },
        ]
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
    term_counts = pd.DataFrame(make_term_counts(myths))

    runs.to_csv(args.out / "run_metrics.csv", index=False)
    rounds.to_csv(args.out / "round_metrics.csv", index=False)
    myths.to_csv(args.out / "myth_metrics.csv", index=False)
    returns.to_csv(args.out / "return_decisions.csv", index=False)
    deductions.to_csv(args.out / "deduction_decisions.csv", index=False)
    token_usage.to_csv(args.out / "token_usage.csv", index=False)
    contrasts.to_csv(args.out / "contrasts.csv", index=False)
    replicate_contrasts.to_csv(args.out / "replicate_contrasts.csv", index=False)
    decisions.to_csv(args.out / "confirmatory_decisions.csv", index=False)
    audit_table.to_csv(args.out / "audit.csv", index=False)
    term_counts.to_csv(args.out / "myth_term_counts.csv", index=False)
    plot_behavior(runs, args.out)
    plot_return_trajectories(rounds, args.out)
    plot_myth_interactions(contrasts, args.out)

    print("Primary contrasts:")
    print(primaries.to_string())
    print("\nConfirmatory decisions:")
    print(decisions.to_string(index=False))
    print(f"\nSaved outputs to {args.out}")


if __name__ == "__main__":
    main()
