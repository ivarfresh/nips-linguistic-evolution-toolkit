#!/usr/bin/env python3
"""Analyze the frozen Gemini hidden-defector punishment confirmation."""

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
from audit_v2_protocol import audit_run


DEFAULT_INPUT = Path(
    "data/json/noise_experiments/defector_punishment_gemini_confirmation_n10_20260821"
)
EXPERIMENT = "noise8i_defector_punishment_gemini_confirmation_n10"
DEFAULT_OUTPUT = Path(
    "docs/figures/defector_punishment_gemini_confirmation_n10_20260821"
)
EXPECTED_IDS = set(range(70, 80))
MODEL = "google/gemini-3.1-flash-lite"


def ci(values):
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


def mean(values):
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def primary_record(values, metric, label):
    values = np.asarray(values, dtype=float)
    low, high = ci(values)
    sd = values.std(ddof=1)
    if np.isclose(sd, 0):
        p_value = 1.0 if np.isclose(values.mean(), 0) else 0.0
        statistic = 0.0 if np.isclose(values.mean(), 0) else math.inf
        dz = 0.0 if np.isclose(values.mean(), 0) else math.inf
    else:
        test = stats.ttest_1samp(values, 0)
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
        dz = float(values.mean() / sd)
    return {
        "metric": metric,
        "label": label,
        "contrast": "defector receiver − ordinary receiver",
        "n_populations": len(values),
        "estimate": float(values.mean()),
        "ci_low": low,
        "ci_high": high,
        "t_statistic": statistic,
        "p_value": p_value,
        "cohens_dz": dz,
    }


def extract(input_root):
    runs = load_runs(input_root / EXPERIMENT)
    if len(runs) != 10:
        raise RuntimeError(f"Found {len(runs)} runs; expected 10")

    audits = [audit_run(path) for path, _run in runs]
    issues = [issue for result in audits for issue in result["issues"]]
    if issues:
        raise RuntimeError("Protocol audit failed:\n" + "\n".join(issues))
    expected_totals = {
        "rounds": 100,
        "dyads": 400,
        "calls": 2400,
        "llm_calls": 1700,
        "forced_responses": 300,
        "notifications": 400,
        "noise_checks": 800,
    }
    for key, expected in expected_totals.items():
        observed = sum(int(result[key]) for result in audits)
        if observed != expected:
            raise RuntimeError(f"Audit total {key}={observed}; expected {expected}")

    run_rows = []
    decision_rows = []
    round_rows = []
    myth_rows = []
    usage_rows = []
    seen_ids = set()
    code_commits = set()
    config_hashes = set()

    for path, run in runs:
        metadata = run.get("run_metadata") or {}
        expected_metadata = {
            "model": MODEL,
            "llm_provider": "google",
            "provider_model": "gemini-3.1-flash-lite",
            "execution_provenance_version": 1,
            "punishment_enabled": True,
            "punishment_budget": 2,
            "punishment_effect_multiplier": 3,
            "punishment_prompt_variant": "current",
            "defector_ratio_actual": 0.25,
            "defector_action_policy": "forced_zero",
            "defector_myth_policy": "normal",
            "memory_capacity": 9,
            "history_policy": "none",
            "self_history_window": 0,
            "coplayer_history_window": 0,
            "population_history_window": 0,
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise RuntimeError(
                    f"{path}: {key}={metadata.get(key)!r}; expected {expected!r}"
                )
        if metadata.get("code_dirty") is not False:
            raise RuntimeError(f"{path}: dirty or missing execution provenance")
        seen_ids.add(int(metadata["replicate_id"]))
        code_commits.add(metadata.get("code_commit"))
        config_hashes.add(metadata.get("config_sha256"))
        defector_ids = set(metadata.get("defector_agent_ids") or [])
        if len(defector_ids) != 2:
            raise RuntimeError(f"{path}: expected two defectors; got {defector_ids}")
        standard_ids = set(run.get("agents") or {}) - defector_ids

        run_decisions = []
        run_returns = []
        run_myths = []
        for entry in run.get("conversation_history") or []:
            round_number = int(entry["round"])
            round_decisions = []
            for dyad in entry.get("dyads") or []:
                investor = dyad["investor"]
                trustee = dyad["trustee"]
                received = float(dyad.get("received") or 0)
                if trustee in standard_ids and received > 0:
                    run_returns.append(float(dyad["returned"]) / received)
                if investor not in standard_ids:
                    continue
                received_visible = float(dyad.get("received_communicated") or 0)
                returned_visible = float(dyad.get("returned_communicated") or 0)
                visible_ratio = (
                    returned_visible / received_visible
                    if received_visible > 0
                    else math.nan
                )
                spent = int(dyad["deduction_spent"])
                row = {
                    "replicate_id": int(metadata["replicate_id"]),
                    "round": round_number,
                    "investor_id": investor,
                    "trustee_id": trustee,
                    "target_type": (
                        "defector" if trustee in defector_ids else "standard"
                    ),
                    "send_ratio": float(dyad["sent"]) / 5.0,
                    "received_visible": received_visible,
                    "returned_visible": returned_visible,
                    "visible_return_ratio": visible_ratio,
                    "deduction_spent": spent,
                    "any_deduction": float(spent > 0),
                    "adequate_half_return": float(
                        np.isfinite(visible_ratio) and visible_ratio >= 0.5
                    ),
                }
                decision_rows.append(row)
                run_decisions.append(row)
                round_decisions.append(row)
            round_rows.append(
                {
                    "replicate_id": int(metadata["replicate_id"]),
                    "round": round_number,
                    "standard_send_ratio": mean(
                        [row["send_ratio"] for row in round_decisions]
                    ),
                    "deduction_spent": mean(
                        [row["deduction_spent"] for row in round_decisions]
                    ),
                    "any_deduction": mean(
                        [row["any_deduction"] for row in round_decisions]
                    ),
                }
            )
            for agent_id, text in (entry.get("myths") or {}).items():
                if agent_id not in standard_ids:
                    continue
                metrics = myth_metrics(text)
                words = re.findall(r"\b[\w'-]+\b", text.lower())
                punishment_matches = len(PUNISH_PATTERN.findall(text))
                row = {
                    "replicate_id": int(metadata["replicate_id"]),
                    "round": round_number,
                    "agent_id": agent_id,
                    "text": text,
                    **metrics,
                    "punishment_density": (
                        100 * punishment_matches / len(words) if words else 0.0
                    ),
                    "punishment_presence": float(punishment_matches > 0),
                }
                myth_rows.append(row)
                run_myths.append(row)

        defector_targets = [
            row for row in run_decisions if row["target_type"] == "defector"
        ]
        standard_targets = [
            row for row in run_decisions if row["target_type"] == "standard"
        ]
        if len(defector_targets) != 8 or len(standard_targets) != 22:
            raise RuntimeError(
                f"{path}: target counts {len(defector_targets)}/"
                f"{len(standard_targets)}; expected 8/22"
            )
        run_rows.append(
            {
                "replicate_id": int(metadata["replicate_id"]),
                "defector_ids": ",".join(sorted(defector_ids)),
                "standard_send_ratio": mean(
                    [row["send_ratio"] for row in run_decisions]
                ),
                "standard_return_ratio": mean(run_returns),
                "defector_target_spent": mean(
                    [row["deduction_spent"] for row in defector_targets]
                ),
                "standard_target_spent": mean(
                    [row["deduction_spent"] for row in standard_targets]
                ),
                "defector_target_any": mean(
                    [row["any_deduction"] for row in defector_targets]
                ),
                "standard_target_any": mean(
                    [row["any_deduction"] for row in standard_targets]
                ),
                "target_contrast_spent": mean(
                    [row["deduction_spent"] for row in defector_targets]
                )
                - mean([row["deduction_spent"] for row in standard_targets]),
                "target_contrast_any": mean(
                    [row["any_deduction"] for row in defector_targets]
                )
                - mean([row["any_deduction"] for row in standard_targets]),
                "myth_coop_density": mean(
                    [row["coop_density"] for row in run_myths]
                ),
                "myth_threat_density": mean(
                    [row["threat_density"] for row in run_myths]
                ),
                "myth_half_rule": mean([row["half_rule"] for row in run_myths]),
                "myth_punishment_density": mean(
                    [row["punishment_density"] for row in run_myths]
                ),
            }
        )
        usage_rows.append(
            {
                "replicate_id": int(metadata["replicate_id"]),
                **usage_metrics(run),
            }
        )

    if seen_ids != EXPECTED_IDS:
        raise RuntimeError(f"Replicate IDs {seen_ids}; expected {EXPECTED_IDS}")
    if len(code_commits) != 1 or None in code_commits:
        raise RuntimeError(f"Expected one recorded code commit; got {code_commits}")
    if len(config_hashes) != 1 or None in config_hashes:
        raise RuntimeError(f"Expected one config hash; got {config_hashes}")
    return run_rows, decision_rows, round_rows, myth_rows, usage_rows, audits


def plot_targeting(runs, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7))
    for ax, suffix, ylabel, ylim in (
        (axes[0], "spent", "Mean deduction points", (0, 2.05)),
        (axes[1], "any", "Probability of any deduction", (0, 1.05)),
    ):
        ordinary = runs[f"standard_target_{suffix}"].to_numpy()
        defector = runs[f"defector_target_{suffix}"].to_numpy()
        for left, right in zip(ordinary, defector):
            ax.plot([0, 1], [left, right], marker="o", color="#7b8794", alpha=.7)
        ax.scatter(
            [0, 1],
            [ordinary.mean(), defector.mean()],
            s=130,
            color="#c14953",
            zorder=5,
        )
        ax.set_xticks([0, 1], ["Ordinary receiver", "Hidden defector"])
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y", alpha=.25)
    fig.suptitle("Independent confirmation of selective deduction", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "deduction_targeting.png", dpi=300)
    plt.close(fig)


def plot_return_bins(decisions, output_dir):
    import matplotlib.pyplot as plt
    import pandas as pd

    finite = decisions[np.isfinite(decisions["visible_return_ratio"])].copy()
    finite["return_bin"] = pd.cut(
        finite["visible_return_ratio"],
        [-np.inf, .001, .25, .5, np.inf],
        labels=["≈0", "(0,.25)", "[.25,.50)", "≥.50"],
        right=False,
    )
    summary = (
        finite.groupby("return_bin", observed=False)
        .agg(
            n=("any_deduction", "size"),
            any_deduction=("any_deduction", "mean"),
            mean_spent=("deduction_spent", "mean"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    bars = ax.bar(
        summary["return_bin"].astype(str),
        summary["any_deduction"].fillna(0),
        color="#c14953",
    )
    for bar, n in zip(bars, summary["n"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + .025,
            f"n={int(n)}",
            ha="center",
        )
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("Visible return / visible amount received")
    ax.set_ylabel("Probability of any deduction")
    ax.set_title("Deduction by visible return", fontweight="bold")
    ax.grid(True, axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(output_dir / "deduction_by_visible_return.png", dpi=300)
    plt.close(fig)
    return summary


def plot_trajectories(rounds, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, metric, ylabel, upper in (
        (axes[0], "standard_send_ratio", "Ordinary-agent proportion sent", 1),
        (axes[1], "deduction_spent", "Mean deduction points", 2),
    ):
        grouped = rounds.groupby("round")[metric].agg(["mean", "sem"])
        x = grouped.index.to_numpy()
        center = grouped["mean"].to_numpy()
        error = stats.t.ppf(.975, 9) * grouped["sem"].fillna(0).to_numpy()
        ax.plot(x, center, marker="o", color="#c14953", linewidth=2.2)
        ax.fill_between(
            x,
            np.clip(center - error, 0, upper),
            np.clip(center + error, 0, upper),
            color="#c14953",
            alpha=.15,
        )
        ax.set_ylim(-.05, upper+.05)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=.25)
    fig.suptitle("Confirmation-population trajectories", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "behavior_trajectories.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    import pandas as pd

    run_rows, decision_rows, round_rows, myth_rows, usage_rows, audits = extract(
        args.input
    )
    runs = pd.DataFrame(run_rows).sort_values("replicate_id")
    decisions = pd.DataFrame(decision_rows).sort_values(
        ["replicate_id", "round", "investor_id"]
    )
    rounds = pd.DataFrame(round_rows).sort_values(["replicate_id", "round"])
    myths = pd.DataFrame(myth_rows).sort_values(
        ["replicate_id", "round", "agent_id"]
    )
    token_usage = pd.DataFrame(usage_rows).sort_values("replicate_id")
    primary = pd.DataFrame(
        [
            primary_record(
                runs["target_contrast_spent"],
                "target_contrast_spent",
                "Mean deduction points",
            ),
            primary_record(
                runs["target_contrast_any"],
                "target_contrast_any",
                "Probability of any deduction",
            ),
        ]
    )
    primary["p_holm"] = holm_adjust(primary["p_value"].tolist())

    defector = decisions[decisions["target_type"] == "defector"]
    adequate = decisions[decisions["adequate_half_return"] == 1]
    gates = pd.DataFrame(
        [
            ["spending contrast", primary.iloc[0]["estimate"], .5, ">="],
            ["any-deduction contrast", primary.iloc[1]["estimate"], .25, ">="],
            ["spending contrast Holm p", primary.iloc[0]["p_holm"], .05, "<"],
            ["any contrast Holm p", primary.iloc[1]["p_holm"], .05, "<"],
            [
                "defector-target any probability",
                defector["any_deduction"].mean(),
                .5,
                ">=",
            ],
            [
                "any after visible half-return",
                adequate["any_deduction"].mean(),
                .25,
                "<=",
            ],
        ],
        columns=["criterion", "observed", "threshold", "comparison"],
    )
    gates["passed"] = [
        value >= threshold
        if op == ">="
        else value <= threshold
        if op == "<="
        else value < threshold
        for value, threshold, op in zip(
            gates["observed"], gates["threshold"], gates["comparison"]
        )
    ]
    if not gates["passed"].all():
        decision = "not_confirmed"
    else:
        decision = "confirmed"
    gates["overall_decision"] = decision

    audit_table = pd.DataFrame(
        [
            {
                key: value
                for key, value in item.items()
                if key not in {"issues", "pairing_signature"}
            }
            for item in audits
        ]
    )
    return_bins = plot_return_bins(decisions, args.out)
    plot_targeting(runs, args.out)
    plot_trajectories(rounds, args.out)

    runs.to_csv(args.out / "run_metrics.csv", index=False)
    decisions.to_csv(args.out / "deduction_decisions.csv", index=False)
    rounds.to_csv(args.out / "round_metrics.csv", index=False)
    myths.to_csv(args.out / "myth_metrics.csv", index=False)
    token_usage.to_csv(args.out / "token_usage.csv", index=False)
    primary.to_csv(args.out / "primary_tests.csv", index=False)
    gates.to_csv(args.out / "confirmation_gate.csv", index=False)
    audit_table.to_csv(args.out / "audit.csv", index=False)
    return_bins.to_csv(args.out / "deduction_by_return_bin.csv", index=False)

    print("Primary tests:")
    print(primary.to_string(index=False))
    print("\nConfirmation gate:")
    print(gates.to_string(index=False))
    print(f"\nOverall decision: {decision}")
    print(f"Saved outputs to {args.out}")


if __name__ == "__main__":
    main()
