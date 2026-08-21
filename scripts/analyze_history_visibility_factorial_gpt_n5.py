#!/usr/bin/env python3
"""Analyze the clean GPT-5 Nano history-visibility by task-order factorial.

The analysis selects five matched protocol seeds in each of six cells:
current-partner dossier versus private interaction memory, crossed with Game only,
Game→Myth, and Myth→Game. Runs with accepted decision-as-myth responses are
excluded and replaced only by exact-seed clean reruns.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyses._shared import configure_matplotlib
from analyze_corrected_v2_confirmatory import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_ORDER,
    holm_adjust,
    load_runs,
    run_metrics,
)


PARTNER_ROOT = Path("data/json/noise_experiments/crossmodel_signed_gpt_n5_20260821")
PARTNER_REPAIRS = Path(
    "data/json/noise_experiments/history_factorial_mythrepairs_20260821"
)
OWN_ROOT = Path("data/json/noise_experiments/history_gate_signed_gpt_n5_20260821")
OWN_REPAIRS = Path(
    "data/json/noise_experiments/history_factorial_own_retries_20260821"
)
DEFAULT_OUTPUT = Path("docs/figures/history_visibility_factorial_gpt_n5_20260821")

HISTORY_ORDER = ["own_only", "partner_dossier"]
HISTORY_LABELS = {
    "own_only": "Private interaction memory",
    "partner_dossier": "Current-partner dossier",
}
HISTORY_COLORS = {
    "own_only": "#2a9d8f",
    "partner_dossier": "#e76f51",
}


def _replicate_id(run):
    return int((run.get("run_metadata") or {})["replicate_id"])


def _selected_runs():
    """Return the six clean cells with one run per paired replicate ID."""
    cells = {
        ("partner_dossier", "game"): load_runs(
            PARTNER_ROOT / "noise8_crossmodel_signed_gpt_n5_game"
        ),
        ("own_only", "game"): load_runs(
            OWN_ROOT / "noise8_history_gate_signed_gpt_n5_ownonly_game"
        ),
        ("own_only", "game_myth"): load_runs(
            OWN_REPAIRS
            / "noise8_history_gate_signed_gpt_n5_ownonly_game_myth_mythrepair"
        ),
        ("own_only", "myth_game"): load_runs(
            OWN_REPAIRS
            / "noise8_history_gate_signed_gpt_n5_ownonly_myth_game_mythrepair"
        ),
    }

    partner_game_myth = [
        item
        for item in load_runs(
            PARTNER_ROOT / "noise8_crossmodel_signed_gpt_n5_game_myth"
        )
        if _replicate_id(item[1]) != 1
    ]
    partner_game_myth.extend(
        load_runs(
            PARTNER_REPAIRS
            / "noise8_crossmodel_signed_gpt_n5_game_myth_mythrepair"
        )
    )
    cells[("partner_dossier", "game_myth")] = partner_game_myth

    partner_myth_game = [
        item
        for item in load_runs(
            PARTNER_ROOT / "noise8_crossmodel_signed_gpt_n5_myth_game"
        )
        if _replicate_id(item[1]) != 0
    ]
    partner_myth_game.extend(
        load_runs(
            PARTNER_REPAIRS
            / "noise8_crossmodel_signed_gpt_n5_myth_game_mythrepair"
        )
    )
    cells[("partner_dossier", "myth_game")] = partner_myth_game

    expected = set(range(5))
    for cell, runs in cells.items():
        observed = [_replicate_id(run) for _, run in runs]
        if len(observed) != 5 or set(observed) != expected:
            raise RuntimeError(f"Cell {cell} has replicate IDs {sorted(observed)}")
        for path, run in runs:
            metadata = run.get("run_metadata") or {}
            if metadata.get("code_dirty"):
                raise RuntimeError(f"Dirty execution provenance in {path}")
            if metadata.get("model") != "openai/gpt-5-nano":
                raise RuntimeError(f"Unexpected model in {path}")
    return cells


def _usage(run):
    totals = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    attempts = 0
    retries = 0
    for agent in (run.get("agents") or {}).values():
        for event in agent.get("interaction_history") or []:
            response = event.get("response") or {}
            usage = response.get("usage") or {}
            if response:
                attempts += 1
            if event.get("error"):
                retries += 1
            for key in totals:
                totals[key] += int(usage.get(key) or 0)
    return totals, attempts, retries


def load_factorial():
    rows = []
    trajectories = []
    usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    attempts = 0
    retries = 0
    for (history, condition), runs in _selected_runs().items():
        for path, run in runs:
            metrics, run_trajectory = run_metrics(path, run)
            replicate_id = _replicate_id(run)
            metrics["return_over_sent"] = (
                metrics["mean_returned"] / metrics["mean_sent"]
                if metrics["mean_sent"] > 0
                else math.nan
            )
            rows.append(
                {
                    "history": history,
                    "history_label": HISTORY_LABELS[history],
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "replicate_id": replicate_id,
                    **metrics,
                }
            )
            trajectories.extend(
                {
                    "history": history,
                    "history_label": HISTORY_LABELS[history],
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "replicate_id": replicate_id,
                    **point,
                }
                for point in run_trajectory
            )
            run_usage, run_attempts, run_retries = _usage(run)
            for key in usage:
                usage[key] += run_usage[key]
            attempts += run_attempts
            retries += run_retries
    return rows, trajectories, usage, attempts, retries


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    return stats.t.interval(
        0.95,
        len(values) - 1,
        loc=values.mean(),
        scale=stats.sem(values),
    )


def paired_record(dataframe, metric, left_query, right_query, label, family):
    left = dataframe.query(left_query).set_index("replicate_id")[metric]
    right = dataframe.query(right_query).set_index("replicate_id")[metric]
    differences = (left - right).sort_index().to_numpy(dtype=float)
    ci_low, ci_high = confidence_interval(differences)
    return {
        "metric": metric,
        "contrast": label,
        "family": family,
        "n_pairs": len(differences),
        "estimate": differences.mean(),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": stats.ttest_1samp(differences, 0).pvalue,
    }


def summaries(dataframe):
    records = []
    for history in HISTORY_ORDER:
        for condition in CONDITION_ORDER:
            cell = dataframe[
                (dataframe["history"] == history)
                & (dataframe["condition"] == condition)
            ]
            for metric in (
                "final_balance",
                "mean_trust_ratio",
                "mean_return_ratio",
                "return_over_sent",
                "mean_sent",
                "mean_returned",
            ):
                values = cell[metric].to_numpy(dtype=float)
                ci_low, ci_high = confidence_interval(values)
                records.append(
                    {
                        "history": history,
                        "history_label": HISTORY_LABELS[history],
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "metric": metric,
                        "n": len(values),
                        "mean": values.mean(),
                        "sd": values.std(ddof=1),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )
    return records


def contrasts(dataframe):
    records = []
    metrics = ["final_balance", "mean_trust_ratio", "mean_return_ratio"]
    for metric in metrics:
        for condition in CONDITION_ORDER:
            records.append(
                paired_record(
                    dataframe,
                    metric,
                    f"history == 'partner_dossier' and condition == '{condition}'",
                    f"history == 'own_only' and condition == '{condition}'",
                    f"Partner dossier − private memory | {CONDITION_LABELS[condition]}",
                    f"visibility_{metric}",
                )
            )
        for history in HISTORY_ORDER:
            for left, right in (
                ("game_myth", "game"),
                ("myth_game", "game"),
                ("myth_game", "game_myth"),
            ):
                records.append(
                    paired_record(
                        dataframe,
                        metric,
                        f"history == '{history}' and condition == '{left}'",
                        f"history == '{history}' and condition == '{right}'",
                        f"{CONDITION_LABELS[left]} − {CONDITION_LABELS[right]} | {HISTORY_LABELS[history]}",
                        f"task_order_{history}_{metric}",
                    )
                )

        pivot = dataframe.pivot(
            index="replicate_id",
            columns=["history", "condition"],
            values=metric,
        )
        game_visibility = (
            pivot[("partner_dossier", "game")] - pivot[("own_only", "game")]
        )
        for condition in ("game_myth", "myth_game"):
            myth_visibility = (
                pivot[("partner_dossier", condition)]
                - pivot[("own_only", condition)]
            )
            differences = (myth_visibility - game_visibility).to_numpy(dtype=float)
            ci_low, ci_high = confidence_interval(differences)
            records.append(
                {
                    "metric": metric,
                    "contrast": (
                        f"Visibility interaction: {CONDITION_LABELS[condition]} − Game only"
                    ),
                    "family": f"interaction_{metric}",
                    "n_pairs": len(differences),
                    "estimate": differences.mean(),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": stats.ttest_1samp(differences, 0).pvalue,
                }
            )

    for family in {record["family"] for record in records}:
        indices = [i for i, record in enumerate(records) if record["family"] == family]
        adjusted = holm_adjust([records[i]["p_value"] for i in indices])
        for index, adjusted_p in zip(indices, adjusted):
            records[index]["holm_p_within_family"] = adjusted_p
    return records


def plot_balances(dataframe, output_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns

    order = [CONDITION_LABELS[value] for value in CONDITION_ORDER]
    hue_order = [HISTORY_LABELS[value] for value in HISTORY_ORDER]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sns.boxplot(
        data=dataframe,
        x="condition_label",
        y="final_balance",
        hue="history_label",
        order=order,
        hue_order=hue_order,
        palette=[HISTORY_COLORS[value] for value in HISTORY_ORDER],
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=dataframe,
        x="condition_label",
        y="final_balance",
        hue="history_label",
        order=order,
        hue_order=hue_order,
        dodge=True,
        palette=["#263238", "#263238"],
        alpha=0.75,
        size=5,
        ax=ax,
        legend=False,
    )
    ax.set_title("History visibility × task order (GPT-5 Nano)", fontweight="bold")
    ax.set_xlabel("Task order")
    ax.set_ylabel("Average final balance per agent")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Decision-time history")
    fig.tight_layout()
    fig.savefig(output_dir / "final_balance_history_by_task.png", dpi=300)
    plt.close(fig)


def plot_visibility_effect(dataframe, output_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    x = np.arange(len(CONDITION_ORDER))
    for replicate_id in range(5):
        run = dataframe[dataframe["replicate_id"] == replicate_id].pivot(
            index="condition", columns="history", values="final_balance"
        )
        values = [
            run.loc[condition, "partner_dossier"] - run.loc[condition, "own_only"]
            for condition in CONDITION_ORDER
        ]
        ax.plot(x, values, color="#607d8b", alpha=0.35, marker="o")
    means = []
    cis = []
    for condition in CONDITION_ORDER:
        paired = dataframe[dataframe["condition"] == condition].pivot(
            index="replicate_id", columns="history", values="final_balance"
        )
        differences = paired["partner_dossier"] - paired["own_only"]
        means.append(differences.mean())
        low, high = confidence_interval(differences)
        cis.append((differences.mean() - low, high - differences.mean()))
    ax.errorbar(
        x,
        means,
        yerr=np.asarray(cis).T,
        color="#9c2f1f",
        linewidth=2.5,
        marker="o",
        capsize=5,
        label="Mean and 95% CI",
    )
    ax.axhline(0, color="#263238", linestyle="--", linewidth=1)
    ax.set_xticks(x, [CONDITION_LABELS[value] for value in CONDITION_ORDER])
    ax.set_title("Effect of showing the current partner's recent games", fontweight="bold")
    ax.set_xlabel("Task order")
    ax.set_ylabel("Partner dossier − private memory\n(final balance per agent)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "partner_dossier_effect.png", dpi=300)
    plt.close(fig)


def plot_trajectories(trajectory_dataframe, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    for ax, history in zip(axes, HISTORY_ORDER):
        history_data = trajectory_dataframe[
            trajectory_dataframe["history"] == history
        ]
        for condition in CONDITION_ORDER:
            cell = history_data[history_data["condition"] == condition]
            means = cell.groupby("round", as_index=False)["trust_ratio"].mean()
            ax.plot(
                means["round"],
                means["trust_ratio"],
                marker="o",
                linewidth=2.5,
                color=CONDITION_COLORS[condition],
                label=CONDITION_LABELS[condition],
            )
        ax.set_title(HISTORY_LABELS[history], fontweight="bold")
        ax.set_xlabel("Round")
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, linestyle="--")
    axes[0].set_ylabel("Mean proportion sent")
    axes[1].legend(title="Task order", loc="lower right")
    fig.suptitle("Trust trajectories by decision-time history", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "trust_trajectories_history_by_task.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    configure_matplotlib()
    import pandas as pd
    import seaborn as sns

    sns.set_style("whitegrid")
    args.out.mkdir(parents=True, exist_ok=True)
    rows, trajectory_rows, usage, attempts, retries = load_factorial()
    dataframe = pd.DataFrame(rows).sort_values(
        ["history", "condition", "replicate_id"]
    )
    trajectory_dataframe = pd.DataFrame(trajectory_rows)
    summary = pd.DataFrame(summaries(dataframe))
    contrast_table = pd.DataFrame(contrasts(dataframe))

    estimated_cost = (
        usage["input_tokens"] / 1_000_000 * 0.05
        + (usage["output_tokens"] + usage["reasoning_tokens"])
        / 1_000_000
        * 0.40
    )
    cost = pd.DataFrame(
        [
            {
                **usage,
                "total_attempts": attempts,
                "recovered_retries": retries,
                "assumed_input_usd_per_million": 0.05,
                "assumed_output_usd_per_million": 0.40,
                "estimated_list_price_usd": estimated_cost,
            }
        ]
    )

    dataframe.to_csv(args.out / "run_metrics.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    contrast_table.to_csv(args.out / "paired_contrasts.csv", index=False)
    cost.to_csv(args.out / "token_cost.csv", index=False)
    plot_balances(dataframe, args.out)
    plot_visibility_effect(dataframe, args.out)
    plot_trajectories(trajectory_dataframe, args.out)

    print(
        summary[
            summary["metric"].isin(
                ["final_balance", "mean_trust_ratio", "mean_return_ratio"]
            )
        ].to_string(index=False)
    )
    print("\nFinal-balance contrasts:")
    print(
        contrast_table[contrast_table["metric"] == "final_balance"].to_string(
            index=False
        )
    )
    print(f"\nAttempts: {attempts}; recovered retries: {retries}")
    print(f"Estimated list-price cost under recorded rates: ${estimated_cost:.4f}")
    print(f"Saved outputs to {args.out}")


if __name__ == "__main__":
    main()
