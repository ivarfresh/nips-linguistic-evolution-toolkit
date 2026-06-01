#!/usr/bin/env python3
"""Plot myth-causal prompt-arm results.

Generates the plot families requested for the Claude myth-causal screen:
- trajectory_plotting summary plots
- trajectory_plotting individual per-run plots
- balance_comparison plots
- delta plots
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analyses._shared import configure_matplotlib
from analyses.trajectory_plotting import plot_numerical_trajectories


configure_matplotlib()


CONDITION_ORDER = ["game_only", "myth_control", "myth_game_directive", "myth_game_normative"]
CONDITION_LABELS = {
    "game_only": "Game Only",
    "myth_control": "Myth Control",
    "myth_game_directive": "Myth-Game Directive",
    "myth_game_normative": "Normative Directive",
}
PALETTE = {
    "game_only": "#4C72B0",
    "myth_control": "#55A868",
    "myth_game_directive": "#C44E52",
    "myth_game_normative": "#8172B3",
}


@dataclass
class RunRecord:
    path: Path
    batch: str
    condition: str
    replicate: int | None
    rounds: list[int]
    sent: list[float]
    returned: list[float]
    received: list[float]
    avg_sent: float
    avg_returned: float
    avg_return_ratio: float
    final_agent_1: float
    final_agent_2: float
    final_mean_balance: float
    final_total_balance: float
    conversation_history: list[dict]


def primary_json_files(input_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for input_dir in input_dirs:
        files.extend(
            p for p in input_dir.rglob("*.json")
            if not p.name.endswith((".results.json", ".checkpoint.json", ".error.json"))
        )
    return sorted(files)


def condition_for(data: dict) -> str:
    order = "_".join(data.get("task_order") or [])
    if order == "game":
        return "game_only"
    arm = data.get("run_metadata", {}).get("myth_prompt_arm_id")
    if arm:
        return arm
    return "unknown"


def load_runs(input_dirs: list[Path]) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for path in primary_json_files(input_dirs):
        with path.open() as f:
            data = json.load(f)
        condition = condition_for(data)
        if condition not in CONDITION_ORDER:
            continue

        history = [
            r for r in data.get("conversation_history", [])
            if r.get("sent") is not None
        ]
        if not history:
            continue

        sent = [float(r["sent"]) for r in history]
        returned = [float(r["returned"]) for r in history]
        received = [float(r["received"]) for r in history]
        ratios = [ret / rec for ret, rec in zip(returned, received) if rec]
        final_balances = history[-1]["balances"]
        batch = path.parts[path.parts.index("data") + 2] if "data" in path.parts else path.parent.name
        runs.append(
            RunRecord(
                path=path,
                batch=batch,
                condition=condition,
                replicate=data.get("run_metadata", {}).get("replicate_id"),
                rounds=[int(r["round"]) for r in history],
                sent=sent,
                returned=returned,
                received=received,
                avg_sent=float(np.mean(sent)),
                avg_returned=float(np.mean(returned)),
                avg_return_ratio=float(np.mean(ratios)) if ratios else float("nan"),
                final_agent_1=float(final_balances["Agent_1"]),
                final_agent_2=float(final_balances["Agent_2"]),
                final_mean_balance=float((final_balances["Agent_1"] + final_balances["Agent_2"]) / 2),
                final_total_balance=float(final_balances["Agent_1"] + final_balances["Agent_2"]),
                conversation_history=history,
            )
        )
    return runs


def runs_dataframe(runs: list[RunRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "batch": r.batch,
                "condition": r.condition,
                "condition_label": CONDITION_LABELS[r.condition],
                "replicate": r.replicate,
                "path": str(r.path),
                "avg_sent": r.avg_sent,
                "avg_returned": r.avg_returned,
                "avg_return_ratio": r.avg_return_ratio,
                "final_agent_1": r.final_agent_1,
                "final_agent_2": r.final_agent_2,
                "final_mean_balance": r.final_mean_balance,
                "final_total_balance": r.final_total_balance,
                "escalated_send": any(v > 3 for v in r.sent),
                "escalated_return": any(v > 6 for v in r.returned),
            }
            for r in runs
        ]
    )


def write_tables(runs: list[RunRecord], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = runs_dataframe(runs)
    df.to_csv(output_dir / "run_metrics.csv", index=False)
    summary = (
        df.groupby(["condition", "condition_label"], as_index=False)
        .agg(
            n=("path", "count"),
            mean_avg_sent=("avg_sent", "mean"),
            sd_avg_sent=("avg_sent", "std"),
            mean_avg_returned=("avg_returned", "mean"),
            sd_avg_returned=("avg_returned", "std"),
            mean_final_balance=("final_mean_balance", "mean"),
            sd_final_balance=("final_mean_balance", "std"),
            escalated_send_runs=("escalated_send", "sum"),
            escalated_return_runs=("escalated_return", "sum"),
        )
    )
    summary.to_csv(output_dir / "summary_by_condition.csv", index=False)


def plot_trajectory_summary(runs: list[RunRecord], output_dir: Path) -> None:
    out = output_dir / "trajectory_plotting"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_idx, run in enumerate(runs):
        for i, round_num in enumerate(run.rounds):
            rows.append(
                {
                    "run_idx": run_idx,
                    "condition": run.condition,
                    "condition_label": CONDITION_LABELS[run.condition],
                    "round": round_num,
                    "sent": run.sent[i],
                    "returned": run.returned[i],
                    "received": run.received[i],
                    "return_ratio": run.returned[i] / run.received[i] if run.received[i] else np.nan,
                    "mean_balance": (
                        run.conversation_history[i]["balances"]["Agent_1"]
                        + run.conversation_history[i]["balances"]["Agent_2"]
                    ) / 2,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "trajectory_points.csv", index=False)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    metric_specs = [
        ("sent", "Investor Sent", "$ Sent"),
        ("returned", "Trustee Returned", "$ Returned"),
        ("return_ratio", "Return Ratio", "Returned / Received"),
        ("mean_balance", "Mean Cumulative Balance", "Mean Balance"),
    ]
    for ax, (metric, title, ylabel) in zip(axes.flat, metric_specs):
        sns.lineplot(
            data=df,
            x="round",
            y=metric,
            hue="condition",
            hue_order=CONDITION_ORDER,
            palette=PALETTE,
            estimator="mean",
            errorbar=("se", 1),
            marker="o",
            ax=ax,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(df["round"].unique()))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=[CONDITION_LABELS.get(l, l) for l in labels], loc="best")
    fig.suptitle("Trajectory Plotting: Myth-Causal Prompt Arms", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "trajectory_plotting_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Individual spaghetti plot per metric for quick run-level inspection.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for ax, metric, title, ylabel in [
        (axes[0], "sent", "Sent Trajectories", "$ Sent"),
        (axes[1], "returned", "Returned Trajectories", "$ Returned"),
    ]:
        sns.lineplot(
            data=df,
            x="round",
            y=metric,
            hue="condition",
            hue_order=CONDITION_ORDER,
            units="run_idx",
            estimator=None,
            palette=PALETTE,
            alpha=0.45,
            linewidth=1.5,
            ax=ax,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(df["round"].unique()))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=[CONDITION_LABELS.get(l, l) for l in labels], loc="best")
    plt.tight_layout()
    plt.savefig(out / "trajectory_plotting_individual_overlay.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_individual_trajectories(runs: list[RunRecord], output_dir: Path) -> None:
    out = output_dir / "trajectory_plotting_individual"
    for run in runs:
        stem = run.path.stem
        run_dir = out / run.condition / stem
        run_dir.mkdir(parents=True, exist_ok=True)
        plot_numerical_trajectories(
            run.conversation_history,
            save_path=str(run_dir / "trajectory_1_numerical.png"),
            title=f"{CONDITION_LABELS[run.condition]}: {stem}",
        )


def plot_balance_comparison(runs: list[RunRecord], output_dir: Path, note: str | None = None) -> None:
    out = output_dir / "_balance_comparison" / "claude-sonnet-4.5"
    out.mkdir(parents=True, exist_ok=True)
    df = runs_dataframe(runs)
    df["condition_label"] = pd.Categorical(
        df["condition_label"],
        [CONDITION_LABELS[c] for c in CONDITION_ORDER],
        ordered=True,
    )

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [
        ("avg_sent", "Mean Sent", "$ Sent"),
        ("avg_returned", "Mean Returned", "$ Returned"),
        ("final_mean_balance", "Final Mean Balance", "Balance per Agent"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, metrics):
        sns.boxplot(
            data=df,
            x="condition_label",
            y=metric,
            hue="condition_label",
            order=[CONDITION_LABELS[c] for c in CONDITION_ORDER],
            hue_order=[CONDITION_LABELS[c] for c in CONDITION_ORDER],
            palette={CONDITION_LABELS[c]: PALETTE[c] for c in CONDITION_ORDER},
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=df,
            x="condition_label",
            y=metric,
            order=[CONDITION_LABELS[c] for c in CONDITION_ORDER],
            color="black",
            size=4,
            alpha=0.65,
            jitter=0.16,
            ax=ax,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Balance Comparison: Myth-Causal Prompt Arms", fontweight="bold", y=1.04)
    if note:
        fig.text(0.5, 0.965, note, ha="center", va="center", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94 if note else 1])
    plt.savefig(out / "balance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_delta_comparison(runs: list[RunRecord], output_dir: Path) -> None:
    out = output_dir / "_balance_comparison" / "claude-sonnet-4.5"
    out.mkdir(parents=True, exist_ok=True)
    df = runs_dataframe(runs)
    baselines = df[df["condition"] == "game_only"]
    base_median = {
        "avg_sent": float(baselines["avg_sent"].median()),
        "avg_returned": float(baselines["avg_returned"].median()),
        "final_mean_balance": float(baselines["final_mean_balance"].median()),
    }

    rows = []
    for condition in [c for c in CONDITION_ORDER if c != "game_only"]:
        subset = df[df["condition"] == condition]
        for metric, label in [
            ("avg_sent", "Mean Sent"),
            ("avg_returned", "Mean Returned"),
            ("final_mean_balance", "Final Mean Balance"),
        ]:
            rows.append(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "metric": metric,
                    "metric_label": label,
                    "delta": float(subset[metric].median() - base_median[metric]),
                }
            )
    delta_df = pd.DataFrame(rows)
    delta_df.to_csv(out / "deltas.csv", index=False)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (metric, label) in zip(
        axes,
        [("avg_sent", "Mean Sent"), ("avg_returned", "Mean Returned"), ("final_mean_balance", "Final Mean Balance")],
    ):
        sub = delta_df[delta_df["metric"] == metric]
        bars = ax.bar(
            sub["condition_label"],
            sub["delta"],
            color=[PALETTE[c] for c in sub["condition"]],
            edgecolor="black",
            linewidth=0.6,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel("Median Delta vs Game Only")
        ax.tick_params(axis="x", rotation=20)
        for bar in bars:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (0.03 if val >= 0 else -0.03),
                f"{val:.2f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=9,
                fontweight="bold",
            )
    fig.suptitle("Delta Comparison: Myth Conditions − Game Only", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "delta_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def write_index(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.suffix.lower() in {".png", ".csv"}:
            rows.append({"artifact": str(path.relative_to(output_dir)), "path": str(path)})
    with (output_dir / "plot_index.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["artifact", "path"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True, help="Input JSON directory; repeatable.")
    parser.add_argument("--output-dir", required=True, help="Output plot directory.")
    parser.add_argument("--skip-individual", action="store_true", help="Skip per-run trajectory plots.")
    parser.add_argument("--balance-note", help="Optional subtitle note for the balance comparison plot.")
    args = parser.parse_args()

    input_dirs = [Path(p) for p in args.input_dir]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(input_dirs)
    if not runs:
        raise SystemExit("No matching primary JSON runs found.")

    write_tables(runs, output_dir)
    plot_trajectory_summary(runs, output_dir)
    if not args.skip_individual:
        plot_individual_trajectories(runs, output_dir)
    plot_balance_comparison(runs, output_dir, args.balance_note)
    plot_delta_comparison(runs, output_dir)
    write_index(output_dir)
    print(f"Generated plots for {len(runs)} runs in {output_dir}")


if __name__ == "__main__":
    main()
