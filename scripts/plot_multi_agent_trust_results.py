#!/usr/bin/env python3
"""Plot multi-agent trust-game overview results.

This mirrors the two-agent myth-causal overview, but aggregates each
multi-agent round over all dyads and all agent balances.
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
    condition: str
    replicate: int | None
    num_agents: int
    rounds: list[int]
    sent: list[float]
    returned: list[float]
    received: list[float]
    return_ratio: list[float]
    mean_balance: list[float]
    final_min_balance: float
    final_max_balance: float
    final_mean_balance: float
    final_total_balance: float

    @property
    def avg_sent(self) -> float:
        return float(np.mean(self.sent))

    @property
    def avg_returned(self) -> float:
        return float(np.mean(self.returned))

    @property
    def avg_return_ratio(self) -> float:
        return float(np.mean(self.return_ratio))


def is_primary_json(path: Path) -> bool:
    return (
        path.suffix == ".json"
        and not path.name.endswith(".results.json")
        and not path.name.endswith(".checkpoint.json")
        and not path.name.endswith(".error.json")
    )


def primary_json_files(input_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for input_dir in input_dirs:
        files.extend(p for p in input_dir.rglob("*.json") if is_primary_json(p))
    return sorted(files)


def condition_for(data: dict) -> str | None:
    task_order = "_".join(data.get("task_order") or [])
    if task_order == "game":
        return "game_only"
    return data.get("run_metadata", {}).get("myth_prompt_arm_id")


def mean_balance(row: dict) -> float:
    balances = row.get("balances") or {}
    values = [float(v) for v in balances.values()]
    if not values:
        raise ValueError(f"Missing balances in round {row.get('round')}")
    return float(np.mean(values))


def round_metrics(row: dict) -> dict | None:
    dyads = row.get("dyads") or []
    if dyads:
        sent = [float(d["sent"]) for d in dyads]
        received = [float(d["received"]) for d in dyads]
        returned = [float(d["returned"]) for d in dyads]
        ratios = [ret / rec for ret, rec in zip(returned, received) if rec]
        return {
            "round": int(row["round"]),
            "sent": float(np.mean(sent)),
            "received": float(np.mean(received)),
            "returned": float(np.mean(returned)),
            "return_ratio": float(np.mean(ratios)) if ratios else float("nan"),
            "mean_balance": mean_balance(row),
            "num_dyads": len(dyads),
        }

    if row.get("sent") is not None and row.get("returned") is not None:
        received = float(row["received"])
        returned = float(row["returned"])
        return {
            "round": int(row["round"]),
            "sent": float(row["sent"]),
            "received": received,
            "returned": returned,
            "return_ratio": returned / received if received else float("nan"),
            "mean_balance": mean_balance(row),
            "num_dyads": 1,
        }

    return None


def load_runs(input_dirs: list[Path]) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for path in primary_json_files(input_dirs):
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)

        condition = condition_for(data)
        if condition not in CONDITION_ORDER:
            continue

        rows = []
        for row in data.get("conversation_history", []):
            metric_row = round_metrics(row)
            if metric_row:
                rows.append(metric_row)
        if not rows:
            continue

        final_balances = data.get("conversation_history", [])[-1].get("balances", {})
        final_values = [float(v) for v in final_balances.values()]
        if not final_values:
            continue

        runs.append(
            RunRecord(
                path=path,
                condition=condition,
                replicate=data.get("run_metadata", {}).get("replicate_id"),
                num_agents=int(data.get("run_metadata", {}).get("num_agents") or len(final_values)),
                rounds=[r["round"] for r in rows],
                sent=[r["sent"] for r in rows],
                returned=[r["returned"] for r in rows],
                received=[r["received"] for r in rows],
                return_ratio=[r["return_ratio"] for r in rows],
                mean_balance=[r["mean_balance"] for r in rows],
                final_min_balance=float(np.min(final_values)),
                final_max_balance=float(np.max(final_values)),
                final_mean_balance=float(np.mean(final_values)),
                final_total_balance=float(np.sum(final_values)),
            )
        )
    return runs


def present_condition_order(runs: list[RunRecord] | pd.DataFrame) -> list[str]:
    if isinstance(runs, pd.DataFrame):
        present = set(runs["condition"].dropna().unique())
    else:
        present = {run.condition for run in runs}
    return [condition for condition in CONDITION_ORDER if condition in present]


def trajectory_dataframe(runs: list[RunRecord]) -> pd.DataFrame:
    rows = []
    for run_idx, run in enumerate(runs):
        for i, round_num in enumerate(run.rounds):
            rows.append(
                {
                    "run_idx": run_idx,
                    "condition": run.condition,
                    "condition_label": CONDITION_LABELS[run.condition],
                    "replicate": run.replicate,
                    "round": round_num,
                    "sent": run.sent[i],
                    "returned": run.returned[i],
                    "received": run.received[i],
                    "return_ratio": run.return_ratio[i],
                    "mean_balance": run.mean_balance[i],
                    "path": str(run.path),
                }
            )
    return pd.DataFrame(rows)


def runs_dataframe(runs: list[RunRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "condition": r.condition,
                "condition_label": CONDITION_LABELS[r.condition],
                "replicate": r.replicate,
                "num_agents": r.num_agents,
                "path": str(r.path),
                "avg_sent": r.avg_sent,
                "avg_returned": r.avg_returned,
                "avg_return_ratio": r.avg_return_ratio,
                "final_min_balance": r.final_min_balance,
                "final_max_balance": r.final_max_balance,
                "final_mean_balance": r.final_mean_balance,
                "final_total_balance": r.final_total_balance,
                "escalated_send": any(v > 3 for v in r.sent),
                "escalated_return": any(v > 6 for v in r.returned),
            }
            for r in runs
        ]
    )


def write_tables(runs: list[RunRecord], output_dir: Path) -> pd.DataFrame:
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
            mean_return_ratio=("avg_return_ratio", "mean"),
            mean_final_balance=("final_mean_balance", "mean"),
            sd_final_balance=("final_mean_balance", "std"),
            median_final_balance=("final_mean_balance", "median"),
            mean_final_min_balance=("final_min_balance", "mean"),
            mean_final_max_balance=("final_max_balance", "mean"),
            escalated_send_runs=("escalated_send", "sum"),
            escalated_return_runs=("escalated_return", "sum"),
        )
    )
    order = {condition: i for i, condition in enumerate(CONDITION_ORDER)}
    summary = summary.sort_values("condition", key=lambda s: s.map(order)).reset_index(drop=True)
    summary.to_csv(output_dir / "summary_by_condition.csv", index=False)
    return summary


def plot_trajectory_summary(runs: list[RunRecord], output_dir: Path) -> None:
    out = output_dir / "trajectory_plotting"
    out.mkdir(parents=True, exist_ok=True)
    df = trajectory_dataframe(runs)
    df.to_csv(out / "trajectory_points.csv", index=False)
    condition_order = present_condition_order(df)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    metric_specs = [
        ("sent", "Mean Sent Per Dyad", "$ Sent"),
        ("returned", "Mean Returned Per Dyad", "$ Returned"),
        ("return_ratio", "Mean Return Ratio", "Returned / Received"),
        ("mean_balance", "Mean Cumulative Balance", "Balance per Agent"),
    ]
    for ax, (metric, title, ylabel) in zip(axes.flat, metric_specs):
        sns.lineplot(
            data=df,
            x="round",
            y=metric,
            hue="condition",
            hue_order=condition_order,
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
        ax.legend(handles=handles, labels=[CONDITION_LABELS.get(label, label) for label in labels], loc="best")
    fig.suptitle("8-Agent Trust Game Trajectories", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "trajectory_plotting_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for ax, metric, title, ylabel in [
        (axes[0], "sent", "Sent Trajectories", "$ Sent per Dyad"),
        (axes[1], "returned", "Returned Trajectories", "$ Returned per Dyad"),
    ]:
        sns.lineplot(
            data=df,
            x="round",
            y=metric,
            hue="condition",
            hue_order=condition_order,
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
        ax.legend(handles=handles, labels=[CONDITION_LABELS.get(label, label) for label in labels], loc="best")
    plt.tight_layout()
    plt.savefig(out / "trajectory_plotting_individual_overlay.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_balance_comparison(runs: list[RunRecord], output_dir: Path, model_label: str) -> None:
    out = output_dir / "_balance_comparison" / model_label
    out.mkdir(parents=True, exist_ok=True)
    df = runs_dataframe(runs)
    condition_order = present_condition_order(df)
    condition_labels = [CONDITION_LABELS[c] for c in condition_order]
    df["condition_label"] = pd.Categorical(df["condition_label"], condition_labels, ordered=True)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [
        ("avg_sent", "Mean Sent", "$ Sent per Dyad"),
        ("avg_returned", "Mean Returned", "$ Returned per Dyad"),
        ("final_mean_balance", "Final Mean Balance", "Balance per Agent"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, metrics):
        sns.boxplot(
            data=df,
            x="condition_label",
            y=metric,
            hue="condition_label",
            order=condition_labels,
            hue_order=condition_labels,
            palette={CONDITION_LABELS[c]: PALETTE[c] for c in condition_order},
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=df,
            x="condition_label",
            y=metric,
            order=condition_labels,
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
    fig.suptitle("8-Agent Balance Comparison", fontweight="bold", y=1.04)
    plt.tight_layout()
    plt.savefig(out / "balance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_delta_comparison(runs: list[RunRecord], output_dir: Path, model_label: str) -> None:
    out = output_dir / "_balance_comparison" / model_label
    out.mkdir(parents=True, exist_ok=True)
    df = runs_dataframe(runs)
    condition_order = present_condition_order(df)
    baselines = df[df["condition"] == "game_only"]
    if baselines.empty:
        return
    base_median = {
        "avg_sent": float(baselines["avg_sent"].median()),
        "avg_returned": float(baselines["avg_returned"].median()),
        "final_mean_balance": float(baselines["final_mean_balance"].median()),
    }

    rows = []
    for condition in [c for c in condition_order if c != "game_only"]:
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
    fig.suptitle("8-Agent Delta Comparison: Myth Conditions - Game Only", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "delta_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_slide_style(runs: list[RunRecord], summary: pd.DataFrame, output_dir: Path, title: str, model_label: str) -> None:
    out = output_dir / "slide_style"
    out.mkdir(parents=True, exist_ok=True)
    df = runs_dataframe(runs)
    present = present_condition_order(df)
    labels = [CONDITION_LABELS[c].replace(" ", "\n", 1) for c in present]
    label_map = {c: CONDITION_LABELS[c].replace(" ", "\n", 1) for c in present}
    df["Prompt Arm"] = df["condition"].map(label_map)
    df.to_csv(out / "slide_style_cumulative_balance_data.csv", index=False)
    summary.to_csv(out / "slide_style_summary.csv", index=False)

    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(0.055, 0.925, title, fontsize=31, ha="left", va="top", color="black")

    ax = fig.add_axes([0.055, 0.07, 0.62, 0.66])
    sns.boxplot(
        data=df,
        x="Prompt Arm",
        y="final_mean_balance",
        order=labels,
        color="#4C72B0",
        width=0.50,
        linewidth=1.4,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="Prompt Arm",
        y="final_mean_balance",
        order=labels,
        color="#4C72B0",
        size=5.5,
        alpha=0.55,
        jitter=0.10,
        ax=ax,
    )
    ax.set_title(f"Cumulative Balance at Round 10: {model_label}", fontsize=17, fontweight="bold", pad=8)
    ax.set_xlabel("Prompt Arm", fontsize=14)
    ax.set_ylabel("Cumulative Balance (avg. of 8 agents)", fontsize=14)
    ymax = max(90, float(df["final_mean_balance"].max()) + 10)
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.9)
    ax.grid(False, axis="x")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    table_ax = fig.add_axes([0.69, 0.60, 0.31, 0.33])
    table_ax.axis("off")
    table_rows = [["No noise", "", "Mean final\nbalance", "Median final\nbalance"], ["Prompt Arm", "n", "", ""]]
    for _, row in summary.iterrows():
        table_rows.append(
            [
                str(row["condition_label"]),
                str(int(row["n"])),
                f"{float(row['mean_final_balance']):.1f}",
                f"{float(row['median_final_balance']):.1f}",
            ]
        )
    table = table_ax.table(
        cellText=table_rows,
        cellLoc="left",
        colWidths=[0.39, 0.16, 0.23, 0.23],
        bbox=[0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#C8C8C8")
        cell.set_linewidth(0.7)
        cell.set_facecolor("white")
        cell.PAD = 0.03
        if row_idx == 0 and col_idx == 0:
            cell.set_text_props(weight="bold")
        if col_idx in {1, 2, 3} and row_idx >= 2:
            cell.set_text_props(ha="right")

    output_png = out / "slide_style_cumulative_balance.png"
    fig.savefig(output_png, dpi=200)
    fig.savefig(output_png.with_suffix(".pdf"))
    plt.close(fig)


def plot_slide_style_chart_only(runs: list[RunRecord], output_dir: Path, title: str, model_label: str) -> None:
    out = output_dir / "slide_style"
    out.mkdir(parents=True, exist_ok=True)
    df = runs_dataframe(runs)
    present = present_condition_order(df)
    labels = [CONDITION_LABELS[c].replace(" ", "\n", 1) for c in present]
    label_map = {c: CONDITION_LABELS[c].replace(" ", "\n", 1) for c in present}
    df["Prompt Arm"] = df["condition"].map(label_map)

    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(12, 8), facecolor="white")
    fig.text(0.075, 0.94, title, fontsize=28, ha="left", va="top", color="black")

    ax = fig.add_axes([0.10, 0.10, 0.86, 0.72])
    sns.boxplot(
        data=df,
        x="Prompt Arm",
        y="final_mean_balance",
        order=labels,
        color="#4C72B0",
        width=0.42,
        linewidth=1.4,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="Prompt Arm",
        y="final_mean_balance",
        order=labels,
        color="#4C72B0",
        size=6,
        alpha=0.60,
        jitter=0.10,
        ax=ax,
    )
    ax.set_title(f"Cumulative Balance at Round 10: {model_label}", fontsize=17, fontweight="bold", pad=10)
    ax.set_xlabel("Prompt Arm", fontsize=14)
    ax.set_ylabel("Cumulative Balance (avg. of 8 agents)", fontsize=14)
    ymax = max(90, float(df["final_mean_balance"].max()) + 10)
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.9)
    ax.grid(False, axis="x")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    output_png = out / "slide_style_cumulative_balance_chart_only.png"
    fig.savefig(output_png, dpi=200)
    fig.savefig(output_png.with_suffix(".pdf"))
    plt.close(fig)


def write_index(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.suffix.lower() in {".png", ".csv", ".pdf"}:
            rows.append({"artifact": str(path.relative_to(output_dir)), "path": str(path)})
    with (output_dir / "plot_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["artifact", "path"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-label", default="claude-sonnet-4.5")
    parser.add_argument("--title", default="8-agent Sonnet 4.5 directive vs game-only")
    parser.add_argument("--skip-slide-style", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dirs = [p if p.is_absolute() else REPO_ROOT / p for p in args.input_dir]
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(input_dirs)
    if not runs:
        raise SystemExit("No matching primary JSON runs found.")

    summary = write_tables(runs, output_dir)
    plot_trajectory_summary(runs, output_dir)
    plot_balance_comparison(runs, output_dir, args.model_label)
    plot_delta_comparison(runs, output_dir, args.model_label)
    if not args.skip_slide_style:
        plot_slide_style(runs, summary, output_dir, args.title, args.model_label)
        plot_slide_style_chart_only(runs, output_dir, args.title, args.model_label)
    write_index(output_dir)
    print(f"Generated multi-agent plots for {len(runs)} runs in {output_dir}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
