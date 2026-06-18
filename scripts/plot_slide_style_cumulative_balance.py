#!/usr/bin/env python3
"""Create a slide-style cumulative balance overview plot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_MPL_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "slide_style_mplconfig"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_LABELS = {
    "game_only": "Game Only",
    "myth_control": "Myth Control",
    "myth_game_directive": "Myth-Game\nDirective",
    "myth_game_normative": "Normative\nDirective",
}
PROMPT_ORDER = ["game_only", "myth_control", "myth_game_directive", "myth_game_normative"]


def is_primary_json(path: Path) -> bool:
    return (
        path.suffix == ".json"
        and not path.name.endswith(".results.json")
        and not path.name.endswith(".checkpoint.json")
        and not path.name.endswith(".error.json")
    )


def condition_for(data: dict) -> str | None:
    task_order = "_".join(data.get("task_order") or [])
    if task_order == "game":
        return "game_only"
    return data.get("run_metadata", {}).get("myth_prompt_arm_id")


def collect_rows(input_dir: Path, target_round: int) -> pd.DataFrame:
    rows = []
    for path in sorted(p for p in input_dir.rglob("*.json") if is_primary_json(p)):
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)

        condition = condition_for(data)
        if condition not in PROMPT_LABELS:
            continue

        history = [
            row
            for row in data.get("conversation_history", [])
            if row.get("sent") is not None and row.get("balances")
        ]
        matches = [row for row in history if int(row.get("round", -1)) == target_round]
        if not matches:
            continue
        row = matches[-1]
        balances = row["balances"]
        mean_balance = (float(balances["Agent_1"]) + float(balances["Agent_2"])) / 2.0
        rows.append(
            {
                "condition": condition,
                "Prompt Arm": PROMPT_LABELS[condition],
                "Noise Condition": "No Noise",
                "round": target_round,
                "mean_cumulative_balance": mean_balance,
                "path": str(path),
            }
        )

    return pd.DataFrame(rows)


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def write_summary_table(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    summary = (
        df.groupby(["condition", "Prompt Arm"], as_index=False)
        .agg(
            n=("path", "count"),
            mean_final_balance=("mean_cumulative_balance", "mean"),
            median_final_balance=("mean_cumulative_balance", "median"),
        )
    )
    order = {condition: i for i, condition in enumerate(PROMPT_ORDER)}
    summary = summary.sort_values("condition", key=lambda s: s.map(order)).reset_index(drop=True)
    summary.to_csv(output_path, index=False)
    return summary


def add_summary_table(ax: plt.Axes, summary: pd.DataFrame) -> None:
    ax.axis("off")
    rows = [
        ["No noise", "", "Mean final\nbalance", "Median final\nbalance"],
        ["Prompt Arm", "n", "", ""],
    ]
    for _, row in summary.iterrows():
        rows.append(
            [
                str(row["Prompt Arm"]).replace("\n", "-"),
                str(int(row["n"])),
                format_number(float(row["mean_final_balance"])),
                format_number(float(row["median_final_balance"])),
            ]
        )

    table = ax.table(
        cellText=rows,
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
        if row_idx in {0, 1}:
            cell.set_text_props(va="center")


def plot(df: pd.DataFrame, summary: pd.DataFrame, output_png: Path, title: str, round_number: int) -> None:
    present = [condition for condition in PROMPT_ORDER if condition in set(df["condition"])]
    x_order = [PROMPT_LABELS[condition] for condition in present]

    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(0.055, 0.925, title, fontsize=31, ha="left", va="top", color="black")

    ax = fig.add_axes([0.055, 0.07, 0.62, 0.66])
    sns.boxplot(
        data=df,
        x="Prompt Arm",
        y="mean_cumulative_balance",
        order=x_order,
        color="#4C72B0",
        width=0.50,
        linewidth=1.4,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="Prompt Arm",
        y="mean_cumulative_balance",
        order=x_order,
        color="#4C72B0",
        size=5.5,
        alpha=0.55,
        jitter=0.10,
        ax=ax,
    )
    ax.set_title("Cumulative Balance at Round 10: claude-sonnet-4.5", fontsize=17, fontweight="bold", pad=8)
    ax.set_xlabel("Prompt Arm", fontsize=14)
    ax.set_ylabel("Cumulative Balance (avg. of both agents)", fontsize=14)
    ax.set_ylim(0, 90)
    ax.set_yticks(range(0, 91, 10))
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.9)
    ax.grid(False, axis="x")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    table_ax = fig.add_axes([0.69, 0.60, 0.31, 0.33])
    add_summary_table(table_ax, summary)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    fig.savefig(output_png.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--title", default="Sonnet 4.5 directive vs normative prompts")
    parser.add_argument("--target-round", default=10, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir if args.input_dir.is_absolute() else PROJECT_ROOT / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_rows(input_dir, args.target_round)
    if df.empty:
        raise SystemExit("No matching run JSON files found.")

    data_path = output_dir / "slide_style_cumulative_balance_data.csv"
    summary_path = output_dir / "slide_style_summary.csv"
    df.to_csv(data_path, index=False)
    summary = write_summary_table(df, summary_path)

    output_png = output_dir / "slide_style_cumulative_balance.png"
    plot(df, summary, output_png, args.title, args.target_round)
    print(f"wrote {output_png}")
    print(f"wrote {output_png.with_suffix('.pdf')}")
    print(f"wrote {summary_path}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
