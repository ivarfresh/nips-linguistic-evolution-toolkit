#!/usr/bin/env python3
"""Reference-style cumulative balance plot for myth-causal prompt-arm runs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

_MPL_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "myth_causal_mplconfig"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
_XDG_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "myth_causal_xdg_cache"
_XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CLEAN_ROOTS = [
    PROJECT_ROOT / "data/json/myth_causal_confirm_claude_fixed_prompt/claude-sonnet-4.5",
    PROJECT_ROOT / "data/json/myth_causal_confirm_claude_fixed_prompt_topup/claude-sonnet-4.5",
]
DEFAULT_NOISE_ROOTS = [
    PROJECT_ROOT
    / "data/json/noise_experiments/myth_causal_negative2_informed_claude_pilot"
    / "myth_causal_noise_negative2_informed_claude_pilot/claude-sonnet-4.5",
    PROJECT_ROOT
    / "data/json/noise_experiments/myth_causal_negative2_informed_claude_topup"
    / "myth_causal_noise_negative2_informed_claude_topup/claude-sonnet-4.5",
]

PROMPT_ORDER = ["Game Only", "Myth Control", "Myth-Game Directive", "Normative Directive"]
PROMPT_LABELS = {
    "game_only": "Game Only",
    "myth_control": "Myth Control",
    "myth_game_directive": "Myth-Game Directive",
    "myth_game_normative": "Normative Directive",
}
NOISE_ORDER = ["No Noise", "Informed Negative U(0,2)"]
PALETTE = {
    "No Noise": "#4C72B0",
    "Informed Negative U(0,2)": "#55A868",
}


def present_prompt_order(df: pd.DataFrame) -> list[str]:
    present = set(df["Prompt Arm"].dropna().unique())
    return [prompt for prompt in PROMPT_ORDER if prompt in present]


def is_primary_json(path: Path) -> bool:
    return (
        path.suffix == ".json"
        and not path.name.endswith(".results.json")
        and not path.name.endswith(".checkpoint.json")
        and not path.name.endswith(".error.json")
    )


def prompt_arm_label(data: dict) -> str | None:
    task_order = "_".join(data.get("task_order") or [])
    if task_order == "game":
        return "Game Only"

    arm = data.get("run_metadata", {}).get("myth_prompt_arm_id")
    return PROMPT_LABELS.get(arm)


def game_rounds(data: dict) -> list[dict]:
    return [row for row in data.get("conversation_history", []) if row.get("sent") is not None]


def balance_at_round(data: dict, target_round: int | None) -> tuple[int, float] | None:
    rounds = game_rounds(data)
    if not rounds:
        return None

    if target_round is None:
        row = rounds[-1]
    else:
        matches = [row for row in rounds if int(row.get("round", -1)) == target_round]
        if not matches:
            return None
        row = matches[-1]

    balances = row.get("balances", {})
    try:
        mean_balance = (float(balances["Agent_1"]) + float(balances["Agent_2"])) / 2.0
    except (KeyError, TypeError, ValueError):
        return None
    return int(row["round"]), mean_balance


def collect_records(roots: list[Path], noise_condition: str, target_round: int | None) -> list[dict]:
    rows: list[dict] = []
    for root in roots:
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*.json") if is_primary_json(p)):
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)

            prompt_label = prompt_arm_label(data)
            if prompt_label is None:
                continue

            balance = balance_at_round(data, target_round)
            if balance is None:
                continue

            round_number, mean_balance = balance
            rows.append(
                {
                    "Prompt Arm": prompt_label,
                    "Noise Condition": noise_condition,
                    "round": round_number,
                    "mean_cumulative_balance": mean_balance,
                    "path": str(path.relative_to(PROJECT_ROOT)),
                }
            )
    return rows


def add_significance_brackets(
    ax,
    df: pd.DataFrame,
    y_top: float,
    noise_order: list[str],
    prompt_order: list[str],
) -> None:
    if len(noise_order) != 2:
        return

    offset = 0.2
    y_range = y_top
    bracket_step = y_range * 0.035
    text_pad = y_range * 0.008

    for x_index, prompt in enumerate(prompt_order):
        values_a = df[
            (df["Prompt Arm"] == prompt) & (df["Noise Condition"] == noise_order[0])
        ]["mean_cumulative_balance"].to_numpy()
        values_b = df[
            (df["Prompt Arm"] == prompt) & (df["Noise Condition"] == noise_order[1])
        ]["mean_cumulative_balance"].to_numpy()
        if len(values_a) < 2 or len(values_b) < 2:
            continue

        result = mannwhitneyu(values_a, values_b, alternative="two-sided")
        if result.pvalue >= 0.05:
            continue

        stars = "***" if result.pvalue < 0.001 else "**" if result.pvalue < 0.01 else "*"
        y = max(values_a.max(), values_b.max()) + bracket_step
        y = min(y, y_top - bracket_step * 1.8)
        x1 = x_index - offset
        x2 = x_index + offset
        ax.plot([x1, x1, x2, x2], [y, y + bracket_step, y + bracket_step, y], color="black", linewidth=1.5)
        ax.text(
            (x1 + x2) / 2,
            y + bracket_step + text_pad,
            stars,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color="black",
        )


def plot(df: pd.DataFrame, output_path: Path, title: str, note: str | None = None) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    noise_order = [label for label in NOISE_ORDER if label in set(df["Noise Condition"])]
    prompt_order = present_prompt_order(df)

    fig, ax = plt.subplots(figsize=(14, 9))
    sns.boxplot(
        data=df,
        x="Prompt Arm",
        y="mean_cumulative_balance",
        hue="Noise Condition",
        order=prompt_order,
        hue_order=noise_order,
        palette=PALETTE,
        width=0.75,
        linewidth=1.5,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="Prompt Arm",
        y="mean_cumulative_balance",
        hue="Noise Condition",
        order=prompt_order,
        hue_order=noise_order,
        palette=PALETTE,
        dodge=True,
        jitter=0.12,
        size=5.5,
        alpha=0.55,
        linewidth=0,
        ax=ax,
    )

    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if label in noise_order and label not in unique:
            unique[label] = handle
    ax.legend(
        unique.values(),
        unique.keys(),
        title="Noise Condition",
        loc="upper right",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
    )

    y_max = float(df["mean_cumulative_balance"].max())
    y_top = max(100.0, math.ceil((y_max + 6.0) / 10.0) * 10.0)
    ax.set_ylim(0, y_top)
    add_significance_brackets(ax, df, y_top, noise_order, prompt_order)

    fig.suptitle(title, fontsize=22, fontweight="bold", y=0.985)
    if note:
        fig.text(
            0.5,
            0.945,
            note,
            ha="center",
            va="bottom",
            fontsize=13,
        )
    ax.set_xlabel("Prompt Arm", fontsize=18)
    ax.set_ylabel("Cumulative Balance (avg. of both agents)", fontsize=18)
    ax.grid(True, axis="y", alpha=0.35)
    ax.grid(False, axis="x")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.965 if note else 0.975])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-round", type=int, default=None, help="Round number to plot; defaults to final round.")
    parser.add_argument(
        "--clean-root",
        action="append",
        type=Path,
        default=None,
        help="Clean/no-noise JSON root. Can be passed multiple times; defaults to the original pilot roots.",
    )
    parser.add_argument(
        "--noise-root",
        action="append",
        type=Path,
        default=None,
        help="Noisy JSON root. Can be passed multiple times; defaults to the original informed-negative-noise roots.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/plots/myth_causal_reference_style",
        help="Output directory relative to the repo root.",
    )
    parser.add_argument("--note", help="Optional subtitle note.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clean_roots = args.clean_root if args.clean_root is not None else DEFAULT_CLEAN_ROOTS
    noise_roots = args.noise_root if args.noise_root is not None else DEFAULT_NOISE_ROOTS

    rows = []
    rows.extend(collect_records(clean_roots, "No Noise", args.target_round))
    rows.extend(collect_records(noise_roots, "Informed Negative U(0,2)", args.target_round))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No matching JSON runs found.")

    rounds = sorted(df["round"].unique())
    round_label = str(rounds[0]) if len(rounds) == 1 else f"{rounds[0]}-{rounds[-1]}"
    title = f"Cumulative Balance at Round {round_label}: claude-sonnet-4.5"

    out = PROJECT_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cumulative_balance_reference_style_data.csv", index=False)
    counts = df.groupby(["Prompt Arm", "Noise Condition"], as_index=False).size().rename(columns={"size": "n"})
    counts.to_csv(out / "cumulative_balance_reference_style_counts.csv", index=False)
    plot(df, out / "cumulative_balance_reference_style.png", title, args.note)

    print(f"wrote {out / 'cumulative_balance_reference_style.png'}")
    print(counts.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
