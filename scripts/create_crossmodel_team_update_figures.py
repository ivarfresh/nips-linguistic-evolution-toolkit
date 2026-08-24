#!/usr/bin/env python3
"""Create slide-ready figures summarizing the August cross-model experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyses._shared import configure_matplotlib


DEFAULT_OUTPUT = Path("docs/figures/crossmodel_team_update_20260824")

GREEN = "#66c2a5"
ORANGE = "#fc8d62"
BLUE = "#8da0cb"
PINK = "#e78ac3"
DARK = "#37474f"
LIGHT = "#d9dee2"

CONDITIONS = ["game", "game_myth", "myth_game"]
CONDITION_LABELS = ["Game only", "Game → Myth", "Myth → Game"]


def finish(fig, output: Path) -> None:
    fig.patch.set_edgecolor("#111111")
    fig.patch.set_linewidth(1.4)
    fig.savefig(output, dpi=300, facecolor="white", edgecolor="#111111")


def errorbar(ax, x, estimate, low, high, *, color, marker="o", label=None):
    lower = np.asarray(estimate) - np.asarray(low)
    upper = np.asarray(high) - np.asarray(estimate)
    ax.errorbar(
        x,
        estimate,
        yerr=np.vstack([lower, upper]),
        color=color,
        marker=marker,
        markersize=7,
        linewidth=2.2,
        capsize=5,
        label=label,
    )


def task_order(output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    gpt = pd.read_csv(
        "docs/figures/crossmodel_signed_gpt_n5_20260821/summary.csv"
    )
    gemini = pd.read_csv(
        "docs/figures/gemini37_flash_task_order_n3_20260823/summary.csv"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    fig.suptitle(
        "Task order generalizes to GPT; Gemini 3.7 is ceiling-limited",
        fontsize=15,
        fontweight="bold",
    )
    x = np.arange(3)
    for ax, metric, title, ylabel, ylim in [
        (axes[0], "final_balance", "Final cumulative balance", "Average per agent", (59, 77)),
        (axes[1], "mean_trust_ratio", "Proportion sent", "Mean trust ratio", (.70, 1.02)),
    ]:
        for model, frame, color, marker in [
            ("GPT-5 Nano (n=5)", gpt, GREEN, "o"),
            ("Gemini 3.7 Flash (n=3)", gemini, ORANGE, "s"),
        ]:
            rows = []
            for condition in CONDITIONS:
                rows.append(
                    frame[(frame["condition"] == condition) & (frame["metric"] == metric)].iloc[0]
                )
            estimate = [row["mean"] for row in rows]
            low = [row["ci_low"] for row in rows]
            high = [row["ci_high"] for row in rows]
            errorbar(
                ax,
                x,
                estimate,
                low,
                high,
                color=color,
                marker=marker,
                label=model,
            )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITION_LABELS, rotation=12)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y", alpha=.3)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[1].annotate(
        "Gemini: 360/360 full sends",
        xy=(1, 1),
        xytext=(1, .94),
        ha="center",
        color=ORANGE,
        fontsize=10,
        fontweight="bold",
        arrowprops={"arrowstyle": "-[, widthB=4", "color": ORANGE, "lw": 1.5},
    )
    fig.tight_layout(rect=(0, 0, 1, .95))
    finish(fig, output_dir / "01_task_order_crossmodel.png")
    plt.close(fig)


def defector_spillovers(output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    data = pd.read_csv(
        "docs/figures/defector_myth_game_crossmodel_confirmation_n10_20260821/"
        "model_paired_contrasts.csv"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    fig.suptitle(
        "Hidden defectors do not trigger a broad collapse; cultural imprint is localized",
        fontsize=15,
        fontweight="bold",
    )
    models = ["gpt", "gemini"]
    labels = ["GPT-5 Nano", "Gemini 3.1\nFlash-Lite"]
    colors = [GREEN, ORANGE]
    selected = data[data["metric"] == "standard_send_ratio"].set_index("model_id")
    for i, (model, label, color) in enumerate(zip(models, labels, colors)):
        row = selected.loc[model]
        errorbar(
            axes[0],
            [i],
            [row["estimate"]],
            [row["ci_low"]],
            [row["ci_high"]],
            color=color,
        )
    axes[0].axhline(0, color=DARK, linestyle="--", linewidth=1.2)
    axes[0].set_xticks(range(2))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Change in ordinary proportion sent")
    axes[0].set_title("2 of 8 defectors − no defectors", fontsize=13, fontweight="bold")
    axes[0].set_ylim(-.065, .035)
    axes[0].grid(True, axis="y", alpha=.3)
    cultural_metrics = [
        (
            "defector_myth_coop_density_rounds2_10",
            "Cooperation/fairness terms",
            "o",
        ),
        (
            "defector_myth_threat_density_rounds2_10",
            "Threat/defection terms",
            "s",
        ),
    ]
    offsets = [-.08, .08]
    for metric_index, (metric, metric_label, marker) in enumerate(cultural_metrics):
        rows = data[data["metric"] == metric].set_index("model_id")
        estimates, lows, highs = [], [], []
        for model in models:
            estimates.append(rows.loc[model, "estimate"])
            lows.append(rows.loc[model, "ci_low"])
            highs.append(rows.loc[model, "ci_high"])
        errorbar(
            axes[1],
            np.arange(2) + offsets[metric_index],
            estimates,
            lows,
            highs,
            color=[BLUE, PINK][metric_index],
            marker=marker,
            label=metric_label,
        )
    axes[1].axhline(0, color=DARK, linestyle="--", linewidth=1.2)
    axes[1].set_xticks(range(2))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel(
        "Difference in category-word frequency\n"
        "(defector − non-defector myths)\n"
        "Matches per 100 words",
        fontsize=10,
    )
    axes[1].set_title(
        "Defector vs non-defector myth vocabulary\n"
        "(after the first forced zero action; rounds 2–10)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].text(
        0.5,
        0.965,
        "Above 0: more frequent in defector myths\n"
        "Below 0: less frequent in defector myths",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color=DARK,
    )
    axes[1].set_ylim(-.9, .3)
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].grid(True, axis="y", alpha=.3)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, .90))
    finish(fig, output_dir / "02_defector_spillovers.png")
    plt.close(fig)


def gpt_information(output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    dossier = pd.read_csv(
        "docs/figures/history_visibility_confirmation_gpt_n10_20260821/contrasts.csv"
    )
    identity = pd.read_csv(
        "docs/figures/identity_persistence_confirmation_gpt_n10_20260821/"
        "paired_contrasts.csv"
    )
    anonymous = pd.read_csv(
        "docs/figures/anonymous_population_record_gpt_n5_20260821/paired_contrasts.csv"
    )
    rows = [
        dossier[(dossier["metric"] == "final_balance") & dossier["contrast"].eq("Dossier − private | Game only")].iloc[0],
        dossier[(dossier["metric"] == "final_balance") & dossier["contrast"].eq("Dossier − private | Myth→Game")].iloc[0],
        identity[identity["metric"] == "final_balance"].iloc[0],
        anonymous[(anonymous["metric"] == "final_balance") & anonymous["contrast"].eq("Anonymous record − private memory")].iloc[0],
    ]
    labels = [
        "Partner dossier\n(Game only, n=10)",
        "Partner dossier\n(Myth → Game, n=10)",
        "Persistent IDs\n(n=10)",
        "Anonymous population record\n(n=5)",
    ]
    colors = [BLUE, BLUE, ORANGE, GREEN]
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(rows))[::-1]
    for pos, row, color in zip(y, rows, colors):
        errorbar(
            ax,
            [row["estimate"]],
            [pos],
            [pos - 0],
            [pos + 0],
            color=color,
        )
        ax.plot([row["ci_low"], row["ci_high"]], [pos, pos], color=color, linewidth=2.2)
        ax.plot([row["ci_low"], row["ci_low"]], [pos-.08, pos+.08], color=color, linewidth=1.8)
        ax.plot([row["ci_high"], row["ci_high"]], [pos-.08, pos+.08], color=color, linewidth=1.8)
        ax.scatter(row["estimate"], pos, color=color, s=65, zorder=3)
    ax.axvline(0, color=DARK, linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Change in final balance per agent (95% paired CI)")
    ax.set_title(
        "GPT-5 Nano: persistent identity helps; partner dossiers tend to hurt",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlim(-7, 8)
    ax.grid(True, axis="x", alpha=.3)
    ax.text(
        .02,
        .03,
        "The predicted dossier × Myth→Game interaction was not confirmed.",
        transform=ax.transAxes,
        fontsize=10,
        color=DARK,
    )
    fig.tight_layout()
    finish(fig, output_dir / "03_gpt_social_information.png")
    plt.close(fig)


def punishment_calibration(output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    old = pd.read_csv(
        "docs/figures/punishment_comprehension_crossmodel_20260821/cell_summary.csv"
    )
    new = pd.read_csv(
        "docs/figures/punishment_comprehension_gemini37_20260823/cell_summary.csv"
    )
    frames = [
        ("GPT-5 Nano", old[old["model_id"] == "gpt"], GREEN, "o"),
        ("Gemini 3.1 Flash-Lite", old[old["model_id"] == "gemini"], BLUE, "s"),
        ("Gemini 3.7 Flash", new[new["model"] == "Gemini 3.7 Flash"], ORANGE, "^"),
    ]
    fig, ax = plt.subplots(figsize=(11, 7))
    for label, frame, color, marker in frames:
        frame = frame.sort_values("return_ratio")
        ax.plot(
            100 * frame["return_ratio"],
            frame["mean_deduction"],
            color=color,
            marker=marker,
            markersize=8,
            linewidth=2.5,
            label=label,
        )
        ax.fill_between(
            100 * frame["return_ratio"],
            frame["ci_low"],
            frame["ci_high"],
            color=color,
            alpha=.14,
        )
    ax.axvline(50, color=DARK, linestyle="--", linewidth=1.2)
    ax.text(51.5, 1.88, "Fair-share threshold", fontsize=10, color=DARK)
    ax.set_title(
        "The same punishment rule elicits sharply model-specific policies",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Visible return (% of receipt)")
    ax.set_ylabel("Mean deduction points spent (0–2)")
    ax.set_xticks([0, 10, 25, 50, 75])
    ax.set_ylim(-.08, 2.18)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    finish(fig, output_dir / "04_punishment_calibration.png")
    plt.close(fig)


def punishment_population(output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    gpt = pd.read_csv(
        "docs/figures/defector_punishment_factorial_gpt_n5_20260821/paired_contrasts.csv"
    )
    lite = pd.read_csv(
        "docs/figures/defector_punishment_gemini_factorial_confirmation_n10_20260822/contrasts.csv"
    )
    flash = pd.read_csv(
        "docs/figures/defector_punishment_gemini37_n3_20260823/contrasts.csv"
    )
    model_labels = ["GPT-5 Nano\n(n=5)", "Gemini 3.1\nFlash-Lite (n=10)", "Gemini 3.7\nFlash (n=3)"]
    colors = [GREEN, BLUE, ORANGE]
    target_defector = [93.2, 85.7, 100.0]
    target_ordinary = [91.5, 2.8, 0.0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    x = np.arange(3)
    width = .34
    axes[0].bar(x-width/2, target_defector, width, color=PINK, label="Defector receiver")
    axes[0].bar(x+width/2, target_ordinary, width, color=LIGHT, edgecolor=DARK, label="Ordinary receiver")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_labels)
    axes[0].set_ylabel("Any deduction (%)")
    axes[0].set_ylim(0, 108)
    axes[0].set_title("Who gets punished?", fontsize=13, fontweight="bold")
    axes[0].legend(loc="lower left", fontsize=9)
    axes[0].grid(True, axis="y", alpha=.3)
    gpt_row = gpt[
        (gpt["metric"] == "standard_return_ratio")
        & (gpt["contrast_type"] == "availability_defectors25")
    ].iloc[0]
    lite_row = lite[
        (lite["metric"] == "standard_return_ratio")
        & (lite["contrast_type"] == "availability_defectors25")
    ].iloc[0]
    flash_row = flash[
        (flash["metric"] == "standard_return_ratio")
        & (flash["contrast_type"] == "availability_defectors25")
    ].iloc[0]
    for i, (row, color) in enumerate(zip([gpt_row, lite_row, flash_row], colors)):
        errorbar(
            axes[1],
            [i],
            [row["estimate"]],
            [row["ci_low"]],
            [row["ci_high"]],
            color=color,
        )
    axes[1].axhline(0, color=DARK, linestyle="--", linewidth=1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_labels)
    axes[1].set_ylabel("Available − unavailable return ratio")
    axes[1].set_ylim(-.09, .13)
    axes[1].set_title("Does punishment crowd out returning?", fontsize=13, fontweight="bold")
    axes[1].grid(True, axis="y", alpha=.3)
    axes[1].annotate(
        "Confirmed crowding",
        xy=(1, lite_row["estimate"]),
        xytext=(1.25, -.075),
        fontsize=9,
        color=BLUE,
        arrowprops={"arrowstyle": "->", "color": BLUE},
    )
    fig.suptitle(
        "Selective targeting generalizes across Gemini; behavioral consequences do not",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, .95))
    finish(fig, output_dir / "05_punishment_population.png")
    plt.close(fig)


def gemini_ceiling(output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    summary = pd.read_csv(
        "docs/figures/gemini37_flash_task_order_n3_20260823/summary.csv"
    )
    return_rules = pd.read_csv(
        "docs/figures/defector_punishment_gemini37_n3_20260823/return_rule_summary.csv"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    x = np.arange(3)
    rates = []
    for condition in CONDITIONS:
        rates.append(
            summary[
                (summary["condition"] == condition)
                & (summary["metric"] == "maximum_send_rate")
            ]["mean"].iloc[0]
        )
    axes[0].bar(x, rates, color=[GREEN, BLUE, PINK], width=.62)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CONDITION_LABELS, rotation=12)
    axes[0].set_ylabel("Maximum-send rate")
    axes[0].set_ylim(.84, 1.02)
    axes[0].set_yticks([.85, .90, .95, 1.00])
    axes[0].set_title("Baseline task-order screen", fontsize=13, fontweight="bold")
    axes[0].grid(True, axis="y", alpha=.3)
    axes[0].text(1, .97, "360 / 360 full sends", ha="center", fontsize=11, fontweight="bold")
    labels = ["Off\n0%", "Off\n25%", "On\n0%", "On\n25%"]
    order = [("off", "control"), ("off", "defectors25"), ("on", "control"), ("on", "defectors25")]
    exact, near_only, other = [], [], []
    for availability, condition in order:
        row = return_rules[
            (return_rules["availability"] == availability)
            & (return_rules["condition"] == condition)
        ].iloc[0]
        n = row["n"]
        exact.append(100 * row["exact_half_visible"] / n)
        near_only.append(100 * (row["within_half_cent"] - row["exact_half_visible"]) / n)
        other.append(100 * (n - row["within_half_cent"]) / n)
    x2 = np.arange(4)
    axes[1].bar(x2, exact, color=GREEN, label="Exact half")
    axes[1].bar(x2, near_only, bottom=exact, color=BLUE, label="Within 0.5¢")
    axes[1].bar(x2, other, bottom=np.asarray(exact)+np.asarray(near_only), color=ORANGE, label="Other")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("Punishment availability / defectors")
    axes[1].set_ylabel("Ordinary receiver decisions (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Return relative to noisy receipt shown", fontsize=13, fontweight="bold")
    axes[1].legend(loc="lower left", fontsize=9)
    axes[1].grid(True, axis="y", alpha=.3)
    axes[1].text(1.5, 94, "327 / 370 within 0.5¢ of half", ha="center", fontsize=10, fontweight="bold")
    fig.suptitle(
        "Gemini 3.7 adopts two rigid defaults: send everything, return half",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, .95))
    finish(fig, output_dir / "06_gemini37_ceiling_and_half_rule.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    task_order(args.out)
    defector_spillovers(args.out)
    gpt_information(args.out)
    punishment_calibration(args.out)
    punishment_population(args.out)
    gemini_ceiling(args.out)


if __name__ == "__main__":
    main()
