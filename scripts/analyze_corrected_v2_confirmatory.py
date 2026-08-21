#!/usr/bin/env python3
"""Summarize and plot the corrected informed-noise confirmatory rerun.

The primary dataset is the fresh 2 x 3 population-regime by task-order batch
completed on 2026-08-12 (n=10 independent runs per cell).  The script also
loads the superseded dyad batch solely for a clearly labelled diagnostic figure
showing the effect of the broken communicated-transfer path.

Outputs:
  - run_metrics.csv
  - summary.csv
  - contrasts.csv
  - final_balance_by_population_taskorder.png
  - behavior_metrics_by_population_taskorder.png
  - trust_trajectories_by_population_taskorder.png
  - task_order_contrasts_by_population.png
  - corrected_vs_invalid_dyad.png

Usage:
  python scripts/analyze_corrected_v2_confirmatory.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from analyses._shared import configure_matplotlib


ENDOWMENT = 5.0
CONDITION_ORDER = ["game", "game_myth", "myth_game"]
CONDITION_LABELS = {
    "game": "Game only",
    "game_myth": "Game → Myth",
    "myth_game": "Myth → Game",
}
CONDITION_COLORS = {
    "game": "#66c2a5",
    "game_myth": "#8da0cb",
    "myth_game": "#e78ac3",
}
POPULATION_COLORS = {
    "2-agent repeated dyad": "#66c2a5",
    "8-agent rotating population": "#fc8d62",
}
POPULATION_ORDER = ["2-agent repeated dyad", "8-agent rotating population"]
POPULATION_SHORT = {
    "2-agent repeated dyad": "2 agents",
    "8-agent rotating population": "8 agents",
}
POPULATION_PLOT_LABELS = {
    "2-agent repeated dyad": "2-agent dyad",
    "8-agent rotating population": "8-agent population",
}

DEFAULT_INPUT = Path(
    "data/json/noise_experiments/corrected_v2_confirmatory_20260812"
)
DEFAULT_INVALID_INPUT = Path("data/json/noise_experiments/v2")
DEFAULT_OUTPUT = Path("docs/figures/corrected_v2_confirmatory_20260812")

PRIMARY_CELLS = {
    "2-agent repeated dyad": {
        "game": "noise2i_memprimary_v2_game",
        "game_myth": "noise2i_memprimary_v2_game_myth",
        "myth_game": "noise2i_memprimary_v2_myth_game",
    },
    "8-agent rotating population": {
        "game": "noise8i_memprimary_v2_game",
        "game_myth": "noise8i_memprimary_v2_game_myth",
        "myth_game": "noise8i_memprimary_v2_myth_game",
    },
}
INVALID_DYAD_CELLS = {
    "game": "noise2i_memprimary_game",
    "game_myth": "noise2i_memprimary_game_myth",
    "myth_game": "noise2i_memtest_memprimary",
}


def load_runs(directory: Path) -> list[tuple[Path, dict]]:
    """Load final simulation JSONs while excluding results and error artifacts."""
    runs: list[tuple[Path, dict]] = []
    for path in sorted(directory.rglob("*.json")):
        if path.name.endswith(
            (".results.json", ".checkpoint.json", ".error.json")
        ) or ".checkpoint." in path.name:
            continue
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("conversation_history"):
            runs.append((path, data))
    return runs


def round_dyads(round_entry: dict) -> list[dict]:
    """Return dyad records for either multi-agent or legacy scalar rounds."""
    dyads = round_entry.get("dyads") or []
    if dyads:
        return [dyad for dyad in dyads if dyad.get("sent") is not None]
    if round_entry.get("sent") is None:
        return []
    return [round_entry]


def safe_return_ratio(dyad: dict) -> float:
    received = float(dyad.get("received") or 0.0)
    returned = float(dyad.get("returned") or 0.0)
    return returned / received if received > 0 else 0.0


def final_balances(run: dict) -> dict[str, float]:
    for entry in reversed(run.get("conversation_history", [])):
        balances = entry.get("balances")
        if balances:
            return {key: float(value) for key, value in balances.items()}
    raise ValueError("Run has no final balances")


def run_metrics(path: Path, run: dict) -> tuple[dict, list[dict]]:
    """Compute run-level outcomes and per-round behavior."""
    all_dyads: list[dict] = []
    trajectory: list[dict] = []
    for entry in run.get("conversation_history", []):
        dyads = round_dyads(entry)
        if not dyads:
            continue
        all_dyads.extend(dyads)
        trajectory.append(
            {
                "round": int(entry.get("round", len(trajectory) + 1)),
                "trust_ratio": float(
                    np.mean([float(dyad["sent"]) / ENDOWMENT for dyad in dyads])
                ),
                "return_ratio": float(np.mean([safe_return_ratio(dyad) for dyad in dyads])),
            }
        )
    if not all_dyads:
        raise ValueError(f"No game interactions in {path}")

    balances = final_balances(run)
    communicated_later = [
        float(dyad["sent_communicated"])
        for entry in run.get("conversation_history", [])
        if int(entry.get("round", 0)) > 1
        for dyad in round_dyads(entry)
        if dyad.get("sent_communicated") is not None
    ]
    metadata = run.get("run_metadata") or {}
    metrics = {
        "path": str(path),
        "replicate_id": metadata.get("replicate_id"),
        "num_agents": int(metadata.get("num_agents") or len(balances)),
        "final_balance": float(np.mean(list(balances.values()))),
        "joint_balance": float(sum(balances.values())),
        "mean_trust_ratio": float(
            np.mean([float(dyad["sent"]) / ENDOWMENT for dyad in all_dyads])
        ),
        "mean_return_ratio": float(np.mean([safe_return_ratio(dyad) for dyad in all_dyads])),
        "mean_sent": float(np.mean([float(dyad["sent"]) for dyad in all_dyads])),
        "mean_returned": float(
            np.mean([float(dyad.get("returned") or 0.0) for dyad in all_dyads])
        ),
        "mean_later_communicated_sent": (
            float(np.mean(communicated_later)) if communicated_later else math.nan
        ),
        "num_interactions": len(all_dyads),
    }
    return metrics, trajectory


def add_cell(
    rows: list[dict],
    trajectories: list[dict],
    directory: Path,
    source: str,
    population: str,
    condition: str,
) -> None:
    runs = load_runs(directory)
    if not runs:
        raise FileNotFoundError(f"No final JSON runs found under {directory}")
    for run_number, (path, run) in enumerate(runs, start=1):
        metrics, run_trajectory = run_metrics(path, run)
        run_id = f"{source}|{population}|{condition}|{run_number:02d}"
        rows.append(
            {
                "source": source,
                "population": population,
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "run_id": run_id,
                **metrics,
            }
        )
        for point in run_trajectory:
            trajectories.append(
                {
                    "source": source,
                    "population": population,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "run_id": run_id,
                    **point,
                }
            )


def welch_difference(a: Iterable[float], b: Iterable[float]) -> dict[str, float]:
    """Welch estimate, 95% CI, and two-sided p value for mean(a)-mean(b)."""
    from scipy import stats

    a_array = np.asarray(list(a), dtype=float)
    b_array = np.asarray(list(b), dtype=float)
    estimate = float(np.mean(a_array) - np.mean(b_array))
    va = float(np.var(a_array, ddof=1) / len(a_array))
    vb = float(np.var(b_array, ddof=1) / len(b_array))
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (
        va**2 / (len(a_array) - 1) + vb**2 / (len(b_array) - 1)
    )
    critical = float(stats.t.ppf(0.975, df))
    test = stats.ttest_ind(a_array, b_array, equal_var=False)
    return {
        "estimate": estimate,
        "ci_low": estimate - critical * se,
        "ci_high": estimate + critical * se,
        "p_value": float(test.pvalue),
        "df": float(df),
    }


def independent_interaction(
    a8: Iterable[float],
    b8: Iterable[float],
    a2: Iterable[float],
    b2: Iterable[float],
) -> dict[str, float]:
    """Independent difference-in-differences: (a8-b8) - (a2-b2)."""
    from scipy import stats

    groups = [np.asarray(list(values), dtype=float) for values in (a8, b8, a2, b2)]
    means = [float(np.mean(values)) for values in groups]
    estimate = (means[0] - means[1]) - (means[2] - means[3])
    variance_terms = [float(np.var(values, ddof=1) / len(values)) for values in groups]
    variance = sum(variance_terms)
    se = math.sqrt(variance)
    df = variance**2 / sum(
        term**2 / (len(values) - 1)
        for term, values in zip(variance_terms, groups)
    )
    critical = float(stats.t.ppf(0.975, df))
    t_value = estimate / se
    p_value = float(2 * stats.t.sf(abs(t_value), df))
    return {
        "estimate": estimate,
        "ci_low": estimate - critical * se,
        "ci_high": estimate + critical * se,
        "p_value": p_value,
        "df": float(df),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm familywise-error adjustment, returned in original order."""
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    family_size = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (family_size - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def build_contrasts(primary_df):
    import pandas as pd

    metrics = ["final_balance", "mean_trust_ratio", "mean_return_ratio"]
    condition_pairs = [
        ("game_myth", "game", "Game → Myth − Game only"),
        ("myth_game", "game", "Myth → Game − Game only"),
        ("myth_game", "game_myth", "Myth → Game − Game → Myth"),
    ]
    rows: list[dict] = []
    for metric in metrics:
        metric_rows: list[dict] = []
        for population in POPULATION_ORDER:
            pop_df = primary_df[primary_df["population"] == population]
            for a, b, label in condition_pairs:
                result = welch_difference(
                    pop_df[pop_df["condition"] == a][metric],
                    pop_df[pop_df["condition"] == b][metric],
                )
                metric_rows.append(
                    {
                        "contrast_type": "within_population",
                        "metric": metric,
                        "population": population,
                        "contrast": label,
                        "condition_a": a,
                        "condition_b": b,
                        **result,
                    }
                )
        adjusted = holm_adjust([row["p_value"] for row in metric_rows])
        for row, p_holm in zip(metric_rows, adjusted):
            row["p_holm_six_tests"] = p_holm
        rows.extend(metric_rows)

        for a, b, label in condition_pairs:
            pop2 = primary_df[primary_df["population"] == POPULATION_ORDER[0]]
            pop8 = primary_df[primary_df["population"] == POPULATION_ORDER[1]]
            result = independent_interaction(
                pop8[pop8["condition"] == a][metric],
                pop8[pop8["condition"] == b][metric],
                pop2[pop2["condition"] == a][metric],
                pop2[pop2["condition"] == b][metric],
            )
            rows.append(
                {
                    "contrast_type": "population_regime_interaction",
                    "metric": metric,
                    "population": "8-agent contrast − 2-agent contrast",
                    "contrast": label,
                    "condition_a": a,
                    "condition_b": b,
                    "p_holm_six_tests": math.nan,
                    **result,
                }
            )
    return pd.DataFrame(rows)


def save_tables(df, contrasts, output_dir: Path) -> None:
    primary_df = df[df["source"] == "Corrected v2"].copy()
    metrics = [
        "final_balance",
        "joint_balance",
        "mean_trust_ratio",
        "mean_return_ratio",
        "mean_sent",
        "mean_returned",
        "mean_later_communicated_sent",
    ]
    summary = (
        primary_df.groupby(["population", "condition", "condition_label"], sort=False)[metrics]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    df.to_csv(output_dir / "run_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    contrasts.to_csv(output_dir / "contrasts.csv", index=False)


def condition_boxplot(sns, data, x, y, order, ax, palette):
    sns.boxplot(
        data=data,
        x=x,
        y=y,
        hue=x,
        order=order,
        hue_order=order,
        palette=palette,
        width=0.62,
        linewidth=1.2,
        fliersize=0,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x=x,
        y=y,
        order=order,
        color="#263238",
        edgecolor="white",
        linewidth=0.45,
        size=5.2,
        alpha=0.72,
        jitter=0.16,
        ax=ax,
    )


def plot_final_balance(primary_df, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = primary_df.copy()
    plot_df["Population"] = plot_df["population"].map(POPULATION_PLOT_LABELS)
    order = [CONDITION_LABELS[key] for key in CONDITION_ORDER]
    population_order = [POPULATION_PLOT_LABELS[value] for value in POPULATION_ORDER]
    population_palette = {
        POPULATION_PLOT_LABELS[key]: value for key, value in POPULATION_COLORS.items()
    }

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.boxplot(
        data=plot_df,
        x="condition_label",
        y="joint_balance",
        hue="Population",
        order=order,
        hue_order=population_order,
        palette=population_palette,
        ax=ax,
    )
    ax.set_title(
        "Population-Wide (Joint) Cumulative Balance: 2-Agent vs 8-Agent\n"
        "(Corrected informed noise, memory-primary, sum of all agents per run)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Joint Cumulative Balance (sum of all agents)", fontsize=12)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.legend(title="Population", loc="upper left")
    fig.patch.set_edgecolor("#111111")
    fig.patch.set_linewidth(1.4)
    fig.tight_layout()
    fig.savefig(
        output_dir / "final_balance_by_population_taskorder.png",
        dpi=300,
        facecolor="white",
        edgecolor="#111111",
    )
    plt.close(fig)


def plot_behavior_metrics(primary_df, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = primary_df.copy()
    plot_df["Population"] = plot_df["population"].map(POPULATION_PLOT_LABELS)
    population_order = [POPULATION_PLOT_LABELS[value] for value in POPULATION_ORDER]
    population_palette = {
        POPULATION_PLOT_LABELS[key]: value for key, value in POPULATION_COLORS.items()
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), sharey=True)
    fig.suptitle(
        "Cooperation Behavior: 2-Agent vs 8-Agent",
        fontsize=15,
        fontweight="bold",
    )
    order = [CONDITION_LABELS[key] for key in CONDITION_ORDER]
    for index, (metric, title, ylabel) in enumerate(
        [
            ("mean_trust_ratio", "Trust Ratio (Proportion Sent)", "Mean Trust Ratio"),
            ("mean_return_ratio", "Return Ratio (Proportion Returned)", "Mean Return Ratio"),
        ]
    ):
        ax = axes[index]
        sns.boxplot(
            data=plot_df,
            x="condition_label",
            y=metric,
            hue="Population",
            order=order,
            hue_order=population_order,
            palette=population_palette,
            ax=ax,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Condition", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(True, alpha=0.3)
        if index == 0:
            ax.legend(title="Population", loc="lower right", fontsize=9)
        else:
            ax.get_legend().remove()
    fig.patch.set_edgecolor("#111111")
    fig.patch.set_linewidth(1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(
        output_dir / "behavior_metrics_by_population_taskorder.png",
        dpi=300,
        facecolor="white",
        edgecolor="#111111",
    )
    plt.close(fig)


def plot_trust_trajectories(trajectory_df, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 7), sharey=True)
    fig.suptitle(
        "Trust trajectories across the ten game rounds",
        fontsize=15,
        fontweight="bold",
    )
    for ax, population in zip(axes, POPULATION_ORDER):
        pop_df = trajectory_df[trajectory_df["population"] == population]
        for condition in CONDITION_ORDER:
            cond_df = pop_df[pop_df["condition"] == condition]
            color = CONDITION_COLORS[condition]
            for _, run_df in cond_df.groupby("run_id"):
                run_df = run_df.sort_values("round")
                ax.plot(
                    run_df["round"],
                    run_df["trust_ratio"],
                    color=color,
                    linewidth=1,
                    alpha=0.24,
                )
            mean_df = cond_df.groupby("round", as_index=False)["trust_ratio"].mean()
            ax.plot(
                mean_df["round"],
                mean_df["trust_ratio"],
                color=color,
                linewidth=3,
                marker="o",
                markersize=4,
                label=CONDITION_LABELS[condition],
            )
        ax.set_title(POPULATION_SHORT[population], fontsize=12, fontweight="bold")
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean trust ratio" if ax is axes[0] else "")
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, linestyle="--")
    axes[1].legend(title="Task order", loc="lower right", fontsize=10)
    fig.patch.set_edgecolor("#111111")
    fig.patch.set_linewidth(1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(
        output_dir / "trust_trajectories_by_population_taskorder.png",
        dpi=300,
        facecolor="white",
        edgecolor="#111111",
    )
    plt.close(fig)


def plot_contrasts(contrasts, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    selected = contrasts[
        (contrasts["contrast_type"] == "within_population")
        & (contrasts["metric"] == "final_balance")
        & contrasts["contrast"].isin(
            ["Game → Myth − Game only", "Myth → Game − Game only"]
        )
    ]
    x_base = np.arange(len(POPULATION_ORDER), dtype=float)
    fig, ax = plt.subplots(figsize=(11, 7))
    styles = [
        ("Game → Myth − Game only", "#8da0cb", "o", -0.06),
        ("Myth → Game − Game only", "#e78ac3", "s", 0.06),
    ]
    for label, color, marker, offset in styles:
        rows = []
        for population in POPULATION_ORDER:
            row = selected[
                (selected["population"] == population) & (selected["contrast"] == label)
            ].iloc[0]
            rows.append(row)
        estimates = np.array([row["estimate"] for row in rows], dtype=float)
        lower = estimates - np.array([row["ci_low"] for row in rows], dtype=float)
        upper = np.array([row["ci_high"] for row in rows], dtype=float) - estimates
        ax.errorbar(
            x_base + offset,
            estimates,
            yerr=np.vstack([lower, upper]),
            color=color,
            marker=marker,
            markersize=8,
            linewidth=2.2,
            capsize=5,
            label=label,
        )
    ax.axhline(0, color="#37474f", linewidth=1.2, linestyle="--")
    ax.set_xticks(x_base)
    ax.set_xticklabels([POPULATION_SHORT[pop] for pop in POPULATION_ORDER])
    ax.set_ylabel("Difference in final balance (95% Welch CI)")
    ax.set_xlabel("Population regime")
    ax.set_title(
        "Task-order contrasts do not show a population-regime reversal",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.patch.set_edgecolor("#111111")
    fig.patch.set_linewidth(1.4)
    fig.tight_layout()
    fig.savefig(
        output_dir / "task_order_contrasts_by_population.png",
        dpi=300,
        facecolor="white",
        edgecolor="#111111",
    )
    plt.close(fig)


def plot_corrected_vs_invalid(df, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    dyad_df = df[df["population"] == POPULATION_ORDER[0]].copy()
    source_order = ["Invalidated 2026-08-10", "Corrected v2"]
    source_palette = {
        "Invalidated 2026-08-10": "#fc8d62",
        "Corrected v2": "#66c2a5",
    }
    order = [CONDITION_LABELS[key] for key in CONDITION_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    fig.suptitle(
        "Diagnostic comparison: invalidated versus corrected dyad batches",
        fontsize=15,
        fontweight="bold",
    )
    for ax, metric, title, ylabel in [
        (
            axes[0],
            "final_balance",
            "Final cumulative balance",
            "Average final balance per agent",
        ),
        (
            axes[1],
            "mean_later_communicated_sent",
            "Transfer signal shown after round 1",
            "Mean communicated amount sent",
        ),
    ]:
        sns.boxplot(
            data=dyad_df,
            x="condition_label",
            y=metric,
            hue="source",
            order=order,
            hue_order=source_order,
            palette=source_palette,
            fliersize=0,
            linewidth=1.2,
            ax=ax,
        )
        sns.stripplot(
            data=dyad_df,
            x="condition_label",
            y=metric,
            hue="source",
            order=order,
            hue_order=source_order,
            dodge=True,
            palette=source_palette,
            edgecolor="white",
            linewidth=0.4,
            size=4.6,
            alpha=0.7,
            ax=ax,
            legend=False,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Task order")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=12)
        ax.grid(True, axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.legend(title="Batch", loc="upper left", fontsize=9)
        else:
            ax.get_legend().remove()
            ax.set_ylim(bottom=0)
    fig.patch.set_edgecolor("#111111")
    fig.patch.set_linewidth(1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(
        output_dir / "corrected_vs_invalid_dyad.png",
        dpi=300,
        facecolor="white",
        edgecolor="#111111",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--invalid-input", type=Path, default=DEFAULT_INVALID_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    configure_matplotlib()
    import pandas as pd
    import seaborn as sns

    sns.set_style("whitegrid")
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    trajectories: list[dict] = []
    for population, cells in PRIMARY_CELLS.items():
        for condition, subdirectory in cells.items():
            add_cell(
                rows,
                trajectories,
                args.input / subdirectory,
                "Corrected v2",
                population,
                condition,
            )
    for condition, subdirectory in INVALID_DYAD_CELLS.items():
        add_cell(
            rows,
            trajectories,
            args.invalid_input / subdirectory,
            "Invalidated 2026-08-10",
            POPULATION_ORDER[0],
            condition,
        )

    df = pd.DataFrame(rows)
    trajectory_df = pd.DataFrame(trajectories)
    primary_df = df[df["source"] == "Corrected v2"].copy()
    primary_trajectory_df = trajectory_df[
        trajectory_df["source"] == "Corrected v2"
    ].copy()

    cell_counts = primary_df.groupby(["population", "condition"]).size()
    if len(primary_df) != 60 or not (cell_counts == 10).all():
        raise RuntimeError(
            "Expected 60 corrected runs with n=10 in every cell; got "
            f"{cell_counts.to_dict()}"
        )

    contrasts = build_contrasts(primary_df)
    save_tables(df, contrasts, args.out)
    plot_final_balance(primary_df, args.out)
    plot_behavior_metrics(primary_df, args.out)
    plot_trust_trajectories(primary_trajectory_df, args.out)
    plot_contrasts(contrasts, args.out)
    plot_corrected_vs_invalid(df, args.out)

    print("\nCorrected v2 cell summary (mean ± sample SD):")
    for population in POPULATION_ORDER:
        print(f"\n{population}")
        for condition in CONDITION_ORDER:
            cell = primary_df[
                (primary_df["population"] == population)
                & (primary_df["condition"] == condition)
            ]
            print(
                f"  {CONDITION_LABELS[condition]:12s} "
                f"balance {cell['final_balance'].mean():.3f} ± {cell['final_balance'].std(ddof=1):.3f}; "
                f"trust {cell['mean_trust_ratio'].mean():.3f}; "
                f"return {cell['mean_return_ratio'].mean():.3f}"
            )
    print(f"\nSaved tables and figures to {args.out}")


if __name__ == "__main__":
    main()
