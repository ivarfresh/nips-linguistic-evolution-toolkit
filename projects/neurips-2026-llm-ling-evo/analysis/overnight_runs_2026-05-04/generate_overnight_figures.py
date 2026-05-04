#!/usr/bin/env python3
"""Generate figures summarising the 2026-05-03 overnight model sweeps."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
ANALYSIS_ROOT = REPO_ROOT / "projects" / "neurips-2026-llm-ling-evo" / "analysis"
DATA_ROOT = REPO_ROOT / "data" / "json" / "noise_experiments"
CELL_SUMMARY = ANALYSIS_ROOT / "cell_summaries" / "cell_summary.csv"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_LABELS = {
    "game": "Game only",
    "game_myth": "Game -> myth",
    "myth_game": "Myth -> game",
}

TASK_COLORS = {
    "game": "#4b5563",
    "game_myth": "#2563eb",
    "myth_game": "#dc2626",
}

NOISE_LABELS = {
    "positive": "Positive noise",
    "negative_5": "Negative noise",
    "bootstrap": "Bootstrap noise",
}


@dataclass(frozen=True)
class CellKey:
    model: str
    version: str
    experiment: str
    noise_condition: str
    informed: bool
    task_order: str


def is_final_json(path: Path) -> bool:
    name = path.name
    return (
        name.endswith(".json")
        and ".checkpoint" not in name
        and ".results" not in name
        and ".error" not in name
    )


def count_final_jsons(experiment: str) -> int:
    root = DATA_ROOT / "v4_direct_provider" / experiment
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*.json") if is_final_json(path))


def balance_trajectory(path: Path) -> np.ndarray | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    points = []
    for entry in sorted(data.get("conversation_history", []), key=lambda e: e.get("round", 0)):
        balances = entry.get("balances") or {}
        a1 = balances.get("Agent_1")
        a2 = balances.get("Agent_2")
        if a1 is None or a2 is None:
            continue
        points.append((float(a1) + float(a2)) / 2.0)
    if not points:
        return None
    return np.array(points, dtype=float)


def collect_trajectories(
    experiment: str,
    model: str,
    task_order: str,
    noise_condition: str,
    version: str = "v4_direct_provider",
) -> list[np.ndarray]:
    root = DATA_ROOT / version / experiment / model / task_order / noise_condition
    if not root.exists():
        return []
    trajectories = []
    for path in root.rglob("*.json"):
        if not is_final_json(path):
            continue
        traj = balance_trajectory(path)
        if traj is not None and len(traj) >= 10:
            trajectories.append(traj)
    return trajectories


def mean_and_se(trajs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    if not trajs:
        return None
    t_max = max(len(t) for t in trajs)
    arr = np.full((len(trajs), t_max), np.nan)
    for i, traj in enumerate(trajs):
        arr[i, : len(traj)] = traj
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=1) if len(trajs) > 1 else np.zeros(t_max)
    se = std / max(np.sqrt(len(trajs)), 1.0)
    return mean, se


def plot_trajectory(ax, trajs: list[np.ndarray], label: str, color: str, linestyle: str = "-") -> None:
    stats = mean_and_se(trajs)
    if stats is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        return
    mean, se = stats
    rounds = np.arange(1, len(mean) + 1)
    ax.plot(rounds, mean, color=color, linestyle=linestyle, linewidth=2.2, label=f"{label} (n={len(trajs)})")
    ax.fill_between(rounds, mean - se, mean + se, color=color, alpha=0.14, linewidth=0)


def style_axis(ax, title: str | None = None) -> None:
    if title:
        ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlim(1, 10)
    ax.set_xticks([1, 5, 10])
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def read_cell_summary() -> dict[CellKey, dict[str, str]]:
    cells: dict[CellKey, dict[str, str]] = {}
    with CELL_SUMMARY.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = CellKey(
                model=row["model"],
                version=row["version"],
                experiment=row["experiment"],
                noise_condition=row["noise_condition"],
                informed=row["informed"] == "True",
                task_order=row["task_order"],
            )
            cells[key] = row
    return cells


def cell_mean(cells: dict[CellKey, dict[str, str]], key: CellKey) -> tuple[float, float, int]:
    row = cells[key]
    mean = float(row["mean"])
    std = float(row["std"]) if row["std"] else 0.0
    n = int(row["n"])
    return mean, std, n


def figure_1_completion() -> None:
    rows = [
        ("Gemini 3.1 Pro", "Positive", "noise_positive_mem3_gemini_3_1_pro", 90),
        ("Gemini 3.1 Pro", "Negative", "noise_negative_mem3_gemini_3_1_pro", 90),
        ("Gemini 3.1 Pro", "Bootstrap", "noise_bootstrap_mem3_gemini_3_1_pro", 90),
        ("GPT-5.5", "Positive", "noise_positive_mem3_gpt5_5", 90),
        ("GPT-5.5", "Negative", "noise_negative_mem3_gpt5_5", 90),
        ("GPT-5.5", "Bootstrap", "noise_bootstrap_mem3_gpt5_5", 90),
    ]
    labels = [f"{model}\n{noise}" for model, noise, _, _ in rows]
    completed = [count_final_jsons(experiment) for _, _, experiment, _ in rows]
    expected = [expected for _, _, _, expected in rows]
    colors = ["#16a34a" if c == e else "#f97316" if c > 1 else "#dc2626" for c, e in zip(completed, expected)]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(rows))
    ax.bar(x, expected, color="#e5e7eb", width=0.68, label="Planned")
    ax.bar(x, completed, color=colors, width=0.68, label="Final JSONs available")
    for i, (c, e) in enumerate(zip(completed, expected)):
        ax.text(i, max(c, 2) + 2, f"{c}/{e}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Runs")
    ax.set_ylim(0, 105)
    ax.set_title("Overnight missing-condition sweep: completion status", fontsize=13, pad=12)
    ax.text(
        0.99,
        -0.20,
        "GPT-5.5 counts include launch-prep smoke files; OpenAI quota stopped the remaining cells.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#4b5563",
    )
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "fig1_completion_status")


def figure_2_gemini_trajectories() -> None:
    rows = [
        ("positive", "noise_positive_mem3_gemini_3_1_pro", "noisy_positive_5"),
        ("negative_5", "noise_negative_mem3_gemini_3_1_pro", "noisy_negative_5"),
        ("bootstrap", "noise_bootstrap_mem3_gemini_3_1_pro", "noisy_bootstrap_cooperation"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(10.4, 8.4), sharex=True, sharey=True)
    for r, (noise_label, experiment, base_condition) in enumerate(rows):
        for c, informed in enumerate([False, True]):
            suffix = "_informed" if informed else ""
            condition = f"{base_condition}{suffix}"
            ax = axes[r, c]
            for task_order in ["game", "game_myth", "myth_game"]:
                plot_trajectory(
                    ax,
                    collect_trajectories(experiment, "gemini-3.1-pro-preview", task_order, condition),
                    TASK_LABELS[task_order],
                    TASK_COLORS[task_order],
                )
            col_title = "agents informed about noise" if informed else "standard prompt"
            style_axis(ax, f"{NOISE_LABELS[noise_label]}: {col_title}")
    fig.suptitle("Gemini-3.1-pro-preview overnight sweep: mean cumulative reward trajectories", fontsize=13, y=0.99)
    fig.supxlabel("Round", fontsize=11, y=0.055)
    fig.supylabel("Mean cumulative reward per agent", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0.02, 0.08, 1, 0.96])
    save(fig, "fig2_gemini_trajectories")


def figure_3_bootstrap_comparison(cells: dict[CellKey, dict[str, str]]) -> None:
    models = [
        ("GPT-5-Nano", "gpt-5-nano", "noise_bootstrap_mem3"),
        ("Gemini 3.1 Pro", "gemini-3.1-pro-preview", "noise_bootstrap_mem3_gemini_3_1_pro"),
    ]
    task_orders = ["game", "game_myth", "myth_game"]
    values = []
    errors = []
    labels = []
    for model_label, model, experiment in models:
        model_values = []
        model_errors = []
        for task_order in task_orders:
            key = CellKey(
                model=model,
                version="v4_direct_provider",
                experiment=experiment,
                noise_condition="noisy_bootstrap_cooperation",
                informed=False,
                task_order=task_order,
            )
            mean, std, n = cell_mean(cells, key)
            model_values.append(mean)
            model_errors.append(std / np.sqrt(n))
        values.append(model_values)
        errors.append(model_errors)
        labels.append(model_label)

    x = np.arange(len(task_orders))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.bar(x - width / 2, values[0], yerr=errors[0], capsize=3, width=width, label=labels[0], color="#2563eb")
    ax.bar(x + width / 2, values[1], yerr=errors[1], capsize=3, width=width, label=labels[1], color="#16a34a")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in task_orders])
    ax.set_ylabel("Final cumulative reward per agent")
    ax.set_ylim(45, 76)
    ax.set_title("Bootstrap standard prompt: GPT-5-Nano destabilises, Gemini does not", fontsize=13, pad=12)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "fig3_bootstrap_model_comparison")


def figure_4_gpt55_positive(cells: dict[CellKey, dict[str, str]]) -> None:
    models = [
        ("GPT-5-Nano", "gpt-5-nano", "noise_positive_mem3_gpt5_nano"),
        ("Gemini 3.1 Pro", "gemini-3.1-pro-preview", "noise_positive_mem3_gemini_3_1_pro"),
        ("GPT-5.5", "gpt-5.5", "noise_positive_mem3_gpt5_5"),
    ]
    task_orders = ["game", "game_myth", "myth_game"]
    x = np.arange(len(task_orders))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    colors = ["#2563eb", "#16a34a", "#9333ea"]
    offsets = [-width, 0, width]
    for (label, model, experiment), color, offset in zip(models, colors, offsets):
        values = []
        errors = []
        ns = []
        for task_order in task_orders:
            key = CellKey(
                model=model,
                version="v4_direct_provider",
                experiment=experiment,
                noise_condition="noisy_positive_5",
                informed=False,
                task_order=task_order,
            )
            mean, std, n = cell_mean(cells, key)
            values.append(mean)
            errors.append(std / np.sqrt(n) if n else 0.0)
            ns.append(n)
        ax.bar(x + offset, values, yerr=errors, capsize=3, width=width, label=label, color=color)
        for xi, value, n in zip(x + offset, values, ns):
            if n < 15:
                ax.text(xi, value + 0.35, f"n={n}", ha="center", va="bottom", fontsize=7, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in task_orders])
    ax.set_ylabel("Final cumulative reward per agent")
    ax.set_ylim(68, 76)
    ax.set_title("Positive-noise standard cells are near ceiling, including GPT-5.5", fontsize=13, pad=12)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "fig4_positive_ceiling_comparison")


def figure_5_gpt55_positive_trajectories() -> None:
    panels = [
        ("standard prompt", "noisy_positive_5"),
        ("agents informed about noise", "noisy_positive_5_informed"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), sharex=True, sharey=True)
    for ax, (title, condition) in zip(axes, panels):
        for task_order in ["game", "game_myth", "myth_game"]:
            plot_trajectory(
                ax,
                collect_trajectories("noise_positive_mem3_gpt5_5", "gpt-5.5", task_order, condition),
                TASK_LABELS[task_order],
                TASK_COLORS[task_order],
            )
        style_axis(ax, title)
        ax.set_ylim(0, 78)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("GPT-5.5 positive-noise runs that completed: near-ceiling trajectories", fontsize=13, y=0.99)
    fig.supxlabel("Round", fontsize=11, y=0.075)
    fig.supylabel("Mean cumulative reward per agent", fontsize=11)
    fig.text(
        0.99,
        0.01,
        "Completed cells: 15,15,15 standard; 15,15,2 informed. Last informed myth->game cell is quota-truncated.",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout(rect=[0.02, 0.12, 1, 0.94])
    save(fig, "fig5_gpt55_positive_trajectories")


def main() -> None:
    cells = read_cell_summary()
    figure_1_completion()
    figure_2_gemini_trajectories()
    figure_3_bootstrap_comparison(cells)
    figure_4_gpt55_positive(cells)
    figure_5_gpt55_positive_trajectories()


if __name__ == "__main__":
    main()
