#!/usr/bin/env python3
"""Plot Claude pre-noise choices against post-noise ledger transfer values.

This is scoped to the Claude cells reported in fig1_main_v4_trajectories:
positive and negative direct-provider noise, standard prompt, across the
three task orders.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

MPL_CACHE_DIR = Path("/private/tmp/nlet_matplotlib_cache")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data" / "json" / "noise_experiments"
OUT_DIR = Path(__file__).resolve().parent / "figures"
RUN_OUT_DIR = OUT_DIR / "claude_noise_value_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "claude-sonnet-4.5"
VERSION = "v4_direct_provider"

NOISE_CELLS = [
    {
        "key": "positive",
        "label": "Positive noise",
        "experiment": "noise_positive_mem3_claude_sonnet_45",
        "condition": "noisy_positive_5",
        "post_color": "#2563eb",
    },
    {
        "key": "negative",
        "label": "Negative noise",
        "experiment": "noise_negative_mem3_claude_sonnet_45",
        "condition": "noisy_negative_5",
        "post_color": "#dc2626",
    },
]

TASKS = [
    ("game", "Game only"),
    ("game_myth", "Game -> myth"),
    ("myth_game", "Myth -> game"),
]

TRANSFER_KINDS = [
    ("sent", "Sent", (-0.25, 5.25)),
    ("returned", "Returned", (-0.75, 15.75)),
]

PRE_COLOR = "#111827"
PRE_LABEL = "Chosen pre-noise"
POST_LABEL = "Ledger post-noise"


@dataclass(frozen=True)
class RunSeries:
    path: Path
    rounds: np.ndarray
    sent_pre: np.ndarray
    sent_post: np.ndarray
    returned_pre: np.ndarray
    returned_post: np.ndarray
    final_balance: float


def is_final_json(path: Path) -> bool:
    name = path.name
    return (
        name.endswith(".json")
        and ".checkpoint" not in name
        and ".results" not in name
        and ".error" not in name
    )


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_run(path: Path) -> RunSeries | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    rows = []
    final_balance = np.nan
    for entry in sorted(data.get("conversation_history", []), key=lambda e: e.get("round", 0)):
        round_id = entry.get("round")
        sent_pre = _as_float(entry.get("sent_decision"))
        sent_post = _as_float(entry.get("sent"))
        returned_pre = _as_float(entry.get("returned_decision"))
        returned_post = _as_float(entry.get("returned"))
        if None in (round_id, sent_pre, sent_post, returned_pre, returned_post):
            continue
        rows.append((int(round_id), sent_pre, sent_post, returned_pre, returned_post))

        balances = entry.get("balances") or {}
        a1 = _as_float(balances.get("Agent_1"))
        a2 = _as_float(balances.get("Agent_2"))
        if a1 is not None and a2 is not None:
            final_balance = (a1 + a2) / 2.0

    if len(rows) < 10:
        return None

    arr = np.asarray(rows, dtype=float)
    return RunSeries(
        path=path,
        rounds=arr[:, 0],
        sent_pre=arr[:, 1],
        sent_post=arr[:, 2],
        returned_pre=arr[:, 3],
        returned_post=arr[:, 4],
        final_balance=float(final_balance),
    )


def collect_runs(experiment: str, task_order: str, noise_condition: str) -> list[RunSeries]:
    root = DATA_ROOT / VERSION / experiment / MODEL / task_order / noise_condition
    if not root.exists():
        return []
    runs = []
    for path in sorted(root.rglob("*.json")):
        if not is_final_json(path):
            continue
        run = load_run(path)
        if run is not None:
            runs.append(run)
    return runs


def stack_values(runs: list[RunSeries], attr: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_max = max(len(run.rounds) for run in runs)
    arr = np.full((len(runs), t_max), np.nan)
    for row, run in enumerate(runs):
        values = getattr(run, attr)
        arr[row, : len(values)] = values
    rounds = np.arange(1, t_max + 1)
    mean = np.nanmean(arr, axis=0)
    if len(runs) > 1:
        se = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(len(runs))
    else:
        se = np.zeros_like(mean)
    return rounds, mean, se


def style_axis(ax, ylim: tuple[float, float], title: str | None = None) -> None:
    if title:
        ax.set_title(title, fontsize=9, pad=7)
    ax.set_xlim(1, 10)
    ax.set_xticks([1, 5, 10])
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.jpg", dpi=240, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_average_grid(all_runs: dict[tuple[str, str], list[RunSeries]]) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(10.8, 8.2), sharex=True)

    for row_idx, noise_cell in enumerate(NOISE_CELLS):
        for kind_idx, (kind, kind_label, ylim) in enumerate(TRANSFER_KINDS):
            ax_row = row_idx * 2 + kind_idx
            for col_idx, (task_key, task_label) in enumerate(TASKS):
                ax = axes[ax_row, col_idx]
                runs = all_runs[(noise_cell["key"], task_key)]
                pre_attr = f"{kind}_pre"
                post_attr = f"{kind}_post"

                title = f"{task_label}\nn={len(runs)}" if kind_idx == 0 else None
                if runs:
                    rounds, pre_mean, pre_se = stack_values(runs, pre_attr)
                    _, post_mean, post_se = stack_values(runs, post_attr)
                    ax.plot(rounds, pre_mean, color=PRE_COLOR, linewidth=2.0, label=PRE_LABEL)
                    ax.fill_between(rounds, pre_mean - pre_se, pre_mean + pre_se, color=PRE_COLOR, alpha=0.12, linewidth=0)
                    ax.plot(
                        rounds,
                        post_mean,
                        color=noise_cell["post_color"],
                        linestyle="--",
                        linewidth=2.0,
                        label=POST_LABEL,
                    )
                    ax.fill_between(
                        rounds,
                        post_mean - post_se,
                        post_mean + post_se,
                        color=noise_cell["post_color"],
                        alpha=0.12,
                        linewidth=0,
                    )
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=9)

                style_axis(ax, ylim, title)
                if col_idx == 0:
                    ax.set_ylabel(f"{noise_cell['label']}\n{kind_label}", fontsize=9)

    fig.suptitle("Claude Sonnet 4.5 reported fig1 cells: mean chosen transfers vs post-noise ledger values", fontsize=13, y=0.99)
    fig.supxlabel("Round", fontsize=11, y=0.055)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0.02, 0.08, 1, 0.955])
    save(fig, "fig1_claude_noise_values_average")


def plot_individual_overlay_grid(all_runs: dict[tuple[str, str], list[RunSeries]]) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(10.8, 8.2), sharex=True)

    for row_idx, noise_cell in enumerate(NOISE_CELLS):
        for kind_idx, (kind, kind_label, ylim) in enumerate(TRANSFER_KINDS):
            ax_row = row_idx * 2 + kind_idx
            for col_idx, (task_key, task_label) in enumerate(TASKS):
                ax = axes[ax_row, col_idx]
                runs = all_runs[(noise_cell["key"], task_key)]
                pre_attr = f"{kind}_pre"
                post_attr = f"{kind}_post"

                title = f"{task_label}\nn={len(runs)}" if kind_idx == 0 else None
                for run in runs:
                    ax.plot(run.rounds, getattr(run, pre_attr), color=PRE_COLOR, linewidth=0.9, alpha=0.17)
                    ax.plot(
                        run.rounds,
                        getattr(run, post_attr),
                        color=noise_cell["post_color"],
                        linestyle="--",
                        linewidth=0.9,
                        alpha=0.20,
                    )

                if runs:
                    rounds, pre_mean, _ = stack_values(runs, pre_attr)
                    _, post_mean, _ = stack_values(runs, post_attr)
                    ax.plot(rounds, pre_mean, color=PRE_COLOR, linewidth=2.2, label=PRE_LABEL)
                    ax.plot(
                        rounds,
                        post_mean,
                        color=noise_cell["post_color"],
                        linestyle="--",
                        linewidth=2.2,
                        label=POST_LABEL,
                    )
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=9)

                style_axis(ax, ylim, title)
                if col_idx == 0:
                    ax.set_ylabel(f"{noise_cell['label']}\n{kind_label}", fontsize=9)

    fig.suptitle("Claude Sonnet 4.5 reported fig1 cells: individual runs with mean overlay", fontsize=13, y=0.99)
    fig.supxlabel("Round", fontsize=11, y=0.055)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0.02, 0.08, 1, 0.955])
    save(fig, "fig1_claude_noise_values_individual_runs")


def representative_run(runs: list[RunSeries]) -> RunSeries | None:
    if not runs:
        return None
    finite = [run for run in runs if np.isfinite(run.final_balance)]
    candidates = finite if finite else runs
    values = np.asarray([run.final_balance for run in candidates], dtype=float)
    if np.all(np.isfinite(values)):
        median = float(np.median(values))
        return min(candidates, key=lambda run: abs(run.final_balance - median))
    return candidates[len(candidates) // 2]


def compact_stem(path: Path) -> str:
    stem = path.stem
    for prefix in ("noise_positive_mem3_claude_sonnet_45_", "noise_negative_mem3_claude_sonnet_45_"):
        if stem.startswith(prefix):
            return stem.removeprefix(prefix)
    return stem


def plot_representative_grid(all_runs: dict[tuple[str, str], list[RunSeries]]) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(10.8, 8.2), sharex=True)

    for row_idx, noise_cell in enumerate(NOISE_CELLS):
        for kind_idx, (kind, kind_label, ylim) in enumerate(TRANSFER_KINDS):
            ax_row = row_idx * 2 + kind_idx
            for col_idx, (task_key, task_label) in enumerate(TASKS):
                ax = axes[ax_row, col_idx]
                run = representative_run(all_runs[(noise_cell["key"], task_key)])
                pre_attr = f"{kind}_pre"
                post_attr = f"{kind}_post"

                if run is not None:
                    ax.plot(run.rounds, getattr(run, pre_attr), color=PRE_COLOR, linewidth=2.0, label=PRE_LABEL)
                    ax.plot(
                        run.rounds,
                        getattr(run, post_attr),
                        color=noise_cell["post_color"],
                        linestyle="--",
                        linewidth=2.0,
                        label=POST_LABEL,
                    )
                    title = f"{task_label}\n{compact_stem(run.path)}" if kind_idx == 0 else None
                else:
                    title = f"{task_label}\nNo data" if kind_idx == 0 else None
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=9)

                style_axis(ax, ylim, title)
                if col_idx == 0:
                    ax.set_ylabel(f"{noise_cell['label']}\n{kind_label}", fontsize=9)

    fig.suptitle("Claude Sonnet 4.5 reported fig1 cells: representative individual runs", fontsize=13, y=0.99)
    fig.supxlabel("Round", fontsize=11, y=0.055)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0.02, 0.08, 1, 0.955])
    save(fig, "fig1_claude_noise_values_representative_runs")


def save_per_run_plot(run: RunSeries, noise_label: str, task_label: str, post_color: str, output_stem: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True)
    series = [
        ("sent", "Sent", (-0.25, 5.25)),
        ("returned", "Returned", (-0.75, 15.75)),
    ]
    for ax, (kind, label, ylim) in zip(axes, series):
        ax.plot(run.rounds, getattr(run, f"{kind}_pre"), color=PRE_COLOR, linewidth=2.0, label=PRE_LABEL)
        ax.plot(run.rounds, getattr(run, f"{kind}_post"), color=post_color, linestyle="--", linewidth=2.0, label=POST_LABEL)
        style_axis(ax, ylim)
        ax.set_ylabel(label)

    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle(f"Claude Sonnet 4.5 | {noise_label} | {task_label}\n{run.path.name}", fontsize=10, y=0.995)
    fig.supxlabel("Round", fontsize=10, y=0.04)
    fig.tight_layout(rect=[0.02, 0.08, 1, 0.93])
    for ext in ("png", "jpg"):
        fig.savefig(RUN_OUT_DIR / f"{output_stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_all_per_run_plots(all_runs: dict[tuple[str, str], list[RunSeries]]) -> None:
    for noise_cell in NOISE_CELLS:
        for task_key, task_label in TASKS:
            runs = all_runs[(noise_cell["key"], task_key)]
            for idx, run in enumerate(runs, start=1):
                output_stem = f"{noise_cell['key']}_{task_key}_{idx:02d}_{run.path.stem}"
                save_per_run_plot(run, noise_cell["label"], task_label, noise_cell["post_color"], output_stem)


def main() -> None:
    all_runs: dict[tuple[str, str], list[RunSeries]] = {}
    for noise_cell in NOISE_CELLS:
        for task_key, _task_label in TASKS:
            all_runs[(noise_cell["key"], task_key)] = collect_runs(
                noise_cell["experiment"],
                task_key,
                noise_cell["condition"],
            )

    plot_average_grid(all_runs)
    plot_individual_overlay_grid(all_runs)
    plot_representative_grid(all_runs)
    save_all_per_run_plots(all_runs)

    for noise_cell in NOISE_CELLS:
        for task_key, task_label in TASKS:
            print(f"{noise_cell['label']}, {task_label}: n={len(all_runs[(noise_cell['key'], task_key)])}")
    print(f"Wrote summary figures to {OUT_DIR}")
    print(f"Wrote individual run figures to {RUN_OUT_DIR}")


if __name__ == "__main__":
    main()
