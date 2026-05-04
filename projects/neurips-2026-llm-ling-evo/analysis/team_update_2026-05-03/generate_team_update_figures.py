#!/usr/bin/env python3
"""Generate figures for the 2026-05-03 team update packet."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data" / "json" / "noise_experiments"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_COLORS = {
    "game": "#6b7280",
    "game_myth": "#2563eb",
    "myth_game": "#dc2626",
}


def is_final_json(path: Path) -> bool:
    name = path.name
    return (
        name.endswith(".json")
        and ".checkpoint" not in name
        and ".results" not in name
        and ".error" not in name
    )


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


def collect(version: str, experiment: str, model: str, task_order: str, noise_cond: str) -> list[np.ndarray]:
    root = DATA_ROOT / version / experiment / model / task_order / noise_cond
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


def plot_line(
    ax,
    trajs: list[np.ndarray],
    label: str,
    color: str,
    linestyle: str = "-",
    linewidth: float = 2.0,
    include_n: bool = True,
):
    stats = mean_and_se(trajs)
    if stats is None:
        ax.text(0.5, 0.5, f"No data\n{label}", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        return
    mean, se = stats
    rounds = np.arange(1, len(mean) + 1)
    line_label = f"{label} (n={len(trajs)})" if include_n else label
    ax.plot(rounds, mean, label=line_label, color=color, linestyle=linestyle, linewidth=linewidth)
    ax.fill_between(rounds, mean - se, mean + se, color=color, alpha=0.14, linewidth=0)


def style_axis(ax, title: str | None = None):
    if title:
        ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlim(1, 10)
    ax.set_xticks([1, 5, 10])
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, stem: str):
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_1_main_trajectories():
    cells = [
        ("GPT-5-Nano", "Positive noise", "v4_direct_provider", "noise_positive_mem3_gpt5_nano", "gpt-5-nano", "noisy_positive_5"),
        ("GPT-5-Nano", "Negative noise", "v4_direct_provider", "noise_negative_mem3_gpt5_nano", "gpt-5-nano", "noisy_negative_5"),
        ("GPT-5-Nano", "Bootstrap noise", "v4_direct_provider", "noise_bootstrap_mem3", "gpt-5-nano", "noisy_bootstrap_cooperation"),
        ("Claude Sonnet 4.5", "Positive noise", "v4_direct_provider", "noise_positive_mem3_claude_sonnet_45", "claude-sonnet-4.5", "noisy_positive_5"),
        ("Claude Sonnet 4.5", "Negative noise", "v4_direct_provider", "noise_negative_mem3_claude_sonnet_45", "claude-sonnet-4.5", "noisy_negative_5"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0), sharex=True, sharey=True)
    axes_by_key = {
        ("GPT-5-Nano", "Positive noise"): axes[0, 0],
        ("GPT-5-Nano", "Negative noise"): axes[0, 1],
        ("GPT-5-Nano", "Bootstrap noise"): axes[0, 2],
        ("Claude Sonnet 4.5", "Positive noise"): axes[1, 0],
        ("Claude Sonnet 4.5", "Negative noise"): axes[1, 1],
    }
    for model_name, noise_name, version, experiment, model, noise_cond in cells:
        ax = axes_by_key[(model_name, noise_name)]
        for task in ["game", "game_myth", "myth_game"]:
            plot_line(
                ax,
                collect(version, experiment, model, task, noise_cond),
                {"game": "Game only", "game_myth": "Game -> myth", "myth_game": "Myth -> game"}[task],
                TASK_COLORS[task],
            )
        style_axis(ax, f"{model_name}\n{noise_name}")
    axes[1, 2].axis("off")
    fig.suptitle("Original v4 trajectories: myth effects are conditional, not uniform", fontsize=13, y=0.99)
    fig.supxlabel("Round", fontsize=11, y=0.09)
    fig.supylabel("Mean cumulative reward per agent", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0.02, 0.14, 1, 0.95])
    save(fig, "fig1_main_v4_trajectories")


def figure_2_bootstrap_mechanism():
    specs = [
        ("Game only", "#4b5563", "v4_direct_provider", "noise_bootstrap_mem3", "gpt-5-nano", "game", "noisy_bootstrap_cooperation", "-", 2.3),
        ("Game -> myth, no story in game prompt", "#dc2626", "v4_direct_provider", "noise_bootstrap_mem3", "gpt-5-nano", "game_myth", "noisy_bootstrap_cooperation", "-", 2.5),
        ("Real partner myth shown as partner story", "#2563eb", "v4_direct_provider_A1_partner_myth", "gpt5nano_partner_myth_injection", "gpt-5-nano", "game_myth", "noisy_bootstrap_cooperation", "-", 2.5),
        ("Other dyad myth shown as partner story", "#16a34a", "v4_direct_provider_controls", "gpt5nano_partner_myth_shuffled_bootstrap", "gpt-5-nano", "game_myth", "noisy_bootstrap_cooperation", "--", 2.2),
        ("Filler paragraph shown as partner story", "#9333ea", "v4_direct_provider_controls", "gpt5nano_partner_myth_filler_bootstrap", "gpt-5-nano", "game_myth", "noisy_bootstrap_cooperation", "--", 2.2),
        ("Own myth shown as partner story", "#f97316", "v4_direct_provider_controls", "gpt5nano_partner_myth_own_bootstrap", "gpt-5-nano", "game_myth", "noisy_bootstrap_cooperation", ":", 2.4),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for label, color, version, experiment, model, task, noise_cond, linestyle, linewidth in specs:
        plot_line(ax, collect(version, experiment, model, task, noise_cond), label, color, linestyle, linewidth)
    style_axis(ax, "GPT-5-Nano bootstrap: text labelled as partner story restores cooperation")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean cumulative reward per agent")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "fig2_bootstrap_mechanism_game_myth")


def figure_3_boundary_conditions():
    panels = [
        (
            "No noise",
            ("v4_direct_provider_baseline", "baseline_v4_mem3_direct", "gpt-5-nano", "game_myth", "default"),
            ("v4_direct_provider_A1_no_noise", "gpt5nano_partner_myth_no_noise", "gpt-5-nano", "game_myth", "default"),
            ("v4_direct_provider_controls", "gpt5nano_partner_myth_filler_no_noise", "gpt-5-nano", "game_myth", "default"),
        ),
        (
            "Positive noise",
            ("v4_direct_provider", "noise_positive_mem3_gpt5_nano", "gpt-5-nano", "game_myth", "noisy_positive_5"),
            ("v4_direct_provider_A1_partner_myth", "gpt5nano_partner_myth_injection", "gpt-5-nano", "game_myth", "noisy_positive_5"),
            ("v4_direct_provider_controls", "gpt5nano_partner_myth_filler_positive_5", "gpt-5-nano", "game_myth", "noisy_positive_5"),
        ),
        (
            "Negative noise",
            ("v4_direct_provider", "noise_negative_mem3_gpt5_nano", "gpt-5-nano", "game_myth", "noisy_negative_5"),
            ("v4_direct_provider_A1_partner_myth", "gpt5nano_partner_myth_injection", "gpt-5-nano", "game_myth", "noisy_negative_5"),
            ("v4_direct_provider_controls", "gpt5nano_partner_myth_filler_negative_5", "gpt-5-nano", "game_myth", "noisy_negative_5"),
        ),
        (
            "Bootstrap noise",
            ("v4_direct_provider", "noise_bootstrap_mem3", "gpt-5-nano", "game_myth", "noisy_bootstrap_cooperation"),
            ("v4_direct_provider_A1_partner_myth", "gpt5nano_partner_myth_injection", "gpt-5-nano", "game_myth", "noisy_bootstrap_cooperation"),
            ("v4_direct_provider_controls", "gpt5nano_partner_myth_filler_bootstrap", "gpt-5-nano", "game_myth", "noisy_bootstrap_cooperation"),
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.6), sharex=True, sharey=True)
    for ax, (title, no_inj, partner, filler) in zip(axes.ravel(), panels):
        plot_line(ax, collect(*no_inj), "No story in game prompt", "#dc2626", "-", 2.3)
        plot_line(ax, collect(*partner), "Real partner myth shown", "#2563eb", "-", 2.3)
        plot_line(ax, collect(*filler), "Filler shown as partner story", "#9333ea", "--", 2.3)
        style_axis(ax, title)
    fig.suptitle("GPT-5-Nano boundary check: partner-story text changes bootstrap, not every regime", fontsize=13, y=0.99)
    fig.supxlabel("Round", fontsize=11, y=0.09)
    fig.supylabel("Mean cumulative reward per agent", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0.02, 0.14, 1, 0.95])
    save(fig, "fig3_boundary_conditions_game_myth")


def figure_4_neutral_framing():
    panels = [
        (
            "GPT-5-Nano\nstandard investor/trustee",
            "v4_direct_provider_baseline",
            "baseline_v4_mem3_direct",
            "gpt-5-nano",
        ),
        (
            "GPT-5-Nano\nneutral ROLE A/B",
            "v4_direct_provider_neutral",
            "neutral_framing_v4_mem3",
            "gpt-5-nano",
        ),
        (
            "Claude Sonnet 4.5\nstandard investor/trustee",
            "v4_direct_provider_baseline",
            "baseline_v4_mem3_direct",
            "claude-sonnet-4.5",
        ),
        (
            "Claude Sonnet 4.5\nneutral ROLE A/B",
            "v4_direct_provider_neutral",
            "neutral_framing_v4_mem3",
            "claude-sonnet-4.5",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.6), sharex=True, sharey=True)
    for ax, (title, version, experiment, model) in zip(axes.ravel(), panels):
        for task in ["game", "game_myth", "myth_game"]:
            plot_line(
                ax,
                collect(version, experiment, model, task, "default"),
                {"game": "Game only", "game_myth": "Game -> myth", "myth_game": "Myth -> game"}[task],
                TASK_COLORS[task],
                include_n=False,
            )
        style_axis(ax, title)
    fig.suptitle("Neutral framing check: ROLE A/B lowers GPT-5-Nano and exposes a myth lift", fontsize=13, y=0.99)
    fig.text(0.985, 0.145, "All cells n=15 except Claude neutral myth -> game: n=14", ha="right", va="bottom", fontsize=7, color="#4b5563")
    fig.supxlabel("Round", fontsize=11, y=0.09)
    fig.supylabel("Mean cumulative reward per agent", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0.02, 0.14, 1, 0.95])
    save(fig, "fig4_neutral_framing_trajectories")


def main():
    figure_1_main_trajectories()
    figure_2_bootstrap_mechanism()
    figure_3_boundary_conditions()
    figure_4_neutral_framing()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
