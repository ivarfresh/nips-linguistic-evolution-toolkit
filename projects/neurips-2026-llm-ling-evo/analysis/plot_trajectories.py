#!/usr/bin/env python3
"""Per-round cumulative balance trajectories, faceted by (model x noise),
with one line per task_order.

Useful for §3.1/§3.2: shows the cross-model regimes and the noise effect
without conflating task orders.

Output: figures/trajectories.png
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
JSON_ROOT = REPO_ROOT / "data" / "json" / "noise_experiments" / "v4_direct_provider"
ANALYSIS_DIR = Path(__file__).parent
FIG_DIR = ANALYSIS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

INCLUDE_MODELS = ["claude-sonnet-4.5", "gpt-5-nano"]
NOISE_LABEL_FROM_EXPERIMENT = {
    "noise_bootstrap_mem3": "bootstrap",
    "noise_negative_mem3_gpt5_nano": "negative_5",
    "noise_negative_mem3_claude_sonnet_45": "negative_5",
    "noise_positive_mem3_claude_sonnet_45": "positive",
    "noise_positive_mem3_gpt5_nano": "positive",
    "noise_deterministic_max_mem3_gpt5_nano": "deterministic_max",
}
NOISE_ORDER = ["positive", "negative_5", "bootstrap", "deterministic_max"]
TASK_ORDER_COLORS = {
    "game": "#999999",
    "game_myth": "#1f77b4",
    "myth_game": "#d62728",
}


def load_balance_trajectory(path: Path):
    """Returns array of round-by-round mean(Agent_1, Agent_2) balances."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    rounds = []
    for entry in sorted(
        data.get("conversation_history", []),
        key=lambda e: e.get("round", 0),
    ):
        b = entry.get("balances") or {}
        a1 = b.get("Agent_1")
        a2 = b.get("Agent_2")
        if a1 is None or a2 is None:
            continue
        rounds.append(0.5 * (a1 + a2))
    return np.array(rounds, dtype=float) if rounds else None


def collect():
    """Returns nested dict: model -> noise_label -> task_order -> list of trajectories."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in JSON_ROOT.rglob("*.json"):
        n = path.name
        if ".checkpoint" in n or ".results" in n or ".error" in n:
            continue
        rel = path.relative_to(JSON_ROOT)
        # Expected: experiment / model / task_order / noise_cond / file
        parts = rel.parts
        if len(parts) < 5:
            continue
        experiment, model, task_order = parts[0], parts[1], parts[2]
        if model not in INCLUDE_MODELS:
            continue
        noise_label = NOISE_LABEL_FROM_EXPERIMENT.get(experiment, experiment)
        traj = load_balance_trajectory(path)
        if traj is None or len(traj) < 5:
            continue
        out[model][noise_label][task_order].append(traj)
    return out


def main():
    data = collect()
    # Grid: rows = models, cols = noise types.
    # Only plot noise types that exist for at least one model.
    noise_types = [
        n for n in NOISE_ORDER
        if any(n in data[m] for m in INCLUDE_MODELS)
    ]
    n_rows = len(INCLUDE_MODELS)
    n_cols = len(noise_types)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.6 * n_cols, 2.4 * n_rows),
        sharey=True, sharex=True, squeeze=False,
    )
    for i, model in enumerate(INCLUDE_MODELS):
        for j, noise in enumerate(noise_types):
            ax = axes[i, j]
            cell = data[model].get(noise, {})
            for task_order in ["game", "game_myth", "myth_game"]:
                trajs = cell.get(task_order, [])
                if not trajs:
                    continue
                T = max(len(t) for t in trajs)
                arr = np.full((len(trajs), T), np.nan)
                for k, t in enumerate(trajs):
                    arr[k, :len(t)] = t
                mean = np.nanmean(arr, axis=0)
                std = np.nanstd(arr, axis=0)
                rounds = np.arange(1, T + 1)
                ax.plot(
                    rounds, mean,
                    color=TASK_ORDER_COLORS[task_order],
                    label=task_order, linewidth=1.5,
                )
                ax.fill_between(
                    rounds, mean - std, mean + std,
                    color=TASK_ORDER_COLORS[task_order], alpha=0.18,
                )
            if i == 0:
                ax.set_title(noise.replace("_", " "), fontsize=10)
            if j == 0:
                ax.set_ylabel(model.split("-")[0], fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xticks([1, 5, 10])
    # Single legend.
    handles = [
        plt.Line2D([], [], color=TASK_ORDER_COLORS[t], label=t)
        for t in ["game", "game_myth", "myth_game"]
    ]
    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.005), ncol=3, frameon=False, fontsize=9,
    )
    fig.suptitle(
        "Mean cumulative balance over rounds, by model × noise × task order\n"
        "v4_direct_provider — shaded ±1 std across seeds",
        fontsize=11,
    )
    fig.text(0.5, 0.04, "Round", ha="center", fontsize=10)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = FIG_DIR / "trajectories.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(FIG_DIR / "trajectories.pdf", bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
