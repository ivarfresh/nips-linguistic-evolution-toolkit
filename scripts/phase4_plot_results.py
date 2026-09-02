"""Phase 4 result plot.

Grouped bar chart comparing joint balance across:
  - 3 task orders (P3 `["game"]`, P4 `["myth","game"]`, P4 `["game","myth"]`)
  - 3 seed cells (baseline, S-start, S-end+)
Each bar = mean of n=5 reps. Whiskers = ±sd. Black dots overlay = individual reps.

Output: data/phase4/plots/01_task_order_comparison.png
"""

import glob
import json
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyses._shared import configure_matplotlib


CELL_ORDER = ["baseline", "s_start", "s_end_plus"]
CELL_LABELS = {
    "baseline": "Baseline (no seed)",
    "s_start": "S-start (round-1 parable)",
    "s_end_plus": "S-end+ (round-10 myth)",
}
CELL_COLORS = {
    "baseline": "#7f7f7f",
    "s_start": "#1f77b4",
    "s_end_plus": "#2ca02c",
}

TASK_ORDER_LABELS = {
    "game": '["game"]\n(no myth-writing)',
    "myth_game": '["myth","game"]\n(write then play)',
    "game_myth": '["game","myth"]\n(play then write)',
}

BASE = "data/json/noise_experiments"
CELLS = {
    # (task_order_key, seed_cell): glob pattern
    ("game", "baseline"):     f"{BASE}/phase3_baseline/phase3_seeded_baseline_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json",
    ("game", "s_start"):      f"{BASE}/phase3_seeded/phase3_seeded_s_start_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json",
    ("game", "s_end_plus"):   f"{BASE}/phase3_seeded/phase3_seeded_s_end_plus_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json",
    ("myth_game", "baseline"):    f"{BASE}/phase4_baseline/phase3_seeded_baseline_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/myth_game/default/*.json",
    ("myth_game", "s_start"):     f"{BASE}/phase4_seeded/phase3_seeded_s_start_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/myth_game/default/*.json",
    ("myth_game", "s_end_plus"):  f"{BASE}/phase4_seeded/phase3_seeded_s_end_plus_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/myth_game/default/*.json",
    ("game_myth", "baseline"):    f"{BASE}/phase4_baseline/phase3_seeded_baseline_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game_myth/default/*.json",
    ("game_myth", "s_start"):     f"{BASE}/phase4_seeded/phase3_seeded_s_start_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game_myth/default/*.json",
    ("game_myth", "s_end_plus"):  f"{BASE}/phase4_seeded/phase3_seeded_s_end_plus_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game_myth/default/*.json",
}

OUT_PATH = Path("data/phase4/plots/01_task_order_comparison.png")
CEILING = 600.0


def joint_of(path):
    with open(path) as f:
        d = json.load(f)
    ch = d.get("conversation_history", [])
    bal = ch[-1].get("balances", {}) if ch else {}
    return sum(bal.values()) if bal else None


def collect():
    out = {}
    for (task_order, cell), pat in CELLS.items():
        files = sorted(glob.glob(pat))
        files = [f for f in files if ".results." not in f and ".checkpoint." not in f]
        js = []
        for f in files:
            j = joint_of(f)
            if j is not None:
                js.append(j)
        out[(task_order, cell)] = js
    return out


def main():
    configure_matplotlib()
    data = collect()

    task_orders = ["game", "myth_game", "game_myth"]

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(task_orders))
    width = 0.27
    rng = np.random.default_rng(seed=3)

    for i, cell in enumerate(CELL_ORDER):
        means = [statistics.mean(data[(t, cell)]) for t in task_orders]
        stds = [statistics.stdev(data[(t, cell)]) if len(data[(t, cell)]) > 1 else 0 for t in task_orders]
        offset = (i - 1) * width
        bar_positions = x + offset
        ax.bar(
            bar_positions,
            means,
            width,
            yerr=stds,
            capsize=4,
            color=CELL_COLORS[cell],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.88,
            label=CELL_LABELS[cell],
        )
        # individual rep dots overlay
        for j, t in enumerate(task_orders):
            vals = data[(t, cell)]
            jitter = rng.uniform(-0.07, 0.07, size=len(vals))
            ax.scatter(
                [bar_positions[j] + dx for dx in jitter],
                vals,
                s=28,
                color="black",
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
        # value labels above each bar
        for j, (m, s) in enumerate(zip(means, stds)):
            ax.text(
                bar_positions[j],
                m + s + 8,
                f"${m:.0f}\n±${s:.0f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax.axhline(CEILING, color="red", linestyle="--", alpha=0.55, label=f"Ceiling (${int(CEILING)})")

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_ORDER_LABELS[t] for t in task_orders], fontsize=10)
    ax.set_ylabel("Joint balance after 10 rounds (sum across 8 agents)")
    ax.set_ylim(0, 690)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax.set_title(
        "Phase 3 + Phase 4: seed effect across all three task orders\n"
        "Sonnet 4.5 · 8-agent · myth-only chat memory · history-block=none · noise neg5 · n=5 per cell\n"
        "Each task order shows the same monotonic lift baseline → S-start → S-end+.",
        fontsize=11,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
