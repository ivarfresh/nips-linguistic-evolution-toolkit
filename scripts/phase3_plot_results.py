"""Phase 3 result plots.

Produces:
  data/phase3/plots/01_cell_means.png — joint balance by cell, n=5 each, with
  per-rep strip dots overlaid on the bar means.

Phase 3 has one task order (`["game"]`) and three cells:
  - baseline (no seed, myth-only chat memory)
  - s_start (round-1 directive parable from team baseline)
  - s_end_plus (round-10 myth from team baseline)
"""

import glob
import json
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
    "baseline": "Baseline\n(no seed)",
    "s_start": "S-start\n(round-1 parable)",
    "s_end_plus": "S-end+\n(round-10 myth)",
}
CELL_COLORS = {
    "baseline": "#7f7f7f",
    "s_start": "#1f77b4",
    "s_end_plus": "#2ca02c",
}

PATHS = {
    "baseline": "data/json/noise_experiments/phase3_baseline/phase3_seeded_baseline_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json",
    "s_start": "data/json/noise_experiments/phase3_seeded/phase3_seeded_s_start_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json",
    "s_end_plus": "data/json/noise_experiments/phase3_seeded/phase3_seeded_s_end_plus_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json",
}

OUT_DIR = Path("data/phase3/plots")
CEILING = 600.0


def _joints(cell):
    files = sorted(glob.glob(PATHS[cell]))
    files = [f for f in files if ".results." not in f and ".checkpoint." not in f]
    joints = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        ch = d.get("conversation_history", [])
        bal = ch[-1].get("balances", {}) if ch else {}
        if bal:
            joints.append(sum(bal.values()))
    return joints


def plot_cell_means(out_path):
    fig, ax = plt.subplots(figsize=(8, 6))

    data = {cell: _joints(cell) for cell in CELL_ORDER}
    x = np.arange(len(CELL_ORDER))

    means = [statistics.mean(data[c]) for c in CELL_ORDER]
    stds = [statistics.stdev(data[c]) if len(data[c]) > 1 else 0 for c in CELL_ORDER]
    ns = [len(data[c]) for c in CELL_ORDER]

    bars = ax.bar(
        x,
        means,
        width=0.55,
        yerr=stds,
        capsize=5,
        color=[CELL_COLORS[c] for c in CELL_ORDER],
        edgecolor="black",
        linewidth=0.7,
        alpha=0.85,
    )

    # Strip plot overlay: one dot per rep
    rng = np.random.default_rng(seed=0)
    for i, c in enumerate(CELL_ORDER):
        jitter = rng.uniform(-0.10, 0.10, size=len(data[c]))
        ax.scatter(
            [x[i] + dx for dx in jitter],
            data[c],
            s=42,
            color="black",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

    ax.axhline(CEILING, color="red", linestyle="--", alpha=0.6, label=f"Cooperation ceiling (${int(CEILING)})")

    # Cell-mean labels above each bar
    for i, (m, s, n) in enumerate(zip(means, stds, ns)):
        ax.text(
            x[i],
            m + max(s, 6) + 8,
            f"${m:.1f}\n(±{s:.1f})\nn={n}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CELL_LABELS[c] for c in CELL_ORDER], fontsize=11)
    ax.set_ylabel("Joint balance after 10 rounds (sum across 8 agents)")
    ax.set_title(
        "Phase 3: cell means — joint balance by seed condition\n"
        "Sonnet 4.5  •  8-agent  •  myth-only chat memory  •  history-block=none\n"
        "noise neg5  •  task order=[\"game\"]  •  n=5 per cell",
        fontsize=11,
    )
    ax.set_ylim(0, 680)
    ax.legend(loc="center right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_cell_means(OUT_DIR / "01_cell_means.png")


if __name__ == "__main__":
    main()
