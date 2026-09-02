"""Phase 5 5-cell result plot.

5 cells × n=5 each, ordered by mean joint balance (left = most suppressive,
right = highest cooperation). Per-rep dots overlaid. Ceiling line at $600.

Output: data/phase5/plots/01_cell_means.png
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


BASE = "data/json/noise_experiments"
CELLS = [
    # (key, label, color, glob)
    ("s_end_minus", "S-end−\n(low-coop source)",       "#d62728",
     f"{BASE}/phase5_seeded/phase3_seeded_s_end_minus_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"),
    ("baseline",    "Baseline\n(no seed)",              "#7f7f7f",
     f"{BASE}/phase3_baseline/phase3_seeded_baseline_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"),
    ("s_filler",    "S-filler\n(Wikipedia)",            "#9467bd",
     f"{BASE}/phase5_seeded/phase3_seeded_s_filler_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"),
    ("s_start",     "S-start\n(round-1 parable)",       "#1f77b4",
     f"{BASE}/phase3_seeded/phase3_seeded_s_start_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"),
    ("s_end_plus",  "S-end+\n(round-10 myth)",          "#2ca02c",
     f"{BASE}/phase3_seeded/phase3_seeded_s_end_plus_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"),
]

OUT_PATH = Path("data/phase5/plots/01_cell_means.png")
CEILING = 600.0


def joint_of(path):
    with open(path) as f:
        d = json.load(f)
    ch = d.get("conversation_history", [])
    bal = ch[-1].get("balances", {}) if ch else {}
    return sum(bal.values()) if bal else None


def collect_joints(pattern):
    files = sorted(glob.glob(pattern))
    files = [f for f in files if ".results." not in f and ".checkpoint." not in f]
    out = []
    for f in files:
        j = joint_of(f)
        if j is not None:
            out.append(j)
    return out


def main():
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(13, 7))

    x = np.arange(len(CELLS))
    rng = np.random.default_rng(seed=4)

    for i, (key, label, color, pat) in enumerate(CELLS):
        joints = collect_joints(pat)
        n = len(joints)
        mean = statistics.mean(joints)
        sd = statistics.stdev(joints) if n > 1 else 0

        ax.bar(
            x[i], mean, width=0.62, yerr=sd, capsize=5,
            color=color, edgecolor="black", linewidth=0.7, alpha=0.88,
        )

        # per-rep dots
        jitter = rng.uniform(-0.1, 0.1, size=n)
        ax.scatter(
            [x[i] + dx for dx in jitter], joints,
            s=46, color="black", edgecolor="white", linewidth=0.8, zorder=3,
        )

        # value label
        ax.text(
            x[i], mean + sd + 12,
            f"${mean:.0f}\n(±${sd:.0f})\nn={n}",
            ha="center", va="bottom", fontsize=10,
        )

    # Ceiling
    ax.axhline(CEILING, color="red", linestyle="--", alpha=0.55,
               label=f"Cooperation ceiling (${int(CEILING)})")
    # Baseline reference line for visual contrast
    baseline_mean = statistics.mean(collect_joints(CELLS[1][3]))
    ax.axhline(baseline_mean, color="#7f7f7f", linestyle=":", alpha=0.6, linewidth=1.5,
               label=f"Baseline mean (${baseline_mean:.0f})")

    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in CELLS], fontsize=10.5)
    ax.set_ylabel("Joint balance after 10 rounds (sum across 8 agents)")
    ax.set_ylim(0, 690)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax.set_title(
        "Phase 5 — seed content matters: five distinct behavioral regimes from changing only the chat-memory text\n"
        "Sonnet 4.5 · 8-agent · myth-only chat memory · history-block=none · noise neg5 · task order=[\"game\"] · n=5 per cell\n"
        "Ordered by mean joint balance: suppression → no effect → moderate lift → ceiling saturation.",
        fontsize=11,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
