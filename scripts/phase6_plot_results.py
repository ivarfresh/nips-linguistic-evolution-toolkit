"""Phase 6 — 6-cell result plot.

All Phase 3 + Phase 5 + Phase 6 cells, sorted by mean joint balance.
Adds S-end+_gemini next to S-end+ (Sonnet) to make the cross-model comparison
visually direct.

Output: data/phase6/plots/01_cell_means.png
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
    # (label, color, glob)
    ("S-end−\n(low-coop source)",       "#d62728",
     f"{BASE}/phase5_seeded/phase3_seeded_s_end_minus_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"),
    ("Baseline\n(no seed)",              "#7f7f7f",
     f"{BASE}/phase3_baseline/phase3_seeded_baseline_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"),
    ("S-filler\n(Wikipedia)",            "#9467bd",
     f"{BASE}/phase5_seeded/phase3_seeded_s_filler_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"),
    ("S-end+ GPT\n(GPT-5-nano round-10)", "#ff7f0e",
     f"{BASE}/phase6_seeded/phase3_seeded_s_end_plus_gpt_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"),
    ("S-start\n(Sonnet round-1)",        "#1f77b4",
     f"{BASE}/phase3_seeded/phase3_seeded_s_start_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"),
    ("S-end+ Gemini\n(Flash-Lite round-10)", "#17becf",
     f"{BASE}/phase6_seeded/phase3_seeded_s_end_plus_gemini_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"),
    ("S-end+ Sonnet\n(Sonnet round-10)", "#2ca02c",
     f"{BASE}/phase3_seeded/phase3_seeded_s_end_plus_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"),
]

OUT_PATH = Path("data/phase6/plots/01_cell_means.png")
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
    fig, ax = plt.subplots(figsize=(15, 7))

    x = np.arange(len(CELLS))
    rng = np.random.default_rng(seed=6)

    baseline_mean = None
    for i, (label, color, pat) in enumerate(CELLS):
        joints = collect_joints(pat)
        n = len(joints)
        if n == 0:
            continue
        mean = statistics.mean(joints)
        sd = statistics.stdev(joints) if n > 1 else 0
        if label.startswith("Baseline"):
            baseline_mean = mean

        ax.bar(
            x[i], mean, width=0.62, yerr=sd, capsize=5,
            color=color, edgecolor="black", linewidth=0.7, alpha=0.88,
        )
        jitter = rng.uniform(-0.1, 0.1, size=n)
        ax.scatter(
            [x[i] + dx for dx in jitter], joints,
            s=42, color="black", edgecolor="white", linewidth=0.8, zorder=3,
        )
        ax.text(
            x[i], mean + sd + 12,
            f"${mean:.0f}\n(±${sd:.0f})\nn={n}",
            ha="center", va="bottom", fontsize=9.5,
        )

    ax.axhline(CEILING, color="red", linestyle="--", alpha=0.55,
               label=f"Cooperation ceiling (${int(CEILING)})")
    if baseline_mean:
        ax.axhline(baseline_mean, color="#7f7f7f", linestyle=":", alpha=0.6, linewidth=1.4,
                   label=f"Baseline mean (${baseline_mean:.0f})")

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in CELLS], fontsize=10)
    ax.set_ylabel("Joint balance after 10 rounds (sum across 8 agents)")
    ax.set_ylim(0, 700)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax.set_title(
        "Phase 3 + 5 + 6 cell means — cross-model transfer is NOT universal\n"
        "Sonnet 4.5 game runs · myth-only chat memory · history-block=none · noise neg5 · n=5 per cell\n"
        "Gemini seeds transfer at \\$549. GPT-5-nano seeds do NOT transfer — indistinguishable from baseline and filler.",
        fontsize=11,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
