"""Phase 7 — cell means with the gowith transplant cell.

Same layout/conventions as phase6_plot_results.py, adding s_end_plus_gowith
(n=4; rep 4 refusal-censored, see researchlog 2026-07-02).

Output: data/phase7/plots/04_cell_means_gowith.png
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

BASE = "data/json/noise_experiments"
# Same cells and order as phase6_plot_results.py (sorted by mean), with the
# Phase 7 gowith cell slotted in between S-end+ Gemini and S-end+ Sonnet.
CELLS = [
    ("S-end−\n(low-coop source)", "#d62728",
     f"{BASE}/phase5_seeded/phase3_seeded_s_end_minus_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"),
    ("Baseline\n(no seed)", "#7f7f7f",
     f"{BASE}/phase3_baseline/phase3_seeded_baseline_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"),
    ("S-filler\n(Wikipedia)", "#9467bd",
     f"{BASE}/phase5_seeded/phase3_seeded_s_filler_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"),
    ("S-end+ GPT\n(GPT-5-nano round-10)", "#ff7f0e",
     f"{BASE}/phase6_seeded/phase3_seeded_s_end_plus_gpt_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"),
    ("S-start\n(Sonnet round-1)", "#1f77b4",
     f"{BASE}/phase3_seeded/phase3_seeded_s_start_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"),
    ("S-end+ Gemini\n(Flash-Lite round-10)", "#17becf",
     f"{BASE}/phase6_seeded/phase3_seeded_s_end_plus_gemini_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"),
    ("S-end+ gowith\n(grammar transplant)", "#a3552e",
     f"{BASE}/phase7_seeded/phase3_seeded_s_end_plus_gowith_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"),
    ("S-end+ Sonnet\n(Sonnet round-10)", "#2ca02c",
     f"{BASE}/phase3_seeded/phase3_seeded_s_end_plus_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"),
]

OUT_PATH = Path("data/phase7/plots/04_cell_means_gowith.png")
CEILING = 600.0


def joint_of(path):
    with open(path) as f:
        d = json.load(f)
    ch = d.get("conversation_history", [])
    bal = ch[-1].get("balances", {}) if ch else {}
    return sum(bal.values()) if bal else None


def collect_joints(pattern):
    files = sorted(glob.glob(pattern))
    files = [f for f in files
             if ".results." not in f and ".checkpoint." not in f and ".error." not in f]
    return [j for j in (joint_of(f) for f in files) if j is not None]


def main():
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(CELLS))
    rng = np.random.default_rng(seed=7)

    baseline_mean = None
    for i, (label, color, pat) in enumerate(CELLS):
        joints = collect_joints(pat)
        if not joints:
            continue
        mean = statistics.mean(joints)
        sd = statistics.stdev(joints) if len(joints) > 1 else 0
        if label.startswith("Baseline"):
            baseline_mean = mean
        ax.bar(x[i], mean, width=0.62, yerr=sd, capsize=5,
               color=color, edgecolor="black", linewidth=0.7, alpha=0.88)
        jitter = rng.uniform(-0.1, 0.1, size=len(joints))
        ax.scatter([x[i] + dx for dx in jitter], joints,
                   s=42, color="black", edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(x[i], mean + sd + 12, f"${mean:.0f}\n(±${sd:.0f})\nn={len(joints)}",
                ha="center", va="bottom", fontsize=9.5)

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
        "Phase 3 + 5 + 6 + 7 cell means — the cooperation recipe survives translation into an alien grammar\n"
        "S-end+ myths rewritten in gowith (subject-less relational English, numbers pinned) reach \\$599 vs original \\$600\n"
        "Sonnet 4.5 game runs · myth-only chat memory · history-block=none · noise neg5 · n=5 per cell (gowith n=4, rep 4 refusal-censored)",
        fontsize=11,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
