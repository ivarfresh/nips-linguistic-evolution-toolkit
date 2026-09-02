"""Phase 3 'puzzle' figure — the result plus the three explanatory metrics.

Single 1×4 row of bar charts comparing s_start vs s_end_plus on:
  (a) Phase 3 game outcome — joint balance ($) — the "result"
  (b) Cooperative word % — analyses/cooperativity_analysis.py
  (c) Mean sentence length — analyses/myth_compression_curves.py
  (d) Number of sentences — analyses/myth_compression_curves.py

Each bar = pool mean (±sd). Black dots overlay = individual myths.
Result: the round-10 pool produces MORE cooperation in the game, while
its myths contain FEWER cooperation words and have MORE / SHORTER
sentences — the structural-vs-verbal puzzle in one panel each.

Output: data/phase3/plots/04_puzzle.png
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


MANIFEST_PATH = Path("data/phase3/seed_manifest.json")
OUT_PATH = Path("data/phase3/plots/04_puzzle.png")

POOL_ORDER = ["s_start", "s_end_plus"]
POOL_LABELS = {"s_start": "S-start\n(round-1 parable)", "s_end_plus": "S-end+\n(round-10 myth)"}
POOL_COLORS = {"s_start": "#1f77b4", "s_end_plus": "#2ca02c"}

PHASE3_JOINT_PATHS = {
    "s_start": "data/json/noise_experiments/phase3_seeded/phase3_seeded_s_start_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json",
    "s_end_plus": "data/json/noise_experiments/phase3_seeded/phase3_seeded_s_end_plus_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json",
}


def load_phase3_joints():
    out = {}
    for pool, pat in PHASE3_JOINT_PATHS.items():
        files = sorted(glob.glob(pat))
        files = [f for f in files if ".results." not in f and ".checkpoint." not in f]
        joints = []
        for f in files:
            with open(f) as fh:
                d = json.load(fh)
            ch = d.get("conversation_history", [])
            bal = ch[-1].get("balances", {}) if ch else {}
            if bal:
                joints.append(sum(bal.values()))
        out[pool] = joints
    return out


def collect(manifest):
    rows = {}
    for pool in POOL_ORDER:
        rows[pool] = {
            "sentence_length": [s["abstractness"]["mean_sentence_length"] for s in manifest["seeds"][pool]],
            "n_sentences": [s["abstractness"]["n_sentences"] for s in manifest["seeds"][pool]],
            "cooperative_pct": [s["cooperativity"]["cooperative_pct"] for s in manifest["seeds"][pool]],
        }
    return rows


def draw_panel(ax, values_by_pool, title, ylabel, fmt="{:.1f}", ceiling=None, ceiling_label=None):
    x = np.arange(len(POOL_ORDER))
    means = [statistics.mean(values_by_pool[p]) for p in POOL_ORDER]
    stds = [statistics.stdev(values_by_pool[p]) if len(values_by_pool[p]) > 1 else 0 for p in POOL_ORDER]

    bars = ax.bar(
        x,
        means,
        width=0.55,
        yerr=stds,
        capsize=5,
        color=[POOL_COLORS[p] for p in POOL_ORDER],
        edgecolor="black",
        linewidth=0.7,
        alpha=0.85,
    )

    rng = np.random.default_rng(seed=2)
    for i, p in enumerate(POOL_ORDER):
        vals = values_by_pool[p]
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(
            [x[i] + dx for dx in jitter],
            vals,
            s=42,
            color="black",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

    # value labels above each bar
    ymax = max([m + s for m, s in zip(means, stds)] + ([ceiling] if ceiling else [0])) * 1.18
    ax.set_ylim(0, ymax)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(
            x[i],
            m + s + ymax * 0.03,
            f"{fmt.format(m)}\n(±{fmt.format(s)})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    if ceiling is not None:
        ax.axhline(ceiling, color="red", linestyle="--", alpha=0.55, label=ceiling_label)
        ax.legend(loc="lower right", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([POOL_LABELS[p] for p in POOL_ORDER], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)


def main():
    configure_matplotlib()
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    metrics = collect(manifest)
    joints = load_phase3_joints()

    # 2×2 grid, more breathing room than 1×4.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left — game outcome (the result)
    draw_panel(
        axes[0, 0],
        joints,
        "GAME OUTCOME (the result)",
        "Joint balance after 10 rounds ($)",
        fmt="${:.0f}",
        ceiling=600,
        ceiling_label="ceiling ($600)",
    )

    # Top-right — cooperative word %
    draw_panel(
        axes[0, 1],
        {p: metrics[p]["cooperative_pct"] for p in POOL_ORDER},
        "COOPERATION VOCABULARY",
        "% of words from cooperation lists",
        fmt="{:.2f}%",
    )

    # Bottom-left — sentence length
    draw_panel(
        axes[1, 0],
        {p: metrics[p]["sentence_length"] for p in POOL_ORDER},
        "SENTENCE LENGTH (flowing vs chunked)",
        "Words per sentence",
        fmt="{:.1f}",
    )

    # Bottom-right — n_sentences
    draw_panel(
        axes[1, 1],
        {p: metrics[p]["n_sentences"] for p in POOL_ORDER},
        "SENTENCE COUNT (chunkiness)",
        "Sentences per myth",
        fmt="{:.1f}",
    )

    fig.suptitle(
        "Phase 3 puzzle — round-10 myths produce MORE cooperation, but have FEWER cooperation words and MORE shorter sentences\n"
        "Bars = pool mean (n=5).  Whiskers = ±sd.  Dots = individual reps.",
        fontsize=12,
        y=1.00,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
