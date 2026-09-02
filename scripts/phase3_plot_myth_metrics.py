"""Phase 3 seed myth: cooperativeness × abstractness scatter.

Per-myth scatter of mean sentence length (x) vs LLM judge cooperativeness (y),
colored by pool. Two panels side-by-side: Sonnet 4.5 judge and Opus 4.7 judge.

Output: data/phase3/plots/02_seed_metrics.png
"""

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
OUT_PATH = Path("data/phase3/plots/02_seed_metrics.png")

POOL_ORDER = ["s_start", "s_end_plus"]
POOL_LABELS = {
    "s_start": "S-start (round-1 parable)",
    "s_end_plus": "S-end+ (round-10 myth)",
}
POOL_COLORS = {
    "s_start": "#1f77b4",
    "s_end_plus": "#2ca02c",
}


def collect(manifest):
    out = {}
    for pool in POOL_ORDER:
        rows = []
        for seed in manifest["seeds"][pool]:
            rows.append(
                {
                    "sentence_length": seed["metrics"]["mean_sentence_length"],
                    "ttr": seed["metrics"]["ttr"],
                    "word_count": seed["metrics"]["word_count"],
                    "sonnet": seed["judge_scores"]["sonnet_45"],
                    "opus": seed["judge_scores"]["opus_4_7"],
                    "joint_at_source": seed.get("joint_at_source"),
                    "agent_id": seed.get("agent_id"),
                }
            )
        out[pool] = rows
    return out


def plot_panel(ax, rows_by_pool, judge_key, judge_label):
    rng = np.random.default_rng(seed=1)
    for pool in POOL_ORDER:
        rows = rows_by_pool[pool]
        xs = [r["sentence_length"] for r in rows]
        ys = [r[judge_key] for r in rows]
        # Tiny y-jitter so overlapping integer scores at 10 don't fully hide.
        ys_j = [y + rng.uniform(-0.06, 0.06) for y in ys]
        ax.scatter(
            xs,
            ys_j,
            s=90,
            color=POOL_COLORS[pool],
            edgecolor="black",
            linewidth=0.8,
            label=POOL_LABELS[pool],
            zorder=3,
        )
        # Mean cross
        mx = statistics.mean(xs)
        my = statistics.mean(ys)
        ax.plot(
            [mx],
            [my],
            marker="x",
            color=POOL_COLORS[pool],
            markersize=14,
            markeredgewidth=3,
            zorder=4,
        )
    ax.set_xlabel("Mean sentence length (words / sentence)")
    ax.set_ylabel(f"{judge_label} cooperativeness score (0–10)")
    ax.set_title(judge_label)
    ax.set_ylim(6.5, 10.5)
    ax.set_xlim(7.5, 16.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)


def main():
    configure_matplotlib()
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    rows = collect(manifest)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    plot_panel(axes[0], rows, "sonnet", "Sonnet 4.5 judge")
    plot_panel(axes[1], rows, "opus", "Opus 4.7 judge")

    fig.suptitle(
        "Phase 3 seed myths: cooperativeness × sentence length\n"
        "Dots = individual myths (n=5 per pool). × = pool mean. Tiny y-jitter for visibility only.",
        fontsize=11,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
