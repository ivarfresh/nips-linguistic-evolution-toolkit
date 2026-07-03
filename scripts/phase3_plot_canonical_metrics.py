"""Phase 3 seed myth: canonical cooperativeness × abstractness scatter.

Uses the canonical repo analyses (instead of the Phase 2 LLM-judge):
  - cooperativeness: `analyses/cooperativity_analysis.py` (cooperative_pct)
  - abstractness:    `analyses/myth_compression_curves.py` (mean_sentence_length, n_sentences)

Per-myth scatter colored by pool, with × marking the pool mean. Two panels:
  (a) cooperative_pct vs mean_sentence_length
  (b) cooperative_pct vs n_sentences

Output: data/phase3/plots/03_canonical_metrics.png
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
OUT_PATH = Path("data/phase3/plots/03_canonical_metrics.png")

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
    rows = {}
    for pool in POOL_ORDER:
        rows[pool] = []
        for seed in manifest["seeds"][pool]:
            rows[pool].append(
                {
                    "sentence_length": seed["abstractness"]["mean_sentence_length"],
                    "n_sentences": seed["abstractness"]["n_sentences"],
                    "ttr": seed["abstractness"]["ttr"],
                    "n_tokens": seed["abstractness"]["n_tokens"],
                    "cooperative_pct": seed["cooperativity"]["cooperative_pct"],
                    "uncooperative_pct": seed["cooperativity"]["uncooperative_pct"],
                    "mutuality_ratio": seed["cooperativity"]["mutuality_ratio"],
                    "joint_at_source": seed.get("joint_at_source"),
                    "agent_id": seed.get("agent_id"),
                }
            )
    return rows


def plot_panel(ax, rows_by_pool, x_key, x_label):
    for pool in POOL_ORDER:
        rows = rows_by_pool[pool]
        xs = [r[x_key] for r in rows]
        ys = [r["cooperative_pct"] for r in rows]
        ax.scatter(
            xs,
            ys,
            s=90,
            color=POOL_COLORS[pool],
            edgecolor="black",
            linewidth=0.8,
            label=POOL_LABELS[pool],
            zorder=3,
        )
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
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cooperative word % (dictionary count / total words)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)


def main():
    configure_matplotlib()
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    rows = collect(manifest)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    plot_panel(axes[0], rows, "sentence_length", "Mean sentence length (words / sentence)")
    plot_panel(axes[1], rows, "n_sentences", "Number of sentences")

    axes[0].set_title("vs sentence length")
    axes[1].set_title("vs sentence count")

    fig.suptitle(
        "Phase 3 seed myths — canonical analyses\n"
        "Cooperativeness from analyses/cooperativity_analysis.py (dictionary word counts)  •  "
        "Abstractness from analyses/myth_compression_curves.py\n"
        "Dots = individual myths (n=5 per pool). × = pool mean.",
        fontsize=10.5,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
