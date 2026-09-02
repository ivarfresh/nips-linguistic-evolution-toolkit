"""Phase 6 — Gemini round-10 myths vs Sonnet's parable and strategic styles.

4-panel comparison on the canonical metrics, showing that Gemini's round-10
myths sit structurally between Sonnet's parable (s_start) and Sonnet's chunked-
strategic style (s_end_plus_sonnet): long sentences like a parable, but with
strategic content and higher cooperation vocabulary.

Output: data/phase6/plots/02_gemini_vs_sonnet_metrics.png
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
OUT_PATH = Path("data/phase6/plots/02_gemini_vs_sonnet_metrics.png")

POOL_ORDER = ["s_start", "s_end_plus_gemini", "s_end_plus"]
POOL_LABELS = {
    "s_start": "S-start\n(Sonnet round-1\nparable)",
    "s_end_plus_gemini": "S-end+ Gemini\n(Gemini round-10\nparable+strategy)",
    "s_end_plus": "S-end+ Sonnet\n(Sonnet round-10\nchunked-strategic)",
}
POOL_COLORS = {
    "s_start": "#1f77b4",
    "s_end_plus_gemini": "#17becf",
    "s_end_plus": "#2ca02c",
}

# Outcome means for annotation
OUTCOMES = {"s_start": 499, "s_end_plus_gemini": 549, "s_end_plus": 600}


def draw_panel(ax, manifest, metric_path, title, ylabel, fmt="{:.2f}", invert_intuition=None):
    rng = np.random.default_rng(seed=7)
    means, stds, vals_per_pool = [], [], []
    for pool in POOL_ORDER:
        seeds = manifest["seeds"][pool]
        # Navigate nested keys e.g. "abstractness.mean_sentence_length"
        vals = []
        for s in seeds:
            v = s
            for key in metric_path.split("."):
                v = v[key]
            vals.append(v)
        means.append(statistics.mean(vals))
        stds.append(statistics.stdev(vals) if len(vals) > 1 else 0)
        vals_per_pool.append(vals)

    x = np.arange(len(POOL_ORDER))
    bars = ax.bar(
        x, means, width=0.55, yerr=stds, capsize=5,
        color=[POOL_COLORS[p] for p in POOL_ORDER],
        edgecolor="black", linewidth=0.7, alpha=0.88,
    )
    # individual seed dots
    for i, vals in enumerate(vals_per_pool):
        jitter = rng.uniform(-0.1, 0.1, size=len(vals))
        ax.scatter(
            [x[i] + dx for dx in jitter], vals,
            s=42, color="black", edgecolor="white", linewidth=0.7, zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([POOL_LABELS[p] for p in POOL_ORDER], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Annotate the mean above each bar
    ymax = max([m + s for m, s in zip(means, stds)]) * 1.18
    ax.set_ylim(0, ymax)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(x[i], m + s + ymax * 0.03,
                fmt.format(m), ha="center", va="bottom", fontsize=10)


def main():
    configure_matplotlib()
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    draw_panel(axes[0, 0], manifest, "cooperativity.cooperative_pct",
               "Cooperative word % (dictionary)",
               "% of words from cooperation list", fmt="{:.2f}%")

    draw_panel(axes[0, 1], manifest, "abstractness.mean_sentence_length",
               "Mean sentence length",
               "Words / sentence  →  higher = more flowing", fmt="{:.1f}")

    draw_panel(axes[1, 0], manifest, "abstractness.n_sentences",
               "Number of sentences",
               "Sentences / myth  →  higher = more chunked", fmt="{:.1f}")

    draw_panel(axes[1, 1], manifest, "abstractness.ttr",
               "Type-token ratio (lexical diversity)",
               "TTR", fmt="{:.3f}")

    # Annotate the outcome ($) at the bottom
    fig.text(0.5, 0.02,
             "Phase 6 game-outcome means (Sonnet runs, n=5):  "
             "S-start = \\$499      S-end+ Gemini = \\$549      S-end+ Sonnet = \\$600",
             ha="center", fontsize=11, color="#222")

    fig.suptitle(
        "Phase 6 — Gemini round-10 myths: structurally a PARABLE, but with strategic content + higher cooperation vocab\n"
        "Gemini's round-10 prose is closer to Sonnet's round-1 parable in structure (long sentences, fewer of them), "
        "but it encodes strategic play rules and uses more cooperation vocabulary.\n"
        "The game outcome ($549) sits halfway between Sonnet's parable ($499) and Sonnet's chunked-strategic ($600).",
        fontsize=11,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
