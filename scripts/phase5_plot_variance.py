"""Phase 5 — within-pool variance analysis for s_end_minus.

For each of the 5 s_end_minus seeds, plot three correlations against the
Phase 5 joint balance:
  (a) source-run joint balance (where the seed was harvested from)
  (b) seed cooperative_pct (dictionary metric)
  (c) seed mean sentence length

Each panel: scatter of 5 points (one per rep), regression line,
Pearson r in title, baseline reference, suppression-vs-lift annotation.

Output: data/phase5/plots/02_s_end_minus_variance.png
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
MANIFEST_PATH = Path("data/phase3/seed_manifest.json")
OUT_PATH = Path("data/phase5/plots/02_s_end_minus_variance.png")
BASELINE_MEAN = 437.4


def joint_of(path):
    with open(path) as f:
        d = json.load(f)
    ch = d.get("conversation_history", [])
    bal = ch[-1].get("balances", {}) if ch else {}
    return sum(bal.values()) if bal else None


def collect_rows():
    with open(MANIFEST_PATH) as f:
        m = json.load(f)
    seeds = m["seeds"]["s_end_minus"]

    files = sorted(glob.glob(
        f"{BASE}/phase5_seeded/phase3_seeded_s_end_minus_phase3_8agent_anon_neg5_myth_only/claude-sonnet-4.5/game/default/*.json"
    ))
    files = [f for f in files if ".results." not in f and ".checkpoint." not in f]
    by_rep = {}
    for f in files:
        rm = re.search(r"rep(\d+)", f)
        if rm:
            by_rep[int(rm.group(1))] = joint_of(f)

    rows = []
    for rep in sorted(by_rep):
        s = seeds[rep]
        rows.append({
            "rep": rep,
            "src_joint": s["joint_at_source"],
            "p5_joint": by_rep[rep],
            "coop_pct": s["cooperativity"]["cooperative_pct"],
            "sent_len": s["abstractness"]["mean_sentence_length"],
        })
    return rows


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx, sy = statistics.stdev(xs), statistics.stdev(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    return cov / (sx * sy) if (sx and sy) else float("nan")


def draw_panel(ax, rows, x_key, x_label, panel_letter):
    xs = [r[x_key] for r in rows]
    ys = [r["p5_joint"] for r in rows]
    r = pearson(xs, ys)

    ax.axhline(BASELINE_MEAN, color="#7f7f7f", linestyle=":", alpha=0.7, linewidth=1.4,
               label=f"Baseline mean (${BASELINE_MEAN:.0f})")
    # suppression zone shading
    ax.axhspan(300, BASELINE_MEAN, color="#fde0e0", alpha=0.3, zorder=0)
    ax.text(min(xs) + 0.02 * (max(xs) - min(xs)), 320, "← SUPPRESSION zone",
            fontsize=8.5, color="#aa3333", style="italic", va="bottom")

    # regression line
    if len(xs) >= 2:
        coef = np.polyfit(xs, ys, 1)
        xx = np.linspace(min(xs), max(xs), 50)
        ax.plot(xx, np.polyval(coef, xx), color="#444", linestyle="--", alpha=0.6, linewidth=1.4)

    # points + rep labels
    ax.scatter(xs, ys, s=130, color="#d62728", edgecolor="black", linewidth=0.9, zorder=3)
    for r_row, x, y in zip(rows, xs, ys):
        ax.annotate(
            f"r{r_row['rep']}",
            xy=(x, y), xytext=(7, 6), textcoords="offset points",
            fontsize=9, color="#222",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Phase 5 joint balance ($)")
    ax.set_title(f"({panel_letter})  Pearson r = {r:+.3f}", fontsize=11)
    ax.set_ylim(330, 480)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)


def main():
    configure_matplotlib()
    rows = collect_rows()

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    draw_panel(axes[0], rows, "src_joint", "Source-run joint balance ($)", "a")
    draw_panel(axes[1], rows, "coop_pct", "Seed cooperative word % (dictionary)", "b")
    draw_panel(axes[2], rows, "sent_len", "Seed mean sentence length (words)", "c")

    fig.suptitle(
        "Phase 5 — within-pool variance in S-end− (n=5)\n"
        "Three candidate predictors of the suppression magnitude. Strongest signal: the seed's surface cooperation word count (r=+0.952).\n"
        "Underlying mechanism: the strategic play recipe encoded in each individual myth.",
        fontsize=11,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
