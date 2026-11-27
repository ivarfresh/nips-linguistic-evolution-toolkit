#!/usr/bin/env python3
"""Figure for the self-inspiration sentence ablation (2026-11-27).

Left: word adoption from the partner's myth in excess of an unseen myth, by reader model and
pairing. Right: final resources per agent, by pairing (dot-and-interval version of the figure;
the paper-format version is analyses/plot_no_self_line_dyads.py). With versus without the sentence;
dots are runs (both task orders, 10 per column), bars are means with 95% t intervals.
Reads per_run.csv and mixed_per_run.csv written by analyses/no_self_line_dyads.py and
analyses/no_self_line_mixed_dyads.py.

  python3 analyses/plot_no_self_line_dyads_dots.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses._shared import configure_matplotlib  # noqa: E402

configure_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "docs/research/no_self_line_dyads_20261127"
ARMS = [("line", "With the sentence", "#3b5b8c"), ("no_line", "Without the sentence", "#d9822b")]
NAME = {"sonnet": "Sonnet-4.5", "gpt": "GPT-5-Nano", "gemini": "Gemini-3.7"}


def column(ax, x, values, color, rng):
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    ax.scatter(x + rng.uniform(-0.07, 0.07, len(values)), values, s=16, color=color, alpha=0.45, linewidths=0, zorder=2)
    mean = values.mean()
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    ax.errorbar(x, mean, yerr=half, fmt="o", color=color, markersize=6.5, capsize=3.5, elinewidth=1.6,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)


def main() -> None:
    single = pd.read_csv(OUT / "per_run.csv")
    mixed = pd.read_csv(OUT / "mixed_per_run.csv")
    rng = np.random.default_rng(0)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.6, 3.9), gridspec_kw={"width_ratios": [7, 4.3]})

    left = [("Sonnet-4.5\nreading\nSonnet-4.5", lambda arm: single[single["arm"] == arm]["borrowing"])]
    for other in ("gpt", "gemini"):
        m = mixed[mixed["pair"] == f"sonnet+{other}"]
        left.append((f"Sonnet-4.5\nreading\n{NAME[other]}", lambda arm, m=m: m[(m["arm"] == arm) & (m["family"] == "sonnet")]["borrowing"]))
        left.append((f"{NAME[other]}\nreading\nSonnet-4.5", lambda arm, m=m, o=other: m[(m["arm"] == arm) & (m["family"] == o)]["borrowing"]))
    m = mixed[mixed["pair"] == "gemini+gpt"]
    for reader, other in (("gemini", "gpt"), ("gpt", "gemini")):
        left.append((f"{NAME[reader]}\nreading\n{NAME[other]}", lambda arm, m=m, r=reader: m[(m["arm"] == arm) & (m["family"] == r)]["borrowing"]))
    for i, (_, get) in enumerate(left):
        for k, (arm, _, color) in enumerate(ARMS):
            column(a, i + (k - 0.5) * 0.34, 100 * get(arm), color, rng)
    a.axhline(0, color="#888888", linewidth=0.8, zorder=1)
    a.set_xticks(range(len(left)), [name for name, _ in left])
    a.set_ylabel("Word adoption from the partner's myth,\nexcess over an unseen myth (pct. points)")

    right = [("Sonnet-4.5 +\nSonnet-4.5", lambda arm: single[single["arm"] == arm]["resources"])]
    for other in ("gpt", "gemini"):
        m = mixed[(mixed["pair"] == f"sonnet+{other}") & (mixed["family"] == "sonnet")]  # one row per run
        right.append((f"Sonnet-4.5 +\n{NAME[other]}", lambda arm, m=m: m[m["arm"] == arm]["resources"]))
    m = mixed[(mixed["pair"] == "gemini+gpt") & (mixed["family"] == "gemini")]  # one row per run
    right.append(("Gemini-3.7 +\nGPT-5-Nano", lambda arm, m=m: m[m["arm"] == arm]["resources"]))
    for i, (_, get) in enumerate(right):
        for k, (arm, _, color) in enumerate(ARMS):
            column(b, i + (k - 0.5) * 0.34, get(arm), color, rng)
    b.set_xticks(range(len(right)), [name for name, _ in right])
    b.set_ylim(25, 77)
    b.set_ylabel("Final resources per agent")

    handles = [plt.Line2D([], [], marker="o", linestyle="", color=color, markersize=6.5, label=label) for _, label, color in ARMS]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    for ax, tag in ((a, "a"), (b, "b")):
        ax.text(-0.17 if ax is a else -0.24, 1.03, tag, transform=ax.transAxes, fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"no_self_line_transmission_cooperation_dots.{ext}", dpi=300, bbox_inches="tight")
    print(OUT / "no_self_line_transmission_cooperation_dots.png")


if __name__ == "__main__":
    main()
