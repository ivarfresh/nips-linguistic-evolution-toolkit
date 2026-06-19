"""Sibling of plot 05 — H5 dose-response scatter but using the Opus 4.7
judge scores instead of Sonnet 4.5.

Layout matches scripts/phase2_plot_results.py:plot_judge_vs_joint
exactly; the only difference is which judge column is read from the
seed manifest.

Output: data/phase2/plots/05_judge_vs_joint_opus.png  (the original
plot 05 is untouched.)
"""

import glob
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEED_COLORS = {
    "s_none": "#7f7f7f",
    "s_filler": "#9467bd",
    "s_end_minus": "#d62728",
    "s_end_plus": "#2ca02c",
    "s_start": "#1f77b4",
}
SEED_LABELS = {
    "s_none": "S-none\n(baseline)",
    "s_filler": "S-filler\n(Wikipedia)",
    "s_end_minus": "S-end−\n(low-coop end myth)",
    "s_end_plus": "S-end+\n(high-coop end myth)",
    "s_start": "S-start\n(round-1 directive)",
}
TASK_ORDERS = ["game", "game_myth", "myth_game"]
TASK_LABELS = {
    "game": '["game"]',
    "game_myth": '["game","myth"]',
    "myth_game": '["myth","game"]',
}
SEEDED_ROOT = "data/json/noise_experiments/phase2_seeded"


def _opus_score(seed):
    js = seed.get("judge_scores", {}) or {}
    return js.get("opus_4_7") or js.get("claude-opus-4-7")


def main():
    manifest = json.load(open("data/phase2/seed_manifest.json"))
    np.random.seed(20260619)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax_i, to in enumerate(TASK_ORDERS):
        ax = axes[ax_i]
        xs, ys, colors = [], [], []
        for stype in ["s_filler", "s_end_minus", "s_end_plus", "s_start"]:
            base = (
                f"{SEEDED_ROOT}/phase2_seeded_{stype}_phase2_8agent_history3_anon_neg5/"
                f"claude-sonnet-4.5/{to}/default"
            )
            files = sorted(glob.glob(f"{base}/*.json"))
            files = [
                f for f in files if ".results." not in f and ".checkpoint." not in f
            ]
            for f in files:
                d = json.load(open(f))
                rep = d.get("run_metadata", {}).get("phase2_rep")
                if rep is None:
                    m = re.search(r"_rep(\d+)_", f)
                    if m:
                        rep = int(m.group(1))
                if rep is None:
                    continue
                seed = manifest["seeds"][stype][rep % len(manifest["seeds"][stype])]
                js = _opus_score(seed)
                if js is None:
                    continue
                joint = sum(d.get("game_data", {}).get("balances", {}).values())
                xs.append(js + np.random.normal(0, 0.1))
                ys.append(joint)
                colors.append(SEED_COLORS[stype])
        ax.scatter(
            xs,
            ys,
            c=colors,
            s=60,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        # Linear fit
        if len(xs) > 1:
            coeffs = np.polyfit(xs, ys, 1)
            xrange = np.array([0, 10])
            ax.plot(xrange, np.polyval(coeffs, xrange), "k--", linewidth=1.5, alpha=0.6)
        # Spearman ρ
        n = len(xs)
        rx = sorted(range(n), key=lambda i: xs[i])
        ry = sorted(range(n), key=lambda i: ys[i])
        rank_x = [0] * n
        rank_y = [0] * n
        for r, i in enumerate(rx):
            rank_x[i] = r
        for r, i in enumerate(ry):
            rank_y[i] = r
        mx = statistics.mean(rank_x)
        my = statistics.mean(rank_y)
        cov = sum((a - mx) * (b - my) for a, b in zip(rank_x, rank_y)) / n
        sx = statistics.pstdev(rank_x)
        sy = statistics.pstdev(rank_y)
        rho = cov / (sx * sy) if sx and sy else 0
        ax.set_title(f"{TASK_LABELS[to]}  ρ={rho:+.3f}")
        ax.set_xlabel("Judge cooperativeness score (0–10) — Opus 4.7")
        if ax_i == 0:
            ax.set_ylabel("Joint balance")
        ax.grid(alpha=0.3)
        ax.set_xticks(range(0, 11, 2))

    # Legend
    handles = [
        plt.scatter(
            [],
            [],
            c=SEED_COLORS[s],
            s=60,
            edgecolor="black",
            linewidth=0.5,
            label=SEED_LABELS[s].replace("\n", " "),
        )
        for s in ["s_filler", "s_end_minus", "s_end_plus", "s_start"]
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=4,
        fontsize=9,
    )
    fig.suptitle(
        "Phase 2 H5 (Opus 4.7 judge): myth cooperativeness × resulting joint balance",
        y=1.02,
    )
    fig.tight_layout()
    out = Path("data/phase2/plots/05_judge_vs_joint_opus.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
