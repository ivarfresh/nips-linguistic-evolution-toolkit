"""Phase 2 result plots.

Produces:
  data/phase2/plots/01_cell_means.png        — joint balance by seed × task order
  data/phase2/plots/02_per_rep_strip.png     — strip plot of per-run joints
  data/phase2/plots/03_trajectories.png      — per-round joint trajectory by seed
  data/phase2/plots/04_contrast_bars.png     — Δ vs s_none with SE bars per contrast
  data/phase2/plots/05_judge_vs_joint.png    — H5 dose-response (judge score × joint)
"""

import glob
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEED_ORDER = ["s_none", "s_filler", "s_end_minus", "s_end_plus", "s_start"]
SEED_LABELS = {
    "s_none": "S-none\n(baseline)",
    "s_filler": "S-filler\n(Wikipedia)",
    "s_end_minus": "S-end−\n(low-coop end myth)",
    "s_end_plus": "S-end+\n(high-coop end myth)",
    "s_start": "S-start\n(round-1 directive)",
}
SEED_COLORS = {
    "s_none": "#7f7f7f",
    "s_filler": "#9467bd",
    "s_end_minus": "#d62728",
    "s_end_plus": "#2ca02c",
    "s_start": "#1f77b4",
}
TASK_ORDERS = ["game", "game_myth", "myth_game"]
TASK_LABELS = {
    "game": '["game"]',
    "game_myth": '["game","myth"]',
    "myth_game": '["myth","game"]',
}

BASELINE_ROOT = "data/json/noise_experiments/phase2_baseline/phase2_baseline_neg5/claude-sonnet-4.5"
SEEDED_ROOT = "data/json/noise_experiments/phase2_seeded"


def _runs_for(seed_type, task_order):
    if seed_type == "s_none":
        pattern = f"{BASELINE_ROOT}/{task_order}/phase2_8agent_history3_anon_neg5/*.json"
    else:
        pattern = (
            f"{SEEDED_ROOT}/phase2_seeded_{seed_type}_phase2_8agent_history3_anon_neg5/"
            f"claude-sonnet-4.5/{task_order}/default/*.json"
        )
    files = sorted(glob.glob(pattern))
    return [f for f in files if ".results." not in f and ".checkpoint." not in f]


def _joints(seed_type, task_order):
    files = _runs_for(seed_type, task_order)
    out = []
    for f in files:
        d = json.load(open(f))
        balances = d.get("game_data", {}).get("balances", {})
        if balances:
            out.append(sum(balances.values()))
    return out


def _round_joints(seed_type, task_order):
    """Cumulative joint balance per round (averaged across runs)."""
    files = _runs_for(seed_type, task_order)
    per_round = defaultdict(list)
    for f in files:
        d = json.load(open(f))
        ch = d.get("conversation_history", [])
        # Reconstruct cumulative balance per round
        # Each entry has "balances" snapshot after that round
        for entry in ch:
            r = entry.get("round")
            b = entry.get("balances")
            if r is not None and b:
                per_round[r].append(sum(b.values()))
    return {r: per_round[r] for r in sorted(per_round.keys())}


def plot_cell_means(out_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(TASK_ORDERS))
    width = 0.16
    for i, seed in enumerate(SEED_ORDER):
        means, stds = [], []
        for to in TASK_ORDERS:
            joints = _joints(seed, to)
            means.append(statistics.mean(joints) if joints else 0)
            stds.append(statistics.stdev(joints) if len(joints) > 1 else 0)
        offset = (i - 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=SEED_LABELS[seed].replace("\n", " "),
            color=SEED_COLORS[seed],
            edgecolor="black",
            linewidth=0.5,
        )
    ax.axhline(600, color="red", linestyle="--", alpha=0.5, label="Cooperation ceiling")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[to] for to in TASK_ORDERS])
    ax.set_ylabel("Joint balance (sum of 8 agents)")
    ax.set_title(
        "Phase 2: joint balance by seed × task order\n"
        "Sonnet 4.5, 8-agent history3 anon directive, noisy_negative_5, n=15 baseline / n=5 seeded"
    )
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.set_ylim(0, 660)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_per_rep_strip(out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax_i, to in enumerate(TASK_ORDERS):
        ax = axes[ax_i]
        for i, seed in enumerate(SEED_ORDER):
            joints = _joints(seed, to)
            if not joints:
                continue
            jitter = np.random.normal(0, 0.05, size=len(joints))
            ax.scatter(
                [i + j for j in jitter],
                joints,
                color=SEED_COLORS[seed],
                s=60,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.hlines(
                statistics.mean(joints),
                i - 0.25,
                i + 0.25,
                colors="black",
                linewidth=2,
            )
        ax.set_xticks(range(len(SEED_ORDER)))
        ax.set_xticklabels([SEED_LABELS[s].replace("\n", "\n") for s in SEED_ORDER], rotation=0, fontsize=8)
        ax.set_title(TASK_LABELS[to])
        ax.axhline(600, color="red", linestyle="--", alpha=0.4)
        ax.grid(axis="y", alpha=0.3)
        if ax_i == 0:
            ax.set_ylabel("Joint balance")
    fig.suptitle("Phase 2: per-run joint balance (black bar = mean)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_trajectories(out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax_i, to in enumerate(TASK_ORDERS):
        ax = axes[ax_i]
        for seed in SEED_ORDER:
            per_round = _round_joints(seed, to)
            if not per_round:
                continue
            rounds = sorted(per_round.keys())
            means = [statistics.mean(per_round[r]) for r in rounds]
            ax.plot(
                rounds,
                means,
                marker="o",
                color=SEED_COLORS[seed],
                label=SEED_LABELS[seed].replace("\n", " "),
                linewidth=2,
            )
        ax.set_title(TASK_LABELS[to])
        ax.set_xlabel("Round")
        if ax_i == 0:
            ax.set_ylabel("Cumulative joint balance (mean)")
        ax.grid(alpha=0.3)
        ax.set_xticks(range(1, 11))
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Phase 2: cumulative joint-balance trajectory by seed type", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_contrast_bars(out_path):
    """Δ vs s_none per task order with SE bars."""
    contrasts = ["s_start", "s_end_plus", "s_end_minus", "s_filler"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax_i, to in enumerate(TASK_ORDERS):
        ax = axes[ax_i]
        none_j = _joints("s_none", to)
        none_mean = statistics.mean(none_j)
        none_var = statistics.variance(none_j) if len(none_j) > 1 else 0
        positions = np.arange(len(contrasts))
        deltas, ses, zs = [], [], []
        for seed in contrasts:
            sj = _joints(seed, to)
            if not sj:
                deltas.append(0); ses.append(0); zs.append(0); continue
            d = statistics.mean(sj) - none_mean
            sv = statistics.variance(sj) if len(sj) > 1 else 0
            se = math.sqrt(sv / len(sj) + none_var / len(none_j))
            deltas.append(d)
            ses.append(se)
            zs.append(d / se if se else 0)
        bars = ax.bar(
            positions,
            deltas,
            yerr=ses,
            capsize=4,
            color=[SEED_COLORS[s] for s in contrasts],
            edgecolor="black",
            linewidth=0.7,
        )
        # Annotate z values
        for j, (d, z) in enumerate(zip(deltas, zs)):
            top = d + (ses[j] if d >= 0 else -ses[j])
            offset = 6 if d >= 0 else -18
            ax.annotate(
                f"z={z:+.2f}",
                xy=(positions[j], top),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold" if abs(z) >= 1.96 else "normal",
                color="black" if abs(z) >= 1.96 else "gray",
            )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(positions)
        ax.set_xticklabels([SEED_LABELS[s].replace("\n", "\n") for s in contrasts], fontsize=8)
        ax.set_title(f"{TASK_LABELS[to]}  (S-none={none_mean:.0f})")
        ax.grid(axis="y", alpha=0.3)
        if ax_i == 0:
            ax.set_ylabel("Δ joint vs S-none")
    fig.suptitle(
        "Phase 2: seed-cell effect on cooperation (Δ joint vs S-none, with SE bars)\n"
        "Bold z = |z| ≥ 1.96 (5% two-tailed)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_judge_vs_joint(out_path):
    manifest = json.load(open("data/phase2/seed_manifest.json"))
    seeds_by_idx = {}
    for stype, seeds in manifest["seeds"].items():
        for i, s in enumerate(seeds):
            seeds_by_idx[(stype, i)] = s

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
            files = [f for f in files if ".results." not in f and ".checkpoint." not in f]
            for f in files:
                d = json.load(open(f))
                rep = d.get("run_metadata", {}).get("phase2_rep")
                if rep is None:
                    import re
                    m = re.search(r"_rep(\d+)_", f)
                    if m:
                        rep = int(m.group(1))
                if rep is None:
                    continue
                seed = manifest["seeds"][stype][rep % len(manifest["seeds"][stype])]
                js = seed.get("judge_score", 0)
                joint = sum(d.get("game_data", {}).get("balances", {}).values())
                xs.append(js + np.random.normal(0, 0.1))
                ys.append(joint)
                colors.append(SEED_COLORS[stype])
        ax.scatter(xs, ys, c=colors, s=60, alpha=0.7, edgecolor="black", linewidth=0.5)
        # Per-seed-type means
        from collections import defaultdict
        bins = defaultdict(list)
        for x, y, c in zip(xs, ys, colors):
            bins[round(x)].append(y)
        # Linear fit
        if len(xs) > 1:
            coeffs = np.polyfit(xs, ys, 1)
            xrange = np.array([0, 10])
            ax.plot(xrange, np.polyval(coeffs, xrange), "k--", linewidth=1.5, alpha=0.6)
        # Spearman ρ
        n = len(xs)
        rx = sorted(range(n), key=lambda i: xs[i])
        ry = sorted(range(n), key=lambda i: ys[i])
        rank_x = [0]*n; rank_y = [0]*n
        for r, i in enumerate(rx): rank_x[i] = r
        for r, i in enumerate(ry): rank_y[i] = r
        mx = statistics.mean(rank_x); my = statistics.mean(rank_y)
        cov = sum((a-mx)*(b-my) for a,b in zip(rank_x, rank_y))/n
        sx = statistics.pstdev(rank_x); sy = statistics.pstdev(rank_y)
        rho = cov/(sx*sy) if sx and sy else 0
        ax.set_title(f"{TASK_LABELS[to]}  ρ={rho:+.3f}")
        ax.set_xlabel("Judge cooperativeness score (0–10)")
        if ax_i == 0:
            ax.set_ylabel("Joint balance")
        ax.grid(alpha=0.3)
        ax.set_xticks(range(0, 11, 2))
    # Legend
    handles = [
        plt.scatter([], [], c=SEED_COLORS[s], s=60, edgecolor="black", linewidth=0.5, label=SEED_LABELS[s].replace("\n", " "))
        for s in ["s_filler", "s_end_minus", "s_end_plus", "s_start"]
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=9)
    fig.suptitle("Phase 2 H5: judge-rated myth cooperativeness × resulting joint balance", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    out_dir = Path("data/phase2/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(20260619)
    plot_cell_means(out_dir / "01_cell_means.png")
    plot_per_rep_strip(out_dir / "02_per_rep_strip.png")
    plot_trajectories(out_dir / "03_trajectories.png")
    plot_contrast_bars(out_dir / "04_contrast_bars.png")
    plot_judge_vs_joint(out_dir / "05_judge_vs_joint.png")


if __name__ == "__main__":
    main()
