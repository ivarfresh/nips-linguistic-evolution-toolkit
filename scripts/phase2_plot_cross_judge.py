"""Cross-judge comparison plot — Sonnet 4.5 vs Opus 4.7 on Phase 2 seed myths.

Writes data/phase2/plots/06_cross_judge_comparison.png — does NOT
overwrite the existing 01-05 plots.

Three panels:
  (a) per-myth scatter Sonnet × Opus with y=x diagonal
  (b) per-pool means side-by-side
  (c) H5 dose-response Spearman ρ side-by-side per task order
"""

import glob
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEED_TYPES = ["s_filler", "s_end_minus", "s_end_plus", "s_start"]
SEED_LABELS = {
    "s_start": "S-start",
    "s_end_plus": "S-end+",
    "s_end_minus": "S-end−",
    "s_filler": "S-filler",
}
SEED_COLORS = {
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


def _scores(seed):
    sonnet = seed.get("judge_score")
    js = seed.get("judge_scores", {}) or {}
    if sonnet is None:
        sonnet = js.get("claude-sonnet-4_5") or js.get("sonnet_45")
    opus = js.get("opus_4_7") or js.get("claude-opus-4-7")
    return sonnet, opus


def _spearman(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
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
    return cov / (sx * sy) if sx and sy else 0.0


def main():
    manifest = json.load(open("data/phase2/seed_manifest.json"))

    np.random.seed(20260619)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # --- Panel (a): per-myth scatter Sonnet × Opus ---
    ax = axes[0]
    all_son, all_opus = [], []
    for st in SEED_TYPES:
        xs, ys = [], []
        for seed in manifest["seeds"].get(st, []):
            sn, op = _scores(seed)
            if sn is None or op is None:
                continue
            # jitter to avoid overlap on integer values
            xs.append(sn + np.random.normal(0, 0.08))
            ys.append(op + np.random.normal(0, 0.08))
            all_son.append(sn)
            all_opus.append(op)
        ax.scatter(
            xs,
            ys,
            color=SEED_COLORS[st],
            s=110,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
            label=SEED_LABELS[st],
        )
    ax.plot([-0.5, 10.5], [-0.5, 10.5], "k--", linewidth=1, alpha=0.5, label="y = x")
    rho = _spearman(all_son, all_opus)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_xticks(range(0, 11, 2))
    ax.set_yticks(range(0, 11, 2))
    ax.set_xlabel("Sonnet 4.5 judge score")
    ax.set_ylabel("Opus 4.7 judge score")
    ax.set_title(f"(a) Per-myth agreement  ρ = {rho:+.3f}, n = {len(all_son)}")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    # --- Panel (b): per-pool means side-by-side ---
    ax = axes[1]
    x = np.arange(len(SEED_TYPES))
    width = 0.38
    sonnet_means, sonnet_stds = [], []
    opus_means, opus_stds = [], []
    for st in SEED_TYPES:
        sons, opuses = [], []
        for seed in manifest["seeds"].get(st, []):
            sn, op = _scores(seed)
            if sn is not None:
                sons.append(sn)
            if op is not None:
                opuses.append(op)
        sonnet_means.append(statistics.mean(sons) if sons else 0)
        sonnet_stds.append(statistics.pstdev(sons) if sons else 0)
        opus_means.append(statistics.mean(opuses) if opuses else 0)
        opus_stds.append(statistics.pstdev(opuses) if opuses else 0)
    bars1 = ax.bar(
        x - width / 2,
        sonnet_means,
        width,
        yerr=sonnet_stds,
        capsize=4,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
        label="Sonnet 4.5",
    )
    bars2 = ax.bar(
        x + width / 2,
        opus_means,
        width,
        yerr=opus_stds,
        capsize=4,
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.5,
        label="Opus 4.7",
    )
    for i, (sm, om) in enumerate(zip(sonnet_means, opus_means)):
        ax.annotate(f"{sm:.1f}", (i - width / 2, sm + 0.15), ha="center", fontsize=8)
        ax.annotate(f"{om:.1f}", (i + width / 2, om + 0.15), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([SEED_LABELS[s] for s in SEED_TYPES])
    ax.set_ylabel("Judge cooperativeness score (0–10)")
    ax.set_title("(b) Per-pool mean cooperativeness")
    ax.set_ylim(0, 11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # --- Panel (c): H5 ρ(judge, joint) per task order, by judge ---
    seed_score_map = {}
    for st in SEED_TYPES:
        for i, s in enumerate(manifest["seeds"].get(st, [])):
            sn, op = _scores(s)
            seed_score_map[(st, i)] = (sn, op)

    rho_sonnet, rho_opus = [], []
    for to in TASK_ORDERS:
        xs_sonnet, xs_opus, ys = [], [], []
        for st in SEED_TYPES:
            n_seeds = len(manifest["seeds"].get(st, []))
            base = (
                f"data/json/noise_experiments/phase2_seeded/"
                f"phase2_seeded_{st}_phase2_8agent_history3_anon_neg5/"
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
                    import re

                    m = re.search(r"_rep(\d+)_", f)
                    if m:
                        rep = int(m.group(1))
                if rep is None:
                    continue
                idx = rep % n_seeds if n_seeds else 0
                sn, op = seed_score_map.get((st, idx), (None, None))
                joint = sum(d["game_data"]["balances"].values())
                if sn is None or op is None:
                    continue
                xs_sonnet.append(sn)
                xs_opus.append(op)
                ys.append(joint)
        rho_sonnet.append(_spearman(xs_sonnet, ys))
        rho_opus.append(_spearman(xs_opus, ys))

    ax = axes[2]
    x = np.arange(len(TASK_ORDERS))
    width = 0.38
    ax.bar(
        x - width / 2,
        rho_sonnet,
        width,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
        label="Sonnet 4.5",
    )
    ax.bar(
        x + width / 2,
        rho_opus,
        width,
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.5,
        label="Opus 4.7",
    )
    for i, (rs, ro) in enumerate(zip(rho_sonnet, rho_opus)):
        ax.annotate(
            f"{rs:+.2f}",
            (i - width / 2, rs + (0.02 if rs >= 0 else -0.04)),
            ha="center",
            fontsize=8,
        )
        ax.annotate(
            f"{ro:+.2f}",
            (i + width / 2, ro + (0.02 if ro >= 0 else -0.04)),
            ha="center",
            fontsize=8,
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[to] for to in TASK_ORDERS])
    ax.set_ylabel("Spearman ρ(judge score, joint balance)")
    ax.set_title("(c) H5 dose-response per task order")
    ax.set_ylim(-0.1, 0.7)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Phase 2 cross-judge comparison — Sonnet 4.5 vs Opus 4.7 on the 20 seed myths",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    out = Path("data/phase2/plots/06_cross_judge_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
