#!/usr/bin/env python3
"""Compare the fixed 8-agent memory-primary runs against the June 8 baseline.

Baseline: sonnet45_8agent_myth_directive_history3_anon_r10_n5 (double memory).
Fixed:    sonnet45_8agent_myth_directive_history3_anon_memprimary_r10_n5.

Reports per condition (mean ±std across runs):
- game: sent and return ratio per round (averaged over the 4 dyads per round)
- myths: mean pairwise embedding similarity across all 8 myths per round
  (population convergence) and per-agent similarity round r vs r-1 (drift)

Usage: python scripts/memprimary_8agent_compare.py [--out DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from analyses._shared import configure_matplotlib


def load_runs(condition_dir):
    runs = []
    for path in sorted(Path(condition_dir).rglob("*.json")):
        if path.name.endswith((".results.json", ".checkpoint.json")):
            continue
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def per_round_game(run):
    """Mean sent and return ratio per round across that round's dyads."""
    sent_rounds, rr_rounds = [], []
    for entry in run.get("conversation_history", []):
        dyads = [d for d in entry.get("dyads") or [] if d.get("sent") is not None]
        if not dyads:
            continue
        sent_rounds.append(np.mean([d["sent"] for d in dyads]))
        rr_rounds.append(
            np.mean([
                d["returned"] / d["received"] if d.get("received") else 0.0
                for d in dyads
                if d.get("returned") is not None
            ])
        )
    return np.array(sent_rounds), np.array(rr_rounds)


def myth_metrics(run, model):
    """Population convergence and per-agent drift from myth embeddings."""
    from sentence_transformers import util

    rounds = []
    for entry in run.get("conversation_history", []):
        myths = entry.get("myths") or {}
        if myths:
            rounds.append((entry["round"], dict(sorted(myths.items()))))
    if not rounds:
        return np.array([]), np.array([])

    agents = list(rounds[0][1])
    texts, index = [], {}
    for r, myths in rounds:
        for a, text in myths.items():
            index[(r, a)] = len(texts)
            texts.append(text)
    emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    sim = util.cos_sim(emb, emb).cpu().numpy()

    cross = []
    for r, myths in rounds:
        idx = [index[(r, a)] for a in myths]
        pair_vals = [sim[i, j] for k, i in enumerate(idx) for j in idx[k + 1:]]
        cross.append(float(np.mean(pair_vals)))

    drift = []
    for (r_prev, m_prev), (r_cur, m_cur) in zip(rounds, rounds[1:]):
        vals = [
            sim[index[(r_prev, a)], index[(r_cur, a)]]
            for a in agents
            if a in m_prev and a in m_cur
        ]
        drift.append(float(np.mean(vals)))
    return np.array(cross), np.array(drift)


def per_round_dyad_values(run, key="sent"):
    """All individual dyad values per round: list of (round, value)."""
    points = []
    for entry in run.get("conversation_history", []):
        for d in entry.get("dyads") or []:
            if d.get("sent") is None:
                continue
            if key == "sent":
                points.append((entry["round"], d["sent"]))
            elif key == "rr" and d.get("received"):
                points.append((entry["round"], d["returned"] / d["received"]))
    return points


def stack_mean_std(series_list):
    n = min(len(s) for s in series_list)
    arr = np.stack([s[:n] for s in series_list])
    return arr.mean(axis=0), arr.std(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", default="data/json/sonnet45_8agent_myth_directive_history3_anon_r10_n5"
    )
    parser.add_argument(
        "--fixed",
        default="data/json/sonnet45_8agent_myth_directive_history3_anon_memprimary_r10_n5",
    )
    parser.add_argument("--out", default="data/plots/memprimary_8agent")
    args = parser.parse_args()

    configure_matplotlib()
    import matplotlib.pyplot as plt
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-mpnet-base-v2")
    conditions = {
        "baseline (double memory)": load_runs(args.baseline),
        "memory-primary (fixed)": load_runs(args.fixed),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, runs in conditions.items():
        print(f"{name}: {len(runs)} runs")
        sents, rrs, crosses, drifts = [], [], [], []
        for run in runs:
            s, r = per_round_game(run)
            c, d = myth_metrics(run, model)
            sents.append(s)
            rrs.append(r)
            crosses.append(c)
            drifts.append(d)
        results[name] = {"sent": sents, "rr": rrs, "cross": crosses, "drift": drifts}

    print("\n=== Run-level means ===")
    for key, label in [
        ("sent", "sent ($/round, dyad mean)"),
        ("rr", "return ratio"),
        ("cross", "population myth similarity"),
        ("drift", "self myth similarity r vs r-1"),
    ]:
        parts = []
        for name, res in results.items():
            v = np.array([np.mean(s) for s in res[key]])
            parts.append(f"{name}: {v.mean():.3f} (±{v.std():.3f})")
        print(f"{label:32s} " + "  ".join(parts))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("sent", "Sent per round ($, dyad mean)"),
        ("rr", "Return ratio per round"),
        ("cross", "Population myth similarity"),
        ("drift", "Self myth similarity (r vs r-1)"),
    ]
    colors = ["tab:gray", "tab:green"]
    for ax, (key, title) in zip(axes.flat, panels):
        for (name, res), color in zip(results.items(), colors):
            mean, std = stack_mean_std(res[key])
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, label=name, color=color)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.legend(fontsize=8)
    fig.suptitle("8-agent regime: June 8 baseline vs memory-primary fix (Sonnet 4.5, n=5)")
    fig.tight_layout()
    fig_path = out_dir / "memprimary_8agent_comparison.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nFigure: {fig_path}")

    # Individual trajectories: one thin line per run (mean over the round's 4
    # dyads), individual dyad decisions scattered underneath, condition mean bold.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title in [
        (axes[0], "sent", "Sent per round — individual runs"),
        (axes[1], "rr", "Return ratio per round — individual runs"),
    ]:
        for (name, runs), color in zip(conditions.items(), colors):
            res_key = "sent" if key == "sent" else "rr"
            for s in results[name][res_key]:
                ax.plot(np.arange(1, len(s) + 1), s, color=color, alpha=0.35, lw=1)
            for run in runs:
                pts = per_round_dyad_values(run, key)
                if pts:
                    r, v = zip(*pts)
                    jitter = (np.random.default_rng(0).random(len(r)) - 0.5) * 0.25
                    ax.scatter(np.array(r) + jitter, v, s=6, color=color, alpha=0.15)
            mean, _ = stack_mean_std(results[name][res_key])
            ax.plot(np.arange(1, len(mean) + 1), mean, color=color, lw=2.5, label=name)
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.legend(fontsize=8)
    fig.suptitle("Individual run trajectories (thin) + dyad decisions (dots) + condition mean (bold)")
    fig.tight_layout()
    traj_path = out_dir / "memprimary_8agent_trajectories.png"
    fig.savefig(traj_path, dpi=150)
    print(f"Figure: {traj_path}")

    # Per-condition boxplots of run-level means (each box: n runs).
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    box_specs = [
        ("sent", "Sent ($/round)"),
        ("rr", "Return ratio"),
        ("cross", "Population myth sim"),
        ("drift", "Self myth sim (r vs r-1)"),
    ]
    for ax, (key, title) in zip(axes, box_specs):
        data = [
            [float(np.mean(s)) for s in results[name][key]] for name in results
        ]
        bp = ax.boxplot(data, labels=[n.split(" (")[0] for n in results], patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        for i, vals in enumerate(data, start=1):
            ax.scatter(
                np.full(len(vals), i) + (np.random.default_rng(1).random(len(vals)) - 0.5) * 0.15,
                vals,
                s=18,
                color=colors[i - 1],
                zorder=3,
            )
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Run-level means per condition (each dot = one run, n=5)")
    fig.tight_layout()
    box_path = out_dir / "memprimary_8agent_boxplots.png"
    fig.savefig(box_path, dpi=150)
    print(f"Figure: {box_path}")


if __name__ == "__main__":
    main()
