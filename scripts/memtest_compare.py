#!/usr/bin/env python3
"""Compare the memory-channel pilot conditions (hybrid vs stateless).

Loads all runs from the two memtest experiment dirs and reports, per condition:
- game behavior: sent and return ratio per round (mean ±std across reps)
- myth dynamics: cross-agent embedding similarity per round (convergence) and
  round-to-round self-similarity per agent (lineage inertia)

Usage:
    python scripts/memtest_compare.py \
        --hybrid data/json/memtest_hybrid_sonnet45_2agent_r10_n5 \
        --stateless data/json/memtest_stateless_sonnet45_2agent_r10_n5 \
        --out data/plots/memtest_memory_channels
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
        name = path.name
        if name.endswith((".results.json", ".checkpoint.json")):
            continue
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def per_round_game(run):
    """Return (rounds, sent, return_ratio) arrays for one run."""
    rounds, sent, rr = [], [], []
    for entry in run.get("conversation_history", []):
        if entry.get("sent") is None or entry.get("returned") is None:
            continue
        rounds.append(entry["round"])
        sent.append(entry["sent"])
        received = entry.get("received") or 0
        rr.append(entry["returned"] / received if received else 0.0)
    return np.array(rounds), np.array(sent, dtype=float), np.array(rr, dtype=float)


def myths_by_round(run):
    """Return {round: {agent_id: myth_text}} for rounds with both myths."""
    out = {}
    for entry in run.get("conversation_history", []):
        myths = entry.get("myths") or {}
        if len(myths) >= 2:
            out[entry["round"]] = myths
    return out


def embed_similarities(runs, model):
    """Per run: cross-agent cosine per round, and per-agent self-similarity
    between consecutive rounds. Returns (cross[rep][round_idx], self_sim[...])."""
    from sentence_transformers import util

    cross_all, self_all = [], []
    for run in runs:
        by_round = myths_by_round(run)
        rounds = sorted(by_round)
        agents = sorted(by_round[rounds[0]]) if rounds else []
        if len(agents) < 2:
            continue
        texts, index = [], {}
        for r in rounds:
            for a in agents:
                index[(r, a)] = len(texts)
                texts.append(by_round[r][a])
        emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        sim = util.cos_sim(emb, emb).cpu().numpy()

        cross = [sim[index[(r, agents[0])], index[(r, agents[1])]] for r in rounds]
        self_sim = []
        for prev, cur in zip(rounds, rounds[1:]):
            vals = [sim[index[(prev, a)], index[(cur, a)]] for a in agents]
            self_sim.append(float(np.mean(vals)))
        cross_all.append(np.array(cross))
        self_all.append(np.array(self_sim))
    return cross_all, self_all


def stack_mean_std(series_list):
    n = min(len(s) for s in series_list)
    arr = np.stack([s[:n] for s in series_list])
    return arr.mean(axis=0), arr.std(axis=0)


def welch(a, b):
    try:
        from scipy import stats

        t, p = stats.ttest_ind(a, b, equal_var=False)
        return float(t), float(p)
    except ImportError:
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", default="data/json/memtest_hybrid_sonnet45_2agent_r10_n5")
    parser.add_argument("--stateless", default="data/json/memtest_stateless_sonnet45_2agent_r10_n5")
    parser.add_argument("--out", default="data/plots/memtest_memory_channels")
    args = parser.parse_args()

    configure_matplotlib()
    import matplotlib.pyplot as plt
    from sentence_transformers import SentenceTransformer

    conditions = {"hybrid": load_runs(args.hybrid), "stateless": load_runs(args.stateless)}
    for name, runs in conditions.items():
        print(f"{name}: {len(runs)} runs")

    model = SentenceTransformer("all-mpnet-base-v2")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    game = {}
    myth = {}
    summary = {}
    for name, runs in conditions.items():
        sents, rrs = [], []
        for run in runs:
            _, sent, rr = per_round_game(run)
            sents.append(sent)
            rrs.append(rr)
        cross, self_sim = embed_similarities(runs, model)
        game[name] = (sents, rrs)
        myth[name] = (cross, self_sim)
        summary[name] = {
            "mean_sent_per_run": [float(np.mean(s)) for s in sents],
            "mean_return_ratio_per_run": [float(np.mean(r)) for r in rrs],
            "mean_cross_sim_per_run": [float(np.mean(c)) for c in cross],
            "mean_self_sim_per_run": [float(np.mean(s)) for s in self_sim],
        }

    print("\n=== Run-level means (each n = number of runs) ===")
    metrics = [
        ("mean_sent_per_run", "sent ($/round)"),
        ("mean_return_ratio_per_run", "return ratio"),
        ("mean_cross_sim_per_run", "cross-agent myth sim"),
        ("mean_self_sim_per_run", "self myth sim (r vs r-1)"),
    ]
    for key, label in metrics:
        h = np.array(summary["hybrid"][key])
        s = np.array(summary["stateless"][key])
        t, p = welch(h, s)
        p_str = f"Welch t={t:.2f} p={p:.3f}" if p is not None else "scipy unavailable"
        print(
            f"{label:26s} hybrid {h.mean():.3f} (±{h.std():.3f})  "
            f"stateless {s.mean():.3f} (±{s.std():.3f})  {p_str}"
        )

    panels = [
        ("Sent per round ($)", [(n, stack_mean_std(game[n][0])) for n in conditions]),
        ("Return ratio per round", [(n, stack_mean_std(game[n][1])) for n in conditions]),
        ("Cross-agent myth similarity", [(n, stack_mean_std(myth[n][0])) for n in conditions]),
        ("Self myth similarity (round r vs r-1)", [(n, stack_mean_std(myth[n][1])) for n in conditions]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {"hybrid": "tab:blue", "stateless": "tab:orange"}
    for ax, (title, series) in zip(axes.flat, panels):
        for name, (mean, std) in series:
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, label=name, color=colors[name])
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=colors[name])
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.legend()
    fig.suptitle("Memory-channel pilot: hybrid vs stateless (Sonnet 4.5, 2-agent dyads, n=5)")
    fig.tight_layout()
    fig_path = out_dir / "memtest_comparison.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nFigure: {fig_path}")

    with open(out_dir / "memtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
