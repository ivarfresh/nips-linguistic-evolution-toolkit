"""Compare cross-judge scores in the Phase 2 seed manifest.

Reads `data/phase2/seed_manifest.json` (assumed to contain both Sonnet 4.5
and Opus 4.7 scores under seed["judge_score"] / seed["judge_scores"]) and
emits a per-myth comparison table + per-pool aggregates + a rank-order
correlation, plus an updated H5 (Spearman ρ judge × joint) per task order.
"""

import glob
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


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


def _scores(seed):
    """Return (sonnet, opus) tuple."""
    sonnet = seed.get("judge_score")
    js = seed.get("judge_scores", {}) or {}
    if sonnet is None:
        sonnet = (
            js.get("claude-sonnet-4_5")
            or js.get("claude-sonnet-4.5")
            or js.get("sonnet_45")
        )
    opus = js.get("opus_4_7") or js.get("claude-opus-4-7")
    return sonnet, opus


def main():
    manifest_path = Path("data/phase2/seed_manifest.json")
    manifest = json.load(open(manifest_path))

    seed_types = ["s_start", "s_end_plus", "s_end_minus", "s_filler"]

    print("=" * 80)
    print("Cross-judge comparison (Sonnet 4.5 vs Opus 4.7)")
    print("=" * 80)
    print()
    print(f"{'Pool':14} {'Sonnet 4.5':12} {'Opus 4.7':12} {'Δ (Opus-Son)':14}")
    pool_aggregate = {}
    all_son, all_opus = [], []
    for st in seed_types:
        seeds = manifest["seeds"].get(st, [])
        sons, opuses = [], []
        for s in seeds:
            sn, op = _scores(s)
            if sn is not None:
                sons.append(sn)
            if op is not None:
                opuses.append(op)
        if not (sons and opuses):
            continue
        son_mean = statistics.mean(sons)
        opus_mean = statistics.mean(opuses)
        pool_aggregate[st] = {
            "sonnet_mean": son_mean,
            "opus_mean": opus_mean,
            "delta": opus_mean - son_mean,
            "sonnet_values": sons,
            "opus_values": opuses,
        }
        print(
            f"{st:14} {son_mean:6.2f} ±{statistics.pstdev(sons):4.2f} "
            f"{opus_mean:6.2f} ±{statistics.pstdev(opuses):4.2f} "
            f"{opus_mean - son_mean:+6.2f}"
        )
        all_son.extend(sons)
        all_opus.extend(opuses)

    print()
    if len(all_son) == len(all_opus) and len(all_son) > 1:
        rho = _spearman(all_son, all_opus)
        pearson_cov = sum(
            (a - statistics.mean(all_son)) * (b - statistics.mean(all_opus))
            for a, b in zip(all_son, all_opus)
        ) / len(all_son)
        sx = statistics.pstdev(all_son)
        sy = statistics.pstdev(all_opus)
        pearson = pearson_cov / (sx * sy) if sx and sy else 0
        print(f"Across all 20 seed myths:")
        print(f"  Spearman ρ(Sonnet, Opus) = {rho:+.3f}")
        print(f"  Pearson  r(Sonnet, Opus) = {pearson:+.3f}")
        print()

    # Surprising-finding sanity checks
    print("=" * 80)
    print("Key Phase 2 surprises — do they replicate across judges?")
    print("=" * 80)
    print()

    sep = pool_aggregate.get("s_end_plus", {})
    sem = pool_aggregate.get("s_end_minus", {})
    ss = pool_aggregate.get("s_start", {})
    if sep and sem:
        print(
            f"1) 'Counter-current': s_end_minus > s_end_plus on cooperativeness?"
        )
        print(
            f"     Sonnet: end_minus {sem['sonnet_mean']:.2f} vs end_plus "
            f"{sep['sonnet_mean']:.2f} → Δ = {sem['sonnet_mean'] - sep['sonnet_mean']:+.2f}"
        )
        print(
            f"     Opus:   end_minus {sem['opus_mean']:.2f} vs end_plus "
            f"{sep['opus_mean']:.2f} → Δ = {sem['opus_mean'] - sep['opus_mean']:+.2f}"
        )
        sonnet_supports = sem["sonnet_mean"] > sep["sonnet_mean"]
        opus_supports = sem["opus_mean"] > sep["opus_mean"]
        verdict = (
            "REPLICATES"
            if sonnet_supports and opus_supports
            else "FAILS TO REPLICATE"
            if not opus_supports and sonnet_supports
            else "MIXED"
        )
        print(f"     Verdict: {verdict}")
        print()

    if ss and sep:
        print(f"2) 'H4 reversal': s_start (round-1 parable) > s_end_plus on cooperativeness?")
        print(
            f"     Sonnet: s_start {ss['sonnet_mean']:.2f} vs s_end_plus "
            f"{sep['sonnet_mean']:.2f} → Δ = {ss['sonnet_mean'] - sep['sonnet_mean']:+.2f}"
        )
        print(
            f"     Opus:   s_start {ss['opus_mean']:.2f} vs s_end_plus "
            f"{sep['opus_mean']:.2f} → Δ = {ss['opus_mean'] - sep['opus_mean']:+.2f}"
        )
        sonnet_supports = ss["sonnet_mean"] > sep["sonnet_mean"]
        opus_supports = ss["opus_mean"] > sep["opus_mean"]
        verdict = (
            "REPLICATES"
            if sonnet_supports and opus_supports
            else "FAILS TO REPLICATE"
            if not opus_supports and sonnet_supports
            else "MIXED"
        )
        print(f"     Verdict (judge-side gap): {verdict}")
        print()

    # H5 Spearman ρ(judge, joint) per task order using BOTH judges
    print("=" * 80)
    print("H5 dose-response — re-computed with each judge")
    print("=" * 80)
    print()

    seed_score_map = {}
    for st in seed_types:
        for i, s in enumerate(manifest["seeds"].get(st, [])):
            sn, op = _scores(s)
            seed_score_map[(st, i)] = (sn, op)

    task_orders = ["game", "game_myth", "myth_game"]
    rows = []
    for to in task_orders:
        xs_sonnet, xs_opus, ys = [], [], []
        for st in seed_types:
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
        if not xs_sonnet:
            continue
        rho_s = _spearman(xs_sonnet, ys)
        rho_o = _spearman(xs_opus, ys)
        rows.append((to, rho_s, rho_o))
        print(
            f"  {to:11}  n={len(ys):3d}  ρ(Sonnet, joint) = {rho_s:+.3f}   "
            f"ρ(Opus, joint) = {rho_o:+.3f}"
        )
    print()

    # Save a structured summary
    out = {
        "manifest_path": str(manifest_path),
        "pool_aggregate": pool_aggregate,
        "h5_by_task_order": [
            {"task_order": to, "rho_sonnet": rs, "rho_opus": ro}
            for to, rs, ro in rows
        ],
        "across_all_seeds": {
            "n": len(all_son),
            "spearman_rho_sonnet_vs_opus": _spearman(all_son, all_opus) if len(all_son) > 1 else 0,
        },
    }
    out_path = Path("data/phase2/judge_compare.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
