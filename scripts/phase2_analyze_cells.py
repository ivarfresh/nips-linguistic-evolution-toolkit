"""Phase 2 cell analysis: compare seeded cells against the S-none baseline
within each task order. Reports H1–H4 contrasts.
"""

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict
from math import sqrt
from pathlib import Path


def _load_runs(root_dir, extract_meta):
    paths = sorted(
        p
        for p in glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True)
        if not any(skip in p for skip in (".checkpoint.", ".results.", ".error."))
    )
    runs = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        balances = data.get("game_data", {}).get("balances", {})
        if not balances:
            continue
        joint = sum(balances.values())
        meta = extract_meta(data, p) or {}
        meta["path"] = p
        meta["joint"] = joint
        meta["per_agent"] = list(balances.values())
        runs.append(meta)
    return runs


def _baseline_meta(data, path):
    task_order = data.get("task_order") or []
    return {
        "task_order": tuple(task_order),
        "seed_type": "s_none",
    }


def _seeded_meta(data, path):
    meta = data.get("run_metadata", {}) or {}
    task_order = data.get("task_order") or meta.get("task_order") or []
    return {
        "task_order": tuple(task_order),
        "seed_type": meta.get("phase2_seed_type", "unknown"),
    }


def _t_two_sample(group_a, group_b):
    n_a = len(group_a)
    n_b = len(group_b)
    mean_a = statistics.mean(group_a)
    mean_b = statistics.mean(group_b)
    var_a = statistics.variance(group_a) if n_a > 1 else 0
    var_b = statistics.variance(group_b) if n_b > 1 else 0
    se = sqrt((var_a / n_a) + (var_b / n_b)) if (n_a > 0 and n_b > 0) else 0
    delta = mean_a - mean_b
    z = delta / se if se else 0
    return delta, se, z, mean_a, mean_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-dir",
        required=True,
        help="Phase 2 baseline runs directory",
    )
    parser.add_argument(
        "--seeded-dir",
        required=True,
        help="Phase 2 seeded cell runs directory",
    )
    parser.add_argument(
        "--out",
        default="data/phase2/cell_analysis.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    baseline_runs = _load_runs(args.baseline_dir, _baseline_meta)
    seeded_runs = _load_runs(args.seeded_dir, _seeded_meta)

    by_to_seed = defaultdict(list)
    for r in baseline_runs:
        by_to_seed[(r["task_order"], "s_none")].append(r["joint"])
    for r in seeded_runs:
        by_to_seed[(r["task_order"], r["seed_type"])].append(r["joint"])

    task_orders = sorted({to for (to, _) in by_to_seed.keys()})
    seed_types = ["s_none", "s_start", "s_end_plus", "s_end_minus", "s_filler"]

    print("=" * 80)
    print("Phase 2 cell-level summary (joint balance)")
    print("=" * 80)
    summary = {}
    for to in task_orders:
        to_str = "/".join(to)
        print(f"\nTask order: {to_str}")
        cell_summary = {}
        none_joints = by_to_seed.get((to, "s_none"), [])
        for seed in seed_types:
            joints = by_to_seed.get((to, seed), [])
            n = len(joints)
            mean = statistics.mean(joints) if joints else 0
            std = statistics.stdev(joints) if n > 1 else 0
            cell_summary[seed] = {
                "n": n,
                "mean": mean,
                "std": std,
                "joints": joints,
            }
            print(f"  {seed:14}: n={n:3d}, mean={mean:7.2f}, std={std:6.2f}")
        summary[to_str] = {"cells": cell_summary}

        if none_joints and len(none_joints) > 1:
            contrasts = {}
            for seed in seed_types:
                if seed == "s_none":
                    continue
                seed_joints = by_to_seed.get((to, seed), [])
                if len(seed_joints) < 2:
                    continue
                delta, se, z, mean_a, mean_b = _t_two_sample(seed_joints, none_joints)
                contrasts[f"{seed}_vs_s_none"] = {
                    "delta": delta,
                    "se": se,
                    "z": z,
                    "seed_mean": mean_a,
                    "baseline_mean": mean_b,
                }
                print(
                    f"  {seed} vs s_none: Δ={delta:+7.2f}, SE={se:5.2f}, z={z:+5.2f}"
                )
            # H2: s_end_plus vs s_filler
            ep = by_to_seed.get((to, "s_end_plus"), [])
            sf = by_to_seed.get((to, "s_filler"), [])
            if len(ep) > 1 and len(sf) > 1:
                delta, se, z, ma, mb = _t_two_sample(ep, sf)
                contrasts["s_end_plus_vs_s_filler"] = {
                    "delta": delta, "se": se, "z": z,
                    "seed_mean": ma, "baseline_mean": mb,
                }
                print(f"  H2 (s_end_plus vs s_filler): Δ={delta:+7.2f}, SE={se:5.2f}, z={z:+5.2f}")
            # H3: s_end_plus vs s_end_minus
            em = by_to_seed.get((to, "s_end_minus"), [])
            if len(ep) > 1 and len(em) > 1:
                delta, se, z, ma, mb = _t_two_sample(ep, em)
                contrasts["s_end_plus_vs_s_end_minus"] = {
                    "delta": delta, "se": se, "z": z,
                    "seed_mean": ma, "baseline_mean": mb,
                }
                print(f"  H3 (s_end_plus vs s_end_minus): Δ={delta:+7.2f}, SE={se:5.2f}, z={z:+5.2f}")
            # H4: s_end_plus vs s_start
            ss = by_to_seed.get((to, "s_start"), [])
            if len(ep) > 1 and len(ss) > 1:
                delta, se, z, ma, mb = _t_two_sample(ep, ss)
                contrasts["s_end_plus_vs_s_start"] = {
                    "delta": delta, "se": se, "z": z,
                    "seed_mean": ma, "baseline_mean": mb,
                }
                print(f"  H4 (s_end_plus vs s_start): Δ={delta:+7.2f}, SE={se:5.2f}, z={z:+5.2f}")
            summary[to_str]["contrasts"] = contrasts

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
