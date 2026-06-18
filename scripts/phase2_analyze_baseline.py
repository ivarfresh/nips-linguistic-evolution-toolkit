"""Phase 2 baseline analysis: compute σ_host per task order, pre-register
text features for source myth pools, and emit thresholds for §17.5.
"""

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


def _load_runs(root_dir):
    paths = sorted(
        p
        for p in glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True)
        if not any(skip in p for skip in (".checkpoint.", ".results.", ".error."))
    )
    runs = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        task_order = data.get("task_order") or data.get("run_metadata", {}).get(
            "task_order"
        )
        if not task_order:
            continue
        balances = data.get("game_data", {}).get("balances", {})
        if not balances:
            continue
        runs.append(
            {
                "path": p,
                "task_order": tuple(task_order),
                "joint": sum(balances.values()),
                "per_agent": list(balances.values()),
                "data": data,
            }
        )
    return runs


def _myth_lengths(run):
    lens = []
    for entry in run["data"].get("conversation_history", []) or []:
        myths = entry.get("myths") or {}
        for text in myths.values():
            if isinstance(text, str):
                lens.append(len(text.split()))
    return lens


def summarize(runs):
    by_task_order = defaultdict(list)
    for r in runs:
        by_task_order[r["task_order"]].append(r)

    out = {}
    for task_order, group in sorted(by_task_order.items()):
        joints = [g["joint"] for g in group]
        per_agent = [v for g in group for v in g["per_agent"]]
        mean_joint = statistics.mean(joints) if joints else 0
        std_joint = statistics.stdev(joints) if len(joints) > 1 else 0
        out[" / ".join(task_order)] = {
            "n": len(group),
            "joint_mean": mean_joint,
            "joint_std": std_joint,
            "joint_range": [min(joints), max(joints)] if joints else [None, None],
            "per_agent_mean": statistics.mean(per_agent) if per_agent else 0,
            "per_agent_std": (
                statistics.stdev(per_agent) if len(per_agent) > 1 else 0
            ),
            "per_run_joints": joints,
        }
    return out


def threshold_table(sigma_host, baseline_mean):
    return {
        "H1 (S-end+ > S-none) reproduces": f"Δjoint ≥ +{sigma_host:.2f} (S-end+ mean ≥ {baseline_mean + sigma_host:.2f})",
        "H1 null": f"|Δjoint| ≤ {0.5 * sigma_host:.2f} with overlapping CI",
        "H2 (content vs filler) significant": f"Δjoint ≥ +{0.75 * sigma_host:.2f}",
        "H3 (cooperative content) significant": f"Δjoint ≥ +{0.75 * sigma_host:.2f}",
        "H4 (refinement) significant": f"Δjoint ≥ +{0.5 * sigma_host:.2f}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_dir",
        help="Path to baseline runs directory (e.g., data/json/noise_experiments/phase2_baseline/phase2_baseline_neg5)",
    )
    parser.add_argument(
        "--out",
        default="data/phase2/baseline_summary.json",
        help="Output summary JSON path",
    )
    args = parser.parse_args()

    runs = _load_runs(args.root_dir)
    if not runs:
        raise SystemExit(f"No runs found under {args.root_dir}")

    summary = summarize(runs)

    print("=" * 80)
    print("Phase 2 baseline summary")
    print("=" * 80)
    for task_order, s in summary.items():
        print(f"\nTask order: {task_order}")
        print(f"  n = {s['n']}")
        print(f"  Joint mean (±std):   {s['joint_mean']:.2f} ({s['joint_std']:.2f})")
        print(
            f"  Joint range: [{s['joint_range'][0]:.0f}, {s['joint_range'][1]:.0f}]"
        )
        print(f"  Per-agent mean (±std): {s['per_agent_mean']:.2f} ({s['per_agent_std']:.2f})")
        print(f"  Thresholds vs S-none:")
        for hyp, rule in threshold_table(s["joint_std"], s["joint_mean"]).items():
            print(f"    {hyp}: {rule}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out}")

    # Myth length distribution across all baseline runs (for filler matching)
    all_lens = [n for r in runs for n in _myth_lengths(r)]
    if all_lens:
        print(
            f"\nMyth length distribution: mean={statistics.mean(all_lens):.1f} "
            f"std={statistics.stdev(all_lens) if len(all_lens) > 1 else 0:.1f} "
            f"range=[{min(all_lens)}, {max(all_lens)}]"
        )


if __name__ == "__main__":
    main()
