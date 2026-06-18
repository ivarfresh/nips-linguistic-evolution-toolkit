"""Phase 2 seed harvester (memory-transplant ablation §17.4).

Scans a directory of completed noisy 8-agent anon-history3-directive
runs and produces a seed manifest with five seed types per task order:

- S-start: round-1 myths from baseline source runs
- S-end+: round-10 myths from highest-joint baseline source runs
- S-end-: round-10 myths from lowest-joint baseline source runs
- S-filler: length-matched Wikipedia paragraphs (loaded from
  data/share/phase2_filler_paragraphs.json if present, else skipped)
- S-none: no seed (no manifest entries; the baseline runs themselves
  serve as the comparator)

Output: data/phase2/seed_manifest.json
"""

import argparse
import glob
import json
import os
import statistics
import sys
from pathlib import Path


def _load_run(path):
    with open(path) as f:
        return json.load(f)


def _is_main_run(path):
    name = os.path.basename(path)
    return not any(skip in name for skip in (".checkpoint.", ".results.", ".error."))


def _joint_balance(data):
    balances = data.get("game_data", {}).get("balances", {})
    return sum(balances.values())


def _myths_by_round(data):
    myths = {}
    for entry in data.get("conversation_history", []) or []:
        round_num = entry.get("round")
        if round_num is None:
            continue
        round_myths = entry.get("myths") or {}
        if round_myths:
            myths[round_num] = round_myths
    return myths


def _median_agent_id(data):
    balances = data.get("game_data", {}).get("balances", {})
    if not balances:
        return None
    items = sorted(balances.items(), key=lambda kv: kv[1])
    return items[len(items) // 2][0]


def _gather_runs(source_dir):
    paths = sorted(
        p
        for p in glob.glob(os.path.join(source_dir, "**", "*.json"), recursive=True)
        if _is_main_run(p)
    )
    runs = []
    for p in paths:
        data = _load_run(p)
        joint = _joint_balance(data)
        myths = _myths_by_round(data)
        if not myths:
            continue
        agent_id = _median_agent_id(data)
        if agent_id is None:
            continue
        task_order = data.get("task_order") or data.get("run_metadata", {}).get(
            "task_order"
        )
        runs.append(
            {
                "path": p,
                "joint": joint,
                "myths": myths,
                "agent_id": agent_id,
                "task_order": task_order,
            }
        )
    return runs


def _draw_round_myth(run, round_num, agent_id_override=None):
    agent_id = agent_id_override or run["agent_id"]
    text = run["myths"].get(round_num, {}).get(agent_id)
    if not text:
        return None
    return {
        "source_run": run["path"],
        "agent_id": agent_id,
        "round": round_num,
        "joint_at_source": run["joint"],
        "text": text.strip(),
        "tokens": len(text.split()),
    }


def _quartile_split(runs, num_per_pool):
    sorted_runs = sorted(runs, key=lambda r: r["joint"], reverse=True)
    n = len(sorted_runs)
    if n == 0:
        return [], []
    take = max(num_per_pool, min(n // 4, num_per_pool))
    top = sorted_runs[:take]
    bottom = sorted_runs[-take:]
    return top, bottom


def harvest(source_dir, num_per_pool, output_path):
    runs = _gather_runs(source_dir)
    if not runs:
        raise SystemExit(f"No usable runs with myths under {source_dir}")

    joints = [r["joint"] for r in runs]
    print(
        f"Loaded {len(runs)} runs with myths. "
        f"Joint balance mean={statistics.mean(joints):.2f} std={statistics.stdev(joints) if len(joints) > 1 else 0:.2f} "
        f"range=[{min(joints):.0f}, {max(joints):.0f}]"
    )

    s_start = []
    for run in runs:
        m = _draw_round_myth(run, 1)
        if m is not None:
            s_start.append(m)
    s_start = sorted(s_start, key=lambda m: m["joint_at_source"], reverse=True)[
        :num_per_pool
    ]

    top, bottom = _quartile_split(runs, num_per_pool)
    s_end_plus = []
    for run in top:
        m = _draw_round_myth(run, 10)
        if m is not None:
            s_end_plus.append(m)
    s_end_plus = s_end_plus[:num_per_pool]

    s_end_minus = []
    for run in bottom:
        m = _draw_round_myth(run, 10)
        if m is not None:
            s_end_minus.append(m)
    s_end_minus = s_end_minus[:num_per_pool]

    target_lengths = [m["tokens"] for m in s_start + s_end_plus + s_end_minus]
    mean_len = (
        statistics.mean(target_lengths) if target_lengths else 200
    )

    filler_pool_path = Path("data/share/phase2_filler_paragraphs.json")
    s_filler = []
    if filler_pool_path.exists():
        with open(filler_pool_path) as f:
            filler_pool = json.load(f)
        filler_pool = sorted(
            filler_pool, key=lambda item: abs(len(item["text"].split()) - mean_len)
        )
        for item in filler_pool[:num_per_pool]:
            s_filler.append(
                {
                    "source_run": None,
                    "agent_id": None,
                    "round": None,
                    "joint_at_source": None,
                    "text": item["text"].strip(),
                    "tokens": len(item["text"].split()),
                    "source": item.get("source"),
                }
            )

    manifest = {
        "source_dir": source_dir,
        "num_per_pool": num_per_pool,
        "baseline_stats": {
            "n_runs": len(runs),
            "joint_mean": statistics.mean(joints),
            "joint_std": statistics.stdev(joints) if len(joints) > 1 else 0,
            "joint_range": [min(joints), max(joints)],
            "myth_mean_tokens": mean_len,
        },
        "seeds": {
            "s_start": s_start,
            "s_end_plus": s_end_plus,
            "s_end_minus": s_end_minus,
            "s_filler": s_filler,
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {output_path}")
    print(
        f"Seeds: s_start={len(s_start)} s_end_plus={len(s_end_plus)} "
        f"s_end_minus={len(s_end_minus)} s_filler={len(s_filler)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", help="Directory containing baseline runs")
    parser.add_argument("--num-per-pool", type=int, default=5)
    parser.add_argument(
        "--output",
        default="data/phase2/seed_manifest.json",
        help="Output manifest path",
    )
    args = parser.parse_args()
    harvest(args.source_dir, args.num_per_pool, args.output)


if __name__ == "__main__":
    main()
