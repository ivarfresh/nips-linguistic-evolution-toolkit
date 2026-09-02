"""
Harvest seed myths for the memory-transplant ablation (Phase 1).

Reads pre-existing v4 direct-provider Sonnet 4.5 negative-noise runs and writes
a JSON manifest with 5 myths per pool:

- S-end+  : round-10 Agent_1 myth from top-25% joint-balance runs
- S-end-  : round-10 Agent_1 myth from bottom-25% joint-balance runs
- S-start : round-1 Agent_1 myth from `myth_game`-only runs (no game history
            baked in)

S-filler is generated separately (see data/seeds/fillers.json) — this script
records the seed-pool token-length distribution so the filler is length-matched.

See docs/memory_transplant_ablation_design.md §11.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path so we can import src/* if needed.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SOURCE_ROOT = (
    PROJECT_ROOT
    / "data/json/noise_experiments/v4_direct_provider"
    / "noise_negative_mem3_claude_sonnet_45/claude-sonnet-4.5"
)

# (task_order, noise_subdir) — informed pooled with uninformed per design §11.
SOURCE_CONDITIONS = [
    ("game_myth", "noisy_negative_5"),
    ("game_myth", "noisy_negative_5_informed"),
    ("myth_game", "noisy_negative_5"),
    ("myth_game", "noisy_negative_5_informed"),
]

OUTPUT_PATH = PROJECT_ROOT / "data/seeds/ablation_phase1_manifest.json"

NUM_PER_POOL = 5

# Reference thresholds from the design doc (§2). Recomputed here for drift check.
DOC_TOP_THRESHOLD = 67.3
DOC_BOTTOM_THRESHOLD = 58.5


def _is_source_run(path: Path) -> bool:
    """True iff path is a primary run file (not .results.json or .checkpoint.json)."""
    name = path.name
    if not name.endswith(".json"):
        return False
    if name.endswith(".results.json"):
        return False
    if name.endswith(".checkpoint.json"):
        return False
    if ".error" in name:
        return False
    return True


def load_runs() -> List[Dict[str, Any]]:
    """Load metadata for every source run across all 4 conditions."""
    runs: List[Dict[str, Any]] = []
    for task_order, noise_subdir in SOURCE_CONDITIONS:
        cond_dir = SOURCE_ROOT / task_order / noise_subdir
        if not cond_dir.is_dir():
            print(f"warning: missing source dir {cond_dir}", file=sys.stderr)
            continue
        for p in sorted(cond_dir.iterdir()):
            if not _is_source_run(p):
                continue
            try:
                with open(p, "r") as f:
                    state = json.load(f)
            except json.JSONDecodeError as e:
                print(f"warning: skipping malformed {p}: {e}", file=sys.stderr)
                continue
            balances = (state.get("game_data") or {}).get("balances", {}) or {}
            joint = float(sum(balances.values())) if balances else 0.0
            runs.append(
                {
                    "path": p,
                    "task_order": task_order,
                    "noise_subdir": noise_subdir,
                    "joint_balance": joint,
                    "conversation_history": state.get("conversation_history", []),
                }
            )
    return runs


def compute_percentile_cutoffs(joint_balances: List[float]) -> Tuple[float, float]:
    """Return (top-25% cutoff, bottom-25% cutoff)."""
    sorted_b = sorted(joint_balances)
    n = len(sorted_b)
    if n == 0:
        return (float("nan"), float("nan"))
    # 75th percentile (top cutoff): runs ABOVE this are top-25%.
    # Use the nearest-rank method for transparency.
    top_idx = max(0, min(n - 1, int(round(0.75 * (n - 1)))))
    bot_idx = max(0, min(n - 1, int(round(0.25 * (n - 1)))))
    return (sorted_b[top_idx], sorted_b[bot_idx])


def get_myth(run: Dict[str, Any], round_idx: int, agent_id: str = "Agent_1") -> Optional[str]:
    """Return Agent_1's myth from a given round (0-indexed), or None if missing."""
    ch = run["conversation_history"]
    if len(ch) <= round_idx:
        return None
    myths = (ch[round_idx] or {}).get("myths", {}) or {}
    text = myths.get(agent_id)
    if not text or not isinstance(text, str) or not text.strip():
        return None
    return text


def token_count(text: str) -> int:
    """Cheap whitespace token count proxy."""
    return len(text.split())


def select_pool(
    runs: List[Dict[str, Any]],
    *,
    seed_type: str,
    round_idx: int,
    filter_fn,
    n: int = NUM_PER_POOL,
) -> List[Dict[str, Any]]:
    """Pick n distinct runs (deterministically by sorted path) that pass filter_fn
    and have a non-empty Agent_1 myth at round_idx.

    Returns a list of seed records.
    """
    candidates = [r for r in runs if filter_fn(r)]
    candidates.sort(key=lambda r: str(r["path"]))

    seeds: List[Dict[str, Any]] = []
    seen_paths: set = set()
    for r in candidates:
        if r["path"] in seen_paths:
            continue
        myth = get_myth(r, round_idx)
        if myth is None:
            continue
        seen_paths.add(r["path"])
        seeds.append(
            {
                "seed_id": f"{seed_type}_{len(seeds):02d}",
                "seed_type": seed_type,
                "myth_text": myth,
                "source_run_path": str(r["path"].relative_to(PROJECT_ROOT)),
                "source_task_order": r["task_order"],
                "source_noise_subdir": r["noise_subdir"],
                "source_joint_balance": round(r["joint_balance"], 4),
                "token_count": token_count(myth),
            }
        )
        if len(seeds) >= n:
            break
    return seeds


def summarize_token_stats(seeds: List[Dict[str, Any]]) -> Dict[str, float]:
    counts = [s["token_count"] for s in seeds]
    if not counts:
        return {"min": 0, "median": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(counts),
        "median": int(statistics.median(counts)),
        "max": max(counts),
        "mean": round(statistics.mean(counts), 1),
    }


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def main() -> int:
    print(f"Loading source runs from {SOURCE_ROOT.relative_to(PROJECT_ROOT)}/ ...")
    runs = load_runs()
    print(f"  Loaded {len(runs)} runs across {len(SOURCE_CONDITIONS)} conditions.")

    joint_balances = [r["joint_balance"] for r in runs]
    top_cutoff, bot_cutoff = compute_percentile_cutoffs(joint_balances)
    print()
    print("Joint balance distribution:")
    print(f"  n              = {len(joint_balances)}")
    if joint_balances:
        print(f"  min/median/max = {min(joint_balances):.2f} / "
              f"{statistics.median(joint_balances):.2f} / {max(joint_balances):.2f}")
        print(f"  mean (sd)      = {statistics.mean(joint_balances):.2f} "
              f"(±{statistics.stdev(joint_balances):.2f})")
    print(f"  top 25% cutoff = {top_cutoff:.2f}  (design doc: {DOC_TOP_THRESHOLD})")
    print(f"  bot 25% cutoff = {bot_cutoff:.2f}  (design doc: {DOC_BOTTOM_THRESHOLD})")

    # Use the doc thresholds (these are pre-registered) but also report the
    # recomputed cutoffs so drift is visible.
    use_top = DOC_TOP_THRESHOLD
    use_bot = DOC_BOTTOM_THRESHOLD

    print()
    print("Selecting pools...")
    s_end_plus = select_pool(
        runs,
        seed_type="s_end_plus",
        round_idx=9,  # round 10 (0-indexed)
        filter_fn=lambda r: r["joint_balance"] >= use_top,
    )
    s_end_minus = select_pool(
        runs,
        seed_type="s_end_minus",
        round_idx=9,
        filter_fn=lambda r: r["joint_balance"] <= use_bot,
    )
    s_start = select_pool(
        runs,
        seed_type="s_start",
        round_idx=0,  # round 1
        filter_fn=lambda r: r["task_order"] == "myth_game",
    )

    pools = {
        "s_end_plus": s_end_plus,
        "s_end_minus": s_end_minus,
        "s_start": s_start,
    }

    print()
    print("Pool summary (token counts via whitespace split):")
    print(f"  {'pool':<14} {'n':>3} {'min':>5} {'median':>7} {'max':>5} {'mean':>6}")
    for name, seeds in pools.items():
        stats = summarize_token_stats(seeds)
        print(f"  {name:<14} {len(seeds):>3} {stats['min']:>5} "
              f"{stats['median']:>7} {stats['max']:>5} {stats['mean']:>6}")

    # Top-level pool of all myth seeds for filler-length matching.
    all_seeds = [s for pool in pools.values() for s in pool]
    seed_pool_token_stats = summarize_token_stats(all_seeds)

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "design_doc": "docs/memory_transplant_ablation_design.md",
        "source_root": str(SOURCE_ROOT.relative_to(PROJECT_ROOT)),
        "source_conditions": [list(c) for c in SOURCE_CONDITIONS],
        "n_source_runs": len(runs),
        "joint_balance_summary": {
            "n": len(joint_balances),
            "min": round(min(joint_balances), 4) if joint_balances else None,
            "max": round(max(joint_balances), 4) if joint_balances else None,
            "mean": round(statistics.mean(joint_balances), 4) if joint_balances else None,
            "median": round(statistics.median(joint_balances), 4) if joint_balances else None,
            "stdev": round(statistics.stdev(joint_balances), 4) if len(joint_balances) > 1 else None,
            "recomputed_top_25_cutoff": round(top_cutoff, 4),
            "recomputed_bottom_25_cutoff": round(bot_cutoff, 4),
            "doc_top_25_cutoff": DOC_TOP_THRESHOLD,
            "doc_bottom_25_cutoff": DOC_BOTTOM_THRESHOLD,
            "applied_top_cutoff": use_top,
            "applied_bottom_cutoff": use_bot,
        },
        "seed_pool_token_stats": seed_pool_token_stats,
        "pools": pools,
    }

    atomic_write_json(OUTPUT_PATH, manifest)
    print()
    print(f"Wrote manifest -> {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
