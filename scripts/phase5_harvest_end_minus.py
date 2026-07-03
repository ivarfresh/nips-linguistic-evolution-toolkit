"""Phase 5 — harvest 5 low-cooperation round-10 myths as the `s_end_minus` pool.

Source: Phase 2 noisy baselines (no-seed runs under noise neg5 — valid baseline
data, distinct from the Phase 2 *ablation* cells). Rank by per-agent final
balance, pick the lowest-cooperation agent's round-10 myth from each of the
lowest-cooperation source runs.

Writes the pool into data/phase3/seed_manifest.json under `seeds.s_end_minus`.
"""

import glob
import json
import re
import statistics
from pathlib import Path


MANIFEST_PATH = Path("data/phase3/seed_manifest.json")
REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_GLOB = str(REPO_ROOT / "data/json/noise_experiments/phase2_baseline/phase2_baseline_neg5/claude-sonnet-4.5/myth_game/phase2_8agent_history3_anon_neg5/*.json")


def word_count(text):
    return len(re.findall(r"\b[a-zA-Z']+\b", text))


def joint_balance(d):
    ch = d.get("conversation_history", [])
    bal = ch[-1].get("balances", {}) if ch else {}
    return sum(bal.values()) if bal else None


def harvest():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    files = sorted(glob.glob(SOURCE_GLOB))
    files = [f for f in files if ".checkpoint" not in f and ".results" not in f]
    print(f"Found {len(files)} Phase 2 noisy myth_game baseline runs")

    # Score each run, collect (joint, path, low_agent, low_balance, round10_myth)
    candidates = []
    for path in files:
        with open(path) as f:
            d = json.load(f)
        ch = d.get("conversation_history", [])
        if not ch:
            continue
        final_balances = ch[-1].get("balances") or {}
        if not final_balances:
            continue
        joint = sum(final_balances.values())

        # Find lowest-balance agent
        low_agent, low_bal = min(final_balances.items(), key=lambda kv: kv[1])

        # Find that agent's round-10 myth
        r10_myth = None
        for entry in ch:
            if entry.get("round") == 10 and "myths" in entry:
                r10_myth = entry["myths"].get(low_agent)
                break
        if not r10_myth:
            continue

        candidates.append({
            "joint": joint,
            "low_agent": low_agent,
            "low_balance": low_bal,
            "r10_myth": r10_myth,
            "path": path,
        })

    # Rank by JOINT balance ascending (lowest cooperation first)
    candidates.sort(key=lambda c: c["joint"])
    picks = candidates[:5]

    end_minus = []
    for c in picks:
        rel = str(Path(c["path"]).relative_to(REPO_ROOT))
        end_minus.append({
            "source_run": rel,
            "agent_id": c["low_agent"],
            "round": 10,
            "joint_at_source": c["joint"],
            "agent_balance_at_source": c["low_balance"],
            "text": c["r10_myth"],
            "tokens": word_count(c["r10_myth"]),
        })

    manifest["seeds"]["s_end_minus"] = end_minus

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(end_minus)} s_end_minus seeds (from lowest-joint Phase 2 baselines)")
    for s in end_minus:
        print(f"  agent={s['agent_id']:8} joint=${s['joint_at_source']:.0f} (agent bal=${s['agent_balance_at_source']:.1f}) words={s['tokens']}")
    print(f"Mean word count: {statistics.mean(s['tokens'] for s in end_minus):.1f}")


if __name__ == "__main__":
    harvest()
