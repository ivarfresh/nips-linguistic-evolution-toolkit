"""Phase 6 — harvest 5 round-10 myths from Gemini Flash Lite 8-agent runs.

Source: data/json/gemini31_flashlite_8agent_myth_directive_history3_r10_n5/
        gemini-3.1-flash-lite/myth_game/  (5 reps, all at $600 ceiling).

For each rep, takes one agent's round-10 myth, rotating across agents so the
5 harvested seeds aren't all from Agent_1.

Writes the pool into data/phase3/seed_manifest.json under `seeds.s_end_plus_gemini`.
"""

import glob
import json
import re
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_GLOB = str(REPO_ROOT / "data/json/gemini31_flashlite_8agent_myth_directive_history3_r10_n5/gemini-3.1-flash-lite/myth_game/*.json")
MANIFEST_PATH = REPO_ROOT / "data/phase3/seed_manifest.json"


def word_count(text):
    return len(re.findall(r"\b[a-zA-Z']+\b", text))


def harvest():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    files = sorted(glob.glob(SOURCE_GLOB))
    files = [f for f in files if ".results" not in f and ".checkpoint" not in f]
    print(f"Found {len(files)} Gemini source runs")

    seeds = []
    # Rotate through agents 1..5 across the 5 reps so the pool isn't all Agent_1
    target_agents = ["Agent_1", "Agent_2", "Agent_3", "Agent_4", "Agent_5"]

    for i, path in enumerate(files[:5]):
        with open(path) as f:
            d = json.load(f)
        ch = d.get("conversation_history", [])
        bal = ch[-1].get("balances", {}) if ch else {}
        joint = sum(bal.values()) if bal else None

        target = target_agents[i]
        agent_balance = bal.get(target)

        # Find round-10 myth from that agent
        myth_text = None
        for entry in ch:
            if entry.get("round") == 10 and "myths" in entry:
                myth_text = entry["myths"].get(target)
                break

        if not myth_text:
            print(f"  [skip] {path}: no round-10 myth for {target}")
            continue

        rel = str(Path(path).relative_to(REPO_ROOT))
        seeds.append({
            "source_run": rel,
            "source_model": "google/gemini-3.1-flash-lite",
            "agent_id": target,
            "round": 10,
            "joint_at_source": joint,
            "agent_balance_at_source": agent_balance,
            "text": myth_text,
            "tokens": word_count(myth_text),
        })
        print(f"  [ok] rep{i}: {target} joint=${joint:.0f} agent_bal=${agent_balance:.1f} words={word_count(myth_text)}")

    manifest["seeds"]["s_end_plus_gemini"] = seeds
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(seeds)} s_end_plus_gemini seeds to {MANIFEST_PATH}")
    print(f"Mean word count: {statistics.mean(s['tokens'] for s in seeds):.1f}")


if __name__ == "__main__":
    harvest()
