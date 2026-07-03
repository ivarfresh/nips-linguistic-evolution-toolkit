"""Phase 6b — harvest 5 round-10 myths from GPT-5-nano 8-agent runs.

Source: data/json/gpt5nano_8agent_myth_directive_history3_anon_r10_n5/
        gpt-5-nano/myth_game/  (5 reps at joint $270-$588).

For each rep, pick the highest-balance agent and take their round-10 myth.
Mirrors the Sonnet team-baseline methodology.

Writes into data/phase3/seed_manifest.json under `seeds.s_end_plus_gpt`.
"""

import glob
import json
import re
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_GLOB = str(REPO_ROOT / "data/json/gpt5nano_8agent_myth_directive_history3_anon_r10_n5/gpt-5-nano/myth_game/*.json")
MANIFEST_PATH = REPO_ROOT / "data/phase3/seed_manifest.json"


def word_count(text):
    return len(re.findall(r"\b[a-zA-Z']+\b", text))


def harvest():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    files = sorted(glob.glob(SOURCE_GLOB))
    files = [f for f in files if ".results" not in f and ".checkpoint" not in f]
    print(f"Found {len(files)} GPT-5-nano source runs")

    seeds = []
    for path in files[:5]:
        with open(path) as f:
            d = json.load(f)
        ch = d.get("conversation_history", [])
        bal = ch[-1].get("balances", {}) if ch else {}
        joint = sum(bal.values()) if bal else None
        if not bal:
            continue

        # Highest-balance agent per rep
        target, agent_bal = max(bal.items(), key=lambda kv: kv[1])

        myth_text = None
        for entry in ch:
            if entry.get("round") == 10 and "myths" in entry:
                myth_text = entry["myths"].get(target)
                break

        if not myth_text:
            print(f"  [skip] {path}: no round-10 myth for {target}")
            continue

        rel = str(Path(path).relative_to(REPO_ROOT))
        rm = re.search(r"rep(\d+)", path)
        rep = rm.group(1) if rm else "?"
        seeds.append({
            "source_run": rel,
            "source_model": "openai/gpt-5-nano",
            "agent_id": target,
            "round": 10,
            "joint_at_source": joint,
            "agent_balance_at_source": agent_bal,
            "text": myth_text,
            "tokens": word_count(myth_text),
        })
        print(f"  [ok] rep{rep}: {target} joint=${joint:.0f} agent_bal=${agent_bal:.1f} words={word_count(myth_text)}")

    manifest["seeds"]["s_end_plus_gpt"] = seeds
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(seeds)} s_end_plus_gpt seeds to {MANIFEST_PATH}")
    print(f"Source joint range: ${min(s['joint_at_source'] for s in seeds):.0f}-${max(s['joint_at_source'] for s in seeds):.0f}")
    print(f"Mean word count: {statistics.mean(s['tokens'] for s in seeds):.1f}")


if __name__ == "__main__":
    harvest()
