"""Phase 3 seed harvest.

Walks the 5 reps in sonnet45_8agent_myth_directive_history3_anon_r10_n5,
picks one high-cooperation agent per rep, and writes
data/phase3/seed_manifest.json with two pools:
  - s_start: that agent's round-1 myth
  - s_end_plus: that agent's round-10 myth

Schema matches Phase 2 manifest for drop-in compatibility.
"""

import glob
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "data/json/sonnet45_8agent_myth_directive_history3_anon_r10_n5/claude-sonnet-4.5/myth_game"
OUT_PATH = REPO_ROOT / "data/phase3/seed_manifest.json"


def load_run(path):
    with open(path) as f:
        return json.load(f)


def harvest():
    files = sorted(glob.glob(str(SOURCE_DIR / "*.json")))
    files = [f for f in files if ".checkpoint" not in f and ".results" not in f]

    s_start = []
    s_end_plus = []
    joint_balances = []

    for path in files:
        d = load_run(path)
        ch = d.get("conversation_history", [])
        balances_final = ch[-1].get("balances", {}) if ch else {}
        joint = sum(balances_final.values()) if balances_final else None
        joint_balances.append(joint)

        # Pick the highest-balance agent in this rep so the harvested myths
        # come from someone who actually cooperated successfully.
        if not balances_final:
            raise RuntimeError(f"No final balances in {path}")
        target_agent = max(balances_final, key=balances_final.get)

        r1_myth = None
        r10_myth = None
        for entry in ch:
            if entry.get("round") == 1 and "myths" in entry:
                r1_myth = entry["myths"].get(target_agent)
            if entry.get("round") == 10 and "myths" in entry:
                r10_myth = entry["myths"].get(target_agent)

        if not r1_myth or not r10_myth:
            raise RuntimeError(
                f"Missing myth in {path}: r1={bool(r1_myth)} r10={bool(r10_myth)}"
            )

        rel_source = str(Path(path).relative_to(REPO_ROOT))
        common = {
            "source_run": rel_source,
            "agent_id": target_agent,
            "joint_at_source": joint,
        }
        s_start.append({**common, "round": 1, "text": r1_myth, "tokens": len(r1_myth.split())})
        s_end_plus.append({**common, "round": 10, "text": r10_myth, "tokens": len(r10_myth.split())})

    manifest = {
        "source_dir": str(SOURCE_DIR.relative_to(REPO_ROOT)),
        "num_per_pool": len(s_start),
        "baseline_stats": {
            "joint_balances": joint_balances,
            "n_reps": len(joint_balances),
        },
        "seeds": {
            "s_start": s_start,
            "s_end_plus": s_end_plus,
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {OUT_PATH}")
    print(f"  s_start: {len(s_start)} myths")
    print(f"  s_end_plus: {len(s_end_plus)} myths")
    print(f"  Source joint balances: {joint_balances}")
    for pool in ("s_start", "s_end_plus"):
        for entry in manifest["seeds"][pool]:
            print(f"  [{pool}] {entry['agent_id']:8} round={entry['round']:2} joint={entry['joint_at_source']:.0f} tokens={entry['tokens']}")


if __name__ == "__main__":
    harvest()
