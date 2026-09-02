"""Register the verified gowith translations as pool `s_end_plus_gowith` in the
Phase 3 seed manifest (same pattern as phase6_harvest_gemini.py).

Source: data/phase7/gowith_seeds_candidate.json (built by phase7_gowith_translate.py;
5/5 passed mechanical number checks + injection probe — see researchlog 2026-07-02).

Usage: python scripts/phase7_register_gowith_seeds.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = REPO_ROOT / "data/phase7/gowith_seeds_candidate.json"
MANIFEST = REPO_ROOT / "data/phase3/seed_manifest.json"
POOL = "s_end_plus_gowith"


def main():
    candidates = json.loads(CANDIDATES.read_text())
    manifest = json.loads(MANIFEST.read_text())

    entries = []
    for seed in candidates["seeds"]:
        src = seed.get("source_meta") or {}
        entries.append({
            "source_run": src.get("source_run"),
            "source_pool": seed["source_pool"],
            "source_index": seed["source_index"],
            "agent_id": src.get("agent_id"),
            "round": src.get("round"),
            "joint_at_source": src.get("joint_at_source"),
            "text": seed["gowith_text"],
            "tokens": len(seed["gowith_text"].split()),
            "translator_model": seed.get("translator_model"),
            "transform": "gowith",
            "mechanical_check_pass": seed["mechanical_check"]["pass"],
            "readback_match": seed.get("readback_match"),
        })

    if POOL in manifest["seeds"]:
        print(f"pool {POOL} already present ({len(manifest['seeds'][POOL])} seeds) — overwriting")
    manifest["seeds"][POOL] = entries
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"registered {len(entries)} seeds as {POOL}; manifest now has {len(manifest['seeds'])} pools")


if __name__ == "__main__":
    main()
