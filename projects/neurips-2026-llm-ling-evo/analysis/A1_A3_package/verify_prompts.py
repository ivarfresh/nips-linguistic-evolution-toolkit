#!/usr/bin/env python3
"""Smoke test: load an A1 run JSON and pretty-print the actual game
prompts that were sent to the agent, so we can visually confirm the
partner-myth-injection took effect.

Use: after a tiny pilot run (num_runs=1) of any partner-myth experiment_set,
point this at one of the produced JSONs and inspect the prompts.

    python3 verify_prompts.py PATH/TO/run.json

Looks for:
  - The {other_agent_last_myth} placeholder being filled (partner myth text appears)
  - The placeholder gracefully empty in round 1 if no prior partner myth
  - The reasoning-prose addendum in the system prompt for A3 runs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def find_a_run(default_root: Path) -> Path | None:
    """If no path provided, find the newest A1 or A3 JSON."""
    candidates = []
    for sub in (
        "v4_direct_provider_A1_partner_myth",
        "v4_direct_provider_A3_forced_reasoning",
        "v4_direct_provider_A1A3_combined",
    ):
        d = default_root / sub
        if d.exists():
            for p in d.rglob("*.json"):
                if ".checkpoint" in p.name or ".results" in p.name or ".error" in p.name:
                    continue
                candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    if len(sys.argv) >= 2:
        path = Path(sys.argv[1])
    else:
        repo_root = Path(__file__).resolve().parents[4]
        path = find_a_run(repo_root / "data" / "json" / "noise_experiments")
        if path is None:
            print("No A1/A3 run found. Run a pilot first or pass a JSON path.")
            sys.exit(1)
        print(f"Using newest A1/A3 run: {path}\n")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("run_metadata", {})
    print("=" * 72)
    print("RUN METADATA")
    print("=" * 72)
    for k in ("model", "system_prompt_template", "round_prompt_templates",
              "myth_topic_id", "noise_config"):
        print(f"  {k}: {meta.get(k)}")
    print()

    # Print system prompt (from agents)
    agents = data.get("agents", {})
    for agent_id, ag in agents.items():
        msgs = ag.get("messages", [])
        if msgs and msgs[0].get("role") == "system":
            print("=" * 72)
            print(f"SYSTEM PROMPT — {agent_id}")
            print("=" * 72)
            print(msgs[0]["content"])
            print()
            break  # both agents have ~the same system prompt

    # Walk through the conversation history and pull out game prompts
    history = data.get("conversation_history", [])
    for entry in history[:3]:  # first 3 rounds is enough to verify
        r = entry.get("round")
        myths = entry.get("myths") or {}
        responses = entry.get("game_responses") or {}
        for ag, resp in responses.items():
            print("=" * 72)
            print(f"ROUND {r} — game prompt sent to {ag}")
            print("=" * 72)
            # Look for the prompt in the agent's message stream:
            ag_msgs = agents.get(ag, {}).get("messages", [])
            # Find the user message that prompted this game response
            game_content = (resp.get("content") or "")[:80].replace("\n", " ")
            for i, m in enumerate(ag_msgs):
                if (m.get("role") == "assistant"
                        and m.get("content", "").startswith(game_content[:40])):
                    if i > 0 and ag_msgs[i - 1].get("role") == "user":
                        print(ag_msgs[i - 1]["content"])
                        break
            print()
            print(f"-- {ag} response (first 400 chars):")
            print((resp.get("content") or "")[:400])
            print()
        if myths:
            print(f"-- ROUND {r} myths:")
            for ag, myth_text in myths.items():
                preview = (myth_text or "")[:200].replace("\n", " ")
                print(f"   {ag}: {preview}...")
            print()

    # Verification checks
    print("=" * 72)
    print("CHECKS")
    print("=" * 72)
    sys_prompt = ""
    for agent_id, ag in agents.items():
        msgs = ag.get("messages", [])
        if msgs and msgs[0].get("role") == "system":
            sys_prompt = msgs[0]["content"]
            break
    if "REASONING REQUIREMENT" in sys_prompt:
        print("  [A3] Forced-reasoning system prompt detected: YES")
    else:
        print("  [A3] Forced-reasoning system prompt detected: NO")
    # Check whether any round-2+ game prompt contains a partner myth fragment
    a1_hits = 0
    for entry in history[1:]:  # round 2 onward
        for ag, resp in (entry.get("game_responses") or {}).items():
            ag_msgs = agents.get(ag, {}).get("messages", [])
            for m in ag_msgs:
                if m.get("role") == "user":
                    body = m.get("content", "")
                    if "Your partner's most recent story" in body or "Your partner" in body:
                        a1_hits += 1
                        break
    print(f"  [A1] Partner-myth header found in {a1_hits} game-prompt user messages")
    print()
    if a1_hits == 0:
        print("  ⚠ A1 expected at least 1 hit. Check that the round_prompt_templates")
        print("    in the experiment_set point to *_with_partner_myth and that the")
        print("    patch in trust_game_noisy.py is applied.")


if __name__ == "__main__":
    main()
