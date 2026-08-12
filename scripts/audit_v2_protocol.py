#!/usr/bin/env python3
"""Audit corrected v2 trust-game runs before substantive analysis.

Checks each final JSON for:
- complete rounds, dyads, game responses, myths, and LLM calls
- no recorded LLM errors or empty responses/messages
- informed-noise notice in every system prompt
- communicated sends and returns within the configured noise bound
- myth-decision link exactly once in each current myth-condition game prompt
- no myth-decision link in game-only current prompts
- chat-memory message counts consistent with the configured interaction capacity
- the established 8-agent co-player reputation block in later game prompts

Usage:
    python scripts/audit_v2_protocol.py <run_dir_or_json> [...]
"""

import argparse
import json
import math
import re
from pathlib import Path


MYTH_DECISION_LINK = (
    "Take any myths written in this session into account when making your decision."
)
IGNORED_SUFFIXES = (".results.json", ".checkpoint.json", ".error.json")


def final_json_paths(inputs):
    paths = []
    for raw_path in inputs:
        path = Path(raw_path)
        candidates = [path] if path.is_file() else path.rglob("*.json")
        for candidate in candidates:
            if candidate.name.endswith(IGNORED_SUFFIXES):
                continue
            paths.append(candidate)
    return sorted(set(paths))


def audit_run(path):
    with path.open() as handle:
        run = json.load(handle)

    issues = []
    metadata = run.get("run_metadata") or {}
    task_order = run.get("task_order") or []
    history = run.get("conversation_history") or []
    agents = run.get("agents") or {}
    game_data = run.get("game_data") or {}
    num_turns = int(metadata.get("num_turns") or 0)
    num_agents = int(metadata.get("num_agents") or len(agents))
    memory_capacity = int(metadata.get("memory_capacity") or 0)
    expected_dyads = num_agents // 2
    noise_config = metadata.get("noise_config") or {}
    noise_range = float(noise_config.get("range") or 0)
    has_myth = "myth" in task_order
    expected_addition_id = "myth_decision_link" if has_myth else None

    def add(message):
        issues.append(f"{path.name}: {message}")

    if len(history) != num_turns:
        add(f"{len(history)} rounds; expected {num_turns}")
    if len(agents) != num_agents:
        add(f"{len(agents)} agents; expected {num_agents}")
    if metadata.get("game_prompt_addition_id") != expected_addition_id:
        add(
            "game_prompt_addition_id "
            f"{metadata.get('game_prompt_addition_id')!r}; "
            f"expected {expected_addition_id!r}"
        )
    expected_addition = MYTH_DECISION_LINK if has_myth else ""
    if metadata.get("game_prompt_addition", "") != expected_addition:
        add("stored game_prompt_addition does not match the task order")

    dyad_count = 0
    noise_checks = 0
    for entry in history:
        round_number = entry.get("round")
        dyads = entry.get("dyads") or []
        dyad_count += len(dyads)
        if len(dyads) != expected_dyads:
            add(
                f"round {round_number}: {len(dyads)} dyads; "
                f"expected {expected_dyads}"
            )
        if len(entry.get("game_responses") or {}) != num_agents:
            add(
                f"round {round_number}: incomplete game responses "
                f"({len(entry.get('game_responses') or {})}/{num_agents})"
            )
        myth_count = len(entry.get("myths") or {})
        expected_myths = num_agents if has_myth else 0
        if myth_count != expected_myths:
            add(
                f"round {round_number}: {myth_count} myths; "
                f"expected {expected_myths}"
            )

        for dyad in dyads:
            for actual_key, communicated_key in (
                ("sent", "sent_communicated"),
                ("returned", "returned_communicated"),
            ):
                actual = dyad.get(actual_key)
                communicated = dyad.get(communicated_key)
                if actual is None or communicated is None:
                    add(
                        f"round {round_number} {dyad.get('dyad_id')}: "
                        f"missing {actual_key} noise values"
                    )
                    continue
                noise_checks += 1
                if abs(float(communicated) - float(actual)) > noise_range + 1e-9:
                    add(
                        f"round {round_number} {dyad.get('dyad_id')}: "
                        f"{communicated_key}={communicated} is outside "
                        f"{actual_key}={actual} +/- {noise_range}"
                    )

    call_count = 0
    expected_calls = num_turns * num_agents * len(task_order)
    for agent_id, agent in agents.items():
        events = agent.get("interaction_history") or []
        expected_agent_calls = num_turns * len(task_order)
        if len(events) != expected_agent_calls:
            add(
                f"{agent_id}: {len(events)} calls; expected {expected_agent_calls}"
            )

        for event_index, event in enumerate(events, start=1):
            call_count += 1
            event_metadata = event.get("metadata") or {}
            round_number = event_metadata.get("round")
            task = event_metadata.get("task")
            tag = f"{agent_id} round {round_number} {task}"
            messages = event.get("messages_sent") or []

            if event.get("error"):
                add(f"{tag}: recorded {event['error'].get('type')} error")
            if not ((event.get("response") or {}).get("content") or "").strip():
                add(f"{tag}: empty response")
            if not messages or messages[0].get("role") != "system":
                add(f"{tag}: system message is not first")
                continue
            if "communication noise" not in (messages[0].get("content") or ""):
                add(f"{tag}: informed-noise notice missing")
            for message_index, message in enumerate(messages):
                if not (message.get("content") or "").strip():
                    add(f"{tag}: empty message {message_index}")

            expected_previous = min(event_index - 1, memory_capacity)
            expected_messages = 2 + 2 * expected_previous
            if len(messages) != expected_messages:
                add(
                    f"{tag}: {len(messages)} messages sent; "
                    f"expected {expected_messages} for capacity {memory_capacity}"
                )

            prompt = messages[-1].get("content") or ""
            link_count = prompt.count(MYTH_DECISION_LINK)
            expected_link_count = 1 if task == "game" and has_myth else 0
            if link_count != expected_link_count:
                add(
                    f"{tag}: current prompt has myth-decision link "
                    f"{link_count} time(s); expected {expected_link_count}"
                )

            if task == "game" and num_agents > 2 and round_number > 1:
                history_entries = len(
                    re.findall(r"^- Round", prompt, flags=re.MULTILINE)
                )
                expected_entries = min(round_number - 1, 3)
                if "last 3 game(s)" not in prompt:
                    add(f"{tag}: co-player reputation block missing")
                elif history_entries != expected_entries:
                    add(
                        f"{tag}: co-player block has {history_entries} entries; "
                        f"expected {expected_entries}"
                    )

    if call_count != expected_calls:
        add(f"{call_count} calls; expected {expected_calls}")

    final_balances = game_data.get("balances") or {}
    communicated_balances = game_data.get("balances_communicated") or {}
    if len(final_balances) != num_agents:
        add(f"{len(final_balances)} final balances; expected {num_agents}")
    if len(communicated_balances) != num_agents:
        add(
            f"{len(communicated_balances)} communicated balances; "
            f"expected {num_agents}"
        )
    for label, balances in (
        ("balances", final_balances),
        ("communicated balances", communicated_balances),
    ):
        if any(not math.isfinite(float(value)) for value in balances.values()):
            add(f"non-finite value in {label}")

    return {
        "path": path,
        "issues": issues,
        "rounds": len(history),
        "dyads": dyad_count,
        "calls": call_count,
        "noise_checks": noise_checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Run directories or final JSON files")
    args = parser.parse_args()

    paths = final_json_paths(args.paths)
    if not paths:
        parser.error("no final run JSON files found")

    results = [audit_run(path) for path in paths]
    for result in results:
        status = "PASS" if not result["issues"] else "FAIL"
        print(
            f"{status} {result['path']}: "
            f"{result['rounds']} rounds, {result['dyads']} dyads, "
            f"{result['calls']} calls, {result['noise_checks']} noise checks"
        )

    issues = [issue for result in results for issue in result["issues"]]
    print(
        f"\nAudited {len(results)} run(s): "
        f"{sum(result['calls'] for result in results)} calls, "
        f"{sum(result['noise_checks'] for result in results)} noise checks."
    )
    if issues:
        print(f"{len(issues)} ISSUE(S):")
        for issue in issues:
            print(f" !! {issue}")
        raise SystemExit(1)
    print("NO ISSUES — corrected v2 protocol checks passed.")


if __name__ == "__main__":
    main()
