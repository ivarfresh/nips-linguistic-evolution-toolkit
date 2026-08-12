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
- configured history-policy/window behavior in shared later game prompts
- fixed-pair partner stability and pairing-aware system instructions

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
    history_policy = metadata.get("history_policy", "minimal")
    self_history_window = int(metadata.get("self_history_window") or 0)
    coplayer_history_window = int(metadata.get("coplayer_history_window") or 0)
    pairing_mode = metadata.get("pairing_mode", "balanced")
    effective_pairing_mode = metadata.get(
        "effective_pairing_mode",
        "fixed" if num_agents == 2 else pairing_mode,
    )
    prompt_regime = metadata.get("prompt_regime", "legacy")
    show_agent_names = bool(metadata.get("show_agent_names", True))
    defector_action_policy = metadata.get(
        "defector_action_policy",
        "prompted",
    )
    defector_myth_policy = metadata.get("defector_myth_policy", "normal")
    defector_role_visible = bool(
        metadata.get("defector_role_visible_to_self", True)
    )
    defector_ids = set(
        metadata.get("defector_agent_ids")
        or game_data.get("defector_agent_ids")
        or []
    )
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
    if prompt_regime == "unified":
        if metadata.get("noise_seed") is None:
            add("unified protocol is missing its recorded noise_seed")
        if metadata.get("pairing_seed") is None:
            add("unified protocol is missing its recorded pairing_seed")

    dyad_count = 0
    noise_checks = 0
    partners_by_agent = {agent_id: set() for agent_id in agents}
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
            dyad_agents = dyad.get("agents") or []
            if len(dyad_agents) == 2:
                first, second = dyad_agents
                partners_by_agent.setdefault(first, set()).add(second)
                partners_by_agent.setdefault(second, set()).add(first)
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
            if defector_action_policy == "forced_zero":
                if dyad.get("investor") in defector_ids and dyad.get("sent") != 0:
                    add(
                        f"round {round_number} {dyad.get('dyad_id')}: "
                        "forced-zero sender made a nonzero actual transfer"
                    )
                if dyad.get("trustee") in defector_ids and dyad.get("returned") != 0:
                    add(
                        f"round {round_number} {dyad.get('dyad_id')}: "
                        "forced-zero receiver made a nonzero actual return"
                    )

    call_count = 0
    llm_call_count = 0
    forced_response_count = 0
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
            response = event.get("response") or {}
            response_source = response.get("response_source", "llm")
            if response_source == "forced_zero":
                forced_response_count += 1
            else:
                llm_call_count += 1

            if event.get("error"):
                add(f"{tag}: recorded {event['error'].get('type')} error")
            if not (response.get("content") or "").strip():
                add(f"{tag}: empty response")
            if defector_action_policy == "forced_zero":
                should_be_forced = task == "game" and agent_id in defector_ids
                if should_be_forced and response_source != "forced_zero":
                    add(f"{tag}: forced defector game response came from the LLM")
                if not should_be_forced and response_source == "forced_zero":
                    add(f"{tag}: unexpected forced-zero response")
                if (
                    task == "myth"
                    and agent_id in defector_ids
                    and defector_myth_policy == "normal"
                    and response_source != "llm"
                ):
                    add(f"{tag}: normal defector myth did not come from the LLM")
            if not messages or messages[0].get("role") != "system":
                add(f"{tag}: system message is not first")
                continue
            if "communication noise" not in (messages[0].get("content") or ""):
                add(f"{tag}: informed-noise notice missing")
            system_prompt = messages[0].get("content") or ""
            if prompt_regime == "unified" or num_agents > 2:
                if effective_pairing_mode == "fixed":
                    if "same opponent throughout the run" not in system_prompt:
                        add(f"{tag}: fixed-pair system instruction missing")
                    if "Pairings are randomized" in system_prompt:
                        add(f"{tag}: fixed-pair system prompt claims random pairing")
                if show_agent_names:
                    if "told your opponent's name" not in system_prompt:
                        add(f"{tag}: named-agent system instruction missing")
                elif "Agent names are hidden" not in system_prompt:
                    add(f"{tag}: hidden-name system instruction missing")
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
            if (
                task == "game"
                and agent_id in defector_ids
                and not defector_role_visible
                and "DESIGNATED DEFECTOR" in prompt
            ):
                add(f"{tag}: hidden defector treatment leaked into the prompt")
            link_count = prompt.count(MYTH_DECISION_LINK)
            expected_link_count = 1 if task == "game" and has_myth else 0
            if link_count != expected_link_count:
                add(
                    f"{tag}: current prompt has myth-decision link "
                    f"{link_count} time(s); expected {expected_link_count}"
                )

            uses_shared_prompt = num_agents > 2 or prompt_regime == "unified"
            if task == "game" and uses_shared_prompt and round_number > 1:
                history_entries = len(
                    re.findall(r"^- Round", prompt, flags=re.MULTILINE)
                )
                if history_policy == "none":
                    if history_entries or "most recent previous game" in prompt:
                        add(f"{tag}: history present under history_policy=none")
                elif history_policy == "minimal":
                    if "most recent previous game" not in prompt:
                        add(f"{tag}: minimal own-history recap missing")
                elif history_policy == "self_and_coplayer":
                    expected_entries = min(round_number - 1, self_history_window)
                    expected_entries += min(
                        round_number - 1, coplayer_history_window
                    )
                    if history_entries != expected_entries:
                        add(
                            f"{tag}: history block has {history_entries} entries; "
                            f"expected {expected_entries}"
                        )
                    if self_history_window and "Your last" not in prompt:
                        add(f"{tag}: configured self-history block missing")
                    if (
                        coplayer_history_window
                        and "game(s):" not in prompt
                    ):
                        add(f"{tag}: configured co-player history block missing")

    if call_count != expected_calls:
        add(f"{call_count} calls; expected {expected_calls}")
    if defector_action_policy == "forced_zero":
        expected_forced = num_turns * len(defector_ids)
        if forced_response_count != expected_forced:
            add(
                f"{forced_response_count} forced-zero responses; "
                f"expected {expected_forced}"
            )
        expected_llm_calls = expected_calls - expected_forced
        if llm_call_count != expected_llm_calls:
            add(
                f"{llm_call_count} LLM calls; expected {expected_llm_calls}"
            )

    if effective_pairing_mode == "fixed":
        for agent_id, partners in partners_by_agent.items():
            if len(partners) != 1:
                add(
                    f"{agent_id}: fixed pairing produced {len(partners)} partners; "
                    "expected 1"
                )

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
        "llm_calls": llm_call_count,
        "forced_responses": forced_response_count,
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
            f"{result['calls']} interactions "
            f"({result['llm_calls']} LLM, {result['forced_responses']} forced), "
            f"{result['noise_checks']} noise checks"
        )

    issues = [issue for result in results for issue in result["issues"]]
    print(
        f"\nAudited {len(results)} run(s): "
        f"{sum(result['calls'] for result in results)} interactions, "
        f"{sum(result['llm_calls'] for result in results)} LLM calls, "
        f"{sum(result['forced_responses'] for result in results)} forced responses, "
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
