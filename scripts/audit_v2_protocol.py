#!/usr/bin/env python3
"""Audit corrected v2 trust-game runs before substantive analysis.

Checks each final JSON for:
- complete rounds, dyads, game responses, myths, and LLM calls
- recovered retry attempts are explicit and do not alter accepted-call counts
- no unrecovered LLM errors or empty accepted responses/messages
- informed-noise notice in every system prompt
- communicated sends and returns within the configured noise bound
- myth-decision link exactly once in each current myth-condition game prompt
- no myth-decision link in game-only current prompts
- chat-memory message counts consistent with the configured interaction capacity
- configured history-policy/window behavior in shared later game prompts
- exact public-ledger identities and noisy records when population-wide
  history is enabled
- fixed-pair partner stability and pairing-aware system instructions
- identical realized schedules across nominally paired seeded runs

Usage:
    python scripts/audit_v2_protocol.py <run_dir_or_json> [...]
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.myth_writer import InvalidMythResponseError, validate_myth_response


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


def classify_interactions(events):
    """Separate accepted interactions from rejected attempts and verify recovery."""
    accepted = []
    retries = []
    unrecovered = []
    for event_index, event in enumerate(events):
        if not event.get("error"):
            accepted.append(event)
            continue

        retries.append(event)
        metadata = event.get("metadata") or {}
        event_key = (metadata.get("round"), metadata.get("task"))
        recovered_later = any(
            not later_event.get("error")
            and (
                (later_event.get("metadata") or {}).get("round"),
                (later_event.get("metadata") or {}).get("task"),
            )
            == event_key
            for later_event in events[event_index + 1 :]
        )
        if not recovered_later:
            unrecovered.append(event)
    return accepted, retries, unrecovered


def audit_paired_schedules(results):
    """Require runs sharing a paired-protocol key to realize one schedule."""
    groups = {}
    for result in results:
        pairing_key = result.get("pairing_key")
        if pairing_key is not None:
            groups.setdefault(pairing_key, []).append(result)

    for pairing_key, group in groups.items():
        if len(group) < 2:
            continue
        signatures = {result["pairing_signature"] for result in group}
        if len(signatures) == 1:
            continue

        pairing_seed, replicate_id, *_ = pairing_key
        filenames = ", ".join(result["path"].name for result in group)
        for result in group:
            result["issues"].append(
                f"{result['path'].name}: realized pairing schedule differs "
                "within nominally paired runs "
                f"(pairing_seed={pairing_seed}, replicate_id={replicate_id}): "
                f"{filenames}"
            )


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
    population_history_window = int(
        metadata.get("population_history_window") or 0
    )
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
    pairing_signature = tuple(
        (
            entry.get("round"),
            tuple(
                (
                    dyad.get("dyad_id"),
                    dyad.get("investor"),
                    dyad.get("trustee"),
                )
                for dyad in (entry.get("dyads") or [])
            ),
        )
        for entry in history
    )
    replicate_id = metadata.get("replicate_id")
    pairing_seed = metadata.get("pairing_seed")
    pairing_key = None
    if pairing_seed is not None and replicate_id is not None:
        pairing_key = (
            pairing_seed,
            replicate_id,
            num_agents,
            num_turns,
            effective_pairing_mode,
            tuple(agents),
        )

    def add(message):
        issues.append(f"{path.name}: {message}")

    agent_ids = list(agents)

    def public_ledger_label(agent_id):
        try:
            index = agent_ids.index(agent_id)
        except ValueError:
            return None
        letters = ""
        value = index + 1
        while value:
            value, remainder = divmod(value - 1, 26)
            letters = chr(ord("A") + remainder) + letters
        return f"Member {letters}"

    history_by_round = {}
    for entry in history:
        try:
            history_by_round[int(entry.get("round"))] = entry
        except (TypeError, ValueError):
            continue

    def expected_public_ledger_lines(round_number):
        previous_rounds = sorted(
            number for number in history_by_round if number < round_number
        )[-population_history_window:]
        lines = []
        for previous_round in previous_rounds:
            for dyad in history_by_round[previous_round].get("dyads") or []:
                investor_id = dyad.get("investor")
                trustee_id = dyad.get("trustee")
                lines.append(
                    f"- Round {previous_round}: {public_ledger_label(investor_id)} "
                    f"(SENDER) sent ${dyad.get('sent_communicated')} to "
                    f"{public_ledger_label(trustee_id)} (RECEIVER); "
                    f"{public_ledger_label(trustee_id)} returned "
                    f"${dyad.get('returned_communicated')}."
                )
        return lines

    def expected_anonymous_record_lines(round_number):
        previous_rounds = sorted(
            number for number in history_by_round if number < round_number
        )[-population_history_window:]
        lines = []
        for previous_round in previous_rounds:
            for pair_index, dyad in enumerate(
                history_by_round[previous_round].get("dyads") or [],
                start=1,
            ):
                lines.append(
                    f"- Round {previous_round}, Pair {pair_index}: a sender sent "
                    f"${dyad.get('sent_communicated')}; the receiver returned "
                    f"${dyad.get('returned_communicated')}."
                )
        return lines

    def round_opponent(agent_id, round_number):
        for dyad in (history_by_round.get(round_number) or {}).get("dyads") or []:
            dyad_agents = dyad.get("agents") or []
            if agent_id in dyad_agents:
                return next(
                    (other_id for other_id in dyad_agents if other_id != agent_id),
                    None,
                )
        return None

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
    if metadata.get("execution_provenance_version"):
        for key in (
            "code_commit",
            "config_sha256",
            "llm_provider",
            "provider_model",
            "max_output_tokens_source",
        ):
            if metadata.get(key) in (None, ""):
                add(f"execution provenance is missing {key}")
        if metadata.get("code_dirty"):
            add("execution provenance reports a dirty worktree")

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
        for agent_id, myth in (entry.get("myths") or {}).items():
            try:
                validate_myth_response(myth)
            except InvalidMythResponseError as exc:
                add(
                    f"{agent_id} round {round_number} myth: "
                    f"invalid accepted myth ({exc})"
                )

        exposures = entry.get("myth_exposures") or {}
        if has_myth and round_number == 1 and exposures:
            add("round 1: unexpected prior-myth exposure records")
        if has_myth and round_number > 1 and (
            defector_myth_policy == "standard_substitute" or exposures
        ):
            if len(exposures) != num_agents:
                add(
                    f"round {round_number}: {len(exposures)} myth exposure "
                    f"records; expected {num_agents}"
                )
            previous_entry = history_by_round.get(round_number - 1) or {}
            previous_myths = previous_entry.get("myths") or {}
            previous_types = (
                previous_entry.get("agent_types")
                or game_data.get("agent_types")
                or {}
            )
            for target_id in agents:
                exposure = exposures.get(target_id) or {}
                original_author_id = round_opponent(target_id, round_number - 1)
                should_substitute = (
                    defector_myth_policy == "standard_substitute"
                    and previous_types.get(target_id, "standard") == "standard"
                    and previous_types.get(original_author_id, "standard")
                    == "defector"
                )
                if exposure.get("policy") != defector_myth_policy:
                    add(
                        f"{target_id} round {round_number}: myth exposure policy "
                        "does not match run metadata"
                    )
                if exposure.get("source_round") != round_number - 1:
                    add(
                        f"{target_id} round {round_number}: myth exposure has "
                        "the wrong source round"
                    )
                if exposure.get("original_author_id") != original_author_id:
                    add(
                        f"{target_id} round {round_number}: original myth author "
                        "is not the prior-round partner"
                    )
                if bool(exposure.get("substitution_applied")) != should_substitute:
                    add(
                        f"{target_id} round {round_number}: myth substitution "
                        f"flag is {exposure.get('substitution_applied')!r}; "
                        f"expected {should_substitute}"
                    )
                presented_author_id = exposure.get("presented_author_id")
                if presented_author_id not in previous_myths:
                    add(
                        f"{target_id} round {round_number}: presented myth author "
                        "has no prior-round myth"
                    )
                if should_substitute:
                    if previous_types.get(presented_author_id, "standard") != "standard":
                        add(
                            f"{target_id} round {round_number}: substituted myth "
                            "was not ordinary-authored"
                        )
                    if presented_author_id in {target_id, original_author_id}:
                        add(
                            f"{target_id} round {round_number}: invalid "
                            "substitute myth author"
                        )
                elif presented_author_id != original_author_id:
                    add(
                        f"{target_id} round {round_number}: normal circulation "
                        "changed the presented myth author"
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

    attempt_count = 0
    call_count = 0
    llm_call_count = 0
    forced_response_count = 0
    retry_count = 0
    expected_calls = num_turns * num_agents * len(task_order)
    for agent_id, agent in agents.items():
        events = agent.get("interaction_history") or []
        accepted_events, retry_events, unrecovered_events = classify_interactions(
            events
        )
        expected_agent_calls = num_turns * len(task_order)
        if len(accepted_events) != expected_agent_calls:
            add(
                f"{agent_id}: {len(accepted_events)} accepted calls; "
                f"expected {expected_agent_calls}"
            )
        retry_count += len(retry_events)
        for event in unrecovered_events:
            event_metadata = event.get("metadata") or {}
            add(
                f"{agent_id} round {event_metadata.get('round')} "
                f"{event_metadata.get('task')}: unrecovered "
                f"{(event.get('error') or {}).get('type')} error"
            )

        accepted_before = 0
        for event in events:
            attempt_count += 1
            is_accepted = not event.get("error")
            if is_accepted:
                call_count += 1
            event_metadata = event.get("metadata") or {}
            round_number = event_metadata.get("round")
            task = event_metadata.get("task")
            tag = f"{agent_id} round {round_number} {task}"
            messages = event.get("messages_sent") or []
            response = event.get("response") or {}
            response_source = response.get("response_source", "llm")
            if is_accepted:
                if response_source == "forced_zero":
                    forced_response_count += 1
                else:
                    llm_call_count += 1

            if is_accepted and not (response.get("content") or "").strip():
                add(f"{tag}: empty response")
            if is_accepted and defector_action_policy == "forced_zero":
                should_be_forced = task == "game" and agent_id in defector_ids
                if should_be_forced and response_source != "forced_zero":
                    add(f"{tag}: forced defector game response came from the LLM")
                if not should_be_forced and response_source == "forced_zero":
                    add(f"{tag}: unexpected forced-zero response")
                if (
                    task == "myth"
                    and agent_id in defector_ids
                    and response_source != "llm"
                ):
                    add(f"{tag}: defector myth did not come from the LLM")
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

            expected_previous = min(accepted_before, memory_capacity)
            expected_messages = 2 + 2 * expected_previous
            if len(messages) != expected_messages:
                add(
                    f"{tag}: {len(messages)} messages sent; "
                    f"expected {expected_messages} for capacity {memory_capacity}"
                )

            prompt = messages[-1].get("content") or ""
            if task == "myth" and round_number > 1:
                exposure = (
                    (history_by_round.get(round_number) or {})
                    .get("myth_exposures", {})
                    .get(agent_id)
                )
                if exposure:
                    source_entry = history_by_round.get(
                        exposure.get("source_round")
                    ) or {}
                    source_myths = source_entry.get("myths") or {}
                    presented_myth = source_myths.get(
                        exposure.get("presented_author_id")
                    )
                    if presented_myth and presented_myth not in prompt:
                        add(f"{tag}: recorded presented myth is absent from prompt")
                    if exposure.get("substitution_applied"):
                        original_myth = source_myths.get(
                            exposure.get("original_author_id")
                        )
                        if (
                            original_myth
                            and original_myth != presented_myth
                            and original_myth in prompt
                        ):
                            add(f"{tag}: quarantined defector myth remains in prompt")
                        if (
                            "standard_substitute" in prompt
                            or "circulation policy" in prompt.lower()
                        ):
                            add(f"{tag}: myth circulation policy leaked into prompt")
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
                if history_policy in {"none", "stable_ids", "relative_pair_ids"}:
                    if history_entries or "most recent previous game" in prompt:
                        add(
                            f"{tag}: history present under "
                            f"history_policy={history_policy}"
                        )
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
                elif history_policy == "population_ledger":
                    expected_entries = (
                        min(round_number - 1, population_history_window)
                        * expected_dyads
                    )
                    if history_entries != expected_entries:
                        add(
                            f"{tag}: public ledger has {history_entries} entries; "
                            f"expected {expected_entries}"
                        )
                    if "Public ledger of communicated/noisy transfers" not in prompt:
                        add(f"{tag}: public population ledger missing")
                elif history_policy == "anonymous_population_record":
                    expected_entries = (
                        min(round_number - 1, population_history_window)
                        * expected_dyads
                    )
                    if history_entries != expected_entries:
                        add(
                            f"{tag}: anonymous population record has "
                            f"{history_entries} entries; expected {expected_entries}"
                        )
                    if (
                        "Anonymous record of communicated/noisy population transfers"
                        not in prompt
                    ):
                        add(f"{tag}: anonymous population record missing")

            if task == "game" and history_policy == "population_ledger":
                if "PUBLIC POPULATION LEDGER:" not in prompt:
                    add(f"{tag}: public-ledger treatment context missing")
                if "stable public ID" not in prompt:
                    add(f"{tag}: stable public IDs missing")
                if "does not reveal hidden true amounts" not in prompt:
                    add(f"{tag}: ledger observability boundary missing")
                own_label = public_ledger_label(agent_id)
                opponent_label = public_ledger_label(
                    round_opponent(agent_id, round_number)
                )
                if f"Your stable public ID is {own_label}." not in prompt:
                    add(f"{tag}: incorrect own public-ledger ID")
                if (
                    "Your current co-player's stable public ID is "
                    f"{opponent_label}."
                ) not in prompt:
                    add(f"{tag}: incorrect current co-player public-ledger ID")
                observed_lines = re.findall(
                    r"^- Round.*$", prompt, flags=re.MULTILINE
                )
                expected_lines = expected_public_ledger_lines(round_number)
                if observed_lines != expected_lines:
                    add(
                        f"{tag}: public ledger does not exactly match the "
                        "communicated transfers in saved prior rounds"
                    )

            if task == "game" and history_policy == "stable_ids":
                if "STABLE POPULATION IDS:" not in prompt:
                    add(f"{tag}: stable-ID treatment context missing")
                own_label = public_ledger_label(agent_id)
                opponent_label = public_ledger_label(
                    round_opponent(agent_id, round_number)
                )
                if f"Your stable population ID is {own_label}." not in prompt:
                    add(f"{tag}: incorrect own stable population ID")
                if (
                    "Your current co-player's stable population ID is "
                    f"{opponent_label}."
                ) not in prompt:
                    add(f"{tag}: incorrect current co-player stable population ID")
                if "No population-wide interaction history is shown" not in prompt:
                    add(f"{tag}: stable-ID no-history boundary missing")

            if task == "game" and history_policy == "anonymous_population_record":
                if "ANONYMOUS POPULATION RECORD:" not in prompt:
                    add(f"{tag}: anonymous-record treatment context missing")
                if "cannot identify your current co-player" not in prompt:
                    add(f"{tag}: anonymous-record identity boundary missing")
                if "does not reveal hidden true amounts" not in prompt:
                    add(f"{tag}: anonymous-record observability boundary missing")
                if re.search(r"\bMember [A-Z]+\b|\bAgent_\d+\b", prompt):
                    add(f"{tag}: stable identity leaked into anonymous record prompt")
                observed_lines = re.findall(
                    r"^- Round.*$", prompt, flags=re.MULTILINE
                )
                expected_lines = expected_anonymous_record_lines(round_number)
                if observed_lines != expected_lines:
                    add(
                        f"{tag}: anonymous population record does not exactly "
                        "match communicated transfers in saved prior rounds"
                    )

            if task == "game" and history_policy == "relative_pair_ids":
                if "ROUND-LOCAL PAIR IDS:" not in prompt:
                    add(f"{tag}: round-local identity context missing")
                if "Your round-local pair ID is Member Self." not in prompt:
                    add(f"{tag}: relative own pair ID missing")
                if (
                    "Your current co-player's round-local pair ID is Member Other."
                    not in prompt
                ):
                    add(f"{tag}: relative co-player pair ID missing")
                if "reassigned every round" not in prompt:
                    add(f"{tag}: round-local reassignment boundary missing")
                if "No population-wide interaction history is shown" not in prompt:
                    add(f"{tag}: round-local no-history boundary missing")
                if re.search(r"\bMember [A-H]\b|\bAgent_\d+\b", prompt):
                    add(f"{tag}: persistent identity leaked into round-local prompt")

            if is_accepted:
                accepted_before += 1

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
        "attempts": attempt_count,
        "llm_calls": llm_call_count,
        "forced_responses": forced_response_count,
        "retry_attempts": retry_count,
        "noise_checks": noise_checks,
        "pairing_key": pairing_key,
        "pairing_signature": pairing_signature,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Run directories or final JSON files")
    args = parser.parse_args()

    paths = final_json_paths(args.paths)
    if not paths:
        parser.error("no final run JSON files found")

    results = [audit_run(path) for path in paths]
    audit_paired_schedules(results)
    for result in results:
        status = "PASS" if not result["issues"] else "FAIL"
        print(
            f"{status} {result['path']}: "
            f"{result['rounds']} rounds, {result['dyads']} dyads, "
            f"{result['calls']} accepted interactions "
            f"({result['llm_calls']} LLM, {result['forced_responses']} forced), "
            f"{result['retry_attempts']} recovered retries, "
            f"{result['noise_checks']} noise checks"
        )

    issues = [issue for result in results for issue in result["issues"]]
    print(
        f"\nAudited {len(results)} run(s): "
        f"{sum(result['calls'] for result in results)} interactions, "
        f"{sum(result['attempts'] for result in results)} total attempts, "
        f"{sum(result['llm_calls'] for result in results)} LLM calls, "
        f"{sum(result['forced_responses'] for result in results)} forced responses, "
        f"{sum(result['retry_attempts'] for result in results)} recovered retries, "
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
