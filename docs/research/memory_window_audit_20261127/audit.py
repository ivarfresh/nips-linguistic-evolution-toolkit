#!/usr/bin/env python3
"""Offline audit of one existing, hash-pinned research final. Standard library only.

No experiment imports, simulations, model clients, network access, or input writes.
Saved contexts are measured directly; the latest-six expectation is compared with
them, never substituted for them. Provider-format lists are labelled projections.
"""
import argparse
import collections
import csv
import hashlib
import html
import json
from pathlib import Path
import subprocess


SOURCE = Path(
    "data/json/noise_experiments/negative_only_crossmodel_reasoning_rerun_20260909/"
    "negative_only_reasoning_rerun_dyad_myth_game_claude_n5/claude-sonnet-4.5/"
    "myth_game/noisy2_crossmodel_negative_twotask_r3/"
    "negative_only_reasoning_rerun_dyad_myth_game_claude_n5_000_neutral_rep00_"
    "memtest_memory_primary_anything.json"
)
EXPECTED_SHA = "4b92a43771e3716308c77957a885cabe996a8bea0a0aecf65af29205c12885e6"
FOCAL = "Agent_1"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")


def assistant(event):
    response = event["response"]
    return {"role": "assistant", "content": response["content"],
            "reasoning": response.get("reasoning"), "usage": response.get("usage")}


def pair(event):
    return [{"role": "user", "content": event["prompt"]}, assistant(event)]


def provider_projection(messages):
    """Exact role/content projection in hash-verified historical utils.py:322.

    This derives input fields; it is NOT a captured HTTP request or a new API call.
    """
    system = []
    chat = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if content is None:
            continue
        if role == "system":
            system.append(content)
        elif role in {"user", "assistant"}:
            chat.append({"role": role, "content": content})
    return {"system": "\n\n".join(system) if system else None, "messages": chat}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    source = root / SOURCE
    raw = source.read_bytes()
    assert digest(raw) == EXPECTED_SHA, "Source changed; stop instead of auditing a different run."
    data = json.loads(raw)
    metadata = data["run_metadata"]
    condition = metadata["experiment_condition"]
    commit = metadata["code_commit"]
    agents = data["agents"]
    capacity = metadata["memory_capacity"]
    assert metadata["chat_memory_mode"] == "memory_primary" and capacity == 6
    assert len(agents) == 2 and metadata["num_turns"] == 10
    assert data["task_order"] == ["myth", "game"]
    assert metadata["code_dirty"] is False
    assert metadata["defector_count"] == 0
    assert metadata["random_defection_probability"] == 0
    assert metadata["seed_myth"] is None and metadata["seed_reinject"] is False
    assert [r["round"] for r in data["conversation_history"]] == list(range(1, 11))
    assert not any(s in source.name for s in (".results.", ".checkpoint.", ".error."))
    assert out != source.parent and not out.is_relative_to(root / "data")
    out.mkdir(parents=True, exist_ok=True)

    def write_json(name, obj):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Export the exact historical files and verify every recorded hash. No checkout.
    source_checks = []
    hashes = dict(condition["implementation"])
    hashes["config/experiments_noisy.yaml"] = metadata["config_sha256"]
    for relative, expected in hashes.items():
        content = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=root)
        actual = digest(content)
        assert actual == expected, (relative, actual, expected)
        target = out / "source_at_run" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        source_checks.append({"path": relative, "commit": commit, "sha256": actual,
                              "recorded_sha256_matches": actual == expected})

    provenance_rows = []
    for relative in (
        "docs/figures/negative_only_crossmodel_reasoning_rerun_20260909/run_manifest.csv",
        "docs/research/statistics_audit_20261123/section44/september_finals_verified.csv",
    ):
        path = root / relative
        matches = []
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                if any(SOURCE.name in str(value) and str(SOURCE.parent) in str(value)
                       for value in row.values()):
                    matches.append({"end_line": reader.line_num, "row": row})
        assert matches, f"Run missing from research provenance: {relative}"
        provenance_rows.append({"path": relative, "sha256": digest(path.read_bytes()), "matches": matches})
    # This independent, earlier audit must contain the same file hash.
    assert EXPECTED_SHA in json.dumps(provenance_rows[1])

    checks = []

    def check(name, passed, **evidence):
        checks.append({"claim": name, "passed": bool(passed), **evidence})

    myth_texts = {
        agent_id: {event["metadata"]["round"]: event["response"]["content"]
                   for event in agent["interaction_history"] if event["metadata"]["task"] == "myth"}
        for agent_id, agent in agents.items()
    }
    later_template = condition["protocol"]["myth"]["later_rounds_template"]
    all_rows, message_changes, enriched_calls, buffer_checks = {}, {}, {}, []
    response_ids = []
    for agent_id, agent in agents.items():
        events = agent["interaction_history"]
        partner_id = next(other for other in agents if other != agent_id)
        check("20 recorded interactions", len(events) == 20, agent=agent_id)
        check("agent capacity matches run", agent["memory_capacity"] == capacity, agent=agent_id)
        round_counts = collections.defaultdict(list)
        system = events[0]["messages_sent"][0]
        check("first message is system", system["role"] == "system", agent=agent_id)
        previous_ids = []
        previous_map = {}
        rows, changes, calls = [], [], []
        for index, event in enumerate(events):
            number = index + 1
            location = f"agents.{agent_id}.interaction_history[{index}]"
            rnd, task = event["metadata"]["round"], event["metadata"]["task"]
            round_counts[rnd].append(task)
            messages = event["messages_sent"]
            history = messages[1:-1]
            usage = event["response"]["usage"]
            response_ids.append(usage.get("response_id"))
            check("recorded successful LLM interaction", "error" not in event and
                  event["response"]["response_source"] == "llm" and
                  usage["finish_reason"] == "end_turn" and usage["outcome"] == "complete",
                  location=location)
            check("system unchanged", messages[0] == system and
                  sum(m["role"] == "system" for m in messages) == 1, location=location)
            check("current prompt recorded exactly", messages[-1] == {"role": "user", "content": event["prompt"]}, location=location)
            check("complete alternating pairs", len(history) % 2 == 0 and
                  [m["role"] for m in history] == ["user", "assistant"] * (len(history) // 2), location=location)
            check("at most six prior pairs", len(history) // 2 <= capacity, location=location, observed=len(history) // 2)
            check("correct event index", event["interaction_index"] == number, location=location)
            check("request plan equals run metadata", usage["request_settings"] == metadata["llm_request"], location=location)
            retained_indices = []
            for pos in range(0, len(history), 2):
                observed_pair = history[pos:pos + 2]
                candidates = [j for j in range(index) if pair(events[j]) == observed_pair]
                check("pair identifies one unchanged original exchange", len(candidates) == 1,
                      location=location, message_index=pos + 1, matches=[j + 1 for j in candidates])
                if len(candidates) != 1:
                    raise AssertionError(f"Cannot label ambiguous/modified pair at {location}[{pos + 1}]")
                retained_indices.append(candidates[0])
            expected_indices = list(range(max(0, index - capacity), index))
            check("retained pairs equal chronological latest-six suffix", retained_indices == expected_indices,
                  location=location, observed=[j + 1 for j in retained_indices], expected=[j + 1 for j in expected_indices])

            ids = ["system"]
            for j in retained_indices:
                ids.extend([f"C{j + 1}.user", f"C{j + 1}.assistant"])
            ids.append(f"C{number}.user")
            mapping = dict(zip(ids, messages))
            removed = [key for key in previous_ids if key not in ids]
            entered = [key for key in ids if key not in previous_ids]
            unchanged = [key for key in ids if key in previous_ids]
            check("overlapping messages unchanged", all(mapping[key] == previous_map[key] for key in unchanged), location=location)
            expected_removed = ([f"C{number - capacity - 1}.user", f"C{number - capacity - 1}.assistant"]
                                if index > capacity else [])
            check("FIFO eviction removes exactly oldest whole pair", removed == expected_removed,
                  location=location, removed=removed, expected=expected_removed)

            occurrences = []
            own_in_prompt = []
            for source_round, text in myth_texts[partner_id].items():
                for message_index, message in enumerate(messages):
                    if message["role"] == "user" and text in message["content"]:
                        occurrences.append({"partner": partner_id, "source_round": source_round,
                                            "message_index": message_index, "message_id": ids[message_index],
                                            "current_prompt": message_index == len(messages) - 1,
                                            "content_offset": message["content"].index(text),
                                            "myth_sha256": digest(text.encode())})
            for source_round, text in myth_texts[agent_id].items():
                if text in event["prompt"]:
                    own_in_prompt.append(source_round)
            historical_partner = sorted({o["source_round"] for o in occurrences if not o["current_prompt"]})
            current_partner = sorted({o["source_round"] for o in occurrences if o["current_prompt"]})
            check("no own full myth in new prompt", not own_in_prompt, location=location)
            if task == "myth":
                check("myth output matches round record", data["conversation_history"][rnd - 1]["myths"][agent_id] == event["response"]["content"], location=location)
                if rnd > 1:
                    expected_prompt = later_template.format(other_agent_myth=myth_texts[partner_id][rnd - 1])
                    check("new myth prompt equals scaffold plus only previous-round partner", event["prompt"] == expected_prompt and current_partner == [rnd - 1], location=location)
                else:
                    check("first myth prompt contains no partner myth", not current_partner, location=location)
            else:
                check("new game prompt contains no whole myth", not current_partner and not own_in_prompt, location=location)

            game_rounds = [events[j]["metadata"]["round"] for j in retained_indices if events[j]["metadata"]["task"] == "game"]
            own_rounds = [events[j]["metadata"]["round"] for j in retained_indices if events[j]["metadata"]["task"] == "myth"]
            row = {"call": number, "round": rnd, "task": task, "total_messages_including_system": len(messages),
                   "provider_chat_messages_excluding_system": len(provider_projection(messages)["messages"]),
                   "retained_previous_exchanges": len(retained_indices), "retained_calls": [j + 1 for j in retained_indices],
                   "previous_game_rounds": game_rounds, "own_myth_rounds": own_rounds,
                   "own_myth_count": len(own_rounds), "partner_myth_rounds_in_historical_prompts": historical_partner,
                   "partner_myth_rounds_in_current_prompt": current_partner,
                   "partner_myth_count": len(set(historical_partner + current_partner)),
                   "removed_messages": removed, "entered_messages": entered,
                   "partner_myth_occurrences": occurrences, "json_location": location + ".messages_sent"}
            rows.append(row)
            changes.append({"call": number, "removed": [{"id": key, "message": previous_map[key]} for key in removed],
                            "entered": [{"id": key, "message": mapping[key]} for key in entered], "preserved": unchanged})
            calls.append({"call": number, "round": rnd, "task": task, "message_ids": ids,
                          "status_since_previous_input": ["entered" if key in entered else "preserved" for key in ids],
                          "messages_sent": messages})
            previous_ids, previous_map = ids, mapping
        check("actual two calls per round: myth then game", dict(round_counts) == {r: ["myth", "game"] for r in range(1, 11)}, agent=agent_id)
        final_messages = agent["messages"]
        buffer_checks.append({"agent": agent_id, "messages": len(final_messages),
                              "complete_exchanges": (len(final_messages) - 1) // 2,
                              "equals_last_input_plus_final_response": final_messages == events[-1]["messages_sent"] + [assistant(events[-1])],
                              "strict_stored_buffer_at_most_six": (len(final_messages) - 1) // 2 <= capacity})
        all_rows[agent_id], message_changes[agent_id], enriched_calls[agent_id] = rows, changes, calls
    check("40 distinct saved response IDs", len(response_ids) == 40 and len(set(response_ids)) == 40 and None not in response_ids)
    failures = [c for c in checks if not c["passed"]]

    focal_rows = all_rows[FOCAL]
    first_myth_call = next(r["call"] for r in focal_rows if r["task"] == "myth")
    eviction_call = next(r["call"] for r in focal_rows if f"C{first_myth_call}.assistant" in r["removed_messages"])
    selected_numbers = [eviction_call - 1, eviction_call, eviction_call + 1]
    selected = [call for call in enriched_calls[FOCAL] if call["call"] in selected_numbers]
    check("selected calls bracket first own-myth eviction", selected_numbers == [7, 8, 9])

    phrase = "Valley of Echoing Waters"
    final_myth_event = agents[FOCAL]["interaction_history"][18]
    original_m2 = myth_texts[FOCAL][2]
    lineage = {"phrase": phrase, "own_myth_rounds_containing_phrase": [r for r, text in myth_texts[FOCAL].items() if phrase in text],
               "partner_myth_rounds_containing_phrase": [r for r, text in myth_texts["Agent_2"].items() if phrase in text],
               "original_myth_2_absent_from_call_19": all(original_m2 not in m["content"] for m in final_myth_event["messages_sent"]),
               "call_19_visible_phrase_messages": [{"message_index": i, "role": m["role"], "content": m["content"]}
                                                   for i, m in enumerate(final_myth_event["messages_sent"]) if phrase in m["content"]],
               "original_myth_2": original_m2, "round_10_response": final_myth_event["response"]["content"],
               "interpretation": "Textual continuity after original message eviction; not proof of causal ancestry or internal model memory."}
    check("textual continuity example", lineage["own_myth_rounds_containing_phrase"] == list(range(2, 11)) and lineage["original_myth_2_absent_from_call_19"])
    check("original input file unchanged", source.read_bytes() == raw)
    failures = [c for c in checks if not c["passed"]]
    verdict = {
        "scope": "One completed research final; both agents; 40 recorded logical calls, not all HTTP attempts.",
        "core_check_count": len(checks), "core_failures": failures, "checks": checks,
        "final_buffer_observations": buffer_checks,
        "contradicted": ["The stored conversation buffer never exceeds six complete exchanges (final buffers contain seven).",
                         "All directly accessible myths are limited to the preceding three rounds (partner myths reach four at steady myth calls)."],
        "unverified": ["Raw HTTP request bytes and provider-internal processing were not recorded.",
                       "Exact number and timing of underlying retry attempts.",
                       "Causal influence/internal recall of old information; phrase continuity is descriptive.",
                       "Universality across other runs, configurations, task orders, providers or error paths."],
        "first_own_myth_eviction_call": eviction_call, "selected_calls": selected_numbers,
    }
    write_json("verification.json", verdict)
    write_json("metadata.json", {"original_path": str(source), "task_order": data["task_order"], "run_metadata": metadata})
    write_json("source_manifest.json", {"original_path": str(source), "relative_path": str(SOURCE), "sha256": EXPECTED_SHA,
                                       "bytes": len(raw), "historical_source_files": source_checks, "research_provenance": provenance_rows})
    write_json("all_saved_interactions.json", {a: agent["interaction_history"] for a, agent in agents.items()})
    write_json("all_call_tables.json", all_rows)
    write_json("message_changes.json", message_changes)
    write_json("selected_calls_saved_snapshots.json", selected)
    write_json("selected_calls_provider_projection.json", {
        "warning": "DERIVED FROM SAVED SNAPSHOTS using the hash-verified historical Anthropic adapter. Not captured network bodies.",
        "source": "source_at_run/src/utils.py:322-336,610-643", "agent": FOCAL,
        "calls": [{"call": call["call"], "round": call["round"], "task": call["task"],
                   "input_fields": provider_projection(call["messages_sent"])} for call in selected]})
    write_json("indirect_textual_continuity.json", lineage)
    (out / "original_input.sha256").write_text(EXPECTED_SHA + "  " + str(source) + "\n")

    full_inputs = ["UNABRIDGED ROLE/CONTENT INPUTS: Agent_1 calls 7, 8, 9\n"
                   "Direct extraction of saved message contents; no summarization or truncation.\n"
                   "System is sent separately by Anthropic; other roles form its messages list.\n"
                   "Stored usage/reasoning are not sent; raw fields are in selected_calls_saved_snapshots.json.\n"
                   "Message IDs and ENTERED/PRESERVED labels are audit annotations, not input text.\n"]
    for call in selected:
        full_inputs.append(f"\n{'=' * 80}\nCALL {call['call']} | ROUND {call['round']} | {call['task']}\n")
        for i, message in enumerate(call["messages_sent"]):
            full_inputs.append(f"\n--- MESSAGE {i}: {call['message_ids'][i]} | {message['role']} | {call['status_since_previous_input'][i].upper()} ---\n")
            full_inputs.append(message["content"])
            full_inputs.append("\n--- END MESSAGE ---\n")
    (out / "selected_calls_full_input.txt").write_text("".join(full_inputs), encoding="utf-8")

    def nums(values):
        return ",".join(str(v) for v in values) or "-"

    headers = ["Call", "Round", "Task", "Messages", "Prior pairs", "Games", "Own myths", "Partner historical", "Partner current", "Removed"]
    table_rows = [[r["call"], r["round"], r["task"], r["total_messages_including_system"], r["retained_previous_exchanges"],
                   nums(r["previous_game_rounds"]), nums(r["own_myth_rounds"]), nums(r["partner_myth_rounds_in_historical_prompts"]),
                   nums(r["partner_myth_rounds_in_current_prompt"]), ", ".join(r["removed_messages"]) or "-"] for r in focal_rows]
    widths = [max(len(str(row[i])) for row in [headers] + table_rows) for i in range(len(headers))]
    table = "\n".join(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) for row in [headers] + table_rows)
    (out / "agent_1_call_table.txt").write_text(table + "\n", encoding="utf-8")
    report = f"""EXISTING-RUN MEMORY AUDIT — 2026-11-27

RESULT: The saved pre-request inputs implement a six-exchange FIFO sliding
window in this run. Two stronger claims fail: the final stored buffers contain
seven exchanges, and steady myth-writing inputs expose four partner myths.
No new simulation or LLM API call was made. No original data/config/source was edited.

ORIGINAL FINAL (not a checkpoint or results-only file)
{source}
SHA256: {EXPECTED_SHA}
Recorded commit: {commit}; code_dirty=false. All 17 implementation hashes plus
the historical config hash match the local Git objects. source_manifest.json
also preserves matching research-manifest and earlier paper-audit rows.

RUN CONFIGURATION
Two agents; ten complete rounds; one myth then one game decision per agent per
round (observed, not assumed): 20 logical calls each. memory_primary; capacity=6.
Replicate 0, fixed dyad, ordinary agents, no seed/reinjection/forced defection.
Myth topic anything; directive memory-primary myth template; minimal game prompts.
Communication noise: uniform negative, range $1, both transfer directions,
agents informed. Native Anthropic claude-sonnet-4-5-20250929; pinned temperature
policy default; thinking budget 8192; output ceiling 64000. Full resolved settings,
templates, seeds and condition hash are in metadata.json. The inherited game_params
temperature=0.8 is not the resolved request temperature: llm_request is authoritative.
Agent_1 saved timestamps: {agents[FOCAL]['interaction_history'][0]['timestamp']} to
{agents[FOCAL]['interaction_history'][-1]['timestamp']} (timezone not recorded).

WHAT THE SNAPSHOTS PROVE
agents.<agent>.interaction_history[i].messages_sent is the deep copy taken
before call_llm, not the cumulative interaction log or the final agent.messages.
Historical agents.py:168-182 captures it; :55-81 and :231-236 record it;
simulation.py:259-285 serializes stored events. These files are hash-matched
to this run and included under source_at_run/. Snapshot lengths grow to 14
then plateau, with original pairs disappearing; they are not cumulative histories.

The saved snapshot is PRE-ADAPTER. Historical utils.py:322-336 separates system
and drops assistant usage/reasoning. :610-643 constructs the Anthropic request.
Thus 14 here means 1 separate system input + 13 provider chat messages:
12 historical messages (6 pairs) + 1 unanswered current prompt. Provider input
projections are derived from saved evidence and verified code, NOT captured HTTP
bodies. The 40 successful, distinct response IDs do not exclude hidden transport
retries. The table enumerates every recorded logical interaction, not every
possible network attempt. Exact wire bytes/attempt counts remain unverified.

EMPIRICAL CHECKS (BOTH AGENTS)
{len(checks)} core checks; {len(failures)} failures. See verification.json for each comparison.
- Same system message in all 40 snapshots; no removal or modification.
- At most six complete previous pairs in every request (observed maximum six).
- Each pair equals the original prompt and response, including all saved fields.
- Exact chronological latest-six suffix; no split pair, skipped pair, mutation,
  reordered pair, or unexpected original-pair reintroduction.
- After filling, each next input removes precisely the oldest user AND assistant
  messages and retains the other original messages unchanged.
- All responses are saved as LLM/end_turn/complete; all 20 myths agree with their
  per-round records. No saved event errors or extra logical calls.

CONTRADICTIONS / LIMITATIONS
- Both final agent.messages contain 15 messages = system + SEVEN completed pairs.
  Each equals the last input plus its new answer. Therefore 'stored history never
  exceeds six' is contradicted. Six is the limit on PREVIOUS exchanges at request
  time. Trimming before the next call explains this; intermediate post-response
  buffer states were not independently snapshotted.
- Six prior exchanges = exactly three previous rounds at MYTH calls once full.
  At GAME calls they span four round labels: oldest round's game, two whole
  intervening rounds, and current round's myth. This is an exchange window, not
  a rule that deletes whole rounds.
- Direct myth source ages differ: at steady myth calls own R-3..R-1 (three),
  partner R-4..R-1 (four). A universal three-round direct-myth limit is false.
- Findings are scoped to this final and its successful path, not every run,
  provider, task ordering, retry path or the current repository configuration.

MYTH RETENTION AND REINTRODUCTION
At myth-writing rounds 1..10, own counts: 0,1,2,3,3,3,3,3,3,3;
partner counts: 0,1,2,3,4,4,4,4,4,4 (including current prompt).
Both agents' nine later myth prompts EXACTLY equal the saved fixed scaffold plus
the whole previous-round partner myth. No own full myth, older partner full myth,
or other variable history block is inserted. Game prompts contain no full myth.
The text 'Use the myth you wrote in the previous round as inspiration' is an
instruction, not a copy of that myth. The own original text is available through
retained assistant content. Although myth_writer.py passes last_myth to format(),
this run's saved template has no last_myth placeholder.

1. DIRECT RETENTION: original own user/assistant pairs remain verbatim. Older
   partner myths remain verbatim INSIDE retained historical user prompts.
2. EXPLICIT DELIVERY: each new later myth prompt inserts partner R-1, its first
   delivery to this agent. No older-than-R-1 full myth is reintroduced. Once a
   delivered partner text's containing prompt leaves, that original delivery is gone.
3. INDIRECT CONTINUITY: later myths can carry names/information forward after
   originals leave. 'Valley of Echoing Waters' occurs in own M2 through M10 and
   no partner myth. At call19, M2 is absent but the name remains in retained
   M7/M8/M9 and is repeated in M10. Exact texts and locations are in
   indirect_textual_continuity.json. This establishes text persistence, not
   causal ancestry, internal recall or the amount of information retained.
   Stored private reasoning is not resent by this run's Anthropic adapter.

AGENT_1 CALL TABLE
Games/own/partner numbers refer to source rounds; Cn means recorded call n.
Messages includes system; Prior pairs excludes system and unanswered current prompt.
Partner historical/current separates retained delivery prompts from the new prompt.
Removed lists exact message identities absent since the preceding input.
{table}

THREE CONSECUTIVE CALLS: 7, 8, 9
C7 R4 myth: retains C1..C6 (M1,G1,M2,G2,M3,G3), plus current M4 prompt.
C8 R4 game: removes C1.user + C1.assistant (M1), retains C2..C7;
  C7's answer enters history, and the G4 prompt is new.
C9 R5 myth: removes C2.user + C2.assistant (G1), retains C3..C8;
  C8's answer enters history, and the M5 prompt delivers partner M4.
Partner M1 remains in the historical M2 prompt through C9, leaves before C10.
Unabridged role/content: selected_calls_full_input.txt and report.html.
All raw snapshot fields: selected_calls_saved_snapshots.json.
Provider-format system/messages: selected_calls_provider_projection.json (derived).
Every event for both agents: all_saved_interactions.json (direct extraction).
Exact full-message additions/removals: message_changes.json.

AUTHORITATIVE SOURCE LOCATIONS (ALL UNDER source_at_run/ AT THE RECORDED COMMIT)
src/agents.py:83-88 truncation; :168-182 snapshot/call; :55-81 recording;
  :217-236 append response and record snapshot.
src/utils.py:322-336 Anthropic conversion; :361 retry default;
  :610-674 request construction and internal retries.
src/simulation.py:259-285 serialization.
src/myth_writer.py:131-204 previous-round selection and prompt formatting.
config/experiments_noisy.yaml:34-41 model profile; :388-395 myth template;
  :983-1000 dyad parameters; :5149-5167 inherited condition; :5179-5183 chosen set.
Current src/agents.py matches this version, but other current files have changed.

REPRODUCTION
From repository root, using Python 3.10+ and local Git history, no dependencies:
python3 docs/research/memory_window_audit_20261127/audit.py --out /tmp/memory-window-recheck
The script fails if the original final or historical implementation hashes differ.
It reads saved JSON and local git objects only. It never imports experiment code.
"""
    (out / "REPORT.txt").write_text(report, encoding="utf-8")
    (out / "REPRODUCE.txt").write_text(report[report.index("REPRODUCTION\n"):], encoding="utf-8")
    # Standalone human-readable report: no JavaScript, external assets or server.
    e = html.escape
    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>Saved-run memory audit</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#18212a}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #bdc7d0;padding:7px;text-align:left}th{background:#eaf0f5}pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5;background:#f4f6f8;padding:16px}.scroll{overflow:auto}.entered{border-left:5px solid #258455}.preserved{border-left:5px solid #627b91}a{color:#07519b}section{margin:40px 0}</style>",
             "<h1>Saved-run memory audit</h1><p>Six previous exchanges per input verified; seven exchanges in final buffers. Three own myths versus four partner myths at steady myth-writing calls.</p>",
             "<p><a href='REPORT.txt'>Full findings and source locations</a> · <a href='verification.json'>Individual checks</a> · <a href='metadata.json'>Metadata</a> · <a href='audit.py'>Offline script</a> · <a href='selected_calls_saved_snapshots.json'>Raw snapshots</a></p>",
             "<h2>Every recorded call for Agent_1</h2><p>Counts include the separately sent system input. Myth/game numbers are source rounds. Removed identifies both messages of an exchange. Full exact-text occurrence locations are in all_call_tables.json.</p><div class='scroll'><table><thead><tr>",
             "".join(f"<th>{e(h)}</th>" for h in headers), "</tr></thead><tbody>"]
    for row in table_rows:
        parts.append("<tr>" + "".join(f"<td>{e(str(cell))}</td>" for cell in row) + "</tr>")
    parts.append("</tbody></table></div><h2>Unabridged inputs around first own-myth eviction</h2><p>All saved role/content text below is unchanged. Message IDs and entered/preserved labels are annotations. System is a separate provider field; usage/reasoning metadata is omitted by that provider adapter. See raw snapshots for every saved field.</p>")
    for call in selected:
        parts.append(f"<section id='call-{call['call']}'><h3>Call {call['call']}: round {call['round']} {call['task']}</h3>")
        change = message_changes[FOCAL][call["call"] - 1]
        parts.append("<p>Removed since previous input: " + e(", ".join(item["id"] for item in change["removed"]) or "none") + "</p>")
        for i, message in enumerate(call["messages_sent"]):
            status = call["status_since_previous_input"][i]
            parts.append(f"<h4>Message {i}: {e(call['message_ids'][i])} · {e(message['role'])} · {status}</h4><pre class='{status}'>" + e(message["content"]) + "</pre>")
        parts.append("</section>")
    parts.append("<h2>Detailed findings</h2><pre>" + e(report) + "</pre></html>")
    (out / "report.html").write_text("".join(parts), encoding="utf-8")
    assert source.read_bytes() == raw, "Original data changed during audit"
    print(json.dumps({"source_sha256": EXPECTED_SHA, "output": str(out), "core_checks": len(checks),
                      "core_failures": len(failures), "selected_calls": selected_numbers,
                      "final_buffers": buffer_checks}, indent=2))
    if failures:
        raise SystemExit("Core checks failed: inspect verification.json; do not report a pass.")


if __name__ == "__main__":
    main()
