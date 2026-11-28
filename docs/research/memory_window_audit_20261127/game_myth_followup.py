#!/usr/bin/env python3
"""Offline follow-up: measure myth exposure in the existing game->myth final."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / (
    "data/json/noise_experiments/negative_only_crossmodel_reasoning_rerun_20260909/"
    "negative_only_reasoning_rerun_dyad_game_myth_claude_n5/claude-sonnet-4.5/"
    "game_myth/noisy2_crossmodel_negative_twotask_r3/"
    "negative_only_reasoning_rerun_dyad_game_myth_claude_n5_000_neutral_rep00_"
    "memtest_memory_primary_anything.json"
)
EXPECTED_SHA = "a44d5243e6b03c5282033e25948cc181d723fedf4f73661a2534a3720bd99e99"


def pair(event):
    response = event["response"]
    return [{"role": "user", "content": event["prompt"]},
            {"role": "assistant", "content": response["content"],
             "reasoning": response.get("reasoning"), "usage": response.get("usage")}]


def main():
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA
    data = json.loads(raw)
    meta = data["run_metadata"]
    assert data["task_order"] == ["game", "myth"]
    assert meta["memory_capacity"] == 6 and meta["chat_memory_mode"] == "memory_primary"
    assert meta["defector_count"] == 0 and meta["random_defection_probability"] == 0
    assert [r["round"] for r in data["conversation_history"]] == list(range(1, 11))
    # Same recorded implementation/configuration as the already archived primary run.
    primary_meta = json.loads((OUT / "metadata.json").read_text())["run_metadata"]
    for key in ("code_commit", "config_sha256"):
        assert meta[key] == primary_meta[key]
    assert meta["experiment_condition"]["implementation"] == primary_meta["experiment_condition"]["implementation"]
    rows = []
    for agent_id, agent in data["agents"].items():
        partner = next(a for a in data["agents"] if a != agent_id)
        events = agent["interaction_history"]
        peer_events = data["agents"][partner]["interaction_history"]
        assert len(events) == 20
        for index, event in enumerate(events):
            rnd, task = event["metadata"]["round"], event["metadata"]["task"]
            messages = event["messages_sent"]
            assert (rnd, task) == (index // 2 + 1, ("game", "myth")[index % 2])
            assert messages[0] == events[0]["messages_sent"][0]
            observed = messages[1:-1]
            expected = [m for previous in events[max(0, index - 6):index] for m in pair(previous)]
            assert observed == expected
            assert messages[-1] == {"role": "user", "content": event["prompt"]}
            assert "error" not in event and event["response"]["response_source"] == "llm"
            assert event["response"]["usage"]["outcome"] == "complete"
            own = [e["metadata"]["round"] for e in events if e["metadata"]["task"] == "myth"
                   and any(m["role"] == "assistant" and m["content"] == e["response"]["content"] for m in messages)]
            historical = [e["metadata"]["round"] for e in peer_events if e["metadata"]["task"] == "myth"
                          and any(m["role"] == "user" and e["response"]["content"] in m["content"] for m in messages[:-1])]
            current = [e["metadata"]["round"] for e in peer_events if e["metadata"]["task"] == "myth"
                       and e["response"]["content"] in messages[-1]["content"]]
            rows.append({"agent": agent_id, "call": index + 1, "round": rnd, "task": task,
                         "messages_including_system": len(messages), "own_myth_rounds": own,
                         "partner_in_history": historical, "partner_in_current_prompt": current,
                         "original_json_location": f"agents.{agent_id}.interaction_history[{index}].messages_sent"})
            if rnd >= 5:
                assert own == list(range(rnd - 3, rnd))
                assert historical == list(range(rnd - 4, rnd - 1))
                assert current == ([rnd - 1] if task == "myth" else [])
    result = {"original_path": str(SOURCE), "sha256": EXPECTED_SHA, "run_metadata": meta,
              "checks_passed": True, "recorded_calls_checked": len(rows), "rows": rows,
              "finding": "At steady game calls: three own myths R-3..R-1 and three partner myths R-4..R-2. At myth calls: same retained myths plus partner R-1 in current prompt. This describes saved pre-adapter inputs, not wire captures or behavioural effects."}
    (OUT / "game_myth_followup.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    assert SOURCE.read_bytes() == raw
    print(json.dumps({"checks_passed": True, "calls": len(rows), "round_5": [r for r in rows if r["agent"] == "Agent_1" and r["round"] == 5]}, indent=2))


if __name__ == "__main__":
    main()
