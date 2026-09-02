"""Phase 3 chat-memory validator.

Loads a Phase 3 output JSON, walks one agent's interaction_history, and:
  1. Prints the `messages_sent` payload at rounds 1, 5, and 10.
  2. Asserts that messages[0:3] are byte-for-byte identical across all three
     rounds (system + seed_user + seed_myth).
  3. Asserts that exactly 4 messages were sent each round (the fourth being
     the round-specific game prompt).

This is the contract the spec promises. If any assertion fails, the
implementation has a bug and we don't spend Sonnet money.

Usage:
  python scripts/phase3_inspect_chat_memory.py <run.json> [--agent Agent_1]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def msg_digest(msg):
    payload = json.dumps(
        {"role": msg.get("role"), "content": msg.get("content")},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def short(text, limit=180):
    if text is None:
        return "<None>"
    text = text.replace("\n", " \\n ")
    return text if len(text) <= limit else text[:limit] + "…"


def get_round_message_sent(agent_block, round_num, task="game"):
    """Find the messages_sent payload for a given round and task ('game' or 'myth')."""
    for event in agent_block.get("interaction_history", []):
        meta = event.get("metadata") or {}
        if meta.get("round") == round_num and meta.get("task") == task:
            return event.get("messages_sent")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_json", help="Path to Phase 3 run JSON")
    parser.add_argument("--agent", default="Agent_1", help="Agent ID to inspect")
    parser.add_argument(
        "--rounds",
        nargs="*",
        type=int,
        default=[1, 5, 10],
        help="Rounds to inspect (default: 1 5 10)",
    )
    args = parser.parse_args()

    path = Path(args.run_json)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(2)

    with open(path) as f:
        data = json.load(f)

    agents = data.get("agents", {})
    if args.agent not in agents:
        print(f"ERROR: agent {args.agent} not in run. Available: {list(agents.keys())}", file=sys.stderr)
        sys.exit(2)

    agent_block = agents[args.agent]
    # Detect whether this run was seeded. The chat-memory prefix shared across
    # tasks within a round is 3 messages when seeded ([system, seed_user, seed_myth])
    # and 1 when not seeded ([system]).
    has_seed = bool(data.get("run_metadata", {}).get("seed_myth"))
    shared_prefix = 3 if has_seed else 1

    print(f"=== Phase 3 chat-memory inspection ===")
    print(f"Run:    {path}")
    print(f"Agent:  {args.agent}")
    print(f"Rounds: {args.rounds}")
    print(f"Seeded: {has_seed}  (shared-prefix slots to check across tasks: {shared_prefix})")
    print()

    # Determine which tasks were run by checking the agent's interaction_history.
    tasks_seen = sorted({
        (event.get("metadata") or {}).get("task")
        for event in agent_block.get("interaction_history", [])
        if (event.get("metadata") or {}).get("task") in ("game", "myth")
    })
    print(f"Tasks recorded: {tasks_seen}")
    print()

    failures = []
    # per_round_messages: {task: {round_num: messages}}
    per_round_messages = {t: {} for t in tasks_seen}

    for task in tasks_seen:
        for r in args.rounds:
            msgs = get_round_message_sent(agent_block, r, task=task)
            if msgs is None:
                print(f"[{task}] ROUND {r}: no {task} interaction recorded — SKIP")
                failures.append(f"{task} round {r} missing")
                continue
            per_round_messages[task][r] = msgs
            print(f"--- [{task}] ROUND {r} — {len(msgs)} messages ---")
            for i, m in enumerate(msgs):
                digest = msg_digest(m)
                print(f"  [{i}] role={m.get('role'):9} digest={digest}  preview: {short(m.get('content'))}")
            print()
            expected_len = shared_prefix + 1  # prefix + 1 current-task prompt
            if len(msgs) != expected_len:
                failures.append(f"{task} round {r}: expected {expected_len} messages, got {len(msgs)}")

        # Within-task: shared-prefix messages identical across all sampled rounds.
        task_rounds = per_round_messages[task]
        if len(task_rounds) >= 2:
            ref_round = min(task_rounds)
            ref_msgs = task_rounds[ref_round]
            for r, msgs in task_rounds.items():
                if r == ref_round:
                    continue
                for i in range(min(shared_prefix, len(ref_msgs), len(msgs))):
                    if msg_digest(ref_msgs[i]) != msg_digest(msgs[i]):
                        failures.append(
                            f"{task} round {r} message[{i}] differs from {task} round {ref_round}"
                        )

    # Cross-task: shared-prefix messages should also match between myth and game in the same round.
    if "game" in per_round_messages and "myth" in per_round_messages:
        common_rounds = set(per_round_messages["game"]) & set(per_round_messages["myth"])
        for r in sorted(common_rounds):
            for i in range(min(shared_prefix, len(per_round_messages["game"][r]), len(per_round_messages["myth"][r]))):
                gd = msg_digest(per_round_messages["game"][r][i])
                md = msg_digest(per_round_messages["myth"][r][i])
                if gd != md:
                    failures.append(
                        f"round {r}: game message[{i}] differs from myth message[{i}]"
                    )

    print("=== Contract check ===")
    if not failures:
        print("PASS: messages[0:3] identical across rounds (and across tasks within a round); exactly 4 messages per round.")
        sys.exit(0)
    else:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
