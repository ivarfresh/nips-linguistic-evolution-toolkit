#!/usr/bin/env python3
"""Audit every LLM call in saved runs for dropped or malformed context.

Per call (from interaction_history.messages_sent), checks:
- system message present, with the informed-noise notice (when expected)
- no empty message content anywhere in the context
- own-myth memory window: the most recent own myth is present, the set of
  own myths in context is a contiguous run of the most recent rounds
- later myth prompts quote the previous-round partner's myth verbatim
- later game prompts carry the co-player block with min(r-1, 3) entries and
  no self-history block / own-myth recap
- responses are non-empty and no interaction recorded an error

Per run: 10 rounds, 4 complete dyads and 8 balances per round, 8 myths per
myth round, 8 game responses per round.

Usage: python scripts/audit_context_integrity.py <run_dir> [<run_dir> ...]
"""

import json
import re
import sys
from pathlib import Path

issues = []
stats = {"runs": 0, "calls": 0}


def check_run(path, run):
    name = Path(path).name
    stats["runs"] += 1
    hist = run["conversation_history"]
    order = run.get("task_order") or []
    informed = bool(
        (run.get("run_metadata") or {}).get("model") and "noise" in path
    )

    # Round-level completeness
    if len(hist) != 10:
        issues.append(f"{name}: {len(hist)} rounds (expected 10)")
    for e in hist:
        r = e["round"]
        dyads = [d for d in e.get("dyads") or [] if d.get("sent") is not None and d.get("returned") is not None]
        if len(dyads) != 4:
            issues.append(f"{name} r{r}: {len(dyads)} complete dyads (expected 4)")
        if len(e.get("balances") or {}) != 8:
            issues.append(f"{name} r{r}: {len(e.get('balances') or {})} balances (expected 8)")
        if len(e.get("game_responses") or {}) != 8:
            issues.append(f"{name} r{r}: {len(e.get('game_responses') or {})} game responses")
        if "myth" in order:
            myths = e.get("myths") or {}
            if len(myths) != 8:
                issues.append(f"{name} r{r}: {len(myths)} myths (expected 8)")
            for aid, m in myths.items():
                if not (m or "").strip():
                    issues.append(f"{name} r{r}: empty myth for {aid}")

    myths_by_round = {e["round"]: e.get("myths") or {} for e in hist}
    pairings_by_round = {e["round"]: e.get("pairings") or [] for e in hist}

    def partner_of(aid, rnd):
        for p in pairings_by_round.get(rnd, []):
            if aid in p.get("agents", []):
                return [a for a in p["agents"] if a != aid][0]
        return None

    for aid, agent in run["agents"].items():
        for ev in agent.get("interaction_history", []):
            stats["calls"] += 1
            md = ev.get("metadata") or {}
            r, task = md.get("round"), md.get("task")
            ms = ev.get("messages_sent") or []
            tag = f"{name} {aid} r{r} {task}"

            if ev.get("error"):
                issues.append(f"{tag}: recorded error {ev['error'].get('type')}")
            resp = (ev.get("response") or {}).get("content", "")
            if not resp.strip():
                issues.append(f"{tag}: empty response")

            if not ms or ms[0].get("role") != "system":
                issues.append(f"{tag}: no system message first")
                continue
            if informed and "communication noise" not in ms[0]["content"]:
                issues.append(f"{tag}: informed-noise notice missing from system prompt")
            for i, m in enumerate(ms):
                if not (m.get("content") or "").strip():
                    issues.append(f"{tag}: empty content at msg[{i}]")
            ctx = " ".join(m.get("content") or "" for m in ms)
            prompt = ms[-1].get("content") or ""

            # own-myth window: contiguous, most recent present
            if "myth" in order:
                avail = [
                    rr for rr in sorted(myths_by_round)
                    if myths_by_round[rr].get(aid)
                    and (rr < r or (rr == r and not (task == "myth")
                         and order[0] == "myth"))
                ]
                found = sorted(
                    rr for rr in avail if myths_by_round[rr][aid][:90] in ctx
                )
                if avail:
                    expected_window = avail[-min(3, len(avail)):]
                    if found != expected_window:
                        issues.append(
                            f"{tag}: own-myth window {found} != expected {expected_window}"
                        )

            if task == "myth" and r and r > 1:
                p_prev = partner_of(aid, r - 1)
                pm = myths_by_round.get(r - 1, {}).get(p_prev)
                if pm and pm[:90] not in prompt:
                    issues.append(f"{tag}: previous-round partner myth missing from prompt")

            if task == "game" and r and r > 1:
                m = re.findall(r"^- Round", prompt, re.MULTILINE)
                expected = min(r - 1, 3)
                if "last 3 game(s)" not in prompt:
                    issues.append(f"{tag}: co-player block missing")
                elif len(m) != expected:
                    issues.append(f"{tag}: co-player block has {len(m)} entries (expected {expected})")
                if "Your last" in prompt and "game(s)" in prompt:
                    issues.append(f"{tag}: self-history block present (should be dropped)")
                if "Here is the myth you wrote" in prompt:
                    issues.append(f"{tag}: own-myth recap in game prompt")


def main():
    dirs = sys.argv[1:]
    for d in dirs:
        for p in sorted(Path(d).rglob("*.json")):
            if p.name.endswith((".results.json", ".checkpoint.json", ".error.json")):
                continue
            with open(p) as f:
                check_run(str(p), json.load(f))
    print(f"Audited {stats['runs']} runs, {stats['calls']} LLM calls.")
    if issues:
        print(f"\n{len(issues)} ISSUES:")
        for i in issues[:60]:
            print(" !!", i)
        if len(issues) > 60:
            print(f" ... and {len(issues)-60} more")
    else:
        print("NO ISSUES — nothing dropped, all contexts complete.")


if __name__ == "__main__":
    main()
