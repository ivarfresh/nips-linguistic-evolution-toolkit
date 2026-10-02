#!/usr/bin/env python3
"""Cultural transmission pilot analysis (rules: docs/research/cultural_transmission_pilot_20261002/PREREGISTRATION.md).

  python3 analyses/cultural_transmission_pilot.py --newcomers   # 10 samples x 2 boards x run (API calls)
  python3 analyses/cultural_transmission_pilot.py --analyze     # markers, $2 sends, newcomer result

Reads the audited finals listed in the pilot's completion receipt. Newcomer replies are appended to
data/analysis/cultural_transmission_pilot_20261002/newcomers.jsonl and the run resumes from it.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses.myth_replay_probe import append, call_cost, done_keys, parse_send, replay  # noqa: E402

RECEIPT = ROOT / "data/json/noise_experiments/cultural_transmission_pilot_20261002/completion_receipt.json"
OUT = ROOT / "data/analysis/cultural_transmission_pilot_20261002"
FIGS = ROOT / "docs/research/cultural_transmission_pilot_20261002"
PLANT_AGENT = "Agent_1"
MARKER = re.compile(r"velmar|two of (the )?five", re.I)
SAMPLES = 10
INTRO = "Here are myths written by members of the population you are joining:\n\n"


def finals(arm=None):
    rows = json.loads(RECEIPT.read_text())["finals"]
    return [r for r in rows if arm is None or r["arm"] == arm]


def load(rec):
    return json.loads((ROOT / rec["path"]).read_text())


def round1_investor_call(d):
    """System prompt and round-1 investor prompt of a non-planted investor, as logged."""
    for agent_id, a in sorted(d["agents"].items()):
        if agent_id == PLANT_AGENT:
            continue
        for x in a["interaction_history"]:
            m = x.get("metadata") or {}
            if m.get("task") == "game" and m.get("round") == 1 and m.get("role") == "investor":
                msgs = x["messages_sent"]
                assert msgs[0]["role"] == "system" and msgs[-1]["role"] == "user"
                return msgs[0]["content"], msgs[-1]["content"], d["run_metadata"]["llm_request"], x.get("temperature", "default")
    raise ValueError("no round-1 investor call")


def board(d, rnd):
    entry = next(e for e in d["conversation_history"] if e["round"] == rnd)
    myths = [t for a, t in sorted(entry["myths"].items()) if a != PLANT_AGENT]
    assert len(myths) == 7
    return "\n\n---\n\n".join(myths)


def newcomers(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "newcomers.jsonl"
    done = done_keys(path)
    jobs = []
    for rec in finals():
        d = load(rec)
        system, investor_prompt, plan, temperature = round1_investor_call(d)
        for rnd in (1, 10):
            text = board(d, rnd)
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": f"{INTRO}{text}\n\n{investor_prompt}"}]
            for s in range(SAMPLES):
                key = f"{rec['path']}|R{rnd}|s{s}"
                if key not in done:
                    jobs.append({"key": key, "run": rec["path"], "arm": rec["arm"], "replicate_id": rec["replicate_id"],
                                 "board_round": rnd, "board_has_marker": bool(MARKER.search(text)),
                                 "ctx": {"plan": plan, "temperature": temperature}, "messages": msgs})
    print(f"PREFLIGHT MODEL=anthropic/claude-sonnet-4.5 (run plan) N={len(jobs)} WORKERS={workers} EST_COST=${len(jobs) * 0.02:.2f}", flush=True)

    def work(j):
        res = replay(j["ctx"], j["messages"])
        rec = {k: j[k] for k in ("key", "run", "arm", "replicate_id", "board_round", "board_has_marker")}
        rec.update(send=res["send"], text=res["text"], error=res["error"], finish_reason=res.get("finish_reason"),
                   usage=res["usage"], messages=j["messages"], cost=call_cost("Sonnet", res["usage"] or {}))
        append(path, rec)
        return rec

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        recs = list(pool.map(work, jobs))
    print(f"newcomers done: {len(recs)} calls, ${sum(r['cost'] for r in recs):.2f}, errors {sum(bool(r['error']) for r in recs)}")


def analyze():
    FIGS.mkdir(parents=True, exist_ok=True)
    myth_rows, send_rows = [], []
    for rec in finals():
        d = load(rec)
        got_from_plant = set()
        for e in sorted(d["conversation_history"], key=lambda e: e["round"]):
            for a, t in e["myths"].items():
                myth_rows.append({"run": rec["replicate_id"], "arm": rec["arm"], "round": e["round"], "agent": a,
                                  "planted_agent": a == PLANT_AGENT, "marker": bool(MARKER.search(t)),
                                  "velmar": "velmar" in t.lower()})
            for dy in e["dyads"]:
                inv, tru = dy["investor"], dy["trustee"]
                send_rows.append({"run": rec["replicate_id"], "arm": rec["arm"], "round": e["round"], "investor": inv,
                                  "planted_agent": inv == PLANT_AGENT, "send": dy["sent_decision"],
                                  "exact2": dy["sent_decision"] == 2.0, "had_send_from_plant": inv in got_from_plant,
                                  "after_board": e["round"] >= 2})
            got_from_plant |= {dy["trustee"] for dy in e["dyads"] if dy["investor"] == PLANT_AGENT}
    myths, sends = pd.DataFrame(myth_rows), pd.DataFrame(send_rows)
    others = myths[~myths["planted_agent"]]
    text = others.groupby(["arm", "run", "round"])[["marker", "velmar"]].mean().unstack(["arm", "run"])
    text.to_csv(FIGS / "marker_share_by_round.csv")
    play = sends.groupby(["arm", "planted_agent", "after_board", "had_send_from_plant", "run"]).agg(
        n=("exact2", "size"), exact2=("exact2", "sum"), mean_send=("send", "mean")).reset_index()
    play.to_csv(FIGS / "exact2_sends.csv", index=False)
    pd.set_option("display.width", 200)
    print("Share of non-Agent_1 myths with a marker, by round (columns: run):")
    print(text["marker"].round(2).to_string())
    print("\nAgent_1 sends by round:", sends[sends.planted_agent].groupby(["arm", "run", "round"])["send"].first().unstack("round").to_string())
    print("\nExact-$2 sends:\n", play.to_string(index=False))
    path = OUT / "newcomers.jsonl"
    if path.exists():
        nc = pd.DataFrame([json.loads(x) for x in path.read_text().splitlines() if x.strip()])
        nc["exact2"] = nc["send"] == 2.0
        g = nc.groupby(["arm", "replicate_id", "board_round"]).agg(n=("send", "size"), parsed=("send", "count"),
                                                             exact2=("exact2", "mean"), mean_send=("send", "mean"),
                                                             board_has_marker=("board_has_marker", "first"))
        g.to_csv(FIGS / "newcomers.csv")
        print("\nNewcomers:\n", g.round(3).to_string())
        diff = g["exact2"].unstack("board_round")
        for arm, d in diff.groupby(level="arm"):
            passed = int(((d[10] - d[1]) >= 0.20).sum())
            print(f"\nRetelling rule ({arm}): R10 - R1 exact-$2 share >= 20 pts in {passed} of {len(d)} runs "
                  f"-> {'ESTABLISHED' if passed >= 2 else 'not established'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--newcomers", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.newcomers:
        newcomers(args.workers)
    if args.analyze:
        analyze()


if __name__ == "__main__":
    main()
