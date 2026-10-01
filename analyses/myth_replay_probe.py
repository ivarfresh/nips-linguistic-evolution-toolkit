#!/usr/bin/env python3
"""Edit-and-replay probe: does the amount stated in a myth cause the next send?

A decision is replayed from the exact logged messages of a September run, through the
repo's own transport (src.utils.call_llm) with the run's recorded request plan. The
models are stateless, so a replay with one sentence edited is the original decision
with only that sentence changed.

This file implements the PILOT (design: docs/research/myth_opening_plan_20260930/README.md
section 7, agreed 2026-10-01):

  noise    5 round-1 sender contexts (myth->game, own myth in context) and 5 round-2
           sender contexts (8-agent myth->game, shown myth in context) per family
           (Sonnet 4.5, GPT-5 Nano, Gemini 3.7 Flash), unedited, 6 samples each.
           Measures the spread of sends on identical input.
  refusal  20 Sonnet round-1 contexts whose own myth states an amount, with the amount
           rewritten to $1 by an editor model; 1 sample each. Measures refusals and
           parse failures on edited own-myth text.

Contexts come only from the 156 no-defector September myth runs (linguistic run list),
homogeneous runs only, sampled with a fixed seed. Raw run JSONs are read from
--data-root (default: the main checkout). Results are appended to JSONL under
data/analysis/myth_replay_probe_20261001/ (gitignored) and the run resumes from them.

  python3 analyses/myth_replay_probe.py --preflight
  python3 analyses/myth_replay_probe.py --stage noise
  python3 analyses/myth_replay_probe.py --stage refusal
  python3 analyses/myth_replay_probe.py --summary
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path
import random
import re
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses.linguistic_corpus import run_list  # noqa: E402

OUT = ROOT / "data/analysis/myth_replay_probe_20261001"
DEFAULT_DATA_ROOT = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-linguistic-evolution-toolkit")
SEED = 20261001
FAMILY = {"claude-sonnet-4.5": "Sonnet", "gpt-5-nano": "GPT", "gemini-3.7-flash": "Gemini"}
# USD per 1M tokens: Sonnet and Nano from analyses/_llm_judge.py; Gemini 3.7 Flash has no repo
# price, so Gemini 2.5 Flash's is used as a stand-in (Gemini calls emit ~11 output tokens).
PRICE = {"Sonnet": (3.00, 15.00), "GPT": (0.05, 0.40), "Gemini": (0.30, 2.50)}
EDITOR_MODEL = "z-ai/glm-5.2"
EDITOR_PRICE = (0.65, 2.042)  # analyses/myth_moral_judge.py PRICES
NOISE_CONTEXTS, NOISE_SAMPLES, REFUSAL_CONTEXTS = 5, 6, 20
_lock = threading.Lock()


# ----------------------------------------------------------------------------- contexts

def family_of(model: str) -> str | None:
    return next((f for k, f in FAMILY.items() if k in model), None)


def load_runs(data_root: Path) -> list[dict]:
    runs = []
    for size in (2, 8):
        for spec in run_list(size, data_root):
            if spec["task_order"] != "myth_game":
                continue
            d = json.loads(Path(spec["abs_path"]).read_text())
            req = d["run_metadata"].get("llm_request") or {}
            if req.get("provider") in (None, "mixed"):
                continue  # homogeneous runs only
            fam = family_of(req["model"])
            if fam:
                runs.append({**spec, "family": fam, "data": d})
    return runs


def sender_calls(run: dict, rnd: int) -> list[dict]:
    calls = []
    for agent, a in run["data"]["agents"].items():
        for x in a["interaction_history"]:
            m = x.get("metadata") or {}
            if m.get("task") == "game" and m.get("round") == rnd and m.get("role") == "investor":
                calls.append({"agent": agent, "messages": x["messages_sent"],
                              "temperature": x.get("temperature", "default"),
                              "original": (x.get("response") or {}).get("content", "")})
    return calls


def contexts(runs: list[dict], fam: str, rnd: int, size: int | None, n: int, rng) -> list[dict]:
    pool = [r for r in runs if r["family"] == fam and (size is None or r["size"] == size)]
    rng.shuffle(pool)
    out = []
    for r in pool:  # one context per run, so contexts spread across runs
        cs = sender_calls(r, rnd)
        if cs:
            c = rng.choice(cs)
            out.append({"run": r["path"], "family": fam, "round": rnd, "size": r["size"],
                        "plan": r["data"]["run_metadata"]["llm_request"], **c})
        if len(out) == n:
            break
    return out


# ----------------------------------------------------------------------------- parsing

NUMBER = r"\$?\s*(-?\d+(?:\.\d*)?|-?\.\d+)"


def parse_send(text: str) -> float | None:
    """Same rule as games/trust_game_noisy.py TrustGame._extract_amount for key 'send'."""
    for key, alt in (("send", None), (None, "return")):
        k = key or alt
        pat = rf"'{k}':\s*{NUMBER}|" + rf'"{k}":\s*{NUMBER}'
        ms = list(re.finditer(pat, text or "", re.IGNORECASE))
        if key and ms:
            return float(next(g for g in ms[0].groups() if g is not None))
        if alt and len(ms) == 1:
            return float(next(g for g in ms[0].groups() if g is not None))
    return None


# ----------------------------------------------------------------------------- calls

def replay(ctx: dict, messages: list[dict]) -> dict:
    from src.llm_settings import RequestPlan, canonical
    from src.utils import call_llm, create_llm_client
    plan = RequestPlan(canonical(ctx["plan"]))
    client = create_llm_client(ctx["plan"]["model"], request_plan=plan)
    try:
        r = call_llm(client, ctx["plan"]["model"], ctx["temperature"], messages)
        text = r.get("content", "") if isinstance(r, dict) else str(r)
        usage = r.get("usage", {}) if isinstance(r, dict) else {}
        return {"text": text, "send": parse_send(text), "usage": usage,
                "finish_reason": usage.get("finish_reason"), "error": None}
    except Exception as err:  # noqa: BLE001  record and continue
        return {"text": "", "send": None, "usage": getattr(err, "usage", {}), "error": f"{type(err).__name__}: {err}"[:300]}


def call_cost(fam: str, usage: dict) -> float:
    pin, pout = PRICE[fam]
    return (usage.get("input_tokens") or 0) * pin / 1e6 + (usage.get("output_tokens") or 0) * pout / 1e6


def append(path: Path, rec: dict) -> None:
    with _lock, path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(line)["key"] for line in path.read_text().splitlines() if line.strip()}


# ----------------------------------------------------------------------------- editor

EDIT_PROMPT = """Below is a short myth. Rewrite ONLY the sentence or phrase that states how much a giver
sends, gives or offers, so that the amount becomes {amount} (out of five). Keep every other word,
the formatting and the length the same. If the amount appears more than once, change each
occurrence consistently. Return only the full rewritten myth, nothing else.

MYTH:
{myth}"""


def edit_myth(myth: str, amount: str) -> tuple[str, float]:
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    r = client.chat.completions.create(
        model=EDITOR_MODEL, temperature=0, max_tokens=1200,
        messages=[{"role": "user", "content": EDIT_PROMPT.format(amount=amount, myth=myth)}],
        extra_body={"reasoning": {"enabled": False}, "usage": {"include": True}})
    u = r.usage
    cost = getattr(u, "cost", None)
    if cost is None:
        cost = u.prompt_tokens * EDITOR_PRICE[0] / 1e6 + u.completion_tokens * EDITOR_PRICE[1] / 1e6
    return r.choices[0].message.content.strip(), float(cost)


def changed_fraction(a: str, b: str) -> float:
    import difflib
    sm = difflib.SequenceMatcher(None, a.split(), b.split())
    same = sum(bl.size for bl in sm.get_matching_blocks())
    return 1 - same / max(len(a.split()), 1)


# ----------------------------------------------------------------------------- stages

def plan_noise(runs, rng) -> list[dict]:
    jobs = []
    for fam in ("Sonnet", "GPT", "Gemini"):
        for rnd, size in ((1, None), (2, 8)):
            for c in contexts(runs, fam, rnd, size, NOISE_CONTEXTS, rng):
                for s in range(NOISE_SAMPLES):
                    jobs.append({**c, "key": f"{c['run']}|{c['agent']}|r{rnd}|s{s}", "sample": s})
    return jobs


def stated_amount(myth_text: str, rules: dict, run: str, agent: str) -> float | None:
    return rules.get((run, agent))


def plan_refusal(runs, rng) -> list[dict]:
    import pandas as pd
    rules = pd.read_csv(ROOT / "data/analysis/linguistic_20260923/myth_rules_september_z-ai__glm-5.2.csv")
    r1 = rules[(rules["round"] == 1) & (rules["task_order"] == "myth_game") & rules["send_amount"].notna()]
    named = {(row.run_id, row.agent) for row in r1.itertuples()}
    pool = []
    for r in (r for r in runs if r["family"] == "Sonnet"):  # every round-1 sender, not one per run
        for c in sender_calls(r, 1):
            if ((Path(r["path"]).stem, c["agent"]) in named and len(c["messages"]) == 4
                    and c["messages"][2]["role"] == "assistant"):
                pool.append({"run": r["path"], "family": "Sonnet", "round": 1, "size": r["size"],
                             "plan": r["data"]["run_metadata"]["llm_request"], **c})
    rng.shuffle(pool)
    return [{**c, "key": f"{c['run']}|{c['agent']}|r1|edit1"} for c in pool[:REFUSAL_CONTEXTS]]


def preflight(runs) -> None:
    rng = random.Random(SEED)
    noise = plan_noise(runs, rng)
    refusal = plan_refusal(runs, random.Random(SEED + 1))
    est = 0.0
    tok = {("Sonnet", 1): (870, 530), ("Sonnet", 2): (1960, 860), ("GPT", 1): (640, 3980),
           ("GPT", 2): (1330, 6320), ("Gemini", 1): (760, 11), ("Gemini", 2): (1510, 13)}
    for j in noise:
        i, o = tok[(j["family"], j["round"])]
        est += call_cost(j["family"], {"input_tokens": i, "output_tokens": o})
    est += len(refusal) * (call_cost("Sonnet", {"input_tokens": 870, "output_tokens": 530}) + 0.002)
    fams = {f: sum(j["family"] == f for j in noise) for f in ("Sonnet", "GPT", "Gemini")}
    print(f"MODEL=sonnet-4.5,gpt-5-nano,gemini-3.7-flash(+{EDITOR_MODEL} editor) "
          f"N=noise:{len(noise)} {fams} refusal:{len(refusal)} WORKERS=4 EST_COST=${est:.2f}")


def run_jobs(jobs, path: Path, make_messages, workers: int) -> None:
    done = done_keys(path)
    todo = [j for j in jobs if j["key"] not in done]
    random.Random(SEED + 7).shuffle(todo)  # interleave families, rounds and samples
    print(f"{len(todo)} calls to run ({len(done)} already done) -> {path}")
    spent = 0.0

    def one(j):
        msgs, extra = make_messages(j)
        if msgs is None:
            return {"key": j["key"], "family": j["family"], "run": j["run"], "agent": j["agent"],
                    "send": None, "error": None, "cost_est": extra.get("editor_cost", 0.0), **extra}
        res = replay(j, msgs)
        return {"key": j["key"], "run": j["run"], "agent": j["agent"], "family": j["family"],
                "round": j["round"], "size": j["size"], "sample": j.get("sample"),
                "original_send": parse_send(j["original"]), **extra, **res,
                "cost_est": call_cost(j["family"], res["usage"] or {}) + extra.get("editor_cost", 0.0)}

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for k, rec in enumerate(ex.map(one, todo), 1):
            append(path, rec)
            spent += rec.get("cost_est", 0.0)
            if k % 20 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} done, est ${spent:.2f} this session", flush=True)


def summary() -> None:
    import pandas as pd
    for name in ("noise", "refusal"):
        p = OUT / f"{name}.jsonl"
        if not p.exists():
            continue
        d = pd.DataFrame([json.loads(x) for x in p.read_text().splitlines() if x.strip()])
        print(f"\n== {name}: {len(d)} calls, est ${d.get('cost_est', pd.Series(dtype=float)).sum():.2f}")
        print("errors:", int(d["error"].notna().sum()), "| parse failures:", int((d["error"].isna() & d["send"].isna()).sum()))
        if name == "noise":
            g = d.dropna(subset=["send"]).groupby(["family", "round", "run", "agent"])
            ctx = g["send"].agg(["mean", "std", "count"]).join(g["original_send"].first())
            ctx["orig_in_range"] = [(o is not None) and (o == o) and abs(o - m) <= max(s, 0.5)
                                    for o, m, s in zip(ctx["original_send"], ctx["mean"], ctx["std"].fillna(0))]
            print(ctx.round(2).to_string())
            print("\nwithin-context sd of send, by family x round (pooled):")
            print(ctx.groupby(["family", "round"])["std"].agg(lambda s: (s.pow(2).mean()) ** 0.5).round(2).to_string())
            out = ctx.reset_index()
            out.to_csv(OUT / "noise_by_context.csv", index=False)
        else:
            print(d[["run", "agent", "stated_before", "changed_frac", "original_send", "send", "finish_reason", "error"]].to_string())


# ----------------------------------------------------------------------------- pilot 2: edit pipeline

AMOUNTS = [0, 1, 3, 5]
WORD = {0: "none of them", 1: "one of the five", 3: "three of the five", 5: "all five"}
RULE = "\n\nAnd so the elders taught: whoever holds five should send {w}."
EDIT_PROMPT_V2 = """Below is a short myth from a trust game, where a giver holds five and decides how many to send.
Change ONLY the amount the giver sends (or should send) so that it becomes {word} ({amount} of five).
You may also change numbers that follow directly from that amount (for example the tripled amount
the receiver gets) so the story stays consistent. Do not change anything else, do not add new
sentences, and do not add dollar signs or phrases such as "out of five".
If the myth does not state a specific amount the giver sends, reply exactly NO_AMOUNT.
Return only the full rewritten myth (or NO_AMOUNT), nothing else.

MYTH:
{myth}"""
P2_PER_FAMILY = 16


def named_senders(runs, fam: str, rng) -> list[dict]:
    import pandas as pd
    rules = pd.read_csv(ROOT / "data/analysis/linguistic_20260923/myth_rules_september_z-ai__glm-5.2.csv")
    r1 = rules[(rules["round"] == 1) & (rules["task_order"] == "myth_game") & rules["send_amount"].notna()]
    amt = {(row.run_id, row.agent): row.send_amount for row in r1.itertuples()}
    pool = []
    for r in (r for r in runs if r["family"] == fam):
        for c in sender_calls(r, 1):
            k = (Path(r["path"]).stem, c["agent"])
            if k in amt and len(c["messages"]) == 4 and c["messages"][2]["role"] == "assistant":
                pool.append({"run": r["path"], "family": fam, "round": 1, "size": r["size"],
                             "plan": r["data"]["run_metadata"]["llm_request"], "stated": amt[k], **c})
    rng.shuffle(pool)
    return pool[:P2_PER_FAMILY]


def editor_v2(myth: str, amount: int) -> tuple[str, float]:
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    r = client.chat.completions.create(
        model=EDITOR_MODEL, temperature=0, max_tokens=1500,
        messages=[{"role": "user", "content": EDIT_PROMPT_V2.format(word=WORD[amount], amount=amount, myth=myth)}],
        extra_body={"reasoning": {"enabled": False}, "usage": {"include": True}})
    u = r.usage
    cost = getattr(u, "cost", None)
    if cost is None:
        cost = u.prompt_tokens * EDITOR_PRICE[0] / 1e6 + u.completion_tokens * EDITOR_PRICE[1] / 1e6
    return (r.choices[0].message.content or "").strip(), float(cost)


_judge = None


def judged_amount(text: str) -> tuple[float | None, float]:
    global _judge
    sys.path.insert(0, str(ROOT / "analyses"))
    from myth_moral_judge import Judge
    from myth_rule_judge import parse as rparse, render as rrender
    if _judge is None:
        _judge = Judge(EDITOR_MODEL)
    res = _judge(rrender(text), lambda raw: rparse(raw)[1] == "ok")
    obj, status = rparse(res.get("raw", ""))
    return (obj["send_amount"] if obj else None), float(res.get("cost") or 0)


def plan_pilot2(runs) -> list[dict]:
    jobs = []
    for i, fam in enumerate(("Sonnet", "GPT", "Gemini")):
        for c in named_senders(runs, fam, random.Random(SEED + 10 + i)):
            for a in AMOUNTS:
                for mode in ("rule", "natural"):
                    jobs.append({**c, "amount": a, "mode": mode, "key": f"{c['run']}|{c['agent']}|r1|{mode}{a}"})
    return jobs


def make_pilot2(j):
    myth = j["messages"][2]["content"]
    extra = {"mode": j["mode"], "edited_amount": float(j["amount"]), "stated_before": j["stated"]}
    if j["mode"] == "rule":
        new, ecost = myth.rstrip() + RULE.format(w=WORD[j["amount"]]), 0.0
    else:
        new, ecost = editor_v2(myth, j["amount"])
    extra["editor_cost"] = ecost
    extra["edited_myth"] = new
    if new.strip() == "NO_AMOUNT":
        return None, {**extra, "check": "no_amount"}
    frac = changed_fraction(myth, new)
    extra["changed_frac"] = round(frac, 3)
    if j["mode"] == "natural":
        added = [w for w in ("$", "out of five", "out of 5") if w in new and w not in myth]
        if frac > 0.15 or added:
            return None, {**extra, "check": f"too_much_changed:{frac:.2f}" + (f" added:{added}" if added else "")}
    got, jcost = judged_amount(new)
    extra["judged_amount"] = got
    extra["editor_cost"] = ecost + jcost
    if got is None or abs(got - j["amount"]) > 0.01:
        return None, {**extra, "check": f"judge_reads:{got}"}
    msgs = [dict(m) for m in j["messages"]]
    msgs[2] = {**msgs[2], "content": new}
    return msgs, {**extra, "check": "pass"}


def summary2() -> None:
    import pandas as pd
    p = OUT / "pilot2.jsonl"
    d = pd.DataFrame([json.loads(x) for x in p.read_text().splitlines() if x.strip()])
    d["check_kind"] = d["check"].str.split(":").str[0]
    print(f"== pilot2: {len(d)} edited myths, est ${d['cost_est'].fillna(0).sum() + d.loc[d['send'].isna() if 'send' in d else [], 'editor_cost'].fillna(0).sum():.2f}")
    print("\nedit check pass rate by family x mode:")
    print(d.groupby(["family", "mode"])["check_kind"].value_counts().unstack(fill_value=0).to_string())
    rep = d[d["check"] == "pass"]
    print("\nreplays: errors", int(rep["error"].notna().sum()), "| parse failures", int((rep["error"].isna() & rep["send"].isna()).sum()),
          "| refusals", int((rep.get("finish_reason") == "refusal").sum()))
    print("\nmean send by edited amount (1 sample per edit; descriptive only):")
    print(rep.groupby(["family", "mode", "edited_amount"])["send"].agg(["mean", "count"]).round(2).unstack("edited_amount").to_string())


# ----------------------------------------------------------------------------- pilot 3: intent-to-treat design

AMOUNTS3 = [1, 2, 3, 5]
WORD.update({2: "two of the five"})
P3_PER_FAMILY = 16


def all_senders(runs, fam: str, rng) -> list[dict]:
    """Every round-1 myth->game sender of a family (homogeneous runs), named amount or not."""
    import pandas as pd
    rules = pd.read_csv(ROOT / "data/analysis/linguistic_20260923/myth_rules_september_z-ai__glm-5.2.csv")
    r1 = rules[(rules["round"] == 1) & (rules["task_order"] == "myth_game")]
    amt = {(row.run_id, row.agent): row.send_amount for row in r1.itertuples()}
    pool = []
    for r in (r for r in runs if r["family"] == fam):
        for c in sender_calls(r, 1):
            if len(c["messages"]) == 4 and c["messages"][2]["role"] == "assistant":
                stated = amt.get((Path(r["path"]).stem, c["agent"]))
                pool.append({"run": r["path"], "family": fam, "round": 1, "size": r["size"],
                             "plan": r["data"]["run_metadata"]["llm_request"],
                             "stated": None if stated != stated else stated, **c})
    rng.shuffle(pool)
    return pool


def plan_pilot3(runs) -> list[dict]:
    jobs = []
    for i, fam in enumerate(("Sonnet", "GPT", "Gemini")):
        ctxs = all_senders(runs, fam, random.Random(SEED + 20 + i))[:P3_PER_FAMILY]
        for c in ctxs:
            modes = ["rule"] + (["natural"] if fam == "Sonnet" and c["stated"] is not None else [])
            for mode in modes:
                for a in AMOUNTS3:
                    jobs.append({**c, "amount": a, "mode": mode,
                                 "key": f"{c['run']}|{c['agent']}|r1|p3|{mode}{a}"})
    return jobs


def make_pilot3(j):
    """Intent to treat: every edit that exists is replayed; the judge reading is recorded as a
    manipulation check, not a gate. Only edits that could not be made (editor NO_AMOUNT, or a
    natural edit that changed too much or added '$' / 'out of five') are not replayed."""
    myth = j["messages"][2]["content"]
    base = {"family": j["family"], "run": j["run"], "agent": j["agent"], "round": 1, "size": j["size"],
            "mode": j["mode"], "edited_amount": float(j["amount"]), "stated_before": j["stated"]}
    if j["mode"] == "rule":
        new, ecost = myth.rstrip() + RULE.format(w=WORD[j["amount"]]), 0.0
    else:
        new, ecost = editor_v2(myth, j["amount"])
    base.update(edited_myth=new, editor_cost=ecost)
    if new.strip() == "NO_AMOUNT":
        return None, {**base, "made": False, "check": "no_amount"}
    frac = changed_fraction(myth, new)
    base["changed_frac"] = round(frac, 3)
    if j["mode"] == "natural":
        added = [w for w in ("$", "out of five", "out of 5") if w in new and w not in myth]
        if frac > 0.15 or added:
            return None, {**base, "made": False, "check": "too_much_changed"}
    got, jcost = judged_amount(new)
    base.update(judged_amount=got, editor_cost=ecost + jcost, made=True,
                check="reads_intended" if got is not None and abs(got - j["amount"]) < 0.01 else f"reads:{got}")
    msgs = [dict(m) for m in j["messages"]]
    msgs[2] = {**msgs[2], "content": new}
    return msgs, base


def summary3() -> None:
    import pandas as pd
    d = pd.DataFrame([json.loads(x) for x in (OUT / "pilot3.jsonl").read_text().splitlines() if x.strip()])
    total = d["cost_est"].fillna(0).sum() + d.loc[d["made"] == False, "editor_cost"].fillna(0).sum()  # noqa: E712
    print(f"== pilot3: {len(d)} edits, est ${total:.2f}")
    print("contexts per family:", d.groupby("family")[["run", "agent"]].apply(lambda g: len(g.drop_duplicates())).to_dict())
    made = d.groupby(["family", "mode"])["made"].agg(["sum", "count"])
    made["made_rate"] = (made["sum"] / made["count"]).round(2)
    print("\n1) edits made:\n" + made.to_string())
    m = d[d["made"] == True]  # noqa: E712
    chk = m.assign(ok=m["check"] == "reads_intended").groupby(["family", "mode", "edited_amount"])["ok"].mean().unstack().round(2)
    chk["all"] = m.assign(ok=m["check"] == "reads_intended").groupby(["family", "mode"])["ok"].mean().round(2)
    print("\n2) judge reads the intended amount (share of made edits):\n" + chk.to_string())
    bad = int(m["error"].notna().sum()) + int((m["error"].isna() & m["send"].isna()).sum())
    refus = int((m["finish_reason"] == "refusal").sum())
    print(f"\n3) replays: {len(m)} | errors+parse failures {bad} | refusals {refus} | rate {(bad + refus) / max(len(m), 1):.1%}")
    bal = m.groupby(["family", "mode", "run", "agent"])["edited_amount"].nunique()
    print(f"\n4) contexts replayed at all 4 amounts: {(bal == 4).sum()} of {len(bal)}")
    print("\nmean send by edited amount, all made edits (intent to treat; 1 sample; descriptive only):")
    print(m.groupby(["family", "mode", "edited_amount"])["send"].mean().unstack().round(2).to_string())


# ----------------------------------------------------------------------------- pilot 4: arms B and C, mixed runs

SHOWN_PREFIX = "Here is the myth the other agent wrote in the previous round:\n"
SHOWN_SUFFIX = "\n\nWrite your own myth."
MYTH_PROMPTS = ("Write a 200-word myth", "Write your own myth")
P4_PER_FAMILY = 12


def load_all_runs(data_root: Path) -> list[dict]:
    """All 156 no-defector September myth runs, homogeneous and mixed, with a per-agent plan."""
    runs = []
    for size in (2, 8):
        for spec in run_list(size, data_root):
            d = json.loads(Path(spec["abs_path"]).read_text())
            req = d["run_metadata"]["llm_request"]
            plans = req["agents"] if req.get("provider") == "mixed" else {a: req for a in d["agents"]}
            runs.append({**spec, "data": d, "plans": plans, "mixed": req.get("provider") == "mixed"})
    return runs


def locate_own(messages) -> int | None:
    """Index of the agent's most recent own myth: an assistant reply to a myth-writing prompt."""
    idx = None
    for i in range(1, len(messages)):
        prev = messages[i - 1]
        if messages[i]["role"] == "assistant" and prev["role"] == "user" and any(k in prev["content"] for k in MYTH_PROMPTS):
            idx = i
    return idx


def locate_shown(messages) -> int | None:
    idx = None
    for i, m in enumerate(messages):
        if m["role"] == "user" and m["content"].startswith(SHOWN_PREFIX) and SHOWN_SUFFIX in m["content"]:
            idx = i
    return idx


def arm_contexts(runs, fam: str, arm: str, rounds=(2, 3)) -> list[dict]:
    """arm B: 8-agent myth_game senders with a shown myth in context; arm C: game_myth senders
    whose own myth (written after the previous game) is in context."""
    import pandas as pd
    rules = pd.read_csv(ROOT / "data/analysis/linguistic_20260923/myth_rules_september_z-ai__glm-5.2.csv")
    amt = {(row.run_id, row.agent, row.round): row.send_amount for row in rules.itertuples()}
    myths = pd.read_csv(ROOT / "data/analysis/linguistic_20260923/myths.csv", usecols=["run_id", "round", "agent", "exposed_author", "exposed_round"])
    expo = {(r.run_id, r.agent, r.round): (r.exposed_author, r.exposed_round) for r in myths.itertuples()}
    order, size = ("myth_game", 8) if arm == "B" else ("game_myth", None)
    out = []
    for r in runs:
        if r["task_order"] != order or (size and r["size"] != size):
            continue
        run_id = Path(r["path"]).stem
        for agent, a in r["data"]["agents"].items():
            plan = r["plans"][agent]
            if family_of(plan["model"]) != fam:
                continue
            for x in a["interaction_history"]:
                m = x.get("metadata") or {}
                if not (m.get("task") == "game" and m.get("role") == "investor" and m.get("round") in rounds):
                    continue
                msgs, rnd = x["messages_sent"], m["round"]
                if arm == "B":
                    i = locate_shown(msgs)
                    # the shown myth in context for round rnd is the one shown before this round's myth
                    src = expo.get((run_id, agent, rnd))
                    stated = amt.get((run_id, src[0], src[1])) if src and src[0] == src[0] else None
                else:
                    i = locate_own(msgs)
                    stated = amt.get((run_id, agent, rnd - 1))
                if i is None:
                    continue
                out.append({"run": r["path"], "family": fam, "round": rnd, "size": r["size"], "mixed": r["mixed"],
                            "plan": plan, "agent": agent, "messages": msgs, "slot": i, "arm": arm,
                            "temperature": x.get("temperature", "default"),
                            "original": (x.get("response") or {}).get("content", ""),
                            "stated": None if stated is None or stated != stated else stated})
    return out


def split_slot(ctx) -> tuple[str, str, str]:
    """(prefix, myth, suffix) of the message holding the myth to edit."""
    t = ctx["messages"][ctx["slot"]]["content"]
    if ctx["arm"] == "B":
        j = t.index(SHOWN_SUFFIX)
        return SHOWN_PREFIX, t[len(SHOWN_PREFIX):j], t[j:]
    return "", t, ""


def plan_pilot4(runs) -> tuple[list[dict], dict]:
    counts, jobs = {}, []
    for i, fam in enumerate(("Sonnet", "GPT", "Gemini")):
        for arm in ("B", "C"):
            pool = arm_contexts(runs, fam, arm)
            counts[(fam, arm)] = {"all": len(pool), "homogeneous": sum(not c["mixed"] for c in pool),
                                  "round2": sum(c["round"] == 2 for c in pool),
                                  "named": sum(c["stated"] is not None for c in pool)}
            rng = random.Random(SEED + 40 + i * 2 + (arm == "C"))
            by_run = {}
            for c in pool:
                by_run.setdefault(c["run"], []).append(c)
            keys = list(by_run)
            rng.shuffle(keys)
            pick = [rng.choice(by_run[k]) for k in keys][:P4_PER_FAMILY]  # one context per run
            for c in pick:
                modes = ["rule"] + (["natural"] if fam == "Sonnet" and c["stated"] is not None else [])
                for mode in modes:
                    for a in AMOUNTS3:
                        jobs.append({**c, "amount": a, "mode": mode,
                                     "key": f"{c['run']}|{c['agent']}|r{c['round']}|p4{arm}|{mode}{a}"})
    return jobs, counts


def make_pilot4(j):
    pre, myth, suf = split_slot(j)
    base = {"family": j["family"], "run": j["run"], "agent": j["agent"], "round": j["round"], "size": j["size"],
            "arm": j["arm"], "mixed": j["mixed"], "mode": j["mode"], "edited_amount": float(j["amount"]),
            "stated_before": j["stated"]}
    if j["mode"] == "rule":
        new, ecost = myth.rstrip() + RULE.format(w=WORD[j["amount"]]), 0.0
    else:
        new, ecost = editor_v2(myth, j["amount"])
    base.update(edited_myth=new, editor_cost=ecost)
    if new.strip() == "NO_AMOUNT":
        return None, {**base, "made": False, "check": "no_amount"}
    frac = changed_fraction(myth, new)
    base["changed_frac"] = round(frac, 3)
    if j["mode"] == "natural":
        added = [w for w in ("$", "out of five", "out of 5") if w in new and w not in myth]
        if frac > 0.15 or added:
            return None, {**base, "made": False, "check": "too_much_changed"}
    msgs = [dict(m) for m in j["messages"]]
    msgs[j["slot"]] = {**msgs[j["slot"]], "content": pre + new + suf}
    # placement check: only the intended message changed, and its frame text is intact
    diff = [k for k, (a, b) in enumerate(zip(j["messages"], msgs)) if a["content"] != b["content"]]
    unchanged = new == myth  # the myth already states the target amount: a valid, identical version
    base["unchanged_same_amount"] = unchanged
    assert diff == ([] if unchanged else [j["slot"]]), (j["key"], diff)
    assert msgs[j["slot"]]["content"].startswith(pre) and msgs[j["slot"]]["content"].endswith(suf), j["key"]
    got, jcost = judged_amount(new)
    base.update(judged_amount=got, editor_cost=ecost + jcost, made=True,
                check="reads_intended" if got is not None and abs(got - j["amount"]) < 0.01 else f"reads:{got}")
    return msgs, base


def summary4() -> None:
    import pandas as pd
    d = pd.DataFrame([json.loads(x) for x in (OUT / "pilot4.jsonl").read_text().splitlines() if x.strip()])
    total = d["cost_est"].fillna(0).sum()
    print(f"== pilot4: {len(d)} edits, est ${total:.2f}")
    print("contexts:", d.groupby(["family", "arm"])[["run", "agent", "round"]].apply(lambda g: len(g.drop_duplicates())).to_dict())
    made = d.groupby(["family", "arm", "mode"])["made"].agg(["sum", "count"])
    made["made_rate"] = (made["sum"] / made["count"]).round(2)
    print("\n1) edits made:\n" + made.to_string())
    m = d[d["made"] == True]  # noqa: E712
    ok = m.assign(ok=m["check"] == "reads_intended")
    chk = ok.groupby(["family", "arm", "mode", "edited_amount"])["ok"].mean().unstack().round(2)
    chk["all"] = ok.groupby(["family", "arm", "mode"])["ok"].mean().round(2)
    print("\n2) judge reads the intended amount:\n" + chk.to_string())
    bad = int(m["error"].notna().sum()) + int((m["error"].isna() & m["send"].isna()).sum())
    refus = int((m["finish_reason"] == "refusal").sum())
    print(f"\n3) replays: {len(m)} | errors+parse failures {bad} | refusals {refus} | rate {(bad + refus) / max(len(m), 1):.1%}")
    bal = m.groupby(["family", "arm", "mode", "run", "agent", "round"])["edited_amount"].nunique()
    print(f"\n4) contexts replayed at all 4 amounts: {(bal == 4).sum()} of {len(bal)}")
    print("\nmean send by edited amount (intent to treat; 1 sample; descriptive only):")
    print(m.groupby(["family", "arm", "mode", "edited_amount"])["send"].mean().unstack().round(2).to_string())
    print("\nmixed-run share of replays:", round(m["mixed"].mean(), 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["noise", "refusal", "pilot2", "pilot3", "pilot4"])
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(args.data_root / ".env")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.summary:
        return {"pilot2": summary2, "pilot3": summary3, "pilot4": summary4}.get(args.stage, summary)()
    if args.stage == "pilot4":
        runs = load_all_runs(args.data_root)
        jobs, counts = plan_pilot4(runs)
        print("available contexts (rounds 2-3), per family x arm:")
        for k, v in counts.items():
            print("  ", k, v)
        tok = {"Sonnet": (1960, 860), "GPT": (1330, 6320), "Gemini": (1510, 13)}
        est = sum(call_cost(j["family"], {"input_tokens": tok[j["family"]][0], "output_tokens": tok[j["family"]][1]}) for j in jobs)
        est += sum(j["mode"] == "natural" for j in jobs) * 0.0012 + len(jobs) * 0.0008
        fams = {f: sum(j["family"] == f for j in jobs) for f in ("Sonnet", "GPT", "Gemini")}
        print(f"MODEL=sonnet-4.5,gpt-5-nano,gemini-3.7-flash(+{EDITOR_MODEL} editor+judge) N={len(jobs)} {fams} "
              f"WORKERS={args.workers} EST_COST=${est:.2f}")
        if args.preflight:
            return None
        return run_jobs(jobs, OUT / "pilot4.jsonl", make_pilot4, args.workers)
    runs = load_runs(args.data_root)
    print(f"homogeneous no-defector myth->game runs: {len(runs)} "
          f"{ {f: sum(r['family'] == f for r in runs) for f in ('Sonnet', 'GPT', 'Gemini')} }")
    if args.stage in ("pilot2", "pilot3"):
        jobs = plan_pilot2(runs) if args.stage == "pilot2" else plan_pilot3(runs)
        fams = {f: sum(j["family"] == f for j in jobs) for f in ("Sonnet", "GPT", "Gemini")}
        tok = {"Sonnet": (870, 530), "GPT": (640, 3980), "Gemini": (760, 11)}
        est = sum(call_cost(j["family"], {"input_tokens": tok[j["family"]][0], "output_tokens": tok[j["family"]][1]}) for j in jobs)
        n_nat = sum(j["mode"] == "natural" for j in jobs)
        est += n_nat * 0.0012 + len(jobs) * 0.0008  # GLM editor (natural only) + GLM judge check (all)
        print(f"MODEL=sonnet-4.5,gpt-5-nano,gemini-3.7-flash(+{EDITOR_MODEL} editor+judge) N={len(jobs)} {fams} "
              f"WORKERS={args.workers} EST_COST=${est:.2f} (upper bound: replays only for edits that pass)")
        if args.preflight:
            return None
        maker = make_pilot2 if args.stage == "pilot2" else make_pilot3
        return run_jobs(jobs, OUT / f"{args.stage}.jsonl", maker, args.workers)
    if args.preflight or not args.stage:
        return preflight(runs)
    if args.stage == "noise":
        jobs = plan_noise(runs, random.Random(SEED))
        run_jobs(jobs, OUT / "noise.jsonl", lambda j: (j["messages"], {}), args.workers)
    else:
        jobs = plan_refusal(runs, random.Random(SEED + 1))

        def edited(j):
            myth = j["messages"][2]["content"]
            new, cost = edit_myth(myth, "$1")
            frac = changed_fraction(myth, new)
            extra = {"edited_amount": 1.0, "stated_before": None, "changed_frac": round(frac, 3),
                     "editor_cost": cost, "edited_myth": new}
            if frac > 0.25 or len(new.split()) < 0.6 * len(myth.split()):
                return None, {**extra, "error": "edit rejected: too much changed"}
            msgs = [dict(m) for m in j["messages"]]
            msgs[2] = {**msgs[2], "content": new}
            return msgs, extra
        run_jobs(jobs, OUT / "refusal.jsonl", edited, args.workers)


if __name__ == "__main__":
    main()
