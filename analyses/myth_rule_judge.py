#!/usr/bin/env python3
"""Extract the play rules each myth endorses (how much to send, return, and what to do when let down).

Arabella's three moral labels say whether a myth leans generous, fair or
cautious. This asks the concrete question behind them: which amounts and
responses does the myth tell a player to use? Rubric:
analyses/rubrics/myth_rule_rubric.txt. Fields: send_rule, send_amount,
test_first, return_rule, after_letdown, consistency, noise_mentioned.

Two corpora, same rubric:
  september  every September myth in data/analysis/linguistic_20260923/myths.csv
             (built by analyses/linguistic_corpus.py), >= 20 words
  donors     the texts injected in the slide-678 transplant reruns (8-agent
             20260916 and dyad 20260917), one row per donor text, so the
             extracted rule can be compared with what hosts actually sent

Judge settings and cache are those of analyses/myth_moral_judge.py (GLM-5.2 via
OpenRouter, temperature 0, reasoning off, cached by exact prompt).

  python3 analyses/myth_rule_judge.py --preflight
  python3 analyses/myth_rule_judge.py --sample 100          # pilot on a random sample
  python3 analyses/myth_rule_judge.py                       # full run, both corpora
  python3 analyses/myth_rule_judge.py --model deepseek/deepseek-v4-flash --sample 1000   # second judge
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from myth_moral_judge import DATA, MIN_WORDS, PRICES, ROOT, Judge

RUBRIC = ROOT / "analyses/rubrics/myth_rule_rubric.txt"
TRANSPLANT_ROOTS = {
    8: ROOT / "data/json/noise_experiments/slide678_rerun_20260916",
    2: ROOT / "data/json/noise_experiments/slide678_dyad_rerun_20260917",
}
FIELDS = {
    "send_rule": ["all", "most", "moderate", "little", "none", "unspecified"],
    "return_rule": ["more_than_half", "half", "at_least_sent", "little", "match_partner", "unspecified"],
    "after_letdown": ["keep_trusting", "reduce", "withdraw", "unspecified"],
}
BOOLS = ["test_first", "consistency", "noise_mentioned"]


def render(text: str) -> str:
    return RUBRIC.read_text().replace("{text}", text)


def parse(raw: str) -> tuple[dict | None, str]:
    s = raw.strip()
    try:
        obj = json.loads(s[s.find("{"): s.rfind("}") + 1])
    except Exception:  # noqa: BLE001
        return None, "bad_json"
    if not isinstance(obj, dict):
        return None, "bad_json"
    out = {}
    for f, allowed in FIELDS.items():
        v = str(obj.get(f, "")).strip().lower()
        if v not in allowed:
            return None, f"bad_{f}"
        out[f] = v
    amt = obj.get("send_amount")
    try:
        amt = None if amt in (None, "", "null") else float(amt)
    except (TypeError, ValueError):
        return None, "bad_send_amount"
    if amt is not None and not 0 <= amt <= 5:
        amt = None  # an amount outside the game's range is a misread, not a rule
    out["send_amount"] = amt
    for f in BOOLS:
        v = obj.get(f)
        out[f] = v if isinstance(v, bool) else str(v).strip().lower() == "true"
    return out, "ok"


def september() -> pd.DataFrame:
    myths = pd.read_csv(DATA / "myths.csv")
    myths = myths[myths["n_words"] >= MIN_WORDS]
    return myths[["run_id", "size", "mixed", "composition", "task_order", "round", "agent", "family", "text"]]


def donors() -> pd.DataFrame:
    rows = []
    for size, root in TRANSPLANT_ROOTS.items():
        for c in json.loads((root / "plan.json").read_text())["combos"]:
            if not c["seed_text"]:
                continue
            rows.append({"size": size, "seed_type": c["seed_type"], "rep": int(c["rep"]),
                         "final": str((root / c["seed_type"] / f"rep{int(c['rep']):02d}.json").relative_to(ROOT)),
                         "text": c["seed_text"]})
    return pd.DataFrame(rows)


def run(judge: Judge, frame: pd.DataFrame, workers: int) -> pd.DataFrame:
    prompts = [render(t) for t in frame["text"]]
    valid = lambda raw: parse(raw)[1] == "ok"  # noqa: E731
    results: list[dict | None] = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(judge, p, valid): i for i, p in enumerate(prompts)}
        for n, fut in enumerate(as_completed(futures), 1):
            results[futures[fut]] = fut.result()
            if n % 500 == 0:
                spent = sum(r.get("cost") or 0 for r in results if r and not r.get("cached"))
                print(f"  {n}/{len(prompts)} done, ${spent:.2f} billed this session", flush=True)
    out = frame.drop(columns=["text"]).reset_index(drop=True).copy()
    parsed = [parse(r.get("raw", "")) for r in results]
    for f in [*FIELDS, "send_amount", *BOOLS]:
        out[f] = [p[0][f] if p[0] else None for p in parsed]
    out["status"] = [p[1] if not r.get("error") else "error" for p, r in zip(parsed, results)]
    out["cost"] = [r.get("cost") or 0 for r in results]
    out["cached"] = [bool(r.get("cached")) for r in results]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="z-ai/glm-5.2")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sample", type=int, help="random sample of N September myths (pilot); skips donors")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()

    corpora = {"september": september()}
    if args.sample:
        corpora["september"] = corpora["september"].sample(args.sample, random_state=0)
    else:
        corpora["donors"] = donors()
    tag = args.model.replace("/", "__") + (f"_sample{args.sample}" if args.sample else "")

    pin, pout = PRICES[args.model]
    n = sum(len(f) for f in corpora.values())
    n_in = sum(len(render(t)) for f in corpora.values() for t in f["text"]) / 4 + 40 * n
    est = (n_in / 1e6 * pin + 70 * n / 1e6 * pout) * 1.5
    print(f"MODEL={args.model} N={n} WORKERS={args.workers} EST_COST=${est:.2f} (upper bound; cached prompts are free)")
    if args.preflight:
        return

    judge = Judge(args.model)
    for name, frame in corpora.items():
        out = run(judge, frame, args.workers)
        path = DATA / f"myth_rules_{name}_{tag}.csv"
        out.to_csv(path, index=False)
        bad = (out["status"] != "ok").sum()
        print(f"{name}: {len(out) - bad} ok, {bad} unparsed; ${out.loc[~out['cached'], 'cost'].sum():.2f} billed -> {path}")


if __name__ == "__main__":
    main()
