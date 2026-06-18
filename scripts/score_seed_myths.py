#!/usr/bin/env python3
"""LLM-judge scoring of seed myths + fillers for H5 graded analysis.

Loads the harvest manifest, scores each seed text on a 0-10 cooperativeness
scale via Claude Sonnet 4.5, and writes results to
data/seeds/ablation_phase1_judge_scores.json.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import create_llm_client, call_llm

MANIFEST = ROOT / "data" / "seeds" / "ablation_phase1_manifest.json"
FILLERS = ROOT / "data" / "seeds" / "fillers.json"
OUT = ROOT / "data" / "seeds" / "ablation_phase1_judge_scores.json"

JUDGE_MODEL = "anthropic/claude-sonnet-4.5"
JUDGE_TEMP = 0.0  # deterministic-ish

JUDGE_SYSTEM = """You are a careful, calibrated rater of short texts.

You will be shown a short story or paragraph. Your task is to score how strongly
the text conveys themes of COOPERATION between individuals, groups, or forces —
including trust, reciprocity, mutual aid, collective benefit, generosity, fair
exchange, or pro-social behavior.

Use this 0-10 scale:
  0  = no cooperative themes; or the text is explicitly anti-cooperative
       (selfishness, betrayal, exploitation, defection presented approvingly)
  1-2 = minimal or incidental cooperative themes
  3-4 = some cooperative elements but not central
  5-6 = cooperation is one of several themes, moderately present
  7-8 = cooperation is a central theme, clearly presented as valuable
  9-10 = cooperation is the explicit moral/lesson of the text, strongly
         endorsed

Be calibrated. Most texts should land in the middle of the scale; reserve
the extremes for clear-cut cases. Score what the text SAYS, not what you
infer the writer's intent was."""

JUDGE_USER_TEMPLATE = """Score the following text on the 0-10 cooperativeness scale described.

TEXT:
\"\"\"
{text}
\"\"\"

Respond ONLY with a JSON object of the form:
{{"score": <int 0-10>, "rationale": "<1-2 sentence justification>"}}"""


def score_one(client, text):
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER_TEMPLATE.format(text=text)},
    ]
    resp = call_llm(client, JUDGE_MODEL, JUDGE_TEMP, messages)
    content = resp["content"]
    # Extract JSON
    import re
    m = re.search(r"\{[^{}]*\"score\"[^{}]*\}", content, re.DOTALL)
    if not m:
        return {"score": None, "rationale": f"PARSE_FAIL: {content[:200]}"}
    try:
        return json.loads(m.group(0))
    except Exception as e:
        return {"score": None, "rationale": f"PARSE_FAIL: {e}: {content[:200]}"}


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    with open(FILLERS) as f:
        fillers = json.load(f)

    client = create_llm_client(JUDGE_MODEL)

    results = {}

    # Score seeds
    for pool_name, seeds in manifest["pools"].items():
        for seed in seeds:
            seed_id = seed["seed_id"]
            print(f"  scoring {seed_id}...", flush=True)
            results[seed_id] = {
                "pool": pool_name,
                "source_joint_balance": seed.get("source_joint_balance"),
                "token_count": seed.get("token_count"),
                **score_one(client, seed["myth_text"]),
            }

    # Score fillers
    for filler in fillers.get("fillers", fillers if isinstance(fillers, list) else []):
        fid = filler.get("filler_id") or filler.get("id") or filler.get("slug")
        text = filler.get("text") or filler.get("content")
        print(f"  scoring {fid}...", flush=True)
        results[fid] = {
            "pool": "filler",
            "token_count": len(text.split()) if text else None,
            **score_one(client, text),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT}")
    print(f"n scored: {len(results)}")

    # Quick summary
    from collections import defaultdict
    by_pool = defaultdict(list)
    for k, v in results.items():
        if v.get("score") is not None:
            by_pool[v["pool"]].append(v["score"])
    print("\nMean score per pool:")
    for pool, scores in sorted(by_pool.items()):
        import statistics as st
        print(f"  {pool:14s}: mean={st.mean(scores):.2f}  values={sorted(scores)}")


if __name__ == "__main__":
    main()
