"""LLM-judge cooperativeness scorer for Phase 2 seed myths.

Scores each distinct seed myth in the manifest on a 0–10 scale of
"how strongly does this myth advocate/exemplify cooperation, generosity,
reciprocity, or trust between agents." Output is added to the manifest.

Supports multi-judge scoring: pass `--label <name>` to store scores under
`seed["judge_scores"][<label>]` so a second judge model can be run without
clobbering the first. The legacy `judge_score` / `judge_model` fields are
preserved for backward compatibility and treated as the first-judge result
when this script runs without `--label`.

Uses the configured Anthropic-direct or OpenRouter client. Default judge
model is Sonnet 4.5 for consistency with the host runs.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import call_llm, create_llm_client


JUDGE_PROMPT = """You will read a short text and score how strongly it advocates or exemplifies cooperation, generosity, reciprocity, or trust between agents.

Score on a 0–10 scale:
- 0: actively anti-cooperative (defection, betrayal, distrust as normative)
- 5: neutral or mixed
- 10: strongly pro-cooperative (cooperation/generosity/reciprocity as central value)

Respond with exactly one integer 0–10 in JSON: {{"score": <int>}}

Text to score:
\"\"\"
{text}
\"\"\""""


def score_myth(client, model, text):
    messages = [{"role": "user", "content": JUDGE_PROMPT.format(text=text)}]
    response = call_llm(client, model, 0.0, messages, max_retries=2)
    content = response.get("content", "")
    m = re.search(r'"score"\s*:\s*(\d+)', content)
    if not m:
        m = re.search(r"\b([0-9]|10)\b", content)
    if not m:
        return None
    return int(m.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--judge-model",
        default="anthropic/claude-sonnet-4.5",
        help="Repo model slug for the judge",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Label for this judge under seed['judge_scores'][label]. "
        "If omitted, writes to the legacy seed['judge_score'].",
    )
    parser.add_argument("--out", default=None, help="Output path (default: overwrite manifest)")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    client = create_llm_client(args.judge_model)

    for seed_type, seeds in manifest.get("seeds", {}).items():
        for seed in seeds:
            text = seed.get("text")
            if not text:
                continue
            scores = seed.setdefault("judge_scores", {})
            if args.label is None:
                if "judge_score" in seed:
                    # Legacy mode: backfill into judge_scores under the
                    # legacy model name, then skip rescoring.
                    legacy_model = seed.get("judge_model", "legacy")
                    legacy_label = legacy_model.split("/")[-1].replace(".", "_")
                    scores.setdefault(legacy_label, seed["judge_score"])
                    continue
                score = score_myth(client, args.judge_model, text)
                seed["judge_score"] = score
                seed["judge_model"] = args.judge_model
                judge_label = args.judge_model.split("/")[-1].replace(".", "_")
                scores[judge_label] = score
                print(f"{seed_type}: score={score} text='{text[:60]}...'")
            else:
                if args.label in scores:
                    print(f"{seed_type}: already scored as {args.label}={scores[args.label]} — skip")
                    continue
                score = score_myth(client, args.judge_model, text)
                scores[args.label] = score
                print(f"{seed_type} [{args.label}]: score={score} text='{text[:60]}...'")

    out_path = args.out or args.manifest
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
