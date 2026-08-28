#!/usr/bin/env python3
"""Blinded LLM-judge relabeling of meme families for every capsule.

The regex ontology in ``analyze_meme_evolution.py`` measures vocabulary; this
script measures meaning.  A judge model reads each agent response *in
isolation* — no round, condition, agent identity, or hypothesis — and reports
which of the nine strategy ideas the text genuinely expresses.  Labels land in
a JSONL file consumed by ``analyze_meme_transmission_null.py --labels``.

Design constraints:
- The judge is deliberately from a different model family than the corpus
  (corpus: Claude Sonnet 4.5; judge default: openai/gpt-5-mini via OpenRouter)
  so the judge does not share the generator's phrasing priors.
- Temperature 0, JSON-object output, one capsule per call.
- Resumable: capsule ids already present in the output JSONL are skipped.

Usage:
    python scripts/judge_meme_labels.py [INPUT_DIR] [--output DIR] \
        [--model openai/gpt-5-mini] [--workers 8] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

try:
    from scripts.analyze_capsule_genealogies import is_primary_json, load_capsules
    from scripts.analyze_meme_evolution import MEMES
except ModuleNotFoundError:  # Direct execution adds scripts/, not repo root.
    from analyze_capsule_genealogies import is_primary_json, load_capsules  # type: ignore
    from analyze_meme_evolution import MEMES  # type: ignore

load_dotenv(find_dotenv(usecwd=True) or None)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data/share/corrected_informed_noise_confirmatory_60runs_2026-08-12/data"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data/analysis/meme_judge_labels_2026_08_28"
DEFAULT_MODEL = "openai/gpt-5-mini"

FAMILIES = [definition.family for definition in MEMES]

RUBRIC = "\n".join(
    f"- {definition.family}: {definition.description}" for definition in MEMES
)

SYSTEM_PROMPT = f"""You label strategy ideas in short texts. The texts are \
messages written during a repeated economic exchange; some are narrative \
stories, some are decision rationales. Judge ONLY what the text in front of \
you genuinely expresses.

The nine idea labels:
{RUBRIC}

Rules:
- Label an idea as present ONLY when the text expresses the idea itself — a \
stated strategy, principle, or lesson. A passing word association is not \
enough (e.g. the word "noise" alone does not express noise_adaptation; \
"begin" near "trust" does not express trust_seeding).
- Mentioning an amount or event is not a strategy; ignore pure bookkeeping.
- Restating the game's rules or arithmetic is NOT a strategy idea. \
Example for prosperity_through_cooperation: "sending gets tripled, creating \
mutual benefit" is a rules restatement — NOT present; "generosity creates a \
shared abundance from which both of us draw" espouses the idea as a \
principle — present. Apply the same standard to every label.
- Be conservative: when in doubt whether a text merely gestures at an idea \
or genuinely espouses it, leave the label out.
- Ideas may be expressed in mythic/metaphorical language; that counts if the \
underlying strategy idea is clearly there.
- Return strict JSON: {{"present": [<label>, ...], "evidence": {{<label>: \
"<shortest verbatim quote showing it>", ...}}}}. Use [] and {{}} when none \
apply. No other keys, no commentary."""


def judge_one(client: OpenAI, model: str, capsule_id: str, text: str) -> dict:
    last_error = None
    for attempt in range(6):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Text to label:\n\n{text}"},
                ],
            )
            payload = json.loads(response.choices[0].message.content)
            present = [
                family for family in payload.get("present", []) if family in FAMILIES
            ]
            evidence = {
                family: str(quote)[:200]
                for family, quote in (payload.get("evidence") or {}).items()
                if family in present
            }
            usage = response.usage
            return {
                "capsule_id": capsule_id,
                "present": present,
                "evidence": evidence,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
        except Exception as error:  # noqa: BLE001 — retry everything, then surface
            last_error = error
            time.sleep(min(60, 2**attempt) + random.random())
    return {"capsule_id": capsule_id, "error": str(last_error)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Judge only the first N capsules (smoke test)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY missing (see .env.example)")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    run_paths = sorted(
        path for path in args.input_dir.rglob("*.json") if is_primary_json(path)
    )
    capsules = []
    for path in run_paths:
        _data, run_capsules = load_capsules(path, args.input_dir)
        capsules.extend(run_capsules)
    capsules.sort(key=lambda c: c.capsule_id)
    if args.limit:
        capsules = capsules[: args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    labels_path = args.output / "judge_labels.jsonl"
    done: set[str] = set()
    if labels_path.exists():
        with labels_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if "error" not in record:
                    done.add(record["capsule_id"])
    todo = [c for c in capsules if c.capsule_id not in done]

    # $/Mtok (input, output), verified against the OpenRouter models endpoint.
    pricing = {
        "openai/gpt-5-mini": (0.25, 2.00),
        "openai/gpt-4o-mini": (0.15, 0.60),
        "google/gemini-2.5-flash-lite": (0.10, 0.40),
    }
    price_in, price_out = pricing.get(args.model, (0.25, 2.00))
    est_in = len(todo) * 700 / 1e6
    est_out = len(todo) * 300 / 1e6
    est_cost = est_in * price_in + est_out * price_out
    print(
        f"MODEL={args.model} N={len(todo)} (of {len(capsules)}, {len(done)} done) "
        f"WORKERS={args.workers} EST_COST=${est_cost:.2f}"
    )
    if not todo:
        print("Nothing to do.")
        return

    lock = Lock()
    completed = 0
    errors = 0
    with labels_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(judge_one, client, args.model, c.capsule_id, c.text): c
                for c in todo
            }
            for future in as_completed(futures):
                record = future.result()
                with lock:
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    completed += 1
                    if "error" in record:
                        errors += 1
                    if completed % 250 == 0 or completed == len(todo):
                        print(f"{completed}/{len(todo)} judged ({errors} errors)")

    print(f"Done: {completed} judged, {errors} errors -> {labels_path}")


if __name__ == "__main__":
    main()
