#!/usr/bin/env python3
"""LLM-judge classification of agent reasoning traces per game action.

For each (run × round × agent), extract the reasoning text from `game_responses`
(content + reasoning fields), and classify into one of four mutually-exclusive
labels with quoted evidence:

  - myth_content: references a character, entity, motif, or specific theme
                  from the myth chain.
  - reciprocity:  references partner behaviour, retaliation, trust-building,
                  or reputation.
  - game_math:    references payoff structure, expected value, optimization,
                  round number, or game rules.
  - other:        persona, meta, confusion, none-of-above.

The judge is prompted with the agent's own myth(s) from prior rounds so it can
recognise indirect myth references. Judgements without a quoted `evidence` span
from the input text are discarded.

Pilot strategy: start with --limit-files to check a single cell before scaling.
Cache in data/judge_cache/ survives across runs, so re-running is cheap.

Example:
    # Pilot on 5 Claude negative-noise game_myth runs.
    python analyses/reason_field_coding.py \
        --input-root data/json/noise_experiments/v2_uniform_distribution_noise/noise_negative_mem3/claude-sonnet-4.5/game_myth \
        --output-dir data/plots/noise_experiments/_reason_coding/pilot \
        --limit-files 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _llm_judge import judge, stats  # noqa: E402
_ = judge  # silence unused warning (used inside classify_task)


JUDGE_SYSTEM = """You are coding an LLM agent's reasoning trace written before it took an action in an iterated Trust Game. Your job: classify what the reasoning is PRIMARILY about.

Respond with strict JSON only, no prose. Schema:
{
  "label": "myth_content" | "reciprocity" | "game_math" | "other",
  "evidence": "<ONE SHORT contiguous verbatim span (5-25 words) copy-pasted VERBATIM from the reasoning text — no ellipsis, no paraphrase, no concatenation of multiple fragments>",
  "confidence": <number in [0, 1]>
}

Pick the SHORTEST single sentence or phrase that unambiguously justifies your label. The evidence must be a substring of the input — we verify this programmatically.

Label definitions (pick the one that best matches; if two apply, pick the dominant one):
- myth_content: references a character name, entity, theme, or motif from the provided MYTH CONTEXT, or uses narrative / mythological vocabulary (gods, spirits, heroes, parables, archetypes). Metaphorical or allegorical language about trust counts.
- reciprocity: references the partner's prior behaviour, retaliation, trust-building, reputation, or what signal this action sends to the partner.
- game_math: references numbers, payoffs, multipliers, expected value, optimization, round counts, or the game's formal rules.
- other: persona talk, confusion, instruction-following meta, or none of the above.

If the text is empty, "[encrypted]", or purely a JSON action like {"send": 3}, return label "other" with evidence "" and confidence 0."""


def build_user_prompt(reason_text: str, own_myths: list[str], partner_myths: list[str]) -> str:
    own_block = "\n\n".join(f"[Round {i + 1}] {m}" for i, m in enumerate(own_myths)) if own_myths else "(none yet)"
    partner_block = "\n\n".join(f"[Round {i + 1}] {m}" for i, m in enumerate(partner_myths)) if partner_myths else "(none yet)"
    return f"""REASONING TEXT TO CLASSIFY:
\"\"\"
{reason_text}
\"\"\"

MYTH CONTEXT — agent's own prior myths:
\"\"\"
{own_block}
\"\"\"

MYTH CONTEXT — partner's prior myths:
\"\"\"
{partner_block}
\"\"\"

Respond with JSON only."""


ENCRYPTED_RE = re.compile(r"\[\s*\d+\s*reasoning tokens used.*?encrypted", re.IGNORECASE)


def extract_reason_text(response: dict) -> str:
    """Return the natural-language reasoning from a game_response object.

    Concatenates `reasoning` and a JSON-stripped `content`, discarding encrypted
    placeholders. Strips trailing JSON action blocks from `content`.
    """
    if not response:
        return ""
    reasoning = response.get("reasoning") or ""
    if ENCRYPTED_RE.search(reasoning):
        reasoning = ""
    content = response.get("content") or ""
    # Strip JSON-only lines or objects.
    content_no_json = re.sub(r"```(?:json)?\s*\{[^}]*\}\s*```", "", content, flags=re.DOTALL | re.IGNORECASE)
    content_no_json = re.sub(r"\{[^{}]*\"(?:send|return)\"[^{}]*\}", "", content_no_json, flags=re.DOTALL)
    text = "\n".join([t for t in [reasoning, content_no_json.strip()] if t.strip()])
    return text.strip()


def iter_valid_jsons(root: Path) -> list[Path]:
    out = []
    for p in root.rglob("*.json"):
        if ".checkpoint" in p.name or ".results" in p.name:
            continue
        out.append(p)
    return sorted(out)


def collect_reason_tasks(
    path: Path,
    max_rounds: int | None = None,
) -> list[dict]:
    """Extract all (agent × round × reason_text + context) records from one run."""
    with open(path) as f:
        data = json.load(f)
    meta = data.get("run_metadata", {}) or {}
    task_order = data.get("task_order") or []
    history = data.get("conversation_history", []) or []

    own_myth_log: dict[str, list[str]] = {}
    tasks = []

    for entry in history:
        r = entry.get("round")
        if max_rounds is not None and r > max_rounds:
            break
        myths = entry.get("myths") or {}
        game_responses = entry.get("game_responses") or {}
        actions = entry.get("actions") or {}

        for agent, response in game_responses.items():
            reason_text = extract_reason_text(response)
            own = own_myth_log.get(agent, [])
            partner = next(
                (own_myth_log.get(a, []) for a in game_responses.keys() if a != agent),
                [],
            )
            base = {
                "path": str(path), "model": meta.get("model"),
                "task_order": "->".join(task_order) if isinstance(task_order, list) else str(task_order),
                "game_params": meta.get("game_params_name"),
                "myth_topic": meta.get("myth_topic"),
                "round": r, "agent": agent,
                "action": (actions.get(agent, {}) or {}).get("action"),
                "amount": (actions.get(agent, {}) or {}).get("amount"),
                "reason_text": reason_text,
                "own_myths": list(own),
                "partner_myths": list(partner),
            }
            tasks.append(base)

        # After game responses, accumulate this round's myths into log.
        for agent, text in myths.items():
            if isinstance(text, str) and text.strip():
                own_myth_log.setdefault(agent, []).append(text)

    return tasks


def classify_task(task: dict, judge_model: str) -> dict:
    """Judge one reason text. Returns an output row (without context fields)."""
    reason_text = task["reason_text"]
    if not reason_text:
        return {
            **{k: v for k, v in task.items() if k not in ("reason_text", "own_myths", "partner_myths")},
            "label": "no_reasoning", "evidence": "", "confidence": 0.0, "reason_len_chars": 0,
        }

    user = build_user_prompt(reason_text, task["own_myths"], task["partner_myths"])
    result = judge(JUDGE_SYSTEM, user, model=judge_model, temperature=0.0)

    label = result.get("label") if isinstance(result, dict) else None
    evidence = result.get("evidence", "") if isinstance(result, dict) else ""
    confidence = result.get("confidence", 0.0) if isinstance(result, dict) else 0.0

    evidence_ok = False
    if evidence:
        fragments = [f.strip() for f in re.split(r"\.{2,}|…", evidence) if len(f.strip()) >= 6]
        if fragments and all(f in reason_text for f in fragments):
            evidence_ok = True
    if label and label != "other" and not evidence_ok:
        label = "unverified"

    return {
        **{k: v for k, v in task.items() if k not in ("reason_text", "own_myths", "partner_myths")},
        "label": label or "parse_error",
        "evidence": evidence,
        "confidence": float(confidence) if confidence is not None else 0.0,
        "reason_len_chars": len(reason_text),
    }


def classify_tasks_parallel(
    tasks: list[dict],
    judge_model: str,
    workers: int = 8,
    on_progress=None,
) -> list[dict]:
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(classify_task, t, judge_model): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows.append(fut.result())
            except Exception as e:  # noqa: BLE001
                t = futs[fut]
                print(f"  ! {t['path']}::r{t['round']}::{t['agent']}: {type(e).__name__}: {e}", flush=True)
            if on_progress is not None:
                on_progress(i, len(tasks))
    rows.sort(key=lambda r: (r.get("path", ""), r.get("round", 0), r.get("agent", "")))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-judge classification of game-action reasoning traces.",
    )
    parser.add_argument("--input-root", required=True, help="Directory to scan for JSONs (recursive)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model", default="anthropic/claude-haiku-4.5")
    parser.add_argument("--limit-files", type=int, default=None,
                        help="Only classify the first N files (pilot mode).")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="Only classify up to round N per run.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel worker threads for judge calls (default: 8).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = iter_valid_jsons(Path(args.input_root))
    if args.limit_files is not None:
        files = files[:args.limit_files]
    print(f"Classifying {len(files)} files with judge={args.judge_model} (workers={args.workers})", flush=True)

    all_tasks = []
    for p in files:
        try:
            all_tasks.extend(collect_reason_tasks(p, max_rounds=args.max_rounds))
        except Exception as e:  # noqa: BLE001
            print(f"  ! collect {p}: {type(e).__name__}: {e}")
    print(f"Total reason tasks: {len(all_tasks)}", flush=True)

    def progress(i: int, n: int) -> None:
        if i % 20 == 0 or i == n:
            print(f"  [{i}/{n}]  cache_hits={stats().cache_hits}  cost~${stats().est_cost_usd:.2f}", flush=True)

    all_rows = classify_tasks_parallel(all_tasks, args.judge_model, workers=args.workers, on_progress=progress)

    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "reason_labels.csv", index=False)
    print(f"\nLabels: {output_dir / 'reason_labels.csv'}  ({len(df)} rows)")

    # Per-cell label rate summary.
    if not df.empty:
        pivot = (
            df.groupby(["task_order", "game_params", "model", "label"])
              .size().unstack("label", fill_value=0)
        )
        pivot["total"] = pivot.sum(axis=1)
        for col in [c for c in pivot.columns if c != "total"]:
            pivot[f"{col}_pct"] = (pivot[col] / pivot["total"] * 100).round(1)
        pivot_out = output_dir / "reason_label_rates.csv"
        pivot.to_csv(pivot_out)
        print(f"Rates: {pivot_out}")
        print(pivot)

    stats().print_summary()


if __name__ == "__main__":
    main()
