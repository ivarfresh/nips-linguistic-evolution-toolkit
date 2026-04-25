#!/usr/bin/env python3
"""Post-hoc trait inference on trust-game transcripts.

Adapted from Abdurahman et al. (2026) "Explicit Trait Inference for Multi-Agent
Coordination". The original protocol has agents infer partner warmth and
competence in-the-loop. We apply the same two-dimensional framework post-hoc:
a third-party LLM-judge rates what one agent would plausibly infer about the
partner, given the conversation history up to round t.

Frame in the paper as: "post-hoc adaptation of Abdurahman et al.'s warmth/
competence inference", not an in-the-loop replication.

For each (run × agent × round), we feed the judge:
  - all (sent, returned) amounts the agent has observed by that round
  - optionally, the partner's myths seen so far (if myth task order)
  - the agent's own myths seen so far (for context)
and ask for warmth (1-7) and competence (1-7) Likert ratings with brief
rationale. Both dimensions are 1-7 to match the original ETI Likert convention.

Mediation test (done post-hoc from the output CSV):
  cooperation_delta_model_noise_cell = f(warmth_inferred_delta_model_noise_cell)

Example:
    python analyses/eti_trait_inference.py \
        --input-root data/json/noise_experiments/v2_uniform_distribution_noise/noise_negative_mem3/claude-sonnet-4.5 \
        --output-dir data/plots/noise_experiments/_eti/pilot \
        --limit-files 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _llm_judge import judge, stats  # noqa: E402


JUDGE_SYSTEM = """You are a research assistant rating perceived partner traits in an iterated Trust Game. The game's rules: each round the investor sends some amount (0-5), the trustee receives 3× that, and returns some amount (0 to 3× sent). Roles alternate each round.

You will be given the interaction history (and, if present, any myths written between agents) from ONE agent's perspective. Rate what that agent would plausibly infer about its partner along two dimensions, using a 1-7 Likert scale (the scale from Abdurahman et al. 2026):

- WARMTH: perceived benevolence, trustworthiness, good will, fairness.
  (1 = cold, exploitative, untrustworthy; 7 = warm, generous, trustworthy)
- COMPETENCE: perceived skill, strategic capability, understanding of the game.
  (1 = incompetent, random, confused; 7 = skilled, strategic, sophisticated)

Respond with strict JSON only:
{
  "warmth": <integer 1-7>,
  "competence": <integer 1-7>,
  "warmth_rationale": "<one-sentence reason grounded in the observed behaviour / myth content>",
  "competence_rationale": "<one-sentence reason grounded in the observed behaviour / myth content>"
}

Base your rating only on the evidence provided. If insufficient evidence (e.g. no rounds completed), return 4/4 with rationales "insufficient evidence"."""


def build_history_block(
    game_history: list[dict],
    own_id: str,
    partner_id: str,
    include_myths: bool,
) -> str:
    if not game_history:
        return "(no rounds yet)"

    lines = []
    for entry in game_history:
        r = entry.get("round")
        roles = entry.get("roles", {}) or {}
        sent_c = entry.get("sent_communicated", entry.get("sent"))
        ret_c = entry.get("returned_communicated", entry.get("returned"))
        own_role = roles.get(own_id)
        partner_role = roles.get(partner_id)

        if own_role == "investor":
            lines.append(f"Round {r}: I (investor) sent ${sent_c}; partner (trustee) returned ${ret_c}.")
        else:
            lines.append(f"Round {r}: partner (investor) sent ${sent_c}; I (trustee) returned ${ret_c}.")

        if include_myths:
            myths = entry.get("myths") or {}
            pm = myths.get(partner_id)
            if isinstance(pm, str) and pm.strip():
                preview = pm.strip().replace("\n", " ")
                lines.append(f"  Partner's myth (round {r}): \"{preview[:400]}{'...' if len(preview) > 400 else ''}\"")
    return "\n".join(lines)


def build_user_prompt(history_block: str) -> str:
    return f"""INTERACTION HISTORY FROM THE AGENT'S PERSPECTIVE:
{history_block}

Respond with strict JSON only:
{{"warmth": <1-7>, "competence": <1-7>, "warmth_rationale": "...", "competence_rationale": "..."}}"""


def infer_traits_for_run(
    path: Path,
    judge_model: str,
    max_rounds: int | None,
    rounds_to_rate: set[int] | None = None,
) -> list[dict]:
    with open(path) as f:
        data = json.load(f)

    meta = data.get("run_metadata", {}) or {}
    task_order = data.get("task_order") or []
    history = data.get("conversation_history", []) or []
    include_myths = "myth" in (task_order if isinstance(task_order, list) else [task_order])

    agents = ["Agent_1", "Agent_2"]
    rows = []

    for rnd_idx in range(len(history)):
        r = history[rnd_idx].get("round")
        if max_rounds is not None and r > max_rounds:
            break
        if rounds_to_rate is not None and r not in rounds_to_rate:
            continue

        history_so_far = history[: rnd_idx + 1]

        for own_id in agents:
            partner_id = "Agent_2" if own_id == "Agent_1" else "Agent_1"
            hist_block = build_history_block(history_so_far, own_id, partner_id, include_myths)
            user = build_user_prompt(hist_block)
            result = judge(JUDGE_SYSTEM, user, model=judge_model, temperature=0.0)

            if not isinstance(result, dict):
                result = {}

            rows.append({
                "path": str(path),
                "model": meta.get("model"),
                "task_order": "->".join(task_order) if isinstance(task_order, list) else str(task_order),
                "game_params": meta.get("game_params_name"),
                "myth_topic": meta.get("myth_topic"),
                "round": r,
                "rater_agent": own_id,
                "rated_agent": partner_id,
                "warmth": result.get("warmth"),
                "competence": result.get("competence"),
                "warmth_rationale": result.get("warmth_rationale", ""),
                "competence_rationale": result.get("competence_rationale", ""),
                "parse_error": result.get("_parse_error"),
            })

    return rows


def iter_valid_jsons(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.json")
        if ".checkpoint" not in p.name and ".results" not in p.name
    )


def summarize(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return

    # Per (model × task_order × game_params × round): mean ± std of warmth/competence.
    grouped = (
        df.dropna(subset=["warmth", "competence"])
          .groupby(["model", "task_order", "game_params", "round"])[["warmth", "competence"]]
          .agg(["mean", "std", "count"])
    )
    grouped.columns = [f"{m}_{s}" for m, s in grouped.columns]
    grouped = grouped.reset_index()
    out = output_dir / "eti_per_round.csv"
    grouped.to_csv(out, index=False)
    print(f"Per-round: {out}")

    # Cell-level summary (final-round rating per run).
    final = (
        df.dropna(subset=["warmth", "competence"])
          .sort_values("round")
          .groupby(["path", "model", "task_order", "game_params", "rater_agent"])
          .tail(1)
    )
    cell = final.groupby(["model", "task_order", "game_params"])[["warmth", "competence"]].agg(["mean", "std", "count"])
    cell.columns = [f"{m}_{s}" for m, s in cell.columns]
    cell = cell.reset_index()
    cell_out = output_dir / "eti_cell_summary.csv"
    cell.to_csv(cell_out, index=False)
    print(f"Cell summary: {cell_out}")
    print(cell.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc warmth/competence inference on trust-game transcripts.",
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model", default="anthropic/claude-haiku-4.5")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="Only rate up to round N per run (default: all).")
    parser.add_argument("--rounds-to-rate", type=int, nargs="*", default=None,
                        help="If set, rate only these round indices (e.g. 1 5 10). "
                             "Overrides --max-rounds behaviour; useful to sample cheaply.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = iter_valid_jsons(Path(args.input_root))
    if args.limit_files is not None:
        files = files[:args.limit_files]
    print(f"Inferring traits from {len(files)} files with judge={args.judge_model}")

    rounds_to_rate = set(args.rounds_to_rate) if args.rounds_to_rate else None

    all_rows = []
    for i, p in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {p.name}", flush=True)
        try:
            rows = infer_traits_for_run(
                p, args.judge_model,
                max_rounds=args.max_rounds,
                rounds_to_rate=rounds_to_rate,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  ! error: {type(e).__name__}: {e}")
            continue
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(output_dir / "eti_raw.csv", index=False)

    df = pd.DataFrame(all_rows)
    print(f"\nRated: {len(df)} rows")
    df.to_csv(output_dir / "eti_raw.csv", index=False)
    summarize(df, output_dir)

    stats().print_summary()


if __name__ == "__main__":
    main()
