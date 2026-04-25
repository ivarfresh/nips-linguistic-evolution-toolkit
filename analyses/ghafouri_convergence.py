#!/usr/bin/env python3
"""Ghafouri-Ferrara convergence + selective-survival measures on myth chains.

Adapts the three findings of Ghafouri & Ferrara (2026) "Lost Before Translation"
to the iterated myth-writing setup:

1. Convergence: per myth, a judge rates
     - certainty (1-7): 1 = hedged/tentative, 7 = confident/asserted
     - emotional_intensity (1-7): 1 = muted/analytical, 7 = vivid/affective
     - perspectival_balance (1-7): 1 = single-voice/monological, 7 = multi-voice/polyphonic
   Convergence hypothesis: variance across myths within a cell shrinks over rounds.

2. Selective survival: per myth, the judge extracts up to 5 key_entities
   (named characters, places, objects that carry narrative weight) and up to
   5 key_motifs (themes/images recurring in the myth). Persistence curves
   measure what fraction of round-1 entities/motifs still appear at round t
   within the same dyad.

3. Competitive filtering: not measured here directly — downstream, you can
   test whether agent-A motifs fixate more than agent-B motifs when they
   compete in the same dyad. Output CSV has per-agent entity lists ready
   for that analysis.

Pilot with --limit-files 2 before scaling. Cache persists across runs.

Example:
    python analyses/ghafouri_convergence.py \
        --input-root data/json/noise_experiments/v2_uniform_distribution_noise/noise_negative_mem3/claude-sonnet-4.5 \
        --output-dir data/plots/noise_experiments/_ghafouri/pilot \
        --limit-files 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _llm_judge import judge, stats  # noqa: E402


JUDGE_SYSTEM = """You rate a short myth written by an LLM agent along three dimensions, then extract its narrative anchors. Respond with strict JSON only, no prose.

Rating dimensions (adapted from Ghafouri & Ferrara 2026 "Lost Before Translation"):
- certainty (1-7): 1 = heavily hedged, tentative, speculative ("perhaps", "some say", "it is told"). 7 = confident, asserted, categorical.
- emotional_intensity (1-7): 1 = muted, analytical, flat affect. 7 = vivid, affectively charged, sensory.
- perspectival_balance (1-7): 1 = single-voice monologue, one point of view. 7 = multi-voice / polyphonic / multiple named perspectives dialogue.

Narrative-anchor extraction:
- key_entities: up to 5 named characters, named places, or named objects that carry narrative weight. Prefer proper names. Use the exact spelling from the myth.
- key_motifs: up to 5 short phrases (2-5 words each) capturing recurring themes / images / moral content.

Schema:
{
  "certainty": <integer 1-7>,
  "emotional_intensity": <integer 1-7>,
  "perspectival_balance": <integer 1-7>,
  "key_entities": ["...", ...],
  "key_motifs": ["...", ...],
  "short_rationale": "<one sentence>"
}"""


def build_user_prompt(myth_text: str) -> str:
    return f"""MYTH:
\"\"\"
{myth_text}
\"\"\"

Respond with strict JSON only."""


def iter_valid_jsons(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.json")
        if ".checkpoint" not in p.name and ".results" not in p.name
    )


def collect_myth_tasks(path: Path, max_rounds: int | None) -> list[dict]:
    """Extract all (round, agent, text, metadata) entries needing a judge call."""
    with open(path) as f:
        data = json.load(f)
    meta = data.get("run_metadata", {}) or {}
    task_order = data.get("task_order") or []
    history = data.get("conversation_history", []) or []
    tasks = []
    for entry in history:
        r = entry.get("round")
        if max_rounds is not None and r > max_rounds:
            break
        myths = entry.get("myths") or {}
        for agent, text in myths.items():
            if not isinstance(text, str) or not text.strip():
                continue
            tasks.append({
                "path": str(path),
                "model": meta.get("model"),
                "task_order": "->".join(task_order) if isinstance(task_order, list) else str(task_order),
                "game_params": meta.get("game_params_name"),
                "myth_topic": meta.get("myth_topic"),
                "round": r,
                "agent": agent,
                "text": text,
            })
    return tasks


def rate_task(task: dict, judge_model: str) -> dict:
    """Judge a single myth task. Returns a row dict."""
    user = build_user_prompt(task["text"])
    result = judge(JUDGE_SYSTEM, user, model=judge_model, temperature=0.0)
    if not isinstance(result, dict):
        result = {}
    row = {k: v for k, v in task.items() if k != "text"}
    row.update({
        "certainty": result.get("certainty"),
        "emotional_intensity": result.get("emotional_intensity"),
        "perspectival_balance": result.get("perspectival_balance"),
        "key_entities": json.dumps(result.get("key_entities", []) or []),
        "key_motifs": json.dumps(result.get("key_motifs", []) or []),
        "rationale": result.get("short_rationale", ""),
        "parse_error": result.get("_parse_error"),
    })
    return row


def rate_tasks_parallel(
    tasks: list[dict],
    judge_model: str,
    workers: int = 8,
    on_progress=None,
) -> list[dict]:
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(rate_task, t, judge_model): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows.append(fut.result())
            except Exception as e:  # noqa: BLE001
                t = futs[fut]
                print(f"  ! {t['path']}::r{t['round']}::{t['agent']}: {type(e).__name__}: {e}", flush=True)
            if on_progress is not None:
                on_progress(i, len(tasks))
    # Sort by path, round, agent for stable output.
    rows.sort(key=lambda r: (r.get("path", ""), r.get("round", 0), r.get("agent", "")))
    return rows


def plot_convergence(df: pd.DataFrame, output_dir: Path) -> None:
    """Convergence: std of each dimension across (dyad × agent) within a cell, vs round."""
    dims = ["certainty", "emotional_intensity", "perspectival_balance"]
    ddf = df.dropna(subset=["model", "task_order", "game_params", "round"]).copy()
    if ddf.empty:
        return

    cells = ddf.groupby(["model", "task_order", "game_params"])
    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4), sharey=False)
    palette = plt.cm.tab10.colors

    for ax, dim in zip(axes, dims):
        for i, (cell, sub) in enumerate(cells):
            grp = sub.dropna(subset=[dim]).groupby("round")[dim].agg(["mean", "std", "count"]).reset_index()
            if grp.empty:
                continue
            label = " / ".join(str(c) for c in cell)
            ax.plot(grp["round"], grp["std"], marker="o", color=palette[i % 10], label=label)
        ax.set_title(f"Std of {dim} across myths (per round)", fontsize=10)
        ax.set_xlabel("Round")
        ax.set_ylabel(f"Std({dim})")
        ax.grid(alpha=0.3)

    axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Ghafouri convergence: does variance collapse across rounds?", fontweight="bold")
    plt.tight_layout()
    out = output_dir / "convergence_std_vs_round.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot: {out}")

    # Also plot means over round
    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4), sharey=False)
    for ax, dim in zip(axes, dims):
        for i, (cell, sub) in enumerate(cells):
            grp = sub.dropna(subset=[dim]).groupby("round")[dim].agg(["mean", "std", "count"]).reset_index()
            if grp.empty:
                continue
            label = " / ".join(str(c) for c in cell)
            ax.plot(grp["round"], grp["mean"], marker="o", color=palette[i % 10], label=label)
            ax.fill_between(
                grp["round"],
                grp["mean"] - grp["std"] / np.sqrt(grp["count"]),
                grp["mean"] + grp["std"] / np.sqrt(grp["count"]),
                alpha=0.15, color=palette[i % 10],
            )
        ax.set_title(f"Mean {dim} (per round)", fontsize=10)
        ax.set_xlabel("Round")
        ax.set_ylabel(f"Mean({dim})")
        ax.grid(alpha=0.3)
    axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Ghafouri convergence: do means drift toward shared defaults?", fontweight="bold")
    plt.tight_layout()
    out = output_dir / "convergence_mean_vs_round.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot: {out}")


def _norm(s: str) -> str:
    return s.strip().lower()


def compute_survival(df: pd.DataFrame) -> pd.DataFrame:
    """Per (run × agent), for each round t, compute fraction of round-1 entities that persist.

    Returns long-format DF with one row per (run, agent, round, kind) and a
    'persistence_rate' column. kind in {entities, motifs}.
    """
    rows = []
    for (path, agent), sub in df.sort_values("round").groupby(["path", "agent"]):
        first = sub[sub["round"] == sub["round"].min()]
        if first.empty:
            continue
        try:
            first_ents = {_norm(x) for x in json.loads(first.iloc[0]["key_entities"] or "[]") if isinstance(x, str) and x.strip()}
            first_mots = {_norm(x) for x in json.loads(first.iloc[0]["key_motifs"] or "[]") if isinstance(x, str) and x.strip()}
        except Exception:
            continue

        for _, r in sub.iterrows():
            try:
                ents = {_norm(x) for x in json.loads(r["key_entities"] or "[]") if isinstance(x, str) and x.strip()}
                mots = {_norm(x) for x in json.loads(r["key_motifs"] or "[]") if isinstance(x, str) and x.strip()}
            except Exception:
                continue
            ent_rate = len(first_ents & ents) / max(1, len(first_ents))
            mot_rate = len(first_mots & mots) / max(1, len(first_mots))
            base = {
                "path": path, "agent": agent, "round": r["round"],
                "model": r["model"], "task_order": r["task_order"],
                "game_params": r["game_params"],
            }
            rows.append({**base, "kind": "entities", "persistence_rate": ent_rate, "n_anchor": len(first_ents)})
            rows.append({**base, "kind": "motifs",   "persistence_rate": mot_rate, "n_anchor": len(first_mots)})
    return pd.DataFrame(rows)


def plot_survival(surv: pd.DataFrame, output_dir: Path) -> None:
    if surv.empty:
        return
    palette = plt.cm.tab10.colors
    for kind in ["entities", "motifs"]:
        sub = surv[surv["kind"] == kind]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        cells = sub.groupby(["model", "task_order", "game_params"])
        for i, (cell, s) in enumerate(cells):
            grp = s.groupby("round")["persistence_rate"].agg(["mean", "std", "count"]).reset_index()
            label = " / ".join(str(c) for c in cell)
            ax.plot(grp["round"], grp["mean"], marker="o", color=palette[i % 10], label=label)
            ax.fill_between(
                grp["round"],
                grp["mean"] - grp["std"] / np.sqrt(grp["count"]),
                grp["mean"] + grp["std"] / np.sqrt(grp["count"]),
                alpha=0.15, color=palette[i % 10],
            )
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Round")
        ax.set_ylabel(f"Fraction of round-1 {kind} still present")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
        ax.set_title(f"Selective survival — {kind}", fontweight="bold")
        plt.tight_layout()
        out = output_dir / f"survival_{kind}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ghafouri convergence + selective-survival on myth chains.",
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model", default="anthropic/claude-haiku-4.5")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel worker threads for judge calls (default: 8).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = iter_valid_jsons(Path(args.input_root))
    if args.limit_files is not None:
        files = files[:args.limit_files]
    print(f"Rating myths from {len(files)} files with judge={args.judge_model} (workers={args.workers})", flush=True)

    all_tasks = []
    for p in files:
        try:
            all_tasks.extend(collect_myth_tasks(p, max_rounds=args.max_rounds))
        except Exception as e:  # noqa: BLE001
            print(f"  ! collect {p}: {type(e).__name__}: {e}")
    print(f"Total myth tasks: {len(all_tasks)}", flush=True)

    def progress(i: int, n: int) -> None:
        if i % 10 == 0 or i == n:
            print(f"  [{i}/{n}]  cache_hits={stats().cache_hits}  cost~${stats().est_cost_usd:.2f}", flush=True)

    all_rows = rate_tasks_parallel(all_tasks, args.judge_model, workers=args.workers, on_progress=progress)
    pd.DataFrame(all_rows).to_csv(output_dir / "ghafouri_raw.csv", index=False)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No rows.")
        stats().print_summary()
        return

    df.to_csv(output_dir / "ghafouri_raw.csv", index=False)
    print(f"\nRaw: {output_dir / 'ghafouri_raw.csv'} ({len(df)} rows)")

    # Summary: per (model × task_order × game_params × round): mean, std, count of 3 dims.
    dims = ["certainty", "emotional_intensity", "perspectival_balance"]
    summ = (
        df.dropna(subset=dims)
          .groupby(["model", "task_order", "game_params", "round"])[dims]
          .agg(["mean", "std", "count"])
    )
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(output_dir / "convergence_per_round.csv", index=False)
    print(f"Convergence: {output_dir / 'convergence_per_round.csv'}")

    surv = compute_survival(df)
    surv.to_csv(output_dir / "survival_per_round.csv", index=False)
    print(f"Survival: {output_dir / 'survival_per_round.csv'}")

    plot_convergence(df, output_dir)
    plot_survival(surv, output_dir)

    stats().print_summary()


if __name__ == "__main__":
    main()
