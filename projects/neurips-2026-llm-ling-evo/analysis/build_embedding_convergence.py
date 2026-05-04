#!/usr/bin/env python3
"""Embedding-based myth convergence and drift analysis.

Scope: this week's runs only — `noise_experiments/v4_direct_provider/`,
Claude + GPT-5-Nano, task orders that include myth.

For each run:
  - between-agent same-round cosine similarity (per round)
  - within-agent drift: cosine(myth at round 1, myth at round t)

Aggregates across runs per (model x noise_label x informed x task_order).

Outputs:
  - embedding_per_run.csv         (one row per run x round)
  - embedding_summary.csv         (per cell, slope of between-agent
                                   similarity over rounds with bootstrap CI)

Uses `all-mpnet-base-v2` (same model as analyses/myth_similarity_embedding.py).
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="urllib3 v2 only supports")

REPO_ROOT = Path(__file__).resolve().parents[3]
JSON_ROOT = REPO_ROOT / "data" / "json"
OUT_DIR = Path(__file__).parent / "cell_summaries"
OUT_DIR.mkdir(exist_ok=True)

INCLUDE_VERSIONS = {
    "v4_direct_provider",
    "v4_direct_provider_A1A3_combined",
    "v4_direct_provider_A1_adversarial_bootstrap",
    "v4_direct_provider_A1_no_noise",
    "v4_direct_provider_A1_partner_myth",
    "v4_direct_provider_shared_context_pilot",
    "v4_direct_provider_shared_context",
    "v4_direct_provider_A1_targeted_bootstrap",
    "v4_direct_provider_A3_forced_reasoning",
    "v4_direct_provider_baseline",
    "v4_direct_provider_targeted_bootstrap",
    "v4_direct_provider_targeted_gpt5nano",
    "v4_direct_provider_targeted_k1_gpt5nano",
    "v4_direct_provider_targeted_k2_gpt5nano",
    "v4_direct_provider_targeted_neutral_gpt5nano",
    "v4_direct_provider_controls",
}
INCLUDE_MODELS = {"claude-sonnet-4.5", "gpt-5-nano"}
TASK_ORDERS_WITH_MYTH = {"game_myth", "myth_game"}

NOISE_LABEL_FROM_EXPERIMENT = {
    "noise_bootstrap_mem3": "bootstrap",
    "noise_negative_mem3_gpt5_nano": "negative_5",
    "noise_negative_mem3_claude_sonnet_45": "negative_5",
    "noise_positive_mem3_claude_sonnet_45": "positive",
    "noise_positive_mem3_gpt5_nano": "positive",
    "gpt5nano_shared_context_bootstrap": "bootstrap",
    "gpt5nano_shared_context_bootstrap_pilot": "bootstrap",
    "noise_deterministic_max_mem3_gpt5_nano": "deterministic_max",
}

EMBED_MODEL = "all-mpnet-base-v2"
N_BOOTSTRAP = 1000
RNG = np.random.default_rng(42)


def parse_relative(rel: Path) -> Optional[dict]:
    parts = rel.parts
    if not parts or parts[0] != "noise_experiments":
        return None
    if len(parts) < 6:
        return None
    version = parts[1]
    experiment = parts[2]
    model = parts[3]
    task_order = parts[4]
    noise_cond = parts[5]
    informed = noise_cond.endswith("_informed")
    noise_label = NOISE_LABEL_FROM_EXPERIMENT.get(experiment, experiment)
    return {
        "version": version,
        "experiment": experiment,
        "model": model,
        "task_order": task_order,
        "noise_condition": noise_cond,
        "noise_label": noise_label,
        "informed": informed,
    }


def collect_runs():
    """Returns list of (rel_path, meta, agent1_myths, agent2_myths) where
    each agent_myths is dict: round -> text.
    """
    runs = []
    for path in JSON_ROOT.rglob("*.json"):
        n = path.name
        if ".checkpoint" in n or ".results" in n or ".error" in n:
            continue
        rel = path.relative_to(JSON_ROOT)
        meta = parse_relative(rel)
        if meta is None:
            continue
        if meta["version"] not in INCLUDE_VERSIONS:
            continue
        if meta["model"] not in INCLUDE_MODELS:
            continue
        if meta["task_order"] not in TASK_ORDERS_WITH_MYTH:
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        myths_by_agent: Dict[str, Dict[int, str]] = defaultdict(dict)
        for entry in data.get("conversation_history", []):
            r = entry.get("round")
            for agent_id, myth in (entry.get("myths") or {}).items():
                if myth:
                    myths_by_agent[agent_id][r] = myth
        if "Agent_1" not in myths_by_agent or "Agent_2" not in myths_by_agent:
            continue
        a1 = myths_by_agent["Agent_1"]
        a2 = myths_by_agent["Agent_2"]
        common = sorted(set(a1) & set(a2))
        if len(common) < 4:
            continue
        runs.append((rel, meta, a1, a2, common))
    return runs


def main():
    print("Loading runs ...")
    runs = collect_runs()
    print(f"  {len(runs)} runs to embed")

    # Batch-encode all myths.
    print(f"Loading sentence-transformer model {EMBED_MODEL} ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    print("  model loaded")

    # Flatten texts for batch encoding.
    flat_texts = []
    flat_keys = []  # (run_idx, agent, round)
    for run_idx, (_, _, a1, a2, common) in enumerate(runs):
        for r in common:
            flat_texts.append(a1[r])
            flat_keys.append((run_idx, "Agent_1", r))
            flat_texts.append(a2[r])
            flat_keys.append((run_idx, "Agent_2", r))
    print(f"Encoding {len(flat_texts)} myth texts ...")
    embeddings = model.encode(
        flat_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"  encoded shape={embeddings.shape}")

    # Reshape into per-run, per-agent vectors.
    by_run: Dict[int, Dict[str, Dict[int, np.ndarray]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (run_idx, agent, r), vec in zip(flat_keys, embeddings):
        by_run[run_idx][agent][r] = vec

    # Compute per-run, per-round between-agent cosine + within-agent drift.
    rows = []
    for run_idx, (rel, meta, _, _, common) in enumerate(runs):
        emb_a = by_run[run_idx]["Agent_1"]
        emb_b = by_run[run_idx]["Agent_2"]
        first_a = emb_a[common[0]]
        first_b = emb_b[common[0]]
        for r in common:
            cos_ab = float(np.dot(emb_a[r], emb_b[r]))
            drift_a = float(np.dot(emb_a[r], first_a))
            drift_b = float(np.dot(emb_b[r], first_b))
            rows.append({
                "path": str(rel),
                **{k: meta[k] for k in
                   ("model", "task_order", "noise_label", "informed",
                    "experiment")},
                "round": r,
                "cos_between": cos_ab,
                "drift_a": drift_a,
                "drift_b": drift_b,
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "embedding_per_run.csv", index=False)
    print(f"  wrote {OUT_DIR / 'embedding_per_run.csv'} ({len(df)} rows)")

    # Per-cell aggregate: slope of cos_between over rounds, plus per-round means.
    keys = ["model", "noise_label", "informed", "task_order"]
    summary_rows = []
    for key, sub in df.groupby(keys):
        # Per-round mean cos_between.
        per_round = sub.groupby("round")["cos_between"].mean()
        # Slope of cos_between vs round (across runs, per round).
        # Use linear regression on (round, cos_between) raw observations.
        x = sub["round"].values.astype(float)
        y = sub["cos_between"].values.astype(float)
        if len(x) >= 4 and np.std(x) > 0:
            slope = float(np.polyfit(x, y, 1)[0])
            # Bootstrap.
            slopes = np.empty(N_BOOTSTRAP)
            for i in range(N_BOOTSTRAP):
                idx = RNG.integers(0, len(x), size=len(x))
                if np.std(x[idx]) == 0:
                    slopes[i] = np.nan
                    continue
                slopes[i] = np.polyfit(x[idx], y[idx], 1)[0]
            slopes = slopes[~np.isnan(slopes)]
            slope_lo = float(np.percentile(slopes, 2.5)) if len(slopes) else float("nan")
            slope_hi = float(np.percentile(slopes, 97.5)) if len(slopes) else float("nan")
        else:
            slope = slope_lo = slope_hi = float("nan")
        row = dict(zip(keys, key))
        row.update({
            "n_runs": sub["path"].nunique(),
            "n_obs": len(sub),
            "cos_between_round1": float(per_round.iloc[0]) if len(per_round) else float("nan"),
            "cos_between_roundN": float(per_round.iloc[-1]) if len(per_round) else float("nan"),
            "cos_between_slope_per_round": slope,
            "cos_between_slope_ci_lo": slope_lo,
            "cos_between_slope_ci_hi": slope_hi,
            "drift_a_round1": float(sub[sub["round"] == sub["round"].min()]["drift_a"].mean()),
            "drift_a_roundN": float(sub[sub["round"] == sub["round"].max()]["drift_a"].mean()),
            "drift_b_round1": float(sub[sub["round"] == sub["round"].min()]["drift_b"].mean()),
            "drift_b_roundN": float(sub[sub["round"] == sub["round"].max()]["drift_b"].mean()),
        })
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "embedding_summary.csv", index=False)
    print(f"  wrote {OUT_DIR / 'embedding_summary.csv'} ({len(summary)} cells)")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n=== EMBEDDING CONVERGENCE SUMMARY ===")
    cols = ["model", "noise_label", "informed", "task_order", "n_runs",
            "cos_between_round1", "cos_between_roundN",
            "cos_between_slope_per_round", "cos_between_slope_ci_lo",
            "cos_between_slope_ci_hi",
            "drift_a_round1", "drift_a_roundN"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
