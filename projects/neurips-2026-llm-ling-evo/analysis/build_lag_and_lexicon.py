#!/usr/bin/env python3
"""Aggregated cooperativity-lexicon and lag-1 cross-agent analysis.

Scope: this week's runs only — `noise_experiments/v4_direct_provider/`,
Claude + GPT-5-Nano, task orders that include myth (game_myth, myth_game).

For each run, for each agent, score every myth round on a small cooperativity
lexicon (collective + connected + giving) minus an uncooperative lexicon
(individual + disconnected + taking).  Then compute three correlations per
dyad:

  1. lag1_cross_AB  — Agent_A's cooperativity score at round t correlated with
                       Agent_B's score at round t+1.  (Project's "Claude
                       r ≈ 0.72" replication test.)
  2. lag1_cross_BA  — symmetric direction.
  3. same_round_AB  — between-agent same-round correlation (convergence proxy).

Also for runs in `game_myth` / `myth_game` orders, compute the within-agent
cooperativity-vs-game-action correlation: agent's own myth score at round t
vs agent's own `sent` (or `returned`, when trustee) at round t.  Tests
whether myth content tracks own behaviour.

Outputs to `cell_summaries/`:
  - lexicon_per_run.csv           (one row per run x agent x round)
  - lag_correlations.csv          (one row per run, with cross/within rs)
  - lag_summary.csv               (aggregated per model x noise x task_order
                                   with bootstrap 95% CIs on mean r)

Usage:
    python3 build_lag_and_lexicon.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
    "noise_negative_mem3": "negative_5",
    "noise_negative_mem3_gpt5_nano": "negative_5",
    "noise_negative_mem3_claude_sonnet_45": "negative_5",
    "noise_positive_mem3_claude_sonnet_45": "positive",
    "noise_positive_mem3_gemini_3_1_pro": "positive",
    "noise_positive_mem3_gpt5_nano": "positive",
    "gpt5nano_shared_context_bootstrap": "bootstrap",
    "gpt5nano_shared_context_bootstrap_pilot": "bootstrap",
    "noise_deterministic_max_mem3_gpt5_nano": "deterministic_max",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "was", "are", "were", "been", "be", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "can",
    "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "them", "their", "his", "her", "its", "our", "your", "who", "which", "what",
    "where", "when", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "s", "t", "just", "now", "then",
}

# Lifted verbatim from analyses/cooperativity_analysis.py.
COOPERATIVE_WORDS = (
    {"together", "shared", "mutual", "partnership", "community", "collaboration",
     "unity", "collective", "bond", "alliance", "cooperation", "cooperate",
     "collaborative"}
    | {"relationship", "connection", "reciprocity", "trust", "friendship",
       "companion", "ally", "partner", "kindness", "generosity", "compassion",
       "solidarity"}
    | {"give", "giving", "gave", "given", "offered", "offer", "returned",
       "return", "share", "generous", "gift", "bestow", "granted"}
)

UNCOOPERATIVE_WORDS = (
    {"alone", "solitary", "independent", "self", "individual", "isolated",
     "separate", "isolation", "lonely", "solo"}
    | {"betrayal", "division", "conflict", "rivalry", "enemy", "hostility",
       "antagonism", "opposition", "discord"}
    | {"took", "taken", "take", "kept", "keep", "withheld", "withhold",
       "refused", "refuse", "denied", "deny", "hoarded", "hoard"}
)

N_BOOTSTRAP = 2000
RNG = np.random.default_rng(42)


def tokenize(text: str) -> List[str]:
    text = text.lower()
    words = re.findall(r"\b[a-z]+\b", text)
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def coop_score(text: str) -> Tuple[int, int, int]:
    """Return (cooperative_count, uncooperative_count, total_tokens)."""
    if not text:
        return 0, 0, 0
    words = tokenize(text)
    coop = sum(1 for w in words if w in COOPERATIVE_WORDS)
    uncoop = sum(1 for w in words if w in UNCOOPERATIVE_WORDS)
    return coop, uncoop, len(words)


def coop_index(text: str) -> Optional[float]:
    """Net cooperative count per 100 tokens, or None if no tokens."""
    coop, uncoop, total = coop_score(text)
    if total == 0:
        return None
    return 100.0 * (coop - uncoop) / total


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


def collect_per_run() -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_lex = []
    rows_lag = []
    seen = 0
    skipped = 0
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
            skipped += 1
            continue

        # Build per-round score trajectories per agent.
        myths_by_agent: Dict[str, Dict[int, str]] = defaultdict(dict)
        for entry in data.get("conversation_history", []):
            r = entry.get("round")
            for agent_id, myth in (entry.get("myths") or {}).items():
                if myth:
                    myths_by_agent[agent_id][r] = myth

        if len(myths_by_agent) < 2:
            skipped += 1
            continue

        # Lexicon per round, per agent.
        agent_scores: Dict[str, Dict[int, float]] = {}
        for agent_id, rounds in myths_by_agent.items():
            scores = {}
            for r, txt in rounds.items():
                ci = coop_index(txt)
                if ci is None:
                    continue
                scores[r] = ci
                rows_lex.append({
                    "path": str(rel),
                    **{k: meta[k] for k in
                       ("version", "experiment", "model", "task_order",
                        "noise_condition", "noise_label", "informed")},
                    "agent": agent_id,
                    "round": r,
                    "coop_index": ci,
                })
            agent_scores[agent_id] = scores

        # Pick the two main agents by name.
        agent_ids = sorted(agent_scores.keys())[:2]
        if len(agent_ids) < 2:
            continue
        a, b = agent_ids
        rounds_common = sorted(set(agent_scores[a]) & set(agent_scores[b]))
        if len(rounds_common) < 4:
            continue
        sa = np.array([agent_scores[a][r] for r in rounds_common], dtype=float)
        sb = np.array([agent_scores[b][r] for r in rounds_common], dtype=float)

        same_round = pearson(sa, sb)
        lag_ab = pearson(sa[:-1], sb[1:])  # A at t, B at t+1
        lag_ba = pearson(sb[:-1], sa[1:])  # B at t, A at t+1

        # Within-agent: own myth cooperativity at round t vs own `sent` at round t
        # (only meaningful when game and myth both happen each round).
        # Each `conversation_history` entry has `sent` and `returned` from the
        # current investor; agents alternate, so we tag by `actions` if present.
        sent_by_agent: Dict[str, Dict[int, float]] = defaultdict(dict)
        ret_by_agent: Dict[str, Dict[int, float]] = defaultdict(dict)
        for entry in data.get("conversation_history", []):
            r = entry.get("round")
            actions = entry.get("actions") or {}
            for ag, act in actions.items():
                if act is None:
                    continue
                if "sent" in act:
                    sent_by_agent[ag][r] = float(act["sent"])
                if "returned" in act:
                    ret_by_agent[ag][r] = float(act["returned"])

        within_corrs = {}
        for ag in agent_ids:
            shared = sorted(set(agent_scores[ag]) & set(sent_by_agent.get(ag, {})))
            if len(shared) >= 4:
                within_corrs[f"within_sent_{ag}"] = pearson(
                    np.array([agent_scores[ag][r] for r in shared]),
                    np.array([sent_by_agent[ag][r] for r in shared]),
                )

        rows_lag.append({
            "path": str(rel),
            **{k: meta[k] for k in
               ("version", "experiment", "model", "task_order",
                "noise_condition", "noise_label", "informed")},
            "n_rounds": len(rounds_common),
            "lag1_AB": lag_ab,
            "lag1_BA": lag_ba,
            "same_round_AB": same_round,
            **within_corrs,
        })
        seen += 1
    print(f"  scored {seen} runs, skipped {skipped}")
    return pd.DataFrame(rows_lex), pd.DataFrame(rows_lag)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3 or len(a) != len(b):
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def bootstrap_mean_ci(values: np.ndarray, n=N_BOOTSTRAP, alpha=0.05):
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    point = float(np.mean(values))
    samples = np.empty(n)
    for i in range(n):
        idx = RNG.integers(0, len(values), size=len(values))
        samples[i] = np.mean(values[idx])
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return point, lo, hi, len(values)


def lag_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "noise_label", "informed", "task_order"]
    rows = []
    for key, sub in df.groupby(keys):
        row = dict(zip(keys, key))
        row["n_runs"] = len(sub)
        for col in ("lag1_AB", "lag1_BA", "same_round_AB"):
            mean, lo, hi, n_used = bootstrap_mean_ci(sub[col].values)
            row[f"{col}_mean"] = mean
            row[f"{col}_ci_lo"] = lo
            row[f"{col}_ci_hi"] = hi
            row[f"{col}_n"] = n_used
        # max lag1 across the two directions per run (the "Claude r=0.72"
        # finding was on whichever direction was strongest in that dyad).
        sub_max = np.maximum(sub["lag1_AB"].values, sub["lag1_BA"].values)
        mean, lo, hi, n_used = bootstrap_mean_ci(sub_max)
        row["lag1_max_mean"] = mean
        row["lag1_max_ci_lo"] = lo
        row["lag1_max_ci_hi"] = hi
        row["lag1_max_n"] = n_used
        # share of runs with |max_lag| > 0.5
        valid = sub_max[~np.isnan(sub_max)]
        row["share_lag_gt_0_5"] = float(np.mean(np.abs(valid) > 0.5)) if len(valid) > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    print(f"Walking {JSON_ROOT}/noise_experiments/v4_direct_provider/ ...")
    lex_df, lag_df = collect_per_run()

    lex_df.to_csv(OUT_DIR / "lexicon_per_run.csv", index=False)
    print(f"  wrote {OUT_DIR / 'lexicon_per_run.csv'} ({len(lex_df)} rows)")

    lag_df.to_csv(OUT_DIR / "lag_correlations.csv", index=False)
    print(f"  wrote {OUT_DIR / 'lag_correlations.csv'} ({len(lag_df)} rows)")

    summary = lag_summary(lag_df)
    summary.to_csv(OUT_DIR / "lag_summary.csv", index=False)
    print(f"  wrote {OUT_DIR / 'lag_summary.csv'} ({len(summary)} cells)")

    # Console preview.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n=== LAG-1 SUMMARY (mean Pearson r across runs, with bootstrap 95% CI) ===")
    cols = ["model", "noise_label", "informed", "task_order", "n_runs",
            "lag1_AB_mean", "lag1_AB_ci_lo", "lag1_AB_ci_hi",
            "lag1_BA_mean", "lag1_BA_ci_lo", "lag1_BA_ci_hi",
            "same_round_AB_mean", "same_round_AB_ci_lo", "same_round_AB_ci_hi",
            "lag1_max_mean", "lag1_max_ci_lo", "lag1_max_ci_hi",
            "share_lag_gt_0_5"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
