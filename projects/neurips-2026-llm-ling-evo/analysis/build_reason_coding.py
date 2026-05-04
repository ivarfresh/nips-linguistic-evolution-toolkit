#!/usr/bin/env python3
"""Reason-field coding: do agents cite myth content when reasoning about
the game?  (Lexical proxy — not an LLM judge.)

Two complementary tests, per run x agent x round:

  1. Thematic-vocabulary hit:  does the reason text contain any of a small
     fixed myth-language lexicon (e.g. story/myth/legend/spirit/elder)?

  2. Own-myth-vocabulary hit:  does the reason text contain any content
     word (length >= 5, not stopword) that also appears in the agent's
     own most recent myth?  This catches thematic carryover even when
     the lexical surface is novel.  We exclude generic high-frequency
     content words by requiring the word to appear in the agent's own
     myth at least once and to be >= 5 chars.

Aggregates per cell, scoped to v4_direct_provider + Claude/GPT-5-Nano.

Outputs:
  - reason_coding_per_round.csv    (one row per run x agent x round)
  - reason_coding_summary.csv      (per cell: share of reasons matching
                                    each criterion, plus mean overlap
                                    counts)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

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
    "noise_negative_mem3_gpt5_nano": "negative_5",
    "noise_negative_mem3_claude_sonnet_45": "negative_5",
    "noise_positive_mem3_claude_sonnet_45": "positive",
    "noise_positive_mem3_gpt5_nano": "positive",
    "gpt5nano_shared_context_bootstrap": "bootstrap",
    "gpt5nano_shared_context_bootstrap_pilot": "bootstrap",
    "noise_deterministic_max_mem3_gpt5_nano": "deterministic_max",
}

MYTH_THEME_LEXICON = {
    "myth", "story", "legend", "tale", "saga", "fable", "narrative",
    "spirit", "spirits", "elder", "elders", "ancestor", "ancestors",
    "ritual", "sacred", "deity", "deities", "god", "gods", "goddess",
    "divine", "prophecy", "omen", "covenant", "ceremony", "ritualistic",
    "chant", "hymn", "shrine", "temple", "pilgrimage", "wisdom",
    "grandmother", "grandfather", "weave", "weaver", "weaving",
    "telling", "told", "tales", "stories", "myths", "legends",
    "guardian", "guide", "vision", "dream", "blessing", "curse",
    "sage", "oracle", "rune", "prophet",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "was", "are", "were", "been", "be",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "this", "that", "these", "those", "i",
    "you", "he", "she", "it", "we", "they", "them", "their", "his", "her",
    "its", "our", "your", "who", "which", "what", "where", "when", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "just", "now", "then", "there", "than", "into",
    "about", "while", "after", "before", "because", "round", "rounds",
    "agent", "agents", "trustee", "investor", "trustees", "investors",
    "amount", "amounts", "send", "sends", "sent", "return", "returns",
    "returned", "endowment",
}

TOKEN_RE = re.compile(r"\b[A-Za-z]+\b")


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
        "version": version, "experiment": experiment, "model": model,
        "task_order": task_order, "noise_condition": noise_cond,
        "noise_label": noise_label, "informed": informed,
    }


def content_tokens(text: str) -> Set[str]:
    if not text:
        return set()
    return {
        w.lower() for w in TOKEN_RE.findall(text)
        if len(w) >= 5 and w.lower() not in STOPWORDS
    }


def all_tokens(text: str) -> list:
    if not text:
        return []
    return [w.lower() for w in TOKEN_RE.findall(text)]


def collect():
    rows = []
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
        history = data.get("conversation_history", [])
        # Build per-agent myth content vocabulary cumulatively.
        myth_vocab_by_agent: Dict[str, Set[str]] = defaultdict(set)
        # Iterate rounds in order.
        history_sorted = sorted(history, key=lambda e: e.get("round", 0))
        for entry in history_sorted:
            r = entry.get("round")
            game_responses = entry.get("game_responses") or {}
            myths = entry.get("myths") or {}

            # For each agent's game-response prose this round, code against
            # the agent's CURRENT myth vocabulary (before updating).
            for ag, resp in game_responses.items():
                if not resp or not isinstance(resp, dict):
                    continue
                # Concatenate visible content + hidden reasoning text.
                content = resp.get("content") or ""
                hidden = resp.get("reasoning") or ""
                reason = (content + " " + hidden).strip()
                if not reason:
                    continue
                tokens = all_tokens(reason)
                if not tokens:
                    continue
                tok_set = set(tokens)
                # Theme-lexicon hit count.
                theme_hits = [t for t in tokens if t in MYTH_THEME_LEXICON]
                # Own-myth-vocab overlap.
                own_vocab = myth_vocab_by_agent[ag]
                own_hits = [
                    t for t in tokens if t in own_vocab and t not in STOPWORDS
                ]
                rows.append({
                    "path": str(rel),
                    **{k: meta[k] for k in
                       ("model", "task_order", "noise_label", "informed",
                        "experiment")},
                    "agent": ag,
                    "round": r,
                    "reason_n_tokens": len(tokens),
                    "theme_hit_count": len(theme_hits),
                    "theme_hit": len(theme_hits) > 0,
                    "own_myth_hit_count": len(own_hits),
                    "own_myth_hit": len(own_hits) > 0,
                    "own_myth_unique_overlap": len(set(own_hits)),
                })
            # After coding this round's reasons, update myth vocab for next
            # round (so round-1 reasons aren't credited with round-1 myth
            # content).
            for ag, myth in myths.items():
                if myth:
                    myth_vocab_by_agent[ag] |= content_tokens(myth)
    return pd.DataFrame(rows)


def cell_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "noise_label", "informed", "task_order"]
    rows = []
    for key, sub in df.groupby(keys):
        row = dict(zip(keys, key))
        row["n_reasons"] = len(sub)
        row["n_runs"] = sub["path"].nunique()
        row["share_theme_hit"] = float(sub["theme_hit"].mean())
        row["mean_theme_hits"] = float(sub["theme_hit_count"].mean())
        row["share_own_myth_hit"] = float(sub["own_myth_hit"].mean())
        row["mean_own_myth_hits"] = float(sub["own_myth_hit_count"].mean())
        row["mean_own_myth_unique_overlap"] = float(
            sub["own_myth_unique_overlap"].mean()
        )
        # Per-run share with at least one hit.
        per_run_theme = sub.groupby("path")["theme_hit"].any()
        per_run_own = sub.groupby("path")["own_myth_hit"].any()
        row["share_runs_any_theme_hit"] = float(per_run_theme.mean())
        row["share_runs_any_own_myth_hit"] = float(per_run_own.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    print(f"Walking {JSON_ROOT}/noise_experiments/v4_direct_provider/ ...")
    df = collect()
    df.to_csv(OUT_DIR / "reason_coding_per_round.csv", index=False)
    print(f"  wrote {OUT_DIR / 'reason_coding_per_round.csv'} ({len(df)} rows)")
    summary = cell_summary(df)
    summary.to_csv(OUT_DIR / "reason_coding_summary.csv", index=False)
    print(f"  wrote {OUT_DIR / 'reason_coding_summary.csv'} ({len(summary)} cells)")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("\n=== REASON-CODING SUMMARY ===")
    cols = ["model", "noise_label", "informed", "task_order",
            "n_reasons", "n_runs",
            "share_theme_hit", "mean_theme_hits",
            "share_own_myth_hit", "mean_own_myth_hits",
            "mean_own_myth_unique_overlap",
            "share_runs_any_theme_hit", "share_runs_any_own_myth_hit"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
