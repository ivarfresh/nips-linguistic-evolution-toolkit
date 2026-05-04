#!/usr/bin/env python3
"""Neologism / coinage analysis on myth chains.

Scope: this week's runs only — `noise_experiments/v4_direct_provider/`,
Claude + GPT-5-Nano, task orders that include myth.

Detection heuristic for a "candidate coinage":
  - alphabetic, length >= 6
  - not in /usr/share/dict/words (case-insensitive)
  - not a standard English plural / -ing / -ed / -s / -er / -es / -ly
    of a dictionary word
  - not appearing in a small list of common proper-noun stems often
    used in myths (e.g. "amaranth", "sylph", "zephyr")

For each chain (run x agent x rounds_with_myth):
  - distinct_coinages: count of unique coinages across the agent's chain
  - max_persistence: max number of rounds any single coinage persists
  - rare_share: share of tokens that are coinages

For each run as a whole, also check whether any coinage from a myth chain
appears in any game `reason` field — the "linguistic content leaks into
game reasoning" observation.

Outputs:
  - neologisms_per_run.csv      (one row per run x agent)
  - neologism_summary.csv       (per cell: mean coinages, persistence,
                                  share of runs with at least one
                                  coinage-in-reason)
  - neologisms_examples.csv     (concrete coinages for spot-checking)
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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

DICT_PATH = Path("/usr/share/dict/words")
COMMON_MYTH_STEMS = {
    "trickster", "trickery", "amaranth", "sylph", "zephyr", "kindred",
    "wanderer", "wisp", "elder", "lore", "tribute", "covenant", "lament",
    "mortal", "immortal", "sentinel", "dweller", "vessel",
}
SUFFIXES = ("s", "es", "ed", "d", "ing", "er", "est", "ly", "ies", "ied",
            "ness", "ment", "ful", "less", "ous", "ish", "able", "ible")
# Irregular past tenses & common forms not always in the system dictionary.
EXTRA_VOCAB = {
    "became", "begun", "thought", "brought", "caught", "fought", "taught",
    "sought", "wrought", "spoken", "broken", "chosen", "frozen", "stolen",
    "wrote", "bore", "tore", "swore", "wove", "drove", "strove",
    "arose", "rose", "shone", "shine", "shaken", "shook", "saw", "seen",
    "knelt", "leapt", "dwelt", "spelt", "dealt", "felt",
    "myth", "myths", "agent", "agents", "round", "rounds",
    "investor", "trustee", "trustees", "investors", "endowment", "trust",
    "cooperate", "cooperated", "cooperating", "cooperator",
    "reciprocate", "reciprocated", "reciprocating", "reciprocity",
}


def load_dictionary() -> Set[str]:
    words = set()
    if DICT_PATH.exists():
        with DICT_PATH.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                w = line.strip().lower()
                if w:
                    words.add(w)
    return words


def is_coinage(token: str, dictionary: Set[str]) -> bool:
    if len(token) < 6:
        return False
    if not token.isalpha():
        return False
    t = token.lower()
    if t in dictionary or t in EXTRA_VOCAB or t in COMMON_MYTH_STEMS:
        return False
    # Strip common suffixes and re-check.
    for s in SUFFIXES:
        if t.endswith(s) and len(t) - len(s) >= 3:
            stem = t[: -len(s)]
            if (stem in dictionary or stem in EXTRA_VOCAB
                    or stem in COMMON_MYTH_STEMS):
                return False
            # add "e" back: "shar" + "e" = "share"
            if stem + "e" in dictionary:
                return False
            # double-letter rule: "running" -> "runn" -> "run"
            if (len(stem) >= 3 and stem[-1] == stem[-2]
                    and stem[:-1] in dictionary):
                return False
    # "ied" -> "y": replied -> reply
    if t.endswith("ied") and len(t) >= 5:
        if (t[:-3] + "y") in dictionary:
            return False
    # "ies" -> "y": stories -> story
    if t.endswith("ies") and len(t) >= 5:
        if (t[:-3] + "y") in dictionary:
            return False
    return True


TOKEN_RE = re.compile(r"\b[A-Za-z]+\b")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text or "")


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


def collect_per_run(dictionary: Set[str]):
    per_run_rows = []
    examples_rows = []
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
        myths_by_agent: Dict[str, Dict[int, str]] = defaultdict(dict)
        reasons_text = []
        for entry in history:
            r = entry.get("round")
            for agent_id, myth in (entry.get("myths") or {}).items():
                if myth:
                    myths_by_agent[agent_id][r] = myth
            actions = entry.get("actions") or {}
            for ag, act in actions.items():
                if act and isinstance(act, dict):
                    reason = act.get("reason")
                    if isinstance(reason, str):
                        reasons_text.append(reason)
        if not myths_by_agent:
            continue
        all_reasons_tokens = set(tokenize(" ".join(reasons_text)))
        all_reasons_lower = {t.lower() for t in all_reasons_tokens}

        for agent_id, rounds in myths_by_agent.items():
            tokens_per_round: Dict[int, List[str]] = {}
            coinage_per_round: Dict[int, List[str]] = {}
            all_tokens = []
            for r, txt in rounds.items():
                tks = tokenize(txt)
                tokens_per_round[r] = tks
                all_tokens.extend(tks)
                coinage_per_round[r] = [
                    t for t in tks if is_coinage(t, dictionary)
                ]
            distinct_coinages = sorted({
                c.lower()
                for cs in coinage_per_round.values()
                for c in cs
            })
            persistence = Counter()
            for cs in coinage_per_round.values():
                for c in {c.lower() for c in cs}:
                    persistence[c] += 1
            max_persistence = max(persistence.values()) if persistence else 0
            n_rounds_with_coinage = sum(
                1 for cs in coinage_per_round.values() if cs
            )
            n_tokens = len(all_tokens)
            rare_share = (
                sum(len(cs) for cs in coinage_per_round.values()) / n_tokens
                if n_tokens > 0
                else 0.0
            )
            # Check if any coinage appears in this run's game `reason` text.
            coinages_in_reasons = sorted(
                set(distinct_coinages) & all_reasons_lower
            )

            per_run_rows.append({
                "path": str(rel),
                **{k: meta[k] for k in
                   ("model", "task_order", "noise_label", "informed",
                    "experiment")},
                "agent": agent_id,
                "n_rounds_with_myth": len(rounds),
                "total_tokens": n_tokens,
                "n_distinct_coinages": len(distinct_coinages),
                "max_persistence": max_persistence,
                "n_rounds_with_coinage": n_rounds_with_coinage,
                "rare_share_pct": 100.0 * rare_share,
                "n_coinages_in_reasons": len(coinages_in_reasons),
                "any_coinage_in_reasons": len(coinages_in_reasons) > 0,
            })

            for c, n_persist in persistence.most_common(5):
                examples_rows.append({
                    "path": str(rel),
                    "model": meta["model"],
                    "noise_label": meta["noise_label"],
                    "informed": meta["informed"],
                    "task_order": meta["task_order"],
                    "agent": agent_id,
                    "coinage": c,
                    "rounds_persisted": n_persist,
                    "in_reasons": c in all_reasons_lower,
                })

    return pd.DataFrame(per_run_rows), pd.DataFrame(examples_rows)


def cell_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "noise_label", "informed", "task_order"]
    rows = []
    for key, sub in df.groupby(keys):
        row = dict(zip(keys, key))
        row["n_chains"] = len(sub)
        row["n_runs"] = sub["path"].nunique()
        for col, label in [
            ("n_distinct_coinages", "coinages"),
            ("max_persistence", "max_persistence"),
            ("rare_share_pct", "rare_share_pct"),
        ]:
            row[f"{label}_mean"] = float(sub[col].mean())
            row[f"{label}_median"] = float(sub[col].median())
        # Share of runs with at least one coinage that reappears in `reason`.
        per_run = sub.groupby("path")["any_coinage_in_reasons"].any()
        row["share_runs_coinage_in_reasons"] = float(per_run.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    dictionary = load_dictionary()
    print(f"Loaded {len(dictionary)} dictionary words")
    print(f"Walking {JSON_ROOT}/noise_experiments/v4_direct_provider/ ...")
    per_run, examples = collect_per_run(dictionary)
    per_run.to_csv(OUT_DIR / "neologisms_per_run.csv", index=False)
    print(f"  wrote {OUT_DIR / 'neologisms_per_run.csv'} ({len(per_run)} rows)")
    examples.to_csv(OUT_DIR / "neologisms_examples.csv", index=False)
    print(f"  wrote {OUT_DIR / 'neologisms_examples.csv'} ({len(examples)} rows)")
    summary = cell_summary(per_run)
    summary.to_csv(OUT_DIR / "neologism_summary.csv", index=False)
    print(f"  wrote {OUT_DIR / 'neologism_summary.csv'} ({len(summary)} cells)")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n=== NEOLOGISM SUMMARY (per cell) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("\n=== TOP 30 LONGEST-PERSISTING COINAGES ===")
    if not examples.empty:
        top = examples.sort_values("rounds_persisted", ascending=False).head(30)
        print(top[["model", "noise_label", "informed", "task_order", "agent",
                   "coinage", "rounds_persisted", "in_reasons"]].to_string(
                       index=False))


if __name__ == "__main__":
    main()
