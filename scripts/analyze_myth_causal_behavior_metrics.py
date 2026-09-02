#!/usr/bin/env python3
"""Behavioral diagnostics for myth-causal trust-game runs.

This script computes run-level and condition-level metrics beyond final balance:
- mean sent / returned
- first low-trust collapse round
- zero-exchange spiral probability
- recovery after low-trust episodes
- simple myth/game semantic-coupling proxies
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, stdev


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_DIRS = [
    PROJECT_ROOT / "data/json/myth_causal_confirm_claude_fixed_prompt_r10/claude-sonnet-4.5",
    PROJECT_ROOT / "data/json/myth_causal_confirm_claude_fixed_prompt_r10_directive_retry/claude-sonnet-4.5",
    PROJECT_ROOT
    / "data/json/noise_experiments/myth_causal_negative2_informed_claude_r10"
    / "myth_causal_noise_negative2_informed_claude_r10/claude-sonnet-4.5",
]

CONDITION_LABELS = {
    "game_only": "Game Only",
    "myth_control": "Myth Control",
    "myth_game_directive": "Myth-Game Directive",
}

NOISE_LABELS = {
    "clean": "No Noise",
    "noise_negative2_informed": "Informed Negative U(0,2)",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "who",
    "with",
    "would",
    "you",
}

GAME_CONCEPT_PATTERN = re.compile(
    r"\b("
    r"trust|trusted|trusting|trustworthy|faith|risk|risky|"
    r"return|returned|returning|send|sent|sending|give|gave|given|gift|offering|"
    r"reciprocity|reciprocal|reciprocate|cooperate|cooperation|cooperative|"
    r"betray|betrayed|betrayal|exploit|exploited|exploitation|defect|defection|"
    r"withhold|withheld|withholding|repair|restore|rebuild|fair|fairness|"
    r"punish|punishment|retaliate|retaliation|consequence|norm|promise|covenant|"
    r"exchange|balance|investor|trustee|payoff|round"
    r")\b",
    re.IGNORECASE,
)

WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z']+")


def is_primary_json(path: Path) -> bool:
    return (
        path.suffix == ".json"
        and not path.name.endswith(".results.json")
        and not path.name.endswith(".checkpoint.json")
        and not path.name.endswith(".error.json")
    )


def condition_for(data: dict) -> str:
    task_order = "_".join(data.get("task_order") or [])
    if task_order == "game":
        return "game_only"
    return data.get("run_metadata", {}).get("myth_prompt_arm_id") or "unknown"


def noise_for(path: Path) -> str:
    if "noise_experiments" in path.parts:
        return "noise_negative2_informed"
    return "clean"


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in WORD_PATTERN.findall(text)
        if len(token) > 2 and token.lower() not in STOPWORDS
    ]


def cosine_from_counters(left: Counter[str], right: Counter[str]) -> float | None:
    if not left or not right:
        return None
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return None
    return dot / (left_norm * right_norm)


def keyword_density(text: str) -> float | None:
    words = WORD_PATTERN.findall(text)
    if not words:
        return None
    return len(GAME_CONCEPT_PATTERN.findall(text)) / len(words) * 100.0


def extract_texts(history: list[dict]) -> tuple[str, str]:
    myth_parts: list[str] = []
    game_parts: list[str] = []
    for row in history:
        for myth in (row.get("myths") or {}).values():
            myth_parts.append(str(myth))
        for response in (row.get("game_responses") or {}).values():
            content = response.get("content")
            if content:
                game_parts.append(str(content))
    return "\n\n".join(myth_parts), "\n\n".join(game_parts)


def max_consecutive(flags: list[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def summarize_values(values: list[float | int | None]) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean:
        return None, None
    return mean(clean), stdev(clean) if len(clean) > 1 else 0.0


def load_run(path: Path) -> dict | None:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    condition = condition_for(data)
    if condition not in CONDITION_LABELS:
        return None

    history = [row for row in data.get("conversation_history", []) if row.get("sent") is not None]
    if not history:
        return None

    sent = [float(row["sent"]) for row in history]
    returned = [float(row["returned"]) for row in history]
    received = [float(row["received"]) for row in history]
    return_ratios = [
        returned_value / received_value if received_value > 0 else None
        for returned_value, received_value in zip(returned, received)
    ]

    low_trust_flags = [
        sent_value <= 1.0 or (ratio is not None and ratio <= 0.10)
        for sent_value, ratio in zip(sent, return_ratios)
    ]
    collapse_round = next(
        (int(row["round"]) for row, flag in zip(history, low_trust_flags) if flag),
        None,
    )

    cooperative_flags = [
        sent_value >= 2.5 and ratio is not None and ratio >= 0.30
        for sent_value, ratio in zip(sent, return_ratios)
    ]
    partial_recovery_flags = [
        sent_value >= 1.0 and ratio is not None and ratio >= 0.25
        for sent_value, ratio in zip(sent, return_ratios)
    ]
    recovery_round = None
    partial_recovery_round = None
    sustained_recovery_round = None
    if collapse_round is not None:
        collapse_index = next(i for i, row in enumerate(history) if int(row["round"]) == collapse_round)
        for i in range(collapse_index + 1, len(history)):
            if partial_recovery_flags[i]:
                partial_recovery_round = int(history[i]["round"])
                break
        for i in range(collapse_index + 1, len(history)):
            if cooperative_flags[i]:
                recovery_round = int(history[i]["round"])
                break
        for i in range(collapse_index + 1, len(history) - 1):
            if cooperative_flags[i] and cooperative_flags[i + 1]:
                sustained_recovery_round = int(history[i]["round"])
                break

    zero_exchange_flags = [sent_value <= 0.0 and returned_value <= 0.0 for sent_value, returned_value in zip(sent, returned)]
    zero_spiral = max_consecutive(zero_exchange_flags) >= 2

    final_balances = history[-1].get("balances", {})
    final_mean_balance = (
        float(final_balances["Agent_1"]) + float(final_balances["Agent_2"])
    ) / 2.0

    myth_text, game_text = extract_texts(history)
    myth_game_cosine = cosine_from_counters(Counter(tokenize(myth_text)), Counter(tokenize(game_text)))
    myth_keyword_density = keyword_density(myth_text)

    return {
        "path": str(path.relative_to(PROJECT_ROOT)) if path.is_absolute() else str(path),
        "noise": noise_for(path),
        "noise_label": NOISE_LABELS[noise_for(path)],
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        "replicate": data.get("run_metadata", {}).get("replicate_id"),
        "n_rounds": len(history),
        "avg_sent": mean(sent),
        "avg_returned": mean(returned),
        "avg_return_ratio": mean([ratio for ratio in return_ratios if ratio is not None]),
        "final_mean_balance": final_mean_balance,
        "first_collapse_round": collapse_round,
        "collapsed": collapse_round is not None,
        "any_zero_send": any(value <= 0.0 for value in sent),
        "any_zero_return": any(value <= 0.0 for value in returned),
        "any_zero_exchange": any(zero_exchange_flags),
        "zero_exchange_spiral": zero_spiral,
        "max_zero_exchange_streak": max_consecutive(zero_exchange_flags),
        "partial_recovered_after_low_trust": partial_recovery_round is not None if collapse_round is not None else None,
        "partial_recovery_round": partial_recovery_round,
        "recovered_after_low_trust": recovery_round is not None if collapse_round is not None else None,
        "recovery_round": recovery_round,
        "sustained_recovery_after_low_trust": sustained_recovery_round is not None if collapse_round is not None else None,
        "sustained_recovery_round": sustained_recovery_round,
        "myth_game_cosine": myth_game_cosine,
        "myth_game_keyword_density_per_100w": myth_keyword_density,
    }


def format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            values = []
            for column in columns:
                text = format_value(row.get(column))
                if any(char in text for char in [",", '"', "\n"]):
                    text = '"' + text.replace('"', '""') + '"'
                values.append(text)
            handle.write(",".join(values) + "\n")


def condition_summary(rows: list[dict]) -> list[dict]:
    summaries: list[dict] = []
    groups = sorted({(row["noise"], row["condition"]) for row in rows})
    for noise, condition in groups:
        group = [row for row in rows if row["noise"] == noise and row["condition"] == condition]
        collapsed = [row for row in group if row["collapsed"]]
        low_trust = [row for row in group if row["collapsed"]]
        partial_recovered = [row for row in low_trust if row["partial_recovered_after_low_trust"] is True]
        recovered = [row for row in low_trust if row["recovered_after_low_trust"] is True]
        sustained = [row for row in low_trust if row["sustained_recovery_after_low_trust"] is True]

        summary = {
            "noise": noise,
            "noise_label": NOISE_LABELS[noise],
            "condition": condition,
            "condition_label": CONDITION_LABELS[condition],
            "n": len(group),
            "mean_sent": summarize_values([row["avg_sent"] for row in group])[0],
            "sd_sent": summarize_values([row["avg_sent"] for row in group])[1],
            "mean_returned": summarize_values([row["avg_returned"] for row in group])[0],
            "sd_returned": summarize_values([row["avg_returned"] for row in group])[1],
            "mean_return_ratio": summarize_values([row["avg_return_ratio"] for row in group])[0],
            "mean_final_balance": summarize_values([row["final_mean_balance"] for row in group])[0],
            "sd_final_balance": summarize_values([row["final_mean_balance"] for row in group])[1],
            "collapse_runs": len(collapsed),
            "collapse_rate": len(collapsed) / len(group) if group else None,
            "mean_first_collapse_round": summarize_values([row["first_collapse_round"] for row in collapsed])[0],
            "zero_send_runs": sum(1 for row in group if row["any_zero_send"]),
            "zero_return_runs": sum(1 for row in group if row["any_zero_return"]),
            "zero_exchange_spiral_runs": sum(1 for row in group if row["zero_exchange_spiral"]),
            "zero_exchange_spiral_rate": sum(1 for row in group if row["zero_exchange_spiral"]) / len(group) if group else None,
            "low_trust_runs": len(low_trust),
            "partial_recovered_runs": len(partial_recovered),
            "partial_recovery_rate_among_low_trust": len(partial_recovered) / len(low_trust) if low_trust else None,
            "recovered_runs": len(recovered),
            "recovery_rate_among_low_trust": len(recovered) / len(low_trust) if low_trust else None,
            "sustained_recovery_runs": len(sustained),
            "sustained_recovery_rate_among_low_trust": len(sustained) / len(low_trust) if low_trust else None,
            "mean_myth_game_cosine": summarize_values([row["myth_game_cosine"] for row in group])[0],
            "mean_myth_game_keyword_density_per_100w": summarize_values(
                [row["myth_game_keyword_density_per_100w"] for row in group]
            )[0],
        }
        summaries.append(summary)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/analysis/myth_causal_behavior_metrics_r10",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dirs = args.input_dir or DEFAULT_INPUT_DIRS
    input_dirs = [path if path.is_absolute() else PROJECT_ROOT / path for path in input_dirs]

    rows: list[dict] = []
    for input_dir in input_dirs:
        for path in sorted(p for p in input_dir.rglob("*.json") if is_primary_json(p)):
            row = load_run(path)
            if row:
                rows.append(row)

    if not rows:
        raise SystemExit("No matching runs found.")

    summaries = condition_summary(rows)
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir

    run_columns = [
        "noise_label",
        "condition_label",
        "replicate",
        "n_rounds",
        "avg_sent",
        "avg_returned",
        "avg_return_ratio",
        "final_mean_balance",
        "first_collapse_round",
        "collapsed",
        "any_zero_send",
        "any_zero_return",
        "zero_exchange_spiral",
        "max_zero_exchange_streak",
        "partial_recovered_after_low_trust",
        "partial_recovery_round",
        "recovered_after_low_trust",
        "recovery_round",
        "sustained_recovery_after_low_trust",
        "sustained_recovery_round",
        "myth_game_cosine",
        "myth_game_keyword_density_per_100w",
        "path",
    ]
    summary_columns = [
        "noise_label",
        "condition_label",
        "n",
        "mean_sent",
        "sd_sent",
        "mean_returned",
        "sd_returned",
        "mean_return_ratio",
        "mean_final_balance",
        "sd_final_balance",
        "collapse_runs",
        "collapse_rate",
        "mean_first_collapse_round",
        "zero_send_runs",
        "zero_return_runs",
        "zero_exchange_spiral_runs",
        "zero_exchange_spiral_rate",
        "low_trust_runs",
        "partial_recovered_runs",
        "partial_recovery_rate_among_low_trust",
        "recovered_runs",
        "recovery_rate_among_low_trust",
        "sustained_recovery_runs",
        "sustained_recovery_rate_among_low_trust",
        "mean_myth_game_cosine",
        "mean_myth_game_keyword_density_per_100w",
    ]
    write_csv(output_dir / "run_behavior_metrics.csv", rows, run_columns)
    write_csv(output_dir / "summary_behavior_metrics.csv", summaries, summary_columns)

    print(f"Wrote {output_dir / 'run_behavior_metrics.csv'}")
    print(f"Wrote {output_dir / 'summary_behavior_metrics.csv'}")
    print()
    print("Definitions:")
    print("- collapse: first round where sent <= $1.00 or returned/received <= 0.10")
    print("- zero-exchange spiral: at least two consecutive rounds with sent == $0 and returned == $0")
    print("- partial recovery: after collapse, a later round with sent >= $1.00 and returned/received >= 0.25")
    print("- recovery: after collapse, a later round with sent >= $2.50 and returned/received >= 0.30")
    print("- sustained recovery: after collapse, two consecutive recovery-quality rounds")
    print("- myth/game coupling: bag-of-words cosine between myth text and visible game-response text")
    print("- keyword density: trust-game concept words per 100 myth words")
    print()
    for summary in summaries:
        print(
            f"{summary['noise_label']} / {summary['condition_label']}: "
            f"n={summary['n']}, sent={summary['mean_sent']:.2f}, returned={summary['mean_returned']:.2f}, "
            f"final={summary['mean_final_balance']:.2f}, collapse={summary['collapse_runs']}/{summary['n']}, "
            f"zero_spiral={summary['zero_exchange_spiral_runs']}/{summary['n']}, "
            f"partial_recovery={summary['partial_recovered_runs']}/{summary['low_trust_runs']}, "
            f"strict_recovery={summary['recovered_runs']}/{summary['low_trust_runs']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
