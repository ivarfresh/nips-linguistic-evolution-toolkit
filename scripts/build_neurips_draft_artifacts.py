#!/usr/bin/env python3
"""Build NeurIPS draft analysis tables and paper figures from v4 JSON runs.

The script is deliberately self-contained so the paper draft can be regenerated
from raw simulation JSON without depending on older analysis scripts that assume
the v2 communicated-ledger noise layout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
Path("/tmp/nlet-matplotlib").mkdir(parents=True, exist_ok=True)
Path("/tmp/nlet-cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nlet-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/nlet-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parent.parent
NOISE_ROOT = REPO_ROOT / "data" / "json" / "noise_experiments"
PROJECT_ROOT = REPO_ROOT / "projects" / "neurips-2026-llm-ling-evo"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "draft_artifacts"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "manuscript" / "figures"

PRIMARY_ROOTS = [
    "v4_direct_provider",
    "v4_direct_provider_baseline",
    "v4_direct_provider_neutral",
    "v4_direct_provider_k1",
]

SKIPPED_EXPERIMENT_PATTERNS = ("deterministic",)
SKIPPED_MODEL_PATTERNS = ("gemini",)

CONDITION_ORDER = [
    "default",
    "neutral_framing_default",
    "noisy_positive_1",
    "noisy_positive_1_informed",
    "noisy_negative_1",
    "noisy_negative_1_informed",
    "noisy_positive_5",
    "noisy_positive_5_informed",
    "noisy_negative_5",
    "noisy_negative_5_informed",
    "noisy_bootstrap_cooperation",
    "noisy_bootstrap_cooperation_informed",
]

CONDITION_LABELS = {
    "default": "No noise",
    "neutral_framing_default": "Neutral framing",
    "noisy_positive_1": "Positive k=1",
    "noisy_positive_1_informed": "Positive k=1 informed",
    "noisy_negative_1": "Negative k=1",
    "noisy_negative_1_informed": "Negative k=1 informed",
    "noisy_positive_5": "Positive k=5",
    "noisy_positive_5_informed": "Positive k=5 informed",
    "noisy_negative_5": "Negative k=5",
    "noisy_negative_5_informed": "Negative k=5 informed",
    "noisy_bootstrap_cooperation": "Bootstrap",
    "noisy_bootstrap_cooperation_informed": "Bootstrap informed",
}

TASK_ORDER_LABELS = {
    "game": "Game only",
    "game_myth": "Game -> myth",
    "myth_game": "Myth -> game",
}

MODEL_LABELS = {
    "openai/gpt-5-nano": "GPT-5-Nano",
    "gpt-5-nano": "GPT-5-Nano",
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
}

MODEL_ORDER = ["Claude Sonnet 4.5", "GPT-5-Nano", "Gemini 3.1 Pro"]
TASK_ORDER = ["game", "game_myth", "myth_game"]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "had", "has", "he", "her", "his", "i", "in", "into", "is", "it", "its",
    "of", "on", "or", "she", "that", "the", "their", "them", "then", "there",
    "they", "this", "to", "was", "were", "when", "where", "who", "with", "you",
}

COOP_WORDS = {
    "alliance", "balance", "care", "courage", "fair", "fairness", "forgive",
    "gift", "give", "harmony", "help", "kindness", "listen", "mercy", "mutual",
    "offer", "pact", "patience", "promise", "reciprocity", "return", "share",
    "shared", "together", "trust", "village",
}

WE_WORDS = {"we", "us", "our", "ours", "ourselves"}
I_WORDS = {"i", "me", "my", "mine", "myself"}


def is_final_json(path: Path) -> bool:
    name = path.name
    if not name.endswith(".json"):
        return False
    if name.endswith(".results.json") or name.endswith(".error.json"):
        return False
    if ".checkpoint" in name:
        return False
    return True


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, MODEL_LABELS.get(model.split("/")[-1], model.split("/")[-1]))


def condition_key(root_name: str, game_params_name: str) -> str:
    if root_name == "v4_direct_provider_neutral" and game_params_name == "default":
        return "neutral_framing_default"
    return game_params_name


def condition_label(key: str) -> str:
    return CONDITION_LABELS.get(key, key.replace("_", " "))


def condition_rank(key: str) -> int:
    try:
        return CONDITION_ORDER.index(key)
    except ValueError:
        return len(CONDITION_ORDER)


def sorted_models(models: set[str]) -> list[str]:
    return sorted(models, key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else 99, m))


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        print(f"WARNING: failed to load {path}: {exc}")
        return None


def iter_run_files(root_names: list[str], include_gemini: bool) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for root_name in root_names:
        root = NOISE_ROOT / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            if not is_final_json(path):
                continue
            rel = path.relative_to(root)
            parts = rel.parts
            if len(parts) < 5:
                continue
            experiment, model_dir = parts[0], parts[1]
            if any(pattern in experiment for pattern in SKIPPED_EXPERIMENT_PATTERNS):
                continue
            if not include_gemini and any(pattern in model_dir.lower() for pattern in SKIPPED_MODEL_PATTERNS):
                continue
            files.append((root_name, path))
    return files


def collect_error_files(root_names: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for root_name in root_names:
        root = NOISE_ROOT / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.error.json")):
            rel = path.relative_to(root)
            parts = rel.parts
            rows.append({
                "root": root_name,
                "experiment": parts[0] if len(parts) > 0 else "",
                "model_dir": parts[1] if len(parts) > 1 else "",
                "task_order": parts[2] if len(parts) > 2 else "",
                "condition": parts[3] if len(parts) > 3 else "",
                "path": str(path.relative_to(REPO_ROOT)),
            })
    return rows


def game_rounds(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in data.get("conversation_history", []) if r.get("sent") is not None]


def safe_sum(values: list[float]) -> float:
    return float(sum(v for v in values if v is not None and not math.isnan(v)))


def safe_mean(values: list[float]) -> float:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return float(mean(clean)) if clean else float("nan")


def ratio(num: float, den: float) -> float:
    if den == 0 or math.isnan(den):
        return float("nan")
    return float(num / den)


def extract_run_metrics(root_name: str, path: Path, data: dict[str, Any]) -> dict[str, Any] | None:
    rel_parts = path.relative_to(NOISE_ROOT / root_name).parts
    if len(rel_parts) < 5:
        return None
    experiment, model_dir, task_order, game_params_name = rel_parts[:4]
    metadata = data.get("run_metadata", {})
    model_raw = metadata.get("model", model_dir)
    noise_config = metadata.get("noise_config") or {}
    condition = condition_key(root_name, metadata.get("game_params_name") or game_params_name)

    rounds = game_rounds(data)
    if not rounds:
        return None

    final_balances = rounds[-1].get("balances", {})
    final_agent_1 = float(final_balances.get("Agent_1", float("nan")))
    final_agent_2 = float(final_balances.get("Agent_2", float("nan")))
    final_total = final_agent_1 + final_agent_2

    sent = [float(r.get("sent", 0.0)) for r in rounds]
    returned = [float(r.get("returned", 0.0)) for r in rounds]
    received = [float(r.get("received", 0.0)) for r in rounds]
    sent_decision = [float(r.get("sent_decision", r.get("sent", 0.0))) for r in rounds]
    returned_decision = [float(r.get("returned_decision", r.get("returned", 0.0))) for r in rounds]
    sent_noise = [float(r.get("sent_noise", 0.0)) for r in rounds]
    returned_noise = [float(r.get("returned_noise", 0.0)) for r in rounds]

    return {
        "root": root_name,
        "experiment": experiment,
        "model": model_label(model_raw),
        "model_raw": model_raw,
        "task_order": task_order,
        "task_order_label": TASK_ORDER_LABELS.get(task_order, task_order),
        "condition": condition,
        "condition_label": condition_label(condition),
        "game_params_name": game_params_name,
        "informed": bool(noise_config.get("inform_agents", False)),
        "noise_type": noise_config.get("type", "none"),
        "noise_direction": noise_config.get("direction", "both" if noise_config else "none"),
        "noise_range": noise_config.get("range", ""),
        "noise_applies_to": noise_config.get("applies_to", ""),
        "num_game_rounds": len(rounds),
        "final_agent_1": final_agent_1,
        "final_agent_2": final_agent_2,
        "final_total": final_total,
        "final_mean_agent_balance": final_total / 2,
        "avg_sent": safe_mean(sent),
        "avg_sent_decision": safe_mean(sent_decision),
        "avg_returned": safe_mean(returned),
        "avg_returned_decision": safe_mean(returned_decision),
        "return_ratio": ratio(safe_sum(returned), safe_sum(received)),
        "decision_return_ratio": ratio(safe_sum(returned_decision), 3.0 * safe_sum(sent_decision)),
        "sent_actual_decision_ratio": ratio(safe_sum(sent), safe_sum(sent_decision)),
        "abs_sent_noise": safe_mean([abs(v) for v in sent_noise]),
        "abs_returned_noise": safe_mean([abs(v) for v in returned_noise]),
        "path": str(path.relative_to(REPO_ROOT)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["condition"], row["task_order"])].append(row)

    out: list[dict[str, Any]] = []
    for (model, condition, task_order), group in sorted(groups.items(), key=lambda item: (
        MODEL_ORDER.index(item[0][0]) if item[0][0] in MODEL_ORDER else 99,
        condition_rank(item[0][1]),
        TASK_ORDER.index(item[0][2]) if item[0][2] in TASK_ORDER else 99,
    )):
        vals = np.array([g["final_total"] for g in group], dtype=float)
        expected_n = 3 if condition == "neutral_framing_default" else 15
        out.append({
            "model": model,
            "condition": condition,
            "condition_label": condition_label(condition),
            "task_order": task_order,
            "task_order_label": TASK_ORDER_LABELS.get(task_order, task_order),
            "n": len(group),
            "expected_n": expected_n,
            "complete": len(group) >= expected_n,
            "mean_final_total": float(np.nanmean(vals)),
            "median_final_total": float(np.nanmedian(vals)),
            "std_final_total": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "iqr_final_total": float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)) if len(vals) else float("nan"),
            "mean_avg_sent": safe_mean([g["avg_sent"] for g in group]),
            "mean_avg_sent_decision": safe_mean([g["avg_sent_decision"] for g in group]),
            "mean_avg_returned": safe_mean([g["avg_returned"] for g in group]),
            "mean_avg_returned_decision": safe_mean([g["avg_returned_decision"] for g in group]),
            "mean_return_ratio": safe_mean([g["return_ratio"] for g in group]),
            "mean_decision_return_ratio": safe_mean([g["decision_return_ratio"] for g in group]),
            "mean_sent_actual_decision_ratio": safe_mean([g["sent_actual_decision_ratio"] for g in group]),
            "mean_abs_sent_noise": safe_mean([g["abs_sent_noise"] for g in group]),
            "mean_abs_returned_noise": safe_mean([g["abs_returned_noise"] for g in group]),
        })
    return out


def bootstrap_delta(a: np.ndarray, b: np.ndarray, stat: str, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float, float]:
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")
    fn = np.nanmedian if stat == "median" else np.nanstd
    point = float(fn(a) - fn(b))
    deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        rs_a = rng.choice(a, size=len(a), replace=True)
        rs_b = rng.choice(b, size=len(b), replace=True)
        deltas[i] = fn(rs_a) - fn(rs_b)
    return point, float(np.nanpercentile(deltas, 2.5)), float(np.nanpercentile(deltas, 97.5))


def build_myth_effects(rows: list[dict[str, Any]], n_bootstrap: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["condition"], row["task_order"])].append(row["final_total"])

    rng = np.random.default_rng(42)
    out: list[dict[str, Any]] = []
    keys = sorted({(model, condition) for model, condition, _task in groups.keys()}, key=lambda item: (
        MODEL_ORDER.index(item[0]) if item[0] in MODEL_ORDER else 99,
        condition_rank(item[1]),
    ))
    for model, condition in keys:
        game_vals = np.array(groups.get((model, condition, "game"), []), dtype=float)
        if len(game_vals) == 0:
            continue
        for task_order in ["game_myth", "myth_game"]:
            myth_vals = np.array(groups.get((model, condition, task_order), []), dtype=float)
            if len(myth_vals) == 0:
                continue
            dmed, dmed_lo, dmed_hi = bootstrap_delta(myth_vals, game_vals, "median", n_bootstrap, rng)
            dstd, dstd_lo, dstd_hi = bootstrap_delta(myth_vals, game_vals, "std", n_bootstrap, rng)
            out.append({
                "model": model,
                "condition": condition,
                "condition_label": condition_label(condition),
                "myth_task_order": task_order,
                "myth_task_order_label": TASK_ORDER_LABELS[task_order],
                "n_game": len(game_vals),
                "n_myth": len(myth_vals),
                "median_game": float(np.nanmedian(game_vals)),
                "median_myth": float(np.nanmedian(myth_vals)),
                "std_game": float(np.nanstd(game_vals, ddof=1)) if len(game_vals) > 1 else 0.0,
                "std_myth": float(np.nanstd(myth_vals, ddof=1)) if len(myth_vals) > 1 else 0.0,
                "delta_median": dmed,
                "delta_median_ci_low": dmed_lo,
                "delta_median_ci_high": dmed_hi,
                "delta_std": dstd,
                "delta_std_ci_low": dstd_lo,
                "delta_std_ci_high": dstd_hi,
            })
    return out


TOKEN_RE = re.compile(r"[a-z][a-z'-]*")


def tokenize(text: str) -> list[str]:
    return [tok for tok in TOKEN_RE.findall(text.lower()) if tok not in STOPWORDS and len(tok) > 2]


def jaccard(a: list[str], b: list[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a or not set_b:
        return float("nan")
    return len(set_a & set_b) / len(set_a | set_b)


def myth_rows(root_name: str, path: Path, data: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = extract_run_metrics(root_name, path, data)
    if metrics is None:
        return []
    out: list[dict[str, Any]] = []
    for entry in data.get("conversation_history", []):
        myths = entry.get("myths") or {}
        for agent, myth in myths.items():
            tokens = tokenize(myth)
            n_tokens = len(tokens)
            counts = Counter(tokens)
            coop_count = sum(counts[w] for w in COOP_WORDS)
            we_count = sum(counts[w] for w in WE_WORDS)
            i_count = sum(counts[w] for w in I_WORDS)
            out.append({
                "root": metrics["root"],
                "experiment": metrics["experiment"],
                "model": metrics["model"],
                "condition": metrics["condition"],
                "condition_label": metrics["condition_label"],
                "task_order": metrics["task_order"],
                "round": int(entry.get("round", 0)),
                "agent": agent,
                "n_tokens": n_tokens,
                "coop_per_100": (100.0 * coop_count / n_tokens) if n_tokens else float("nan"),
                "we_count": we_count,
                "i_count": i_count,
                "we_i_ratio": we_count / (i_count + 1),
                "text": myth,
                "path": metrics["path"],
            })
    return out


def build_linguistic_tables(all_myth_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    by_run_round: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_run_agent: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_myth_rows:
        by_run_round[(row["path"], row["round"])].append(row)
        by_run_agent[(row["path"], row["agent"])].append(row)

    round_rows: list[dict[str, Any]] = []
    for (path, round_num), rows in sorted(by_run_round.items()):
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda r: r["agent"])
        sim = jaccard(tokenize(rows[0]["text"]), tokenize(rows[1]["text"]))
        base = rows[0]
        round_rows.append({
            "path": path,
            "model": base["model"],
            "condition": base["condition"],
            "condition_label": base["condition_label"],
            "task_order": base["task_order"],
            "round": round_num,
            "between_agent_jaccard": sim,
            "mean_coop_per_100": safe_mean([r["coop_per_100"] for r in rows]),
            "mean_we_i_ratio": safe_mean([r["we_i_ratio"] for r in rows]),
        })

    within_rows: list[dict[str, Any]] = []
    for (_path, _agent), rows in by_run_agent.items():
        rows = sorted(rows, key=lambda r: r["round"])
        for prev, current in zip(rows, rows[1:]):
            within_rows.append({
                "path": current["path"],
                "model": current["model"],
                "condition": current["condition"],
                "condition_label": current["condition_label"],
                "task_order": current["task_order"],
                "round": current["round"],
                "within_agent_prev_jaccard": jaccard(tokenize(prev["text"]), tokenize(current["text"])),
            })

    within_lookup: dict[tuple[str, str, str, str, int], list[float]] = defaultdict(list)
    for row in within_rows:
        within_lookup[(row["path"], row["model"], row["condition"], row["task_order"], row["round"])].append(row["within_agent_prev_jaccard"])

    merged_round_rows: list[dict[str, Any]] = []
    for row in round_rows:
        key = (row["path"], row["model"], row["condition"], row["task_order"], row["round"])
        row = dict(row)
        row["within_agent_prev_jaccard"] = safe_mean(within_lookup.get(key, []))
        merged_round_rows.append(row)

    summary_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in merged_round_rows:
        summary_groups[(row["model"], row["condition"], row["task_order"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (model, condition, task_order), rows in sorted(summary_groups.items(), key=lambda item: (
        MODEL_ORDER.index(item[0][0]) if item[0][0] in MODEL_ORDER else 99,
        condition_rank(item[0][1]),
        TASK_ORDER.index(item[0][2]) if item[0][2] in TASK_ORDER else 99,
    )):
        final_round = max(r["round"] for r in rows)
        final_rows = [r for r in rows if r["round"] == final_round]
        early_rows = [r for r in rows if r["round"] == min(rr["round"] for rr in rows)]
        xs = np.array([r["round"] for r in rows], dtype=float)
        ys = np.array([r["between_agent_jaccard"] for r in rows], dtype=float)
        valid = ~np.isnan(ys)
        slope = float(np.polyfit(xs[valid], ys[valid], 1)[0]) if valid.sum() >= 2 else float("nan")
        summary_rows.append({
            "model": model,
            "condition": condition,
            "condition_label": condition_label(condition),
            "task_order": task_order,
            "task_order_label": TASK_ORDER_LABELS.get(task_order, task_order),
            "n_round_observations": len(rows),
            "mean_between_agent_jaccard": safe_mean([r["between_agent_jaccard"] for r in rows]),
            "early_between_agent_jaccard": safe_mean([r["between_agent_jaccard"] for r in early_rows]),
            "final_between_agent_jaccard": safe_mean([r["between_agent_jaccard"] for r in final_rows]),
            "between_agent_jaccard_slope": slope,
            "mean_within_agent_prev_jaccard": safe_mean([r["within_agent_prev_jaccard"] for r in rows]),
            "mean_coop_per_100": safe_mean([r["mean_coop_per_100"] for r in rows]),
            "mean_we_i_ratio": safe_mean([r["mean_we_i_ratio"] for r in rows]),
        })

    example = choose_linguistic_example(all_myth_rows)
    return merged_round_rows, summary_rows, example


def choose_linguistic_example(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No myth rows found.\n"
    by_path = defaultdict(list)
    for row in rows:
        by_path[row["path"]].append(row)
    scored: list[tuple[float, str, list[dict[str, Any]]]] = []
    for path, run_rows in by_path.items():
        final_round = max(r["round"] for r in run_rows)
        final_rows = [r for r in run_rows if r["round"] == final_round]
        if len(final_rows) < 2:
            continue
        score = jaccard(tokenize(final_rows[0]["text"]), tokenize(final_rows[1]["text"]))
        if not math.isnan(score):
            scored.append((score, path, run_rows))
    if not scored:
        return "No paired-agent myth examples found.\n"
    score, path, run_rows = max(scored, key=lambda item: item[0])
    final_round = max(r["round"] for r in run_rows)
    final_rows = sorted([r for r in run_rows if r["round"] == final_round], key=lambda r: r["agent"])
    token_counts = Counter()
    for row in run_rows:
        token_counts.update(tokenize(row["text"]))
    repeated = [tok for tok, count in token_counts.most_common(20) if count >= 4 and tok not in COOP_WORDS]
    excerpts = []
    for row in final_rows[:2]:
        text = re.sub(r"\s+", " ", row["text"]).strip()
        excerpts.append(f"- {row['agent']}: {text[:450]}...")
    return (
        f"# Linguistic Example\n\n"
        f"Selected run: `{path}`\n\n"
        f"Final-round between-agent lexical Jaccard: {score:.3f}\n\n"
        f"Repeated non-stopword motifs: {', '.join(repeated[:12]) or 'none'}\n\n"
        f"Final-round excerpts:\n" + "\n".join(excerpts) + "\n"
    )


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


def save_figure(fig: plt.Figure, figure_dir: Path, basename: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{basename}.png", bbox_inches="tight")
    fig.savefig(figure_dir / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig1_design(figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    panel_titles = ["A. Trust-game round", "B. Task order", "C. Interventions and measures"]
    x0s = [0.2, 3.45, 6.7]
    for x0, title in zip(x0s, panel_titles):
        panel = FancyBboxPatch((x0, 0.25), 2.9, 3.45, boxstyle="round,pad=0.02,rounding_size=0.05",
                               linewidth=0.8, edgecolor="#333333", facecolor="#f7f7f2")
        ax.add_patch(panel)
        ax.text(x0 + 0.12, 3.48, title, weight="bold", fontsize=10, va="top")

    ax.text(0.75, 2.85, "Investor", ha="center", va="center", weight="bold")
    ax.text(2.55, 2.85, "Trustee", ha="center", va="center", weight="bold")
    ax.annotate("send s", xy=(2.15, 2.6), xytext=(1.1, 2.6), arrowprops=dict(arrowstyle="->", lw=1.2), ha="center")
    ax.annotate("return r", xy=(1.1, 2.0), xytext=(2.15, 2.0), arrowprops=dict(arrowstyle="->", lw=1.2), ha="center")
    ax.text(1.65, 1.45, "Trustee receives 3 x s", ha="center")
    ax.text(1.65, 0.95, "Roles swap each round", ha="center")

    y_orders = [2.8, 2.1, 1.4]
    labels = ["Game only", "Game -> myth", "Myth -> game"]
    for y, label in zip(y_orders, labels):
        ax.text(4.9, y, label, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#555555", linewidth=0.7))
    ax.text(4.9, 0.9, "Same dyad, same base model\nN independent seeds per cell", ha="center", va="center")

    ax.text(8.15, 2.9, "Action channel", ha="center", weight="bold")
    ax.text(8.15, 2.45, "No noise / directional noise\nneutral role framing pilot", ha="center")
    ax.text(8.15, 1.75, "Behavior", ha="center", weight="bold")
    ax.text(8.15, 1.35, "sent, returned, balances,\ndecisions vs actual ledger", ha="center")
    ax.text(8.15, 0.75, "Language", ha="center", weight="bold")
    ax.text(8.15, 0.45, "myth convergence,\ncooperativity, motifs", ha="center")

    save_figure(fig, figure_dir, "fig1_design_schematic")


def plot_fig2_behavior(summary_rows: list[dict[str, Any]], figure_dir: Path) -> None:
    primary_conditions = [
        "default",
        "noisy_positive_5",
        "noisy_positive_5_informed",
        "noisy_negative_5",
        "noisy_negative_5_informed",
        "noisy_bootstrap_cooperation",
        "noisy_bootstrap_cooperation_informed",
    ]
    rows = [r for r in summary_rows if r["condition"] in primary_conditions]
    if not rows:
        return
    models = sorted_models({r["model"] for r in rows})
    fig, axes = plt.subplots(1, len(models), figsize=(4.1 * len(models), 4.2), sharey=True)
    if len(models) == 1:
        axes = [axes]
    markers = {"game": "o", "game_myth": "s", "myth_game": "^"}
    offsets = {"game": -0.18, "game_myth": 0.0, "myth_game": 0.18}
    colors = {"game": "#2f5d8c", "game_myth": "#b65d3a", "myth_game": "#2f7d5c"}
    for ax, model in zip(axes, models):
        model_rows = [r for r in rows if r["model"] == model]
        conds = [c for c in primary_conditions if any(r["condition"] == c for r in model_rows)]
        for task_order in TASK_ORDER:
            task_rows = [r for r in model_rows if r["task_order"] == task_order]
            for row in task_rows:
                x = conds.index(row["condition"]) + offsets[task_order]
                ax.errorbar(
                    x,
                    row["mean_final_total"],
                    yerr=row["std_final_total"],
                    marker=markers[task_order],
                    color=colors[task_order],
                    markersize=5,
                    capsize=2,
                    linestyle="none",
                    label=TASK_ORDER_LABELS[task_order],
                )
        ax.set_title(model, weight="bold")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([condition_label(c) for c in conds], rotation=35, ha="right")
        ax.set_xlabel("Condition")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Final dyad reward after 10 game rounds")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Behavioral regimes across models and action-channel conditions", y=1.04, weight="bold")
    save_figure(fig, figure_dir, "fig2_behavioral_regimes")


def plot_fig3_myth_effects(effect_rows: list[dict[str, Any]], figure_dir: Path) -> None:
    primary_conditions = {
        "default",
        "noisy_positive_5",
        "noisy_positive_5_informed",
        "noisy_negative_5",
        "noisy_negative_5_informed",
        "noisy_bootstrap_cooperation",
        "noisy_bootstrap_cooperation_informed",
    }
    rows = [r for r in effect_rows if r["condition"] in primary_conditions]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: (
        MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99,
        condition_rank(r["condition"]),
        0 if r["myth_task_order"] == "game_myth" else 1,
    ))
    height = max(4.5, 0.34 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(8.0, height))
    y_positions = np.arange(len(rows))
    colors = ["#b65d3a" if r["myth_task_order"] == "game_myth" else "#2f7d5c" for r in rows]
    for y, row, color in zip(y_positions, rows, colors):
        lo = row["delta_median_ci_low"]
        hi = row["delta_median_ci_high"]
        x = row["delta_median"]
        ax.plot([lo, hi], [y, y], color=color, lw=2)
        ax.scatter([x], [y], color=color, s=28, zorder=3)
    labels = [f"{r['model']} | {condition_label(r['condition'])} | {TASK_ORDER_LABELS[r['myth_task_order']]}" for r in rows]
    ax.axvline(0, color="#444444", lw=1, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Median final dyad reward delta vs game only")
    ax.set_title("Myth-present conditions show small, model-dependent behavioral deltas", weight="bold")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, figure_dir, "fig3_myth_effects")


def plot_fig4_linguistic(round_rows: list[dict[str, Any]], figure_dir: Path) -> None:
    rows = [r for r in round_rows if r["condition"] in {"default", "noisy_positive_5", "noisy_negative_5", "noisy_bootstrap_cooperation"}]
    if not rows:
        return
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["condition"])].append(row)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    palette = ["#2f5d8c", "#b65d3a", "#2f7d5c", "#7b609a", "#8a7a32", "#555555"]
    for color, ((model, condition), group) in zip(palette, sorted(groups.items(), key=lambda item: (
        MODEL_ORDER.index(item[0][0]) if item[0][0] in MODEL_ORDER else 99,
        condition_rank(item[0][1]),
    ))):
        by_round: dict[int, list[float]] = defaultdict(list)
        for row in group:
            if not math.isnan(row["between_agent_jaccard"]):
                by_round[int(row["round"])].append(row["between_agent_jaccard"])
        xs = sorted(by_round.keys())
        ys = [safe_mean(by_round[x]) for x in xs]
        if xs:
            ax.plot(xs, ys, marker="o", label=f"{model} | {condition_label(condition)}", color=color)
    ax.set_xlabel("Myth round")
    ax.set_ylabel("Between-agent lexical Jaccard")
    ax.set_title("Myth chains converge lexically within dyads", weight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, figure_dir, "fig4_linguistic_evolution")


def plot_appendix_diagnostics(summary_rows: list[dict[str, Any]], figure_dir: Path) -> None:
    rows = [r for r in summary_rows if r["condition"] in CONDITION_ORDER]
    if not rows:
        return

    noise_rows = [r for r in rows if r["condition"] != "default" and r["condition"] != "neutral_framing_default"]
    if noise_rows:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = []
        actual = []
        decision = []
        for row in sorted(noise_rows, key=lambda r: (r["model"], condition_rank(r["condition"]), r["task_order"])):
            if row["task_order"] != "game":
                continue
            labels.append(f"{row['model']}\n{condition_label(row['condition'])}")
            actual.append(row["mean_avg_returned"])
            decision.append(row["mean_avg_returned_decision"])
        x = np.arange(len(labels))
        width = 0.38
        ax.bar(x - width / 2, actual, width, label="actual returned", color="#2f5d8c")
        ax.bar(x + width / 2, decision, width, label="decision returned", color="#b65d3a")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("Mean return amount")
        ax.set_title("Decision-vs-actual ledger diagnostics for noisy game-only cells", weight="bold")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        save_figure(fig, figure_dir, "appendix_decision_vs_actual")

    neutral_rows = [r for r in rows if r["condition"] in {"default", "neutral_framing_default"}]
    if any(r["condition"] == "neutral_framing_default" for r in neutral_rows):
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        labels = []
        values = []
        errors = []
        colors = []
        for row in sorted(neutral_rows, key=lambda r: (r["model"], condition_rank(r["condition"]), r["task_order"])):
            if row["task_order"] != "game":
                continue
            labels.append(f"{row['model']}\n{condition_label(row['condition'])}")
            values.append(row["mean_final_total"])
            errors.append(row["std_final_total"])
            colors.append("#2f5d8c" if row["condition"] == "default" else "#b65d3a")
        ax.bar(np.arange(len(labels)), values, yerr=errors, color=colors, capsize=3)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Final dyad reward")
        ax.set_title("Neutral-framing pilot diagnostic", weight="bold")
        ax.grid(axis="y", alpha=0.25)
        save_figure(fig, figure_dir, "appendix_neutral_framing")

    k1_rows = [r for r in rows if "_1" in r["condition"]]
    if k1_rows:
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        labels = []
        values = []
        errors = []
        for row in sorted(k1_rows, key=lambda r: (r["model"], condition_rank(r["condition"]), r["task_order"])):
            if row["task_order"] != "game":
                continue
            labels.append(f"{row['model']}\n{condition_label(row['condition'])}")
            values.append(row["mean_final_total"])
            errors.append(row["std_final_total"])
        ax.bar(np.arange(len(labels)), values, yerr=errors, color="#2f7d5c", capsize=3)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Final dyad reward")
        ax.set_title("k=1 directional-uniform sensitivity", weight="bold")
        ax.grid(axis="y", alpha=0.25)
        save_figure(fig, figure_dir, "appendix_k1_sensitivity")


def fmt_float(value: Any, digits: int = 2) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_findings_markdown(
    run_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    effect_rows: list[dict[str, Any]],
    linguistic_summary: list[dict[str, Any]],
    error_rows: list[dict[str, str]],
) -> str:
    completed = [r for r in summary_rows if r["complete"]]
    incomplete = [r for r in summary_rows if not r["complete"]]
    lines = [
        "# Draft Findings Summary",
        "",
        "Generated from raw v4 simulation JSON by `scripts/build_neurips_draft_artifacts.py`.",
        "",
        f"- Final run JSONs analysed: {len(run_rows)}",
        f"- Behavioural cells: {len(summary_rows)} ({len(completed)} complete, {len(incomplete)} incomplete)",
        f"- Error checkpoint files discovered: {len(error_rows)}",
        "",
    ]

    if incomplete:
        rows = []
        for row in incomplete:
            rows.append([
                row["model"],
                row["condition_label"],
                row["task_order_label"],
                str(row["n"]),
                str(row["expected_n"]),
            ])
        lines.extend([
            "## Incomplete Cells",
            "",
            markdown_table(["Model", "Condition", "Task order", "n", "expected"], rows),
            "",
        ])

    game_rows = [r for r in summary_rows if r["task_order"] == "game"]
    game_rows = sorted(game_rows, key=lambda r: (
        MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99,
        condition_rank(r["condition"]),
    ))
    rows = []
    for row in game_rows:
        rows.append([
            row["model"],
            row["condition_label"],
            str(row["n"]),
            fmt_float(row["mean_final_total"]),
            fmt_float(row["std_final_total"]),
            fmt_float(row["mean_avg_sent"]),
            fmt_float(row["mean_avg_returned"]),
            fmt_float(row["mean_return_ratio"]),
        ])
    lines.extend([
        "## Game-Only Behavioural Regimes",
        "",
        markdown_table(
            ["Model", "Condition", "n", "Mean final dyad reward", "SD", "Avg sent", "Avg returned", "Return ratio"],
            rows,
        ),
        "",
    ])

    supported_positive = [
        r for r in effect_rows
        if not math.isnan(float(r["delta_median_ci_low"])) and float(r["delta_median_ci_low"]) > 0
    ]
    supported_negative = [
        r for r in effect_rows
        if not math.isnan(float(r["delta_median_ci_high"])) and float(r["delta_median_ci_high"]) < 0
    ]
    variance_down = [
        r for r in effect_rows
        if not math.isnan(float(r["delta_std_ci_high"])) and float(r["delta_std_ci_high"]) < 0
    ]
    for title, subset in [
        ("Supported Positive Myth Deltas", supported_positive),
        ("Supported Negative Myth Deltas", supported_negative),
        ("Supported Variance Reductions", variance_down),
    ]:
        rows = []
        for row in sorted(subset, key=lambda r: abs(float(r["delta_median"])), reverse=True):
            rows.append([
                row["model"],
                row["condition_label"],
                row["myth_task_order_label"],
                f"{fmt_float(row['delta_median'])} [{fmt_float(row['delta_median_ci_low'])}, {fmt_float(row['delta_median_ci_high'])}]",
                f"{fmt_float(row['delta_std'])} [{fmt_float(row['delta_std_ci_low'])}, {fmt_float(row['delta_std_ci_high'])}]",
            ])
        lines.extend([f"## {title}", ""])
        lines.append(markdown_table(["Model", "Condition", "Myth order", "Delta median", "Delta SD"], rows) if rows else "_None in current scan._")
        lines.append("")

    ling_rows = sorted(linguistic_summary, key=lambda r: (
        MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99,
        condition_rank(r["condition"]),
        TASK_ORDER.index(r["task_order"]) if r["task_order"] in TASK_ORDER else 99,
    ))
    rows = []
    for row in ling_rows[:30]:
        rows.append([
            row["model"],
            row["condition_label"],
            row["task_order_label"],
            str(row["n_round_observations"]),
            fmt_float(row["early_between_agent_jaccard"], 3),
            fmt_float(row["final_between_agent_jaccard"], 3),
            fmt_float(row["between_agent_jaccard_slope"], 4),
            fmt_float(row["mean_coop_per_100"]),
        ])
    lines.extend([
        "## Linguistic Convergence Proxy",
        "",
        markdown_table(
            ["Model", "Condition", "Task order", "round obs.", "Early Jaccard", "Final Jaccard", "Slope", "Coop words/100"],
            rows,
        ),
        "",
    ])

    return "\n".join(lines) + "\n"


def build_artifacts(args: argparse.Namespace) -> None:
    configure_plot_style()
    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    root_names = args.roots or PRIMARY_ROOTS

    run_rows: list[dict[str, Any]] = []
    all_myth_rows: list[dict[str, Any]] = []
    for root_name, path in iter_run_files(root_names, include_gemini=args.include_gemini):
        data = read_json(path)
        if data is None:
            continue
        metrics = extract_run_metrics(root_name, path, data)
        if metrics is not None:
            run_rows.append(metrics)
        all_myth_rows.extend(myth_rows(root_name, path, data))

    error_rows = collect_error_files(root_names)
    summary_rows = summarize_runs(run_rows)
    effect_rows = build_myth_effects(run_rows, args.n_bootstrap)
    round_rows, linguistic_summary, linguistic_example = build_linguistic_tables(all_myth_rows)

    write_csv(output_dir / "manifest.csv", run_rows)
    write_csv(output_dir / "error_manifest.csv", error_rows)
    write_csv(output_dir / "behavioral_summary.csv", summary_rows)
    write_csv(output_dir / "myth_effects.csv", effect_rows)
    write_csv(output_dir / "linguistic_rounds.csv", round_rows)
    write_csv(output_dir / "linguistic_summary.csv", linguistic_summary)
    (output_dir / "linguistic_example.md").write_text(linguistic_example, encoding="utf-8")
    (output_dir / "draft_findings.md").write_text(
        build_findings_markdown(run_rows, summary_rows, effect_rows, linguistic_summary, error_rows),
        encoding="utf-8",
    )

    plot_fig1_design(figure_dir)
    plot_fig2_behavior(summary_rows, figure_dir)
    plot_fig3_myth_effects(effect_rows, figure_dir)
    plot_fig4_linguistic(round_rows, figure_dir)
    plot_appendix_diagnostics(summary_rows, figure_dir)

    print(f"run rows: {len(run_rows)}")
    print(f"behavioral cells: {len(summary_rows)}")
    print(f"myth-effect cells: {len(effect_rows)}")
    print(f"myth rows: {len(all_myth_rows)}")
    print(f"outputs: {output_dir}")
    print(f"figures: {figure_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--root", dest="roots", action="append", help="Noise output subdir to scan; repeatable.")
    parser.add_argument("--include-gemini", action="store_true", help="Include partial Gemini runs in tables and figures.")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    build_artifacts(args)


if __name__ == "__main__":
    main()
