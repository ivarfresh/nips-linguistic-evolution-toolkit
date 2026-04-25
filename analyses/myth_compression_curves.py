#!/usr/bin/env python3
"""Compression curves across myth-chain rounds.

For each myth, compute:
  - type-token ratio (TTR)
  - Shannon entropy over word-type distribution
  - mean sentence length (words per sentence)
  - word-count (for sanity)

Aggregate by (model × condition × round) and plot trajectories.
Tests the "concrete → abstract" observation quantitatively. Rising entropy or
rising TTR across rounds = diversifying vocabulary; falling = compression.
Falling sentence length = chunking; rising = elaboration.

Usage example:
    python analyses/myth_compression_curves.py \
        --input-root data/json/noise_experiments/v2_uniform_distribution_noise \
        --output-dir data/plots/noise_experiments/v2_uniform_distribution_noise/_compression_curves
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"\b[a-z']+\b")


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"^\s*Myth:\s*", "", text, flags=re.IGNORECASE)
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def shannon_entropy(tokens: Iterable[str]) -> float:
    counts = Counter(tokens)
    n = sum(counts.values())
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def myth_metrics(text: str) -> dict:
    tokens = tokenize(text)
    sentences = split_sentences(text)
    n_tokens = len(tokens)
    n_types = len(set(tokens))
    n_sents = len(sentences)

    sent_lens = [len(tokenize(s)) for s in sentences] if sentences else [0]

    return {
        "n_tokens": n_tokens,
        "n_types": n_types,
        "ttr": n_types / n_tokens if n_tokens else float("nan"),
        "entropy_bits": shannon_entropy(tokens),
        "mean_sentence_length": float(np.mean(sent_lens)) if sent_lens else float("nan"),
        "n_sentences": n_sents,
    }


def iter_valid_jsons(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.json"):
        if ".checkpoint" in p.name or ".results" in p.name:
            continue
        yield p


def classify_condition(path: Path) -> dict:
    """Extract (model, task_order, noise_condition, myth_topic) from path + metadata."""
    parts = path.parts
    out = {"path": str(path), "model": None, "task_order": None, "noise_condition": None, "experiment": None, "myth_topic": None}

    # Walk path parts looking for known markers.
    for i, p in enumerate(parts):
        if p in {"game", "game_myth", "myth_game", "myth"}:
            out["task_order"] = p
        if p in {"baseline"} and i + 1 < len(parts):
            out["noise_condition"] = "no_noise"
            out["experiment"] = "baseline"
        if p.startswith("noise_") and i + 1 < len(parts):
            out["experiment"] = p

    # Try to read metadata for model + topic.
    try:
        with open(path) as f:
            data = json.load(f)
        meta = data.get("run_metadata", {}) or {}
        out["model"] = meta.get("model")
        out["myth_topic"] = meta.get("myth_topic") or meta.get("myth_topic_id")
        out["game_params_name"] = meta.get("game_params_name")
        if out["noise_condition"] is None:
            # Infer from game_params_name
            gpn = meta.get("game_params_name") or ""
            if "_informed" in gpn:
                out["noise_condition"] = "noise_informed"
            elif gpn and gpn != "default":
                out["noise_condition"] = "noise"
            else:
                out["noise_condition"] = "no_noise"
        return out
    except Exception:
        return out


def extract_myths_from_run(path: Path) -> list[dict]:
    """Return a list of {round, agent, text} entries for all myths in a run."""
    with open(path) as f:
        data = json.load(f)
    rows = []
    for entry in data.get("conversation_history", []):
        myths = entry.get("myths") or {}
        for agent, text in myths.items():
            if not isinstance(text, str) or not text.strip():
                continue
            rows.append({
                "round": entry.get("round"),
                "agent": agent,
                "text": text,
            })
    return rows


def build_myth_dataframe(root: Path) -> pd.DataFrame:
    rows = []
    n_files = 0
    n_myths = 0
    for p in iter_valid_jsons(root):
        info = classify_condition(p)
        for m in extract_myths_from_run(p):
            metrics = myth_metrics(m["text"])
            rows.append({
                **{k: v for k, v in info.items() if k != "path"},
                "run_path": info["path"],
                "round": m["round"],
                "agent": m["agent"],
                **metrics,
            })
            n_myths += 1
        n_files += 1
    print(f"Scanned {n_files} files, extracted {n_myths} myths.")
    return pd.DataFrame(rows)


METRIC_LABELS = {
    "ttr": "Type-Token Ratio",
    "entropy_bits": "Shannon Entropy (bits)",
    "mean_sentence_length": "Mean Sentence Length (words)",
    "n_tokens": "Token Count",
}

METRICS = ["ttr", "entropy_bits", "mean_sentence_length", "n_tokens"]


def plot_compression_curves(df: pd.DataFrame, output_dir: Path) -> None:
    """One figure per metric: rows = models, cols = myth task order. Lines per noise cond."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df.dropna(subset=["model", "task_order", "noise_condition", "round"]).copy()
    df = df[df["task_order"].isin(["game_myth", "myth_game", "myth"])]

    models = sorted(df["model"].dropna().unique())
    task_orders = [t for t in ["game_myth", "myth_game", "myth"] if t in df["task_order"].unique()]
    noise_conds = sorted(df["noise_condition"].dropna().unique())

    palette = {"no_noise": "#4C72B0", "noise": "#DD8452", "noise_informed": "#55A868"}

    for metric in METRICS:
        fig, axes = plt.subplots(
            len(models), max(1, len(task_orders)),
            figsize=(4.5 * max(1, len(task_orders)), 3.2 * len(models)),
            sharex=True, sharey=True, squeeze=False,
        )
        for i, model in enumerate(models):
            for j, to in enumerate(task_orders):
                ax = axes[i][j]
                sub = df[(df["model"] == model) & (df["task_order"] == to)]
                for nc in noise_conds:
                    s = sub[sub["noise_condition"] == nc]
                    if s.empty:
                        continue
                    grp = s.groupby("round")[metric].agg(["mean", "std", "count"]).reset_index()
                    grp = grp.sort_values("round")
                    ax.plot(grp["round"], grp["mean"], marker="o", label=nc, color=palette.get(nc))
                    ax.fill_between(
                        grp["round"],
                        grp["mean"] - grp["std"] / np.sqrt(grp["count"]),
                        grp["mean"] + grp["std"] / np.sqrt(grp["count"]),
                        alpha=0.15, color=palette.get(nc),
                    )
                ax.set_title(f"{model} / {to}", fontsize=10)
                if i == len(models) - 1:
                    ax.set_xlabel("Round")
                if j == 0:
                    ax.set_ylabel(METRIC_LABELS[metric])
                ax.grid(alpha=0.3)
        axes[0][-1].legend(title="Noise", fontsize=8, loc="best")
        fig.suptitle(f"Myth compression curve: {METRIC_LABELS[metric]}", fontweight="bold")
        plt.tight_layout()
        fig_path = output_dir / f"compression_{metric}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved: {fig_path}")


def summarize(df: pd.DataFrame, output_dir: Path) -> None:
    # Per (model, task_order, noise, round): mean ± std for each metric.
    grouped = (
        df.dropna(subset=["model", "task_order", "noise_condition", "round"])
          .groupby(["model", "task_order", "noise_condition", "round"])[METRICS]
          .agg(["mean", "std", "count"])
    )
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    out = output_dir / "compression_per_round.csv"
    grouped.to_csv(out, index=False)
    print(f"Per-round summary: {out}")

    # Slope test — fit a linear regression metric ~ round within each (model, to, nc).
    slope_rows = []
    for (m, to, nc), sub in df.groupby(["model", "task_order", "noise_condition"]):
        for metric in METRICS:
            s = sub.dropna(subset=[metric, "round"])
            if len(s) < 4:
                continue
            x = s["round"].to_numpy(dtype=float)
            y = s[metric].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            slope_rows.append({
                "model": m, "task_order": to, "noise_condition": nc,
                "metric": metric, "slope": float(slope), "intercept": float(intercept),
                "n": len(s),
            })
    slopes = pd.DataFrame(slope_rows)
    slopes_out = output_dir / "compression_slopes.csv"
    slopes.to_csv(slopes_out, index=False)
    print(f"Slope table: {slopes_out}")
    return grouped, slopes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Myth compression curves (TTR, entropy, sentence length) across rounds.",
    )
    parser.add_argument("--input-root", required=True,
                        help="Root directory to scan for simulation JSONs (recursive).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--also-scan", action="append", default=[],
                        help="Additional root to scan (repeatable, e.g. baseline dir).")
    args = parser.parse_args()

    input_roots = [Path(args.input_root)] + [Path(p) for p in args.also_scan]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for root in input_roots:
        if not root.exists():
            print(f"WARNING: path does not exist: {root}")
            continue
        print(f"Scanning {root}")
        dfs.append(build_myth_dataframe(root))
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if df.empty:
        print("No myth data found.")
        return

    print(f"\nTotal myths: {len(df)}")
    print(f"Models: {sorted(df['model'].dropna().unique())}")
    print(f"Task orders: {sorted(df['task_order'].dropna().unique())}")
    print(f"Noise conditions: {sorted(df['noise_condition'].dropna().unique())}")

    raw_csv = output_dir / "myth_metrics_raw.csv"
    df.drop(columns=["run_path"], errors="ignore").to_csv(raw_csv, index=False)
    print(f"Raw metrics: {raw_csv}")

    summarize(df, output_dir)
    plot_compression_curves(df, output_dir)
    print(f"\nDone. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
