#!/usr/bin/env python3
"""Walk every simulation JSON and build a tidy cell-level summary.

For each run, extracts:
  - balance_r10: mean of Agent_1 + Agent_2 balances at round 10
  - mean_sent / mean_returned over rounds
Tags each row with (version, experiment_set, model, task_order, noise_condition).

Outputs:
  - runs.csv: one row per run (raw)
  - cell_summary.csv: one row per (model x noise_label x task_order)
                      with n, mean, median, std, IQR
  - deltas.csv: one row per (model x noise_label x myth_task_order)
                with delta_mean, delta_median, var_ratio (myth/game-only),
                bootstrap 95% CIs.

Usage:
    python3 build_cell_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
JSON_ROOT = REPO_ROOT / "data" / "json"
OUT_DIR = Path(__file__).parent / "cell_summaries"
OUT_DIR.mkdir(exist_ok=True)

TARGET_ROUND = 10
N_BOOTSTRAP = 2000
RNG = np.random.default_rng(42)

# Scope: this week's runs only; include the late missing-condition model sweep.
MODELS = {
    "claude-sonnet-4.5": "claude-sonnet-4.5",
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "gpt-5.5": "gpt-5.5",
    "gpt-5-nano": "gpt-5-nano",
}
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

NOISE_LABEL_FROM_EXPERIMENT = {
    "noise_bootstrap_mem3": "bootstrap",
    "noise_bootstrap_mem3_gemini_3_1_pro": "bootstrap",
    "noise_bootstrap_mem3_gpt5_5": "bootstrap",
    "noise_negative_mem3": "negative_5",
    "noise_negative_mem3_gpt5_nano": "negative_5",
    "noise_negative_mem3_claude_sonnet_45": "negative_5",
    "noise_negative_mem3_gemini_3_1_pro": "negative_5",
    "noise_negative_mem3_gpt5_5": "negative_5",
    "noise_positive_mem3_claude_sonnet_45": "positive",
    "noise_positive_mem3_gemini_3_1_pro": "positive",
    "noise_positive_mem3_gpt5_5": "positive",
    "noise_positive_mem3_gpt5_nano": "positive",
    "gpt5nano_shared_context_bootstrap": "bootstrap",
    "gpt5nano_shared_context_bootstrap_pilot": "bootstrap",
    "noise_deterministic_max_mem3_gpt5_nano": "deterministic_max",
    "noise_deterministic_zero_mem3": "deterministic_zero",
    "noise_pilot": "pilot",
}


def load_balance_r10(path: Path) -> Optional[dict]:
    """Return dict with balance_r10 + descriptive metrics, or None on failure."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    history = data.get("conversation_history", [])
    rounds = [r for r in history if r.get("sent") is not None]
    if len(rounds) < TARGET_ROUND:
        return None

    target = rounds[TARGET_ROUND - 1]
    balances = target.get("balances") or {}
    a1 = balances.get("Agent_1")
    a2 = balances.get("Agent_2")
    if a1 is None or a2 is None:
        return None

    sent = np.array([r.get("sent", 0.0) for r in rounds[:TARGET_ROUND]], dtype=float)
    returned = np.array([r.get("returned", 0.0) for r in rounds[:TARGET_ROUND]], dtype=float)
    return {
        "balance_r10": float((a1 + a2) / 2.0),
        "mean_sent": float(np.mean(sent)),
        "mean_returned": float(np.mean(returned)),
    }


def parse_relative(rel: Path) -> Optional[dict]:
    """Tag a JSON path with version/experiment/model/task_order/noise_condition.

    Path conventions handled:
      baseline/{model}/{task_order}/{file}.json
      noise_experiments/{version}/{experiment}/{model}/{task_order}/{noise_cond}/{file}.json
      noise_experiments/{version}/{experiment}/{model}/{task_order}/{file}.json
        (where {noise_cond} is collapsed if not present)
    """
    parts = rel.parts
    if not parts:
        return None
    if parts[0] == "baseline":
        if len(parts) < 4:
            return None
        _, model, task_order, *_ = parts
        return {
            "version": "baseline",
            "experiment": "baseline",
            "model": model,
            "task_order": task_order,
            "noise_condition": "no_noise",
            "noise_label": "no_noise",
            "informed": False,
        }
    if parts[0] == "noise_experiments":
        # noise_experiments / version / experiment / model / task_order / noise_cond / file
        if len(parts) < 6:
            return None
        version = parts[1]
        experiment = parts[2]
        model = parts[3]
        task_order = parts[4]
        # noise_cond may exist as a sub-dir or be implicit
        if len(parts) >= 7:
            noise_cond = parts[5]
        else:
            noise_cond = "default"
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
    return None


def collect_rows() -> pd.DataFrame:
    rows = []
    for path in JSON_ROOT.rglob("*.json"):
        name = path.name
        if ".checkpoint" in name or ".results" in name or ".error" in name:
            continue
        rel = path.relative_to(JSON_ROOT)
        meta = parse_relative(rel)
        if meta is None:
            continue
        if meta["model"] not in MODELS:
            continue
        if meta["version"] not in INCLUDE_VERSIONS:
            continue
        if meta["task_order"] not in {"game", "game_myth", "myth_game", "myth"}:
            continue
        metrics = load_balance_r10(path)
        if metrics is None:
            continue
        row = {**meta, **metrics, "path": str(rel)}
        rows.append(row)
    return pd.DataFrame(rows)


def cell_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["model", "version", "experiment", "noise_label",
                    "noise_condition", "informed", "task_order"])["balance_r10"]
        .agg([
            ("n", "size"),
            ("mean", "mean"),
            ("median", "median"),
            ("std", "std"),
            ("q25", lambda v: float(np.percentile(v, 25))),
            ("q75", lambda v: float(np.percentile(v, 75))),
            ("min", "min"),
            ("max", "max"),
        ])
        .reset_index()
    )
    grouped["iqr"] = grouped["q75"] - grouped["q25"]
    return grouped


def bootstrap_ci(values: np.ndarray, stat, n=N_BOOTSTRAP, alpha=0.05):
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(stat(values))
    samples = np.empty(n)
    for i in range(n):
        idx = RNG.integers(0, len(values), size=len(values))
        samples[i] = stat(values[idx])
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return point, lo, hi


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, stat, n=N_BOOTSTRAP, alpha=0.05):
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(stat(a) - stat(b))
    samples = np.empty(n)
    for i in range(n):
        ai = RNG.integers(0, len(a), size=len(a))
        bi = RNG.integers(0, len(b), size=len(b))
        samples[i] = stat(a[ai]) - stat(b[bi])
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return point, lo, hi


def bootstrap_var_ratio_ci(a: np.ndarray, b: np.ndarray, n=N_BOOTSTRAP, alpha=0.05):
    """var(a) / var(b). a = myth, b = game-only."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(np.var(a, ddof=1) / np.var(b, ddof=1))
    samples = np.empty(n)
    for i in range(n):
        ai = RNG.integers(0, len(a), size=len(a))
        bi = RNG.integers(0, len(b), size=len(b))
        va = np.var(a[ai], ddof=1)
        vb = np.var(b[bi], ddof=1)
        samples[i] = va / vb if vb > 0 else np.nan
    samples = samples[~np.isnan(samples)]
    if len(samples) == 0:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return point, lo, hi


def deltas(df: pd.DataFrame) -> pd.DataFrame:
    """For each (model x version x experiment x noise_condition), compare
    each myth task order to game-only on mean and variance."""
    rows = []
    keys = ["model", "version", "experiment", "noise_label",
            "noise_condition", "informed"]
    for key, sub in df.groupby(keys):
        game = sub[sub["task_order"] == "game"]["balance_r10"].values
        if len(game) < 2:
            continue
        for myth_to in ("game_myth", "myth_game"):
            myth = sub[sub["task_order"] == myth_to]["balance_r10"].values
            if len(myth) < 2:
                continue
            d_mean, d_mean_lo, d_mean_hi = bootstrap_diff_ci(myth, game, np.mean)
            d_med, d_med_lo, d_med_hi = bootstrap_diff_ci(myth, game, np.median)
            vr, vr_lo, vr_hi = bootstrap_var_ratio_ci(myth, game)
            row = dict(zip(keys, key))
            row.update({
                "myth_task_order": myth_to,
                "n_game": len(game),
                "n_myth": len(myth),
                "mean_game": float(np.mean(game)),
                "mean_myth": float(np.mean(myth)),
                "delta_mean": d_mean,
                "delta_mean_ci_lo": d_mean_lo,
                "delta_mean_ci_hi": d_mean_hi,
                "median_game": float(np.median(game)),
                "median_myth": float(np.median(myth)),
                "delta_median": d_med,
                "delta_median_ci_lo": d_med_lo,
                "delta_median_ci_hi": d_med_hi,
                "std_game": float(np.std(game, ddof=1)),
                "std_myth": float(np.std(myth, ddof=1)),
                "var_ratio_myth_over_game": vr,
                "var_ratio_ci_lo": vr_lo,
                "var_ratio_ci_hi": vr_hi,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def classify_cell(delta_mean_lo, delta_mean_hi, var_ratio_lo, var_ratio_hi):
    """3x3 mean-shift x variance-shift classification using bootstrap CIs.

    Mean shift: 'up' if CI excludes 0 above; 'down' if excludes below; 'flat' otherwise.
    Variance shift: 'down' if CI excludes 1 below (variance reduced);
                    'up' if excludes 1 above; 'flat' otherwise.
    """
    if np.isnan(delta_mean_lo) or np.isnan(var_ratio_lo):
        return "missing"
    if delta_mean_lo > 0:
        m = "up"
    elif delta_mean_hi < 0:
        m = "down"
    else:
        m = "flat"
    if var_ratio_hi < 1:
        v = "down"
    elif var_ratio_lo > 1:
        v = "up"
    else:
        v = "flat"
    table = {
        ("up", "down"): "lift+consolidation",
        ("up", "flat"): "lift",
        ("up", "up"): "lift+destabilizing",
        ("flat", "down"): "consolidation",
        ("flat", "flat"): "null",
        ("flat", "up"): "pure_noise",
        ("down", "down"): "harmful-lock-in",
        ("down", "flat"): "harmful",
        ("down", "up"): "destabilizing",
    }
    return table[(m, v)]


def main():
    print(f"Walking {JSON_ROOT} ...")
    runs = collect_rows()
    print(f"  loaded {len(runs)} runs across {runs['model'].nunique()} models")
    runs.to_csv(OUT_DIR / "runs.csv", index=False)
    print(f"  wrote {OUT_DIR / 'runs.csv'}")

    summary = cell_summary(runs)
    summary.to_csv(OUT_DIR / "cell_summary.csv", index=False)
    print(f"  wrote {OUT_DIR / 'cell_summary.csv'} ({len(summary)} cells)")

    d = deltas(runs)
    if len(d) > 0:
        d["classification"] = d.apply(
            lambda r: classify_cell(
                r["delta_mean_ci_lo"], r["delta_mean_ci_hi"],
                r["var_ratio_ci_lo"], r["var_ratio_ci_hi"],
            ),
            axis=1,
        )
    d.to_csv(OUT_DIR / "deltas.csv", index=False)
    print(f"  wrote {OUT_DIR / 'deltas.csv'} ({len(d)} rows)")

    # Console preview
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n=== CELL N PER (model, version, experiment, noise_condition, task_order) ===")
    pivot_n = (
        runs.groupby(["model", "version", "experiment", "noise_label",
                      "noise_condition", "task_order"])
        .size()
        .reset_index(name="n")
    )
    print(pivot_n.to_string(index=False))

    if len(d) > 0:
        print("\n=== DELTAS (myth - game on mean, variance ratio myth/game) ===")
        cols = ["model", "version", "experiment", "noise_label", "informed",
                "myth_task_order", "n_game", "n_myth",
                "delta_mean", "delta_mean_ci_lo", "delta_mean_ci_hi",
                "var_ratio_myth_over_game", "var_ratio_ci_lo", "var_ratio_ci_hi",
                "classification"]
        print(d[cols].to_string(index=False))


if __name__ == "__main__":
    main()
