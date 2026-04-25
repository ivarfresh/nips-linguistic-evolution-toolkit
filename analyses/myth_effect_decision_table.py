#!/usr/bin/env python3
"""3×3 mean/variance decision table for the myth effect per (model × noise) cell.

For each (model × noise_condition × myth_task_order) cell, compare the cumulative
balance distribution against the matching (model × noise_condition × game) cell.
Classify based on bootstrap 95% CIs for delta_mean (of medians) and delta_std:

                    mean up          mean flat         mean down
    var down    lift+consolidation  consolidation     harmful-lock-in
    var flat    lift only           null              harmful only
    var up      lift+destabilizing  pure noise        destabilizing

"mean up" = bootstrap CI for delta_mean excludes 0 on the positive side, etc.

Usage mirrors noise_balance_comparison.py so it reads the same experiment layout.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from noise_balance_comparison import (  # noqa: E402
    NOISE_CONDITIONS,
    NOISE_LABELS,
    TASK_ORDERS,
    TASK_ORDER_LABELS,
    build_dataframe,
    build_model_data_paths,
)
import noise_balance_comparison as nbc  # noqa: E402


MYTH_TASK_ORDERS = ["game_myth", "myth_game"]

N_BOOTSTRAP = 2000
CI_ALPHA = 0.05

# 3×3 classification labels
CLASSIFICATION = {
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

CLASSIFICATION_COLORS = {
    "lift+consolidation": "#1a9850",
    "lift": "#66bd63",
    "lift+destabilizing": "#a6d96a",
    "consolidation": "#fee08b",
    "null": "#f7f7f7",
    "pure_noise": "#fdae61",
    "harmful-lock-in": "#d73027",
    "harmful": "#f46d43",
    "destabilizing": "#fdae61",
}


def bootstrap_delta(
    myth_vals: np.ndarray,
    game_vals: np.ndarray,
    stat: str,
    n_bootstrap: int = N_BOOTSTRAP,
    rng: np.random.Generator | None = None,
) -> dict:
    """Bootstrap delta = stat(myth) - stat(game) across resamples.

    stat in {"median", "std"}. Returns point estimate and 95% percentile CI.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if len(myth_vals) == 0 or len(game_vals) == 0:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_myth": len(myth_vals), "n_game": len(game_vals)}

    fn = np.median if stat == "median" else np.std

    deltas = np.empty(n_bootstrap)
    n_m, n_g = len(myth_vals), len(game_vals)
    for i in range(n_bootstrap):
        rs_m = rng.choice(myth_vals, size=n_m, replace=True)
        rs_g = rng.choice(game_vals, size=n_g, replace=True)
        deltas[i] = fn(rs_m) - fn(rs_g)

    point = float(fn(myth_vals) - fn(game_vals))
    lo = float(np.percentile(deltas, 100 * CI_ALPHA / 2))
    hi = float(np.percentile(deltas, 100 * (1 - CI_ALPHA / 2)))
    return {
        "delta": point,
        "ci_low": lo,
        "ci_high": hi,
        "n_myth": n_m,
        "n_game": n_g,
    }


def classify_direction(ci_low: float, ci_high: float) -> str:
    """Returns 'up' if CI > 0, 'down' if CI < 0, 'flat' if it straddles 0."""
    if np.isnan(ci_low) or np.isnan(ci_high):
        return "unknown"
    if ci_low > 0:
        return "up"
    if ci_high < 0:
        return "down"
    return "flat"


def build_decision_table(df: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP) -> pd.DataFrame:
    """One row per (model × noise × myth_task_order).

    Columns: delta_median, delta_std, CIs, direction tags, classification.
    """
    rows = []
    rng = np.random.default_rng(42)

    models = sorted(df["model"].unique())
    for model in models:
        for noise_cond in NOISE_CONDITIONS:
            game_vals = df[
                (df["model"] == model)
                & (df["noise_condition"] == noise_cond)
                & (df["task_order"] == "game")
            ]["mean_balance_r10"].to_numpy()

            if len(game_vals) == 0:
                continue

            for myth_to in MYTH_TASK_ORDERS:
                myth_vals = df[
                    (df["model"] == model)
                    & (df["noise_condition"] == noise_cond)
                    & (df["task_order"] == myth_to)
                ]["mean_balance_r10"].to_numpy()

                if len(myth_vals) == 0:
                    continue

                mean_res = bootstrap_delta(myth_vals, game_vals, "median", n_bootstrap=n_bootstrap, rng=rng)
                std_res = bootstrap_delta(myth_vals, game_vals, "std", n_bootstrap=n_bootstrap, rng=rng)

                mean_dir = classify_direction(mean_res["ci_low"], mean_res["ci_high"])
                std_dir = classify_direction(std_res["ci_low"], std_res["ci_high"])
                classification = CLASSIFICATION.get((mean_dir, std_dir), "unknown")

                rows.append({
                    "model": model,
                    "noise_condition": noise_cond,
                    "myth_task_order": myth_to,
                    "n_game": mean_res["n_game"],
                    "n_myth": mean_res["n_myth"],
                    "median_game": float(np.median(game_vals)),
                    "median_myth": float(np.median(myth_vals)),
                    "std_game": float(np.std(game_vals)),
                    "std_myth": float(np.std(myth_vals)),
                    "delta_median": mean_res["delta"],
                    "delta_median_ci_low": mean_res["ci_low"],
                    "delta_median_ci_high": mean_res["ci_high"],
                    "delta_std": std_res["delta"],
                    "delta_std_ci_low": std_res["ci_low"],
                    "delta_std_ci_high": std_res["ci_high"],
                    "mean_direction": mean_dir,
                    "std_direction": std_dir,
                    "classification": classification,
                })

    return pd.DataFrame(rows)


def plot_decision_table(table: pd.DataFrame, output_dir: Path) -> None:
    """One figure per myth task order: heatmap of classifications (model × noise)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(table["model"].unique())
    noise_order = NOISE_CONDITIONS

    for myth_to in MYTH_TASK_ORDERS:
        sub = table[table["myth_task_order"] == myth_to]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(max(5, 1.8 * len(noise_order) + 2), max(3, 0.9 * len(models) + 1.5)))

        for j, nc in enumerate(noise_order):
            for i, m in enumerate(models):
                row = sub[(sub["model"] == m) & (sub["noise_condition"] == nc)]
                if row.empty:
                    ax.add_patch(plt.Rectangle((j, len(models) - 1 - i), 1, 1, facecolor="#e0e0e0"))
                    ax.text(j + 0.5, len(models) - 0.5 - i, "no data", ha="center", va="center", fontsize=8, color="#606060")
                    continue
                r = row.iloc[0]
                cls = r["classification"]
                color = CLASSIFICATION_COLORS.get(cls, "#f7f7f7")
                ax.add_patch(plt.Rectangle((j, len(models) - 1 - i), 1, 1, facecolor=color, edgecolor="black", linewidth=0.5))
                dmed = r["delta_median"]
                dstd = r["delta_std"]
                ax.text(j + 0.5, len(models) - 0.35 - i, cls, ha="center", va="center", fontsize=9, fontweight="bold")
                ax.text(j + 0.5, len(models) - 0.65 - i, f"Δmed={dmed:+.1f}\nΔstd={dstd:+.1f}", ha="center", va="center", fontsize=7)

        ax.set_xlim(0, len(noise_order))
        ax.set_ylim(0, len(models))
        ax.set_xticks(np.arange(len(noise_order)) + 0.5)
        ax.set_xticklabels([NOISE_LABELS[n] for n in noise_order], fontsize=10)
        ax.set_yticks(np.arange(len(models)) + 0.5)
        ax.set_yticklabels(list(reversed(models)), fontsize=10)
        ax.set_title(f"Myth effect decision table — {TASK_ORDER_LABELS[myth_to]} vs. Game Only", fontweight="bold", fontsize=12)
        ax.tick_params(axis="both", which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        fig_path = output_dir / f"decision_table_{myth_to}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved: {fig_path}")


def main() -> None:
    nbc.configure_matplotlib() if hasattr(nbc, "configure_matplotlib") else None

    parser = argparse.ArgumentParser(
        description="3×3 mean/variance decision table for the myth effect (bootstrap 95% CIs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--noise-dir", required=True, help="Path to noise experiment directory")
    parser.add_argument("--baseline", action="append", required=True,
                        help="Baseline data: model=path (repeatable)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--target-round", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)

    args = parser.parse_args()

    baseline_dirs = {}
    for b in args.baseline:
        if "=" not in b:
            parser.error(f"--baseline must be model=path, got: {b}")
        name, path = b.split("=", 1)
        baseline_dirs[name] = path

    nbc.MODEL_DATA_PATHS = build_model_data_paths(baseline_dirs, args.noise_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data (target round {args.target_round})...")
    df = build_dataframe(target_round=args.target_round)
    print(f"Total observations: {len(df)}\n")

    n_bootstrap = args.n_bootstrap

    print(f"Building decision table (n_bootstrap={n_bootstrap})...")
    table = build_decision_table(df, n_bootstrap=n_bootstrap)

    if table.empty:
        print("No data — nothing to classify.")
        return

    csv_path = output_dir / "decision_table.csv"
    table.to_csv(csv_path, index=False)
    print(f"Table saved: {csv_path}")

    # Print a compact summary
    print(f"\n{'=' * 110}")
    print(f"{'model':<25}{'noise':<20}{'myth_order':<12}{'Δmed':>8}{' [95% CI]':>20}{'Δstd':>8}{' [95% CI]':>20}{'class':>18}")
    print("-" * 110)
    for _, r in table.iterrows():
        ci_med = f"[{r['delta_median_ci_low']:+.2f}, {r['delta_median_ci_high']:+.2f}]"
        ci_std = f"[{r['delta_std_ci_low']:+.2f}, {r['delta_std_ci_high']:+.2f}]"
        print(f"{r['model']:<25}{NOISE_LABELS.get(r['noise_condition'], r['noise_condition']):<20}{r['myth_task_order']:<12}"
              f"{r['delta_median']:>+8.2f}{ci_med:>20}{r['delta_std']:>+8.2f}{ci_std:>20}{r['classification']:>18}")

    plot_decision_table(table, output_dir)
    print(f"\nDone. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
