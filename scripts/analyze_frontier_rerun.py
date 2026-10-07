#!/usr/bin/env python3
"""Resource boxplot grid for the 2026-09-18 frontier rerun against the n = 10 main-model runs.

Rows: 2-agent and 8-agent runs. Columns: one per provider, frontier arm beside the main
model from the same provider (Opus 5 vs Sonnet 4.5, Gemini 3.1 Pro vs 3.7 Flash, GPT-5.6 Sol
at effort high vs GPT-5 Nano at high). Three task-order groups per panel; within each group
the frontier box sits left and the main-model box right. Each dot is one run: the mean
cumulative resources over that run's agents after round 10 (so no dot exceeds 75).
Frontier runs come from the launcher-audited finals under data/json/noise_experiments/
frontier_rerun_20260918/ (pilot: one run per cell; main stage adds four more). Main-model
runs are the single-model pools of Table 1 at n = 10 (analyses/table1_n10.n10_pools():
the five September runs plus the five 2026-10-01 extension runs per cell).

Outputs (docs/figures/frontier_rerun_20260918/): resources_boxplots.{png,svg,pdf},
cell_summary.csv (per arm x size x task order: runs, mean (sd over runs) resources).
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses._shared import configure_matplotlib  # noqa: E402
from src.experiment_condition import output_provenance  # noqa: E402
from scripts.analyze_negative_only_crossmodel_batch import ALLOWED_DIFFERENCES  # noqa: E402

FRONTIER_ROOT = ROOT / "data/json/noise_experiments/frontier_rerun_20260918"
SEPTEMBER_ROOT = ROOT / "data/json/noise_experiments/negative_only_crossmodel_reasoning_rerun_20260909"
OUTPUT = ROOT / "docs/figures/frontier_rerun_20260918"
NON_FINAL = (".results.json", ".checkpoint.json", ".error.json")
NO_DEFECTOR_PARAMS = {"noisy2_crossmodel_negative_game_r3", "noisy2_crossmodel_negative_twotask_r3",
                      "noisy8_crossmodel_negative_game_r3", "noisy8_crossmodel_negative_twotask_r3"}
ARM_OF = {  # provider_model -> (label, column)
    "claude-opus-5": ("Opus 5", 0), "claude-sonnet-4-5-20250929": ("Sonnet 4.5", 0),
    "claude-opus-5-5": ("Opus 5.5", 0),  # 2026-09-23 check; not in PLOTTED_ARMS (see compare_opus55_opus5.py)
    "gemini-3.1-pro-preview": ("Gemini 3.1 Pro", 1), "gemini-3.7-flash": ("Gemini 3.7 Flash", 1),
    "gpt-5.6-sol": ("Sol", 2), "gpt-5-nano": ("GPT-5 Nano", 2),
}
TASK_ORDERS = ("game", "game_myth", "myth_game")
ORDER_LABELS = ("Game only", "Game → Myth", "Myth → Game")
BOX_COLORS = ("#9a9a9a", "#f4a37a", "#7fcbb4")
DOT_COLORS = ("#6f6f6f", "#f07f3c", "#3fae8b")


def finals(root, keep_dirs=None):
    for p in sorted(root.rglob("*.json")):
        if p.name.endswith(NON_FINAL) or "receipt" in p.name or "worker_logs" in p.parts or "quarantine" in p.parts:
            continue
        if keep_dirs is not None and p.parent.name not in keep_dirs:
            continue
        yield p


def rows_for(path, source):
    run = json.loads(path.read_text())
    m = run["run_metadata"]
    plan = m["llm_request"]
    if plan["provider_model"] not in ARM_OF:  # later sets (e.g. the 2026-09-28 update) share this folder
        return []
    label, column = ARM_OF[plan["provider_model"]]
    effort = (plan.get("parameters") or {}).get("reasoning_effort")
    if plan["provider_model"] == "gpt-5.6-sol":
        label = f"Sol ({effort})"
    history = run["conversation_history"]
    if len(history) != 10:
        raise RuntimeError(f"{path}: expected ten rounds, found {len(history)}")
    balances = history[-1]["balances"]
    n_agents = int(m["num_agents"])
    if len(balances) != n_agents:
        raise RuntimeError(f"{path}: {len(balances)} balances for {n_agents} agents")
    return [{"source": source, "arm": label, "column": column, "num_agents": n_agents,
             "task_order": "_".join(run["task_order"]), "replicate_id": m.get("replicate_id"),
             "agent": agent, "resources": float(value), "path": str(path.relative_to(ROOT))}
            for agent, value in balances.items()]


_MODEL_SWAP = "Frontier rerun: the model and its pinned request profile are the design factor (D011)"
ALLOWED = {
    **ALLOWED_DIFFERENCES,
    # Opus 5 declares output_config.effort, which no September profile has.
    "llm.parameters.output_config": _MODEL_SWAP,
    "llm.policy.reasoning.output_config": _MODEL_SWAP,
    "implementation": "Frontier runs use the run/frontier-rerun-20260918 launcher commits; games/ and prompts are unchanged and the launcher asserts every non-model input equals the September combination",
}


PLOTTED_ARMS = ("Opus 5", "Gemini 3.1 Pro", "Sol (high)", "Sonnet 4.5", "Gemini 3.7 Flash", "GPT-5 Nano")


def audited_frontier_finals():
    """Frontier finals are pooled only if a launcher receipt lists their sha256."""
    import hashlib
    receipts = sorted(FRONTIER_ROOT.glob("*_receipt.json"))
    if not receipts:
        raise RuntimeError(f"no launcher receipts under {FRONTIER_ROOT}")
    audited = {f["sha256"] for r in receipts for f in json.loads(r.read_text())["finals"]}
    for p in finals(FRONTIER_ROOT):
        if hashlib.sha256(p.read_bytes()).hexdigest() in audited:
            yield p
        else:
            print(f"skipping unaudited final (not in any receipt): {p.relative_to(ROOT)}")


def main_model_n10_finals():
    """Single-model dyads and populations of Table 1 at n = 10 (September runs + 2026-10-01 extension)."""
    from analyses.table1_n10 import n10_pools
    pools = n10_pools()
    return pools["dyad"][1] + pools["dyad"][3] + pools["pop"][1] + pools["pop"][3]


def load():
    rows = []
    for p in audited_frontier_finals():
        rows += rows_for(p, "frontier")
    for p in main_model_n10_finals():
        rows += rows_for(p, "main_n10")
    df = pd.DataFrame(rows)
    df = df[df["arm"].isin(PLOTTED_ARMS)].copy()  # the Sol-none smoke run is not part of any comparison
    runs = df.drop_duplicates("path")
    dupes = runs[runs.duplicated(["source", "arm", "num_agents", "task_order", "replicate_id"], keep=False)]
    if not dupes.empty:
        raise RuntimeError("More than one final for a cell/replicate:\n" + "\n".join(dupes["path"]))
    return df


def write_provenance(df):
    """Hash every output and validate all run conditions against the declared differences."""
    paths = sorted({ROOT / p for p in df["path"].unique()})
    outputs = [p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "provenance.json"]
    document = output_provenance(paths, outputs, allowed_differences=ALLOWED, output_root=OUTPUT)
    (OUTPUT / "provenance.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def per_run(df):
    """One row per run: mean cumulative resources over the run's agents after round 10."""
    out = (df.groupby(["source", "arm", "column", "num_agents", "task_order", "replicate_id", "path"])
           .agg(agents=("resources", "size"), resources=("resources", "mean")).reset_index())
    if out["resources"].max() > 75 + 1e-9:
        raise RuntimeError("a run mean exceeds 75; per-run aggregation is wrong")
    return out


def cell_summary(runs):
    g = runs.groupby(["source", "arm", "num_agents", "task_order"])
    out = g.agg(runs=("path", "nunique"), agents=("agents", "sum"),
                resources_mean=("resources", "mean"), resources_sd=("resources", "std")).reset_index()
    out["resources"] = out.apply(lambda r: f"{r.resources_mean:.1f} (±{0 if np.isnan(r.resources_sd) else r.resources_sd:.1f})", axis=1)
    return out


def plot(df):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    configure_matplotlib()
    columns = [("Opus 5", "Sonnet 4.5"), ("Gemini 3.1 Pro", "Gemini 3.7 Flash"), ("Sol (high)", "GPT-5 Nano")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharey=True, squeeze=False)
    for r, n_agents in enumerate((2, 8)):
        for c, (front, sept) in enumerate(columns):
            ax = axes[r][c]
            counts = {}
            for pos, task_order in enumerate(TASK_ORDERS, 1):
                for side, arm, offset, hatch in ((0, front, -.19, None), (1, sept, .19, "////")):
                    sel = df[(df["arm"] == arm) & (df["num_agents"] == n_agents) & (df["task_order"] == task_order)]
                    v = np.sort(sel["resources"].to_numpy())
                    counts[(side, task_order)] = sel["path"].nunique()
                    if len(v) == 0:
                        continue
                    ax.boxplot(v, positions=[pos + offset], widths=.32, patch_artist=True, showfliers=False, whis=1.5,
                               boxprops=dict(facecolor=BOX_COLORS[pos - 1] if side == 0 else "white", edgecolor="#666666", hatch=hatch),
                               medianprops=dict(color="#222222", linewidth=1.6),
                               whiskerprops=dict(color="#666666"), capprops=dict(color="#666666"))
                    ax.scatter(pos + offset + np.linspace(-.07, .07, len(v)), v, s=22, c=DOT_COLORS[pos - 1],
                               edgecolors="white", linewidths=.5, zorder=3, alpha=.9)
            ax.set_xticks([1, 2, 3], ORDER_LABELS)
            ax.set_xlim(.5, 3.5)
            ax.set_ylim(0, 80)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=.22)
            ax.spines[["top", "right"]].set_visible(False)
            n_front = sorted({counts[(0, t)] for t in TASK_ORDERS}); n_sept = sorted({counts[(1, t)] for t in TASK_ORDERS})
            ax.set_title(f"{front} (frontier, n = {'/'.join(map(str, n_front))} runs)\nvs {sept} (n = {'/'.join(map(str, n_sept))} runs)",
                         fontsize=11, fontweight="bold", pad=8)
        axes[r][0].set_ylabel(f"{n_agents} agents", fontsize=12, fontweight="bold", labelpad=14)
    fig.legend(handles=[Patch(facecolor="#cccccc", edgecolor="#666666", label="frontier arm (left box)"),
                        Patch(facecolor="white", edgecolor="#666666", hatch="////", label="main model, same provider (right box)")],
               loc="upper center", bbox_to_anchor=(.5, .935), ncol=2, fontsize=9, frameon=False)
    fig.suptitle("Final cumulative resources (mean over a run's agents)\nFrontier rerun vs n = 10 main-model runs · Informed negative-only noise · No defectors · Round 10",
                 fontsize=14, fontweight="bold")
    fig.text(.5, .012, "Each dot = one run: mean resources over its agents at round 10\n"
             "Box = middle 50% · Line = median · Whiskers = up to 1.5 × IQR",
             ha="center", fontsize=9, color="#444444")
    fig.supylabel("Mean resources per agent at round 10 (one dot per run)", fontsize=12, x=.006)
    fig.tight_layout(rect=(.045, .05, 1, .905), h_pad=2.4, w_pad=2.2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUTPUT / f"resources_boxplots.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load()
    runs = per_run(df)
    summary = cell_summary(runs)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "cell_summary.csv", index=False)
    plot(runs)
    document = write_provenance(df)
    print(summary[["source", "arm", "num_agents", "task_order", "runs", "resources"]].to_string(index=False))
    print(f"figure: {OUTPUT / 'resources_boxplots.png'}; provenance: {document['n_runs']} runs, "
          f"{len(document['observed_differences'])} declared differences observed")


if __name__ == "__main__":
    main()
