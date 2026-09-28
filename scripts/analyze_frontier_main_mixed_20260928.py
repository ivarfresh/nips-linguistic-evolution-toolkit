#!/usr/bin/env python3
"""Main-frontier mixed-model results (2026-09-28): Opus 5, Gemini 3.1 Pro, GPT-5.6 Sol.

Frontier versions of the mixed-model Figures 7 (dyads) and 8 (eight-agent populations),
comparing the mixed runs of scripts/run_frontier_main_mixed.py with the homogeneous
2026-09-18 frontier runs of the same models (scripts/run_frontier_rerun.py). Only
launcher-audited finals are read (a run is used only if a receipt lists its sha256).

Outputs (docs/figures/frontier_main_mixed_20260928/):
  frontier_mixed_dyads_resources_boxplots.png        Figure 7 style: homogeneous vs mixed dyads
  frontier_mixed_populations_resources_boxplots.png  Figure 8 style: homogeneous vs mixed population
  cell_summary.csv     per panel x task order: runs, mean (±sd over runs) of run-mean resources
  family_summary.csv   per setting x family x task order: resources, send, return share
                       (family's per-run mean, then mean ±sd over runs)
  provenance.json
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
from scripts.analyze_frontier_update_20260928 import ALLOWED, POOL_REASON, audited, arm_of, fmt  # noqa: E402

FRONTIER_ROOT = ROOT / "data/json/noise_experiments/frontier_rerun_20260918"
MIXED_ROOT = ROOT / "data/json/noise_experiments/frontier_mixed_main_20260928"
OUTPUT = ROOT / "docs/figures/frontier_main_mixed_20260928"
MAIN_ARMS = {"Opus 5": "Opus 5", "Gemini 3.1 Pro": "Gemini", "GPT-5.6 Sol": "Sol"}
FAMILY_OF_MODEL = {"claude-opus-5": "Opus 5", "gpt-5.6-sol": "Sol", "gemini-3.1-pro-preview": "Gemini"}
TASK_ORDERS = ("game", "game_myth", "myth_game")
ORDER_LABELS = ["Game only", "Game → Myth", "Myth → Game"]
BOX_COLORS = ["#999999", "#e99675", "#72b6a1"]  # as the mixed-model Figures 7 and 8
DOT_COLORS = ["#777777", "#fc8d62", "#66c2a5"]


def rows(path, homogeneous_family=None):
    run = json.loads(path.read_text())
    m = run["run_metadata"]
    history = run["conversation_history"]
    if len(history) != 10:
        raise RuntimeError(f"{path}: expected ten rounds, found {len(history)}")
    per_agent = m.get("agent_models") or {a: m["model"] for a in run["agents"]}
    family = {a: homogeneous_family or FAMILY_OF_MODEL[model.split("/", 1)[-1]] for a, model in per_agent.items()}
    fams = sorted(set(family.values()))
    if homogeneous_family:
        panel = f"{homogeneous_family} + {homogeneous_family}" if len(family) == 2 else f"8 {homogeneous_family}"
    elif len(family) == 2:
        panel = " + ".join(family[a] for a in sorted(family))  # Agent_1 first
        panel = " + ".join(sorted(fams, key=["Opus 5", "Gemini", "Sol"].index))
    else:
        panel = "2 Gemini + 3 Opus 5 + 3 Sol"
    sends, shares = {}, {}
    for r in history:
        for dy in r["dyads"]:
            sends.setdefault(dy["investor"], []).append(dy["sent_decision"])
            if dy["received"]:
                shares.setdefault(dy["trustee"], []).append(dy["returned_decision"] / dy["received"])
    setting = "homogeneous" if homogeneous_family else "mixed"
    return [{"setting": setting, "panel": panel, "num_agents": len(family), "task_order": "_".join(run["task_order"]),
             "replicate_id": m["replicate_id"], "agent": a, "family": family[a], "resources": float(v),
             "send": np.mean(sends.get(a, [np.nan])), "return_share": np.mean(shares.get(a, [np.nan])),
             "path": str(path.resolve())} for a, v in history[-1]["balances"].items()]


def load():
    frontier = [p for p in audited(FRONTIER_ROOT, sorted(FRONTIER_ROOT.glob("main_reasoning_on_receipt.json")))
                if arm_of(p) in MAIN_ARMS]
    mixed = list(audited(MIXED_ROOT, [MIXED_ROOT / "all_receipt.json"]))
    if len(frontier) != 90 or len(mixed) != 69:
        raise RuntimeError(f"expected 90 homogeneous and 69 mixed finals, found {len(frontier)} and {len(mixed)}")
    df = pd.DataFrame([r for p in frontier for r in rows(p, MAIN_ARMS[arm_of(p)])] + [r for p in mixed for r in rows(p)])
    if df.drop_duplicates("path").duplicated(["setting", "panel", "num_agents", "task_order", "replicate_id"]).any():
        raise RuntimeError("more than one final for a panel/task order/replicate")
    return df, frontier, mixed


def cell_summary(df):
    per_run = df.groupby(["setting", "panel", "num_agents", "task_order", "path"]).resources.mean().reset_index()
    out = per_run.groupby(["num_agents", "setting", "panel", "task_order"]).resources.agg(runs="count", mean="mean", sd="std").reset_index()
    out["resources"] = [fmt(m, s) for m, s in zip(out["mean"], out["sd"])]
    return out


def family_summary(df):
    per_run = df.groupby(["num_agents", "setting", "panel", "family", "task_order", "path"])[["resources", "send", "return_share"]].mean().reset_index()
    g = per_run.groupby(["num_agents", "setting", "panel", "family", "task_order"])
    out = g.size().rename("runs").reset_index()
    for k in ("resources", "send", "return_share"):
        stats = g[k].agg(["mean", "std"]).reset_index(drop=True)
        out[k] = [fmt(m, s) for m, s in zip(stats["mean"], stats["std"])]
        out[f"{k}_mean"] = stats["mean"].to_numpy()
    return out


def draw(ax, df, panel, n_agents, setting):
    per_run = df[(df.panel == panel) & (df.num_agents == n_agents) & (df.setting == setting)]
    per_run = per_run.groupby(["task_order", "path"]).resources.mean().reset_index()
    n = []
    for pos, order in enumerate(TASK_ORDERS, 1):
        v = np.sort(per_run[per_run.task_order == order].resources.to_numpy())
        n.append(len(v))
        ax.boxplot(v, positions=[pos], widths=.52, patch_artist=True, showfliers=False, whis=1.5,
                   boxprops=dict(facecolor=BOX_COLORS[pos - 1], edgecolor="#666666"),
                   medianprops=dict(color="#222222", linewidth=1.6),
                   whiskerprops=dict(color="#666666"), capprops=dict(color="#666666"))
        ax.scatter(pos + np.linspace(-.1, .1, len(v)), v, s=30, c=DOT_COLORS[pos - 1], edgecolors="white", linewidths=.6, zorder=3)
    ax.set_xticks([1, 2, 3], ORDER_LABELS, fontsize=9)
    ax.set_xlim(.5, 3.5)
    ax.set_ylim(0, 80)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.22)
    ax.spines[["top", "right"]].set_visible(False)
    return max(n)


def plot_dyads(df):
    import matplotlib.pyplot as plt
    configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), sharey=True, squeeze=False)
    top = ["Opus 5 + Opus 5", "Sol + Sol", "Gemini + Gemini"]
    bottom = ["Opus 5 + Sol", "Opus 5 + Gemini", "Gemini + Sol"]
    for ax, panel in zip(axes[0], top):
        n = draw(ax, df, panel, 2, "homogeneous")
        ax.set_title(f"{panel}\nhomogeneous · n = {n} per box", fontsize=12, fontweight="bold", pad=8)
    for ax, panel in zip(axes[1], bottom):
        n = draw(ax, df, panel, 2, "mixed")
        ax.set_title(f"{panel}\nmixed · n = {n} per box", fontsize=12, fontweight="bold", pad=8)
    axes[0][0].set_ylabel("Homogeneous dyads\n(frontier, 2026-09-18)", fontsize=12, fontweight="bold", labelpad=14)
    axes[1][0].set_ylabel("Mixed dyads", fontsize=12, fontweight="bold", labelpad=14)
    fig.suptitle("Final cumulative resources · Frontier models (Opus 5, GPT-5.6 Sol, Gemini 3.1 Pro)\n"
                 "Homogeneous vs mixed-model dyads · Informed negative-only noise · No defectors · Round 10", fontsize=14, fontweight="bold")
    fig.text(.5, .012, "Each dot = one run, mean of its 2 agents (n = 5 per homogeneous box; 6 per mixed box, 3 with each family sending first)\n"
             "Box = middle 50% · Line = median · Whiskers = up to 1.5 × IQR", ha="center", fontsize=9, color="#444444")
    fig.supylabel("Cumulative resources per agent", fontsize=12, x=.006)
    fig.tight_layout(rect=(.02, .05, 1, .91), h_pad=2.4, w_pad=1.6)
    fig.savefig(OUTPUT / "frontier_mixed_dyads_resources_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_populations(df):
    import matplotlib.pyplot as plt
    configure_matplotlib()
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), sharey=True, squeeze=False)
    panels = [("8 Opus 5", "homogeneous"), ("8 Sol", "homogeneous"), ("8 Gemini", "homogeneous"), ("2 Gemini + 3 Opus 5 + 3 Sol", "mixed")]
    for ax, (panel, setting) in zip(axes[0], panels):
        n = draw(ax, df, panel, 8, setting)
        ax.set_title(f"{panel}\n{setting} · n = {n}", fontsize=11, fontweight="bold", pad=8)
    fig.suptitle("Final cumulative resources per agent (all agents) · Frontier models (Opus 5, GPT-5.6 Sol, Gemini 3.1 Pro)\n"
                 "8-agent populations · Informed negative-only noise · No defectors · Round 10", fontsize=13, fontweight="bold")
    fig.text(.5, .005, "Each dot = one run (mean over its 8 agents) · Box = middle 50% · Line = median · Whiskers = up to 1.5 × IQR",
             ha="center", fontsize=9, color="#444444")
    fig.supylabel("Cumulative resources per agent", fontsize=11, x=.006)
    fig.tight_layout(rect=(.02, .05, 1, .86), w_pad=1.6)
    fig.savefig(OUTPUT / "frontier_mixed_populations_resources_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    df, frontier, mixed = load()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cells = cell_summary(df)
    cells.drop(columns=["mean", "sd"]).to_csv(OUTPUT / "cell_summary.csv", index=False)
    fam = family_summary(df)
    fam.drop(columns=[c for c in fam if c.endswith("_mean")]).to_csv(OUTPUT / "family_summary.csv", index=False)
    plot_dyads(df)
    plot_populations(df)
    outputs = [p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "provenance.json"]
    homo = [p.resolve() for p in frontier]
    mix = [p.resolve() for p in mixed]
    document = output_provenance(homo + mix, outputs, ALLOWED, output_root=OUTPUT,
                                 pools={"homogeneous": homo, "mixed": mix}, pool_reason=POOL_REASON)
    (OUTPUT / "provenance.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    print(cells.pivot_table(index=["num_agents", "setting", "panel"], columns="task_order", values="resources", aggfunc="first").to_string())
    print()
    print(fam[["num_agents", "setting", "panel", "family", "task_order", "runs", "resources", "send", "return_share"]].to_string(index=False))
    print(f"provenance: {document['n_runs']} runs")


if __name__ == "__main__":
    main()
