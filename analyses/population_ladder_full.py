#!/usr/bin/env python3
"""Figure 3 with all fifteen mixed eight-agent compositions (n = 10 each).

Paper Figure 3 shows 6 of the 15 two-family splits (1+7, 2+6, 4+4, 6+2, 7+1 for three
family pairs). The other 9 were run on 2026-11-23/27 (PR #50; Gemini via Vertex AI so
the answering model is gemini-3.7-flash, checked on every call by the launcher audit).
This script pools the paper's n = 10 tables with the new runs and draws one row per
family pair. Each single-model population is drawn once, in the first row it belongs to:

  row 1  8 GPT ... GPT/Gemini splits ... 8 Gemini
  row 2        ... GPT/Sonnet splits ... 8 Sonnet   (GPT-heavy end on the left)
  row 3        ... Sonnet/Gemini splits ...         (Sonnet-heavy end on the left)

Inputs:
  docs/figures/mixed_vs_average_n10_20261001/population_{agent_finals,games}.csv  (paper, 270 runs)
  new finals listed in the completion receipts under --new-root (270 runs):
    mixed_model_populations_20261127_vertex/completion_receipt{,_n5ext}.json
    mixed_model_populations_20261123/completion_receipt{,_n5ext}.json  (GPT+Sonnet only)
Outputs: docs/figures/population_ladder_full_20261127/
No API calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses._shared import configure_matplotlib, load_simulation_runs  # noqa: E402
import analyses.mixed_model_cooperation_per_round as pr  # noqa: E402
import scripts.analyze_mixed_model_populations as po  # noqa: E402

N10 = ROOT / "docs/figures/mixed_vs_average_n10_20261001"
OUTPUT = ROOT / "docs/figures/population_ladder_full_20261127"
NEW_ROOT = ROOT.parent / "fig3-population-completion"
RECEIPTS = [
    "mixed_model_populations_20261127_vertex/completion_receipt.json",
    "mixed_model_populations_20261127_vertex/completion_receipt_n5ext.json",
    "mixed_model_populations_20261123/completion_receipt.json",
    "mixed_model_populations_20261123/completion_receipt_n5ext.json",
]
FAMILY_ORDER = ["Gemini", "GPT", "Sonnet"]
ROWS = [
    ("GPT / Gemini", "GPT", "Gemini"),
    ("GPT / Sonnet", "GPT", "Sonnet"),
    ("Sonnet / Gemini", "Sonnet", "Gemini"),
]
SPLITS = [7, 6, 4, 2, 1]  # agents of the left-hand family, left to right after the pure end
SHORT_ORDERS = ["Game\nonly", "Game\n→ Myth", "Myth\n→ Game"]


def canonical(composition: str) -> str:
    """'4 Sonnet + 4 Gemini' and '4 Gemini + 4 Sonnet' name the same population."""
    parts = [p.split(" ", 1) for p in composition.split(" + ")]
    if len(parts) == 1:
        return composition
    parts.sort(key=lambda p: (int(p[0]), FAMILY_ORDER.index(p[1])))
    return " + ".join(f"{n} {f}" for n, f in parts)


def name(left: str, n_left: int, right: str) -> str:
    if n_left == 8:
        return f"8 {left}"
    if n_left == 0:
        return f"8 {right}"
    return canonical(f"{n_left} {left} + {8 - n_left} {right}")


def layout():
    """Each panel once: a single-model population already shown in an earlier row is left out."""
    seen, rows = set(), []
    for label, left, right in ROWS:
        row = []
        for n in [8, *SPLITS, 0]:
            c = name(left, n, right)
            row.append(None if c in seen else c)
            seen.add(c)
        rows.append((label, row))
    return rows


def new_tables(new_root: Path):
    paths = []
    for r in RECEIPTS:
        receipt = json.loads((new_root / "data/json/noise_experiments" / r).read_text())
        assert not receipt.get("held") or r.endswith("20261123/completion_receipt.json"), r
        paths += [new_root / f["path"] for f in receipt["finals"]]
    assert len(paths) == len(set(paths)) == 270, len(paths)
    po.ROOT = new_root  # extract() records paths relative to the data's own checkout
    runs = load_simulation_runs(paths, allowed_differences=po.ALLOWED)
    games, agents = [], []
    for p in paths:
        g, a = po.extract(p, runs[str(p.resolve())])
        games += g
        agents += a
    return pd.DataFrame(games), pd.DataFrame(agents)


def draw_box(ax, per_run, composition):
    for pos, task_order in enumerate(po.TASK_ORDERS, 1):
        v = np.sort(per_run[(per_run["composition"] == composition) & (per_run["task_order"] == task_order)]["final_balance"].to_numpy())
        ax.boxplot(v, positions=[pos], widths=.52, patch_artist=True, showfliers=False, whis=1.5,
                   boxprops=dict(facecolor=po.BOX_COLORS[pos - 1], edgecolor="#666666"),
                   medianprops=dict(color="#222222", linewidth=1.6),
                   whiskerprops=dict(color="#666666"), capprops=dict(color="#666666"))
        ax.scatter(pos + np.linspace(-.1, .1, len(v)), v, s=20, c=po.DOT_COLORS[pos - 1], edgecolors="white", linewidths=.5, zorder=3)
    ax.set_xticks([1, 2, 3], SHORT_ORDERS, fontsize=7.5)
    ax.set_xlim(.5, 3.5)
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=.22)


def draw_round(ax, stats, composition):
    d = stats[(stats["composition"] == composition) & (stats["metric"] == "send")]
    for task_order in pr.TASK_ORDERS:
        t = d[d["task_order"] == task_order].sort_values("round")
        x, mean, sd = t["round"].to_numpy(), t["mean"].to_numpy(), np.nan_to_num(t["sd"].to_numpy())
        color = pr.LINE_COLORS[task_order]
        ax.fill_between(x, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=1.8, marker="o", ms=2.5)
    ax.set_xticks([1, 5, 10])
    ax.set_xlim(0.6, 10.4)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Round", fontsize=8)
    ax.grid(alpha=0.2)


def plot(draw, data, ylabel, filename, new, legend):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = layout()
    ncols = 7
    fig = plt.figure(figsize=(17, 3.4 * len(rows) + 0.5))
    # Every panel sits in its ladder column, so the same split lines up across rows;
    # a single-model population already shown in an earlier row leaves its cell empty.
    grid = fig.add_gridspec(len(rows), ncols)
    shared = None
    for r, (row_label, compositions) in enumerate(rows):
        panels = [(col, c) for col, c in enumerate(compositions) if c is not None]
        for i, (col, composition) in enumerate(panels):
            ax = fig.add_subplot(grid[r, col], sharey=shared)
            shared = shared or ax
            draw(ax, data, composition)
            title = composition.replace(" + ", " +\n") if "+" in composition else composition
            ax.set_title(title, fontsize=9.5, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            if i:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(row_label, fontsize=10.5, fontweight="bold", labelpad=8)
    fig.supylabel(ylabel, fontsize=11.5, x=0.004)
    bottom = 0.0
    if legend:
        handles = [Line2D([], [], color=pr.LINE_COLORS[t], linewidth=2, marker="o", ms=4, label=pr.ORDER_LABELS[t]) for t in pr.TASK_ORDERS]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10)
        bottom = 0.04
    fig.tight_layout(rect=(0.015, bottom, 1, 1), h_pad=1.4, w_pad=0.6)
    # The y labels sit left of the panels; leave the same margin on the right so the
    # panels themselves are centred on the page.
    left = min(ax.get_position().x0 for ax in fig.axes)
    fig.tight_layout(rect=(0.015, bottom, 1 - left + 0.01, 1), h_pad=1.4, w_pad=0.6)
    fig.savefig(OUTPUT / f"{filename}.png", dpi=200)
    fig.savefig(OUTPUT / f"{filename}.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-root", type=Path, default=NEW_ROOT, help="checkout holding the Nov 2026 run data")
    args = ap.parse_args()
    configure_matplotlib()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    old_agents = pd.read_csv(N10 / "population_agent_finals.csv")
    old_games = pd.read_csv(N10 / "population_games.csv")
    assert old_agents["path"].nunique() == old_games["path"].nunique() == 270
    new_games, new_agents = new_tables(args.new_root.resolve())
    for df in (old_agents, old_games, new_agents, new_games):
        df["composition"] = df["composition"].map(canonical)
    new = set(new_agents["composition"])
    assert len(new) == 9 and not new & set(old_agents["composition"]), sorted(new)

    agents = pd.concat([old_agents, new_agents], ignore_index=True)
    games = pd.concat([old_games, new_games], ignore_index=True)
    per_run = agents.groupby(["composition", "task_order", "path"], sort=False)["final_balance"].mean().reset_index()
    n = per_run.groupby(["composition", "task_order"]).size()
    shown = {c for _, cs in layout() for c in cs if c is not None}
    assert shown <= set(n.index.get_level_values(0)), sorted(shown - set(n.index.get_level_values(0)))
    assert (n.loc[sorted(shown)] == 10).all(), n[n != 10]

    agents.to_csv(OUTPUT / "population_agent_finals.csv", index=False)
    games.to_csv(OUTPUT / "population_games.csv", index=False)
    summary = per_run.groupby(["composition", "task_order"])["final_balance"].agg(n="size", mean="mean", sd="std").round(2).reset_index()
    summary.to_csv(OUTPUT / "resources_per_agent_summary.csv", index=False)
    stats = pr.per_round_stats(pr.run_round_ratios(games, "population"))
    plot(draw_box, per_run, "Resources per agent", "population_resources_boxplots_full", new, legend=False)
    plot(draw_round, stats, "Send fraction (sent / $5)", "population_send_per_round_full", new, legend=True)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
