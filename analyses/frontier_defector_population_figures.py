#!/usr/bin/env python3
"""Mixed-model figure set for the frontier defector populations (2026-10-01/02).

The eight-agent figures made for the mid-tier mixed-model runs, redrawn for the
45 frontier defector finals (8 Sol, 4 Opus 5 + 4 Sol, 8 Opus 5; Agent_4 and
Agent_8 always send and return $0). The "ladder" has three points: 0, 4 and 8
Opus agents among Sol. Defectors' own decisions are forced, so every behaviour
measure below uses ordinary agents only: sending = ordinary senders (to any
partner, defectors included); returning = ordinary receivers; resources =
ordinary agents' final balances.

Input: data/json/noise_experiments/frontier_defector_pilot_20261001/all_receipt.json
(audited by scripts/run_frontier_defector_populations.py).

Outputs (docs/figures/frontier_defector_populations_20261002/, next to resources_boxplots.png):
  games.csv                                   one row per game
  ladder.png                                  ordinary-agent resources by family vs number of Opus agents
  fig8_populations_send_per_round.png         send fraction per round, ordinary senders
  fig8_populations_return_per_round.png       return ratio per round, ordinary receivers
  mixed-model-simulation-8-agent-split.png    amount sent by family vs number of Opus agents
  population_family_split_return.png          return proportion by family vs number of Opus agents

No API calls.
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
import analyses.mixed_model_cooperation_per_round as per_round  # noqa: E402

RECEIPT = ROOT / "data/json/noise_experiments/frontier_defector_pilot_20261001/all_receipt.json"
OUTPUT = ROOT / "docs/figures/frontier_defector_populations_20261002"
DEFECTORS = ("Agent_4", "Agent_8")
FAMILY = {"claude-opus-5": "Opus", "gpt-5.6-sol": "Sol"}
FAMILY_LONG = {"Opus": "Opus 5", "Sol": "GPT-5.6 Sol"}
FAMILY_COLORS = {"Opus": "#7570b3", "Sol": "#d95f02"}  # Anthropic / OpenAI colours of the mid-tier split figures
COMPOSITIONS = {"sol8_d2": ("8 Sol", 0), "opus4_sol4_d2": ("4 Opus+4 Sol", 4), "opus8_d2": ("8 Opus", 8)}
TASK_ORDERS = ["game", "game_myth", "myth_game"]
ORDER_LABELS = {"game": "Game only", "game_myth": "Game → Myth", "myth_game": "Myth → Game"}
XS = [0, 4, 8]
XLABELS = ["0\n(8 Sol)", "4\n(4 + 4)", "8\n(8 Opus)"]
SUBTITLE = "Rotating 8-agent populations · Informed negative-only noise · 2 forced defectors (Agent_4, Agent_8) · 10 rounds"


def families_of(run):
    agent_models = run["run_metadata"].get("agent_models")
    if agent_models:
        return {a: FAMILY[m.split("/")[-1]] for a, m in agent_models.items()}
    model = run["run_metadata"]["llm_request"]["provider_model"]
    return {a: FAMILY[model] for a in run["agents"]}


def load():
    games, agents = [], []
    for r in json.loads(RECEIPT.read_text())["finals"]:
        run = json.loads((ROOT / r["path"]).read_text())
        families = families_of(run)
        composition, opus_count = COMPOSITIONS[r["composition"]]
        base = {"path": r["path"], "composition": composition, "opus_count": opus_count,
                "task_order": r["task_order"], "replicate_id": r["replicate_id"]}
        for entry in run["conversation_history"]:
            for dyad in entry["dyads"]:
                sender, receiver = dyad["investor"], dyad["trustee"]
                received, returned = float(dyad["received"]), float(dyad["returned"])
                games.append({**base, "round": int(entry["round"]), "sender": sender, "receiver": receiver,
                              "sender_family": families[sender], "receiver_family": families[receiver],
                              "sender_is_defector": sender in DEFECTORS, "receiver_is_defector": receiver in DEFECTORS,
                              "sent": float(dyad["sent"]), "received": received, "returned": returned,
                              "return_proportion": returned / received if received > 0 else np.nan})
        balances = run["conversation_history"][-1]["balances"]
        agents += [{**base, "agent_id": a, "family": f, "final_balance": float(balances[a])}
                   for a, f in families.items() if a not in DEFECTORS]
    games, agents = pd.DataFrame(games), pd.DataFrame(agents)
    assert games["path"].nunique() == 45 and len(agents) == 45 * 6
    return games, agents


def by_family_runs(df, value, family_column):
    """Per run and family: mean of `value` (NaN dropped)."""
    out = df.dropna(subset=[value]).groupby(["path", "opus_count", "task_order", family_column])[value].mean()
    return out.rename("value").reset_index().rename(columns={family_column: "family"})


def plot_family_lines(runs, ylabel, ylim, filename, title, note, hline=None):
    """One panel per task order: each family's per-run means against the number of Opus agents."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    for ax, task_order in zip(axes, TASK_ORDERS):
        for family, marker in (("Sol", "o"), ("Opus", "s")):
            sub = runs[(runs["task_order"] == task_order) & (runs["family"] == family)]
            pos, means = [], []
            for i, x in enumerate(XS):
                v = sub[sub["opus_count"] == x]["value"].to_numpy()
                if v.size == 0:
                    continue
                pos.append(i)
                means.append(v.mean())
                ax.scatter(np.full(len(v), i) + np.linspace(-0.08, 0.08, len(v)), v, s=16,
                           c=FAMILY_COLORS[family], alpha=0.55, edgecolors="none", zorder=2)
            ax.plot(pos, means, color=FAMILY_COLORS[family], linewidth=2, marker=marker, ms=7, zorder=3)
        if hline is not None:
            ax.axhline(hline, color="#999999", linewidth=0.8, linestyle=":", zorder=1)
        ax.set_xticks(range(len(XS)), XLABELS)
        ax.set_xlim(-0.4, len(XS) - 0.6)
        ax.set_ylim(*ylim)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(ORDER_LABELS[task_order], fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of Opus 5 agents (of 8)", fontsize=10)
    axes[0].set_ylabel(ylabel, fontsize=10.5)
    handles = [Line2D([], [], color=FAMILY_COLORS[f], linewidth=2, marker=m, ms=7, label=FAMILY_LONG[f])
               for f, m in (("Opus", "s"), ("Sol", "o"))]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.suptitle(f"{title}\n{SUBTITLE}", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.06, note, ha="center", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, 0.1, 1, 0.9))
    fig.savefig(OUTPUT / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    configure_matplotlib()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    games, agents = load()
    games.to_csv(OUTPUT / "games.csv", index=False)
    runs_note = "Line = mean over runs of each family's per-run mean; small dots = runs (n = 5 per point). Defectors excluded."

    # ladder.png: ordinary-agent final resources by family.
    resources = agents.groupby(["path", "opus_count", "task_order", "family"])["final_balance"].mean().rename("value").reset_index()
    plot_family_lines(resources, "Final resources per ordinary agent", (25, 75), "ladder.png",
                      "Final resources by family, from 8 Sol to 8 Opus 5", runs_note)

    # Family split: ordinary senders (to any partner) and ordinary receivers.
    ordinary_send = games[~games["sender_is_defector"]]
    ordinary_return = games[~games["receiver_is_defector"]]
    plot_family_lines(by_family_runs(ordinary_send, "sent", "sender_family"), "Amount sent (of $5)", (0, 5.3),
                      "mixed-model-simulation-8-agent-split.png", "Amount sent, by family, from 8 Sol to 8 Opus 5",
                      runs_note + " Sends to defectors are included.")
    plot_family_lines(by_family_runs(ordinary_return, "return_proportion", "receiver_family"),
                      "Return proportion (returned / received)", (0, 1.0), "population_family_split_return.png",
                      "Return proportion, by family, from 8 Sol to 8 Opus 5",
                      runs_note + " Only games where something arrived.", hline=0.5)

    # Figure 8 per-round panels, reusing the mid-tier plotting code.
    send_runs = per_round.run_round_ratios(ordinary_send, "population")
    return_runs = per_round.run_round_ratios(ordinary_return, "population")
    runs = send_runs.drop(columns="return_ratio").merge(
        return_runs[["path", "round", "return_ratio"]], on=["path", "round"], validate="one_to_one")
    assert len(runs) == len(send_runs) == 450, "every run and round must survive the merge"
    stats = per_round.per_round_stats(runs)
    per_round.OUTPUT = OUTPUT
    rows = [("Opus 5 / Sol\n2 forced defectors", [c for c, _ in COMPOSITIONS.values()])]
    band = "Line = mean over runs, band = ± 1 sd over runs. n = 5 runs per panel. Defectors' own decisions excluded."
    per_round.plot_grid(stats, "population", rows, "send", "fig8_populations_send_per_round",
                        "How much ordinary senders trust, round by round\nFrontier populations · Informed negative-only noise · 2 forced defectors",
                        band + " Each run's value is the mean over the round's ordinary senders, sends to defectors included.")
    per_round.plot_grid(stats, "population", rows, "return", "fig8_populations_return_per_round",
                        "How much ordinary receivers give back, round by round\nFrontier populations · Informed negative-only noise · 2 forced defectors",
                        band + " Per run and round: total returned / total received where something arrived.")
    print(f"wrote 5 figures and games.csv to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
