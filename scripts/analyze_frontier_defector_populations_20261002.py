#!/usr/bin/env python3
"""Figures and tables for the frontier defector populations (2026-10-01/02).

Reads the audited receipt written by scripts/run_frontier_defector_populations.py
(45 finals: 4 Opus 5 + 4 Sol, 8 Opus 5, 8 Sol; two forced-zero defectors,
Agent_4 and Agent_8; game, game->myth, myth->game; 5 replicates) and writes to
docs/figures/frontier_defector_populations_20261002/:

  resources_boxplots.png   ordinary-agent final resources, one dot per run
  cell_summary.csv         mean (sd) per population and task order
  myth_effects.csv         seed-paired myth minus game, Welch p
  mix_vs_parts.csv         mix minus 0.5 * 8 Opus + 0.5 * 8 Sol, bootstrap 95% CI
  sends.csv                ordinary agents' mean send to ordinary and to defector partners
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses._shared import configure_matplotlib  # noqa: E402
from src.experiment_condition import output_provenance  # noqa: E402
from scripts.analyze_frontier_update_20260928 import POOL_REASON, audited  # noqa: E402

RECEIPT = ROOT / "data/json/noise_experiments/frontier_defector_pilot_20261001/all_receipt.json"
OUTPUT = ROOT / "docs/figures/frontier_defector_populations_20261002"
DEFECTORS = ("Agent_4", "Agent_8")
# Exactly the condition differences present in these 45 finals (provenance.json), with this
# design's reasons; anything else would fail the manifest check.
_MODEL = "Design factor: model (Opus 5 vs GPT-5.6 Sol), each at its D011 request profile"
_ORDER = "Design factor: task order (game, game -> myth, myth -> game)"
_SEED = "Replicate seeds 0-4, matched across populations and task orders"
_IDENTITY = "Run identity (set name, output path, replicate id)"
ALLOWED = {
    **{key: _MODEL for key in (
        "llm.agents", "llm.endpoint", "llm.model", "llm.provider", "llm.provider_model",
        "llm.parameters.max_completion_tokens", "llm.parameters.max_tokens", "llm.parameters.output_config",
        "llm.parameters.reasoning_effort", "llm.parameters.thinking", "llm.policy.max_output_tokens",
        "llm.policy.provider", "llm.policy.reasoning.output_config", "llm.policy.reasoning.reasoning_effort",
        "llm.policy.reasoning.thinking")},
    "protocol.simulation.task_order": _ORDER,
    "protocol.simulation.memory_capacity": _ORDER + "; two-task runs keep 6 messages so every decision still sees three rounds",
    "protocol.game.game_prompt_addition": _ORDER + "; myth runs add the instruction to take the myths into account",
    "protocol.myth.round1_template": _ORDER + "; game-only runs have no myth templates",
    "protocol.myth.later_rounds_template": _ORDER + "; game-only runs have no myth templates",
    **{key: _SEED for key in ("replicate.defector_seed", "replicate.noise_seed", "replicate.pairing_seed",
                              "replicate.random_defection_seed", "replicate.run_seed")},
    **{key: _IDENTITY for key in ("replicate.identity.experiment", "replicate.identity.output_path",
                                  "replicate.identity.replicate_id")},
}
POPULATIONS = {"opus4_sol4_d2": "4 Opus 5 + 4 Sol", "opus8_d2": "8 Opus 5", "sol8_d2": "8 Sol"}
TASK_ORDERS = ("game", "game_myth", "myth_game")
ORDER_LABELS = ["Game\nonly", "Game\n→ Myth", "Myth\n→ Game"]
BOX_COLORS = ["#999999", "#e99675", "#72b6a1"]  # as the mixed-model Figures 7 and 8
DOT_COLORS = ["#777777", "#fc8d62", "#66c2a5"]


def load():
    rows = []
    for r in json.loads(RECEIPT.read_text())["finals"]:
        d = json.loads((ROOT / r["path"]).read_text())
        to_ordinary, to_defector = [], []
        for entry in d["conversation_history"]:
            for dyad in entry["dyads"]:
                if dyad["investor"] in DEFECTORS:
                    continue
                send = dyad["actions"][dyad["investor"]]["decision"]
                (to_defector if dyad["trustee"] in DEFECTORS else to_ordinary).append(send)
        rows.append({"population": r["composition"], "task_order": r["task_order"], "replicate_id": r["replicate_id"],
                     "resources": r["ordinary_mean_final"], "send_to_ordinary": np.mean(to_ordinary),
                     "send_to_defector": np.mean(to_defector), "cost_usd": r["standard_rate_usd"]})
    df = pd.DataFrame(rows)
    assert len(df) == 45 and df.groupby(["population", "task_order"]).size().eq(5).all()
    return df


def fmt(values):
    return f"{np.mean(values):.1f} (±{np.std(values, ddof=1):.1f})"


def tables(df):
    cell = df.groupby(["population", "task_order"]).resources.agg(["mean", "std"]).round(2).reset_index()
    effects = []
    for pop in POPULATIONS:
        game = df[(df.population == pop) & (df.task_order == "game")].sort_values("replicate_id").resources.to_numpy()
        for order in TASK_ORDERS[1:]:
            myth = df[(df.population == pop) & (df.task_order == order)].sort_values("replicate_id").resources.to_numpy()
            diff = myth - game
            effects.append({"population": pop, "task_order": order, "paired_mean": diff.mean().round(2),
                            "paired_sd": diff.std(ddof=1).round(2), "pairs_positive": int((diff > 0).sum()),
                            "welch_p": round(stats.ttest_ind(myth, game, equal_var=False).pvalue, 4)})
    rng = np.random.default_rng(0)
    parts = []
    for order in TASK_ORDERS:
        get = lambda pop: df[(df.population == pop) & (df.task_order == order)].resources.to_numpy()
        mix, opus, sol = get("opus4_sol4_d2"), get("opus8_d2"), get("sol8_d2")
        boots = [rng.choice(mix, 5).mean() - (rng.choice(opus, 5).mean() + rng.choice(sol, 5).mean()) / 2 for _ in range(10000)]
        parts.append({"task_order": order, "mix_minus_parts": round(mix.mean() - (opus.mean() + sol.mean()) / 2, 2),
                      "ci_low": round(np.percentile(boots, 2.5), 2), "ci_high": round(np.percentile(boots, 97.5), 2)})
    sends = df.groupby(["population", "task_order"])[["send_to_ordinary", "send_to_defector"]].agg(["mean", "std"]).round(2)
    sends.columns = ["_".join(c) for c in sends.columns]
    return cell, pd.DataFrame(effects), pd.DataFrame(parts), sends.reset_index()


def plot(df):
    import matplotlib.pyplot as plt
    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True)
    for ax, (pop, label) in zip(axes, POPULATIONS.items()):
        for pos, order in enumerate(TASK_ORDERS, 1):
            v = np.sort(df[(df.population == pop) & (df.task_order == order)].resources.to_numpy())
            ax.boxplot(v, positions=[pos], widths=.52, patch_artist=True, showfliers=False, whis=1.5,
                       boxprops=dict(facecolor=BOX_COLORS[pos - 1], edgecolor="#666666"),
                       medianprops=dict(color="#222222", linewidth=1.6),
                       whiskerprops=dict(color="#666666"), capprops=dict(color="#666666"))
            ax.scatter(pos + np.linspace(-.1, .1, len(v)), v, s=30, c=DOT_COLORS[pos - 1], edgecolors="white", linewidths=.6, zorder=3)
        ax.set_xticks([1, 2, 3], ORDER_LABELS, fontsize=9)
        ax.set_xlim(.5, 3.5)
        ax.set_ylim(25, 75)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{label}\n2 forced defectors · n = 5 per box", fontsize=12, fontweight="bold", pad=8)
    fig.suptitle("Final cumulative resources of the 6 ordinary agents · Frontier populations with defectors\n"
                 "Informed negative-only noise · Agent_4 and Agent_8 always send and return $0 · Round 10",
                 fontsize=13, fontweight="bold")
    fig.text(.5, .005, "Each dot = one run, mean of its 6 ordinary agents · Box = middle 50% · Line = median · "
             "Whiskers = up to 1.5 × IQR", ha="center", fontsize=9, color="#444444")
    fig.supylabel("Cumulative resources per ordinary agent", fontsize=11, x=.006)
    fig.tight_layout(rect=(.02, .05, 1, .86), w_pad=1.6)
    fig.savefig(OUTPUT / "resources_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_provenance():
    """provenance.json over every file in OUTPUT (scripts/check_safeguards.py requires it).

    Run finals come from the audited receipt and are re-hashed against it. Shared with
    analyses/frontier_defector_population_figures.py, which writes into the same folder.
    """
    finals = [p.resolve() for p in audited(ROOT, [RECEIPT])]
    receipt = {f["sha256"]: f for f in json.loads(RECEIPT.read_text())["finals"]}
    mixed = {(ROOT / f["path"]).resolve() for f in receipt.values() if f["composition"] == "opus4_sol4_d2"}
    outputs = sorted(p for p in OUTPUT.iterdir() if p.is_file() and p.name != "provenance.json" and not p.name.startswith("."))
    document = output_provenance(finals, outputs, ALLOWED, output_root=OUTPUT,
                                 pools={"homogeneous": [p for p in finals if p not in mixed], "mixed": [p for p in finals if p in mixed]},
                                 pool_reason=POOL_REASON)
    (OUTPUT / "provenance.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def main():
    df = load()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cell, effects, parts, sends = tables(df)
    cell.to_csv(OUTPUT / "cell_summary.csv", index=False)
    effects.to_csv(OUTPUT / "myth_effects.csv", index=False)
    parts.to_csv(OUTPUT / "mix_vs_parts.csv", index=False)
    sends.to_csv(OUTPUT / "sends.csv", index=False)
    plot(df)
    write_provenance()
    for pop, label in POPULATIONS.items():
        print(label, *(fmt(df[(df.population == pop) & (df.task_order == o)].resources) for o in TASK_ORDERS), sep=" | ")
    print(effects.to_string(index=False), parts.to_string(index=False), sends.to_string(index=False), sep="\n\n")
    print(f"\ncost ${df.cost_usd.sum():.2f}")


if __name__ == "__main__":
    main()
