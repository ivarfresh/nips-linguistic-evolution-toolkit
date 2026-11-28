"""Figures for the myth-injection ablation at the main-figure noise setting.

Per group size: bars of final resources per agent by injected text, in the
design of analyses/plot_slide678_rerun.py. Plus one figure relating each
injected run's outcome to the final resources of the run its myth came from.
All values are re-derived from finals whose hashes match the completion receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyses._shared import configure_matplotlib, write_output_provenance
from src.experiment_condition import read_final_run

RUN_ROOT = ROOT / "data/json/noise_experiments/myth_injection_noise1_20261127"
OUT_DIR = ROOT / "docs/figures/myth_injection_noise1_20261127"
MANIFEST = ROOT / "data/phase3/seed_manifest_main_figures_20261127.json"
# (cell, tick label, colour, legend label for the donor-score figure)
CELLS = [
    ("baseline", "Nothing\ninjected", "#7f7f7f", None),
    ("s_filler", "Filler\n(Wikipedia)", "#9467bd", None),
    ("s_end_minus_gpt", "GPT-5-Nano myth\nround 10, low", "#ffbb78", "GPT-5-Nano, round 10, low"),
    ("s_end_plus_gpt", "GPT-5-Nano myth\nround 10, highest", "#ff7f0e", "GPT-5-Nano, round 10, highest"),
    ("s_end_minus", "Sonnet-4.5 myth\nround 10, low", "#d62728", "Sonnet-4.5, round 10, low"),
    ("s_start", "Sonnet-4.5 myth\nround 1", "#1f77b4", "Sonnet-4.5, round 1"),
    ("s_end_plus", "Sonnet-4.5 myth\nround 10, high", "#2ca02c", "Sonnet-4.5, round 10, high"),
    ("s_end_plus_gemini", "Gemini-3.7 myth\nround 10", "#17becf", "Gemini-3.7, round 10"),
]
SIZES = {"dyad": (2, "dyads", "myth_injection_dyads"), "population": (8, "populations", "myth_injection_populations")}
CEILING = 75.0
FLOOR = 25.0
ALLOWED_DIFFERENCES = {
    "protocol.simulation.seed_myth": "Design factor: each cell injects a different donor text (or none).",
    "protocol.simulation.seed_reinject": "Design factor: seeded cells re-inject the same text every round.",
    "protocol.simulation.seed_user_prompt": "Design factor: seeded cells carry the myth-writing user prompt.",
    "replicate.identity": "Five donor/run replicates per cell.",
    "protocol.simulation.num_agents": "Design factor: fixed dyads and eight-agent populations.",
    **{f"protocol.game.personas.Agent_{i}": "Group size: agents 3 to 8 exist only in populations (all neutral personas)."
       for i in range(3, 9)},
}


def load(size):
    """Per-agent final resources by cell and replicate, verified against the receipt."""
    n_agents = SIZES[size][0]
    root = RUN_ROOT / size
    receipt = json.loads((root / "completion_receipt.json").read_text())
    planned = len(CELLS) * 5
    if receipt.get("completed") != planned or len(receipt.get("finals") or []) != planned:
        raise ValueError(f"Receipt does not certify all {planned} runs: {size}")
    values = {cell: {} for cell, *_ in CELLS}
    sources = []
    for entry in receipt["finals"]:
        path = root / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Final hash changed since completion: {path}")
        data = read_final_run(path)
        history = data["conversation_history"]
        if [row.get("round") for row in history] != list(range(1, 11)) or len(data["agents"]) != n_agents:
            raise ValueError(f"Not a complete ten-round final of the right size: {path}")
        joint = float(sum(history[-1]["balances"].values()))
        if not np.isclose(joint, entry["joint_resources"]):
            raise ValueError(f"Receipt value mismatch: {path}")
        values[entry["cell"]][entry["rep"]] = joint / n_agents
        sources.append(path)
    for cell, by_rep in values.items():
        if sorted(by_rep) != list(range(5)):
            raise ValueError(f"Cell {cell} lacks five replicates: {size}")
    return values, receipt, sources


def bar_figure(size, values, out_dir):
    _, group_label, stem = SIZES[size]
    fig, ax = plt.subplots(figsize=(15, 7))
    positions = np.arange(len(CELLS))
    rng = np.random.default_rng(seed=6)
    tops = []
    # Bars run from the lowest cell mean (left) to the highest (right), per figure.
    cells = sorted(CELLS, key=lambda item: statistics.mean(values[item[0]].values()))
    for index, (cell, _label, color, _legend) in enumerate(cells):
        observations = [values[cell][rep] for rep in range(5)]
        mean = statistics.mean(observations)
        sd = statistics.stdev(observations)
        ax.bar(positions[index], mean, width=0.62, yerr=sd, capsize=5,
               color=color, edgecolor="black", linewidth=0.7, alpha=0.88)
        ax.scatter(positions[index] + rng.uniform(-0.1, 0.1, size=5), observations,
                   s=42, color="black", edgecolor="white", linewidth=0.8, zorder=3)
        top = max(mean + sd, max(observations)) + 1.5
        tops.append(top)
        ax.text(positions[index], top, f"{mean:.1f}\n(±{sd:.1f})\nn=5", ha="center", va="bottom", fontsize=9.5)
    baseline = statistics.mean(values["baseline"].values())
    ax.axhline(CEILING, color="red", linestyle="--", alpha=0.55, label=f"Full cooperation ({CEILING:.0f})")
    ax.axhline(baseline, color="#7f7f7f", linestyle=":", alpha=0.6, linewidth=1.4,
               label=f"Nothing injected, mean ({baseline:.1f})")
    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label, _, _ in cells], fontsize=10)
    ax.set_xlabel("Text injected at every decision")
    ax.set_ylabel(f"Final resources per agent ({group_label})")
    ax.set_ylim(0, max(tops) + 12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    outputs = []
    for suffix in ("png", "pdf"):
        target = out_dir / f"{stem}.{suffix}"
        fig.savefig(target, dpi=180, bbox_inches="tight")
        outputs.append(target.name)
    plt.close(fig)
    return outputs


def donor_figure(all_values, manifest, out_dir):
    """Outcome of each injected run against the final resources of its myth's source run."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
    rng = np.random.default_rng(seed=6)
    correlations = {}
    for ax, size in zip(axes, ("dyad", "population")):
        xs, ys = [], []
        for cell, _label, color, legend in CELLS:
            if legend is None:
                continue
            donors = manifest["seeds"][cell]
            x = np.array([donors[rep]["mean_final_per_agent_at_source"] for rep in range(5)], dtype=float)
            y = np.array([all_values[size][cell][rep] for rep in range(5)], dtype=float)
            xs.extend(x)
            ys.extend(y)
            # Small horizontal jitter only where donors tie exactly (Gemini, all at 75).
            shown_x = x + (rng.uniform(-0.5, 0.5, size=5) if len(set(x)) == 1 else 0)
            ax.scatter(shown_x, y, s=70, color=color, edgecolor="black", linewidth=0.7, alpha=0.9, label=legend, zorder=3)
        rho = float(np.corrcoef(xs, ys)[0, 1])
        correlations[size] = {"pearson_r": rho, "n_runs": len(xs)}
        baseline = statistics.mean(all_values[size]["baseline"].values())
        ax.axhline(baseline, color="#7f7f7f", linestyle=":", linewidth=1.4, alpha=0.7)
        ax.text(FLOOR - 1.5, baseline - 0.5, f"nothing injected ({baseline:.1f})", ha="left", va="top",
                fontsize=9, color="#555555")
        ax.axhline(CEILING, color="red", linestyle="--", alpha=0.55)
        ax.plot([FLOOR, CEILING], [FLOOR, CEILING], color="black", linewidth=0.8, alpha=0.25)
        ax.set_xlim(FLOOR - 2, CEILING + 3)
        ax.set_ylim(40, CEILING + 3)
        ax.set_xlabel("Final resources per agent in the myth's source run")
        ax.text(0.03, 0.96, f"{SIZES[size][1].capitalize()}  (r = {rho:.2f}, 30 runs)", transform=ax.transAxes,
                ha="left", va="top", fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Final resources per agent after injection")
    axes[1].legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    outputs = []
    for suffix in ("png", "pdf"):
        target = out_dir / f"myth_injection_by_donor_score.{suffix}"
        fig.savefig(target, dpi=180, bbox_inches="tight")
        outputs.append(target.name)
    plt.close(fig)
    return outputs, correlations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    manifest = json.loads(MANIFEST.read_text())
    all_values, summary, sources, outputs = {}, {}, [], []
    for size in SIZES:
        values, receipt, size_sources = load(size)
        all_values[size] = values
        sources += size_sources
        outputs += bar_figure(size, values, args.out_dir)
        summary[size] = {
            "plan_sha256": receipt["plan_sha256"],
            "completion_receipt_sha256": hashlib.sha256((RUN_ROOT / size / "completion_receipt.json").read_bytes()).hexdigest(),
            "cells": {cell: {"n": 5, "mean": statistics.mean(by_rep.values()), "sd": statistics.stdev(by_rep.values()),
                             "values_by_replicate": [by_rep[rep] for rep in range(5)]} for cell, by_rep in values.items()},
        }
    donor_outputs, correlations = donor_figure(all_values, manifest, args.out_dir)
    summary["donor_score_correlation"] = correlations
    summary["donor_manifest_sha256"] = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    summary["unit"] = "final resources per agent (25 to 75)"
    summary["outputs"] = outputs + donor_outputs
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_output_provenance(args.out_dir, sources, allowed_differences=ALLOWED_DIFFERENCES)
    for name in summary["outputs"]:
        print(args.out_dir / name)
    print(json.dumps(correlations))


if __name__ == "__main__":
    main()
