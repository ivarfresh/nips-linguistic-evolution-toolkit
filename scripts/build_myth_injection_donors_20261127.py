"""Select donor myths for the myth-injection rerun from the paper's main-figure runs.

Rule, fixed before any myth text was read:
- Source: completed mid-tier single-model eight-agent populations with a myth
  task, no defectors, informed negative-only noise of $1 (the September rerun
  and its n=10 extension). Ten runs per family and task order.
- Rank runs by mean final resources per agent; ties broken by replicate id.
- High-cooperation cells: the five highest myth->game runs of the family.
- Low-cooperation Sonnet cell: the five lowest Sonnet runs over both task orders.
- One agent per run, drawn with random.Random(20261127) in a fixed order.
  The early (round 1) and late (round 10) Sonnet cells use the same run and agent.
- Filler texts are reused unchanged from data/phase3/seed_manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    "data/json/noise_experiments/negative_only_crossmodel_reasoning_rerun_20260909",
    "data/json/noise_experiments/table1_n10_extension_20261001",
]
FAMILIES = {"claude": "anthropic/claude-sonnet-4.5", "gemini": "google/gemini-3.7-flash", "gpt": "openai/gpt-5-nano"}
SEED = 20261127
SKIP = (".results.json", ".checkpoint.json", ".error.json")


def load_runs(data_root: Path):
    runs = []
    for source in SOURCES:
        for family in FAMILIES:
            for order in ("myth_game", "game_myth"):
                for folder in sorted((data_root / source).glob(f"negative_only_reasoning_rerun_population_{order}_{family}_n5*")):
                    for path in sorted(folder.rglob("*.json")):
                        if path.name.endswith(SKIP) or "quarantine" in path.parts:
                            continue
                        data = json.loads(path.read_text())
                        meta = data.get("run_metadata") or {}
                        history = data.get("conversation_history") or []
                        noise = meta.get("noise_config") or {}
                        if (
                            meta.get("defector_count")
                            or meta.get("model") != FAMILIES[family]
                            or meta.get("num_agents") != 8
                            or len(history) != 10
                            or (noise.get("range"), noise.get("direction"), noise.get("inform_agents")) != (1.0, "negative", True)
                        ):
                            continue
                        balances = history[-1]["balances"]
                        runs.append({
                            "family": family, "order": order, "rep": meta["replicate_id"],
                            "path": str(path.relative_to(data_root)),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "mean_final": statistics.mean(balances.values()),
                            "joint_final": sum(balances.values()),
                            "history": history,
                        })
    return runs


def seed_entry(run, agent, round_number):
    text = run["history"][round_number - 1]["myths"][agent]
    if not isinstance(text, str) or len(text.split()) < 100:
        raise ValueError(f"Unusable myth: {run['path']} {agent} round {round_number}")
    return {
        "source_run": run["path"], "source_sha256": run["sha256"], "source_model": FAMILIES[run["family"]],
        "source_task_order": run["order"], "source_replicate_id": run["rep"],
        "agent_id": agent, "round": round_number,
        "joint_at_source": run["joint_final"], "mean_final_per_agent_at_source": round(run["mean_final"], 3),
        "word_count": len(text.split()), "tokens": None, "text": text,
    }


def build(data_root: Path):
    runs = load_runs(data_root)
    cells = {}
    for family in FAMILIES:
        for order in ("myth_game", "game_myth"):
            group = [r for r in runs if r["family"] == family and r["order"] == order]
            reps = sorted(r["rep"] for r in group)
            if reps != list(range(10)):
                raise ValueError(f"Expected replicates 0-9 for {family}/{order}, got {reps}")
            cells[(family, order)] = group
    rng = random.Random(SEED)
    agents = [f"Agent_{i}" for i in range(1, 9)]

    def top(group):
        return sorted(group, key=lambda r: (-r["mean_final"], r["rep"]))[:5]

    seeds = {}
    high = top(cells[("claude", "myth_game")])
    picks = [rng.choice(agents) for _ in high]
    seeds["s_start"] = [seed_entry(r, a, 1) for r, a in zip(high, picks)]
    seeds["s_end_plus"] = [seed_entry(r, a, 10) for r, a in zip(high, picks)]
    low = sorted(cells[("claude", "myth_game")] + cells[("claude", "game_myth")], key=lambda r: (r["mean_final"], r["order"], r["rep"]))[:5]
    seeds["s_end_minus"] = [seed_entry(r, rng.choice(agents), 10) for r in low]
    for family, cell in (("gemini", "s_end_plus_gemini"), ("gpt", "s_end_plus_gpt")):
        seeds[cell] = [seed_entry(r, rng.choice(agents), 10) for r in top(cells[(family, "myth_game")])]
    old = json.loads((ROOT / "data/phase3/seed_manifest.json").read_text())
    seeds["s_filler"] = old["seeds"]["s_filler"]
    return {
        "version": 1, "rule": __doc__, "agent_draw_seed": SEED, "sources": SOURCES,
        "runs_considered": {f"{f}/{o}": sorted(round(r["mean_final"], 2) for r in g) for (f, o), g in cells.items()},
        "seeds": seeds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT, help="checkout that holds the gitignored data/json tree")
    parser.add_argument("--output", type=Path, default=ROOT / "data/phase3/seed_manifest_main_figures_20261127.json")
    args = parser.parse_args()
    manifest = build(args.data_root.resolve())
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    for cell, entries in manifest["seeds"].items():
        for i, e in enumerate(entries):
            print(cell, i, e.get("source_task_order"), e.get("source_replicate_id"), e.get("agent_id"), e.get("round"),
                  e.get("mean_final_per_agent_at_source"), e.get("word_count") or len(e["text"].split()))
    print("sha256", hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
