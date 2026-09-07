#!/usr/bin/env python3
"""Describe the exploratory decision-format runs, or compare controlled runs.

The archived September 4 comparison changed myth prompts and retry policy
as well as decision format. It does not isolate a format or memory effect.
Use --exploratory to reproduce that descriptive table. New controlled runs
must pass a complete-condition comparison before any output is written.

  reference            no format line (Aron's 2026-08-25 cross-model cell;
                       Claude wrote ~1,000 chars of strategy per decision)
  json_only            "Reply with the JSON object only."
  reasoning_then_json  "At most two sentences of reasoning, then the JSON."

Per run it reports, for ordinary (non-defector) agents: mean send fraction,
mean return ratio, final balance per agent, and the mean visible length of
game replies. Prints a table with mean (+-std) per cell and writes a CSV.

Usage (from repo root):
  python analyses/decision_format_confound.py
  python analyses/decision_format_confound.py --reference <dir> --cells json_only=<dir> reasoning_then_json=<dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from analyses._shared import (  # noqa: E402
    calculate_return_ratios,
    infer_endowment,
    llm_settings_signature,
    load_simulation_runs,
)
from scripts.write_provenance import build_provenance

EXPLORATORY_REASON = (
    "September 4 runs lack complete conditions. The reference used game-directed "
    "memory-primary myth prompts; both format arms used generic myth prompts. "
    "JSON-only used corrective game retries. Differences cannot be attributed "
    "to output format or retained prose alone."
)
EXPLORATORY_DIFFERENCES = {
    "replicate": "Five separately seeded replicates per condition.",
    "llm.reasoning": "Reference effort was unrecorded; newer runs record off.",
    "protocol.decision_format": "Different requested output formats.",
    "protocol.myth_default_prompt_key": "Uncontrolled change from game-directed to generic myths.",
    "protocol.myth_later_prompt_key": "Uncontrolled change in myth instructions and self-myth repetition.",
    "protocol.myth_injection_mode": "Unrecorded in the reference metadata.",
    "protocol.noise_semantics": "Unrecorded in the reference metadata.",
    "implementation.code_commit": "Different code versions, including the corrective-retry change.",
    "implementation.code_dirty": "Some original runs used a dirty worktree.",
}

REFERENCE_DIR = (
    "data/shared_runs/uploaders/vallinder/data/json/noise_experiments/"
    "negative_only_crossmodel_defectors_n5_20260825/"
    "negative_only_crossmodel_population_myth_game_n5/claude-sonnet-4.5/myth_game/"
    "noisy8_crossmodel_negative_defectors25_twotask_r3"
)
CELL_DIRS = {
    "json_only": (
        "data/json/noise_experiments/fmt_confound_20260904/"
        "fmt_claude_defectors25_myth_game_json_only_n5/claude-sonnet-4.5/myth_game/"
        "noisy8_negative_defectors25_twotask_r3_json_only"
    ),
    "reasoning_then_json": (
        "data/json/noise_experiments/fmt_confound_20260904/"
        "fmt_claude_defectors25_myth_game_reasoning_then_json_n5/claude-sonnet-4.5/myth_game/"
        "noisy8_negative_defectors25_twotask_r3_reasoning_then_json"
    ),
}


def final_jsons(directory: str):
    return sorted(
        str(p)
        for p in Path(directory).glob("*.json")
        if not p.name.endswith((".results.json", ".checkpoint.json", ".error.json"))
    )


import re

DECISION_KEY = re.compile(r"['\"](send|return)['\"]\s*:\s*\$?\s*(-?\d+(?:\.\d+)?)", re.I)
ENDOWMENT = 5.0  # both cells use the $5 endowment (game_params in config)


def ordinary_agents(data):
    """Non-defector agents, from game_data.agent_types (falls back to
    each agent's population_role)."""
    types = data.get("game_data", {}).get("agent_types") or {}
    agents = list(data.get("agents", {}).keys())
    if types:
        return [a for a in agents if types.get(a, "standard") == "standard"]
    return [a for a in agents if data["agents"][a].get("population_role", "standard") == "standard"]


def run_metrics(data):
    """Per-run metrics over ordinary agents.

    Round records carry true amounts keyed by agent: ``sent`` (investors),
    ``received`` / ``returned`` (trustees) and ``roles``. Reply length comes
    from each agent's interaction_history (scripted defector moves excluded).
    """
    ordinary = set(ordinary_agents(data))
    sends, ratios, reply_lens = [], [], []
    for rnd in data.get("conversation_history", []):
        # Population rounds record one entry per dyad with true amounts.
        for dyad in rnd.get("dyads") or []:
            investor, trustee = dyad.get("investor"), dyad.get("trustee")
            if investor in ordinary and dyad.get("sent") is not None:
                sends.append(float(dyad["sent"]))
            if trustee in ordinary:
                got = float(dyad.get("received") or 0.0)
                if got > 0 and dyad.get("returned") is not None:
                    ratios.append(float(dyad["returned"]) / got)
    for aid, agent in data.get("agents", {}).items():
        if aid not in ordinary:
            continue
        for it in agent.get("interaction_history", []):
            if it.get("error"):
                continue
            resp = it.get("response") or {}
            if not isinstance(resp, dict) or resp.get("response_source") == "scripted":
                continue
            content = resp.get("content") or ""
            if DECISION_KEY.search(content):
                reply_lens.append(len(content))
    balances = data.get("game_data", {}).get("balances") or {}
    ordinary_balance = [float(v) for a, v in balances.items() if a in ordinary]
    return {
        "send_fraction": statistics.mean(sends) / ENDOWMENT if sends else float("nan"),
        "return_ratio": statistics.mean(ratios) if ratios else float("nan"),
        "final_balance": statistics.mean(ordinary_balance) if ordinary_balance else float("nan"),
        "reply_chars": statistics.mean(reply_lens) if reply_lens else float("nan"),
        "n_decisions": len(reply_lens),
    }


def summarize(rows):
    out = {}
    for metric in ("send_fraction", "return_ratio", "final_balance", "reply_chars"):
        vals = [r[metric] for r in rows if r[metric] == r[metric]]
        if not vals:
            out[metric] = "n/a"
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out[metric] = f"{mean:.3f} (±{std:.3f})" if metric != "reply_chars" else f"{mean:.0f} (±{std:.0f})"
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reference", default=REFERENCE_DIR)
    parser.add_argument("--cells", nargs="*", default=[f"{k}={v}" for k, v in CELL_DIRS.items()])
    parser.add_argument("--out", default="data/analysis/decision_format_confound_2026_09_04/summary.csv")
    parser.add_argument("--exploratory", action="store_true", help="Acknowledge the archived study's uncontrolled prompt and retry differences")
    args = parser.parse_args()

    cells = {"reference": args.reference}
    for spec in args.cells:
        name, _, path = spec.partition("=")
        cells[name] = path

    allowed = EXPLORATORY_DIFFERENCES if args.exploratory else {
        "replicate": "Independent replicates within each format condition.",
        "protocol.game.decision_format": "The format instruction is the manipulated variable.",
    }
    paths_by_cell = {name: final_jsons(directory) for name, directory in cells.items()}
    if any(not paths for paths in paths_by_cell.values()):
        parser.error("Every comparison cell must contain completed final runs")
    all_paths = [path for paths in paths_by_cell.values() for path in paths]
    checked_runs = load_simulation_runs(
        all_paths, allowed_differences=allowed,
        legacy_reason=EXPLORATORY_REASON if args.exploratory else None,
    )
    if args.exploratory:
        print("EXPLORATORY: " + EXPLORATORY_REASON)

    all_rows = []
    for name, directory in cells.items():
        paths = final_jsons(directory)
        if not paths:
            print(f"{name}: no runs in {directory}")
            continue
        runs = {path: checked_runs[path] for path in paths}
        sigs = {str(llm_settings_signature(d)) for d in runs.values()}
        for path, data in runs.items():
            row = {"cell": name, "run": os.path.basename(path), **run_metrics(data)}
            all_rows.append(row)
        print(f"{name}: {len(paths)} runs; settings signatures: {sigs}")

    print()
    print(f"{'cell':22s} {'n':>3s} {'send frac':>18s} {'return ratio':>18s} {'final balance':>18s} {'reply chars':>14s}")
    for name in cells:
        rows = [r for r in all_rows if r["cell"] == name]
        if not rows:
            continue
        s = summarize(rows)
        print(f"{name:22s} {len(rows):3d} {s['send_fraction']:>18s} {s['return_ratio']:>18s} {s['final_balance']:>18s} {s['reply_chars']:>14s}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()) if all_rows else ["cell"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwrote {out}")
    provenance = build_provenance(
        all_paths, allowed_differences=allowed,
        legacy_reason=EXPLORATORY_REASON if args.exploratory else None,
        output_dir=out.parent,
    )
    (out.parent / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
