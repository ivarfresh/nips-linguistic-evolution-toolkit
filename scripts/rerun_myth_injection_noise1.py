"""Myth-injection ablation at the main-figure noise setting.

Same apparatus and seven cells as scripts/rerun_slide678.py (populations) and
scripts/rerun_slide678_dyad.py (dyads), with three changes:

- Noise is informed negative-only U(-1, 0), as in the main figures. The
  September batches used uninformed U(-5, 0).
- Injected myths come from the paper's main-figure population runs
  (scripts/build_myth_injection_donors_20261127.py), not the June donors.
- An eighth cell injects late myths from low-cooperation GPT-5 Nano populations.

Modes: prepare freezes the plan; run executes pending runs; audit prints cell
means; supervise runs batches until all finals validate and writes a receipt.
No automatic resampling of failed runs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_phase3_seeded_cells import _load_config, _make_combo
from scripts import rerun_slide678 as population
from src.experiment_condition import condition_from_run, digest, read_final_run
from src.llm_settings import RequestPlan, resolve_request_plan
from src.simulation import run_simulation
from src.utils import DIRECT_MODEL_ALIASES

PARAMS = "phase3_8agent_anon_neg5_myth_only"
NOISE = {"type": "uniform", "range": 1.0, "direction": "negative", "applies_to": "both", "inform_agents": True}
CELLS = ["baseline", "s_end_plus", "s_start", "s_end_minus", "s_filler", "s_end_plus_gemini", "s_end_plus_gpt", "s_end_minus_gpt"]
REPS = 5
RUNS = len(CELLS) * REPS
NUM_TURNS = 10
MANIFEST = ROOT / "data/phase3/seed_manifest_main_figures_20261127.json"
OUTPUT_ROOT = ROOT / "data/json/noise_experiments/myth_injection_noise1_20261127"
SIZES = {"population": 8, "dyad": 2}
# Planning allowance per batch; September recorded $27.50 (population) and $8.49 (dyad).
ESTIMATE_USD = {"population": 40, "dyad": 13}
STOP_ABOVE_USD = {"population": 60, "dyad": 20}
PLAN_KEY = "myth_injection_noise1_plan_sha256"


def source_hashes():
    files = [Path(__file__), Path(population.__file__), ROOT / "experiments/run_phase3_seeded_cells.py",
             ROOT / "scripts/build_myth_injection_donors_20261127.py"]
    files += [p for folder in ("src", "games") for p in sorted((ROOT / folder).glob("*.py"))]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def prepare(size):
    config = _load_config()
    manifest = json.loads(MANIFEST.read_text())
    request = resolve_request_plan(config["base_models"]["claude_sonnet_45"], config["llm_profiles"]["september8_claude"], DIRECT_MODEL_ALIASES)
    combos = []
    # Interleave cells so the first two jobs are baseline and late Sonnet.
    for rep in range(REPS):
        for cell in CELLS:
            seed = {"text": ""} if cell == "baseline" else manifest["seeds"][cell][rep]
            combo = _make_combo(cell, seed, rep, config, PARAMS, "claude_sonnet_45")
            combo["game_params"]["noise_config"] = dict(NOISE)
            combo["game_params"]["num_agents"] = SIZES[size]
            combo["seed_meta"].update({k: seed.get(k) for k in (
                "source_sha256", "source_model", "source_task_order", "source_replicate_id",
                "mean_final_per_agent_at_source", "word_count", "topic")})
            combos.append(combo)
    return {
        "version": 1, "size": size, "num_agents": SIZES[size], "request": request.as_dict(), "combos": combos,
        "donor_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "source_hashes": source_hashes(), "estimate_usd": ESTIMATE_USD[size],
        "differences_from_september_batch": [
            "Informed negative-only U(-1,0) communication noise instead of uninformed U(-5,0).",
            "Donor myths from the main-figure population runs instead of the June donors; filler texts unchanged.",
            "Added cell: late myths from low-cooperation GPT-5 Nano populations.",
        ],
        "preserved": [
            "the seven September cells, five donor/run replicates, ten game-only rounds",
            "myth-only context with the identical text re-injected each round; no game history block",
            "hidden names, balanced roles, fresh unseeded pairing and noise draws",
            "pinned native Anthropic September Sonnet 4.5 profile; same-prompt retry once",
        ],
    }


def validate_memory(data, combo, num_agents):
    if data["task_order"] != ["game"] or [r["round"] for r in data["conversation_history"]] != list(range(1, NUM_TURNS + 1)):
        raise ValueError("Not a complete ten-round game-only final")
    if len(data["agents"]) != num_agents:
        raise ValueError("Wrong group size")
    # This runner path does not record noise_config in run_metadata; check the draws.
    draws = [action["noise"] for row in data["conversation_history"] for action in row["actions"].values()]
    if not draws or min(draws) < -NOISE["range"] or max(draws) > 0 or not any(draws):
        raise ValueError(f"Noise draws outside U(-1,0): min={min(draws, default=None)} max={max(draws, default=None)}")
    seeded = combo["seed_type"] != "baseline"
    game, _ = population.game_and_writer(combo)
    game.configure_agents(list(data["agents"]), data["run_metadata"]["agent_names"])
    for agent_id, agent in data["agents"].items():
        system = game.get_system_prompt(agent_id, SimpleNamespace(initial_bias=""))
        if "differ" not in system.lower():
            raise ValueError("System prompt lacks the noise notice")
        prefix = [{"role": "system", "content": system}]
        if seeded:
            prefix += [{"role": "user", "content": combo["seed_user_prompt"]}, {"role": "assistant", "content": combo["seed_text"]}]
        rounds = []
        for event in agent["interaction_history"]:
            messages = event["messages_sent"]
            if messages[:-1] != prefix or messages[-1] != {"role": "user", "content": event["prompt"]}:
                raise ValueError(f"Context mismatch: {combo['seed_type']} {agent_id}")
            if event["metadata"]["task"] != "game":
                raise ValueError("Unexpected myth-generation call")
            if "error" not in event:
                rounds.append(event["metadata"]["round"])
        if rounds != list(range(1, NUM_TURNS + 1)):
            raise ValueError(f"Missing/duplicated agent decisions: {agent_id}: {rounds}")


def final_path(output, combo):
    return output / combo["seed_type"] / f"rep{combo['rep']:02d}.json"


def verify_final(path, plan, combo):
    data = read_final_run(path)
    condition = condition_from_run(data)
    if data["run_metadata"].get(PLAN_KEY) != digest(plan):
        raise ValueError(f"Final belongs to a different frozen plan: {path}")
    identity = f"{combo['seed_type']}/rep{combo['rep']:02d}"
    if condition["llm"] != plan["request"] or condition["replicate"]["identity"] != identity:
        raise ValueError(f"Final request/identity mismatch: {path}")
    validate_memory(data, combo, plan["num_agents"])
    return data


def run_one(plan, combo, output):
    path = final_path(output, combo)
    if path.exists():
        verify_final(path, plan, combo)
        return {"status": "existing", "path": str(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = path.with_suffix(".checkpoint.json")
    game, writer = population.game_and_writer(combo)
    baseline = combo["seed_type"] == "baseline"
    try:
        with path.with_suffix(".log").open("a") as log, redirect_stdout(log), redirect_stderr(log):
            result = run_simulation(
                game=game, model=combo["model"], temperature="default", num_turns=NUM_TURNS,
                num_agents=plan["num_agents"], memory_capacity=combo["game_params"]["memory_capacity"],
                agent_biases="", myth_writer=writer, task_order=["game"],
                checkpoint_path=str(checkpoint), checkpoint_every=1,
                resume_from=str(checkpoint) if checkpoint.exists() else None,
                seed_myth=None if baseline else combo["seed_text"],
                seed_user_prompt=None if baseline else combo["seed_user_prompt"],
                chat_memory_mode="myth_only", seed_reinject=not baseline,
                request_plan=RequestPlan(json.dumps(plan["request"])),
                run_identity=f"{combo['seed_type']}/rep{combo['rep']:02d}",
                game_response_retry_policy="repeat_same_prompt_once",
                run_metadata_extra={PLAN_KEY: digest(plan), "phase3_seed_type": combo["seed_type"],
                                    "phase3_rep": combo["rep"], "phase3_seed_meta": combo["seed_meta"]},
            )
            validate_memory(result.to_state(), combo, plan["num_agents"])
            result.save_state(str(path))
            verify_final(path, plan, combo)
        return {"status": "complete", "path": str(path)}
    except Exception:
        return {"status": "failed", "path": str(path), "error": traceback.format_exc()}


def usage_of(data):
    usages = [(event.get("response") or {}).get("usage") or {}
              for agent in data["agents"].values() for event in agent["interaction_history"]]
    return {"calls": len(usages),
            "input_tokens": sum(u.get("input_tokens") or 0 for u in usages),
            "output_tokens": sum(u.get("output_tokens") or 0 for u in usages)}


def audit(output, plan):
    entries, values = [], defaultdict(list)
    for combo in plan["combos"]:
        path = final_path(output, combo)
        if not path.exists():
            continue
        data = verify_final(path, plan, combo)
        usage = usage_of(data)
        usage["standard_rate_usd"] = usage["input_tokens"] * 3e-6 + usage["output_tokens"] * 15e-6
        balance = sum(data["conversation_history"][-1]["balances"].values())
        values[combo["seed_type"]].append(balance)
        entries.append({"path": str(path.relative_to(output)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "cell": combo["seed_type"], "rep": combo["rep"], "joint_resources": balance, **usage})
    cost = sum(e["standard_rate_usd"] for e in entries)
    estimate = plan["estimate_usd"]
    projected = max(estimate, cost / len(entries) * RUNS * 1.25) if entries else estimate
    cells = {cell: {"n": len(v), "mean": statistics.mean(v), "sd": statistics.stdev(v) if len(v) > 1 else None, "values": v}
             for cell, v in values.items()}
    return {"plan_sha256": digest(plan), "completed": len(entries), "planned": RUNS,
            "standard_rate_usd": cost, "projected_usd_with_margin": projected,
            "cost_caveat": "Recorded calls in finals only; not invoice cost. Failed transport attempts may be unrecorded.",
            "cells": cells, "finals": entries}


def write_status(output, status, **extra):
    value = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat(), **extra}
    target = output / "supervisor_status.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps(value), flush=True)


def supervise(output, plan, workers):
    lock = output / "supervisor.lock"
    with lock.open("x"):
        pass
    try:
        while True:
            if source_hashes() != plan["source_hashes"]:
                raise RuntimeError("Frozen implementation changed")
            receipt = audit(output, plan)
            if receipt["completed"] == RUNS:
                (output / "completion_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
                write_status(output, "complete", completed=RUNS, standard_rate_usd=receipt["standard_rate_usd"], cells=receipt["cells"])
                return
            if receipt["projected_usd_with_margin"] > STOP_ABOVE_USD[plan["size"]]:
                write_status(output, "approval_required", completed=receipt["completed"], projected_usd=receipt["projected_usd_with_margin"])
                return
            write_status(output, "running", completed=receipt["completed"], pending=RUNS - receipt["completed"],
                         projected_usd=receipt["projected_usd_with_margin"], standard_rate_usd=receipt["standard_rate_usd"])
            result = subprocess.run([sys.executable, str(Path(__file__)), "run", "--size", plan["size"], "--output", str(output),
                                     "--workers", str(workers), "--limit", str(workers)], cwd=ROOT)
            if result.returncode:
                raise RuntimeError("Batch failed; inspect error snapshots, no automatic resampling")
    except Exception as exc:
        write_status(output, "stopped", error=str(exc))
        raise
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "audit", "supervise"])
    parser.add_argument("--size", choices=sorted(SIZES), required=True)
    parser.add_argument("--output", type=Path, help="batch folder; default data/json/noise_experiments/myth_injection_noise1_20261127/<size>")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    output = (args.output or OUTPUT_ROOT / args.size).resolve()
    plan_path = output / "plan.json"
    if args.mode == "prepare":
        plan = prepare(args.size)
        output.mkdir(parents=True, exist_ok=True)
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("Refusing to overwrite a different frozen plan")
        if not plan_path.exists():
            plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        calls = RUNS * SIZES[args.size] * NUM_TURNS
        print(f"Prepared {RUNS} runs / {calls} calls; plan={plan_path}; sha256={digest(plan)}")
        return
    plan = json.loads(plan_path.read_text())
    if plan["size"] != args.size:
        raise ValueError("Plan belongs to the other group size")
    if source_hashes() != plan["source_hashes"]:
        raise ValueError("Implementation changed since plan was frozen")
    if args.mode == "supervise":
        supervise(output, plan, args.workers)
        return
    pending, results = [], {}
    for combo in plan["combos"]:
        path = final_path(output, combo)
        if path.exists():
            data = verify_final(path, plan, combo)
            results.setdefault(combo["seed_type"], []).append(sum(data["conversation_history"][-1]["balances"].values()))
        else:
            pending.append(combo)
    print(f"Validated complete={RUNS - len(pending)}/{RUNS}; pending={len(pending)}", flush=True)
    if args.mode == "audit":
        for cell, values in results.items():
            sd = statistics.stdev(values) if len(values) > 1 else float("nan")
            print(f"{cell}: n={len(values)} mean={statistics.mean(values):.2f} sd={sd:.2f}")
        return
    selected = pending[:args.limit] if args.limit else pending
    print(f"Resolved MODEL={plan['request']['provider_model']} N={len(selected)} WORKERS={args.workers} "
          f"EST_COST=${plan['estimate_usd'] * len(selected) / RUNS:.2f}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, plan, combo, output) for combo in selected]
        failed = False
        for future in as_completed(futures):
            outcome = future.result()
            print(json.dumps(outcome), flush=True)
            failed |= outcome["status"] == "failed"
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
