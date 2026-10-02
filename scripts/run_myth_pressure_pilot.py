"""Myth-pressure pilot (2026-09-28): 20 Sonnet 4.5 myth-first dyads, word budget x council.

Dry-run by default. `--smoke` runs one full 10-round tight+council replicate into a
separate folder so the compressed-myth regime is probed with the exact runtime
prompts before the pilot. `--execute` runs the missing pilot finals; every final is
audited (pinned request, clean code, delivery and council records) into a receipt.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_myth_pressure_config import SET_NAME, SEPTEMBER_SET, SEPTEMBER_GAME_PARAMS, SCHEDULES, COUNCIL_EXCHANGES, arm_names, game_params_name
from scripts.run_noisy_missing import load_combinations, expected_output_path, check_existing_final, run_missing_job
from scripts.rerun_negative_only_crossmodel import TRUNCATION_REASONS
from experiments.run_noisy_batch import build_noisy_protocol
from src.utils import is_exhausted_quota

CONFIG = ROOT / "config/myth_pressure_pilot_20260928.yaml"
OUTPUT = "myth_pressure_pilot_20260928"
SMOKE_OUTPUT = OUTPUT + "_smoke"
SONNET_RATES = (3, 15)  # USD per million input / output tokens
N = 20


def plan():
    with contextlib.redirect_stdout(io.StringIO()):
        combos = load_combinations(SET_NAME, str(CONFIG))
        september = load_combinations(SEPTEMBER_SET, str(ROOT / "config/experiments_noisy.yaml"))
    base = {c["replicate_id"]: c for c in september if c["game_params_name"] == SEPTEMBER_GAME_PARAMS}
    assert len(combos) == N and sorted(base) == [0, 1, 2, 3, 4]
    jobs = []
    for index, combo in enumerate(combos):
        ref = base[combo["replicate_id"]]
        assert combo["task_order"] == ["myth", "game"] and combo["model"] == ref["model"]
        assert combo["request_plan"].as_dict() == ref["request_plan"].as_dict()
        assert {k: v for k, v in combo["game_params"].items() if k != "myth_pressure"} == ref["game_params"]
        assert combo["game_params"]["noise_config"]["inform_agents"] is True
        arm = next(a for a in arm_names() if combo["game_params_name"] == game_params_name(a))
        budget, council = arm.split("_")
        pressure = combo["myth_pressure"]
        assert pressure["word_budget_schedule"] == SCHEDULES[budget]
        assert pressure["council_exchanges"] == COUNCIL_EXCHANGES[council]
        game, writer = build_noisy_protocol(combo, index)
        assert writer.pressure["word_budget_schedule"] == SCHEDULES[budget]
        assert not game.defector_agent_ids and not game.punishment_enabled
        jobs.append((SET_NAME, index, combo, expected_output_path(combo, SET_NAME, index, OUTPUT), arm))
    assert len({str(j[3]) for j in jobs}) == N
    # Council arms first (longer), replicates interleaved.
    jobs.sort(key=lambda j: (0 if j[4].endswith("_council") else 1, j[2]["replicate_id"]))
    return jobs


def audit(job, path=None):
    name, index, combo, planned, arm = job
    path = path or planned
    check_existing_final(path, combo)
    data = json.loads(path.read_text())
    meta = data["run_metadata"]
    assert not meta["code_dirty"], path
    assert meta["llm_request"] == combo["request_plan"].as_dict()
    assert meta["experiment_condition"]["protocol"]["myth"]["pressure"]["word_budget_schedule"] == combo["myth_pressure"]["word_budget_schedule"]
    exchanges = combo["myth_pressure"]["council_exchanges"]
    history = data["conversation_history"]
    assert len(history) == 10
    for entry in history:
        assert set(entry["myth_delivery"]) == {"Agent_1", "Agent_2"}, (path, entry["round"])
        has_council = bool(entry.get("council"))
        assert has_council == (exchanges > 0 and entry["round"] < 10), (path, entry["round"])
        for council in (entry.get("council") or {}).values():
            assert len(council["messages"]) == 2 * exchanges
            assert all(m["content"].strip() for m in council["messages"])
    calls, council_calls, cost = 0, 0, 0.0
    rate_in, rate_out = SONNET_RATES
    for agent in data["agents"].values():
        for event in agent.get("interaction_history", []):
            response = event.get("response") or {}
            if response.get("response_source", "llm") != "llm":
                continue
            usage = response.get("usage") or {}
            assert usage.get("request_settings") == combo["request_plan"].as_dict()
            assert usage.get("outcome") == "complete" and usage.get("finish_reason") not in TRUNCATION_REASONS
            calls += 1
            council_calls += event["metadata"].get("task") == "council"
            cost += ((usage.get("input_tokens") or 0) * rate_in + (usage.get("output_tokens") or 0) * rate_out) / 1e6
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "arm": arm, "replicate_id": combo["replicate_id"], "calls": calls,
            "council_calls": council_calls, "standard_rate_usd": cost}


def execute(pending, output, workers, total):
    os.environ["HF_DATASET_AUTO_UPLOAD"] = "0"
    os.environ["TRUST_BATCH_QUIET"] = "1"
    logdir = str(ROOT / "data/json/noise_experiments" / output / "worker_logs")
    receipts = []
    for attempt in range(1, 11):
        if not pending:
            break
        failed, quota = [], False
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_missing_job, j[2], j[0], j[1], output, logdir): j for j in pending}
            for future in as_completed(futures):
                job = futures[future]
                if future.cancelled():
                    failed.append(job)
                    continue
                try:
                    result = future.result()
                    if not result.get("success") and is_exhausted_quota(result.get("error", "")):
                        quota = True
                        for queued in futures:
                            queued.cancel()
                    if not result.get("success"):
                        raise RuntimeError(f"Worker failed; inspect {result.get('worker_log')}")
                    path = expected_output_path(job[2], job[0], job[1], output)
                    receipt = audit(job, path)
                    receipts.append(receipt)
                    print(f"COMPLETE {len(receipts)}/{total} {job[4]} rep={job[2]['replicate_id']} cost=${receipt['standard_rate_usd']:.3f}", flush=True)
                except Exception as error:
                    print(f"FAILED {job[4]} rep={job[2]['replicate_id']} {type(error).__name__}: {error}", flush=True)
                    if expected_output_path(job[2], job[0], job[1], output).exists():
                        raise  # never resample a final that fails validation
                    failed.append(job)
        if quota:
            raise RuntimeError("Provider credits exhausted; completed finals preserved.")
        pending = failed
        if pending:
            if attempt == 10:
                raise RuntimeError(f"{len(pending)} runs failed after ten attempts")
            workers = 1
            print(f"CONTINUATION PENDING={len(pending)} WORKERS=1 ATTEMPT={attempt + 1}", flush=True)
            time.sleep(min(60, 2 ** attempt))
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    jobs = plan()
    if args.smoke or args.execute:
        assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(), "Clean checkout required"
    if args.smoke:
        job = next(j for j in jobs if j[4] == "tight_council" and j[2]["replicate_id"] == 0)
        print("PREFLIGHT MODEL=anthropic/claude-sonnet-4.5 N=1 (tight_council rep0, 10 rounds) WORKERS=1 EST_COST=$2", flush=True)
        receipts = execute([job], SMOKE_OUTPUT, 1, 1)
        target = ROOT / "data/json/noise_experiments" / SMOKE_OUTPUT / "smoke_receipt.json"
        target.write_text(json.dumps({"runs": len(receipts), "finals": receipts}, indent=2) + "\n")
        print(f"SMOKE PASSED receipt={target}", flush=True)
        return
    receipts, pending = [], []
    for job in jobs:
        (receipts.append(audit(job)) if job[3].exists() else pending.append(job))
    print(f"VALIDATED N={N} ARMS={arm_names()} EXISTING={len(receipts)} PENDING={len(pending)}", flush=True)
    if args.audit_only:
        assert not pending
    elif not args.execute:
        return
    else:
        print(f"PREFLIGHT MODEL=anthropic/claude-sonnet-4.5 N={len(pending)} WORKERS={args.workers} EST_COST=$25 (~$0.8 per no-council run, ~$1.6 per council run)", flush=True)
        receipts += execute(pending, OUTPUT, args.workers, N)
    target = ROOT / "data/json/noise_experiments" / OUTPUT / "completion_receipt.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"runs": len(receipts), "standard_rate_usd": sum(r["standard_rate_usd"] for r in receipts), "finals": receipts}, indent=2) + "\n")
    print(f"AUDIT PASSED {len(receipts)}/{N}; receipt={target}", flush=True)


if __name__ == "__main__":
    main()
