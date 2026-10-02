"""Cultural transmission pilot (2026-10-02): Sonnet 4.5 8-agent myth-first populations on a
shared myth board, Agent_1's round-1 myth planted. Dry-run by default.

`--arm planted` (or `control`, or `all`) picks the arm; `--reps 0` limits replicates (the first
launch runs replicate 0 alone as the smoke run). `--execute` runs the missing finals; every final
is audited (pinned request, clean code, planted myth, board records) into a receipt.
Config: scripts/build_cultural_transmission_config.py. Plan:
docs/research/cultural_transmission_pilot_2026-10-02.md.
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
from scripts.build_cultural_transmission_config import (  # noqa: E402
    ADDED, BOARD_LATER, PLANT_AGENT, REPLICATES, SEED_MYTH, SEPTEMBER_GAME_PARAMS, SEPTEMBER_SET, TARGET as CONFIG,
    game_params_name)
from scripts.run_noisy_missing import load_combinations, expected_output_path, check_existing_final, run_missing_job  # noqa: E402
from scripts.rerun_negative_only_crossmodel import TRUNCATION_REASONS  # noqa: E402
from experiments.run_noisy_batch import build_noisy_protocol  # noqa: E402
from src.utils import is_exhausted_quota  # noqa: E402

OUTPUT = "cultural_transmission_pilot_20261002"
SONNET_RATES = (3, 15)  # USD per million input / output tokens
# $3.41: mean Sonnet 8-agent myth->game run (table1_n10_extension_20261001 receipt). Board: by
# round r it holds 8(r-1) myths of ~300 tokens, read once per agent per round (8 x 45 x 300 tokens).
EST_RUN_USD = 3.41 + 8 * (8 * 45 * 300) * SONNET_RATES[0] / 1e6


def set_name(arm):
    return f"cultural_transmission_{arm}_population_myth_game_claude"


def plan(arms):
    with contextlib.redirect_stdout(io.StringIO()):
        september = load_combinations(SEPTEMBER_SET, str(ROOT / "config/experiments_noisy.yaml"))
        later = __import__("yaml").safe_load(CONFIG.read_text())["prompt_templates"][BOARD_LATER]
        sets = {arm: load_combinations(set_name(arm), str(CONFIG)) for arm in arms}
    base = {c["replicate_id"]: c for c in september if c["game_params_name"] == SEPTEMBER_GAME_PARAMS}
    jobs = []
    for arm, combos in sets.items():
        assert [c["replicate_id"] for c in combos] == REPLICATES, arm
        for index, combo in enumerate(combos):
            ref = base[combo["replicate_id"]]
            assert combo["task_order"] == ["myth", "game"] and combo["model"] == ref["model"]
            assert combo["request_plan"].as_dict() == ref["request_plan"].as_dict()
            extra = {"myth_board", "myth_plant"}
            assert {k: v for k, v in combo["game_params"].items() if k not in extra} == ref["game_params"]
            assert combo["game_params_name"] == game_params_name(arm)
            assert combo["myth_board"] == "persistent" and combo["myth_writing_later_rounds"] == later
            assert combo["myth_writing_default"] == ref["myth_writing_default"]
            assert combo["myth_plant"] == {"agent": PLANT_AGENT, "text": f"{SEED_MYTH}\n\n{ADDED[arm]}"}
            game, writer = build_noisy_protocol(combo, index)
            assert writer.board == "persistent" and writer.plant["agent"] == PLANT_AGENT
            assert not game.defector_agent_ids and not game.punishment_enabled
            jobs.append((set_name(arm), index, combo, expected_output_path(combo, set_name(arm), index, OUTPUT), arm))
    assert len({str(j[3]) for j in jobs}) == len(jobs)
    return jobs


def audit(job, path=None):
    name, index, combo, planned, arm = job
    path = path or planned
    check_existing_final(path, combo)
    data = json.loads(path.read_text())
    meta = data["run_metadata"]
    assert not meta["code_dirty"], path
    assert meta["llm_request"] == combo["request_plan"].as_dict()
    myth_protocol = meta["experiment_condition"]["protocol"]["myth"]
    assert myth_protocol["plant"] == combo["myth_plant"] and myth_protocol["board"] == "persistent"
    history = data["conversation_history"]
    assert len(history) == 10
    first = history[0]
    assert first["myths"][PLANT_AGENT].strip() and first["myth_responses"][PLANT_AGENT]["response_source"] == "planted"
    assert first["myth_responses"][PLANT_AGENT]["content"] == combo["myth_plant"]["text"]
    for entry in history[1:]:
        r = entry["round"]
        exposures = entry.get("myth_exposures") or {}
        assert set(exposures) == set(data["agents"]), (path, r)
        for record in exposures.values():
            assert record.get("board") == "persistent" and len(record["board_items"]) == 8 * (r - 1), (path, r)
    calls, cost = 0, 0.0
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
            cost += ((usage.get("input_tokens") or 0) * rate_in + (usage.get("output_tokens") or 0) * rate_out) / 1e6
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "arm": arm, "replicate_id": combo["replicate_id"], "calls": calls, "standard_rate_usd": cost}


def execute(pending, workers, total):
    os.environ["HF_DATASET_AUTO_UPLOAD"] = "0"
    os.environ["TRUST_BATCH_QUIET"] = "1"
    logdir = str(ROOT / "data/json/noise_experiments" / OUTPUT / "worker_logs")
    receipts = []
    for attempt in range(1, 11):
        if not pending:
            break
        failed, quota = [], False
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_missing_job, j[2], j[0], j[1], OUTPUT, logdir): j for j in pending}
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
                    receipt = audit(job, expected_output_path(job[2], job[0], job[1], OUTPUT))
                    receipts.append(receipt)
                    print(f"COMPLETE {len(receipts)}/{total} {job[4]} rep={job[2]['replicate_id']} cost=${receipt['standard_rate_usd']:.3f}", flush=True)
                except Exception as error:
                    print(f"FAILED {job[4]} rep={job[2]['replicate_id']} {type(error).__name__}: {error}", flush=True)
                    if expected_output_path(job[2], job[0], job[1], OUTPUT).exists():
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
    parser.add_argument("--arm", choices=[*ADDED, "all"], required=True)
    parser.add_argument("--reps", type=int, nargs="*", default=REPLICATES)
    parser.add_argument("--workers", type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    arms = list(ADDED) if args.arm == "all" else [args.arm]
    jobs = [j for j in plan(arms) if j[2]["replicate_id"] in args.reps]
    receipts, pending = [], []
    for job in jobs:
        (receipts.append(audit(job)) if job[3].exists() else pending.append(job))
    print(f"VALIDATED ARMS={arms} REPS={args.reps} N={len(jobs)} EXISTING={len(receipts)} PENDING={len(pending)}", flush=True)
    print(f"PREFLIGHT MODEL=anthropic/claude-sonnet-4.5 (September profile, thinking 8192) N={len(pending)} "
          f"WORKERS={min(args.workers, max(len(pending), 1))} EST_COST=${len(pending) * EST_RUN_USD:.2f}", flush=True)
    if args.audit_only:
        assert not pending
    elif not args.execute:
        print("DRY RUN: pass --execute to launch", flush=True)
        return
    else:
        assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(), "Clean checkout required"
        receipts += execute(pending, min(args.workers, len(pending)) or 1, len(jobs))
    target = ROOT / "data/json/noise_experiments" / OUTPUT / "completion_receipt.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(target.read_text())["finals"] if target.exists() else []
    finals = {r["path"]: r for r in [*old, *receipts]}
    target.write_text(json.dumps({"runs": len(finals), "standard_rate_usd": sum(r["standard_rate_usd"] for r in finals.values()),
                                  "finals": list(finals.values())}, indent=2) + "\n")
    print(f"AUDIT PASSED {len(receipts)}/{len(jobs)}; receipt={target}", flush=True)


if __name__ == "__main__":
    main()
