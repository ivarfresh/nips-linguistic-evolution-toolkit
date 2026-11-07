#!/usr/bin/env python3
"""Myth board pilot (2026-11-07): board runs and the Sonnet 5.5 baseline cell; dry-run by default.

--stage board (default): 10 Sonnet 4.5 + 10 GPT-5 Nano board populations.
--stage sonnet55_baseline: 10 Claude Sonnet 5.5 partner-myth populations (no board), seed twins
of the September Sonnet 4.5 cells, to see whether Sonnet 5.5 has headroom before any board run.

Twenty 8-agent runs from the four myth_board_pilot_* sets in config/experiments_noisy.yaml:
Sonnet 4.5 and GPT-5 Nano, game->myth and myth->game, replicates 0-4. Each run is the
seed twin of its September partner-myth baseline (negative_only_reasoning_rerun_*_n5):
same pairings and noise draws, same request plan, and the only differences are the
myth later-round template (board instead of partner myth) and the game params
(myth_board: persistent_game, which also pastes the board ahead of every game decision
from round 2; src/myth_writer.py).

The plan step proves this before any paid call. The audit step re-checks every final:
request settings, complete calls, the board in every round-2+ game and myth prompt with
8*(r-1) myths, no board text in chat memory, and prices the run from recorded usage.
Design: docs/research/myth_board_design_v2_20261106.md.
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
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_noisy_missing import load_combinations, expected_output_path, check_existing_final, run_missing_job  # noqa: E402
from scripts.rerun_negative_only_crossmodel import TRUNCATION_REASONS  # noqa: E402
from experiments.run_noisy_batch import build_noisy_protocol, resolve_protocol_seeds  # noqa: E402
from src.myth_writer import MYTH_BOARD_GAME_BLOCK  # noqa: E402
from src.utils import is_exhausted_quota  # noqa: E402

CONFIG = ROOT / "config/experiments_noisy.yaml"
OUTPUT = "myth_board_pilot_20261107"
BOARD_KIND = "persistent_game"
BOARD_TEMPLATE = "myth_writing_later_rounds_board_memory_primary"
BOARD_PARAMS = "noisy8_crossmodel_negative_twotask_board_r3"
REPLICATES = [0, 1, 2, 3, 4]
RATES = {"anthropic": (3, 15), "openai": (0.05, 0.4)}
# Mean standard-rate cost of a September 8-agent two-task baseline run, from the recorded
# usage of the 10 runs per cell (2026-11-07), plus the board: 8 agents x 2 calls x
# sum_{r=2..10} 8(r-1) = 360 myths x ~300 tokens = ~1.73M extra input tokens per run.
BASE_USD = {"anthropic": 3.35, "openai": 0.44}
BOARD_INPUT_TOKENS = 8 * 2 * 360 * 300
GAME_BOARD_HEAD = MYTH_BOARD_GAME_BLOCK.split("\n")[0].split("{")[0]
# Sonnet 5.5 at $2/$10 per M with the Sonnet 4.5 token profile (~560k in, ~110k out).
SONNET55_BASE_USD = 2.20

# Stages. ``board``: the 20 board runs (seed twins of the September baselines; inputs
# may differ only in the board template and game params). ``sonnet55_baseline``: the
# September Sonnet 4.5 partner-myth cells rerun on Claude Sonnet 5.5 (2026-11-07,
# "baselines first, then decide"); inputs may differ only in the model and request.
STAGES = {
    "board": {
        "sets": [(f"myth_board_pilot_population_{shape}_{model}_n5",
                  f"negative_only_reasoning_rerun_population_{shape}_{model}_n5", provider)
                 for shape in ("game_myth", "myth_game")
                 for model, provider in (("claude", "anthropic"), ("gpt", "openai"))],
        "allowed_diffs": {"game_params", "myth_board", "myth_writing_later_rounds"},
        "board": True,
        "runs": 20,
    },
    "sonnet55_baseline": {
        "sets": [(f"sonnet55_baseline_population_{shape}_n5",
                  f"negative_only_reasoning_rerun_population_{shape}_claude_n5", "anthropic")
                 for shape in ("game_myth", "myth_game")],
        "allowed_diffs": {"model", "llm_request"},
        "board": False,
        "runs": 10,
    },
}
SONNET55_MODEL = "claude-sonnet-5-5"


def _quiet(name):
    with contextlib.redirect_stdout(io.StringIO()):
        return load_combinations(name, str(CONFIG))


def _seeds(c):
    return resolve_protocol_seeds(c["game_params"], c)


def plan(stage):
    spec = STAGES[stage]
    jobs = []
    for name, src, provider in spec["sets"]:
        combos = _quiet(name)
        twins = {o["replicate_id"]: o for o in _quiet(src)
                 if o["game_params"].get("defector_ratio", 0) == 0
                 and o["game_params"].get("random_defection_probability", 0) == 0}
        assert [c["replicate_id"] for c in combos] == REPLICATES, (name, [c["replicate_id"] for c in combos])
        for i, c in enumerate(combos):
            twin = twins[c["replicate_id"]]
            inputs, ref = c["comparison_inputs"], twin["comparison_inputs"]
            differing = sorted(k for k in set(inputs) | set(ref) if inputs.get(k) != ref.get(k))
            assert set(differing) <= spec["allowed_diffs"], (name, "unexpected differences from the baseline twin", differing)
            assert set(differing) == spec["allowed_diffs"], (name, "expected differences missing", differing)
            request = c["request_plan"].as_dict()
            assert request["provider"] == provider
            assert c["myth_default_prompt_key"] == twin["myth_default_prompt_key"]
            if spec["board"]:
                gp, gp_ref = dict(c["game_params"]), dict(twin["game_params"])
                assert gp.pop("myth_board") == BOARD_KIND and "myth_board" not in gp_ref and gp == gp_ref, (name, "game params")
                assert c["game_params_name"] == BOARD_PARAMS and c["myth_later_prompt_key"] == BOARD_TEMPLATE
                assert request == twin["request_plan"].as_dict(), (name, "request plan differs")
            else:
                assert c["game_params"] == twin["game_params"] and c["myth_later_prompt_key"] == twin["myth_later_prompt_key"]
                assert request["provider_model"] == SONNET55_MODEL, request
                assert request["parameters"].get("thinking") == {"type": "adaptive"}, request
                assert request["parameters"].get("output_config") == {"effort": "high"}, request
            assert _seeds(c) == _seeds(twin), (name, "seeds differ from the baseline twin", _seeds(c), _seeds(twin))
            game, writer = build_noisy_protocol(c, i)
            assert len(game.defector_agent_ids) == 0 and game.random_defection_probability == 0
            assert not game.punishment_enabled
            if spec["board"]:
                assert writer.board == BOARD_KIND and writer.board_seed == c["replicate_id"]
            else:
                assert writer.board is None
            jobs.append((name, i, c, expected_output_path(c, name, i, OUTPUT), provider, stage))
    assert len(jobs) == spec["runs"] and len({str(j[3]) for j in jobs}) == spec["runs"], len(jobs)
    jobs.sort(key=lambda j: (0 if j[4] == "anthropic" else 1, j[0], j[1]))  # slow Sonnet runs first
    return jobs


def estimate(jobs):
    total = {"anthropic": 0.0, "openai": 0.0}
    for *_, provider, stage in jobs:
        if STAGES[stage]["board"]:
            total[provider] += BASE_USD[provider] + BOARD_INPUT_TOKENS * RATES[provider][0] / 1e6
        else:
            total[provider] += SONNET55_BASE_USD
    return total


class ErroredCall(Exception):
    """A provider call failed (e.g. a dropped connection) and was retried inside the run."""


def quarantine(job, reason):
    """Move a final with an errored call aside so it is resampled under the same seed."""
    path = job[3]
    stem = path.with_suffix("")
    target = ROOT / "data/json/noise_experiments" / OUTPUT / "quarantine" / f"{stem.name}_{time.strftime('%Y%m%dT%H%M%S')}"
    target.mkdir(parents=True, exist_ok=True)
    for f in path.parent.glob(stem.name + ".*"):
        f.rename(target / f.name)
    (target / "QUARANTINE_REASON.txt").write_text(reason + "\n")
    print(f"QUARANTINED {path.name}: {reason}", flush=True)


def audit(job):
    name, i, c, path, provider, stage = job
    board = STAGES[stage]["board"]
    check_existing_final(path, c)
    d = json.loads(path.read_text())
    m = d["run_metadata"]
    request = c["request_plan"].as_dict()
    n_agents = c["game_params"]["num_agents"]
    assert m["defector_count"] == 0 and m["random_defection_probability"] == 0
    assert not m["code_dirty"]
    assert m["llm_request"] == request
    assert m["replicate_id"] == c["replicate_id"]
    assert m["noise_config"] == c["game_params"]["noise_config"]
    cond = m["experiment_condition"]["protocol"]["myth"]
    if board:
        assert cond["board"] == BOARD_KIND and cond["board_seed"] == c["replicate_id"], cond
    else:
        assert "board" not in cond, cond
    # Board exposure records: every agent, every round from 2, 8*(r-1) myths, in myth and game.
    for entry in d["conversation_history"] if board else []:
        r = entry["round"]
        if r == 1:
            assert "board_in_game" not in entry
            continue
        for key in ("myth_exposures", "board_in_game"):
            records = entry.get(key) or {}
            assert set(records) == set(d["agents"]), (name, r, key, "missing records")
            for agent_id, rec in records.items():
                assert rec["board"] == BOARD_KIND and rec["board_seed"] == c["replicate_id"], (name, r, key, agent_id)
                assert len(rec["board_items"]) == n_agents * (r - 1), (name, r, key, agent_id, "board size")
        same = {str(rec["board_items"]) for rec in entry["board_in_game"].values()}
        assert len(same) == 1 and same == {str(rec["board_items"]) for rec in entry["myth_exposures"].values()}
    calls, output_per_call = 0, []
    cost = {"anthropic": 0.0, "openai": 0.0}
    ir, orr = RATES[provider]
    for agent_id, a in d["agents"].items():
        assert a["model"] == request["model"], (agent_id, a["model"])
        # Chat memory (the saved window) never holds the board text.
        for msg in a["messages"]:
            if msg.get("role") == "user":
                assert "Round 1, myth 1:" not in msg["content"], (name, agent_id, "board text in chat memory")
        for e in a.get("interaction_history", []):
            r = e.get("response") or {}
            meta = e.get("metadata") or {}
            if r.get("response_source", "llm") != "llm":
                raise AssertionError((name, agent_id, "non-LLM response"))
            u = r.get("usage") or {}
            assert u.get("request_settings") == request, (name, agent_id)
            if u.get("outcome") == "error":
                raise ErroredCall(f"{agent_id} round {meta.get('round')}: call recorded outcome=error")
            assert u.get("outcome") == "complete" and u.get("finish_reason") not in TRUNCATION_REASONS
            if board and meta.get("task") == "game" and meta.get("round", 0) >= 2:
                assert e["prompt"].startswith(GAME_BOARD_HEAD), (name, agent_id, meta.get("round"), "board missing from game prompt")
                assert e["prompt"].count("\nRound 1, myth ") == n_agents, (name, agent_id, meta.get("round"))
            if (not board or meta.get("round") == 1) and meta.get("task") == "game":
                assert GAME_BOARD_HEAD not in e["prompt"]
            calls += 1
            output_per_call.append(u.get("output_tokens") or 0)
            cost[provider] += ((u.get("input_tokens") or 0) * ir + (u.get("output_tokens") or 0) * orr) / 1e6
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "set": name,
        "stage": stage,
        "provider": provider,
        "task_order": "_".join(c["task_order"]),
        "replicate_id": c["replicate_id"],
        "calls": calls,
        "output_tokens_per_call_mean": statistics.mean(output_per_call) if output_per_call else 0,
        "mean_final_resources": statistics.mean(d["game_data"]["balances"].values()),
        "standard_rate_usd": sum(cost.values()),
    }


def main():
    os.chdir(ROOT)  # workers write finals relative to the cwd; the audit reads ROOT-relative paths
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=sorted(STAGES), default="board")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--smoke", nargs="?", const="openai", choices=sorted(RATES),
                   help="only that provider's game->myth replicate 0 (default openai: one GPT-5 Nano run)")
    p.add_argument("--provider", choices=sorted(RATES), help="restrict to one provider's runs")
    args = p.parse_args()
    assert 1 <= args.workers <= 20
    jobs = plan(args.stage)
    if args.provider:
        jobs = [j for j in jobs if j[4] == args.provider]
    if args.smoke:
        jobs = [j for j in jobs if j[4] == args.smoke and j[2]["task_order"] == ["game", "myth"] and j[1] == 0]
        assert len(jobs) == 1
    pending, receipts = [], []
    for j in jobs:
        if j[3].exists():
            try:
                receipts.append(audit(j))
                continue
            except ErroredCall as e:
                if not args.execute:
                    raise
                quarantine(j, str(e))
        pending.append(j)
    est = estimate(pending)
    print(f"VALIDATED STAGE={args.stage} N={len(jobs)} BOARD={BOARD_KIND if STAGES[args.stage]['board'] else 'none'} "
          f"SEEDS=EQUAL_TO_BASELINE_TWIN EXISTING={len(receipts)} PENDING={len(pending)}", flush=True)
    models = SONNET55_MODEL + " (adaptive thinking, effort high)" if args.stage == "sonnet55_baseline" else "claude-sonnet-4.5/gpt-5-nano (September profiles)"
    print(f"MODEL={models} N={len(pending)} WORKERS={args.workers} "
          f"EST_COST=${sum(est.values()):.2f} (anthropic ${est['anthropic']:.2f}, openai ${est['openai']:.2f}; "
          f"+20% allowance ${1.2 * sum(est.values()):.2f})", flush=True)
    if args.audit_only:
        assert not pending, f"{len(pending)} runs missing"
    elif not args.execute:
        print("DRY RUN: pass --execute to launch", flush=True)
        return
    if args.execute and pending:
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
            raise SystemExit("Clean checkout required")
        os.environ["HF_DATASET_AUTO_UPLOAD"] = "0"
        os.environ["TRUST_BATCH_QUIET"] = "1"
        logdir = str(ROOT / "data/json/noise_experiments" / OUTPUT / "worker_logs")
        workers = args.workers
        for attempt in range(1, 11):
            if not pending:
                break
            failed = []
            quota_exhausted = False
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_missing_job, j[2], j[0], j[1], OUTPUT, logdir): j for j in pending}
                for f in as_completed(futures):
                    j = futures[f]
                    if f.cancelled():
                        failed.append(j)
                        continue
                    try:
                        result = f.result()
                        if not result.get("success") and is_exhausted_quota(result.get("error", "")):
                            quota_exhausted = True
                            for queued in futures:
                                queued.cancel()
                            print("BILLING EXHAUSTED: canceling queued jobs; no further retry passes", flush=True)
                        if not result.get("success"):
                            raise RuntimeError(f"Worker failed; inspect {result.get('worker_log')}: {result.get('error')}")
                        receipt = audit(j)
                        receipts.append(receipt)
                        print(f"COMPLETE {len(receipts)}/{len(jobs)} {j[0]} index={j[1]} resources={receipt['mean_final_resources']:.1f} "
                              f"out/call={receipt['output_tokens_per_call_mean']:.0f} cost=${receipt['standard_rate_usd']:.3f}", flush=True)
                    except ErroredCall as e:
                        quarantine(j, str(e))
                        failed.append(j)
                    except Exception as e:
                        print(f"FAILED {j[0]} index={j[1]} {type(e).__name__}: {e}", flush=True)
                        # Never silently resample a final that fails validation; stop the queue first.
                        if j[3].exists():
                            for queued in futures:
                                queued.cancel()
                            raise
                        failed.append(j)
            if quota_exhausted:
                raise RuntimeError("Provider credits exhausted; top up before resuming. Completed finals preserved.")
            pending = failed
            if pending:
                if attempt == 10:
                    raise RuntimeError(f"{len(pending)} runs failed after ten attempts")
                workers = max(1, workers // 2)
                print(f"CONTINUATION PENDING={len(pending)} WORKERS={workers} ATTEMPT={attempt + 1}", flush=True)
                time.sleep(min(60, 2 ** attempt))
    by_provider = {k: sum(r["standard_rate_usd"] for r in receipts if r["provider"] == k) for k in RATES}
    target = ROOT / "data/json/noise_experiments" / OUTPUT / f"{args.stage}_{'smoke' if args.smoke else 'completion'}_receipt.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "runs": len(receipts),
        "standard_rate_usd": sum(r["standard_rate_usd"] for r in receipts),
        "standard_rate_usd_by_provider": by_provider,
        "finals": receipts,
    }, indent=2) + "\n")
    print(f"AUDIT PASSED {len(receipts)}/{len(jobs)}; cost=${sum(by_provider.values()):.2f} {by_provider}; receipt={target}", flush=True)


if __name__ == "__main__":
    main()
