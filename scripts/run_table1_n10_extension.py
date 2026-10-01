#!/usr/bin/env python3
"""Extend paper Table 1 (mixed groups vs the average of their parts) to n = 10; dry-run by default.

216 runs in the *ext sets of config/experiments_noisy.yaml (added 2026-10-01):
- 90 single-model runs: Sonnet 4.5, GPT-5 Nano and Gemini 3.7 Flash, dyads and
  8-agent populations, game / game_myth / myth_game, no-defector condition only,
  replicates 5-9 (the September controls hold 0-4);
- 36 mixed dyads: the six orderings of Sonnet+GPT, Sonnet+Gemini, Gemini+GPT,
  replicates 6/8 (first-named family sends first) and 7/9 (reversed); the
  originals hold 0-5;
- 90 mixed populations: the six contagion-ladder compositions, replicates 5-9.

The plan step proves, before any paid call, that every *ext run has exactly the
inputs and request plan of its original set except the replicate id, and that
its paired noise/pairing seed is new for that set. The audit step re-checks
every final and prices it per provider from recorded token usage.
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
from scripts.run_noisy_missing import load_combinations, expected_output_path, check_existing_final, run_missing_job  # noqa: E402
from scripts.rerun_negative_only_crossmodel import TRUNCATION_REASONS  # noqa: E402
from experiments.run_noisy_batch import build_noisy_protocol, resolve_protocol_seeds  # noqa: E402
from src.utils import is_exhausted_quota  # noqa: E402

CONFIG = ROOT / "config/experiments_noisy.yaml"
OUTPUT = "table1_n10_extension_20261001"
SHAPES = ("game", "game_myth", "myth_game")
RATES = {"anthropic": (3, 15), "openai": (0.05, 0.4), "google": (0.75, 3.75)}
# Mean standard-rate cost per agent in the September informed-noise controls
# (as in run_mixed_model_dyads.py / run_mixed_model_populations.py).
PER_AGENT_USD = {
    2: {"anthropic": {"game": 0.1094, "game_myth": 0.4041, "myth_game": 0.3955},
        "openai": {"game": 0.0070, "game_myth": 0.0566, "myth_game": 0.0570},
        "google": {"game": 0.0171, "game_myth": 0.1116, "myth_game": 0.1166}},
    8: {"anthropic": {"game": 0.1186, "game_myth": 0.4085, "myth_game": 0.4241},
        "openai": {"game": 0.0071, "game_myth": 0.0555, "myth_game": 0.0563},
        "google": {"game": 0.0193, "game_myth": 0.1059, "myth_game": 0.1092}},
}


def ext_sets():
    """(extension set, original set, expected replicate ids)."""
    out = []
    for scale in ("dyad", "population"):
        for shape in SHAPES:
            for model in ("claude", "gpt", "gemini"):
                src = f"negative_only_reasoning_rerun_{scale}_{shape}_{model}_n5"
                out.append((src + "ext", src, [5, 6, 7, 8, 9]))
    for shape in SHAPES:
        for pair, reps in [("sonnet_gpt", [6, 8]), ("gpt_sonnet", [7, 9]), ("sonnet_gemini", [6, 8]),
                           ("gemini_sonnet", [7, 9]), ("gemini_gpt", [6, 8]), ("gpt_gemini", [7, 9])]:
            src = f"mixed_dyad_{shape}_{pair}_n3"
            out.append((f"mixed_dyad_{shape}_{pair}_n2ext", src, reps))
    for shape in SHAPES:
        for comp in ("gemini1_gpt7", "gpt1_sonnet7", "gemini2_gpt6", "gpt2_sonnet6", "gemini4_gpt4", "gpt4_sonnet4"):
            src = f"mixed_pop_{shape}_{comp}_n5"
            out.append((f"mixed_pop_{shape}_{comp}_n5ext", src, [5, 6, 7, 8, 9]))
    return out


def _quiet(name):
    with contextlib.redirect_stdout(io.StringIO()):
        return load_combinations(name, str(CONFIG))


def _seeds(c):
    return resolve_protocol_seeds(c["game_params"], c)


def plan():
    jobs = []
    for name, src, reps in ext_sets():
        combos = _quiet(name)
        originals = [o for o in _quiet(src) if o["game_params"].get("defector_ratio", 0) == 0
                     and o["game_params"].get("random_defection_probability", 0) == 0]
        assert [c["replicate_id"] for c in combos] == reps, (name, [c["replicate_id"] for c in combos])
        reference = originals[0]
        ref_inputs = {k: v for k, v in reference["comparison_inputs"].items() if k != "replicate_id"}
        used_seeds = {_seeds(o) for o in originals}
        for i, c in enumerate(combos):
            inputs = {k: v for k, v in c["comparison_inputs"].items() if k != "replicate_id"}
            differing = sorted(k for k in set(inputs) | set(ref_inputs) if inputs.get(k) != ref_inputs.get(k))
            assert not differing, (name, "inputs differ from the original set", differing)
            assert c["request_plan"].as_dict() == reference["request_plan"].as_dict(), (name, "request plan differs")
            assert c["game_params_name"] == reference["game_params_name"]
            seeds = _seeds(c)
            base = int(c["game_params"].get("protocol_seed_base", 0))
            assert seeds == (base + c["replicate_id"], base + c["replicate_id"]), (name, seeds)
            assert seeds not in used_seeds, (name, "seed already used by the original set", seeds)
            assert c["game_params"].get("defector_ratio", 0) == 0
            assert c["game_params"].get("random_defection_probability", 0) == 0
            game, _ = build_noisy_protocol(c, i)
            assert len(game.defector_agent_ids) == 0 and game.random_defection_probability == 0
            assert not game.punishment_enabled
            jobs.append((name, i, c, expected_output_path(c, name, i, OUTPUT)))
    assert len(jobs) == 216 and len({str(j[3]) for j in jobs}) == 216, len(jobs)
    # Longer two-task, eight-agent jobs first.
    jobs.sort(key=lambda j: (0 if "myth" in j[2]["task_order"] else 1, -j[2]["game_params"]["num_agents"], j[0], j[1]))
    return jobs


def agent_plans(c):
    request = c["request_plan"].as_dict()
    if "agents" in request:
        return request["agents"]
    n = c["game_params"]["num_agents"]
    return {f"Agent_{k + 1}": request for k in range(n)}


def estimate(jobs):
    total = {"anthropic": 0.0, "openai": 0.0, "google": 0.0}
    for _, _, c, _ in jobs:
        shape = "_".join(c["task_order"])
        n = c["game_params"]["num_agents"]
        for plan_ in agent_plans(c).values():
            total[plan_["provider"]] += PER_AGENT_USD[n][plan_["provider"]][shape]
    return total


def audit(job):
    name, i, c, path = job
    check_existing_final(path, c)
    d = json.loads(path.read_text())
    m = d["run_metadata"]
    request = c["request_plan"].as_dict()
    plans = agent_plans(c)
    assert m["defector_count"] == 0 and m["random_defection_probability"] == 0
    assert not m["code_dirty"]
    assert m["llm_request"] == request
    assert m["replicate_id"] == c["replicate_id"]
    assert m["noise_config"] == c["game_params"]["noise_config"]
    assert set(d["agents"]) == set(plans)
    calls = 0
    cost = {"anthropic": 0.0, "openai": 0.0, "google": 0.0}
    for agent_id, a in d["agents"].items():
        expected = plans[agent_id]
        assert a["model"] == expected["model"], (agent_id, a["model"])
        provider = expected["provider"]
        ir, orr = RATES[provider]
        for e in a.get("interaction_history", []):
            r = e.get("response") or {}
            if r.get("response_source", "llm") != "llm":
                continue
            u = r.get("usage") or {}
            assert u.get("request_settings") == expected, (name, agent_id)
            assert u.get("outcome") == "complete" and u.get("finish_reason") not in TRUNCATION_REASONS
            calls += 1
            output = (u.get("output_tokens") or 0) + ((u.get("reasoning_tokens") or 0) if provider == "google" else 0)
            cost[provider] += ((u.get("input_tokens") or 0) * ir + output * orr) / 1e6
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "set": name,
        "task_order": "_".join(c["task_order"]),
        "num_agents": c["game_params"]["num_agents"],
        "replicate_id": c["replicate_id"],
        "calls": calls,
        "standard_rate_usd_by_provider": cost,
        "standard_rate_usd": sum(cost.values()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="two game-only dyads (one single-model GPT, one Gemini+GPT)")
    args = p.parse_args()
    assert 1 <= args.workers <= 20
    jobs = plan()
    if args.smoke:
        jobs = [j for j in jobs if j[0] in {"negative_only_reasoning_rerun_dyad_game_gpt_n5ext",
                                            "mixed_dyad_game_gemini_gpt_n2ext"} and j[1] == 0]
        assert len(jobs) == 2
    pending, receipts = [], []
    for j in jobs:
        if j[3].exists():
            receipts.append(audit(j))
        else:
            pending.append(j)
    est = estimate(pending)
    print(f"VALIDATED N={len(jobs)} SINGLE=90 MIXED_DYADS=36 MIXED_POPS=90 NO_DEFECTORS=1 NEW_SEEDS=1 "
          f"EXISTING={len(receipts)} PENDING={len(pending)}", flush=True)
    print(f"MODEL=sonnet-4.5/gpt-5-nano/gemini-3.7-flash (September profiles) N={len(pending)} WORKERS={args.workers} "
          f"EST_COST=${sum(est.values()):.2f} (anthropic ${est['anthropic']:.2f}, openai ${est['openai']:.2f}, "
          f"google ${est['google']:.2f}; +20% allowance ${1.2 * sum(est.values()):.2f})", flush=True)
    if args.audit_only:
        assert not pending
    elif not args.execute:
        return
    if args.execute and pending:
        assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(), "Clean checkout required"
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
                        print(f"COMPLETE {len(receipts)}/{len(jobs)} {j[0]} index={j[1]} cost=${receipt['standard_rate_usd']:.3f}", flush=True)
                    except Exception as e:
                        print(f"FAILED {j[0]} index={j[1]} {type(e).__name__}: {e}", flush=True)
                        # Never silently resample a final that fails scientific validation.
                        if j[3].exists():
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
    by_provider = {k: sum(r["standard_rate_usd_by_provider"][k] for r in receipts) for k in RATES}
    target = ROOT / "data/json/noise_experiments" / OUTPUT / ("smoke_receipt.json" if args.smoke else "completion_receipt.json")
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
