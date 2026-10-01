#!/usr/bin/env python3
"""Frontier defector pilot (2026-10-01); dry-run by default.

Question: do permanent defectors open room below the ceiling, so the myth effect
can show in a main-frontier mixed population? Without defectors the 2 Gemini /
3 Opus / 3 Sol population ended at 72.2 in game only and 74.7 with myth -> game.

Design: 8 agents, balanced rotating pairs, Agent_1-4 Claude Opus 5 and Agent_5-8
GPT-5.6 Sol (effort high), each at its D011 request profile; no Gemini (it is
ceiling-locked). Two permanent forced-zero defectors, Agent_4 (Opus) and Agent_8
(Sol): they always send and return $0 without a model call, still write myths,
and do not know they are defectors (the September defectors25 settings).
Task orders game and myth_game, replicates 0-2: 6 runs.

The plan step proves, before any paid call, that every non-model input equals the
September ``negative_only_reasoning_rerun_population_*`` defectors25 cell except
the explicit defector ids. Outcomes are scored on the six ordinary agents.
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
from scripts.build_frontier_rerun_config import ARMS, PROFILES, DEFECTOR_PILOT_POPULATION, DEFECTOR_PILOT_NAME, DEFECTOR_PILOT_IDS  # noqa: E402
from experiments.run_noisy_batch import build_noisy_protocol  # noqa: E402
from src.llm_settings import is_mixed_plan  # noqa: E402
from src.utils import is_exhausted_quota  # noqa: E402
import yaml  # noqa: E402

CONFIG = ROOT / "config/frontier_rerun_20260918.yaml"
SEPTEMBER = ROOT / "config/experiments_noisy.yaml"
OUTPUT = "frontier_defector_pilot_20261001"
SHAPES = ("game", "myth_game")
REPLICATES = [0, 1, 2]
_BASE_MODELS = yaml.safe_load(CONFIG.read_text())["base_models"]
# USD per MTok by provider model, as priced in the 2026-09-18 frontier receipts (verified then).
RATES = {"claude-opus-5": (5.0, 25.0), "gpt-5.6-sol": (4.0, 20.0)}
# Mean standard-rate cost per agent in the homogeneous 2026-09-18 8-agent frontier runs. An upper
# bound here: defectors make no game calls.
PER_AGENT_USD = {"claude-opus-5": {"game": 0.081, "myth_game": 0.501},
                 "gpt-5.6-sol": {"game": 0.061, "myth_game": 0.435}}
# Inputs that legitimately differ from the homogeneous September defectors25 cell.
MODEL_ONLY_KEYS = {"model", "agent_models", "llm_request", "replicate_id"}
PILOT_ONLY_KEYS = {"game_params_name", "defector_agent_ids"}


def _comparable(inputs):
    """Comparison inputs minus model-only keys, with the explicit defector ids removed from game_params."""
    out = {k: v for k, v in inputs.items() if k not in MODEL_ONLY_KEYS and k not in PILOT_ONLY_KEYS}
    if isinstance(out.get("game_params"), dict):
        out["game_params"] = {k: v for k, v in out["game_params"].items() if k != "defector_agent_ids"}
    return out


def require(condition, *message):
    """Validation that survives ``python -O`` (a bare assert would not)."""
    if not condition:
        raise RuntimeError(" ".join(str(part) for part in message))


def _quiet_combinations(name, config):
    with contextlib.redirect_stdout(io.StringIO()):
        return load_combinations(name, str(config))


def _expected(arms):
    """Agent id -> (model slug, profile) for an ordered list of arms."""
    return {f"Agent_{i + 1}": (_BASE_MODELS[ARMS[arm][0]], PROFILES[ARMS[arm][1]]) for i, arm in enumerate(arms)}


def _expected():
    arms = [arm for arm, count in DEFECTOR_PILOT_POPULATION for _ in range(count)]
    return {f"Agent_{i + 1}": (_BASE_MODELS[ARMS[arm][0]], PROFILES[ARMS[arm][1]]) for i, arm in enumerate(arms)}


def plan():
    jobs = []
    expected = _expected()
    require(len(expected) == 8 and {expected[a][0] for a in DEFECTOR_PILOT_IDS} == {expected["Agent_1"][0], expected["Agent_8"][0]},
            "one defector per family")
    for shape in SHAPES:
        suffix = "defectors25_game_r3" if shape == "game" else "defectors25_twotask_r3"
        reference = [c for c in _quiet_combinations(f"negative_only_reasoning_rerun_population_{shape}_claude_n5", SEPTEMBER)
                     if c["game_params_name"].endswith(suffix) and c["replicate_id"] == 0]
        require(len(reference) == 1, shape)
        reference_inputs = _comparable(reference[0]["comparison_inputs"])
        name = f"frontier_defector_pilot_population_{shape}_{DEFECTOR_PILOT_NAME}_n3"
        combos = _quiet_combinations(name, CONFIG)
        require([c["replicate_id"] for c in combos] == REPLICATES, name, [c["replicate_id"] for c in combos])
        for i, c in enumerate(combos):
            require(c["agent_models"] == {a: m for a, (m, _) in expected.items()}, name, c["agent_models"])
            params = c["game_params"]
            require(params["num_agents"] == 8 and params["pairing_mode"] == "balanced", name, "population pairing")
            require(params["noise_config"]["inform_agents"] is True, name, "inform_agents")
            require(params["show_agent_names"] is False and params["history_policy"] == "self_and_coplayer"
                    and params["coplayer_history_window"] == 3, name, "population history/identity settings")
            request = c["request_plan"].as_dict()
            require(is_mixed_plan(request) and request["model"] == c["model"], name, "mixed request plan")
            for agent_id, (model, profile) in expected.items():
                agent_request = request["agents"][agent_id]
                require(agent_request["model"] == model, name, agent_id, "model")
                require(agent_request["provider"] == profile["provider"], name, agent_id, "provider")
                require(agent_request["policy"] == profile, name, agent_id, agent_request["policy"])
            actual_inputs = _comparable(c["comparison_inputs"])
            differing = sorted(k for k in set(actual_inputs) | set(reference_inputs) if actual_inputs.get(k) != reference_inputs.get(k))
            if differing == ["game_params"]:
                differing = sorted(k for k in set(actual_inputs["game_params"]) | set(reference_inputs["game_params"])
                                   if actual_inputs["game_params"].get(k) != reference_inputs["game_params"].get(k))
            require(not differing, name, "non-model inputs differ from September defectors25 control", differing)
            require(params["defector_ratio"] == 0.25 and params.get("random_defection_probability", 0) == 0, name, "defector ratio")
            game, _ = build_noisy_protocol(c, i)
            game.configure_agents(list(expected))  # what the simulation does; assigns the defectors
            require(game.defector_agent_ids == DEFECTOR_PILOT_IDS, name, "defector ids", game.defector_agent_ids)
            require(game.defector_action_policy == "forced_zero" and game.defector_myth_policy == "normal"
                    and game.defector_role_visible_to_self is False, name, "defector policies")
            require(game.random_defection_probability == 0 and not game.punishment_enabled, name, "random defection/punishment")
            jobs.append({"name": name, "index": i, "combo": c, "size": 8, "shape": shape,
                         "path": expected_output_path(c, name, i, OUTPUT)})
    require(len(jobs) == 6 and len({str(j["path"]) for j in jobs}) == 6, f"expected 6 unique jobs, found {len(jobs)}")
    jobs.sort(key=lambda j: (0 if "myth" in j["shape"] else 1, j["combo"]["replicate_id"]))
    return jobs


def estimate(jobs):
    total = {m: 0.0 for m in RATES}
    for j in jobs:
        for agent_request in j["combo"]["request_plan"].as_dict()["agents"].values():
            total[agent_request["provider_model"]] += PER_AGENT_USD[agent_request["provider_model"]][j["shape"]]
    return total


def audit(job):
    c, path, name = job["combo"], job["path"], job["name"]
    check_existing_final(path, c)
    d = json.loads(path.read_text())
    m = d["run_metadata"]
    request = c["request_plan"].as_dict()
    require(m["defector_count"] == 2 and m["defector_agent_ids"] == DEFECTOR_PILOT_IDS and m["random_defection_probability"] == 0, path, "defectors")
    require(not m["code_dirty"], path, "produced from a dirty checkout")
    require(m["llm_request"] == request and m["llm_provider"] == "mixed", path, "request plan")
    require(m["agent_models"] == c["agent_models"], path, "agent models")
    require(m["noise_config"] == c["game_params"]["noise_config"], path, "noise config")
    require(set(d["agents"]) == set(request["agents"]), path, "agent ids")
    calls = 0
    cost = {model: 0.0 for model in RATES}
    for agent_id, a in d["agents"].items():
        expected = request["agents"][agent_id]
        require(a["model"] == expected["model"], path, agent_id, a["model"])
        provider_model = expected["provider_model"]
        ir, orr = RATES[provider_model]
        for e in a.get("interaction_history", []):
            r = e.get("response") or {}
            if r.get("response_source", "llm") != "llm":
                continue
            u = r.get("usage") or {}
            require(u.get("request_settings") == expected, path, agent_id, "per-call request settings differ from the plan")
            require(u.get("outcome") == "complete" and u.get("finish_reason") not in TRUNCATION_REASONS, path, agent_id, "truncated or incomplete call", u.get("finish_reason"))
            calls += 1
            output = (u.get("output_tokens") or 0) + ((u.get("reasoning_tokens") or 0) if expected["provider"] == "google" else 0)
            cost[provider_model] += ((u.get("input_tokens") or 0) * ir + output * orr) / 1e6
    require(calls > 0, path, "no LLM calls recorded")
    final = d["conversation_history"][-1]
    require(final["round"] == 10, path, "did not finish round 10")
    defector_actions = 0
    for row in d["conversation_history"]:
        for agent_id in DEFECTOR_PILOT_IDS:
            action = (row.get("actions") or {}).get(agent_id)
            if action:
                defector_actions += 1
                require(action["decision"] == 0, path, agent_id, "defector moved money", row["round"], action)
    require(defector_actions == 2 * 10, path, "expected one action per defector per round", defector_actions)
    ordinary = [a for a in d["agents"] if a not in DEFECTOR_PILOT_IDS]
    by_family = {}
    for a in ordinary:
        by_family.setdefault(request["agents"][a]["provider_model"], []).append(final["balances"][a])
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "set": name,
            "num_agents": job["size"], "agent_models": c["agent_models"], "task_order": job["shape"],
            "replicate_id": c["replicate_id"], "calls": calls, "standard_rate_usd_by_model": cost,
            "standard_rate_usd": sum(cost.values()),
            "ordinary_mean_final": sum(final["balances"][a] for a in ordinary) / len(ordinary),
            "ordinary_mean_final_by_model": {k: sum(v) / len(v) for k, v in by_family.items()}}


STAGES = {
    "smoke": lambda j: j["shape"] == "game" and j["combo"]["replicate_id"] == 0,
    "all": lambda j: True,
}


def main():
    os.chdir(ROOT)  # workers write finals relative to the cwd; the audit reads ROOT-relative paths
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=sorted(STAGES), required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args()
    require(1 <= args.workers <= 20, "--workers must be 1-20")
    selected = [j for j in plan() if STAGES[args.stage](j)]
    pending, receipts = [], []
    for j in selected:
        if j["path"].exists():
            receipts.append(audit(j))
        else:
            pending.append(j)
    est = estimate(pending)
    print(f"VALIDATED STAGE={args.stage} N={len(selected)} EXISTING={len(receipts)} PENDING={len(pending)} "
          f"POPULATION={DEFECTOR_PILOT_NAME} DEFECTORS={','.join(DEFECTOR_PILOT_IDS)} FORCED_ZERO=1", flush=True)
    print(f"MODEL=mixed({','.join(RATES)}) N={len(pending)} WORKERS={args.workers} EST_COST=${sum(est.values()):.2f} "
          f"({', '.join(f'{k} ${v:.2f}' for k, v in est.items())}; +20% allowance ${1.2 * sum(est.values()):.2f})", flush=True)
    if args.audit_only:
        require(not pending, f"{len(pending)} runs missing")
    elif not args.execute:
        print("DRY RUN: pass --execute to launch", flush=True)
        return
    if args.execute and pending:
        require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(), "Clean checkout required")
        os.environ["HF_DATASET_AUTO_UPLOAD"] = "0"
        os.environ["TRUST_BATCH_QUIET"] = "1"
        logdir = str(ROOT / "data/json/noise_experiments" / OUTPUT / "worker_logs")
        workers = min(args.workers, len(pending))
        for attempt in range(1, 11):
            if not pending:
                break
            failed = []
            quota_exhausted = False
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_missing_job, j["combo"], j["name"], j["index"], OUTPUT, logdir): j for j in pending}
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
                            raise RuntimeError(f"Worker failed; inspect {result.get('worker_log')}: {str(result.get('error'))[:300]}")
                        receipt = audit(j)
                        receipts.append(receipt)
                        print(f"COMPLETE {len(receipts)}/{len(selected)} {j['name']} index={j['index']} calls={receipt['calls']} cost=${receipt['standard_rate_usd']:.3f}", flush=True)
                    except Exception as e:
                        print(f"FAILED {j['name']} index={j['index']} {type(e).__name__}: {e}", flush=True)
                        # Never silently resample a final that fails scientific validation.
                        if j["path"].exists():
                            raise
                        failed.append(j)
            if quota_exhausted:
                raise RuntimeError("Provider credits exhausted; top up before resuming. Completed finals preserved.")
            pending = failed
            if pending:
                if attempt == 10:
                    raise RuntimeError(f"{len(pending)} runs failed after ten attempts")
                workers = max(1, min(workers, len(pending)) // 2)
                print(f"CONTINUATION PENDING={len(pending)} WORKERS={workers} ATTEMPT={attempt + 1}", flush=True)
                time.sleep(min(60, 2 ** attempt))
    by_model = {k: sum(r["standard_rate_usd_by_model"][k] for r in receipts) for k in RATES}
    target = ROOT / "data/json/noise_experiments" / OUTPUT / f"{args.stage}_receipt.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"stage": args.stage, "runs": len(receipts), "standard_rate_usd": sum(r["standard_rate_usd"] for r in receipts),
                                  "standard_rate_usd_by_model": by_model, "finals": receipts}, indent=2) + "\n")
    for shape in SHAPES:
        rows = [r for r in receipts if r["task_order"] == shape]
        if rows:
            values = [r["ordinary_mean_final"] for r in rows]
            print(f"RESULT {shape}: ordinary-agent mean final {sum(values) / len(values):.1f} over {len(rows)} runs {[round(v, 1) for v in values]}", flush=True)
    print(f"AUDIT PASSED {len(receipts)}/{len(selected)}; cost=${sum(by_model.values()):.2f} {by_model}; receipt={target}", flush=True)


if __name__ == "__main__":
    main()
