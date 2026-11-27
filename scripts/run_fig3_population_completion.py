#!/usr/bin/env python3
"""Figure 3 completion: the nine missing mixed eight-agent compositions; dry-run by default.

The 2026-09-18 ladder (scripts/run_mixed_model_populations.py) ran 1, 2 or 4
Gemini 3.7 Flash among GPT-5 Nano and 1, 2 or 4 GPT-5 Nano among Sonnet 4.5.
Letting each family take a turn as the minority (1, 2 or 4 of 8) among each of
the other two gives 15 distinct compositions; this launcher runs the nine not
yet run: 1 or 2 GPT among Gemini, 1 or 2 Sonnet among GPT, 1, 2 or 4 Gemini
among Sonnet, and 1 or 2 Sonnet among Gemini. Task orders game, game_myth and
myth_game; five replicates per cell (ids 0-4); 135 runs. ``--replicates 5-9``
ran the top-up (the ``*_n5ext`` sets). The minority family occupies the lowest
agent ids.

Gemini 3.8 note (2026-11-27): from about 2026-10-10 the Gemini API answers
requests for ``gemini-3.7-flash`` with ``gemini-3.8-flash``, so every 2026-11-23
run with a Gemini agent is a 3.8 run. ``--replicates vertex`` repeats the seven
Gemini compositions (replicates 0-4, the ``*_vertex_n5`` sets) with Gemini sent
through Vertex AI, which still serves 3.7, into a separate output folder.
``--replicates vertex-5-9`` adds their replicates 5-9 (``*_vertex_n5ext``).
``--replicates 5-9`` now covers only the two compositions without Gemini.

The plan step proves, before any paid call, that every non-model input equals
the September ``negative_only_reasoning_rerun_population_*`` control cell and
that each agent's pinned request policy equals the September profile for its
model. The audit step is the 2026-09-18 launcher's, unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_noisy_missing import expected_output_path, run_missing_job  # noqa: E402
from scripts.rerun_negative_only_crossmodel import EXPECTED_POLICIES  # noqa: E402
from scripts.run_mixed_model_populations import FAMILY, MODEL_ONLY_KEYS, RATES, SHAPES, _quiet_combinations, audit  # noqa: E402
from experiments.run_noisy_batch import build_noisy_protocol, resolve_protocol_seeds  # noqa: E402
from src.llm_settings import is_mixed_plan  # noqa: E402
from src.utils import is_exhausted_quota  # noqa: E402

OUTPUT = "mixed_model_populations_20261123"
VERTEX_OUTPUT = "mixed_model_populations_20261127_vertex"
NO_GEMINI = ("sonnet2_gpt6", "sonnet1_gpt7")
# --replicates -> (set-name suffix, replicate ids, receipt file, compositions, output folder, Gemini provider).
BATCHES = {
    "0-4": ("n5", [0, 1, 2, 3, 4], "completion_receipt.json", None, OUTPUT, "google"),
    "5-9": ("n5ext", [5, 6, 7, 8, 9], "completion_receipt_n5ext.json", NO_GEMINI, OUTPUT, "google"),
    "vertex": ("vertex_n5", [0, 1, 2, 3, 4], "completion_receipt.json", "gemini", VERTEX_OUTPUT, "vertex"),
    "vertex-5-9": ("vertex_n5ext", [5, 6, 7, 8, 9], "completion_receipt_n5ext.json", "gemini", VERTEX_OUTPUT, "vertex"),
}
# Set-name suffix -> (minority family, minority count, majority family).
COMPOSITIONS = {
    "gpt2_gemini6": ("gpt", 2, "gemini"), "gpt1_gemini7": ("gpt", 1, "gemini"),
    "sonnet2_gpt6": ("sonnet", 2, "gpt"), "sonnet1_gpt7": ("sonnet", 1, "gpt"),
    "gemini1_sonnet7": ("gemini", 1, "sonnet"), "gemini2_sonnet6": ("gemini", 2, "sonnet"),
    "gemini4_sonnet4": ("gemini", 4, "sonnet"),
    "sonnet2_gemini6": ("sonnet", 2, "gemini"), "sonnet1_gemini7": ("sonnet", 1, "gemini"),
}
# Mean standard-rate cost per agent in the 2026-09-18 mixed ladder
# (mixed_model_populations_20260918/completion_receipt.json, 90 runs).
PER_AGENT_USD = {
    "anthropic": {"game": 0.1207, "game_myth": 0.4080, "myth_game": 0.4267},
    "openai": {"game": 0.0115, "game_myth": 0.0550, "myth_game": 0.0564},
    "google": {"game": 0.0188, "game_myth": 0.1150, "myth_game": 0.1163},
}
PER_AGENT_USD["vertex"] = PER_AGENT_USD["google"]  # same model, same list price


def batch_compositions(batch):
    selected = BATCHES[batch][3]
    if selected is None:
        return dict(COMPOSITIONS)
    if selected == "gemini":
        return {k: v for k, v in COMPOSITIONS.items() if "gemini" in (v[0], v[2])}
    return {k: COMPOSITIONS[k] for k in selected}


def expected_policy(model, gemini_provider):
    policy = dict(EXPECTED_POLICIES[model])
    if model == FAMILY["gemini"]:
        policy["provider"] = gemini_provider
    return policy


def plan(batch="0-4"):
    suffix, replicates, _, _, output, gemini_provider = BATCHES[batch]
    compositions = batch_compositions(batch)
    jobs = []
    for shape in SHAPES:
        reference = [
            c for c in _quiet_combinations(f"negative_only_reasoning_rerun_population_{shape}_claude_n5")
            if c["game_params_name"].endswith(("negative_game_r3", "negative_twotask_r3")) and c["replicate_id"] == 0
        ]
        assert len(reference) == 1, shape
        reference_inputs = {k: v for k, v in reference[0]["comparison_inputs"].items() if k not in MODEL_ONLY_KEYS}
        for comp, (minority, count, majority) in compositions.items():
            name = f"mixed_pop_{shape}_{comp}_{suffix}"
            combos = _quiet_combinations(name)
            assert [c["replicate_id"] for c in combos] == replicates, name
            expected_models = {f"Agent_{i+1}": FAMILY[minority if i < count else majority] for i in range(8)}
            for i, c in enumerate(combos):
                assert c["agent_models"] == expected_models, (name, c["agent_models"])
                params = c["game_params"]
                assert params["num_agents"] == 8 and params["pairing_mode"] == "balanced" and params["show_agent_names"] is False
                assert params["history_policy"] == "self_and_coplayer" and params["coplayer_history_window"] == 3
                request = c["request_plan"].as_dict()
                assert is_mixed_plan(request) and request["model"] == c["model"]
                for agent_id, model in expected_models.items():
                    agent_request = request["agents"][agent_id]
                    assert agent_request["model"] == model
                    assert agent_request["policy"] == expected_policy(model, gemini_provider), (name, agent_id)
                # Paired noise/pairing seeds follow the replicate id, so the top-up continues the sequence.
                base = int(params.get("protocol_seed_base", 0))
                assert resolve_protocol_seeds(params, c) == (base + c["replicate_id"], base + c["replicate_id"]), name
                actual_inputs = {k: v for k, v in c["comparison_inputs"].items() if k not in MODEL_ONLY_KEYS}
                differing = sorted(k for k in set(actual_inputs) | set(reference_inputs) if actual_inputs.get(k) != reference_inputs.get(k))
                assert not differing, (name, "non-model inputs differ from September control", differing)
                assert params.get("defector_ratio", 0) == 0 and params.get("random_defection_probability", 0) == 0
                game, _ = build_noisy_protocol(c, i)
                assert len(game.defector_agent_ids) == 0 and game.random_defection_probability == 0
                assert not game.punishment_enabled
                jobs.append((name, i, c, expected_output_path(c, name, i, output)))
    n_runs = len(SHAPES) * len(compositions) * len(replicates)
    assert len(jobs) == n_runs and len({str(j[3]) for j in jobs}) == n_runs
    # Sonnet-heavy two-task jobs are the slow, expensive path: start them first.
    jobs.sort(key=lambda j: (0 if "myth" in j[2]["task_order"] else 1, -sum(m.startswith("anthropic") for m in j[2]["agent_models"].values()), j[1]))
    return jobs


def estimate(jobs):
    total = dict.fromkeys(PER_AGENT_USD, 0.0)
    for _, _, c, _ in jobs:
        shape = "_".join(c["task_order"])
        for agent_request in c["request_plan"].as_dict()["agents"].values():
            total[agent_request["provider"]] += PER_AGENT_USD[agent_request["provider"]][shape]
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="one game-only replicate of the cheapest cell in the batch (1 run)")
    p.add_argument("--replicates", choices=sorted(BATCHES), default="0-4")
    args = p.parse_args()
    assert 1 <= args.workers <= 40
    jobs = plan(args.replicates)
    output = BATCHES[args.replicates][4]
    if args.smoke:
        smoke_set = f"mixed_pop_game_gpt1_gemini7_{BATCHES[args.replicates][0]}"
        if args.replicates == "5-9":
            smoke_set = "mixed_pop_game_sonnet1_gpt7_n5ext"
        jobs = [j for j in jobs if j[0] == smoke_set and j[1] == 0]
        assert len(jobs) == 1
    # A final that fails the audit (a failed provider call, say) is held: left on
    # disk, kept out of the receipt and never resampled here. What to do with it
    # is a decision for a person.
    pending, receipts, held = [], [], []
    for j in jobs:
        if j[3].exists():
            try:
                receipts.append(audit(j))
            except AssertionError as e:
                held.append({"set": j[0], "index": j[1], "path": str(j[3].relative_to(ROOT)), "reason": f"{type(e).__name__}: {e}"})
        else:
            pending.append(j)
    est = estimate(pending)
    print(f"VALIDATED N={len(jobs)} POPULATIONS_8 COMPOSITIONS={len(batch_compositions(args.replicates))} BALANCED_PAIRING=1 NO_DEFECTORS=1 EXISTING={len(receipts)} HELD={len(held)} PENDING={len(pending)}", flush=True)
    print(
        f"MODEL=mixed({', '.join(batch_compositions(args.replicates))}) GEMINI_PROVIDER={BATCHES[args.replicates][5]} OUTPUT={output} N={len(pending)} WORKERS={args.workers} "
        f"EST_COST=${sum(est.values()):.2f} ({', '.join(f'{k} ${v:.2f}' for k, v in est.items() if v)}; +20% allowance ${1.2*sum(est.values()):.2f})",
        flush=True,
    )
    if args.audit_only:
        assert not pending
    elif not args.execute:
        return
    if args.execute and pending:
        assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(), "Clean checkout required"
        os.environ["HF_DATASET_AUTO_UPLOAD"] = "0"
        os.environ["TRUST_BATCH_QUIET"] = "1"
        logdir = str(ROOT / "data/json/noise_experiments" / output / "worker_logs")
        workers = args.workers
        for attempt in range(1, 11):
            if not pending:
                break
            failed = []
            quota_exhausted = False
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_missing_job, j[2], j[0], j[1], output, logdir): j for j in pending}
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
                        # Never silently resample a final that fails scientific validation.
                        if j[3].exists():
                            held.append({"set": j[0], "index": j[1], "path": str(j[3].relative_to(ROOT)), "reason": f"{type(e).__name__}: {e}"})
                            print(f"HELD {j[0]} index={j[1]} {type(e).__name__}: {e}", flush=True)
                            continue
                        print(f"FAILED {j[0]} index={j[1]} {type(e).__name__}: {e}", flush=True)
                        failed.append(j)
            if quota_exhausted:
                raise RuntimeError("Provider credits exhausted; top up before resuming. Completed finals preserved.")
            pending = failed
            if pending:
                if attempt == 10:
                    raise RuntimeError(f"{len(pending)} runs failed after ten attempts")
                # Failures at high concurrency are usually rate limits: halve and retry.
                workers = max(1, workers // 2)
                print(f"CONTINUATION PENDING={len(pending)} WORKERS={workers} ATTEMPT={attempt+1}", flush=True)
                time.sleep(min(60, 2 ** attempt))
    by_provider = {k: sum(r["standard_rate_usd_by_provider"][k] for r in receipts) for k in RATES}
    target = ROOT / "data/json/noise_experiments" / output / ("smoke_receipt.json" if args.smoke else BATCHES[args.replicates][2])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "runs": len(receipts),
        "standard_rate_usd": sum(r["standard_rate_usd"] for r in receipts),
        "standard_rate_usd_by_provider": by_provider,
        "finals": receipts,
        "held": held,
    }, indent=2) + "\n")
    print(f"AUDIT PASSED {len(receipts)}/{len(jobs)} HELD={len(held)}; cost=${sum(by_provider.values()):.2f} {by_provider}; receipt={target}", flush=True)


if __name__ == "__main__":
    main()
