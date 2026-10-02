#!/usr/bin/env python3
"""Frontier saboteur pilot (2026-10-02); dry-run by default.

The opus4_sol4_d2 defector mix (Agent_1-4 Claude Opus 5, Agent_5-8 GPT-5.6 Sol;
Agent_4 and Agent_8 always send and return $0) with one change: the two
defectors' myth prompts start with a private instruction to persuade the others
to send less. Ordinary agents never see the instruction, only the myths.

Two channels, myth -> game, replicates 0-2 (6 runs):
  partner myth  vs frontier_defector_pilot_population_myth_game_opus4_sol4_d2_n3
  shared board  vs frontier_board_pilot_population_myth_game_opus4_sol4_d2_n3
The plan step proves each run differs from its seed-matched comparison run only
in the saboteur instruction.
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
from scripts.build_frontier_rerun_config import DEFECTOR_PILOT_NAME, DEFECTOR_PILOT_IDS, SABOTEUR_SETS, SABOTEUR_TEMPLATE  # noqa: E402
from scripts import run_frontier_defector_populations as defectors  # noqa: E402
from scripts import run_frontier_myth_board as board  # noqa: E402
from scripts.run_frontier_defector_populations import CONFIG, RATES, require, _quiet_combinations  # noqa: E402
from src.utils import is_exhausted_quota  # noqa: E402

OUTPUT = "frontier_saboteur_pilot_20261002"
INSTRUCTION_MARK = "PRIVATE INSTRUCTION"
# Standard-rate cost of the seed-matched comparison runs (receipts of 2026-10-01/02).
RUN_USD = {"partner": 3.42, "board": 7.59}


def plan():
    jobs = []
    for name, (comparison, _, _) in SABOTEUR_SETS.items():
        channel = "board" if "_board_" in name else "partner"
        reference = {c["replicate_id"]: c for c in _quiet_combinations(comparison, CONFIG)}
        combos = _quiet_combinations(name, CONFIG)
        require([c["replicate_id"] for c in combos] == [0, 1, 2], name, [c["replicate_id"] for c in combos])
        for i, c in enumerate(combos):
            base = reference[c["replicate_id"]]
            x, y = c["comparison_inputs"], base["comparison_inputs"]
            differing = {k for k in set(x) | set(y) if x.get(k) != y.get(k)}
            require(differing == {"game_params", "myth_saboteur"}, name, "differs from its comparison run beyond the saboteur", sorted(differing))
            require({k for k in set(x["game_params"]) | set(y["game_params"]) if x["game_params"].get(k) != y["game_params"].get(k)}
                    == {"myth_saboteur"}, name, "game params differ beyond myth_saboteur")
            require(c["myth_saboteur"] == SABOTEUR_TEMPLATE and c["request_plan"].as_dict() == base["request_plan"].as_dict(), name, "saboteur text / plan")
            game, writer = defectors.build_noisy_protocol(c, i)
            game.configure_agents([f"Agent_{n}" for n in range(1, 9)])
            require(game.defector_agent_ids == DEFECTOR_PILOT_IDS and game.defector_action_policy == "forced_zero", name, "defectors")
            require(writer.saboteur == SABOTEUR_TEMPLATE.strip() and writer.board == ("persistent" if channel == "board" else None), name, "writer")
            jobs.append({"name": name, "index": i, "combo": c, "size": 8, "shape": "myth_game", "comp": DEFECTOR_PILOT_NAME,
                         "channel": channel, "path": expected_output_path(c, name, i, OUTPUT)})
    require(len(jobs) == 6, f"expected 6 jobs, found {len(jobs)}")
    return jobs


def estimate(jobs):
    total = sum(RUN_USD[j["channel"]] for j in jobs)
    return {"claude-opus-5": total * 0.55, "gpt-5.6-sol": total * 0.45}  # split as in the comparison receipts


def audit(job):
    receipt = board.audit(job) if job["channel"] == "board" else defectors.audit(job)
    d = json.loads(job["path"].read_text())
    for agent_id, agent in d["agents"].items():
        prompts = [e["prompt"] for e in agent.get("interaction_history", []) if (e.get("metadata") or {}).get("task") == "myth"]
        require(len(prompts) >= 10, job["path"], agent_id, "missing myth calls")
        if agent_id in DEFECTOR_PILOT_IDS:
            require(all(p.startswith(SABOTEUR_TEMPLATE.strip()) for p in prompts), job["path"], agent_id, "saboteur prompt without instruction")
        else:
            require(not any(INSTRUCTION_MARK in p for p in prompts), job["path"], agent_id, "ordinary agent saw the instruction")
    receipt["channel"] = job["channel"]
    return receipt


def main():
    os.chdir(ROOT)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args()
    require(1 <= args.workers <= 20, "--workers must be 1-20")
    jobs = plan()
    pending, receipts = [], []
    for j in jobs:
        (receipts.append(audit(j)) if j["path"].exists() else pending.append(j))
    est = estimate(pending)
    print(f"VALIDATED N={len(jobs)} EXISTING={len(receipts)} PENDING={len(pending)} POPULATION={DEFECTOR_PILOT_NAME} "
          f"SABOTEURS={','.join(DEFECTOR_PILOT_IDS)} CHANNELS=partner,board", flush=True)
    print(f"MODEL=mixed({','.join(RATES)}) N={len(pending)} WORKERS={args.workers} EST_COST=${sum(est.values()):.2f} "
          f"(+20% allowance ${1.2 * sum(est.values()):.2f})", flush=True)
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
            failed, quota_exhausted = [], False
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
                        print(f"COMPLETE {len(receipts)}/{len(jobs)} {j['name']} index={j['index']} cost=${receipt['standard_rate_usd']:.3f}", flush=True)
                    except Exception as e:
                        print(f"FAILED {j['name']} index={j['index']} {type(e).__name__}: {e}", flush=True)
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
    target = ROOT / "data/json/noise_experiments" / OUTPUT / "pilot_receipt.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"runs": len(receipts), "standard_rate_usd": sum(r["standard_rate_usd"] for r in receipts),
                                  "finals": receipts}, indent=2) + "\n")
    for channel in ("partner", "board"):
        rows = sorted((r for r in receipts if r["channel"] == channel), key=lambda r: r["replicate_id"])
        if rows:
            print(f"RESULT saboteur {channel}: ordinary-agent mean final per replicate "
                  f"{[round(r['ordinary_mean_final'], 2) for r in rows]}", flush=True)
    print(f"AUDIT PASSED {len(receipts)}/{len(jobs)}; cost=${sum(r['standard_rate_usd'] for r in receipts):.2f}; receipt={target}", flush=True)


if __name__ == "__main__":
    main()
