#!/usr/bin/env python3
"""Frontier myth-board runs (2026-10-02); dry-run by default.

The 2026-10-01 defector mix (Agent_1-4 Claude Opus 5, Agent_5-8 GPT-5.6 Sol,
Agent_4 and Agent_8 forced-zero defectors) with one change: from round 2 every
agent reads every myth any agent has written in earlier rounds (a shared,
persistent board, without author names) instead of its previous partner's myth.
The board is shown in the myth prompt only; chat memory keeps a one-line note,
so it is not repeated in later prompts. Inspired by the 2026 Hugging Face
incident board; agents neither find nor organise this board.

Task orders game_myth and myth_game, replicates 0-4 (10 runs). Stage ``pilot``
is myth_game replicates 0-2. The comparison is the same replicate of the
defector mix in scripts/run_frontier_defector_populations.py: the plan step
proves that each board run differs from it only in the board.
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
from scripts.build_frontier_rerun_config import DEFECTOR_PILOT_NAME, DEFECTOR_PILOT_IDS, BOARD_TEMPLATE, board_sets  # noqa: E402
from scripts import run_frontier_defector_populations as defectors  # noqa: E402
from scripts.run_frontier_defector_populations import CONFIG, RATES, PER_AGENT_USD, require, _quiet_combinations, _agent_plans  # noqa: E402
from src.utils import is_exhausted_quota  # noqa: E402

OUTPUT = "frontier_myth_board_20261002"
SHAPES = ("game_myth", "myth_game")
# Extra input per agent per run: by round r the board holds 8(r-1) myths of about 300 tokens,
# read once in that round's myth call (rounds 2-10: 8 x 45 x 300 = 108k tokens). Input rates only.
BOARD_INPUT_TOKENS_PER_AGENT = 8 * 45 * 300
BOARD_DIFF = {"myth_board", "myth_writing_later_rounds", "game_params"}


def plan():
    reference = {(j["shape"], j["combo"]["replicate_id"]): j["combo"] for j in defectors.plan() if j["comp"] == DEFECTOR_PILOT_NAME}
    jobs = []
    for name, set_shape, replicate_ids in board_sets():
        shape = set_shape.removeprefix("population_")
        combos = _quiet_combinations(name, CONFIG)
        require([c["replicate_id"] for c in combos] == replicate_ids, name, [c["replicate_id"] for c in combos])
        for i, c in enumerate(combos):
            base = reference[(shape, c["replicate_id"])]
            x, y = c["comparison_inputs"], base["comparison_inputs"]
            differing = {k for k in set(x) | set(y) if x.get(k) != y.get(k)}
            require(differing == BOARD_DIFF, name, "differs from the defector mix in more than the board", sorted(differing))
            require({k for k in set(x["game_params"]) | set(y["game_params"]) if x["game_params"].get(k) != y["game_params"].get(k)} == {"myth_board"},
                    name, "game params differ beyond myth_board")
            require(c["myth_board"] == "persistent" and c["game_params"]["myth_board"] == "persistent", name, "board flag")
            require(c["myth_writing_later_rounds"] == BOARD_TEMPLATE, name, "board template")
            require(c["request_plan"].as_dict() == base["request_plan"].as_dict(), name, "request plan")
            _, writer = defectors.build_noisy_protocol(c, i)
            require(writer.board == "persistent", name, "MythWriter board")
            jobs.append({"name": name, "index": i, "combo": c, "size": 8, "shape": shape, "comp": DEFECTOR_PILOT_NAME,
                         "path": expected_output_path(c, name, i, OUTPUT)})
    require(len(jobs) == 10 and len({(j["shape"], j["combo"]["replicate_id"]) for j in jobs}) == 10, f"expected 10 jobs, found {len(jobs)}")
    jobs.sort(key=lambda j: (0 if j["shape"] == "myth_game" else 1, j["combo"]["replicate_id"]))
    return jobs


def estimate(jobs):
    total = {m: 0.0 for m in RATES}
    for j in jobs:
        for agent_request in _agent_plans(j["combo"]["request_plan"].as_dict()).values():
            model = agent_request["provider_model"]
            total[model] += PER_AGENT_USD[model][j["shape"]] + BOARD_INPUT_TOKENS_PER_AGENT * RATES[model][0] / 1e6
    return total


def audit(job):
    receipt = defectors.audit(job)
    d = json.loads(job["path"].read_text())
    for entry in d["conversation_history"]:
        r = entry["round"]
        if r == 1:
            continue
        exposures = entry.get("myth_exposures") or {}
        require(set(exposures) == set(d["agents"]), job["path"], r, "missing myth exposure records")
        for agent_id, record in exposures.items():
            require(record.get("board") == "persistent" and len(record["board_items"]) == 8 * (r - 1), job["path"], r, agent_id, "board size")
    return receipt


STAGES = {
    "pilot": lambda j: j["name"].startswith("frontier_board_pilot_"),
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
          f"POPULATION={DEFECTOR_PILOT_NAME} DEFECTORS={','.join(DEFECTOR_PILOT_IDS)} MYTH_BOARD=persistent", flush=True)
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
        rows = sorted((r for r in receipts if r["task_order"] == shape), key=lambda r: r["replicate_id"])
        if rows:
            values = [r["ordinary_mean_final"] for r in rows]
            print(f"RESULT board {shape}: ordinary-agent mean final {sum(values) / len(values):.1f} over {len(rows)} runs "
                  f"{[round(v, 1) for v in values]}", flush=True)
    print(f"AUDIT PASSED {len(receipts)}/{len(selected)}; cost=${sum(by_model.values()):.2f} {by_model}; receipt={target}", flush=True)


if __name__ == "__main__":
    main()
