#!/usr/bin/env python3
"""Full-run ablation of the self-inspiration sentence in Sonnet 4.5 dyads; dry-run by default.

20 runs in the no_self_line_* sets of config/experiments_noisy.yaml (added 2026-11-27):
Sonnet 4.5 dyads, game_myth and myth_game, no-defector condition, replicates 0-9. The
later-round myth prompt drops the sentence "Use the myth you wrote in the previous round as
inspiration, but adapt it in your own way." and nothing else.

The plan step proves, before any paid call, that every run has the inputs, request plan and
paired noise seed of its September baseline run (replicates 0-4 in the *_n5 set, 5-9 in the
*_n5ext set) except for that one sentence. The audit step re-checks every final, confirms
the sentence is absent from every prompt sent, and prices the run from recorded token usage.
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
from scripts.run_table1_n10_extension import ErroredCall, _quiet, _seeds, audit as audit_final  # noqa: E402
from experiments.run_noisy_batch import build_noisy_protocol  # noqa: E402
from src.utils import is_exhausted_quota  # noqa: E402

OUTPUT = "no_self_line_dyads_20261127"
LINE = "Use the myth you wrote in the previous round as inspiration, but adapt it in your own way. "
ARM = "memtest_memory_primary_noselfline"
PER_RUN_USD = {"game_myth": 2 * 0.4041, "myth_game": 2 * 0.3955}  # run_table1_n10_extension.PER_AGENT_USD


def plan():
    jobs = []
    for shape in ("game_myth", "myth_game"):
        name = f"no_self_line_dyad_{shape}_claude_n10"
        src = f"negative_only_reasoning_rerun_dyad_{shape}_claude_n5"
        combos = _quiet(name)
        baseline = {o["replicate_id"]: o for o in _quiet(src) + _quiet(src + "ext")
                    if o["game_params_name"] == "noisy2_crossmodel_negative_twotask_r3"}
        assert [c["replicate_id"] for c in combos] == list(range(10)) and sorted(baseline) == list(range(10))
        for i, c in enumerate(combos):
            o = baseline[c["replicate_id"]]
            a, b = c["comparison_inputs"], o["comparison_inputs"]
            differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            assert differing == ["myth_writing_later_rounds"], (name, differing)
            assert b["myth_writing_later_rounds"].count(LINE) == 1
            assert a["myth_writing_later_rounds"] == b["myth_writing_later_rounds"].replace(LINE, "")
            assert c["request_plan"].as_dict() == o["request_plan"].as_dict()
            assert c["game_params"] == o["game_params"] and _seeds(c) == _seeds(o)
            assert c["myth_prompt_arm_id"] == ARM
            game, _ = build_noisy_protocol(c, i)
            assert len(game.defector_agent_ids) == 0 and game.random_defection_probability == 0
            jobs.append((name, i, c, expected_output_path(c, name, i, OUTPUT)))
    assert len(jobs) == 20 and len({str(j[3]) for j in jobs}) == 20
    return jobs


def audit(job):
    receipt = audit_final(job)
    d = json.loads(job[3].read_text())
    myth_calls = 0
    for a in d["agents"].values():
        for e in a.get("interaction_history", []):
            for m in e.get("messages_sent") or []:
                assert LINE.strip() not in str(m.get("content")), (job[0], "sentence present in a sent prompt")
            myth_calls += (e.get("metadata") or {}).get("task") == "myth"
    assert myth_calls >= 20, (job[0], myth_calls)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args()
    jobs = plan()
    pending, receipts = [], []
    for j in jobs:
        if j[3].exists():
            receipts.append(audit(j))
        else:
            pending.append(j)
    est = sum(PER_RUN_USD["_".join(j[2]["task_order"])] for j in pending)
    print(f"VALIDATED N={len(jobs)} ONLY_DIFFERENCE=self-inspiration sentence SEEDS=baseline EXISTING={len(receipts)} "
          f"PENDING={len(pending)}", flush=True)
    print(f"MODEL=anthropic/claude-sonnet-4.5 (September profile, thinking 8192) N={len(pending)} WORKERS={args.workers} "
          f"EST_COST=${est:.2f} (+20% allowance ${1.2 * est:.2f})", flush=True)
    if args.audit_only:
        assert not pending
    elif not args.execute:
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
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_missing_job, j[2], j[0], j[1], OUTPUT, logdir): j for j in pending}
                for f in as_completed(futures):
                    j = futures[f]
                    try:
                        result = f.result()
                        if not result.get("success"):
                            if is_exhausted_quota(result.get("error", "")):
                                raise SystemExit("Provider credits exhausted; completed finals preserved.")
                            raise RuntimeError(f"Worker failed; inspect {result.get('worker_log')}: {result.get('error')}")
                        receipts.append(audit(j))
                        print(f"COMPLETE {len(receipts)}/{len(jobs)} {j[0]} index={j[1]} "
                              f"cost=${receipts[-1]['standard_rate_usd']:.3f}", flush=True)
                    except ErroredCall as e:
                        # A final with an errored call is kept aside and resampled under the same seed.
                        aside = j[3].with_suffix(f".errored_{time.strftime('%H%M%S')}.json")
                        j[3].rename(aside)
                        print(f"SET ASIDE {aside.name}: {e}", flush=True)
                        failed.append(j)
                    except Exception as e:
                        print(f"FAILED {j[0]} index={j[1]} {type(e).__name__}: {e}", flush=True)
                        if j[3].exists():  # never silently resample a final that fails validation
                            raise
                        failed.append(j)
            pending = failed
            if pending:
                if attempt == 10:
                    raise RuntimeError(f"{len(pending)} runs failed after ten attempts")
                workers = max(1, workers // 2)
                print(f"CONTINUATION PENDING={len(pending)} WORKERS={workers} ATTEMPT={attempt + 1}", flush=True)
                time.sleep(min(60, 2 ** attempt))
    target = ROOT / "data/json/noise_experiments" / OUTPUT / "completion_receipt.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    total = sum(r["standard_rate_usd"] for r in receipts)
    target.write_text(json.dumps({"runs": len(receipts), "standard_rate_usd": total, "finals": receipts}, indent=2) + "\n")
    print(f"AUDIT PASSED {len(receipts)}/{len(jobs)}; cost=${total:.2f}; receipt={target}", flush=True)


if __name__ == "__main__":
    main()
