#!/usr/bin/env python3
"""Self-inspiration sentence ablation in mixed dyads (Sonnet 4.5 with GPT-5 Nano / Gemini 3.7 Flash).

20 runs in the no_self_line_mixed_dyad_* sets of config/experiments_noisy.yaml (2026-11-27):
two pairs x game_myth / myth_game x replicates 0-4 (0/2/4 Sonnet sends first, 1/3 the other
family). Dry-run by default.

The plan step proves, before any paid call, that every run has the inputs and paired seeds of
its September mixed-dyad baseline run except the deleted sentence. For Sonnet+Gemini the
request route also differs: Gemini 3.7 Flash goes through Vertex AI with otherwise the same
request policy (the Gemini API now answers 3.7 requests with 3.8). The audit re-checks every
final, including which model answered, and confirms the sentence is in no prompt sent.
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
from scripts.run_mixed_model_populations import RATES, audit as audit_final  # noqa: E402
from scripts.run_table1_n10_extension import PER_AGENT_USD, _quiet, _seeds  # noqa: E402
from experiments.run_noisy_batch import build_noisy_protocol  # noqa: E402
from src.utils import is_exhausted_quota  # noqa: E402

OUTPUT = "no_self_line_mixed_dyads_20261127"
LINE = "Use the myth you wrote in the previous round as inspiration, but adapt it in your own way. "
ARM = "memtest_memory_primary_noselfline"
ORDERINGS = [("sonnet", "gpt", [0, 2, 4]), ("gpt", "sonnet", [1, 3]),
             ("sonnet", "gemini", [0, 2, 4]), ("gemini", "sonnet", [1, 3])]


def plan():
    jobs = []
    for shape in ("game_myth", "myth_game"):
        for first, second, reps in ORDERINGS:
            name = f"no_self_line_mixed_dyad_{shape}_{first}_{second}"
            combos = _quiet(name)
            baseline = {o["replicate_id"]: o for o in _quiet(f"mixed_dyad_{shape}_{first}_{second}_n3")}
            assert [c["replicate_id"] for c in combos] == reps, (name, [c["replicate_id"] for c in combos])
            vertex = "gemini" in (first, second)
            for i, c in enumerate(combos):
                o = baseline[c["replicate_id"]]
                a, b = c["comparison_inputs"], o["comparison_inputs"]
                differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
                assert differing == (["llm_request", "myth_writing_later_rounds"] if vertex else ["myth_writing_later_rounds"]), (name, differing)
                assert b["myth_writing_later_rounds"].count(LINE) == 1
                assert a["myth_writing_later_rounds"] == b["myth_writing_later_rounds"].replace(LINE, "")
                new, old = c["request_plan"].as_dict()["agents"], o["request_plan"].as_dict()["agents"]
                for agent in new:
                    if new[agent]["provider"] == "vertex":
                        assert old[agent]["provider"] == "google" and new[agent]["provider_model"] == old[agent]["provider_model"]
                        assert ({k: v for k, v in new[agent]["policy"].items() if k != "provider"}
                                == {k: v for k, v in old[agent]["policy"].items() if k != "provider"})
                    else:
                        assert new[agent] == old[agent], (name, agent)
                assert c["agent_models"] == o["agent_models"]
                assert c["game_params"] == o["game_params"] and _seeds(c) == _seeds(o)
                assert c["myth_prompt_arm_id"] == ARM
                game, _ = build_noisy_protocol(c, i)
                assert len(game.defector_agent_ids) == 0 and game.random_defection_probability == 0
                jobs.append((name, i, c, expected_output_path(c, name, i, OUTPUT)))
    assert len(jobs) == 20 and len({str(j[3]) for j in jobs}) == 20
    return jobs


def estimate(jobs):
    total = dict.fromkeys(RATES, 0.0)
    for _, _, c, _ in jobs:
        for req in c["request_plan"].as_dict()["agents"].values():
            total[req["provider"]] += PER_AGENT_USD[2]["google" if req["provider"] == "vertex" else req["provider"]]["_".join(c["task_order"])]
    return total


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
    p.add_argument("--smoke", action="store_true", help="run only one Sonnet+Gemini dyad (checks the Vertex route)")
    args = p.parse_args()
    jobs = plan()
    if args.smoke:
        jobs = [j for j in jobs if j[0] == "no_self_line_mixed_dyad_game_myth_sonnet_gemini" and j[1] == 0]
    pending, receipts = [], []
    for j in jobs:
        if j[3].exists():
            receipts.append(audit(j))
        else:
            pending.append(j)
    est = estimate(pending)
    print(f"VALIDATED N={len(jobs)} ONLY_DIFFERENCE=self-inspiration sentence (+Vertex route for Gemini) SEEDS=baseline "
          f"EXISTING={len(receipts)} PENDING={len(pending)}", flush=True)
    print(f"MODEL=sonnet-4.5 + gpt-5-nano / gemini-3.7-flash via Vertex (September profiles) N={len(pending)} "
          f"WORKERS={args.workers} EST_COST=${sum(est.values()):.2f} " + str({k: round(v, 2) for k, v in est.items() if v}), flush=True)
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
    if args.smoke:
        print(f"SMOKE PASSED {len(receipts)}/{len(jobs)} cost=${sum(r['standard_rate_usd'] for r in receipts):.2f}", flush=True)
        return
    target = ROOT / "data/json/noise_experiments" / OUTPUT / "completion_receipt.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    total = sum(r["standard_rate_usd"] for r in receipts)
    target.write_text(json.dumps({"runs": len(receipts), "standard_rate_usd": total, "finals": receipts}, indent=2) + "\n")
    print(f"AUDIT PASSED {len(receipts)}/{len(jobs)}; cost=${total:.2f}; receipt={target}", flush=True)


if __name__ == "__main__":
    main()
