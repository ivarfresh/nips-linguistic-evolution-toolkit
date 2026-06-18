"""Run baseline-matching ablations for the trust-game prompt shift.

This script keeps the diagnostic runs out of the main experiment configs. It
creates three no-noise GPT-5-nano cells:

1. old_prompt_old_runner_15: config/experiments.yaml prompts + TrustGame, 15 turns
2. old_prompt_old_runner_10: config/experiments.yaml prompts + TrustGame, 10 turns
3. old_prompt_noisy_runner_10: config/experiments.yaml prompts + TrustGameNoisy, 10 turns

The current v4 baseline remains the comparison cell:
data/json/noise_experiments/v4_direct_provider_baseline/baseline_v4_mem3_direct
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from games.trust_game import TrustGame
from games.trust_game_noisy import TrustGameNoisy
from src.batch_utils import sanitize_for_filename
from src.myth_writer import MythWriter
from src.simulation import run_simulation


MODEL = "openai/gpt-5-nano"
MODEL_NAME = "gpt-5-nano"
PERSONA_ID = "neutral"
MYTH_TOPIC_ID = "anything"
TASK_ORDERS = (("game",), ("game", "myth"), ("myth", "game"))


class JobTimeoutError(TimeoutError):
    pass


def _handle_job_timeout(signum, frame):
    raise JobTimeoutError("baseline-match job timed out")


@dataclass(frozen=True)
class Cell:
    name: str
    game_kind: str
    prompt_source: str
    num_turns: int


CELLS = {
    "old_prompt_old_runner_15": Cell(
        name="old_prompt_old_runner_15",
        game_kind="old",
        prompt_source="old",
        num_turns=15,
    ),
    "old_prompt_old_runner_10": Cell(
        name="old_prompt_old_runner_10",
        game_kind="old",
        prompt_source="old",
        num_turns=10,
    ),
    "old_prompt_noisy_runner_10": Cell(
        name="old_prompt_noisy_runner_10",
        game_kind="noisy",
        prompt_source="old",
        num_turns=10,
    ),
    "new_prompt_noisy_runner_10": Cell(
        name="new_prompt_noisy_runner_10",
        game_kind="noisy",
        prompt_source="noisy",
        num_turns=10,
    ),
}


DEFAULT_CELLS = (
    "old_prompt_old_runner_15",
    "old_prompt_old_runner_10",
    "old_prompt_noisy_runner_10",
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prompt_bundle(prompt_source: str) -> dict[str, Any]:
    if prompt_source == "old":
        config = load_yaml(PROJECT_ROOT / "config" / "experiments.yaml")
    elif prompt_source == "noisy":
        config = load_yaml(PROJECT_ROOT / "config" / "experiments_noisy.yaml")
    else:
        raise ValueError(f"Unknown prompt_source={prompt_source!r}")

    prompts = config["prompt_templates"]
    personas = config["personas"]
    myth_topics = config["myth_topics"]

    return {
        "system": prompts["trust_game_default"],
        "round1_investor": prompts["trust_game_round1_investor"],
        "round1_trustee": prompts["trust_game_round1_trustee"],
        "later_investor": prompts["trust_game_later_investor"],
        "later_trustee": prompts["trust_game_later_trustee"],
        "myth_round1": prompts["myth_writing_default"],
        "myth_later": prompts["myth_writing_later_rounds"],
        "persona": personas[PERSONA_ID],
        "myth_topic": myth_topics[MYTH_TOPIC_ID],
    }


def output_path(cell: Cell, run: int, task_order: tuple[str, ...], output_root: Path) -> Path:
    task_order_str = "_".join(task_order)
    myth_topic_suffix = ""
    if "myth" in task_order:
        myth_topic_suffix = "_" + sanitize_for_filename(MYTH_TOPIC_ID)

    filename = f"{cell.name}_{run:03d}_{PERSONA_ID}{myth_topic_suffix}.json"
    return output_root / cell.name / MODEL_NAME / task_order_str / "default" / filename


def build_jobs(
    cells: list[Cell],
    num_runs: int,
    output_root: Path,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for cell in cells:
        for task_order in TASK_ORDERS:
            for run in range(num_runs):
                path = output_path(cell, run, task_order, output_root)
                if path.exists():
                    continue
                jobs.append(
                    {
                        "cell": cell,
                        "run": run,
                        "task_order": task_order,
                        "output_root": str(output_root),
                        "save_path": str(path),
                    }
                )
    if limit is not None:
        jobs = jobs[:limit]
    return jobs


def make_game(cell: Cell, bundle: dict[str, Any]):
    personas = {"Agent_1": bundle["persona"], "Agent_2": bundle["persona"]}
    kwargs = {
        "endowment": 5,
        "multiplier": 3,
        "system_prompt_template": bundle["system"],
        "personas": personas,
        "round1_investor_template": bundle["round1_investor"],
        "round1_trustee_template": bundle["round1_trustee"],
        "later_investor_template": bundle["later_investor"],
        "later_trustee_template": bundle["later_trustee"],
    }
    if cell.game_kind == "old":
        return TrustGame(**kwargs)
    if cell.game_kind == "noisy":
        return TrustGameNoisy(
            **kwargs,
            noise_config=None,
            other_player_names="default",
        )
    raise ValueError(f"Unknown game_kind={cell.game_kind!r}")


def run_job(job: dict[str, Any], log_root: str, timeout_seconds: int) -> dict[str, Any]:
    cell = job["cell"]
    run = job["run"]
    task_order = tuple(job["task_order"])
    save_path = Path(job["save_path"])

    log_dir = Path(log_root) / cell.name
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_log = log_dir / f"{cell.name}_{'_'.join(task_order)}_{run:03d}.out"

    old_handler = signal.signal(signal.SIGALRM, _handle_job_timeout)
    if timeout_seconds > 0:
        signal.alarm(timeout_seconds)

    try:
        with worker_log.open("w", encoding="utf-8") as handle:
            with redirect_stdout(handle), redirect_stderr(handle):
                bundle = prompt_bundle(cell.prompt_source)
                game = make_game(cell, bundle)
                myth_writer = MythWriter(
                    myth_topic=bundle["myth_topic"],
                    round1_template=bundle["myth_round1"],
                    later_rounds_template=bundle["myth_later"],
                )

                save_path.parent.mkdir(parents=True, exist_ok=True)
                base_no_ext = save_path.with_suffix("")
                results_path = str(base_no_ext) + ".results.json"
                checkpoint_path = str(base_no_ext) + ".checkpoint.json"
                log_path = str(base_no_ext) + ".log"
                resume_from = checkpoint_path if Path(checkpoint_path).exists() else None

                sim_data = run_simulation(
                    game=game,
                    model=MODEL,
                    temperature=0.8,
                    num_turns=cell.num_turns,
                    num_agents=2,
                    memory_capacity=3,
                    agent_biases="",
                    myth_writer=myth_writer,
                    task_order=list(task_order),
                    results_path=results_path,
                    checkpoint_path=checkpoint_path,
                    checkpoint_every=10,
                    resume_from=resume_from,
                    log_file=log_path,
                )

                sim_data.run_metadata.update(
                    {
                        "baseline_match_cell": cell.name,
                        "baseline_match_prompt_source": cell.prompt_source,
                        "baseline_match_game_kind": cell.game_kind,
                        "baseline_match_run": run,
                        "myth_topic_id": MYTH_TOPIC_ID,
                        "myth_topic": bundle["myth_topic"],
                        "game_params_name": "default",
                        "noise_config": None,
                        "other_player_names": "default" if cell.game_kind == "noisy" else None,
                        "provider_env": os.environ.get("LLM_PROVIDER", ""),
                        "openai_reasoning_effort_env": os.environ.get("OPENAI_REASONING_EFFORT", ""),
                    }
                )
                sim_data.save_state(str(save_path))
                transcript_path = str(base_no_ext) + ".transcript.pdf"
                sim_data.save_transcript_pdf(transcript_path, source_path=str(save_path))

                cp_path = Path(checkpoint_path)
                if cp_path.exists():
                    cp_path.unlink()

        return {
            "success": True,
            "cell": cell.name,
            "run": run,
            "task_order": list(task_order),
            "file_path": str(save_path),
            "transcript_path": transcript_path,
            "worker_log": str(worker_log),
        }
    except Exception as exc:
        return {
            "success": False,
            "cell": cell.name,
            "run": run,
            "task_order": list(task_order),
            "file_path": str(save_path),
            "worker_log": str(worker_log),
            "error": repr(exc),
        }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def parse_cells(raw_cells: str) -> list[Cell]:
    names = [name.strip() for name in raw_cells.split(",") if name.strip()]
    unknown = [name for name in names if name not in CELLS]
    if unknown:
        raise SystemExit(f"Unknown cells: {', '.join(unknown)}")
    return [CELLS[name] for name in names]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        default=",".join(DEFAULT_CELLS),
        help=f"Comma-separated cells. Available: {', '.join(CELLS)}",
    )
    parser.add_argument("--num-runs", type=int, default=15)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output-root",
        default="data/json/baseline_match_ablation/direct",
        help="Root output directory.",
    )
    parser.add_argument(
        "--log-dir",
        default="/tmp/nlet-runs/baseline-match-ablation",
        help="Worker stdout/stderr log directory.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--job-timeout-seconds",
        type=int,
        default=900,
        help="Wall-clock timeout per run; 0 disables the timeout.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cells = parse_cells(args.cells)
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    jobs = build_jobs(cells, args.num_runs, output_root, args.limit)

    print(
        f"baseline_match_ablation cells={','.join(cell.name for cell in cells)} "
        f"num_runs={args.num_runs} jobs_missing={len(jobs)} workers={args.workers} "
        f"output_root={output_root}",
        flush=True,
    )

    if args.dry_run or not jobs:
        return 0

    completed = 0
    failed: list[dict[str, Any]] = []

    if args.workers == 1:
        for job in jobs:
            result = run_job(job, args.log_dir, args.job_timeout_seconds)
            completed += 1
            if result["success"]:
                print(
                    f"[{completed}/{len(jobs)}] ok {result['cell']} "
                    f"{'_'.join(result['task_order'])} run={result['run']:03d}",
                    flush=True,
                )
            else:
                failed.append(result)
                print(
                    f"[{completed}/{len(jobs)}] FAIL {result['cell']} "
                    f"{'_'.join(result['task_order'])} run={result['run']:03d} "
                    f"log={result['worker_log']}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {
                executor.submit(run_job, job, args.log_dir, args.job_timeout_seconds): job
                for job in jobs
            }
            for future in as_completed(future_to_job):
                completed += 1
                result = future.result()
                if result["success"]:
                    print(
                        f"[{completed}/{len(jobs)}] ok {result['cell']} "
                        f"{'_'.join(result['task_order'])} run={result['run']:03d}",
                        flush=True,
                    )
                else:
                    failed.append(result)
                    print(
                        f"[{completed}/{len(jobs)}] FAIL {result['cell']} "
                        f"{'_'.join(result['task_order'])} run={result['run']:03d} "
                        f"log={result['worker_log']}",
                        flush=True,
                    )

    if failed:
        summary_path = Path(args.log_dir) / "failed.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")
        print(f"failed={len(failed)} summary={summary_path}", flush=True)
        return 1

    print(f"complete={completed} failed=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
