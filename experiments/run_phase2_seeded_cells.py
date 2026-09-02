"""Phase 2 seeded-cell runner (memory-transplant ablation §17).

Runs the 60 seeded cells from §17.5: 4 seed types × 3 task orders ×
5 reps. Reuses TrustGameNoisy + run_simulation with seed_myth and
seed_user_prompt threaded through.

Reads the seed manifest produced by scripts/phase2_harvest_seeds.py and
the noise + game settings from a Phase 2 cell config in
config/experiments_noisy.yaml (named entry under game_params).

Usage:
  python experiments/run_phase2_seeded_cells.py \\
      --manifest data/phase2/seed_manifest.json \\
      --noise-game-params phase2_8agent_history3_anon_neg5 \\
      --output-subdir phase2_seeded \\
      --workers 4
"""

import argparse
import contextlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.batch_utils import unique_json_path as _unique_json_path
from src.batch_utils import sanitize_for_filename as _sanitize_for_filename
from src.myth_writer import MythWriter
from src.simulation import run_simulation
from games.trust_game_noisy import TrustGameNoisy

CONFIG_PATH = Path("config/experiments_noisy.yaml")
SEED_TYPES = ["s_start", "s_end_plus", "s_end_minus", "s_filler"]
TASK_ORDERS = [["game"], ["game", "myth"], ["myth", "game"]]
MYTH_PROMPT_ARM = {
    "id": "myth_game_directive",
    "default": "myth_writing_default_game_directive",
    "later": "myth_writing_later_rounds_directive",
}


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _resolve_template(config, key):
    try:
        return config["prompt_templates"][key]
    except KeyError as exc:
        raise KeyError(f"Missing prompt template '{key}'") from exc


def _seed_user_prompt(config):
    template = _resolve_template(config, "myth_writing_default_game_directive")
    return template.format(
        topic_instruction="You may choose any mythic setting, characters, or symbols.",
        shared_context_block="",
        myth_topic="anything",
    )


def _make_combo(seed_type, seed, task_order, rep, config, noise_game_params_name):
    game_params = dict(config["game_params"][noise_game_params_name])
    task_order_str = "_".join(task_order)
    return {
        "seed_type": seed_type,
        "seed_text": seed["text"],
        "seed_meta": {
            "source_run": seed.get("source_run"),
            "agent_id": seed.get("agent_id"),
            "round": seed.get("round"),
            "joint_at_source": seed.get("joint_at_source"),
            "tokens": seed.get("tokens"),
        },
        "task_order": task_order,
        "task_order_str": task_order_str,
        "rep": rep,
        "model": config["base_models"]["claude_sonnet_45"],
        "template": _resolve_template(config, "trust_game_default"),
        "system_prompt_template": "trust_game_default",
        "noise_game_params_name": noise_game_params_name,
        "game_params": game_params,
        "round1_investor": _resolve_template(config, "trust_game_round1_investor"),
        "round1_trustee": _resolve_template(config, "trust_game_round1_trustee"),
        "later_investor": _resolve_template(config, "trust_game_later_investor"),
        "later_trustee": _resolve_template(config, "trust_game_later_trustee"),
        "myth_default": _resolve_template(config, MYTH_PROMPT_ARM["default"]),
        "myth_later": _resolve_template(config, MYTH_PROMPT_ARM["later"]),
        "seed_user_prompt": _seed_user_prompt(config),
    }


def run_one(combo, index, output_subdir):
    try:
        game_params = combo["game_params"]
        game = TrustGameNoisy(
            endowment=game_params["endowment"],
            multiplier=game_params["multiplier"],
            system_prompt_template=combo["template"],
            personas={f"Agent_{i+1}": {"description": "neutral", "system_addition": ""} for i in range(game_params["num_agents"])},
            round1_investor_template=combo["round1_investor"],
            round1_trustee_template=combo["round1_trustee"],
            later_investor_template=combo["later_investor"],
            later_trustee_template=combo["later_trustee"],
            noise_config=game_params.get("noise_config"),
            other_player_names=game_params.get("other_player_names", "default"),
            history_policy=game_params.get("history_policy", "minimal"),
            self_history_window=game_params.get("self_history_window", 1),
            coplayer_history_window=game_params.get("coplayer_history_window", 0),
            show_agent_names=game_params.get("show_agent_names", True),
        )
        myth_writer = MythWriter(
            myth_topic="anything",
            round1_template=combo["myth_default"],
            later_rounds_template=combo["myth_later"],
        )

        save_dir = (
            f"data/json/noise_experiments/{output_subdir}/"
            f"phase2_seeded_{combo['seed_type']}_{combo['noise_game_params_name']}/"
            f"claude-sonnet-4.5/{combo['task_order_str']}/default"
        )
        os.makedirs(save_dir, exist_ok=True)

        filename = (
            f"phase2_seeded_{combo['seed_type']}_{index:03d}_neutral_rep{combo['rep']:02d}_"
            f"{MYTH_PROMPT_ARM['id']}_anything.json"
        )
        save_path = _unique_json_path(os.path.join(save_dir, filename))
        base_no_ext = save_path[:-5] if save_path.endswith(".json") else save_path
        results_path = base_no_ext + ".results.json"
        checkpoint_path = base_no_ext + ".checkpoint.json"
        log_path = base_no_ext + ".log"
        resume_from = checkpoint_path if os.path.exists(checkpoint_path) else None

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"PHASE 2 SEEDED CELL\n")
            f.write(f"Seed type: {combo['seed_type']}\n")
            f.write(f"Task order: {combo['task_order']}\n")
            f.write(f"Noise params: {combo['noise_game_params_name']}\n")
            f.write(f"Rep: {combo['rep']}\n")
            f.write(f"Seed meta: {json.dumps(combo['seed_meta'])}\n")
            f.write(f"Seed text (first 200 chars): {combo['seed_text'][:200]}\n")
            f.write("=" * 80 + "\n\n")

        kwargs = {
            "game": game,
            "model": combo["model"],
            "temperature": game_params.get("temperature", 0.8),
            "num_turns": game_params["num_turns"],
            "num_agents": game_params["num_agents"],
            "memory_capacity": game_params["memory_capacity"],
            "agent_biases": "",
            "myth_writer": myth_writer,
            "task_order": combo["task_order"],
            "results_path": results_path,
            "checkpoint_path": checkpoint_path,
            "checkpoint_every": 10,
            "resume_from": resume_from,
            "log_file": log_path,
            "seed_myth": combo["seed_text"],
            "seed_user_prompt": combo["seed_user_prompt"],
        }

        quiet = os.environ.get("TRUST_BATCH_QUIET", "").lower() in {"1", "true", "yes"}
        if quiet:
            with open(log_path, "a", encoding="utf-8") as log_stream:
                with contextlib.redirect_stdout(log_stream):
                    sim_data = run_simulation(**kwargs)
        else:
            sim_data = run_simulation(**kwargs)

        sim_data.run_metadata["phase2_seed_type"] = combo["seed_type"]
        sim_data.run_metadata["phase2_seed_meta"] = combo["seed_meta"]
        sim_data.run_metadata["phase2_noise_game_params"] = combo["noise_game_params_name"]
        sim_data.run_metadata["phase2_rep"] = combo["rep"]
        sim_data.run_metadata["myth_prompt_arm_id"] = MYTH_PROMPT_ARM["id"]
        sim_data.save_state(save_path)
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
            except OSError:
                pass
        return {
            "success": True,
            "file_path": save_path,
            "seed_type": combo["seed_type"],
            "task_order": combo["task_order_str"],
            "rep": combo["rep"],
        }
    except Exception as exc:
        import traceback
        return {
            "success": False,
            "error": f"{exc}\n{traceback.format_exc()}",
            "seed_type": combo["seed_type"],
            "task_order": combo["task_order_str"],
            "rep": combo["rep"],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Seed manifest JSON path")
    parser.add_argument(
        "--noise-game-params",
        required=True,
        help="Named game_params block in config/experiments_noisy.yaml (e.g., phase2_8agent_history3_anon_neg5)",
    )
    parser.add_argument(
        "--output-subdir", default="phase2_seeded", help="Output subdirectory"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--seed-types",
        nargs="*",
        default=SEED_TYPES,
        help="Subset of seed types to run",
    )
    parser.add_argument(
        "--task-orders",
        nargs="*",
        default=None,
        help="Subset of task orders (e.g., 'game game_myth myth_game')",
    )
    parser.add_argument(
        "--reps", type=int, default=5, help="Number of reps per seed/task-order cell"
    )
    args = parser.parse_args()

    config = _load_config()
    with open(args.manifest) as f:
        manifest = json.load(f)

    task_orders = TASK_ORDERS
    if args.task_orders:
        order_lookup = {
            "game": ["game"],
            "game_myth": ["game", "myth"],
            "myth_game": ["myth", "game"],
        }
        task_orders = [order_lookup[o] for o in args.task_orders]

    combos = []
    for seed_type in args.seed_types:
        seeds = manifest["seeds"].get(seed_type, [])
        if not seeds:
            print(f"[warn] no seeds for {seed_type}, skipping")
            continue
        for task_order in task_orders:
            for rep in range(args.reps):
                seed = seeds[rep % len(seeds)]
                combo = _make_combo(
                    seed_type,
                    seed,
                    task_order,
                    rep,
                    config,
                    args.noise_game_params,
                )
                combos.append(combo)

    print(f"Total seeded combos: {len(combos)}")
    print(
        f"Preflight: MODEL=anthropic/claude-sonnet-4.5 N={len(combos)} "
        f"WORKERS={args.workers} EST_COST≈${len(combos) * 2.30:.2f}"
    )

    start = time.time()
    results = []
    if args.workers <= 1:
        for i, combo in enumerate(combos):
            results.append(run_one(combo, i, args.output_subdir))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(run_one, combo, i, args.output_subdir)
                for i, combo in enumerate(combos)
            ]
            for f in as_completed(futures):
                r = f.result()
                results.append(r)
                status = "OK" if r["success"] else "FAIL"
                print(
                    f"[{status}] {r['seed_type']:14} {r['task_order']:12} rep{r['rep']:02d}"
                )

    elapsed = time.time() - start
    ok = sum(1 for r in results if r["success"])
    print(f"\n{ok}/{len(results)} succeeded in {elapsed/60:.1f} min")
    if ok != len(results):
        print("Failures:")
        for r in results:
            if not r["success"]:
                print(f"  {r['seed_type']} {r['task_order']} rep{r['rep']}: {r['error'][:120]}")


if __name__ == "__main__":
    main()
