"""Phase 3 seeded-cell runner.

Setup (see docs/phase3_chat_memory_spec.md):
- Chat memory at the start of every round = [system, seed_user, seed_myth].
- The seeded myth is re-injected at the top of each round; nothing else
  is ever appended to chat memory.
- No history-block in the prompt text. The round-N prompt is round number,
  balance, role, action request.
- Task order: ["game"] only.
- Variants: s_start (round-1 myth from team baseline) and s_end_plus
  (round-10 myth from team baseline).

Usage:
  python experiments/run_phase3_seeded_cells.py \\
      --manifest data/phase3/seed_manifest.json \\
      --noise-game-params phase3_8agent_anon_neg5_myth_only \\
      --model claude_haiku_45 \\
      --reps 2 \\
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
from src.myth_writer import MythWriter
from src.simulation import run_simulation
from games.trust_game_noisy import TrustGameNoisy

CONFIG_PATH = Path("config/experiments_noisy.yaml")
SEED_TYPES = ["s_start", "s_end_plus"]
TASK_ORDER = ["game"]
TASK_ORDER_LOOKUP = {
    "game": ["game"],
    "myth_game": ["myth", "game"],
    "game_myth": ["game", "myth"],
}
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


def _make_combo(seed_type, seed, rep, config, noise_game_params_name, model_key, task_order=None):
    game_params = dict(config["game_params"][noise_game_params_name])
    task_order = list(task_order) if task_order is not None else list(TASK_ORDER)
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
        "task_order_str": "_".join(task_order),
        "rep": rep,
        "model": config["base_models"][model_key],
        "model_key": model_key,
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
            history_policy=game_params.get("history_policy", "none"),
            self_history_window=game_params.get("self_history_window", 0),
            coplayer_history_window=game_params.get("coplayer_history_window", 0),
            show_agent_names=game_params.get("show_agent_names", False),
        )
        myth_writer = MythWriter(
            myth_topic="anything",
            round1_template=combo["myth_default"],
            later_rounds_template=combo["myth_later"],
        )

        model_slug = combo["model"].split("/")[-1]
        save_dir = (
            f"data/json/noise_experiments/{output_subdir}/"
            f"phase3_seeded_{combo['seed_type']}_{combo['noise_game_params_name']}/"
            f"{model_slug}/{combo['task_order_str']}/default"
        )
        os.makedirs(save_dir, exist_ok=True)

        filename = (
            f"phase3_seeded_{combo['seed_type']}_{index:03d}_neutral_rep{combo['rep']:02d}_"
            f"{MYTH_PROMPT_ARM['id']}_anything.json"
        )
        save_path = _unique_json_path(os.path.join(save_dir, filename))
        base_no_ext = save_path[:-5] if save_path.endswith(".json") else save_path
        results_path = base_no_ext + ".results.json"
        checkpoint_path = base_no_ext + ".checkpoint.json"
        log_path = base_no_ext + ".log"
        resume_from = checkpoint_path if os.path.exists(checkpoint_path) else None

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"PHASE 3 SEEDED CELL\n")
            f.write(f"Model: {combo['model']}\n")
            f.write(f"Seed type: {combo['seed_type']}\n")
            f.write(f"Task order: {combo['task_order']}\n")
            f.write(f"Noise params: {combo['noise_game_params_name']}\n")
            f.write(f"Rep: {combo['rep']}\n")
            f.write(f"chat_memory_mode: myth_only  |  seed_reinject: True\n")
            f.write(f"Seed meta: {json.dumps(combo['seed_meta'])}\n")
            f.write(f"Seed text (first 200 chars): {combo['seed_text'][:200]}\n")
            f.write("=" * 80 + "\n\n")

        is_baseline = combo["seed_type"] == "baseline"
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
            "seed_myth": None if is_baseline else combo["seed_text"],
            "seed_user_prompt": None if is_baseline else combo["seed_user_prompt"],
            "chat_memory_mode": "myth_only",
            "seed_reinject": (not is_baseline),
        }

        quiet = os.environ.get("TRUST_BATCH_QUIET", "").lower() in {"1", "true", "yes"}
        if quiet:
            with open(log_path, "a", encoding="utf-8") as log_stream:
                with contextlib.redirect_stdout(log_stream):
                    sim_data = run_simulation(**kwargs)
        else:
            sim_data = run_simulation(**kwargs)

        sim_data.run_metadata["phase3_seed_type"] = combo["seed_type"]
        sim_data.run_metadata["phase3_seed_meta"] = combo["seed_meta"]
        sim_data.run_metadata["phase3_noise_game_params"] = combo["noise_game_params_name"]
        sim_data.run_metadata["phase3_rep"] = combo["rep"]
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
        help="Named game_params block (e.g., phase3_8agent_anon_neg5_myth_only)",
    )
    parser.add_argument(
        "--model",
        default="claude_sonnet_45",
        help="Key into config base_models (e.g., claude_haiku_45 or claude_sonnet_45)",
    )
    parser.add_argument("--output-subdir", default="phase3_seeded")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-types", nargs="*", default=SEED_TYPES)
    parser.add_argument("--reps", type=int, default=3, help="Upper bound; runs reps [start_rep, reps)")
    parser.add_argument("--start-rep", type=int, default=0, help="First rep to run (lets you extend an existing pilot without re-running its reps)")
    parser.add_argument("--rep-list", type=int, nargs="*", default=None, help="Explicit list of rep numbers to run (overrides --start-rep/--reps). Useful for retrying scattered failures.")
    parser.add_argument("--no-seed", action="store_true", help="Run unseeded baselines under the same Phase 3 regime (myth_only chat memory, history_policy=none)")
    parser.add_argument(
        "--task-orders",
        nargs="*",
        default=["game"],
        choices=list(TASK_ORDER_LOOKUP.keys()),
        help="One or more of: game, myth_game, game_myth (default: game)",
    )
    args = parser.parse_args()

    config = _load_config()
    with open(args.manifest) as f:
        manifest = json.load(f)

    task_orders = [TASK_ORDER_LOOKUP[t] for t in args.task_orders]
    rep_iter = args.rep_list if args.rep_list is not None else list(range(args.start_rep, args.reps))

    combos = []
    if args.no_seed:
        # Synthetic "baseline" pool: one entry with empty text. The runner
        # treats seed_type=="baseline" specially in run_one by passing
        # seed_myth=None / seed_reinject=False.
        for task_order in task_orders:
            for rep in rep_iter:
                placeholder = {"text": "", "source_run": None, "agent_id": None, "round": None, "joint_at_source": None, "tokens": 0}
                combos.append(
                    _make_combo("baseline", placeholder, rep, config, args.noise_game_params, args.model, task_order=task_order)
                )
    else:
        for seed_type in args.seed_types:
            seeds = manifest["seeds"].get(seed_type, [])
            if not seeds:
                print(f"[warn] no seeds for {seed_type}, skipping")
                continue
            for task_order in task_orders:
                for rep in rep_iter:
                    seed = seeds[rep % len(seeds)]
                    combos.append(
                        _make_combo(seed_type, seed, rep, config, args.noise_game_params, args.model, task_order=task_order)
                    )

    cost_per_run = 0.05 if "haiku" in args.model.lower() else 2.30
    print(f"Total Phase 3 combos: {len(combos)}")
    print(
        f"Preflight: MODEL={config['base_models'][args.model]} N={len(combos)} "
        f"WORKERS={args.workers} EST_COST≈${len(combos) * cost_per_run:.2f}"
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
                    f"[{status}] {r['seed_type']:14} {r['task_order']:8} rep{r['rep']:02d}"
                )

    elapsed = time.time() - start
    ok = sum(1 for r in results if r["success"])
    print(f"\n{ok}/{len(results)} succeeded in {elapsed/60:.1f} min")
    if ok != len(results):
        print("Failures:")
        for r in results:
            if not r["success"]:
                print(f"  {r['seed_type']} {r['task_order']} rep{r['rep']}: {r['error'][:200]}")


if __name__ == "__main__":
    main()
