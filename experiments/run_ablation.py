"""
Memory-transplant ablation runner (Phase 1).

Dispatches cells from the design doc §7 sweep:

    M1 cells (myth-only, memory wiped each round):
        m1_s_start, m1_s_end_plus, m1_s_end_minus, m1_s_filler

    M2 cells (seeded-start, normal memory):
        m2_s_start, m2_s_end_plus, m2_s_end_minus, m2_s_filler, m2_s_none

Each cell × 5 reps -> 45 runs total (M1 has no s_none — degenerate).

Usage:
    # List what would run for one cell without invoking the LLM
    python experiments/run_ablation.py --cells m1_s_end_plus --dry-run

    # Run the pilot variance cell first (m2_s_none)
    python experiments/run_ablation.py --cells m2_s_none --reps 5

    # Run all 9 cells
    python experiments/run_ablation.py --cells all --reps 5 --workers 4

See docs/memory_transplant_ablation_design.md for the full design.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.batch_utils import unique_json_path as _unique_json_path  # noqa: E402
from src.myth_writer import MythWriter  # noqa: E402
from src.simulation import run_simulation  # noqa: E402
from games.trust_game_noisy import TrustGameNoisy  # noqa: E402


# ---------------------------------------------------------------------------
# Constants — pinned from design doc §7
# ---------------------------------------------------------------------------

MODEL_ID = "anthropic/claude-sonnet-4.5"
GAME_PARAMS_NAME = "noisy_negative_5"
NOISE_CONFIG_NAME = "noisy_negative_5"
DEFAULT_REPS = 5
NUM_AGENTS = 2
NUM_TURNS = 10
TEMPERATURE = 0.8
MEMORY_CAPACITY = 3
TASK_ORDER = ["game"]  # host runs are game-only per design doc §7

SEED_MANIFEST_PATH = PROJECT_ROOT / "data/seeds/ablation_phase1_manifest.json"
FILLERS_PATH = PROJECT_ROOT / "data/seeds/fillers.json"
NOISY_CONFIG_PATH = PROJECT_ROOT / "config/experiments_noisy.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "data/json/ablation_phase1"

# Cost estimate: 45 runs * ~20 LLM calls/run (10 rounds x 2 agents) at
# Sonnet 4.5 (~$0.003 in + $0.015 out per 1k). Per design doc §7: <$10 host.
COST_PER_RUN_USD = 0.20  # conservative upper bound

# Mapping cell -> seed pool key in manifest (or special tokens).
CELL_TO_POOL = {
    "m1_s_start":    "s_start",
    "m1_s_end_plus": "s_end_plus",
    "m1_s_end_minus":"s_end_minus",
    "m1_s_filler":   "__filler__",
    "m1_s_none":     "__none__",  # isolation cell — memory-wipe alone, no seed text
    "m2_s_start":    "s_start",
    "m2_s_end_plus": "s_end_plus",
    "m2_s_end_minus":"s_end_minus",
    "m2_s_filler":   "__filler__",
    "m2_s_none":     "__none__",
}

ALL_CELLS = list(CELL_TO_POOL.keys())


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_yaml_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_seed_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Seed manifest not found at {path}. "
            "Run `python scripts/harvest_seeds.py` first."
        )
    with open(path, "r") as f:
        return json.load(f)


def load_fillers(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    return data["fillers"]


def resolve_seed_for_cell(
    cell: str,
    rep_index: int,
    manifest: Dict[str, Any],
    fillers: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return a dict with keys {seed_text, seed_id, seed_meta} or None for s_none."""
    pool_key = CELL_TO_POOL[cell]
    if pool_key == "__none__":
        return None
    if pool_key == "__filler__":
        if rep_index >= len(fillers):
            raise IndexError(
                f"rep_index {rep_index} out of range for fillers (len={len(fillers)})"
            )
        f = fillers[rep_index]
        return {
            "seed_text": f["text"],
            "seed_id": f["filler_id"],
            "seed_meta": {
                "seed_type": "s_filler",
                "topic": f.get("topic"),
                "filler_id": f["filler_id"],
            },
        }
    pool = manifest["pools"].get(pool_key, [])
    if rep_index >= len(pool):
        raise IndexError(
            f"rep_index {rep_index} out of range for pool '{pool_key}' (len={len(pool)})"
        )
    s = pool[rep_index]
    return {
        "seed_text": s["myth_text"],
        "seed_id": s["seed_id"],
        "seed_meta": {
            "seed_type": s["seed_type"],
            "source_run_path": s["source_run_path"],
            "source_joint_balance": s["source_joint_balance"],
            "token_count": s["token_count"],
        },
    }


# ---------------------------------------------------------------------------
# Templates & game params from config
# ---------------------------------------------------------------------------

def resolve_run_components(noisy_config: Dict[str, Any]) -> Dict[str, Any]:
    """Pull templates, model id, and game params from the noisy config."""
    prompt_templates = noisy_config["prompt_templates"]
    game_params = noisy_config["game_params"][GAME_PARAMS_NAME]
    model_id = noisy_config["base_models"]["claude_sonnet_45"]
    # Sanity: must match the pinned constant.
    if model_id != MODEL_ID:
        raise ValueError(
            f"Model id mismatch: config has {model_id}, expected {MODEL_ID}"
        )

    # Personas: neutral (the source runs used neutral).
    persona = noisy_config["personas"]["neutral"]

    return {
        "prompt_templates": prompt_templates,
        "game_params": game_params,
        "model_id": model_id,
        "persona": persona,
    }


def build_seed_user_prompt(prompt_templates: Dict[str, Any]) -> str:
    """`myth_writing_default` formatted with myth_topic='anything'.

    This is the controlled fixture per design doc §6 — uniform conversational
    framing across all seed types.
    """
    template = prompt_templates["myth_writing_default"]
    return template.format(myth_topic="anything")


# ---------------------------------------------------------------------------
# Single-run executor (process-safe; standalone for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def run_single_cell_rep(
    cell: str,
    rep_index: int,
    seed_record: Optional[Dict[str, Any]],
    components: Dict[str, Any],
    seed_user_prompt: str,
    output_root: str,
) -> Dict[str, Any]:
    """Run one (cell, rep) pair and save to disk. Returns a result summary dict."""
    try:
        templates = components["prompt_templates"]
        game_params = components["game_params"]
        model_id = components["model_id"]
        persona = components["persona"]

        memory_mode = "m1" if cell.startswith("m1_") else "normal"
        seed_myth = seed_record["seed_text"] if seed_record else None
        seed_id = seed_record["seed_id"] if seed_record else "none"

        # Build the noisy game with both agents getting the neutral persona.
        personas = {"Agent_1": persona, "Agent_2": persona}
        game = TrustGameNoisy(
            endowment=game_params["endowment"],
            multiplier=game_params["multiplier"],
            system_prompt_template=templates["trust_game_default"],
            personas=personas,
            round1_investor_template=templates["trust_game_round1_investor"],
            round1_trustee_template=templates["trust_game_round1_trustee"],
            later_investor_template=templates["trust_game_later_investor"],
            later_trustee_template=templates["trust_game_later_trustee"],
            noise_config=game_params.get("noise_config"),
            other_player_names=game_params.get("other_player_names", "default"),
            myth_injection_mode="partner",
            shuffled_myth_pool_path=None,
            run_seed=rep_index,
        )

        # Myth writer is constructed but only used when "myth" appears in
        # task_order; we keep it here so run_simulation has a valid object.
        myth_writer = MythWriter(
            myth_topic="",
            round1_template=templates["myth_writing_default"],
            later_rounds_template=templates["myth_writing_later_rounds"],
        )

        # Output paths
        out_dir = Path(output_root) / cell
        out_dir.mkdir(parents=True, exist_ok=True)
        base = f"ablation_{cell}_rep{rep_index:02d}_{seed_id}"
        save_path = _unique_json_path(str(out_dir / f"{base}.json"))
        base_no_ext = save_path[:-5] if save_path.endswith(".json") else save_path
        results_path = base_no_ext + ".results.json"
        checkpoint_path = base_no_ext + ".checkpoint.json"
        log_path = base_no_ext + ".log"
        resume_from = checkpoint_path if os.path.exists(checkpoint_path) else None

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("MEMORY-TRANSPLANT ABLATION RUN\n")
            f.write("=" * 80 + "\n")
            f.write(f"Cell: {cell}\n")
            f.write(f"Rep index: {rep_index}\n")
            f.write(f"Memory mode: {memory_mode}\n")
            f.write(f"Seed id: {seed_id}\n")
            f.write(f"Model: {model_id}\n")
            f.write(f"Game params: {GAME_PARAMS_NAME}\n")
            if seed_record:
                f.write(f"Seed meta: {seed_record['seed_meta']}\n")
            f.write("=" * 80 + "\n\n")

        sim_data = run_simulation(
            game=game,
            model=model_id,
            temperature=TEMPERATURE,
            num_turns=NUM_TURNS,
            num_agents=NUM_AGENTS,
            memory_capacity=MEMORY_CAPACITY,
            agent_biases="",
            myth_writer=myth_writer,
            task_order=TASK_ORDER,
            results_path=results_path,
            checkpoint_path=checkpoint_path,
            checkpoint_every=10,
            resume_from=resume_from,
            log_file=log_path,
            memory_mode=memory_mode,
            seed_myth=seed_myth,
            seed_user_prompt=seed_user_prompt if seed_myth is not None else None,
        )

        # Tag run metadata so analysis can group cleanly.
        sim_data.run_metadata.update(
            {
                "ablation_cell": cell,
                "ablation_rep": rep_index,
                "ablation_seed_id": seed_id,
                "ablation_seed_meta": seed_record["seed_meta"] if seed_record else None,
                "game_params_name": GAME_PARAMS_NAME,
                "noise_config": game_params.get("noise_config"),
                "other_player_names": game_params.get("other_player_names", "default"),
                "system_prompt_template": "trust_game_default",
                "round_prompt_templates": {
                    "round1_investor": "trust_game_round1_investor",
                    "round1_trustee": "trust_game_round1_trustee",
                    "later_investor": "trust_game_later_investor",
                    "later_trustee": "trust_game_later_trustee",
                },
            }
        )

        sim_data.save_state(save_path)
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
            except OSError:
                pass

        return {
            "success": True,
            "cell": cell,
            "rep": rep_index,
            "seed_id": seed_id,
            "save_path": save_path,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "cell": cell,
            "rep": rep_index,
            "seed_id": (seed_record or {}).get("seed_id"),
            "save_path": None,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


# ---------------------------------------------------------------------------
# Plan + dispatcher
# ---------------------------------------------------------------------------

def expand_cells(cells_arg: List[str]) -> List[str]:
    if "all" in cells_arg:
        return list(ALL_CELLS)
    unknown = [c for c in cells_arg if c not in CELL_TO_POOL]
    if unknown:
        raise ValueError(
            f"Unknown cell(s): {unknown}. Known: {ALL_CELLS + ['all']}"
        )
    return cells_arg


def build_plan(
    cells: List[str],
    reps: int,
    manifest: Dict[str, Any],
    fillers: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for cell in cells:
        for rep_index in range(reps):
            seed_record = resolve_seed_for_cell(cell, rep_index, manifest, fillers)
            plan.append({"cell": cell, "rep": rep_index, "seed_record": seed_record})
    return plan


def print_preflight(plan: List[Dict[str, Any]], cells: List[str], workers: int) -> None:
    est_cost = len(plan) * COST_PER_RUN_USD
    print("=" * 70)
    print("ABLATION PREFLIGHT")
    print("=" * 70)
    print(f"MODEL={MODEL_ID}")
    print(f"CELLS={','.join(cells)}")
    print(f"RUNS={len(plan)}")
    print(f"WORKERS={workers}")
    print(f"EST_COST=${est_cost:.2f} (at ${COST_PER_RUN_USD:.2f}/run upper bound)")
    print(f"SEED_TYPES={sorted({CELL_TO_POOL[c] for c in cells})}")
    print(f"OUTPUT_ROOT={OUTPUT_ROOT.relative_to(PROJECT_ROOT)}")
    print("=" * 70)


def print_plan_table(plan: List[Dict[str, Any]]) -> None:
    print()
    print(f"{'cell':<16} {'rep':>3} {'seed_id':<25} {'memory_mode':<12}")
    print("-" * 60)
    for item in plan:
        cell = item["cell"]
        rep = item["rep"]
        seed_id = (item["seed_record"] or {}).get("seed_id", "(none)")
        mode = "m1" if cell.startswith("m1_") else "normal"
        print(f"{cell:<16} {rep:>3} {seed_id:<25} {mode:<12}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", required=True,
                        help="Comma-separated cell names, or 'all'.")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS,
                        help=f"Reps per cell (default: {DEFAULT_REPS}).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default: 1).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve plan + preflight; do not call the LLM.")
    args = parser.parse_args()

    cells = expand_cells([c.strip() for c in args.cells.split(",") if c.strip()])

    noisy_config = load_yaml_config(NOISY_CONFIG_PATH)
    components = resolve_run_components(noisy_config)
    seed_user_prompt = build_seed_user_prompt(components["prompt_templates"])

    manifest = load_seed_manifest(SEED_MANIFEST_PATH)
    fillers = load_fillers(FILLERS_PATH)

    plan = build_plan(cells, args.reps, manifest, fillers)

    print_preflight(plan, cells, args.workers)
    print_plan_table(plan)
    print(f"Fake user prompt (seed elicitation): {seed_user_prompt!r}")
    print()

    if args.dry_run:
        print("Dry run: skipping all LLM calls.")
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    successes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    if args.workers <= 1:
        for item in plan:
            print(f"\n--- {item['cell']} rep {item['rep']} "
                  f"(seed={(item['seed_record'] or {}).get('seed_id', 'none')}) ---")
            result = run_single_cell_rep(
                item["cell"], item["rep"], item["seed_record"],
                components, seed_user_prompt, str(OUTPUT_ROOT),
            )
            (successes if result["success"] else failures).append(result)
            if result["success"]:
                print(f"OK -> {result['save_path']}")
            else:
                print(f"FAIL: {result['error'].splitlines()[0]}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futs = {
                executor.submit(
                    run_single_cell_rep,
                    item["cell"], item["rep"], item["seed_record"],
                    components, seed_user_prompt, str(OUTPUT_ROOT),
                ): item
                for item in plan
            }
            done = 0
            for fut in as_completed(futs):
                done += 1
                item = futs[fut]
                try:
                    result = fut.result()
                except Exception as e:  # worker crash
                    result = {
                        "success": False,
                        "cell": item["cell"],
                        "rep": item["rep"],
                        "seed_id": (item["seed_record"] or {}).get("seed_id"),
                        "save_path": None,
                        "error": f"worker exception: {e}",
                    }
                (successes if result["success"] else failures).append(result)
                tag = "OK" if result["success"] else "FAIL"
                print(f"[{done}/{len(plan)}] {tag} {result['cell']} "
                      f"rep {result['rep']} seed={result.get('seed_id')}")

    print()
    print("=" * 70)
    print(f"Batch complete: {len(successes)}/{len(plan)} succeeded, "
          f"{len(failures)} failed")
    print("=" * 70)
    if failures:
        print("Failures:")
        for f in failures:
            head = (f["error"] or "").splitlines()[0] if f["error"] else "(no error msg)"
            print(f"  - {f['cell']} rep {f['rep']} seed={f.get('seed_id')}: {head}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
