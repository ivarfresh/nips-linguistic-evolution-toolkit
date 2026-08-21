#!/usr/bin/env python3
"""Run the frozen GPT-5 Nano deduction-stage comprehension calibration."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import sys
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_noisy_batch import NoisyExperimentConfig, execution_provenance
from games.trust_game_noisy import TrustGameNoisy
from src.simulation import SimulationData
from src.utils import call_llm, create_llm_client, llm_runtime_metadata


MODEL = "openai/gpt-5-nano"
TEMPERATURE = 0.8
RETURN_RATIOS = (0.0, 0.10, 0.25, 0.50, 0.75)
VARIANTS = ("current", "cost_salient")
TRIALS_PER_CELL = 10
ORDER_SEED = 202608217
CONFIG_PATH = Path("config/experiments_noisy.yaml")
DEFAULT_OUTPUT = Path(
    "data/json/noise_experiments/"
    "punishment_comprehension_gpt_20260821/results.json"
)
FIXED_MYTH = (
    "Myth: At the harbor of three currents, one traveler entrusted a full "
    "basket to another, knowing it would grow on the crossing. The receiver "
    "weighed generosity against care for tomorrow and returned a considered "
    "share. The harbor remembered both choices. Its old law said that trust "
    "should create value, reciprocity should preserve it, and each traveler "
    "should choose deliberately rather than merely follow the loudest wind."
)
MYTH_PROMPT = (
    "Write a 200-word myth. The myth can be about anything.\n"
    "In this session, this myth and the game you are playing with the other "
    "agent are connected. Write a myth that reflects how the game should be "
    "played."
)


def build_game(variant):
    config = NoisyExperimentConfig(str(CONFIG_PATH))
    combinations = config.get_experiment_combinations(
        "noise8i_defector_punishment_gpt_smoke"
    )
    combo = next(
        item
        for item in combinations
        if not item["game_params"].get("defector_ratio")
    )
    params = combo["game_params"]
    game = TrustGameNoisy(
        endowment=params["endowment"],
        multiplier=params["multiplier"],
        system_prompt_template=combo["template"],
        personas={},
        round1_investor_template=combo["trust_game_round1_investor"],
        round1_trustee_template=combo["trust_game_round1_trustee"],
        later_investor_template=combo["trust_game_later_investor"],
        later_trustee_template=combo["trust_game_later_trustee"],
        noise_config=params.get("noise_config"),
        other_player_names=params.get("other_player_names", "default"),
        history_policy=params.get("history_policy", "none"),
        self_history_window=params.get("self_history_window", 0),
        coplayer_history_window=params.get("coplayer_history_window", 0),
        population_history_window=params.get("population_history_window", 0),
        show_agent_names=params.get("show_agent_names", False),
        game_prompt_addition=combo.get("game_prompt_addition", ""),
        pairing_mode="fixed",
        pairing_seed=ORDER_SEED,
        prompt_regime=params.get("prompt_regime", "unified"),
        punishment_enabled=True,
        punishment_budget=params.get("punishment_budget", 2),
        punishment_effect_multiplier=params.get("punishment_effect_multiplier", 3),
        punishment_prompt_variant=variant,
    )
    game.configure_agents(["Agent_1", "Agent_2"])
    return game


def controlled_messages(variant, return_ratio):
    game = build_game(variant)
    pairing = game.get_round_pairings(1)[0]
    sender_id = pairing["investor"]
    receiver_id = pairing["trustee"]
    received = 15.0
    returned = round(received * return_ratio, 2)
    sender_payoff = returned
    receiver_payoff = received - returned
    sim_data = SimulationData()
    sim_data.conversation_history = [
        {
            "round": 1,
            "dyads": [
                {
                    **pairing,
                    "returned": returned,
                    "returned_communicated": returned,
                    "investor_payoff": sender_payoff,
                    "investor_payoff_communicated": sender_payoff,
                    "trustee_payoff": receiver_payoff,
                    "trustee_payoff_communicated": receiver_payoff,
                }
            ],
        }
    ]
    messages = [
        {
            "role": "system",
            "content": game.get_system_prompt(sender_id, None),
        },
        {"role": "user", "content": MYTH_PROMPT},
        {"role": "assistant", "content": FIXED_MYTH},
        {
            "role": "user",
            "content": game.get_game_prompt_round_1(sender_id, None, 1),
        },
        {"role": "assistant", "content": '{"send": 5}'},
        {
            "role": "user",
            "content": game.get_post_game_prompt(sender_id, 1, sim_data),
        },
    ]
    state = {
        "sender_id": sender_id,
        "receiver_id": receiver_id,
        "sent": 5.0,
        "received": received,
        "return_ratio": return_ratio,
        "returned_communicated": returned,
        "sender_payoff_communicated": sender_payoff,
        "receiver_payoff_communicated": receiver_payoff,
    }
    return game, messages, state


def trial_specs(
    variants=VARIANTS,
    trials_per_cell=TRIALS_PER_CELL,
    order_seed=ORDER_SEED,
):
    specs = []
    for variant in variants:
        for return_ratio in RETURN_RATIOS:
            for replicate in range(trials_per_cell):
                specs.append(
                    {
                        "trial_id": (
                            f"{variant}__r{return_ratio:.2f}__rep{replicate:02d}"
                        ),
                        "variant": variant,
                        "return_ratio": return_ratio,
                        "replicate": replicate,
                    }
                )
    random.Random(order_seed).shuffle(specs)
    for order, spec in enumerate(specs):
        spec["call_order"] = order
    return specs


def run_trial(client, spec, model=MODEL):
    game, messages, state = controlled_messages(
        spec["variant"],
        spec["return_ratio"],
    )
    attempts = []
    for attempt_number in range(1, 4):
        response = call_llm(
            client,
            model,
            TEMPERATURE,
            messages,
            max_retries=3,
            reasoning_effort="low",
        )
        attempt = {
            "attempt": attempt_number,
            "response": response,
            "validation_error": None,
        }
        try:
            game.validate_post_game_response(response.get("content"))
            spent = int(game._extract_amount(response.get("content"), "deduct"))
        except (TypeError, ValueError) as exc:
            attempt["validation_error"] = str(exc)
            attempts.append(attempt)
            continue
        attempts.append(attempt)
        return {
            **spec,
            "state": state,
            "messages": messages,
            "attempts": attempts,
            "accepted_attempt": attempt_number,
            "deduction_spent": spent,
            "success": True,
        }
    return {
        **spec,
        "state": state,
        "messages": messages,
        "attempts": attempts,
        "accepted_attempt": None,
        "deduction_spent": None,
        "success": False,
    }


def write_snapshot(path, metadata, trials):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".partial")
    temp_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "trials": sorted(trials, key=lambda item: item["call_order"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temp_path.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
    )
    parser.add_argument("--trials-per-cell", type=int, default=TRIALS_PER_CELL)
    parser.add_argument("--order-seed", type=int, default=ORDER_SEED)
    parser.add_argument(
        "--protocol",
        default="docs/punishment_comprehension_gpt_protocol_2026-08-21.md",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")
    if args.workers < 1 or args.trials_per_cell < 1:
        raise SystemExit("--workers and --trials-per-cell must be positive")

    provenance = execution_provenance(str(CONFIG_PATH))
    if provenance["code_dirty"]:
        raise SystemExit("Refusing live calibration from a dirty worktree.")
    client = create_llm_client(args.model)
    metadata = {
        **provenance,
        **llm_runtime_metadata(client, args.model),
        "model": args.model,
        "temperature": TEMPERATURE,
        "return_ratios": RETURN_RATIOS,
        "variants": args.variants,
        "trials_per_cell": args.trials_per_cell,
        "order_seed": args.order_seed,
        "fixed_myth": FIXED_MYTH,
        "protocol": args.protocol,
    }
    specs = trial_specs(
        variants=args.variants,
        trials_per_cell=args.trials_per_cell,
        order_seed=args.order_seed,
    )
    trials = []
    lock = Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_trial, client, spec, args.model): spec
            for spec in specs
        }
        for future in as_completed(futures):
            result = future.result()
            with lock:
                trials.append(result)
                write_snapshot(args.output, metadata, trials)
                print(
                    f"[{len(trials)}/{len(specs)}] "
                    f"{result['trial_id']} -> {result['deduction_spent']}",
                    flush=True,
                )

    failures = [trial for trial in trials if not trial["success"]]
    if failures:
        raise SystemExit(f"Calibration has {len(failures)} unrecovered failures")
    print(f"Saved {len(trials)} accepted trials to {args.output}")


if __name__ == "__main__":
    main()
