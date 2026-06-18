import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from src.agents import Agent
from src.utils import create_llm_client, print_simulation_header
from concurrent.futures import ThreadPoolExecutor


DEFAULT_AGENT_NAMES = [
    "Aster",
    "Briar",
    "Cyra",
    "Dorian",
    "Elara",
    "Finn",
    "Galen",
    "Hana",
    "Iris",
    "Jules",
    "Kael",
    "Lina",
    "Mira",
    "Niko",
    "Orin",
    "Pia",
]


def _build_agent_names(agent_ids, configured_names=None):
    if configured_names is None:
        names = {}
    elif isinstance(configured_names, dict):
        names = {agent_id: str(name) for agent_id, name in configured_names.items()}
    elif isinstance(configured_names, list):
        names = {
            agent_id: str(configured_names[idx])
            for idx, agent_id in enumerate(agent_ids)
            if idx < len(configured_names)
        }
    else:
        raise ValueError("agent_names must be a list, dict, or omitted.")

    for idx, agent_id in enumerate(agent_ids):
        if agent_id in names:
            continue
        if idx < len(DEFAULT_AGENT_NAMES):
            names[agent_id] = DEFAULT_AGENT_NAMES[idx]
        else:
            names[agent_id] = f"AgentName_{idx + 1}"

    resolved = {agent_id: names[agent_id] for agent_id in agent_ids}
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("Agent display names must be unique.")

    return resolved


def _configure_game_agents(game, agent_ids, agent_names):
    if hasattr(game, "configure_agents"):
        game.configure_agents(agent_ids, agent_names)


def _restore_population_assignment(game, sim_data):
    saved_defectors = sim_data.game_data.get("defector_agent_ids")
    if saved_defectors is not None and hasattr(game, "restore_defector_agent_ids"):
        game.restore_defector_agent_ids(saved_defectors)


def _sync_population_metadata(game, sim_data):
    if not hasattr(game, "get_population_metadata"):
        return {}

    metadata = game.get_population_metadata()
    sim_data.game_data.update(metadata)
    agent_types = metadata.get("agent_types", {})
    for agent_id, agent in sim_data.agents.items():
        agent.population_role = agent_types.get(agent_id, "standard")
    return metadata


def _get_round_pairings(game, turn, sim_data):
    if hasattr(game, "get_round_pairings"):
        return game.get_round_pairings(turn, sim_data)

    if hasattr(game, "get_roles_for_round"):
        try:
            roles = game.get_roles_for_round(turn, sim_data)
        except TypeError:
            roles = game.get_roles_for_round(turn)
        if isinstance(roles, dict) and "investor" in roles and "trustee" in roles:
            investor_id = roles["investor"]
            trustee_id = roles["trustee"]
            return [
                {
                    "round": turn,
                    "dyad_id": "dyad_1",
                    "agents": [investor_id, trustee_id],
                    "investor": investor_id,
                    "trustee": trustee_id,
                    "roles": {investor_id: "investor", trustee_id: "trustee"},
                    "agent_names": sim_data.game_data.get("agent_names", {}),
                }
            ]

    return []


def _roles_by_agent(pairings):
    roles = {}
    for pairing in pairings:
        roles.update(pairing.get("roles", {}))
    return roles


def _pairing_for_agent(pairings, agent_id):
    for pairing in pairings:
        if agent_id in (pairing.get("agents") or []):
            return pairing
    return None


def _unique_order(agent_ids):
    seen = set()
    ordered = []
    for agent_id in agent_ids:
        if agent_id in seen:
            continue
        seen.add(agent_id)
        ordered.append(agent_id)
    return ordered


def _role_label(role):
    if role == "investor":
        return "Sender"
    if role == "trustee":
        return "Receiver"
    return "Agent"


def _interaction_metadata(
    turn,
    task,
    agent_id,
    roles_by_agent,
    pairings,
    task_index,
    move_index,
    agent_type=None,
):
    role = roles_by_agent.get(agent_id)
    pairing = _pairing_for_agent(pairings, agent_id)
    metadata = {
        "round": turn,
        "task": task,
        "task_index": task_index,
        "move_index": move_index,
        "role": role,
        "role_label": _role_label(role),
        "agent_type": agent_type or "standard",
    }
    if pairing:
        opponent_id = next(
            (other_id for other_id in pairing.get("agents", []) if other_id != agent_id),
            None,
        )
        metadata.update(
            {
                "dyad_id": pairing.get("dyad_id"),
                "opponent_id": opponent_id,
                "pairing": {
                    "agents": pairing.get("agents"),
                    "investor": pairing.get("investor"),
                    "trustee": pairing.get("trustee"),
                    "roles": pairing.get("roles"),
                    "agent_names": pairing.get("agent_names"),
                },
            }
        )
    return metadata


class SimulationData:
    """Centralized state management for multi-agent conversations"""

    def __init__(self):
        self.agents = {}
        self.conversation_history = []
        self.game_data = {}
        self.task_order = None  # Store task order used in simulation
        # Optional: store basic run metadata for easier debugging/resume
        self.run_metadata = {}

    def add_agent(self, agent_id, agent):
        self.agents[agent_id] = agent

    def get_agent_messages(self, agent_id):
        return self.agents[agent_id].messages

    @staticmethod
    def _atomic_json_write(filepath: str, data: Dict[str, Any], *, indent: Optional[int] = 2) -> None:
        path = Path(filepath)
        if path.parent != Path(""):
            path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp_path, path)

    def to_state(self, include_agent_histories: bool = True):
        state = {
            "conversation_history": self.conversation_history,
            "game_data": self.game_data,
            "task_order": self.task_order, # Include task_order in saved state
            "run_metadata": self.run_metadata,
        }
        if include_agent_histories:
            state["agents"] = {
                agent_id: {
                    "agent_id": agent.agent_id,
                    "display_name": getattr(agent, "display_name", agent.agent_id),
                    "model": agent.model,
                    "temperature": agent.temperature,
                    "memory_capacity": agent.memory_capacity,
                    "initial_bias": agent.initial_bias,
                    "population_role": getattr(agent, "population_role", "standard"),
                    "system_prompt": agent.system_prompt,
                    "messages": agent.messages,
                    "interaction_history": getattr(agent, "interaction_history", []),
                }
                for agent_id, agent in self.agents.items()
            }
        return state

    def save_state(self, filepath):
        self._atomic_json_write(filepath, self.to_state(include_agent_histories=True), indent=2)

    def save_results_only(self, filepath):
        """Lightweight save: results/state only (no agent message histories)."""
        self._atomic_json_write(filepath, self.to_state(include_agent_histories=False), indent=2)

    def save_transcript_pdf(self, filepath, source_path: Optional[str] = None):
        from src.transcript import write_pdf_transcript

        write_pdf_transcript(
            self.to_state(include_agent_histories=True),
            filepath,
            source_path=source_path,
        )

    @classmethod
    def load_state(cls, filepath: str, client, log_file: Optional[str] = None) -> "SimulationData":
        """
        Load simulation state from a JSON file.
        Loads the full state of the simulation, including the message history of each agent.
        """
        with open(filepath, "r") as f:
            state = json.load(f)
        sim_data = cls()
        sim_data.conversation_history = state.get("conversation_history", [])
        sim_data.game_data = state.get("game_data", {})
        sim_data.task_order = state.get("task_order")
        sim_data.run_metadata = state.get("run_metadata", {})

        agents_state = state.get("agents", {}) or {}
        for agent_id, a in agents_state.items():
            agent = Agent(
                a["agent_id"],
                a["model"],
                a.get("temperature", sim_data.run_metadata.get("temperature", 0.8)),
                client,
                memory_capacity=a["memory_capacity"],
                initial_bias=a.get("initial_bias"),
                system_prompt=a.get("system_prompt"),
                log_file=log_file,
            )
            # Preserve message history exactly for faithful resume
            agent.messages = a.get("messages", [])
            agent.interaction_history = a.get("interaction_history", [])
            agent.display_name = a.get("display_name", agent.agent_id)
            agent.population_role = a.get("population_role", "standard")
            sim_data.add_agent(agent_id, agent)

        return sim_data

def run_simulation(
    game,
    model,
    temperature,
    num_turns,
    num_agents,
    memory_capacity,
    agent_biases,
    myth_writer,
    task_order=["game", "myth"],
    *,
    results_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 10,
    resume_from: Optional[str] = None,
    log_file: Optional[str] = None,
    agent_names: Optional[Any] = None,
    seed_myth: Optional[str] = None,
    seed_user_prompt: Optional[str] = None,
):
    """
    Run a multi-agent simulation with any game.
    Now supports sequential moves within a turn.
    Args:
    task_order: List of tasks to execute in order. Options: "game", "myth"
                Examples: ["game"], ["myth"], ["game", "myth"], ["myth", "game"]
    """
    client = create_llm_client(model)
    if resume_from and Path(resume_from).exists():
        sim_data = SimulationData.load_state(resume_from, client, log_file=log_file)
        if task_order is not None:
            sim_data.task_order = task_order
        agent_ids = list(sim_data.agents.keys())
        saved_agent_names = (
            sim_data.run_metadata.get("agent_names")
            or sim_data.game_data.get("agent_names")
            or {
                agent_id: getattr(agent, "display_name", agent_id)
                for agent_id, agent in sim_data.agents.items()
            }
        )
        resolved_agent_names = _build_agent_names(agent_ids, agent_names or saved_agent_names)
        for agent_id, agent in sim_data.agents.items():
            agent.display_name = resolved_agent_names[agent_id]
        sim_data.game_data["agent_names"] = resolved_agent_names
        _configure_game_agents(game, agent_ids, resolved_agent_names)
        _restore_population_assignment(game, sim_data)
    else:
        sim_data = SimulationData()
        sim_data.task_order = task_order  # Store task_order in sim_data
        agent_ids = [f"Agent_{i+1}" for i in range(num_agents)]
        resolved_agent_names = _build_agent_names(agent_ids, agent_names)
        sim_data.game_data["agent_names"] = resolved_agent_names
        _configure_game_agents(game, agent_ids, resolved_agent_names)

        # Initialize agents
        for i, agent_id in enumerate(agent_ids):
            bias = agent_biases[i] if agent_biases and i < len(agent_biases) else None
            agent = Agent(agent_id, model, temperature, client, memory_capacity=memory_capacity, initial_bias=bias, log_file=log_file)
            agent.display_name = resolved_agent_names[agent_id]
            system_prompt = game.get_system_prompt(agent_id, agent)
            agent.system_prompt = system_prompt
            agent.messages.append({"role": "system", "content": system_prompt})
            # Phase 2 memory-transplant: seed lands at messages[1:3] so
            # the agent enters round 1 with a prior "myth-writing" turn in
            # its memory. See docs/memory_transplant_ablation_design.md §17.
            if seed_myth is not None:
                if not seed_user_prompt:
                    raise ValueError(
                        "seed_myth was provided but seed_user_prompt is empty."
                    )
                agent.messages.append({"role": "user", "content": seed_user_prompt})
                agent.messages.append({"role": "assistant", "content": seed_myth})
            sim_data.add_agent(agent_id, agent)

    population_metadata = _sync_population_metadata(game, sim_data)

    # Store run metadata (useful for debugging/resume)
    actual_num_agents = len(sim_data.agents) if sim_data.agents else num_agents
    sim_data.run_metadata.update(
        {
            "model": model,
            "temperature": temperature,
            "num_turns": num_turns,
            "num_agents": actual_num_agents,
            "memory_capacity": memory_capacity,
            "agent_names": sim_data.game_data.get("agent_names", {}),
            "seed_myth": seed_myth,
            "seed_user_prompt": seed_user_prompt,
            **{
                key: value
                for key, value in population_metadata.items()
                if key != "agent_types"
            },
        }
    )

    print_simulation_header(game, num_turns, actual_num_agents, memory_capacity, agent_biases)
    last_responses = {}

    # Main simulation loop
    start_turn = len(sim_data.conversation_history) + 1 if sim_data.conversation_history else 1
    if start_turn > num_turns:
        return sim_data

    for turn in range(start_turn, num_turns + 1):
        print("\n" + "=" * 80)
        print(f"ROUND {turn}")
        print("=" * 80)

        try:
            pairings = _get_round_pairings(game, turn, sim_data)
            roles_by_agent = _roles_by_agent(pairings)
            move_order = game.get_move_order(turn, sim_data)
            active_agent_order = _unique_order(move_order) or list(sim_data.agents.keys())

            if pairings:
                print("Pairings this round:")
                for pairing in pairings:
                    investor_id = pairing["investor"]
                    trustee_id = pairing["trustee"]
                    agent_names = pairing.get("agent_names") or sim_data.game_data.get("agent_names", {})
                    print(
                        f"  {pairing['dyad_id']}: "
                        f"{investor_id} ({agent_names.get(investor_id, investor_id)}) = SENDER, "
                        f"{trustee_id} ({agent_names.get(trustee_id, trustee_id)}) = RECEIVER"
                    )
            print(f"Move order this round: {move_order}")
            # Pre-create complete conversation_history entry for this round with all fields
            round_entry = {
                "round": turn,
                "roles": roles_by_agent,
                "agent_types": sim_data.game_data.get("agent_types", {}),
                "pairings": pairings,
                "dyads": [],
                "sent": None,
                "received": None,
                "returned": None,
                "investor_payoff": None,
                "trustee_payoff": None,
                "payoffs": None,
                "balances": None,
                "actions": None,
                "myths": {},
                "game_responses": {},  # Store game decision responses
                "myth_responses": {}  # Store myth writing responses
            }
            sim_data.conversation_history.append(round_entry)

            agent_responses = {}
            agent_myths = {}

            # Execute tasks in specified order
            for task_index, task in enumerate(task_order):
                if task == "game":
                    # PHASE 1: GAME PLAY
                    print("\n--- PHASE 1: GAME PLAY ---")

                    for move_index, agent_id in enumerate(move_order):
                        agent = sim_data.agents[agent_id]
                        role_name = _role_label(roles_by_agent.get(agent_id))

                        if turn == 1:
                            prompt = game.get_game_prompt_round_1(agent_id, agent, turn)
                        else:
                            prompt = game.get_game_prompt_later_round(agent_id, turn, sim_data, last_responses)

                        response_data = agent.respond(
                            prompt,
                            transcript_metadata=_interaction_metadata(
                                turn,
                                "game",
                                agent_id,
                                roles_by_agent,
                                pairings,
                                task_index,
                                move_index,
                                getattr(agent, "population_role", "standard"),
                            ),
                        )
                        agent_responses[agent_id] = response_data

                        # Store full game response data in round_entry
                        round_entry["game_responses"][agent_id] = {
                            "content": response_data["content"],
                            "reasoning": response_data.get("reasoning"),
                            "usage": response_data.get("usage")
                        }

                        print(f"\n{agent_id} ({role_name}) prompt: {prompt}")
                        print(f"{agent_id} ({role_name}) response: {response_data['content']}")

                        # Allow game to update state after each move (for sequential games)
                        if hasattr(game, 'process_intermediate_response'):
                            game.process_intermediate_response(agent_id, response_data, turn, sim_data)

                    # Process turn with game logic
                    last_responses = game.process_turn(turn, agent_responses, sim_data)

                elif task == "myth":
                    # PHASE 2: MYTH WRITING
                    print("\n--- PHASE 2: MYTH WRITING ---")

                    # PARALLELIZED MYTH WRITING
                    # Prepare prompts for all agents (no dependencies)
                    prompts = {}
                    for agent_id in active_agent_order:
                        if turn == 1:
                            prompts[agent_id] = myth_writer.get_myth_prompt_round_1(agent_id, turn, sim_data)
                        else:
                            prompts[agent_id] = myth_writer.get_myth_prompt_round_later(agent_id, turn, sim_data)

                    # Parallelize LLM calls for myth writing
                    with ThreadPoolExecutor(max_workers=len(active_agent_order)) as executor:
                        myth_metadata = {
                            agent_id: _interaction_metadata(
                                turn,
                                "myth",
                                agent_id,
                                roles_by_agent,
                                pairings,
                                task_index,
                                move_index,
                                getattr(
                                    sim_data.agents[agent_id],
                                    "population_role",
                                    "standard",
                                ),
                            )
                            for move_index, agent_id in enumerate(active_agent_order)
                        }
                        futures = {
                            agent_id: executor.submit(
                                sim_data.agents[agent_id].respond,
                                prompts[agent_id],
                                myth_metadata[agent_id],
                            )
                            for agent_id in active_agent_order
                        }

                        # Collect results as they complete (with one retry on failure)
                        for agent_id, future in futures.items():
                            try:
                                myth_response_data = future.result()
                                agent_myths[agent_id] = myth_response_data
                            except Exception as e:
                                print(f"⚠️  Myth writing failed for {agent_id}: {type(e).__name__}: {e}. Retrying once...")
                                time.sleep(1.0)
                                myth_response_data = sim_data.agents[agent_id].respond(
                                    prompts[agent_id],
                                    myth_metadata[agent_id],
                                )
                                agent_myths[agent_id] = myth_response_data

                            # Store full myth response data in round_entry
                            round_entry["myth_responses"][agent_id] = {
                                "content": myth_response_data["content"],
                                "reasoning": myth_response_data.get("reasoning"),
                                "usage": myth_response_data.get("usage")
                            }

                            current_role = _role_label(roles_by_agent.get(agent_id))
                            print(f"\n{agent_id} ({current_role}) myth prompt:\n{prompts[agent_id]}")
                            print(f"\n{agent_id} ({current_role}) myth response:\n{myth_response_data['content']}")

                    myth_writer.process_myths(turn, agent_myths, sim_data)

            # Print turn summary (only if game was run)
            if "game" in task_order:
                game.print_turn_summary(turn, agent_responses, sim_data)

            # Print myths (only if myth was run)
            if "myth" in task_order and sim_data.conversation_history and sim_data.conversation_history[-1].get("myths"):
                print(f"\n{'~' * 80}")
                print("MYTHS WRITTEN THIS ROUND:")
                print(f"{'~' * 80}")
                for agent_id, myth in sim_data.conversation_history[-1]["myths"].items():
                    current_role = _role_label(roles_by_agent.get(agent_id))
                    print(f"\n{agent_id} ({current_role}):")
                    print(myth)
                    print("-" * 40)

            # Hybrid saving after successful round
            if results_path:
                sim_data.save_results_only(results_path)
            if checkpoint_path and checkpoint_every > 0 and (turn % checkpoint_every == 0):
                sim_data.save_state(checkpoint_path)

        except Exception:
            # Always write a final save on error before stopping
            if results_path:
                sim_data.save_results_only(results_path)
            if checkpoint_path:
                sim_data.save_state(checkpoint_path + ".error.json")
            raise
    
    # Print game summary (only if game was run)
    if "game" in task_order:
        game.print_game_summary(sim_data)
    return sim_data
