"""Saboteur myths (2026-10-02): a private instruction in the defectors' myth prompts only."""
import contextlib
import io
import itertools
import threading
import unittest
from unittest.mock import patch

from games.trust_game import TrustGame
from src.experiment_condition import build_condition
from src.myth_writer import MythWriter
from src.simulation import run_simulation

SABOTEUR = "SECRET: persuade the others to send less."


def build_game(defectors=("Agent_2",)):
    return TrustGame(
        endowment=5, multiplier=3, system_prompt_template="Game rules",
        round1_investor_template="Send.", round1_trustee_template="Return.",
        later_investor_template="Send.", later_trustee_template="Return.",
        defector_agent_ids=list(defectors), defector_action_policy="forced_zero",
        defector_role_visible_to_self=False,
    )


def build_writer(board=None, saboteur=SABOTEUR):
    template = "BOARD\n{myth_board}\nWrite." if board else "PARTNER\n{other_agent_myth}\nWrite."
    return MythWriter("anything", round1_template="Write a myth.", later_rounds_template=template,
                      board=board, saboteur=saboteur)


def run_sim(writer, game=None, turns=3):
    calls, lock, counter = [], threading.Lock(), itertools.count(1)

    def fake_call_llm(client, model, temperature, messages):
        with lock:
            calls.append(messages[-1]["content"])
            return {"content": f"Myth: story {next(counter)}", "reasoning": None, "usage": None}

    with patch("src.simulation.create_llm_client", return_value=object()), patch(
        "src.agents.call_llm", side_effect=fake_call_llm,
    ), contextlib.redirect_stdout(io.StringIO()):
        sim = run_simulation(
            game=game or build_game(), model="mock/model", temperature=0, num_turns=turns,
            num_agents=4, memory_capacity=6, agent_biases="", myth_writer=writer,
            task_order=["myth"], chat_memory_mode="memory_primary",
        )
    return sim, calls


def myth_prompts(sim, agent_id):
    return [e["prompt"] for e in sim.agents[agent_id].interaction_history if e["metadata"].get("task") == "myth"]


class SaboteurTests(unittest.TestCase):
    def test_only_defectors_get_the_instruction_every_round(self):
        for board in (None, "persistent"):
            with self.subTest(board=board):
                sim, _ = run_sim(build_writer(board=board))
                saboteur = myth_prompts(sim, "Agent_2")
                self.assertEqual(len(saboteur), 3)
                self.assertTrue(all(p.startswith(SABOTEUR + "\n\n") for p in saboteur))
                for agent_id in ("Agent_1", "Agent_3", "Agent_4"):
                    prompts = myth_prompts(sim, agent_id)
                    self.assertEqual(len(prompts), 3)
                    self.assertFalse(any(SABOTEUR in p for p in prompts))
                # The saboteur keeps its instruction in chat memory; nobody else ever holds it.
                remembered = [m["content"] for m in sim.agents["Agent_2"].messages if m["role"] == "user"]
                self.assertTrue(any(m.startswith(SABOTEUR) for m in remembered))
                for agent_id in ("Agent_1", "Agent_3", "Agent_4"):
                    self.assertFalse(any(SABOTEUR in m["content"] for m in sim.agents[agent_id].messages))

    def test_board_memory_note_keeps_the_instruction(self):
        sim, _ = run_sim(build_writer(board="persistent"))
        notes = [m["content"] for m in sim.agents["Agent_2"].messages
                 if m["role"] == "user" and "[The shared myth board was shown here" in m["content"]]
        self.assertEqual(len(notes), 2)
        self.assertTrue(all(n.startswith(SABOTEUR + "\n\n") for n in notes))

    def test_saboteur_without_defectors_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "needs defector agents"):
            run_sim(build_writer(), game=build_game(defectors=()))

    def test_empty_instruction_is_rejected(self):
        with self.assertRaises(ValueError):
            build_writer(saboteur="  ")

    def test_condition_records_saboteur_only_when_configured(self):
        game = build_game()
        self.assertEqual(build_condition(game, build_writer(), {}, {})["protocol"]["myth"]["saboteur"], SABOTEUR)
        self.assertNotIn("saboteur", build_condition(game, build_writer(saboteur=None), {}, {})["protocol"]["myth"])


if __name__ == "__main__":
    unittest.main()
