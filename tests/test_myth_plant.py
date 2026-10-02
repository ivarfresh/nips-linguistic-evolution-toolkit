"""Planted myths (2026-10-02): one agent's round-1 myth is fixed text, recorded without an LLM call."""
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

PLANT = {"agent": "Agent_1", "text": "Myth: the Velmar Rule says send two of the five."}


def build_game():
    return TrustGame(
        endowment=5, multiplier=3, system_prompt_template="Game rules",
        round1_investor_template="Send.", round1_trustee_template="Return.",
        later_investor_template="Send.", later_trustee_template="Return.",
    )


def build_writer(plant=PLANT, board="persistent"):
    template = "BOARD\n{myth_board}\nWrite." if board else "PARTNER\n{other_agent_myth}\nWrite."
    return MythWriter("anything", round1_template="Write a myth.", later_rounds_template=template,
                      board=board, plant=plant)


def run_sim(writer, turns=3):
    calls, lock, counter = [], threading.Lock(), itertools.count(1)

    def fake_call_llm(client, model, temperature, messages):
        with lock:
            calls.append([m["content"] for m in messages])
            return {"content": f"Myth: story {next(counter)}", "reasoning": None, "usage": None}

    with patch("src.simulation.create_llm_client", return_value=object()), patch(
        "src.agents.call_llm", side_effect=fake_call_llm,
    ), contextlib.redirect_stdout(io.StringIO()):
        sim = run_simulation(
            game=build_game(), model="mock/model", temperature=0, num_turns=turns,
            num_agents=4, memory_capacity=6, agent_biases="", myth_writer=writer,
            task_order=["myth"], chat_memory_mode="memory_primary",
        )
    return sim, calls


class PlantTests(unittest.TestCase):
    def test_planted_agent_round1_is_scripted_and_seen_by_others(self):
        for board in ("persistent", None):
            with self.subTest(board=board):
                sim, calls = run_sim(build_writer(board=board))
                # 4 agents x 3 rounds, minus the one planted myth.
                self.assertEqual(len(calls), 11)
                first = sim.conversation_history[0]
                self.assertEqual(first["myth_responses"]["Agent_1"]["response_source"], "planted")
                self.assertEqual(first["myth_responses"]["Agent_1"]["content"], PLANT["text"])
                self.assertIn("Velmar", first["myths"]["Agent_1"])
                # Agent_1 remembers the planted myth as its own; later rounds call the model.
                own = [m["content"] for m in sim.agents["Agent_1"].messages if m["role"] == "assistant"]
                self.assertIn(PLANT["text"], own)
                sources = [e["response"]["response_source"] for e in sim.agents["Agent_1"].interaction_history
                           if e["metadata"].get("task") == "myth"]
                self.assertEqual(sources, ["planted", "llm", "llm"])
                # Someone other than Agent_1 reads the planted text in round 2.
                round2_readers = [c for c in calls if any("Velmar" in m for m in c[-1:])]
                self.assertTrue(round2_readers)

    def test_invalid_plant_is_rejected(self):
        for bad in ({"agent": "Agent_1"}, {"agent": "Agent_1", "text": "  "}, "text"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                build_writer(plant=bad)

    def test_condition_records_plant_only_when_configured(self):
        game = build_game()
        self.assertEqual(build_condition(game, build_writer(), {}, {})["protocol"]["myth"]["plant"], PLANT)
        self.assertNotIn("plant", build_condition(game, build_writer(plant=None), {}, {})["protocol"]["myth"])


if __name__ == "__main__":
    unittest.main()
