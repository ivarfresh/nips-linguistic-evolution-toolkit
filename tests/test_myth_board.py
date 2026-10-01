"""Persistent myth board (2026-10-02): every agent reads every earlier myth."""
import contextlib
import io
import itertools
import threading
import unittest
from unittest.mock import patch

from games.trust_game import TrustGame
from src.experiment_condition import build_condition
from src.myth_writer import MythWriter, validate_myth_board
from src.simulation import run_simulation

BOARD_TEMPLATE = "BOARD\n{myth_board}\nWrite your own myth."
PARTNER_TEMPLATE = "PARTNER\n{other_agent_myth}\nWrite your own myth."


def build_game():
    return TrustGame(
        endowment=5, multiplier=3, system_prompt_template="Game rules",
        round1_investor_template="Send.", round1_trustee_template="Return.",
        later_investor_template="Send.", later_trustee_template="Return.",
    )


def build_writer(board="persistent", template=BOARD_TEMPLATE):
    return MythWriter("anything", round1_template="Write a myth.", later_rounds_template=template, board=board)


def run_sim(writer, turns=3, num_agents=4):
    prompts, lock, counter = [], threading.Lock(), itertools.count(1)

    def fake_call_llm(client, model, temperature, messages):
        with lock:
            prompts.append(messages[-1]["content"])
            return {"content": f"Myth: story {next(counter)}", "reasoning": None, "usage": None}

    with patch("src.simulation.create_llm_client", return_value=object()), patch(
        "src.agents.call_llm", side_effect=fake_call_llm,
    ), contextlib.redirect_stdout(io.StringIO()):
        sim = run_simulation(
            game=build_game(), model="mock/model", temperature=0, num_turns=turns,
            num_agents=num_agents, memory_capacity=6, agent_biases="", myth_writer=writer,
            task_order=["myth"], chat_memory_mode="memory_primary",
        )
    return sim, prompts


class BoardTests(unittest.TestCase):
    def test_board_shows_every_earlier_myth_and_memory_keeps_a_note(self):
        sim, prompts = run_sim(build_writer())
        r1, r2, _ = sim.conversation_history
        boards = [p for p in prompts if p.startswith("BOARD")]
        self.assertEqual(len(boards), 8)  # 4 agents x rounds 2 and 3
        # Round 2: all four round-1 myths, own included, and nothing else.
        for prompt in boards[:4]:
            for myth in r1["myths"].values():
                self.assertIn(myth, prompt)
            self.assertNotIn("Round 2,", prompt)
        # Round 3: all eight myths from rounds 1 and 2.
        for prompt in boards[4:]:
            for myth in list(r1["myths"].values()) + list(r2["myths"].values()):
                self.assertIn(myth, prompt)
        # Every reader sees the same board in the same order.
        self.assertEqual(len(set(boards[:4])), 1)
        self.assertEqual(len(set(boards[4:])), 1)
        # Chat memory holds the note, not the board text.
        for agent_id, agent in sim.agents.items():
            remembered = [m["content"] for m in agent.messages if m["role"] == "user" and m["content"].startswith("BOARD")]
            self.assertEqual(remembered, [
                "BOARD\n[The shared myth board was shown here: 4 myths from rounds 1-1.]\nWrite your own myth.",
                "BOARD\n[The shared myth board was shown here: 8 myths from rounds 1-2.]\nWrite your own myth.",
            ])
            # The interaction audit keeps what the model was actually sent.
            sent = [e["prompt"] for e in agent.interaction_history if e["prompt"].startswith("BOARD")]
            self.assertEqual(len(sent), 2)
            self.assertIn(r1["myths"][agent_id], sent[0])

    def test_exposure_record_lists_board_items(self):
        sim, _ = run_sim(build_writer())
        record = sim.conversation_history[2]["myth_exposures"]["Agent_1"]
        self.assertEqual(record["board"], "persistent")
        self.assertEqual(len(record["board_items"]), 8)
        self.assertEqual(sorted(r for r, _ in record["board_items"]), [1] * 4 + [2] * 4)
        self.assertEqual({a for _, a in record["board_items"]}, set(sim.agents))

    def test_partner_arm_is_unchanged(self):
        sim, prompts = run_sim(build_writer(board=None, template=PARTNER_TEMPLATE))
        later = [p for p in prompts if p.startswith("PARTNER")]
        self.assertEqual(len(later), 8)
        self.assertTrue(all(p.count("Myth: story") == 1 for p in later))
        record = sim.conversation_history[1]["myth_exposures"]["Agent_1"]
        self.assertNotIn("board", record)
        # Without a board the full prompt is remembered, as before.
        self.assertTrue(any(m["content"] in later for m in sim.agents["Agent_1"].messages))

    def test_board_rejected_in_myth_only_memory(self):
        with patch("src.simulation.create_llm_client", return_value=object()), \
                self.assertRaisesRegex(ValueError, "myth_only"), contextlib.redirect_stdout(io.StringIO()):
            run_simulation(game=build_game(), model="mock/model", temperature=0, num_turns=2, num_agents=4,
                           memory_capacity=6, agent_biases="", myth_writer=build_writer(), task_order=["myth"],
                           chat_memory_mode="myth_only")

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_myth_board("broadcast")
        with self.assertRaisesRegex(ValueError, "placeholder"):
            build_writer(template=PARTNER_TEMPLATE)

    def test_condition_records_board_only_when_configured(self):
        game = build_game()
        self.assertEqual(build_condition(game, build_writer(), {}, {})["protocol"]["myth"]["board"], "persistent")
        self.assertNotIn("board", build_condition(game, build_writer(None, PARTNER_TEMPLATE), {}, {})["protocol"]["myth"])


if __name__ == "__main__":
    unittest.main()
