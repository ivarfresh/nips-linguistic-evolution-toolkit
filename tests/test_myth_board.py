"""Persistent myth board (2026-10-02): every agent reads every earlier myth.

Game-board kind ``persistent_game`` (2026-11-07 pilot): the same board is also
pasted ahead of every game decision prompt from round 2 and kept out of chat memory."""
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


def build_writer(board="persistent", template=BOARD_TEMPLATE, board_seed=None):
    return MythWriter("anything", round1_template="Write a myth.", later_rounds_template=template, board=board,
                      board_seed=board_seed)


def run_sim(writer, turns=3, num_agents=4, task_order=("myth",)):
    prompts, lock, counter = [], threading.Lock(), itertools.count(1)

    def fake_call_llm(client, model, temperature, messages):
        with lock:
            content = messages[-1]["content"]
            prompts.append(content)
            if content == "Write a myth." or content.rstrip().endswith("Write your own myth."):
                return {"content": f"Myth: story {next(counter)}", "reasoning": None, "usage": None}
            # A game decision: both keys satisfy the validator whatever the role.
            return {"content": '{"send": 1, "return": 1, "reason": "test"}', "reasoning": None, "usage": None}

    with patch("src.simulation.create_llm_client", return_value=object()), patch(
        "src.agents.call_llm", side_effect=fake_call_llm,
    ), contextlib.redirect_stdout(io.StringIO()):
        sim = run_simulation(
            game=build_game(), model="mock/model", temperature=0, num_turns=turns,
            num_agents=num_agents, memory_capacity=6, agent_biases="", myth_writer=writer,
            task_order=list(task_order), chat_memory_mode="memory_primary",
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


GAME_BOARD_HEAD = "Here is the shared myth board."


class GameBoardTests(unittest.TestCase):
    def check_order(self, task_order):
        sim, _ = run_sim(build_writer(board="persistent_game", board_seed=3), task_order=task_order)
        history = sim.conversation_history
        for agent_id, agent in sim.agents.items():
            game_calls = [e for e in agent.interaction_history if e["metadata"]["task"] == "game"]
            self.assertEqual([e["metadata"]["round"] for e in game_calls], [1, 2, 3])
            # Round 1: nothing written yet, so no board.
            self.assertNotIn(GAME_BOARD_HEAD, game_calls[0]["prompt"])
            for e in game_calls[1:]:
                r = e["metadata"]["round"]
                self.assertTrue(e["prompt"].startswith(GAME_BOARD_HEAD))
                self.assertNotIn("Write your own myth.", e["prompt"])
                # Every myth from earlier rounds is on it; this round's myths (already
                # written in myth->game) are not.
                for earlier in history[:r - 1]:
                    for myth in earlier["myths"].values():
                        self.assertIn(myth, e["prompt"])
                for myth in history[r - 1]["myths"].values():
                    self.assertNotIn(myth, e["prompt"])
                record = history[r - 1]["board_in_game"][agent_id]
                self.assertEqual(record["board"], "persistent_game")
                self.assertEqual(record["board_seed"], 3)
                self.assertEqual(len(record["board_items"]), 4 * (r - 1))
                self.assertEqual(record["source_rounds"], list(range(1, r)))
            # Chat memory never holds the board text, only the note ahead of the prompt.
            remembered = [m["content"] for m in agent.messages if m["role"] == "user"]
            self.assertTrue(all("Round 1, myth 1:" not in m for m in remembered))
            self.assertTrue(any(m.startswith("[The shared myth board was shown here: 4 myths from rounds 1-1.]\n\n")
                                and "Write your own myth." not in m for m in remembered))
            # The myth prompt shows the board too, as for the legacy kind.
            myth_calls = [e for e in agent.interaction_history if e["metadata"]["task"] == "myth"]
            self.assertTrue(myth_calls[1]["prompt"].startswith("BOARD"))
            self.assertEqual(history[1]["myth_exposures"][agent_id]["board"], "persistent_game")
            self.assertEqual(history[1]["myth_exposures"][agent_id]["board_seed"], 3)
        return sim

    def test_game_myth_order(self):
        self.check_order(["game", "myth"])

    def test_myth_game_order(self):
        self.check_order(["myth", "game"])

    def test_board_and_myth_prompt_show_the_same_order(self):
        sim, _ = run_sim(build_writer(board="persistent_game", board_seed=3), task_order=["game", "myth"])
        r3 = sim.conversation_history[2]
        self.assertEqual(r3["board_in_game"]["Agent_1"]["board_items"], r3["myth_exposures"]["Agent_1"]["board_items"])
        # Same board for every reader in a round.
        self.assertEqual(len({str(v["board_items"]) for v in r3["board_in_game"].values()}), 1)

    def test_shuffle_differs_by_seed_and_legacy_order_is_unchanged(self):
        class Fake:
            conversation_history = [{"round": 1, "myths": {f"Agent_{i}": f"m{i}" for i in range(1, 9)}}]
        orders = {seed: [a for _, a, _ in build_writer(board="persistent_game", board_seed=seed)._board_items(Fake(), 2)]
                  for seed in range(4)}
        self.assertGreater(len({str(o) for o in orders.values()}), 1)
        legacy = [a for _, a, _ in build_writer()._board_items(Fake(), 2)]
        import random
        expected = sorted(Fake.conversation_history[0]["myths"])
        random.Random("myth-board-v1|1").shuffle(expected)
        self.assertEqual(legacy, expected)

    def test_game_board_rejects_unsupported_setups(self):
        with self.assertRaisesRegex(ValueError, "board_seed"):
            build_writer(board="persistent_game")
        for order, pattern in ((["game"], "both a game and a myth task"), (["myth"], "both a game and a myth task")):
            with patch("src.simulation.create_llm_client", return_value=object()), \
                    self.assertRaisesRegex(ValueError, pattern), contextlib.redirect_stdout(io.StringIO()):
                run_simulation(game=build_game(), model="mock/model", temperature=0, num_turns=2, num_agents=4,
                               memory_capacity=6, agent_biases="", task_order=order,
                               myth_writer=build_writer(board="persistent_game", board_seed=0),
                               chat_memory_mode="memory_primary")
        with patch("src.simulation.create_llm_client", return_value=object()), \
                self.assertRaisesRegex(ValueError, "memory_primary"), contextlib.redirect_stdout(io.StringIO()):
            run_simulation(game=build_game(), model="mock/model", temperature=0, num_turns=2, num_agents=4,
                           memory_capacity=6, agent_biases="", task_order=["game", "myth"],
                           myth_writer=build_writer(board="persistent_game", board_seed=0),
                           chat_memory_mode="stateless")

    def test_condition_records_seed_for_game_board_only(self):
        game = build_game()
        myth = build_condition(game, build_writer(board="persistent_game", board_seed=7), {}, {})["protocol"]["myth"]
        self.assertEqual((myth["board"], myth["board_seed"]), ("persistent_game", 7))
        self.assertNotIn("board_seed", build_condition(game, build_writer(), {}, {})["protocol"]["myth"])


if __name__ == "__main__":
    unittest.main()
