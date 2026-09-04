import contextlib
import io
import unittest
from unittest.mock import patch

from games.base_game import InvalidGameResponseError
from games.trust_game import TrustGame
from src.simulation import run_simulation


def build_game():
    return TrustGame(
        endowment=5,
        multiplier=3,
        system_prompt_template="Game rules",
        round1_investor_template='Respond exactly as JSON: {{"send": amount}}',
        round1_trustee_template='Respond exactly as JSON: {{"return": amount}}',
        later_investor_template="Send an amount.",
        later_trustee_template="Return an amount.",
    )


class GameResponseValidationTests(unittest.TestCase):
    def test_strict_boundary_requires_quoted_expected_key(self):
        game = build_game()
        game.validate_game_response('{"send": 4}', "investor")

        for malformed in ("{send: 4}", '{"return": 4}', "4"):
            with self.subTest(malformed=malformed):
                with self.assertRaises(InvalidGameResponseError):
                    game.validate_game_response(malformed, "investor")

    def test_simulation_retries_without_remembering_rejected_decision(self):
        responses = iter(
            [
                {"content": "{send: 4}", "reasoning": None, "usage": None},
                {"content": '{"send": 4}', "reasoning": None, "usage": None},
                {"content": '{"return": 3}', "reasoning": None, "usage": None},
            ]
        )

        with patch("src.simulation.create_llm_client", return_value=object()), patch(
            "src.agents.call_llm",
            side_effect=lambda *args, **kwargs: next(responses),
        ), patch("src.simulation.time.sleep"):
            with contextlib.redirect_stdout(io.StringIO()):
                sim_data = run_simulation(
                    game=build_game(),
                    model="mock/model",
                    temperature=0,
                    num_turns=1,
                    num_agents=2,
                    memory_capacity=3,
                    agent_biases="",
                    myth_writer=None,
                    task_order=["game"],
                    chat_memory_mode="memory_primary",
                )

        investor = sim_data.agents["Agent_1"]
        self.assertEqual(len(investor.interaction_history), 2)
        rejected, accepted = investor.interaction_history
        self.assertEqual(rejected["error"]["type"], "InvalidGameResponseError")
        self.assertEqual(rejected["response"]["content"], "{send: 4}")
        self.assertNotIn(
            "{send: 4}",
            [message.get("content") for message in investor.messages],
        )
        self.assertNotIn("error", accepted)

        round_entry = sim_data.conversation_history[0]
        self.assertEqual(round_entry["game_responses"]["Agent_1"]["content"], '{"send": 4}')
        self.assertEqual(round_entry["sent"], 4)
        self.assertEqual(round_entry["returned"], 3)

    def test_wrong_role_key_gets_a_corrective_retry(self):
        # 2026-09-04: JSON-only Claude replies answered with the other role's
        # key; a plain repeat reproduced it. The retry now names role + key.
        prompts_seen = []

        def fake_call_llm(client, model, temperature, messages, **kwargs):
            prompts_seen.append(messages[-1]["content"])
            if len(prompts_seen) == 1:
                return {"content": '{"return": 3.5}', "reasoning": None, "usage": None}
            if len(prompts_seen) == 2:
                return {"content": '{"send": 4}', "reasoning": None, "usage": None}
            return {"content": '{"return": 3}', "reasoning": None, "usage": None}

        with patch("src.simulation.create_llm_client", return_value=object()), patch(
            "src.agents.call_llm", side_effect=fake_call_llm
        ), patch("src.simulation.time.sleep"):
            with contextlib.redirect_stdout(io.StringIO()):
                sim_data = run_simulation(
                    game=build_game(),
                    model="mock/model",
                    temperature=0,
                    num_turns=1,
                    num_agents=2,
                    memory_capacity=3,
                    agent_biases="",
                    myth_writer=None,
                    task_order=["game"],
                    chat_memory_mode="memory_primary",
                )

        self.assertIn("you are the SENDER", prompts_seen[1])
        self.assertIn("'send'", prompts_seen[1])
        self.assertTrue(prompts_seen[1].startswith(prompts_seen[0]))
        investor = sim_data.agents["Agent_1"]
        rejected, accepted = investor.interaction_history
        self.assertEqual(rejected["error"]["type"], "InvalidGameResponseError")
        self.assertNotIn("error", accepted)
        self.assertEqual(sim_data.conversation_history[0]["sent"], 4)

    def test_run_dies_after_exhausting_corrective_retries(self):
        responses = iter(
            [{"content": '{"return": 1}', "reasoning": None, "usage": None}] * 3
        )
        with patch("src.simulation.create_llm_client", return_value=object()), patch(
            "src.agents.call_llm", side_effect=lambda *a, **k: next(responses)
        ), patch("src.simulation.time.sleep"):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(InvalidGameResponseError):
                    run_simulation(
                        game=build_game(),
                        model="mock/model",
                        temperature=0,
                        num_turns=1,
                        num_agents=2,
                        memory_capacity=3,
                        agent_biases="",
                        myth_writer=None,
                        task_order=["game"],
                        chat_memory_mode="memory_primary",
                    )


if __name__ == "__main__":
    unittest.main()
