import contextlib
import io
import unittest
from unittest.mock import patch

from games.trust_game import TrustGame
from src.myth_writer import MythWriter
from src.experiment_config import ExperimentConfig
from src.simulation import run_simulation


DEFECTOR_PROMPT = """
MANDATORY GAME STRATEGY — DESIGNATED DEFECTOR:
When you are the SENDER, respond exactly: {"send": 0}
When you are the RECEIVER, respond exactly: {"return": 0}
This applies only to game decisions; participate normally in myth writing.
"""


def build_game(**kwargs):
    return TrustGame(
        endowment=5,
        multiplier=3,
        system_prompt_template="Game rules",
        round1_investor_template="Send an amount.",
        round1_trustee_template="Return an amount.",
        later_investor_template="Send an amount.",
        later_trustee_template="Return an amount.",
        defector_prompt_template=DEFECTOR_PROMPT,
        **kwargs,
    )


class DefectorPopulationTests(unittest.TestCase):
    def setUp(self):
        self.agent_ids = [f"Agent_{index}" for index in range(1, 9)]
        self.agent_names = {
            agent_id: f"Name {index}"
            for index, agent_id in enumerate(self.agent_ids, start=1)
        }

    def test_ratio_assigns_reproducible_fraction(self):
        first = build_game(defector_ratio=0.25, defector_seed=7)
        second = build_game(defector_ratio=0.25, defector_seed=7)

        first.configure_agents(self.agent_ids, self.agent_names)
        second.configure_agents(self.agent_ids, self.agent_names)

        self.assertEqual(first.defector_agent_ids, second.defector_agent_ids)
        self.assertEqual(len(first.defector_agent_ids), 2)
        self.assertEqual(first.get_population_metadata()["defector_ratio_actual"], 0.25)

    def test_only_defectors_receive_game_instruction(self):
        game = build_game(
            defector_agent_ids=["Agent_2", "Agent_7"],
        )
        game.configure_agents(self.agent_ids, self.agent_names)

        defector_prompt = game.with_prompt_context(
            "Round 1 game decision.",
            "Agent_2",
            "Agent_1",
        )
        standard_prompt = game.with_prompt_context(
            "Round 1 game decision.",
            "Agent_1",
            "Agent_2",
        )

        self.assertIn("DESIGNATED DEFECTOR", defector_prompt)
        self.assertIn('{"send": 0}', defector_prompt)
        self.assertIn("participate normally in myth writing", defector_prompt)
        self.assertNotIn("DESIGNATED DEFECTOR", standard_prompt)

    def test_explicit_assignment_is_recorded_by_agent(self):
        game = build_game(defector_agent_ids=["Agent_3", "Agent_8"])
        game.configure_agents(self.agent_ids, self.agent_names)

        self.assertEqual(
            game.get_agent_types(),
            {
                agent_id: (
                    "defector" if agent_id in {"Agent_3", "Agent_8"} else "standard"
                )
                for agent_id in self.agent_ids
            },
        )

    def test_hidden_forced_defector_has_scripted_zero_responses(self):
        game = build_game(
            defector_agent_ids=["Agent_2"],
            defector_action_policy="forced_zero",
            defector_myth_policy="normal",
            defector_role_visible_to_self=False,
        )
        game.configure_agents(self.agent_ids, self.agent_names)

        prompt = game.with_prompt_context(
            "Round 1 game decision.",
            "Agent_2",
            "Agent_1",
        )

        self.assertNotIn("DESIGNATED DEFECTOR", prompt)
        self.assertEqual(
            game.get_forced_game_response("Agent_2", "investor")["content"],
            '{"send": 0}',
        )
        self.assertEqual(
            game.get_forced_game_response("Agent_2", "trustee")["content"],
            '{"return": 0}',
        )
        self.assertIsNone(
            game.get_forced_game_response("Agent_1", "investor")
        )

    def test_forced_defectors_skip_game_llm_but_write_normal_myths(self):
        game = build_game(
            defector_agent_ids=["Agent_1", "Agent_2"],
            defector_action_policy="forced_zero",
            defector_myth_policy="normal",
            defector_role_visible_to_self=False,
        )
        myth_writer = MythWriter(
            myth_topic="anything",
            round1_template="Write a myth. {topic_instruction}",
            later_rounds_template="Rewrite your myth.",
        )
        llm_calls = []

        def fake_call_llm(client, model, temperature, messages):
            prompt = messages[-1]["content"]
            llm_calls.append(prompt)
            if "Write a myth" in prompt:
                content = "A normal myth about a difficult exchange."
            elif "Send an amount" in prompt:
                content = '{"send": 1}'
            else:
                content = '{"return": 0}'
            return {"content": content, "reasoning": None, "usage": None}

        with patch("src.simulation.create_llm_client", return_value=object()), patch(
            "src.agents.call_llm",
            side_effect=fake_call_llm,
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                sim_data = run_simulation(
                    game=game,
                    model="mock/model",
                    temperature=0,
                    num_turns=1,
                    num_agents=8,
                    memory_capacity=6,
                    agent_biases="",
                    myth_writer=myth_writer,
                    task_order=["game", "myth"],
                    chat_memory_mode="memory_primary",
                )

        self.assertEqual(len(llm_calls), 14)
        self.assertEqual(len(sim_data.conversation_history[0]["myths"]), 8)
        self.assertEqual(
            sim_data.run_metadata["defector_action_policy"],
            "forced_zero",
        )
        self.assertEqual(sim_data.run_metadata["defector_myth_policy"], "normal")
        self.assertFalse(
            sim_data.run_metadata["defector_role_visible_to_self"]
        )
        defectors = {"Agent_1", "Agent_2"}
        for dyad in sim_data.conversation_history[0]["dyads"]:
            if dyad["investor"] in defectors:
                self.assertEqual(dyad["sent"], 0)
            if dyad["trustee"] in defectors:
                self.assertEqual(dyad["returned"], 0)

        for agent_id in defectors:
            agent = sim_data.agents[agent_id]
            game_event, myth_event = agent.interaction_history
            self.assertEqual(
                sim_data.conversation_history[0]["game_responses"][agent_id][
                    "response_source"
                ],
                "forced_zero",
            )
            self.assertEqual(
                sim_data.conversation_history[0]["myth_responses"][agent_id][
                    "response_source"
                ],
                "llm",
            )
            self.assertEqual(
                game_event["response"]["response_source"],
                "forced_zero",
            )
            self.assertEqual(myth_event["response"]["response_source"], "llm")
            self.assertNotIn("DESIGNATED DEFECTOR", game_event["prompt"])
            self.assertIn(
                game_event["response"]["content"],
                [message.get("content") for message in agent.messages],
            )
            self.assertIn(
                "A normal myth about a difficult exchange.",
                sim_data.conversation_history[0]["myths"][agent_id],
            )

    def test_invalid_ratio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            build_game(defector_ratio=1.1)

        with self.assertRaisesRegex(ValueError, "defector_action_policy"):
            build_game(defector_action_policy="typo")

        with self.assertRaisesRegex(ValueError, "defector_myth_policy"):
            build_game(defector_myth_policy="typo")

    def test_configured_8_agent_condition_uses_quarter_defectors(self):
        config = ExperimentConfig("config/experiments.yaml")
        combinations = config.get_experiment_combinations(
            "sonnet45_8agent_defectors25_history3_r10_n5"
        )

        self.assertTrue(combinations)
        self.assertTrue(
            all(combo["game_params"]["num_agents"] == 8 for combo in combinations)
        )
        self.assertTrue(
            all(combo["game_params"]["defector_ratio"] == 0.25 for combo in combinations)
        )
        self.assertTrue(
            all(combo["defector_game_instruction"] for combo in combinations)
        )


if __name__ == "__main__":
    unittest.main()
