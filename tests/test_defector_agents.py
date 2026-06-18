import unittest

from games.trust_game import TrustGame
from src.experiment_config import ExperimentConfig


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

    def test_invalid_ratio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            build_game(defector_ratio=1.1)

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
