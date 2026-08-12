import unittest
from unittest.mock import patch

from experiments.run_noisy_batch import NoisyExperimentConfig
from games.trust_game_noisy import TrustGameNoisy
from src.simulation import SimulationData


MYTH_DECISION_LINK = (
    "Take any myths written in this session into account when making your decision."
)


def build_game(num_agents=2, game_prompt_addition=""):
    game = TrustGameNoisy(
        endowment=5,
        multiplier=3,
        system_prompt_template="Game rules",
        round1_investor_template="Send an amount.",
        round1_trustee_template="Return after seeing {sent}.",
        later_investor_template="Round {turn}: send an amount.",
        later_trustee_template="Round {turn}: current send is {current_round_sent}.",
        noise_config={
            "type": "uniform",
            "range": 1,
            "direction": "both",
            "applies_to": "sent",
        },
        history_policy="none",
        game_prompt_addition=game_prompt_addition,
    )
    agent_ids = [f"Agent_{index}" for index in range(1, num_agents + 1)]
    game.configure_agents(agent_ids)
    return game


def build_dyad_state():
    sim_data = SimulationData()
    sim_data.game_data = {
        "balances": {"Agent_1": 5, "Agent_2": 5},
        "balances_communicated": {"Agent_1": 5, "Agent_2": 5},
        "pending_sents": {},
        "pending_sents_communicated": {},
    }
    sim_data.conversation_history = [
        {
            "round": 1,
            "dyads": [
                {
                    "round": 1,
                    "dyad_id": "dyad_1",
                    "agents": ["Agent_1", "Agent_2"],
                    "investor": "Agent_1",
                    "trustee": "Agent_2",
                    "sent": 2,
                    "sent_communicated": 2.5,
                    "returned": 1,
                    "returned_communicated": 1,
                    "investor_payoff": 4,
                    "trustee_payoff": 5,
                }
            ],
        }
    ]
    return sim_data


def build_multi_agent_state():
    sim_data = SimulationData()
    agent_ids = [f"Agent_{index}" for index in range(1, 9)]
    sim_data.game_data = {
        "balances": {agent_id: 5 for agent_id in agent_ids},
        "balances_communicated": {agent_id: 5 for agent_id in agent_ids},
        "pending_sents": {},
        "pending_sents_communicated": {},
    }
    sim_data.conversation_history = [
        {
            "round": 2,
            "pairings": [
                {
                    "dyad_id": "dyad_1",
                    "investor": "Agent_1",
                    "trustee": "Agent_2",
                },
                {
                    "dyad_id": "dyad_2",
                    "investor": "Agent_3",
                    "trustee": "Agent_4",
                },
                {
                    "dyad_id": "dyad_3",
                    "investor": "Agent_5",
                    "trustee": "Agent_6",
                },
                {
                    "dyad_id": "dyad_4",
                    "investor": "Agent_7",
                    "trustee": "Agent_8",
                },
            ],
        }
    ]
    return sim_data


class NoisyDyadTransferPromptTests(unittest.TestCase):
    def setUp(self):
        self.game = build_game()
        self.sim_data = build_dyad_state()

    def test_investor_prompt_does_not_generate_or_cache_transfer_noise(self):
        with patch.object(
            self.game, "_apply_noise", wraps=self.game._apply_noise
        ) as apply_noise:
            prompt = self.game.get_game_prompt_later_round(
                "Agent_2", 2, self.sim_data, {}
            )

        self.assertIn("send an amount", prompt)
        apply_noise.assert_not_called()
        self.assertEqual(self.sim_data.game_data["pending_sents_communicated"], {})

    def test_trustee_prompt_requires_the_current_actual_send(self):
        with self.assertRaisesRegex(ValueError, "Investor should have responded first"):
            self.game.get_game_prompt_later_round("Agent_1", 2, self.sim_data, {})

        self.assertEqual(self.sim_data.game_data["pending_sents_communicated"], {})

    def test_trustee_noise_is_based_on_actual_send_and_cached_once(self):
        self.game.process_intermediate_response(
            "Agent_2", '{"send": 2.5}', 2, self.sim_data
        )

        with patch.object(
            self.game, "_apply_noise", return_value=(3.25, 2.5)
        ) as apply_noise:
            first_prompt = self.game.get_game_prompt_later_round(
                "Agent_1", 2, self.sim_data, {}
            )
            second_prompt = self.game.get_game_prompt_later_round(
                "Agent_1", 2, self.sim_data, {}
            )

        apply_noise.assert_called_once_with(2.5, 5)
        self.assertIn("current send is 3.25", first_prompt)
        self.assertEqual(first_prompt, second_prompt)
        self.assertEqual(
            self.sim_data.game_data["pending_sents_communicated"]["dyad_1"],
            3.25,
        )


class NoisyMultiAgentTransferPromptTests(unittest.TestCase):
    def test_multi_agent_path_has_the_same_sender_then_receiver_contract(self):
        game = build_game(num_agents=8)
        sim_data = build_multi_agent_state()

        with patch.object(
            game, "_apply_noise", return_value=(3.25, 2.5)
        ) as apply_noise:
            game.get_game_prompt_later_round("Agent_1", 2, sim_data, {})
            apply_noise.assert_not_called()

            with self.assertRaisesRegex(
                ValueError, "Investor should have responded first"
            ):
                game.get_game_prompt_later_round("Agent_2", 2, sim_data, {})

            game.process_intermediate_response(
                "Agent_1", '{"send": 2.5}', 2, sim_data
            )
            trustee_prompt = game.get_game_prompt_later_round(
                "Agent_2", 2, sim_data, {}
            )

        apply_noise.assert_called_once_with(2.5, 5)
        self.assertIn("They sent you $3.25", trustee_prompt)


class MythDecisionLinkPromptTests(unittest.TestCase):
    def assert_link_once(self, prompt):
        self.assertEqual(prompt.count(MYTH_DECISION_LINK), 1)

    def test_dyad_round1_and_later_prompts_each_include_link_once(self):
        game = build_game(game_prompt_addition=MYTH_DECISION_LINK)

        self.assert_link_once(game.get_game_prompt_round_1("Agent_1", None, 1))

        round1_state = build_dyad_state()
        game.process_intermediate_response(
            "Agent_1", '{"send": 2.5}', 1, round1_state
        )
        self.assert_link_once(
            game.get_game_prompt_round_1("Agent_2", None, 1)
        )

        later_state = build_dyad_state()
        self.assert_link_once(
            game.get_game_prompt_later_round("Agent_2", 2, later_state, {})
        )
        game.process_intermediate_response(
            "Agent_2", '{"send": 2.5}', 2, later_state
        )
        self.assert_link_once(
            game.get_game_prompt_later_round("Agent_1", 2, later_state, {})
        )

    def test_multi_agent_later_prompts_each_include_link_once(self):
        game = build_game(
            num_agents=8,
            game_prompt_addition=MYTH_DECISION_LINK,
        )
        sim_data = build_multi_agent_state()

        self.assert_link_once(
            game.get_game_prompt_later_round("Agent_1", 2, sim_data, {})
        )
        game.process_intermediate_response(
            "Agent_1", '{"send": 2.5}', 2, sim_data
        )
        self.assert_link_once(
            game.get_game_prompt_later_round("Agent_2", 2, sim_data, {})
        )


class CorrectedV2ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = NoisyExperimentConfig("config/experiments_noisy.yaml")

    def test_v2_cells_match_three_round_memory_horizon(self):
        expected = {
            "noise2i_memprimary_v2_game": (2, 3),
            "noise2i_memprimary_v2_game_myth": (2, 6),
            "noise2i_memprimary_v2_myth_game": (2, 6),
            "noise8i_memprimary_v2_game": (8, 3),
            "noise8i_memprimary_v2_game_myth": (8, 6),
            "noise8i_memprimary_v2_myth_game": (8, 6),
        }

        for experiment_name, (num_agents, memory_capacity) in expected.items():
            with self.subTest(experiment_name=experiment_name):
                combinations = self.config.get_experiment_combinations(
                    experiment_name
                )
                self.assertEqual(len(combinations), 5)
                self.assertTrue(
                    all(
                        combo["game_params"]["num_agents"] == num_agents
                        and combo["game_params"]["memory_capacity"]
                        == memory_capacity
                        for combo in combinations
                    )
                )

    def test_v2_myth_cells_use_runtime_link_not_template_mutation(self):
        myth_cells = [
            "noise2i_memprimary_v2_game_myth",
            "noise2i_memprimary_v2_myth_game",
            "noise8i_memprimary_v2_game_myth",
            "noise8i_memprimary_v2_myth_game",
        ]
        template_keys = [
            "trust_game_round1_investor",
            "trust_game_round1_trustee",
            "trust_game_later_investor",
            "trust_game_later_trustee",
        ]

        for experiment_name in myth_cells:
            with self.subTest(experiment_name=experiment_name):
                combo = self.config.get_experiment_combinations(
                    experiment_name
                )[0]
                self.assertEqual(
                    combo["game_prompt_addition"],
                    MYTH_DECISION_LINK,
                )
                self.assertEqual(
                    combo["game_prompt_addition_id"],
                    "myth_decision_link",
                )
                self.assertTrue(
                    all(
                        MYTH_DECISION_LINK not in combo[key]
                        for key in template_keys
                    )
                )

        game_only = self.config.get_experiment_combinations(
            "noise2i_memprimary_v2_game"
        )[0]
        self.assertEqual(game_only["game_prompt_addition"], "")
        self.assertIsNone(game_only["game_prompt_addition_id"])


if __name__ == "__main__":
    unittest.main()
