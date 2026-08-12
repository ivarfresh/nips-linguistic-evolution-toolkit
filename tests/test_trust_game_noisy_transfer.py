import unittest
from unittest.mock import patch

from experiments.run_noisy_batch import NoisyExperimentConfig, resolve_protocol_seeds
from games.trust_game_noisy import TrustGameNoisy
from src.simulation import SimulationData


MYTH_DECISION_LINK = (
    "Take any myths written in this session into account when making your decision."
)


def build_game(
    num_agents=2,
    game_prompt_addition="",
    history_policy="none",
    self_history_window=0,
    coplayer_history_window=0,
    show_agent_names=True,
    pairing_mode="balanced",
    pairing_seed=None,
    noise_seed=None,
    prompt_regime="legacy",
):
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
        history_policy=history_policy,
        self_history_window=self_history_window,
        coplayer_history_window=coplayer_history_window,
        show_agent_names=show_agent_names,
        game_prompt_addition=game_prompt_addition,
        pairing_mode=pairing_mode,
        pairing_seed=pairing_seed,
        noise_seed=noise_seed,
        prompt_regime=prompt_regime,
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
                    "roles": {
                        "Agent_1": "investor",
                        "Agent_2": "trustee",
                    },
                    "sent": 2,
                    "sent_communicated": 2.5,
                    "received": 6,
                    "received_communicated": 7.5,
                    "returned": 1,
                    "returned_communicated": 1,
                    "investor_payoff": 4,
                    "trustee_payoff": 5,
                    "payoffs_communicated": {
                        "Agent_1": 4,
                        "Agent_2": 6.5,
                    },
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


class PairingAndPromptRegimeTests(unittest.TestCase):
    def test_fixed_pairing_keeps_partners_and_swaps_roles(self):
        game = build_game(num_agents=8, pairing_mode="fixed")
        sim_data = SimulationData()

        round_1 = game.get_round_pairings(1, sim_data)
        round_2 = game.get_round_pairings(2, sim_data)

        self.assertEqual(
            [pairing["agents"] for pairing in round_1],
            [
                ["Agent_1", "Agent_2"],
                ["Agent_3", "Agent_4"],
                ["Agent_5", "Agent_6"],
                ["Agent_7", "Agent_8"],
            ],
        )
        self.assertEqual(round_1[0]["investor"], "Agent_1")
        self.assertEqual(round_2[0]["investor"], "Agent_2")
        self.assertEqual(set(round_1[0]["agents"]), set(round_2[0]["agents"]))

    def test_fixed_system_prompt_is_truthful(self):
        game = build_game(
            num_agents=8,
            pairing_mode="fixed",
            prompt_regime="unified",
            show_agent_names=False,
        )

        prompt = game.get_system_prompt("Agent_1", None)

        self.assertIn("There are 8 agents", prompt)
        self.assertIn("same opponent throughout the run", prompt)
        self.assertIn("roles alternate every round", prompt)
        self.assertNotIn("Pairings are randomized", prompt)

    def test_unified_dyad_uses_population_prompt_and_shared_later_builder(self):
        game = build_game(
            prompt_regime="unified",
            pairing_mode="fixed",
            show_agent_names=False,
        )
        sim_data = build_dyad_state()

        system_prompt = game.get_system_prompt("Agent_1", None)
        later_prompt = game.get_game_prompt_later_round(
            "Agent_2", 2, sim_data, {}
        )

        self.assertIn("There are 2 agents", system_prompt)
        self.assertIn("same opponent throughout the run", system_prompt)
        self.assertIn("This round, you are the SENDER", later_prompt)
        self.assertIn("Respond exactly as JSON", later_prompt)
        self.assertNotIn("Round 2: send an amount", later_prompt)

    def test_unified_fixed_2_and_8_render_identical_decision_prompts(self):
        dyad_game = build_game(
            prompt_regime="unified",
            pairing_mode="fixed",
            noise_seed=31,
            show_agent_names=False,
        )
        population_game = build_game(
            num_agents=8,
            prompt_regime="unified",
            pairing_mode="fixed",
            noise_seed=31,
            show_agent_names=False,
        )
        dyad_state = build_dyad_state()
        population_state = SimulationData()
        population_state.game_data = {
            "balances": {
                f"Agent_{index}": 5 for index in range(1, 9)
            },
            "balances_communicated": {
                f"Agent_{index}": 5 for index in range(1, 9)
            },
            "pending_sents": {},
            "pending_sents_communicated": {},
        }

        dyad_investor = dyad_game.get_game_prompt_later_round(
            "Agent_2", 2, dyad_state, {}
        )
        population_investor = population_game.get_game_prompt_later_round(
            "Agent_2", 2, population_state, {}
        )
        dyad_game.process_intermediate_response(
            "Agent_2", '{"send": 2.5}', 2, dyad_state
        )
        population_game.process_intermediate_response(
            "Agent_2", '{"send": 2.5}', 2, population_state
        )
        dyad_trustee = dyad_game.get_game_prompt_later_round(
            "Agent_1", 2, dyad_state, {}
        )
        population_trustee = population_game.get_game_prompt_later_round(
            "Agent_1", 2, population_state, {}
        )

        self.assertEqual(dyad_investor, population_investor)
        self.assertEqual(dyad_trustee, population_trustee)
        self.assertEqual(
            dyad_game.get_system_prompt("Agent_1", None).replace(
                "There are 2 agents", "There are 8 agents"
            ),
            population_game.get_system_prompt("Agent_1", None),
        )

    def test_history_policies_have_explicit_nonoverlapping_meanings(self):
        sim_data = build_dyad_state()

        no_history = build_game(
            num_agents=8,
            history_policy="none",
        )._format_multi_agent_history(
            "Agent_1", "Agent_2", 2, sim_data
        )
        zero_windows = build_game(
            num_agents=8,
            history_policy="self_and_coplayer",
            self_history_window=0,
            coplayer_history_window=0,
        )._format_multi_agent_history(
            "Agent_1", "Agent_2", 2, sim_data
        )
        minimal = build_game(
            num_agents=8,
            history_policy="minimal",
        )._format_multi_agent_history(
            "Agent_1", "Agent_2", 2, sim_data
        )

        self.assertEqual(no_history, "")
        self.assertEqual(zero_windows, "")
        self.assertIn("most recent previous game round", minimal)

    def test_invalid_protocol_enums_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "history_policy"):
            build_game(history_policy="typo")
        with self.assertRaisesRegex(ValueError, "pairing_mode"):
            build_game(pairing_mode="typo")
        with self.assertRaisesRegex(ValueError, "prompt_regime"):
            build_game(prompt_regime="typo")

    def test_seeded_pairing_and_noise_are_reproducible(self):
        first_game = build_game(
            num_agents=8,
            pairing_seed=17,
            noise_seed=23,
        )
        second_game = build_game(
            num_agents=8,
            pairing_seed=17,
            noise_seed=23,
        )
        first_state = SimulationData()
        first_state.run_metadata["num_turns"] = 10
        second_state = SimulationData()
        second_state.run_metadata["num_turns"] = 10

        first_schedule = [
            first_game.get_round_pairings(turn, first_state)
            for turn in range(1, 11)
        ]
        second_schedule = [
            second_game.get_round_pairings(turn, second_state)
            for turn in range(1, 11)
        ]
        first_noise = first_game._apply_noise_for_event(
            2.5,
            5,
            turn=4,
            dyad_id="dyad_2",
            action_type="sent",
        )
        second_noise = second_game._apply_noise_for_event(
            2.5,
            5,
            turn=4,
            dyad_id="dyad_2",
            action_type="sent",
        )

        self.assertEqual(first_schedule, second_schedule)
        self.assertEqual(first_noise, second_noise)


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
                self.assertEqual(len(combinations), 10)
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

    def test_max_runs_limits_replicates_without_losing_replicate_id(self):
        combinations = self.config.get_experiment_combinations(
            "noise8i_memprimary_v2_myth_game",
            max_runs=1,
        )

        self.assertEqual(len(combinations), 1)
        self.assertEqual(combinations[0]["replicate_id"], 0)

        with self.assertRaisesRegex(ValueError, "at least 1"):
            self.config.get_experiment_combinations(
                "noise8i_memprimary_v2_myth_game",
                max_runs=0,
            )

    def test_population_isolation_is_a_complete_matched_two_by_three_design(self):
        expected = {
            "noise_population_isolation_v2_game": (["game"], 3),
            "noise_population_isolation_v2_game_myth": (
                ["game", "myth"],
                6,
            ),
            "noise_population_isolation_v2_myth_game": (
                ["myth", "game"],
                6,
            ),
        }

        for experiment_name, (task_order, memory_capacity) in expected.items():
            with self.subTest(experiment_name=experiment_name):
                combinations = self.config.get_experiment_combinations(
                    experiment_name
                )
                self.assertEqual(len(combinations), 20)
                self.assertEqual(
                    {combo["game_params"]["num_agents"] for combo in combinations},
                    {2, 8},
                )
                for combo in combinations:
                    params = combo["game_params"]
                    self.assertEqual(combo["task_order"], task_order)
                    self.assertEqual(params["memory_capacity"], memory_capacity)
                    self.assertEqual(params["history_policy"], "none")
                    self.assertEqual(params["pairing_mode"], "fixed")
                    self.assertEqual(params["prompt_regime"], "unified")
                    self.assertFalse(params["show_agent_names"])
                    self.assertTrue(params["paired_protocol_seeds"])

    def test_population_isolation_differs_only_in_num_agents(self):
        combinations = self.config.get_experiment_combinations(
            "noise_population_isolation_v2_game"
        )
        params_by_name = {
            combo["game_params_name"]: combo["game_params"]
            for combo in combinations
        }

        self.assertEqual(len(combinations), 20)
        self.assertEqual(len(params_by_name), 2)
        dyad = dict(
            params_by_name["noisy_bidirectional_informed_population_v2_dyad"]
        )
        fixed_8 = dict(
            params_by_name["noisy8_bidirectional_informed_population_v2_fixed"]
        )
        self.assertEqual(dyad.pop("num_agents"), 2)
        self.assertEqual(fixed_8.pop("num_agents"), 8)
        self.assertEqual(dyad, fixed_8)
        self.assertEqual(dyad["prompt_regime"], "unified")
        self.assertEqual(dyad["pairing_mode"], "fixed")
        self.assertEqual(dyad["history_policy"], "none")
        self.assertEqual(dyad["memory_capacity"], 3)

    def test_identity_isolation_differs_only_in_name_visibility(self):
        combinations = self.config.get_experiment_combinations(
            "noise8i_identity_v2_game"
        )
        params_by_name = {
            combo["game_params_name"]: combo["game_params"]
            for combo in combinations
        }

        self.assertEqual(len(combinations), 20)
        hidden = dict(
            params_by_name["noisy8_bidirectional_informed_identity_v2_hidden"]
        )
        named = dict(
            params_by_name["noisy8_bidirectional_informed_identity_v2_named"]
        )
        self.assertFalse(hidden.pop("show_agent_names"))
        self.assertTrue(named.pop("show_agent_names"))
        self.assertEqual(hidden, named)
        self.assertEqual(hidden["pairing_mode"], "balanced")
        self.assertEqual(hidden["history_policy"], "none")
        self.assertEqual(hidden["memory_capacity"], 3)

    def test_defector_cells_differ_only_in_defector_ratio(self):
        expected_capacity = {
            "noise8i_defector_v2_game": 3,
            "noise8i_defector_v2_game_myth": 6,
            "noise8i_defector_v2_myth_game": 6,
        }
        for experiment_name, memory_capacity in expected_capacity.items():
            combinations = self.config.get_experiment_combinations(
                experiment_name
            )
            params_by_name = {
                combo["game_params_name"]: combo["game_params"]
                for combo in combinations
            }
            control_name = next(
                name for name in params_by_name if name.endswith("control")
            )
            treatment_name = next(
                name for name in params_by_name if name.endswith("treatment25")
            )
            control = dict(params_by_name[control_name])
            treatment = dict(params_by_name[treatment_name])

            with self.subTest(experiment_name=experiment_name):
                self.assertEqual(len(combinations), 10)
                self.assertEqual(control.pop("defector_ratio"), 0.0)
                self.assertEqual(treatment.pop("defector_ratio"), 0.25)
                self.assertEqual(control, treatment)
                self.assertEqual(
                    control["defector_action_policy"],
                    "forced_zero",
                )
                self.assertEqual(control["defector_myth_policy"], "normal")
                self.assertFalse(control["defector_role_visible_to_self"])
                self.assertEqual(control["history_policy"], "none")
                self.assertEqual(control["pairing_mode"], "balanced")
                self.assertEqual(control["prompt_regime"], "unified")
                self.assertEqual(control["memory_capacity"], memory_capacity)

    def test_new_causal_cells_pair_protocol_seeds_by_replicate(self):
        for experiment_name in (
            "noise_population_isolation_v2_game",
            "noise_population_isolation_v2_game_myth",
            "noise_population_isolation_v2_myth_game",
            "noise8i_identity_v2_game",
            "noise8i_defector_v2_game",
            "noise8i_defector_v2_game_myth",
            "noise8i_defector_v2_myth_game",
        ):
            combinations = self.config.get_experiment_combinations(experiment_name)
            seeds_by_replicate = {}
            for combo in combinations:
                seeds_by_replicate.setdefault(combo["replicate_id"], set()).add(
                    resolve_protocol_seeds(combo["game_params"], combo)
                )

            with self.subTest(experiment_name=experiment_name):
                expected_replicates = 5 if "defector" in experiment_name else 10
                self.assertEqual(
                    set(seeds_by_replicate),
                    set(range(expected_replicates)),
                )
                self.assertTrue(
                    all(len(seed_pairs) == 1 for seed_pairs in seeds_by_replicate.values())
                )
                seed_base = (
                    202608121
                    if "defector" in experiment_name
                    else 202608120
                )
                self.assertEqual(seeds_by_replicate[0], {(seed_base, seed_base)})


if __name__ == "__main__":
    unittest.main()
