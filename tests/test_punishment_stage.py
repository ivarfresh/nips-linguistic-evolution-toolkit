import contextlib
import io
import unittest
from unittest.mock import patch

from experiments.run_noisy_batch import NoisyExperimentConfig
from games.trust_game_noisy import TrustGameNoisy
from src.myth_writer import MythWriter
from src.simulation import SimulationData, run_simulation


def build_game(**kwargs):
    settings = {
        "endowment": 5,
        "multiplier": 3,
        "system_prompt_template": "Game rules: {endowment}, {multiplier}",
        "round1_investor_template": "Send. {{'send': 0}}",
        "round1_trustee_template": "Saw {sent}; return. {{'return': 0}}",
        "later_investor_template": "Round {turn}; send.",
        "later_trustee_template": "Round {turn}; return {current_round_sent}.",
        "noise_config": None,
        "history_policy": "none",
        "show_agent_names": False,
        "prompt_regime": "unified",
        "punishment_enabled": True,
        "punishment_budget": 2,
        "punishment_effect_multiplier": 3,
    }
    settings.update(kwargs)
    game = TrustGameNoisy(**settings)
    return game


class PunishmentStageTests(unittest.TestCase):
    def test_frozen_smoke_config_preserves_three_round_memory(self):
        config = NoisyExperimentConfig("config/experiments_noisy.yaml")
        combinations = config.get_experiment_combinations(
            "noise8i_defector_punishment_gpt_smoke"
        )
        self.assertEqual(len(combinations), 2)
        self.assertEqual(
            {combo["replicate_id"] for combo in combinations},
            {60},
        )
        for combo in combinations:
            params = combo["game_params"]
            self.assertEqual(params["memory_capacity"], 9)
            self.assertTrue(params["punishment_enabled"])
            self.assertEqual(params["punishment_budget"], 2)
            self.assertEqual(params["punishment_effect_multiplier"], 3)

    def test_gemini_population_pilot_is_three_matched_pairs(self):
        config = NoisyExperimentConfig("config/experiments_noisy.yaml")
        combinations = config.get_experiment_combinations(
            "noise8i_defector_punishment_gemini_n3"
        )
        self.assertEqual(len(combinations), 6)
        self.assertEqual(
            {combo["replicate_id"] for combo in combinations},
            {66, 67, 68},
        )
        self.assertEqual(
            {combo["model"] for combo in combinations},
            {"google/gemini-3.1-flash-lite"},
        )
        self.assertEqual(
            {combo["game_params"]["defector_ratio"] for combo in combinations},
            {0.0, 0.25},
        )
        for combo in combinations:
            params = combo["game_params"]
            self.assertTrue(params["punishment_enabled"])
            self.assertEqual(params["punishment_prompt_variant"], "current")
            self.assertEqual(params["memory_capacity"], 9)

    def test_gemini_confirmation_is_ten_new_treatment_populations(self):
        config = NoisyExperimentConfig("config/experiments_noisy.yaml")
        combinations = config.get_experiment_combinations(
            "noise8i_defector_punishment_gemini_confirmation_n10"
        )
        self.assertEqual(len(combinations), 10)
        self.assertEqual(
            {combo["replicate_id"] for combo in combinations},
            set(range(70, 80)),
        )
        self.assertEqual(
            {combo["model"] for combo in combinations},
            {"google/gemini-3.1-flash-lite"},
        )
        for combo in combinations:
            params = combo["game_params"]
            self.assertEqual(params["defector_ratio"], 0.25)
            self.assertTrue(params["punishment_enabled"])
            self.assertEqual(params["punishment_prompt_variant"], "current")
            self.assertEqual(params["memory_capacity"], 9)

    def test_rules_and_response_boundary(self):
        game = build_game()
        game.configure_agents(["Agent_1", "Agent_2"])

        system = game.get_system_prompt("Agent_1", None)
        self.assertIn("DEDUCTION-POINT STAGE", system)
        self.assertIn("budget of 2 deduction points", system)
        self.assertIn("by up to $3", system)

        game.validate_post_game_response('{"deduct": 0}')
        game.validate_post_game_response('{"deduct": 2}')
        for malformed in ('{"deduct": 1.5}', '{"deduct": 3}', '{"send": 1}'):
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    game.validate_post_game_response(malformed)

    def test_cost_salient_variant_states_optional_sender_cost(self):
        game = build_game(punishment_prompt_variant="cost_salient")
        game.configure_agents(["Agent_1", "Agent_2"])

        system = game.get_system_prompt("Agent_1", None)
        self.assertIn("Spending is optional", system)
        self.assertIn("choosing 0 is valid", system)
        self.assertIn("costs the sender $1", system)

        sim_data = SimulationData()
        sim_data.conversation_history = [
            {
                "round": 1,
                "dyads": [
                    {
                        "agents": ["Agent_1", "Agent_2"],
                        "investor": "Agent_1",
                        "trustee": "Agent_2",
                        "returned": 3,
                        "returned_communicated": 3,
                        "investor_payoff": 3,
                        "investor_payoff_communicated": 3,
                        "trustee_payoff": 12,
                        "trustee_payoff_communicated": 12,
                    }
                ],
            }
        ]
        prompt = game.get_post_game_prompt("Agent_1", 1, sim_data)
        self.assertIn("Spending is optional", prompt)
        self.assertIn("Each point spent costs you $1", prompt)

    def test_unknown_punishment_prompt_variant_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "punishment_prompt_variant"):
            build_game(punishment_prompt_variant="moralizing")

    def test_sender_budget_and_three_to_one_effect_are_applied(self):
        game = build_game()
        game.configure_agents(["Agent_1", "Agent_2"])
        sim_data = SimulationData()
        sim_data.run_metadata["num_turns"] = 1
        pairings = game.get_round_pairings(1, sim_data)
        sim_data.conversation_history = [
            {
                "round": 1,
                "pairings": pairings,
                "dyads": [],
                "payoffs": {},
                "balances": {},
            }
        ]

        game.process_intermediate_response(
            "Agent_1",
            '{"send": 2}',
            1,
            sim_data,
        )
        game.process_turn(
            1,
            {
                "Agent_1": {"content": '{"send": 2}'},
                "Agent_2": {"content": '{"return": 1}'},
            },
            sim_data,
        )
        records = game.process_post_game_stage(
            1,
            {"Agent_1": {"content": '{"deduct": 1}'}},
            sim_data,
        )

        dyad = sim_data.conversation_history[0]["dyads"][0]
        self.assertEqual(dyad["pre_deduction_payoffs"], {"Agent_1": 4, "Agent_2": 5})
        self.assertEqual(dyad["payoffs"], {"Agent_1": 5, "Agent_2": 2})
        self.assertEqual(dyad["deduction_actual_loss"], 3)
        self.assertEqual(sim_data.game_data["balances"], {"Agent_1": 5, "Agent_2": 2})
        self.assertEqual(records["dyad_1"]["spent"], 1)

        prompt, notice = game.get_post_game_notification(
            "Agent_2",
            1,
            sim_data,
        )
        self.assertIn("spent 1 deduction point", prompt)
        self.assertIn("reduced your visible round payoff by $3", prompt)
        self.assertEqual(notice["response_source"], "deduction_notification")

    def test_target_loss_is_capped_at_zero(self):
        game = build_game()
        game.configure_agents(["Agent_1", "Agent_2"])
        sim_data = SimulationData()
        sim_data.run_metadata["num_turns"] = 1
        pairings = game.get_round_pairings(1, sim_data)
        sim_data.conversation_history = [
            {"round": 1, "pairings": pairings, "dyads": [], "payoffs": {}}
        ]
        game.process_intermediate_response("Agent_1", '{"send": 1}', 1, sim_data)
        game.process_turn(
            1,
            {
                "Agent_1": {"content": '{"send": 1}'},
                "Agent_2": {"content": '{"return": 2}'},
            },
            sim_data,
        )
        game.process_post_game_stage(
            1,
            {"Agent_1": {"content": '{"deduct": 2}'}},
            sim_data,
        )
        dyad = sim_data.conversation_history[0]["dyads"][0]
        self.assertEqual(dyad["pre_deduction_payoffs"]["Agent_2"], 1)
        self.assertEqual(dyad["deduction_intended_loss"], 6)
        self.assertEqual(dyad["deduction_actual_loss"], 1)
        self.assertEqual(dyad["trustee_payoff"], 0)

    def test_scripted_defector_sender_never_deducts(self):
        game = build_game(
            defector_agent_ids=["Agent_1"],
            defector_action_policy="forced_zero",
            defector_role_visible_to_self=False,
        )
        game.configure_agents(["Agent_1", "Agent_2"])
        response = game.get_forced_post_game_response("Agent_1")
        self.assertEqual(response["content"], '{"deduct": 0}')
        self.assertEqual(response["response_source"], "forced_zero")
        self.assertIsNone(game.get_forced_post_game_response("Agent_2"))

    def test_simulation_records_one_post_game_exchange_per_agent(self):
        game = build_game()
        myth_writer = MythWriter(
            myth_topic="anything",
            round1_template="Write a myth. {topic_instruction}",
            later_rounds_template="Rewrite the myth.",
        )

        def fake_call_llm(client, model, temperature, messages):
            prompt = messages[-1]["content"]
            if "Write a myth" in prompt:
                content = "Myth: Two travelers shared a careful promise."
            elif "deduction-point stage" in prompt:
                content = '{"deduct": 1}'
            elif "RECEIVER" in prompt or "Saw" in prompt:
                content = '{"return": 1}'
            else:
                content = '{"send": 2}'
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
                    num_agents=2,
                    memory_capacity=3,
                    agent_biases="",
                    myth_writer=myth_writer,
                    task_order=["myth", "game"],
                    chat_memory_mode="memory_primary",
                )

        entry = sim_data.conversation_history[0]
        self.assertEqual(set(entry["deduction_responses"]), {"Agent_1"})
        self.assertEqual(set(entry["deduction_notifications"]), {"Agent_2"})
        self.assertEqual(len(sim_data.agents["Agent_1"].interaction_history), 3)
        self.assertEqual(len(sim_data.agents["Agent_2"].interaction_history), 3)
        self.assertEqual(
            sim_data.agents["Agent_2"].interaction_history[-1]["response"][
                "response_source"
            ],
            "deduction_notification",
        )


if __name__ == "__main__":
    unittest.main()
