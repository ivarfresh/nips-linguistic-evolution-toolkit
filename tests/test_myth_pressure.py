"""Myth-pressure pilot (2026-09-28): word budget, delivery note and council."""
import contextlib
import io
import threading
import unittest
from unittest.mock import patch

from games.trust_game import TrustGame
from src.experiment_condition import build_condition
from src.myth_writer import MythWriter, deliver_myth, validate_myth_pressure
from src.simulation import run_simulation

MYTH = "Myth: one two three four five six seven eight"
PRESSURE = {
    "word_budget_schedule": [10, 3],
    "council_exchanges": 1,
    "delivery_note_template": "Last myth had {written_words} words; they saw {delivered_words}.\n",
    "council_prompt_template": "COUNCIL own={own_myth} other={other_agent_myth} next={next_word_budget}\n{transcript}",
    "council_block_template": "Council notes:\n{transcript}\n",
}


def build_game():
    return TrustGame(
        endowment=5, multiplier=3, system_prompt_template="Game rules",
        round1_investor_template="Send.", round1_trustee_template="Return.",
        later_investor_template="Send.", later_trustee_template="Return.",
    )


def build_writer(pressure=PRESSURE):
    return MythWriter(
        myth_topic="anything",
        round1_template="Write a myth; budget {word_budget}.",
        later_rounds_template="{delivery_note}{council_block}Other: {other_agent_myth}\nBudget {word_budget}.",
        pressure=pressure,
    )


class DeliveryTests(unittest.TestCase):
    def test_under_budget_is_untouched(self):
        self.assertEqual(deliver_myth(MYTH, 8), (MYTH, 8, 8))

    def test_cut_keeps_label_and_original_spacing(self):
        text = "Myth:  one two\n\nthree four"
        self.assertEqual(deliver_myth(text, 3), ("Myth:  one two\n\nthree", 4, 3))

    def test_label_does_not_count(self):
        self.assertEqual(deliver_myth("**Myth:** a b c", 2), ("**Myth:** a b", 3, 2))

    def test_invalid_blocks_are_rejected(self):
        bad = [
            {**PRESSURE, "word_budget_schedule": []},
            {**PRESSURE, "word_budget_schedule": [10, 0]},
            {**PRESSURE, "council_exchanges": -1},
            {**PRESSURE, "surprise": 1},
            {k: v for k, v in PRESSURE.items() if k != "council_prompt_template"},
        ]
        for pressure in bad:
            with self.subTest(pressure=pressure), self.assertRaises(ValueError):
                validate_myth_pressure(pressure)

    def test_budget_holds_last_value(self):
        writer = build_writer()
        self.assertEqual([writer.word_budget(t) for t in (1, 2, 3, 9)], [10, 3, 3, 3])
        self.assertIsNone(build_writer(None).word_budget(1))


class PressureSimulationTests(unittest.TestCase):
    def run_sim(self, writer, turns=3):
        prompts, lock = [], threading.Lock()

        def fake_call_llm(client, model, temperature, messages):
            prompt = messages[-1]["content"]
            with lock:
                prompts.append(prompt)
            content = "Keep it short." if prompt.startswith("COUNCIL") else MYTH
            return {"content": content, "reasoning": None, "usage": None}

        with patch("src.simulation.create_llm_client", return_value=object()), patch(
            "src.agents.call_llm", side_effect=fake_call_llm,
        ), contextlib.redirect_stdout(io.StringIO()):
            sim = run_simulation(
                game=build_game(), model="mock/model", temperature=0, num_turns=turns,
                num_agents=2, memory_capacity=6, agent_biases="", myth_writer=writer,
                task_order=["myth"], chat_memory_mode="memory_primary",
            )
        return sim, prompts

    def test_delivery_council_and_prompts(self):
        sim, prompts = self.run_sim(build_writer())
        r1, r2, r3 = sim.conversation_history
        # Round 1: under budget, delivered whole. Round 2+: cut to 3 words.
        self.assertEqual(set(r1["myths"].values()), {MYTH})
        self.assertEqual(set(r2["myths"].values()), {"Myth: one two three"})
        self.assertEqual(r2["myth_delivery"]["Agent_1"],
                         {"word_budget": 3, "written_words": 8, "delivered_words": 3, "truncated": True})
        # Full text survives in myth_responses.
        self.assertEqual(r2["myth_responses"]["Agent_1"]["content"], MYTH)
        # Council after rounds 1 and 2, not after the last round.
        self.assertIn("council", r1)
        self.assertIn("council", r2)
        self.assertNotIn("council", r3)
        council = next(iter(r1["council"].values()))
        self.assertEqual([m["agent"] for m in council["messages"]], council["agents"])
        # Council calls stay out of chat memory but are audited.
        for agent in sim.agents.values():
            self.assertFalse(any(m["content"].startswith("COUNCIL") for m in agent.messages))
            tasks = [e["metadata"].get("task") for e in agent.interaction_history]
            self.assertEqual(tasks.count("council"), 2)
        # The council prompt shows next round's budget and the delivered myths.
        self.assertTrue(any("next=3" in p and "own=" + MYTH in p for p in prompts))
        # Round-2 myth prompt carries the delivery note and the council transcript.
        later = [p for p in prompts if p.startswith("Last myth had")]
        self.assertTrue(later)
        self.assertIn("Last myth had 8 words; they saw 8.", later[0])
        self.assertIn("Council notes:\n", later[0])
        self.assertIn("You: Keep it short.", later[0])
        self.assertIn("The other agent: Keep it short.", later[0])
        self.assertIn("Other: " + MYTH, later[0])

    def test_no_council_arm_runs_without_council(self):
        pressure = {k: v for k, v in PRESSURE.items() if not k.startswith("council")}
        sim, prompts = self.run_sim(build_writer(pressure))
        self.assertFalse(any("council" in entry for entry in sim.conversation_history))
        self.assertFalse(any(p.startswith("COUNCIL") for p in prompts))
        self.assertTrue(any(p.startswith("Last myth had 8 words; they saw 3.") for p in prompts))

    def test_condition_records_pressure_only_when_configured(self):
        game = build_game()
        with_pressure = build_condition(game, build_writer(), {}, {})
        without = build_condition(game, build_writer(None), {}, {})
        self.assertEqual(with_pressure["protocol"]["myth"]["pressure"]["word_budget_schedule"], [10, 3])
        self.assertNotIn("pressure", without["protocol"]["myth"])


class ConfigTests(unittest.TestCase):
    def test_config_typo_in_pressure_block_raises(self):
        import copy
        from experiments.run_noisy_batch import NoisyExperimentConfig, build_noisy_protocol
        config = NoisyExperimentConfig("config/myth_pressure_pilot_20260928.yaml")
        params = copy.deepcopy(config.config["game_params"]["myth_pressure_tight_council_noisy2_negative_twotask_r3"])
        params["myth_pressure"]["council_exchange"] = params["myth_pressure"].pop("council_exchanges")
        combo = {"myth_pressure": config._get_myth_pressure(params)}
        with self.assertRaisesRegex(ValueError, "Unknown myth_pressure keys"):
            MythWriter("anything", "a", "b", pressure=combo["myth_pressure"])

    def test_pilot_combos_and_september_combos(self):
        from scripts.run_noisy_missing import load_combinations
        with contextlib.redirect_stdout(io.StringIO()):
            pilot = load_combinations("myth_pressure_pilot_dyad_myth_game_claude_n5",
                                      "config/myth_pressure_pilot_20260928.yaml")
            september = load_combinations("negative_only_reasoning_rerun_dyad_myth_game_claude_n5",
                                          "config/experiments_noisy.yaml")
        self.assertEqual(len(pilot), 20)
        self.assertTrue(all("myth_pressure" in c for c in pilot))
        self.assertFalse(any("myth_pressure" in c for c in september))
        # Apart from myth prompts and the pressure block, the pilot keeps the
        # September informed-noise game protocol and request profile.
        base = next(c for c in september if c["game_params_name"] == "noisy2_crossmodel_negative_twotask_r3")
        for combo in pilot:
            params = {k: v for k, v in combo["game_params"].items() if k != "myth_pressure"}
            self.assertEqual(params, base["game_params"])
            self.assertEqual(combo["request_plan"].as_dict(), base["request_plan"].as_dict())
            self.assertEqual(combo["template"], base["template"])
            self.assertEqual(combo["trust_game_later_investor"], base["trust_game_later_investor"])


if __name__ == "__main__":
    unittest.main()
