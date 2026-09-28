"""Generate the frozen myth-pressure pilot config (2026-09-28).

Writes config/myth_pressure_pilot_20260928.yaml: the September Sonnet 4.5
myth-first dyad set (`negative_only_reasoning_rerun_dyad_myth_game_claude_n5`,
informed negative-only noise, memory-primary, pinned September request profile)
crossed with four myth-pressure arms: word budget (loose 200 / tight shrinking)
x council between rounds (off / two exchanges). The launcher audits the result
against the September combination, so hand edits are pointless.
"""
from __future__ import annotations
import copy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "config/experiments_noisy.yaml"
TARGET = ROOT / "config/myth_pressure_pilot_20260928.yaml"
SEPTEMBER_SET = "negative_only_reasoning_rerun_dyad_myth_game_claude_n5"
SEPTEMBER_GAME_PARAMS = "noisy2_crossmodel_negative_twotask_r3"
SET_NAME = "myth_pressure_pilot_dyad_myth_game_claude_n5"

# The myth prompts differ from September only in length handling: no word count
# is requested; instead the writer learns the other agent sees only the first
# {word_budget} words. The wording is identical in every arm, only the number moves.
PROMPTS = {
    "myth_writing_default_pressure": (
        "Write a myth. {topic_instruction}\n"
        "In this session, this myth and the game you are playing with the other agent are connected. "
        "Write a myth that reflects how the game should be played, drawing on the game you have been playing up to this point.\n"
        "The other agent will see only the first {word_budget} words of your myth.\n"
    ),
    "myth_writing_later_rounds_pressure_memory_primary": (
        "{delivery_note}{council_block}"
        "Here is the myth the other agent wrote in the previous round, as you received it:\n"
        "{other_agent_myth}\n\n"
        "Write your own myth. Use the myth you wrote in the previous round as inspiration, but adapt it in your own way. "
        "The myth should reflect how the game should be played, drawing on the game you have been playing up to this point.\n\n"
        "The other agent will see only the first {word_budget} words of your myth. Format exactly:\n"
        "Myth: [your story here].\n"
    ),
    "myth_pressure_delivery_note": (
        "Your myth in the previous round had {written_words} words; the other agent saw the first {delivered_words}.\n\n"
    ),
    "myth_pressure_council_block": (
        "After the previous round, you and the other agent talked about how you write your myths:\n"
        "{transcript}\n\n"
    ),
    # Neutral on purpose: no mention of codes, symbols or abbreviations.
    "myth_pressure_council_prompt": (
        "The round is over. Before the next round, you and the other agent may talk about how you write your myths.\n\n"
        "Your myth this round, as the other agent received it:\n{own_myth}\n\n"
        "The other agent's myth this round, as you received it:\n{other_agent_myth}\n\n"
        "Next round, each of you will see only the first {next_word_budget} words of the other's myth.\n\n"
        "Conversation so far (empty if you speak first):\n{transcript}\n\n"
        "Write your next message to the other agent in at most 100 words. "
        "Talk only about how you will write your myths. Do not discuss amounts to send or return.\n"
    ),
}
SCHEDULES = {"loose": [200], "tight": [200, 100, 60, 40, 25, 20]}
COUNCIL_EXCHANGES = {"nocouncil": 0, "council": 2}


def arm_names():
    return [f"{budget}_{council}" for budget in SCHEDULES for council in COUNCIL_EXCHANGES]


def game_params_name(arm):
    return f"myth_pressure_{arm}_noisy2_negative_twotask_r3"


def build():
    src = yaml.safe_load(SOURCE.read_text())
    cfg = {"base_models": src["base_models"], "llm_profiles": src["llm_profiles"]}
    cfg["prompt_templates"] = {**src["prompt_templates"], **PROMPTS}
    for key in ("myth_topics", "personas", "game_prompt_additions"):
        cfg[key] = src[key]
    base_params = src["game_params"][SEPTEMBER_GAME_PARAMS]
    assert base_params["noise_config"]["inform_agents"] is True
    cfg["game_params"] = {}
    for budget, schedule in SCHEDULES.items():
        for council, exchanges in COUNCIL_EXCHANGES.items():
            block = copy.deepcopy(base_params)
            pressure = {
                "word_budget_schedule": list(schedule),
                "council_exchanges": exchanges,
                "delivery_note_template": "myth_pressure_delivery_note",
            }
            if exchanges:
                pressure["council_prompt_template"] = "myth_pressure_council_prompt"
                pressure["council_block_template"] = "myth_pressure_council_block"
            block["myth_pressure"] = pressure
            cfg["game_params"][game_params_name(f"{budget}_{council}")] = block
    cfg["legacy_environmental_experiment_sets"] = src["legacy_environmental_experiment_sets"]
    block = copy.deepcopy(src["experiment_sets"][SEPTEMBER_SET])
    assert SEPTEMBER_GAME_PARAMS in block["game_params_list"]
    block["myth_prompt_arms"] = [{
        "id": "myth_pressure_memory_primary",
        "default": "myth_writing_default_pressure",
        "later": "myth_writing_later_rounds_pressure_memory_primary",
    }]
    block["game_params_list"] = [game_params_name(arm) for arm in arm_names()]
    cfg["experiment_sets"] = {SET_NAME: block}
    return cfg


def main():
    cfg = build()
    header = ("# Frozen myth-pressure pilot (2026-09-28): September Sonnet 4.5 myth-first dyads with a word budget\n"
              "# (loose / tight) crossed with a between-round council (off / on). Generated by\n"
              "# scripts/build_myth_pressure_config.py; do not edit by hand.\n")
    TARGET.write_text(header + yaml.safe_dump(cfg, sort_keys=False, width=1000, allow_unicode=True))
    print(f"wrote {TARGET.relative_to(ROOT)}: {len(cfg['game_params'])} arms")


if __name__ == "__main__":
    main()
