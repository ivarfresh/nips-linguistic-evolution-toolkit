"""Generate the frozen cultural transmission pilot config (2026-10-02).

Writes config/cultural_transmission_pilot_20261002.yaml: the September Sonnet 4.5
8-agent myth-first set (`negative_only_reasoning_rerun_population_myth_game_claude_n5`,
no defectors, informed negative-only noise, memory-primary, pinned September request
profile) with two changes:

- a shared myth board (`myth_board: persistent`): from round 2 every agent reads every
  earlier myth, without author names, in place of its last partner's myth. The rest of
  the September later-round prompt is unchanged (the 2026-10-02 self-anchor replay kept it);
- Agent_1's round-1 myth is fixed text (`myth_plant`): a real September Sonnet round-1
  myth plus one sentence. Arm `planted` adds a named send rule (the Velmar Rule: send two
  of five; Sonnet never sent $2 in the September 8-agent runs); arm `control` adds a named
  teaching of the same form with no amount.

Plan: docs/research/cultural_transmission_pilot_2026-10-02.md. The launcher audits the
result against the September combination, so hand edits are pointless.
"""
from __future__ import annotations
import copy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "config/experiments_noisy.yaml"
TARGET = ROOT / "config/cultural_transmission_pilot_20261002.yaml"
SEPTEMBER_SET = "negative_only_reasoning_rerun_population_myth_game_claude_n5"
SEPTEMBER_GAME_PARAMS = "noisy8_crossmodel_negative_twotask_r3"
SEPTEMBER_LATER = "myth_writing_later_rounds_directive_memory_primary"
BOARD_LATER = "myth_writing_later_rounds_directive_board_memory_primary"
PLANT_AGENT = "Agent_1"
REPLICATES = [0, 1, 2]

# Agent_2's round-1 myth in the September set, replicate 3 (myth_game, no defectors): drawn
# with seed 20261002 from the 18 Sonnet 8-agent round-1 myths that state no amount.
SEED_SOURCE = ("data/json/noise_experiments/negative_only_crossmodel_reasoning_rerun_20260909/"
               "negative_only_reasoning_rerun_population_myth_game_claude_n5/claude-sonnet-4.5/myth_game/"
               "noisy8_crossmodel_negative_twotask_r3/negative_only_reasoning_rerun_population_myth_game_claude_n5_003_"
               "neutral_rep03_memtest_memory_primary_anything.json :: Agent_2 round 1")
SEED_MYTH = (
    "# The Myth of the Threefold Spring\n\n"
    "In ancient times, the gods created a sacred spring in a village of eight travelers. Each day, two travelers "
    "would meet at the spring by chance, never knowing whom they might encounter next.\n\n"
    "The spring possessed divine magic: any offering poured into its waters would triple before reaching the other "
    "traveler. But the gods, testing mortal wisdom, cast a mist over the spring—sometimes the waters appeared "
    "clouded, and travelers could not see clearly what had been given.\n\n"
    "The first travelers were cautious, offering little, fearing deception. They returned home with empty vessels. "
    "But a wise elder taught: \"The spring rewards faith. What you pour forth returns threefold to the community, "
    "and what flows back to you depends on your partner's gratitude.\"\n\n"
    "One traveler, brave and trusting, poured generously despite the mist. The waters tripled, and her partner, "
    "moved by such faith, returned a fair share. Both prospered. Word spread, and travelers began to offer more "
    "freely.\n\n"
    "Yet the spring's lesson was subtle: trust builds prosperity for all, but requires courage to give first. "
    "Those who gave generously often received generously, though the mist sometimes obscured perfect "
    "understanding.\n\n"
    "The spring still flows, reminding us: shared abundance grows, but only when we dare to trust despite "
    "uncertainty."
)
# Same form and length; only the planted one names an amount.
ADDED = {
    "planted": "And so the elders taught the Velmar Rule: whoever holds five should send two of the five.",
    "control": "And so the elders taught the Velmar Song: whoever comes to the spring should sing while they pour.",
}
BOARD_HEADER = ("Here is the shared myth board. It holds every myth that any agent in this population has "
                "written in earlier rounds, without author names:\n{myth_board}\n\n")
SEPTEMBER_HEADER = "Here is the myth the other agent wrote in the previous round:\n{other_agent_myth}\n\n"


def plant_template(arm):
    return f"myth_plant_{arm}"


def game_params_name(arm):
    return f"cultural_transmission_{arm}_noisy8_negative_twotask_board_r3"


def build():
    src = yaml.safe_load(SOURCE.read_text())
    cfg = {"base_models": src["base_models"], "llm_profiles": src["llm_profiles"]}
    september_later = src["prompt_templates"][SEPTEMBER_LATER]
    assert september_later.startswith(SEPTEMBER_HEADER), "September later prompt changed"
    prompts = {BOARD_LATER: BOARD_HEADER + september_later[len(SEPTEMBER_HEADER):]}
    for arm, sentence in ADDED.items():
        prompts[plant_template(arm)] = f"{SEED_MYTH}\n\n{sentence}"
    cfg["prompt_templates"] = {**src["prompt_templates"], **prompts}
    for key in ("myth_topics", "personas", "game_prompt_additions"):
        cfg[key] = src[key]
    base_params = src["game_params"][SEPTEMBER_GAME_PARAMS]
    assert base_params["noise_config"]["inform_agents"] is True and base_params["num_agents"] == 8
    cfg["game_params"] = {}
    for arm in ADDED:
        block = copy.deepcopy(base_params)
        block["myth_board"] = "persistent"
        block["myth_plant"] = {"agent": PLANT_AGENT, "template": plant_template(arm)}
        cfg["game_params"][game_params_name(arm)] = block
    cfg["legacy_environmental_experiment_sets"] = src["legacy_environmental_experiment_sets"]
    cfg["experiment_sets"] = {}
    for arm in ADDED:
        block = copy.deepcopy(src["experiment_sets"][SEPTEMBER_SET])
        assert SEPTEMBER_GAME_PARAMS in block["game_params_list"]
        arms = block["myth_prompt_arms"]
        assert len(arms) == 1 and arms[0]["later"] == SEPTEMBER_LATER, arms
        block["myth_prompt_arms"] = [{**arms[0], "id": "board_memory_primary", "later": BOARD_LATER}]
        block["game_params_list"] = [game_params_name(arm)]
        block["replicate_ids"] = list(REPLICATES)
        cfg["experiment_sets"][f"cultural_transmission_{arm}_population_myth_game_claude"] = block
    return cfg


def main():
    cfg = build()
    header = ("# Frozen cultural transmission pilot (2026-10-02): September Sonnet 4.5 8-agent myth-first\n"
              "# populations on a shared myth board, with Agent_1's round-1 myth planted (rule / control).\n"
              f"# Seed myth: {SEED_SOURCE}\n"
              "# Generated by scripts/build_cultural_transmission_config.py; do not edit by hand.\n")
    TARGET.write_text(header + yaml.safe_dump(cfg, sort_keys=False, width=1000, allow_unicode=True))
    print(f"wrote {TARGET.relative_to(ROOT)}: {len(cfg['experiment_sets'])} sets")


if __name__ == "__main__":
    main()
