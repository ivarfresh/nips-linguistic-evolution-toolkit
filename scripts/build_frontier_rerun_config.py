"""Generate the frozen frontier-rerun config from the September no-defector sets.

Writes config/frontier_rerun_20260918.yaml: the September prompts, personas, topics and
the four no-defector game-parameter blocks, plus 36 experiment sets (6 shapes x 6 arms)
that differ from the September `negative_only_reasoning_rerun_*` sets only in the model
and its pinned request profile, and (from 2026-09-28) three frontier mixed-model
eight-agent sets (2 Gemini 3.1 Pro + 3 GPT-6 Sol + 3 Opus 5.5, one per task order). Re-run to regenerate; the launcher audits the result
against the September combinations, so hand edits are pointless.
"""
from __future__ import annotations
import copy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "config/experiments_noisy.yaml"
TARGET = ROOT / "config/frontier_rerun_20260918.yaml"

NEW_MODELS = {
    "claude_opus_5": "anthropic/claude-opus-5",
    "claude_opus_5_5": "anthropic/claude-opus-5-5",
    "gpt56_sol": "openai/gpt-5.6-sol",
    "gpt6_sol": "openai/gpt-6-sol",
}
PROFILES = {
    "september18_opus5": {
        "provider": "anthropic",
        "reasoning": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
        "temperature": "default",
        "max_output_tokens": 64000,
    },
    # Opus 5.5 (added 2026-09-23): identical request to Opus 5. Effort is pinned to high on
    # purpose; Opus 5.5 would otherwise default to medium, one level below Opus 5.
    "september23_opus55": {
        "provider": "anthropic",
        "reasoning": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
        "temperature": "default",
        "max_output_tokens": 64000,
    },
    "september18_gemini31pro": {
        "provider": "google",
        "reasoning": {"thinkingConfig": {"thinkingLevel": "high"}},
        "temperature": 0.8,
        "max_output_tokens": 65536,
    },
    "september18_sol_high": {
        "provider": "openai",
        "reasoning": {"reasoning_effort": "high"},
        "temperature": "default",
        "max_output_tokens": 128000,
    },
    # GPT-6 Sol (added 2026-09-28): the Sol-high request unchanged. On GPT-6 Sol, high is no
    # longer the top effort level (xhigh and max exist); it is kept at high to match GPT-5.6 Sol.
    "september28_sol6_high": {
        "provider": "openai",
        "reasoning": {"reasoning_effort": "high"},
        "temperature": "default",
        "max_output_tokens": 128000,
    },
    "september18_sol_none": {
        "provider": "openai",
        "reasoning": {"reasoning_effort": "none"},
        "temperature": "default",
        "max_output_tokens": 128000,
    },
}
ARMS = {
    "opus5": ("claude_opus_5", "september18_opus5"),
    "gemini31pro": ("gemini_3_1_pro", "september18_gemini31pro"),
    "sol_high": ("gpt56_sol", "september18_sol_high"),
    "sol_none": ("gpt56_sol", "september18_sol_none"),
    "opus55": ("claude_opus_5_5", "september23_opus55"),
    "sol6_high": ("gpt6_sol", "september28_sol6_high"),
}
# Frontier mixed-model populations (2026-09-28): the current frontier arm of each family.
# Agents in contiguous blocks, smallest family first: Agent_1-2 Gemini, Agent_3-5 Sol, Agent_6-8 Opus.
MIXED_POPULATION = [("gemini31pro", 2), ("sol6_high", 3), ("opus55", 3)]
MIXED_NAME = "gemini2_sol3_opus3"
# Main frontier mixed runs (2026-09-28, Ivar): the 2026-09-18 main frontier trio (Opus 5,
# Gemini 3.1 Pro, GPT-5.6 Sol at effort high). Populations: Agent_1-2 Gemini, Agent_3-5 Opus 5,
# Agent_6-8 Sol. Dyads copy the September mixed-dyad design (D010): the first-named arm is
# Agent_1 (round-1 sender) in replicates 0/2/4, the reversed pair in replicates 1/3/5.
MAIN_MIXED_POPULATION = [("gemini31pro", 2), ("opus5", 3), ("sol_high", 3)]
MAIN_MIXED_NAME = "gemini2_opus3_sol3"
MAIN_MIXED_DYADS = {  # set-name suffix -> (Agent_1 arm, Agent_2 arm, replicate ids)
    "opus5_sol": ("opus5", "sol_high", [0, 2, 4]), "sol_opus5": ("sol_high", "opus5", [1, 3, 5]),
    "opus5_gemini": ("opus5", "gemini31pro", [0, 2, 4]), "gemini_opus5": ("gemini31pro", "opus5", [1, 3, 5]),
    "gemini_sol": ("gemini31pro", "sol_high", [0, 2, 4]), "sol_gemini": ("sol_high", "gemini31pro", [1, 3, 5]),
}
# Frontier defector runs (2026-10-01, Ivar): can permanent defectors open room below the
# ceiling for a myth effect, and does the mixed-vs-parts comparison of Table 1 hold for frontier
# models? Main-frontier Opus 5 and GPT-5.6 Sol (Gemini 3.1 Pro is ceiling-locked) in a 4 + 4 mix
# and in single-model populations. Agent_4 and Agent_8 are permanent forced-zero defectors (25%,
# the September defectors25 settings); in the mix that is one defector per family.
DEFECTOR_POPULATIONS = {
    "opus4_sol4_d2": [("opus5", 4), ("sol_high", 4)],
    "opus8_d2": [("opus5", 8)],
    "sol8_d2": [("sol_high", 8)],
}
DEFECTOR_PILOT_NAME = "opus4_sol4_d2"
DEFECTOR_PILOT_POPULATION = DEFECTOR_POPULATIONS[DEFECTOR_PILOT_NAME]
DEFECTOR_PILOT_IDS = ["Agent_4", "Agent_8"]
# shape -> (September set to copy, September defectors25 block, frontier block with explicit ids)
DEFECTOR_PILOT_SHAPES = {
    "population_game": ("negative_only_reasoning_rerun_population_game_claude_n5", "noisy8_crossmodel_negative_defectors25_game_r3",
                        "frontier_noisy8_negative_defectors25_split_game_r3"),
    "population_game_myth": ("negative_only_reasoning_rerun_population_game_myth_claude_n5", "noisy8_crossmodel_negative_defectors25_twotask_r3",
                             "frontier_noisy8_negative_defectors25_split_twotask_r3"),
    "population_myth_game": ("negative_only_reasoning_rerun_population_myth_game_claude_n5", "noisy8_crossmodel_negative_defectors25_twotask_r3",
                             "frontier_noisy8_negative_defectors25_split_twotask_r3"),
}
# The 2026-10-01 pilot ran replicates 0-2 of the mix in game and myth_game; the n=5 sets add the rest.
DEFECTOR_PILOT_REPLICATES = {"population_game": [0, 1, 2], "population_myth_game": [0, 1, 2]}
DEFECTOR_REPLICATES = [0, 1, 2, 3, 4]


def defector_sets():
    """(set name, composition, shape, replicate ids) for every frontier defector set."""
    out = []
    for shape in DEFECTOR_PILOT_SHAPES:
        if shape in DEFECTOR_PILOT_REPLICATES:
            out.append((f"frontier_defector_pilot_{shape}_{DEFECTOR_PILOT_NAME}_n3", DEFECTOR_PILOT_NAME, shape, DEFECTOR_PILOT_REPLICATES[shape]))
        for comp in DEFECTOR_POPULATIONS:
            done = DEFECTOR_PILOT_REPLICATES.get(shape, []) if comp == DEFECTOR_PILOT_NAME else []
            out.append((f"frontier_defector_{shape}_{comp}_n5", comp, shape, [r for r in DEFECTOR_REPLICATES if r not in done]))
    return out
# Frontier myth board (2026-10-02, Ivar): the opus4_sol4_d2 defector mix, but from round 2 every
# agent reads every myth written so far (a shared, persistent board inspired by the 2026 Hugging
# Face incident) instead of its previous partner's myth. The later-round prompt is the September
# memory-primary prompt with the partner-myth line replaced by the board; nothing else changes.
BOARD_TEMPLATE_NAME = "myth_writing_later_rounds_board_memory_primary"
BOARD_TEMPLATE = (
    "Here is the shared myth board. It holds every myth that any agent in this population has "
    "written in earlier rounds, without author names:\n{myth_board}\n\n"
    "Write your own myth. Use the myth you wrote in the previous round as inspiration, but adapt it "
    "in your own way. The myth should reflect how the game should be played, drawing on the game you "
    "have been playing up to this point.\n\nWrite 200 words. Format exactly:\nMyth: [your story here].\n"
)
BOARD_PARAMS = "frontier_noisy8_negative_defectors25_split_twotask_board_r3"
# shape -> replicate ids of the board pilot; the n=5 sets add the rest.
BOARD_PILOT_REPLICATES = {"population_myth_game": [0, 1, 2]}
BOARD_SHAPES = ("population_game_myth", "population_myth_game")


def board_sets():
    """(set name, shape, replicate ids) for every frontier myth-board set."""
    out = []
    for shape in BOARD_SHAPES:
        if shape in BOARD_PILOT_REPLICATES:
            out.append((f"frontier_board_pilot_{shape}_{DEFECTOR_PILOT_NAME}_n3", shape, BOARD_PILOT_REPLICATES[shape]))
        done = BOARD_PILOT_REPLICATES.get(shape, [])
        out.append((f"frontier_board_{shape}_{DEFECTOR_PILOT_NAME}_n5", shape, [r for r in DEFECTOR_REPLICATES if r not in done]))
    return out


# Cross-tier myth-board pilot (2026-10-02, Ivar): the main frontier mix with GPT-5 Nano, the
# mid-tier model that withholds, in place of GPT-5.6 Sol: Agent_1-2 Gemini 3.1 Pro, Agent_3-5
# Opus 5, Agent_6-8 GPT-5 Nano (September profile). No scripted defectors. Game only, myth -> game
# with partner myths, and myth -> game with the persistent board; replicates 0-2.
CROSSTIER_POPULATION = [("gemini31pro", 2), ("opus5", 3), ("nano_sept", 3)]
# GPT-5 Nano keeps its September request profile (september8_gpt, copied from experiments_noisy.yaml).
CROSSTIER_ARMS = {**ARMS, "nano_sept": ("gpt5_nano", "september8_gpt")}
CROSSTIER_NAME = "gemini2_opus3_nano3"
CROSSTIER_BOARD_PARAMS = "frontier_noisy8_negative_twotask_board_r3"
CROSSTIER_REPLICATES = [0, 1, 2]


def crosstier_sets():
    """(set name, shape, board) for the cross-tier pilot."""
    return [(f"frontier_crosstier_pilot_population_game_{CROSSTIER_NAME}_n3", "population_game", False),
            (f"frontier_crosstier_pilot_population_myth_game_{CROSSTIER_NAME}_n3", "population_myth_game", False),
            (f"frontier_crosstier_board_pilot_population_myth_game_{CROSSTIER_NAME}_n3", "population_myth_game", True)]


# shape -> (September set to copy, the single no-defector game-params block to keep)
SHAPES = {
    "dyad_game": ("negative_only_reasoning_rerun_dyad_game_claude_n5", "noisy2_crossmodel_negative_game_r3"),
    "dyad_game_myth": ("negative_only_reasoning_rerun_dyad_game_myth_claude_n5", "noisy2_crossmodel_negative_twotask_r3"),
    "dyad_myth_game": ("negative_only_reasoning_rerun_dyad_myth_game_claude_n5", "noisy2_crossmodel_negative_twotask_r3"),
    "population_game": ("negative_only_reasoning_rerun_population_game_claude_n5", "noisy8_crossmodel_negative_game_r3"),
    "population_game_myth": ("negative_only_reasoning_rerun_population_game_myth_claude_n5", "noisy8_crossmodel_negative_twotask_r3"),
    "population_myth_game": ("negative_only_reasoning_rerun_population_myth_game_claude_n5", "noisy8_crossmodel_negative_twotask_r3"),
}


def build():
    src = yaml.safe_load(SOURCE.read_text())
    cfg = {}
    cfg["base_models"] = {**src["base_models"], **NEW_MODELS}
    cfg["llm_profiles"] = {**src["llm_profiles"], **PROFILES}
    for key in ("prompt_templates", "myth_topics", "personas", "game_prompt_additions"):
        cfg[key] = src[key]
    keep = sorted({gp for _, gp in SHAPES.values()})
    cfg["game_params"] = {name: src["game_params"][name] for name in keep}
    cfg["legacy_environmental_experiment_sets"] = src["legacy_environmental_experiment_sets"]
    sets = {}
    for shape, (sept_set, game_params) in SHAPES.items():
        base = src["experiment_sets"][sept_set]
        assert game_params in base["game_params_list"], (shape, game_params)
        for arm, (model_key, profile) in ARMS.items():
            block = copy.deepcopy(base)
            block["models"] = [model_key]
            block["game_params_list"] = [game_params]
            block["llm_settings"] = copy.deepcopy(PROFILES[profile])
            sets[f"frontier_{shape}_{arm}_n5"] = block
        if shape.startswith("population_"):
            block = copy.deepcopy(base)
            del block["models"], block["llm_settings"]
            block["agent_models"] = [ARMS[arm][0] for arm, count in MIXED_POPULATION for _ in range(count)]
            block["game_params_list"] = [game_params]
            block["llm_settings_by_model"] = {ARMS[arm][0]: copy.deepcopy(PROFILES[ARMS[arm][1]]) for arm, _ in MIXED_POPULATION}
            sets[f"frontier_mixed_{shape}_{MIXED_NAME}_n5"] = block
            block = copy.deepcopy(base)
            del block["models"], block["llm_settings"]
            block["agent_models"] = [ARMS[arm][0] for arm, count in MAIN_MIXED_POPULATION for _ in range(count)]
            block["game_params_list"] = [game_params]
            block["llm_settings_by_model"] = {ARMS[arm][0]: copy.deepcopy(PROFILES[ARMS[arm][1]]) for arm, _ in MAIN_MIXED_POPULATION}
            sets[f"frontier_main_mixed_{shape}_{MAIN_MIXED_NAME}_n5"] = block
        else:
            for pair, (first, second, replicate_ids) in MAIN_MIXED_DYADS.items():
                block = copy.deepcopy(base)
                del block["models"], block["llm_settings"]
                block["agent_models"] = [ARMS[first][0], ARMS[second][0]]
                block["game_params_list"] = [game_params]
                block["replicate_ids"] = replicate_ids
                block["llm_settings_by_model"] = {ARMS[a][0]: copy.deepcopy(PROFILES[ARMS[a][1]]) for a in (first, second)}
                sets[f"frontier_main_mixed_{shape}_{pair}_n3"] = block
    for shape, (sept_set, sept_params, pilot_params) in DEFECTOR_PILOT_SHAPES.items():
        assert sept_params in src["experiment_sets"][sept_set]["game_params_list"], (shape, sept_params)
        cfg["game_params"][pilot_params] = {**copy.deepcopy(src["game_params"][sept_params]), "defector_agent_ids": list(DEFECTOR_PILOT_IDS)}
    for name, comp, shape, replicate_ids in defector_sets():
        sept_set, _, pilot_params = DEFECTOR_PILOT_SHAPES[shape]
        block = copy.deepcopy(src["experiment_sets"][sept_set])
        arms = DEFECTOR_POPULATIONS[comp]
        if len(arms) == 1:  # single-model population: the ordinary single-model set form
            block["models"] = [ARMS[arms[0][0]][0]]
            block["llm_settings"] = copy.deepcopy(PROFILES[ARMS[arms[0][0]][1]])
        else:
            del block["models"], block["llm_settings"]
            block["agent_models"] = [ARMS[arm][0] for arm, count in arms for _ in range(count)]
            block["llm_settings_by_model"] = {ARMS[arm][0]: copy.deepcopy(PROFILES[ARMS[arm][1]]) for arm, _ in arms}
        block["game_params_list"] = [pilot_params]
        block["replicate_ids"] = replicate_ids
        sets[name] = block
    template = cfg["prompt_templates"]["myth_writing_later_rounds_directive_memory_primary"]
    first, rest = template.split("\n\n", 1)
    assert first == "Here is the myth the other agent wrote in the previous round:\n{other_agent_myth}", first
    assert BOARD_TEMPLATE.endswith(rest), "board prompt must keep the September instructions"
    cfg["prompt_templates"][BOARD_TEMPLATE_NAME] = BOARD_TEMPLATE
    cfg["game_params"][BOARD_PARAMS] = {**copy.deepcopy(cfg["game_params"]["frontier_noisy8_negative_defectors25_split_twotask_r3"]),
                                        "myth_board": "persistent"}
    for name, shape, replicate_ids in board_sets():
        block = copy.deepcopy(sets[f"frontier_defector_{shape}_{DEFECTOR_PILOT_NAME}_n5"])
        arm = block["myth_prompt_arms"]
        assert len(arm) == 1 and arm[0]["later"] == "myth_writing_later_rounds_directive_memory_primary", arm
        block["myth_prompt_arms"] = [{**arm[0], "id": "board_memory_primary", "later": BOARD_TEMPLATE_NAME}]
        block["game_params_list"] = [BOARD_PARAMS]
        block["replicate_ids"] = replicate_ids
        sets[name] = block
    cfg["game_params"][CROSSTIER_BOARD_PARAMS] = {**copy.deepcopy(cfg["game_params"]["noisy8_crossmodel_negative_twotask_r3"]),
                                                  "myth_board": "persistent"}
    for name, shape, board in crosstier_sets():
        block = copy.deepcopy(sets[f"frontier_main_mixed_{shape}_{MAIN_MIXED_NAME}_n5"])
        block["agent_models"] = [CROSSTIER_ARMS[arm][0] for arm, count in CROSSTIER_POPULATION for _ in range(count)]
        block["llm_settings_by_model"] = {CROSSTIER_ARMS[arm][0]: copy.deepcopy(cfg["llm_profiles"][CROSSTIER_ARMS[arm][1]])
                                          for arm, _ in CROSSTIER_POPULATION}
        block["replicate_ids"] = list(CROSSTIER_REPLICATES)
        if board:
            arm = block["myth_prompt_arms"]
            assert len(arm) == 1 and arm[0]["later"] == "myth_writing_later_rounds_directive_memory_primary", arm
            block["myth_prompt_arms"] = [{**arm[0], "id": "board_memory_primary", "later": BOARD_TEMPLATE_NAME}]
            block["game_params_list"] = [CROSSTIER_BOARD_PARAMS]
        sets[name] = block
    cfg["experiment_sets"] = sets
    return cfg


def main():
    cfg = build()
    header = "# Frozen frontier rerun (2026-09-18): September no-defector protocol on Claude Opus 5,\n# Gemini 3.1 Pro Preview, GPT-5.6 Sol (effort high and none), from 2026-09-23 Claude Opus 5.5 and, from\n# 2026-09-28, GPT-6 Sol (effort high) plus the Gemini 3.1 Pro / GPT-6 Sol / Opus 5.5 mixed populations and the main-frontier (Opus 5, Gemini 3.1 Pro,\n# GPT-5.6 Sol) mixed dyads and populations, from 2026-10-01 the Opus 5 / Sol defector populations, and from 2026-10-02 the myth-board runs and the cross-tier (GPT-5 Nano) board pilot. Generated by\n# scripts/build_frontier_rerun_config.py; do not edit by hand.\n"
    TARGET.write_text(header + yaml.safe_dump(cfg, sort_keys=False, width=1000, allow_unicode=True))
    print(f"wrote {TARGET.relative_to(ROOT)}: {len(cfg['experiment_sets'])} sets, {len(cfg['game_params'])} game-param blocks")


if __name__ == "__main__":
    main()
