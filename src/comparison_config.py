"""Check declared experimental contrasts after expanding their configurations."""

from copy import deepcopy

from src.experiment_condition import ConditionMismatchError, check_conditions
from src.llm_settings import resolve_llm_settings


LABEL_FIELDS = {
    "game_params_name", "game_prompt_addition_id", "initial_system_template_name",
    "myth_default_prompt_key", "myth_later_prompt_key", "myth_prompt_arm_id",
    "myth_prompt_template_names", "myth_topic_id", "round_prompt_template_names",
    "run_number", "template_name", "output_dir", "persona_key", "system_addition_key",
    "execution_provenance", "llm_settings",
}


def resolved_comparison_inputs(combo, settings):
    result = {key: deepcopy(value) for key, value in combo.items() if key not in LABEL_FIELDS}
    result["llm_settings"] = settings.as_dict() if settings is not None else None
    return result


def validate_config_comparisons(config, experiment_name=None, environ=None):
    for name, declaration in config.config.get("comparison_sets", {}).items():
        sets = declaration.get("experiment_sets", [])
        if len(sets) < 2 or len(set(sets)) != len(sets):
            raise ConditionMismatchError(f"Comparison {name!r} needs distinct experiment sets")
        if experiment_name is not None and experiment_name not in sets:
            continue
        by_replicate = {}
        expected_replicates = None
        for set_name in sets:
            settings = resolve_llm_settings(config.config["experiment_sets"][set_name], set_name, environ=environ)
            combinations = config.get_experiment_combinations(set_name)
            replicates = []
            for combo in combinations:
                replicate = combo.get("replicate_id")
                replicates.append(replicate)
                by_replicate.setdefault(replicate, []).append(resolved_comparison_inputs(combo, settings))
            if len(replicates) != len(set(replicates)):
                raise ConditionMismatchError(f"Comparison {name!r} requires one cell per set and replicate")
            if expected_replicates is not None and set(replicates) != expected_replicates:
                raise ConditionMismatchError(f"Comparison {name!r} has unmatched replicate identities")
            expected_replicates = set(replicates)
        for conditions in by_replicate.values():
            check_conditions(conditions, declaration.get("allowed_differences"))
