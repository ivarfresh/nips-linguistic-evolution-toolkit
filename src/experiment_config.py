import yaml
from itertools import product
from typing import Dict, List

class ExperimentConfig:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_experiment_combinations(self, experiment_name: str) -> List[Dict]:
        """Generate all parameter combinations for an experiment set"""
        exp_set = self.config['experiment_sets'][experiment_name]

        # Resolve "all" references
        models = self._resolve_all(exp_set['models'], 'base_models')
        templates = self._resolve_all(exp_set['templates'], 'prompt_templates')
        personas = self._resolve_all(exp_set['personas'], 'personas')
        task_orders = self._resolve_all(exp_set['task_orders'], 'task_orders',
                                       from_game_params=True)
        # Only expand myth_topics if explicitly set, otherwise use "anything" as default
        myth_topics_spec = exp_set.get("myth_topics", None)
        if myth_topics_spec is None:
            # No myth_topics specified - use "anything" as default topic
            myth_topic_ids = ["anything"]
        else:
            myth_topic_ids = self._resolve_all(myth_topics_spec, "myth_topics")

        # Determine which myth prompts to use. Existing prefix behavior is kept for
        # variants such as instruct_non_coop_*, while exact keys allow additive
        # prompt arms without overwriting the control template.
        myth_prompt_prefix = exp_set.get("myth_prompt_prefix", "")
        myth_default_key = exp_set.get(
            "myth_default_prompt_key",
            f"{myth_prompt_prefix}myth_writing_default",
        )
        myth_later_keys = exp_set.get("myth_later_prompt_keys")
        if myth_later_keys is None:
            myth_later_keys = [
                exp_set.get(
                    "myth_later_prompt_key",
                    f"{myth_prompt_prefix}myth_writing_later_rounds",
                )
            ]
        else:
            myth_later_keys = self._as_list(myth_later_keys)

        myth_prompt_arms = exp_set.get("myth_prompt_arms")
        if myth_prompt_arms is None:
            myth_prompt_arms = [
                {
                    "id": myth_later_key,
                    "default": myth_default_key,
                    "later": myth_later_key,
                }
                for myth_later_key in myth_later_keys
            ]
        else:
            myth_prompt_arms = [
                {
                    "id": arm["id"],
                    "default": arm.get("default", myth_default_key),
                    "later": arm.get("later", f"{myth_prompt_prefix}myth_writing_later_rounds"),
                }
                for arm in myth_prompt_arms
            ]

        repetitions = int(exp_set.get("repetitions", 1))
        replicate_ids = list(range(repetitions))

        # Generate all combinations
        combinations = []
        for model, template, persona, order, myth_topic_id, myth_prompt_arm, replicate_id in product(
            models, templates, personas, task_orders, myth_topic_ids, myth_prompt_arms, replicate_ids
        ):
            # Keep non-myth task orders from multiplying across all topics
            if "myth" not in order and myth_topic_id != myth_topic_ids[0]:
                continue
            # Keep non-myth task orders from multiplying across myth prompt variants
            if "myth" not in order and myth_prompt_arm != myth_prompt_arms[0]:
                continue

            myth_topic = "" if myth_topic_id == "" else self.config["myth_topics"][myth_topic_id]

            active_myth_default_key = myth_prompt_arm["default"]
            active_myth_later_key = myth_prompt_arm["later"]
            myth_default_template = self._get_prompt_template(active_myth_default_key)
            myth_later_template = self._get_prompt_template(active_myth_later_key)
            persona_config = self._get_persona_config(
                persona,
                exp_set.get("system_addition_key"),
            )
            game_prompt_addition = self._get_game_prompt_addition(order)

            # Optional per-set override of game prompt templates, e.g.
            # game_prompt_keys: {trust_game_later_investor: my_variant_key}
            game_prompt_keys = exp_set.get("game_prompt_keys", {})

            def _game_template(name):
                key = game_prompt_keys.get(name, name)
                return self._with_game_prompt_addition(
                    self.config["prompt_templates"].get(key),
                    game_prompt_addition,
                )

            combo = {
                "model": self.config["base_models"][model],
                "template": self.config["prompt_templates"][template],
                "persona": persona_config,
                "persona_key": persona,
                "system_addition_key": exp_set.get("system_addition_key"),
                "task_order": order,
                # Optional top-level output folder override (default: experiment name).
                "output_dir": exp_set.get("output_dir"),
                "game_params": self._get_game_params(exp_set.get("game_params", {})),
                "myth_topic_id": myth_topic_id,
                "myth_topic": myth_topic,
                "replicate_id": replicate_id if repetitions > 1 else None,
                # Add all prompt templates for games and myths
                "trust_game_round1_investor": _game_template("trust_game_round1_investor"),
                "trust_game_round1_trustee": _game_template("trust_game_round1_trustee"),
                "trust_game_later_investor": _game_template("trust_game_later_investor"),
                "trust_game_later_trustee": _game_template("trust_game_later_trustee"),
                "defector_game_instruction": self.config["prompt_templates"].get(
                    "defector_game_instruction"
                ),
                "myth_prompt_arm_id": myth_prompt_arm["id"] if "myth" in order else None,
                "myth_default_prompt_key": active_myth_default_key,
                "myth_later_prompt_key": active_myth_later_key,
                "myth_writing_default": myth_default_template,
                "myth_writing_later_rounds": myth_later_template,
            }
            combinations.append(combo)

        return combinations

    def _resolve_all(self, param, config_key, from_game_params=False):
        if param == "all":
            if from_game_params:
                return self.config['game_parameters'][config_key]
            return list(self.config[config_key].keys())
        return param

    def _get_game_params(self, exp_game_params: Dict) -> Dict:
        """Merge experiment-specific game params with defaults"""
        default_params = self.config['game_parameters'].copy()
        default_params.update(exp_game_params)
        return default_params

    def _as_list(self, value):
        if isinstance(value, list):
            return value
        return [value]

    def _get_prompt_template(self, key: str) -> str:
        try:
            return self.config["prompt_templates"][key]
        except KeyError as exc:
            raise KeyError(f"Unknown prompt template key: {key}") from exc

    def _get_game_prompt_addition(self, task_order) -> str:
        if "game" not in task_order or "myth" not in task_order:
            return ""
        return self.config.get("game_prompt_additions", {}).get("myth_decision_link", "")

    def _with_game_prompt_addition(self, template: str, addition: str) -> str:
        if not template or not addition:
            return template
        return f"{template.rstrip()}\n\n{addition.strip()}\n"

    def _get_persona_config(self, persona_key: str, system_addition_key: str = None) -> Dict:
        persona = dict(self.config["personas"][persona_key])
        if not system_addition_key:
            return persona

        additions = self.config.get("system_prompt_additions", {})
        try:
            addition = additions[system_addition_key]
        except KeyError as exc:
            raise KeyError(f"Unknown system prompt addition key: {system_addition_key}") from exc

        existing = persona.get("system_addition", "")
        persona["system_addition"] = "\n\n".join(part for part in [existing, addition] if part)
        persona["description"] = f"{persona.get('description', persona_key)}_{system_addition_key}"
        return persona
