"""Record and compare the actual inputs to a scientific run."""

import hashlib
import json
from pathlib import Path


CONDITION_VERSION = 1
FULL_STATE_KEYS = {"agents", "conversation_history", "game_data", "task_order"}
NON_FINAL_SUFFIXES = (".results.json", ".checkpoint.json", ".error.json")
GAME_FIELDS = (
    "endowment", "multiplier", "multiplier_distribution", "personas",
    "system_prompt_template", "round1_investor_template", "round1_trustee_template",
    "later_investor_template", "later_trustee_template", "noise_config",
    "noise_semantics", "other_player_names", "myth_injection_mode",
    "history_policy", "self_history_window", "coplayer_history_window",
    "population_history_window", "show_agent_names", "defector_ratio",
    "defector_prompt_template", "defector_action_policy", "defector_myth_policy",
    "defector_role_visible_to_self", "game_prompt_addition", "pairing_mode",
    "prompt_regime", "punishment_enabled", "punishment_budget",
    "punishment_effect_multiplier", "punishment_prompt_variant", "decision_format",
)
SEED_FIELDS = ("pairing_seed", "noise_seed", "run_seed", "defector_seed")
LLM_FIELDS = {
    "provider", "provider_model", "reasoning", "reasoning_param",
    "temperature_sent", "temperature_value", "max_output_tokens",
}


class ConditionMismatchError(ValueError):
    """The requested comparison or resume changes an undeclared condition."""


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def build_condition(game, myth_writer, runtime_metadata, simulation, replicate_id=None):
    protocol = {
        "game_type": f"{type(game).__module__}.{type(game).__name__}",
        "game": {name: getattr(game, name, None) for name in GAME_FIELDS},
        "myth": {
            name: getattr(myth_writer, name, None)
            for name in ("myth_topic", "round1_template", "later_rounds_template")
        },
        "simulation": simulation,
        "game_retry": {"policy": "corrective_v1", "attempts": getattr(game, "GAME_RETRY_ATTEMPTS", 1)},
        "myth_retry": {"policy": "task_boundary_v1", "attempts": 2},
    }
    pool = getattr(game, "_shuffled_myth_pool", None)
    protocol["game"]["shuffled_myth_pool_sha256"] = digest(pool) if pool is not None else None
    condition = {
        "version": CONDITION_VERSION,
        "llm": runtime_metadata.get("llm_settings_effective"),
        "protocol": protocol,
        "replicate": {
            "replicate_id": replicate_id,
            **{name: getattr(game, name, None) for name in SEED_FIELDS},
            "defector_agent_ids": getattr(game, "requested_defector_agent_ids", None),
        },
        "implementation": {
            str(path.relative_to(Path(__file__).resolve().parent.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ("src", "games")
            for path in sorted((Path(__file__).resolve().parent.parent / folder).glob("*.py"))
        },
    }
    return json.loads(json.dumps(condition, allow_nan=False))


def validate_condition(condition):
    if not isinstance(condition, dict) or condition.get("version") != CONDITION_VERSION:
        raise ConditionMismatchError("Missing or unsupported experiment_condition version")
    effective = condition.get("llm")
    if not isinstance(effective, dict) or not LLM_FIELDS.issubset(effective):
        raise ConditionMismatchError("Condition lacks resolved LLM request settings")
    if any(not isinstance(effective[name], str) or not effective[name].strip() for name in ("provider", "provider_model")):
        raise ConditionMismatchError("Condition lacks provider/model identity")
    if not isinstance(effective["temperature_sent"], bool):
        raise ConditionMismatchError("temperature_sent must be boolean")
    if effective["temperature_sent"] != (effective["temperature_value"] is not None):
        raise ConditionMismatchError("Temperature value contradicts its omission policy")
    if effective["reasoning"] not in ("off", "minimal", "low", "medium", "high"):
        raise ConditionMismatchError("Condition has unknown reasoning settings")
    cap = effective["max_output_tokens"]
    if cap is not None and (type(cap) is not int or cap <= 0):
        raise ConditionMismatchError("Output cap must be a positive integer or omitted")
    protocol = condition.get("protocol")
    if not isinstance(protocol, dict) or not {"game", "myth", "simulation", "game_retry", "myth_retry", "game_type"}.issubset(protocol):
        raise ConditionMismatchError("Condition lacks protocol settings")
    if not isinstance(condition.get("replicate"), dict):
        raise ConditionMismatchError("Condition lacks replicate identity")
    if not isinstance(condition.get("implementation"), dict) or not condition["implementation"]:
        raise ConditionMismatchError("Condition lacks implementation identity")
    digest(condition)
    return condition


def condition_from_run(data):
    metadata = data.get("run_metadata")
    if not isinstance(metadata, dict):
        raise ConditionMismatchError("Run metadata is missing or invalid")
    condition = validate_condition(metadata.get("experiment_condition"))
    if metadata.get("condition_sha256") != digest(condition):
        raise ConditionMismatchError("Run condition digest is missing or inconsistent")
    if metadata.get("llm_settings_effective") != condition["llm"]:
        raise ConditionMismatchError("Run metadata contradicts its condition")
    for key in ("provider", "provider_model"):
        metadata_key = "llm_provider" if key == "provider" else key
        if metadata.get(metadata_key) != condition["llm"][key]:
            raise ConditionMismatchError(f"Run metadata contradicts condition {key}")
    for agent in data.get("agents", {}).values():
        for event in agent.get("interaction_history", []):
            response = event.get("response") or {}
            if not response or response.get("response_source") == "scripted":
                continue
            usage = response.get("usage") or {}
            if usage.get("request_settings") != condition["llm"]:
                raise ConditionMismatchError("Per-call request settings differ from the run condition")
            if not usage.get("finish_reason"):
                raise ConditionMismatchError("Per-call finish reason is missing")
    return condition


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            name = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result.append(name)
            else:
                result.extend(differences(left[key], right[key], name))
        return result
    return [] if left == right else [prefix]


def check_conditions(conditions, allowed_differences=None):
    allowed = allowed_differences or {}
    if not isinstance(allowed, dict) or any(not isinstance(path, str) or not isinstance(reason, str) or not reason.strip() for path, reason in allowed.items()):
        raise ConditionMismatchError("Every allowed difference needs a written reason")
    if any(path in {"", "llm", "protocol", "protocol.game", "protocol.myth", "protocol.simulation"} for path in allowed):
        raise ConditionMismatchError("Declare individual differing fields, not whole settings blocks")
    conditions = list(conditions)
    if not conditions:
        raise ConditionMismatchError("No run conditions supplied")
    baseline = conditions[0]
    all_differences = set()
    for condition in conditions[1:]:
        all_differences.update(differences(baseline, condition))
    unexpected = sorted(path for path in all_differences if not any(path == accepted or path.startswith(accepted + ".") for accepted in allowed))
    if unexpected:
        raise ConditionMismatchError("Undeclared condition differences: " + ", ".join(unexpected))
    return sorted(all_differences)


def read_final_run(path):
    path = Path(path)
    if path.suffix != ".json" or path.name.endswith(NON_FINAL_SUFFIXES):
        raise ConditionMismatchError(f"Not a final run JSON: {path}")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not FULL_STATE_KEYS.issubset(data):
        raise ConditionMismatchError(f"Not a full-state run: {path}")
    return data


def comparison_condition(data, legacy_reason=None):
    metadata = data.get("run_metadata") or {}
    if (metadata.get("experiment_condition") or {}).get("llm") is not None:
        return condition_from_run(data)
    if not isinstance(legacy_reason, str) or not legacy_reason.strip():
        raise ConditionMismatchError("Historical run lacks a complete condition; provide legacy_reason for an explicitly exploratory comparison")
    effective = metadata.get("llm_settings_effective") or {}
    fields = (
        "num_agents", "num_turns", "memory_capacity", "chat_memory_mode",
        "noise_config", "noise_semantics", "history_policy", "self_history_window",
        "coplayer_history_window", "population_history_window", "show_agent_names",
        "defector_ratio_requested", "defector_action_policy", "defector_myth_policy",
        "defector_role_visible_to_self", "punishment_enabled", "punishment_budget",
        "punishment_effect_multiplier", "punishment_prompt_variant", "prompt_regime",
        "seed_myth", "seed_user_prompt", "seed_reinject", "myth_injection_mode",
        "game_prompt_addition", "myth_default_prompt_key", "myth_later_prompt_key",
        "decision_format", "pairing_mode", "other_player_names",
    )
    return {
        "version": 0,
        "llm": {
            "provider": metadata.get("llm_provider", "unrecorded"),
            "provider_model": metadata.get("provider_model", "unrecorded"),
            "reasoning": effective.get("reasoning", metadata.get("thinking_level", "unrecorded")),
            "temperature": effective.get("temperature_value") if effective else metadata.get("temperature", "unrecorded"),
            "max_output_tokens": metadata.get("max_output_tokens", "unrecorded"),
        },
        "protocol": {
            **{field: metadata.get(field, "unrecorded") for field in fields},
            "task_order": data.get("task_order"),
            "system_prompts": {name: agent.get("system_prompt") for name, agent in data.get("agents", {}).items()},
        },
        "implementation": {name: metadata.get(name, "unrecorded") for name in ("code_commit", "code_dirty")},
        "replicate": {name: metadata.get(name, "unrecorded") for name in ("replicate_id", *SEED_FIELDS)},
    }


def validate_output_provenance(document):
    if not isinstance(document, dict) or document.get("provenance_version") != 2:
        raise ConditionMismatchError("Output provenance must use version 2")
    runs = document.get("runs")
    if not isinstance(runs, list) or not runs or document.get("n_runs") != len(runs):
        raise ConditionMismatchError("Output provenance has an invalid run count")
    identities = set()
    for run in runs:
        if not run.get("path") or not isinstance(run.get("sha256"), str) or len(run["sha256"]) != 64:
            raise ConditionMismatchError("Every input run needs a path and SHA-256")
        if run["sha256"] in identities:
            raise ConditionMismatchError("Duplicate input run in output provenance")
        identities.add(run["sha256"])
        if run.get("condition", {}).get("version") == 0:
            if not isinstance(document.get("legacy_reason"), str) or not document["legacy_reason"].strip():
                raise ConditionMismatchError("Historical output inputs require a legacy_reason")
            if not {"llm", "protocol", "replicate", "implementation"}.issubset(run["condition"]):
                raise ConditionMismatchError("Historical input lacks its recorded condition fields")
        else:
            validate_condition(run.get("condition"))
    observed = check_conditions([run["condition"] for run in runs], document.get("allowed_differences"))
    if document.get("observed_differences") != observed:
        raise ConditionMismatchError("Output provenance misstates its observed differences")
