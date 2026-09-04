"""Experiment-level LLM settings: provider route, reasoning level, temperature.

Why this exists (researchlog 2026-09-04): provider routing used to be decided
by whichever API keys sat in the runner's ``.env`` (``LLM_PROVIDER=auto``), and
each route applied its own reasoning/temperature defaults. The same model slug
therefore ran under different regimes on different machines, and the cross-model
defector set compared a non-thinking Claude with a minimal-reasoning GPT and a
medium-thinking Gemini without anyone choosing that.

The fix is config over env:

* every experiment set declares an ``llm_settings`` block in the YAML config;
* the batch runners refuse to start without it (fail closed);
* environment variables may override it, but overrides are recorded in
  ``run_metadata`` instead of being silent defaults;
* the effective values (what was actually sent) are written to
  ``run_metadata`` and every per-call ``usage`` dict carries ``finish_reason``.

Schema (per experiment set)::

    llm_settings:
      provider: direct | openrouter
      reasoning: off | minimal | low | medium | high
      temperature: default | <float>

``temperature: default`` means "send no temperature parameter" (the vendor's
default, 1.0 on every current model). A float is sent only when the model
honours it; models known to reject or ignore temperature (GPT-5 family,
Gemini 3.6+ Flash, Claude Opus 4.7+, Fable) fail closed instead of silently
running at the vendor default while the metadata claims otherwise.

Override environment variables (recorded when applied):

    LLM_PROVIDER      -> provider   (direct | openrouter; legacy per-vendor
                                     modes openai/anthropic/google are accepted
                                     and recorded as-is)
    LLM_REASONING     -> reasoning
    LLM_TEMPERATURE   -> temperature ("default" or a float)

The legacy per-vendor knobs (``OPENAI_REASONING_EFFORT``,
``GEMINI_THINKING_LEVEL``, ``OPENROUTER_REASONING_EFFORT``) are ignored once an
``llm_settings`` block is active; a warning is printed if they are set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

VALID_PROVIDERS = ("direct", "openrouter")
LEGACY_PROVIDER_MODES = ("openai", "anthropic", "google", "gemini")
VALID_REASONING = ("off", "minimal", "low", "medium", "high")
REQUIRED_KEYS = ("provider", "reasoning", "temperature")
LEGACY_ENV_KNOBS = (
    "OPENAI_REASONING_EFFORT",
    "GEMINI_THINKING_LEVEL",
    "OPENROUTER_REASONING_EFFORT",
)


class LLMSettingsError(ValueError):
    """Raised when an experiment set's llm_settings block is missing or invalid."""


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    reasoning: str
    temperature: Optional[float]  # None == vendor default (parameter omitted)
    source: str = "config"
    overrides: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "reasoning": self.reasoning,
            "temperature": "default" if self.temperature is None else self.temperature,
        }

    @property
    def provider_mode(self) -> str:
        """Value to hand to create_llm_client(provider=...)."""
        return self.provider


def missing_block_message(experiment_name: str, config_path: str = "") -> str:
    where = f" in {config_path}" if config_path else ""
    return (
        f"Experiment set {experiment_name!r}{where} has no `llm_settings` block. "
        "Every experiment set must pin its LLM regime so the provider route and "
        "reasoning level do not depend on the runner's .env. Add:\n"
        "  llm_settings:\n"
        "    provider: direct        # or openrouter\n"
        "    reasoning: off          # off | minimal | low | medium | high\n"
        "    temperature: default    # or a float the model honours\n"
        "See src/llm_settings.py and docs/architecture/design-constraints.md §6."
    )


def _parse_temperature(value: Any, origin: str) -> Optional[float]:
    if value is None:
        raise LLMSettingsError(
            f"{origin}: temperature must be 'default' or a number, got null."
        )
    if isinstance(value, str):
        if value.strip().lower() == "default":
            return None
        try:
            value = float(value)
        except ValueError as exc:
            raise LLMSettingsError(
                f"{origin}: temperature must be 'default' or a number, got {value!r}."
            ) from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LLMSettingsError(
            f"{origin}: temperature must be 'default' or a number, got {value!r}."
        )
    temperature = float(value)
    if not 0.0 <= temperature <= 2.0:
        raise LLMSettingsError(
            f"{origin}: temperature {temperature} is outside [0, 2]."
        )
    return temperature


def parse_llm_settings_block(block: Any, origin: str) -> Dict[str, Any]:
    """Validate a raw llm_settings mapping; return normalized fields."""
    if not isinstance(block, Mapping):
        raise LLMSettingsError(f"{origin}: llm_settings must be a mapping.")
    missing = [key for key in REQUIRED_KEYS if key not in block]
    if missing:
        raise LLMSettingsError(
            f"{origin}: llm_settings is missing required key(s): {', '.join(missing)}."
        )
    unknown = sorted(set(block) - set(REQUIRED_KEYS))
    if unknown:
        raise LLMSettingsError(
            f"{origin}: llm_settings has unknown key(s): {', '.join(unknown)}."
        )
    provider = str(block["provider"]).strip().lower()
    if provider not in VALID_PROVIDERS:
        raise LLMSettingsError(
            f"{origin}: llm_settings.provider must be one of "
            f"{', '.join(VALID_PROVIDERS)}; got {block['provider']!r}."
        )
    reasoning = str(block["reasoning"]).strip().lower()
    if reasoning not in VALID_REASONING:
        raise LLMSettingsError(
            f"{origin}: llm_settings.reasoning must be one of "
            f"{', '.join(VALID_REASONING)}; got {block['reasoning']!r}."
        )
    temperature = _parse_temperature(block["temperature"], origin)
    return {"provider": provider, "reasoning": reasoning, "temperature": temperature}


def resolve_llm_settings(
    exp_set: Mapping[str, Any],
    experiment_name: str,
    *,
    config_path: str = "",
    environ: Optional[Mapping[str, str]] = None,
    warn=print,
) -> LLMSettings:
    """Resolve the LLM regime for an experiment set. Fails closed if unpinned.

    Env overrides (LLM_PROVIDER, LLM_REASONING, LLM_TEMPERATURE) are applied on
    top of the config block and recorded in ``LLMSettings.overrides``.
    """
    environ = os.environ if environ is None else environ
    origin = f"experiment set {experiment_name!r}"
    block = exp_set.get("llm_settings") if isinstance(exp_set, Mapping) else None
    if block is None:
        raise LLMSettingsError(missing_block_message(experiment_name, config_path))
    fields = parse_llm_settings_block(block, origin)

    overrides: Dict[str, str] = {}
    provider_env = (environ.get("LLM_PROVIDER") or "").strip().lower()
    if provider_env and provider_env != "auto":
        if provider_env not in VALID_PROVIDERS + LEGACY_PROVIDER_MODES:
            raise LLMSettingsError(
                f"LLM_PROVIDER={provider_env!r} is not a valid override "
                f"({', '.join(VALID_PROVIDERS + LEGACY_PROVIDER_MODES)})."
            )
        if provider_env != fields["provider"]:
            overrides["LLM_PROVIDER"] = provider_env
            fields["provider"] = provider_env
    elif provider_env == "auto":
        # `auto` is exactly the per-machine behaviour this module removes.
        warn(
            "⚠️  LLM_PROVIDER=auto is ignored when an llm_settings block is "
            f"active; using provider={fields['provider']!r} from config."
        )

    reasoning_env = (environ.get("LLM_REASONING") or "").strip().lower()
    if reasoning_env:
        if reasoning_env not in VALID_REASONING:
            raise LLMSettingsError(
                f"LLM_REASONING={reasoning_env!r} is not one of {', '.join(VALID_REASONING)}."
            )
        if reasoning_env != fields["reasoning"]:
            overrides["LLM_REASONING"] = reasoning_env
            fields["reasoning"] = reasoning_env

    temperature_env = (environ.get("LLM_TEMPERATURE") or "").strip()
    if temperature_env:
        parsed = _parse_temperature(temperature_env, "LLM_TEMPERATURE")
        if parsed != fields["temperature"]:
            overrides["LLM_TEMPERATURE"] = temperature_env
            fields["temperature"] = parsed

    stale = [knob for knob in LEGACY_ENV_KNOBS if (environ.get(knob) or "").strip()]
    if stale:
        warn(
            "⚠️  Ignoring legacy env knob(s) "
            + ", ".join(stale)
            + " because llm_settings is active; use LLM_REASONING to override."
        )
    if overrides:
        warn(
            "⚠️  llm_settings overridden from the environment (recorded in "
            f"run_metadata): {overrides}"
        )

    return LLMSettings(
        provider=fields["provider"],
        reasoning=fields["reasoning"],
        temperature=fields["temperature"],
        source="config+env" if overrides else "config",
        overrides=overrides,
    )


# ---------------------------------------------------------------------------
# Model capability tables used to fail closed on impossible combinations.
# Every entry is backed by a live probe or vendor doc; see
# docs/verified-facts.md and data/analysis/api_equivalence_audit_2026_09_04/.
# ---------------------------------------------------------------------------

def model_ignores_temperature(provider_model: str) -> Optional[str]:
    """Return a reason string if the model rejects or ignores temperature."""
    m = provider_model.lower()
    native = m.split("/", 1)[-1]
    if native.startswith("gpt-5"):
        return "GPT-5 family accepts only the default temperature (400 otherwise)"
    if native in {"gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"}:
        return "Gemini 3.6+ Flash accepts temperature but ignores it (probe 2026-09-04)"
    if (
        native.startswith("claude-opus-4-7")
        or native.startswith("claude-opus-4-8")
        or native.startswith("claude-opus-5")
        or native.startswith("claude-sonnet-5")
        or native.startswith("claude-fable")
        or native.startswith("claude-mythos")
    ):
        return "this Claude model rejects an explicit temperature (400: deprecated)"
    return None


def model_cannot_disable_reasoning(provider_model: str) -> Optional[str]:
    """Return a reason string if reasoning='off' is impossible for the model."""
    native = provider_model.lower().split("/", 1)[-1]
    if native.startswith("gpt-5"):
        return "GPT-5 family cannot switch reasoning off; the floor is 'minimal'"
    if native.startswith("gemini-3"):
        return (
            "Gemini 3.x cannot switch thinking off; use 'minimal' on 3.6/3.5 "
            "Flash or 'low' on 3.7/3.8 Flash and 3.x Pro"
        )
    return None


def anthropic_uses_adaptive_thinking(native_model: str) -> bool:
    """Claude 4.6+ models take thinking={type: adaptive} + output_config.effort;
    older models (Sonnet/Haiku 4.5, Opus 4.5 and earlier) take budget_tokens."""
    m = native_model.lower()
    return (
        m.startswith("claude-opus-4-6")
        or m.startswith("claude-opus-4-7")
        or m.startswith("claude-opus-4-8")
        or m.startswith("claude-opus-5")
        or m.startswith("claude-sonnet-4-6")
        or m.startswith("claude-sonnet-5")
        or m.startswith("claude-fable")
        or m.startswith("claude-mythos")
    )


# budget_tokens for the pre-4.6 Claude thinking API (minimum accepted is 1024).
ANTHROPIC_BUDGET_BY_LEVEL = {
    "minimal": 1024,
    "low": 2048,
    "medium": 4096,
    "high": 16000,
}
