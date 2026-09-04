"""Config-over-env LLM settings: resolution, request planning, metadata,
per-call finish_reason, and the fail-closed runner path."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.llm_settings import (
    LLMSettings,
    LLMSettingsError,
    parse_llm_settings_block,
    resolve_llm_settings,
)
from src.utils import (
    LLMClient,
    _call_anthropic,
    _call_gemini,
    _call_openai_compatible,
    llm_runtime_metadata,
    plan_request_settings,
)


def _settings(provider="direct", reasoning="off", temperature=None, **kw):
    return LLMSettings(provider=provider, reasoning=reasoning, temperature=temperature, **kw)


class ResolveTests(unittest.TestCase):
    def test_missing_block_fails_closed_with_guidance(self):
        with self.assertRaises(LLMSettingsError) as ctx:
            resolve_llm_settings({"models": ["x"]}, "my_set", config_path="cfg.yaml", environ={})
        msg = str(ctx.exception)
        self.assertIn("my_set", msg)
        self.assertIn("llm_settings", msg)
        self.assertIn("provider:", msg)

    def test_valid_block_parses(self):
        s = resolve_llm_settings(
            {"llm_settings": {"provider": "direct", "reasoning": "off", "temperature": "default"}},
            "s",
            environ={},
        )
        self.assertEqual(s.as_dict(), {"provider": "direct", "reasoning": "off", "temperature": "default"})
        self.assertEqual(s.source, "config")
        self.assertEqual(s.overrides, {})

    def test_invalid_values_rejected(self):
        for bad in (
            {"provider": "auto", "reasoning": "off", "temperature": "default"},
            {"provider": "direct", "reasoning": "xhigh", "temperature": "default"},
            {"provider": "direct", "reasoning": "off", "temperature": 3.0},
            {"provider": "direct", "reasoning": "off"},
            {"provider": "direct", "reasoning": "off", "temperature": "default", "extra": 1},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(LLMSettingsError):
                    parse_llm_settings_block(bad, "t")

    def test_env_overrides_are_applied_and_recorded(self):
        warnings = []
        s = resolve_llm_settings(
            {"llm_settings": {"provider": "direct", "reasoning": "off", "temperature": 0.8}},
            "s",
            environ={"LLM_PROVIDER": "openrouter", "LLM_REASONING": "low", "LLM_TEMPERATURE": "default"},
            warn=warnings.append,
        )
        self.assertEqual(s.provider, "openrouter")
        self.assertEqual(s.reasoning, "low")
        self.assertIsNone(s.temperature)
        self.assertEqual(
            s.overrides,
            {"LLM_PROVIDER": "openrouter", "LLM_REASONING": "low", "LLM_TEMPERATURE": "default"},
        )
        self.assertEqual(s.source, "config+env")
        self.assertTrue(any("overridden" in w for w in warnings))

    def test_auto_and_legacy_knobs_are_ignored_with_warning(self):
        warnings = []
        s = resolve_llm_settings(
            {"llm_settings": {"provider": "direct", "reasoning": "off", "temperature": "default"}},
            "s",
            environ={"LLM_PROVIDER": "auto", "GEMINI_THINKING_LEVEL": "medium"},
            warn=warnings.append,
        )
        self.assertEqual(s.provider, "direct")
        self.assertEqual(s.overrides, {})
        self.assertTrue(any("LLM_PROVIDER=auto is ignored" in w for w in warnings))
        self.assertTrue(any("GEMINI_THINKING_LEVEL" in w for w in warnings))


class PlanTests(unittest.TestCase):
    def test_claude_45_off_sends_nothing_and_temperature_when_set(self):
        plan = plan_request_settings("anthropic", "claude-sonnet-4-5-20250929", 0.8, _settings(temperature=0.8))
        self.assertEqual(plan["temperature"], 0.8)
        self.assertIsNone(plan["reasoning_param"])
        self.assertEqual(plan["reasoning"], "off")

    def test_claude_45_levels_use_budget_tokens(self):
        plan = plan_request_settings("anthropic", "claude-sonnet-4-5-20250929", 0.8, _settings(reasoning="low"))
        self.assertEqual(plan["reasoning_param"], {"thinking": {"type": "enabled", "budget_tokens": 2048}})
        self.assertIsNone(plan["temperature"])

    def test_opus5_off_uses_disabled_and_rejects_temperature(self):
        plan = plan_request_settings("anthropic", "claude-opus-5", 0.8, _settings())
        self.assertEqual(plan["reasoning_param"], {"thinking": {"type": "disabled"}})
        with self.assertRaises(LLMSettingsError):
            plan_request_settings("anthropic", "claude-opus-5", 0.8, _settings(temperature=0.8))

    def test_gpt5_cannot_disable_and_rejects_temperature(self):
        with self.assertRaises(LLMSettingsError):
            plan_request_settings("openai", "gpt-5-nano", 0.8, _settings())
        with self.assertRaises(LLMSettingsError):
            plan_request_settings("openai", "gpt-5-nano", 0.8, _settings(reasoning="low", temperature=0.8))
        plan = plan_request_settings("openai", "gpt-5-nano", 0.8, _settings(reasoning="low"))
        self.assertEqual(plan["reasoning_param"], {"reasoning_effort": "low"})
        self.assertIsNone(plan["temperature"])

    def test_gemini_36_minimal_ok_off_refused_temperature_refused(self):
        plan = plan_request_settings("google", "gemini-3.6-flash", 0.8, _settings(reasoning="minimal"))
        self.assertEqual(plan["reasoning_param"], {"thinkingLevel": "minimal"})
        with self.assertRaises(LLMSettingsError):
            plan_request_settings("google", "gemini-3.6-flash", 0.8, _settings())
        with self.assertRaises(LLMSettingsError):
            plan_request_settings("google", "gemini-3.6-flash", 0.8, _settings(reasoning="minimal", temperature=0.8))

    def test_openrouter_off_sends_no_reasoning_body_even_for_claude(self):
        plan = plan_request_settings("openrouter", "anthropic/claude-sonnet-4.5", 0.8, _settings(provider="openrouter"))
        self.assertIsNone(plan["reasoning_param"])
        plan = plan_request_settings("openrouter", "openai/gpt-5-nano", 0.8, _settings(provider="openrouter", reasoning="low"))
        self.assertEqual(plan["reasoning_param"], {"reasoning": {"effort": "low"}})

    def test_legacy_plan_matches_historical_behaviour(self):
        with patch.dict("os.environ", {}, clear=True):
            plan = plan_request_settings("openrouter", "anthropic/claude-sonnet-4.5", 0.8, None)
        self.assertEqual(plan["source"], "legacy_env")
        self.assertEqual(plan["temperature"], 0.8)
        self.assertEqual(plan["reasoning_param"], {"reasoning": {"effort": "medium"}})
        with patch.dict("os.environ", {}, clear=True):
            plan = plan_request_settings("openai", "gpt-5-nano", 0.8, None)
        self.assertIsNone(plan["temperature"])
        self.assertEqual(plan["reasoning_param"], {"reasoning_effort": "minimal"})


class MetadataTests(unittest.TestCase):
    def test_runtime_metadata_records_effective_settings(self):
        client = LLMClient("anthropic", object())
        with patch.dict("os.environ", {"ANTHROPIC_MAX_TOKENS": "4096"}, clear=True):
            meta = llm_runtime_metadata(
                client,
                "anthropic/claude-sonnet-4.5",
                _settings(temperature=0.8, source="config+env", overrides={"LLM_REASONING": "off"}),
            )
        self.assertEqual(meta["llm_provider_mode"], "direct")
        self.assertEqual(meta["llm_settings"], {"provider": "direct", "reasoning": "off", "temperature": 0.8})
        self.assertEqual(meta["llm_settings_source"], "config+env")
        self.assertEqual(meta["llm_settings_overrides"], {"LLM_REASONING": "off"})
        eff = meta["llm_settings_effective"]
        self.assertEqual(eff["provider_model"], "claude-sonnet-4-5-20250929")
        self.assertTrue(eff["temperature_sent"])
        self.assertEqual(eff["temperature_value"], 0.8)
        self.assertEqual(eff["reasoning"], "off")
        self.assertIsNone(eff["reasoning_param"])
        # Metadata is JSON-serialisable (it lands in run JSONs).
        json.dumps(meta)

    def test_runtime_metadata_legacy_marks_source(self):
        client = LLMClient("openrouter", object())
        meta = llm_runtime_metadata(client, "openai/gpt-5-nano")
        self.assertEqual(meta["llm_settings_source"], "legacy_env")
        self.assertNotIn("llm_settings_effective", meta)


class _Choice:
    def __init__(self, content, finish_reason="stop"):
        self.message = SimpleNamespace(content=content)
        self.finish_reason = finish_reason


class _OpenAIResponse:
    def __init__(self):
        self.choices = [_Choice('{"send": 3}', "stop")]
        self.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)


class _FakeChat:
    def __init__(self):
        self.calls = []

    def create(self, **params):
        self.calls.append(params)
        return _OpenAIResponse()


class _FakeOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeChat())


class _FakeAnthropicClient:
    def __init__(self):
        self.calls = []
        self.messages = self

    def create(self, **params):
        self.calls.append(params)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"send": 4}')],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=12, output_tokens=6),
        )


class PerCallTests(unittest.TestCase):
    messages = [{"role": "system", "content": "Rules"}, {"role": "user", "content": "Go"}]

    def test_openai_direct_records_finish_reason_and_sends_effort_only(self):
        client = _FakeOpenAIClient()
        plan = plan_request_settings("openai", "gpt-5-nano", 0.8, _settings(reasoning="low"))
        out = _call_openai_compatible("openai", client, "gpt-5-nano", plan, self.messages, 1, "medium")
        params = client.chat.completions.calls[0]
        self.assertNotIn("temperature", params)
        self.assertEqual(params["reasoning_effort"], "low")
        self.assertEqual(out["usage"]["finish_reason"], "stop")

    def test_openrouter_off_sends_no_extra_body(self):
        client = _FakeOpenAIClient()
        plan = plan_request_settings("openrouter", "anthropic/claude-sonnet-4.5", 0.8, _settings(provider="openrouter"))
        _call_openai_compatible("openrouter", client, "anthropic/claude-sonnet-4.5", plan, self.messages, 1, "medium")
        params = client.chat.completions.calls[0]
        self.assertNotIn("extra_body", params)
        self.assertNotIn("temperature", params)

    def test_anthropic_off_sends_no_thinking_and_records_stop_reason(self):
        client = _FakeAnthropicClient()
        with patch.dict("os.environ", {"ANTHROPIC_MAX_TOKENS": "4096"}, clear=True):
            plan = plan_request_settings("anthropic", "claude-sonnet-4-5-20250929", 0.8, _settings(temperature=0.8))
            out = _call_anthropic(client, "claude-sonnet-4-5-20250929", plan, self.messages, 1)
        params = client.calls[0]
        self.assertNotIn("thinking", params)
        self.assertEqual(params["temperature"], 0.8)
        self.assertEqual(out["usage"]["finish_reason"], "end_turn")

    def test_anthropic_budget_must_fit_under_max_tokens(self):
        client = _FakeAnthropicClient()
        with patch.dict("os.environ", {"ANTHROPIC_MAX_TOKENS": "1024"}, clear=True):
            plan = plan_request_settings("anthropic", "claude-sonnet-4-5-20250929", 0.8, _settings(reasoning="low"))
            with self.assertRaises(LLMSettingsError):
                _call_anthropic(client, "claude-sonnet-4-5-20250929", plan, self.messages, 1)

    def test_gemini_minimal_sends_thinking_level_and_records_finish_reason(self):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(
                    {
                        "candidates": [{"content": {"parts": [{"text": '{"send": 5}'}]}, "finishReason": "STOP"}],
                        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "thoughtsTokenCount": 0},
                    }
                ).encode()

        with patch("src.utils.urllib.request.urlopen", return_value=_Resp()) as mocked:
            plan = plan_request_settings("google", "gemini-3.6-flash", 0.8, _settings(reasoning="minimal"))
            out = _call_gemini({"api_key": "k", "base_url": "https://x/v1beta"}, "gemini-3.6-flash", plan, self.messages, 1)
        payload = json.loads(mocked.call_args.args[0].data.decode())
        self.assertEqual(payload["generationConfig"]["thinkingConfig"], {"thinkingLevel": "minimal"})
        self.assertNotIn("temperature", payload["generationConfig"])
        self.assertEqual(out["usage"]["finish_reason"], "STOP")


class RunnerFailClosedTests(unittest.TestCase):
    def test_noisy_runner_refuses_unpinned_set(self):
        import tempfile
        import textwrap
        from experiments.run_noisy_batch import run_experiment_set

        cfg = textwrap.dedent(
            """
            base_models: {m: "anthropic/claude-haiku-4.5"}
            prompt_templates: {trust_game_default: "x", trust_game_round1_investor: "x",
              trust_game_round1_trustee: "x", trust_game_later_investor: "x",
              trust_game_later_trustee: "x", myth_writing_default: "x",
              myth_writing_later_rounds: "x"}
            personas: {neutral: {description: n, system_addition: ""}}
            myth_topics: {anything: "a"}
            game_params:
              default: {endowment: 5, multiplier: 3, num_turns: 1, num_agents: 2,
                temperature: 0.8, memory_capacity: 3}
            experiment_sets:
              unpinned:
                models: [m]
                templates: [trust_game_default]
                personas: [neutral]
                task_orders: [["game"]]
                game_params_list: [default]
                num_runs: 1
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write(cfg)
            path = fh.name
        with self.assertRaises(LLMSettingsError) as ctx:
            run_experiment_set("unpinned", workers=1, config_path=path)
        self.assertIn("unpinned", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
