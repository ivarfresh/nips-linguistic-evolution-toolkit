import json
import unittest
from unittest.mock import patch

from src.utils import (
    _call_gemini,
    _gemini_request_timeout_seconds,
    _gemini_supports_temperature,
)


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(
            {
                "candidates": [
                    {"content": {"parts": [{"text": '{"send": 5}'}]}}
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 4,
                    "thoughtsTokenCount": 2,
                },
            }
        ).encode("utf-8")


class GeminiProviderTests(unittest.TestCase):
    def setUp(self):
        self.client = {
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1beta",
        }
        self.messages = [
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "Choose"},
        ]

    def _payload(self, mocked_urlopen):
        request = mocked_urlopen.call_args.args[0]
        return json.loads(request.data.decode("utf-8"))

    @patch("src.utils.urllib.request.urlopen", return_value=_FakeResponse())
    def test_gemini_37_omits_deprecated_temperature(self, mocked_urlopen):
        result = _call_gemini(
            self.client,
            "gemini-3.7-flash",
            0.8,
            self.messages,
            max_retries=1,
        )

        self.assertEqual(result["content"], '{"send": 5}')
        self.assertNotIn("temperature", self._payload(mocked_urlopen)["generationConfig"])
        self.assertFalse(_gemini_supports_temperature("gemini-3.7-flash"))

    @patch("src.utils.urllib.request.urlopen", return_value=_FakeResponse())
    def test_earlier_gemini_keeps_historical_temperature(self, mocked_urlopen):
        _call_gemini(
            self.client,
            "gemini-3.1-flash-lite",
            0.8,
            self.messages,
            max_retries=1,
        )

        self.assertEqual(
            self._payload(mocked_urlopen)["generationConfig"]["temperature"],
            0.8,
        )
        self.assertTrue(_gemini_supports_temperature("gemini-3.1-flash-lite"))

    @patch("src.utils.urllib.request.urlopen", return_value=_FakeResponse())
    def test_gemini_36_minimal_thinking_and_temperature(self, mocked_urlopen):
        # 3.6 Flash is the matched-reasoning Gemini arm: it accepts
        # thinking_level=minimal (3.7/3.8 stop at low). Like 3.7 it ignores
        # temperature (live probe 2026-09-04, scripts/probe_gemini_settings.py:
        # 5/5 distinct outputs at temperature 0), so the parameter is omitted
        # and run metadata records temperature_sent=false.
        with patch.dict("os.environ", {"GEMINI_THINKING_LEVEL": "minimal"}):
            _call_gemini(
                self.client,
                "gemini-3.6-flash",
                0.8,
                self.messages,
                max_retries=1,
            )

        config = self._payload(mocked_urlopen)["generationConfig"]
        self.assertNotIn("temperature", config)
        self.assertEqual(config["thinkingConfig"], {"thinkingLevel": "minimal"})
        self.assertFalse(_gemini_supports_temperature("gemini-3.6-flash"))

    def test_gemini_36_slug_resolves_to_direct_model_id(self):
        from src.utils import DIRECT_MODEL_ALIASES

        self.assertEqual(
            DIRECT_MODEL_ALIASES["google/gemini-3.6-flash"], "gemini-3.6-flash"
        )

    def test_timeout_defaults_and_validates_override(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_gemini_request_timeout_seconds(), 120.0)
        with patch.dict(
            "os.environ", {"GEMINI_REQUEST_TIMEOUT_SECONDS": "300"}, clear=True
        ):
            self.assertEqual(_gemini_request_timeout_seconds(), 300.0)
        with patch.dict(
            "os.environ", {"GEMINI_REQUEST_TIMEOUT_SECONDS": "zero"}, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "positive number"):
                _gemini_request_timeout_seconds()



def _plan(provider):
    from src.llm_settings import resolve_request_plan
    block = {"provider": provider, "reasoning": {"thinkingConfig": {"thinkingLevel": "high"}},
             "temperature": "default", "max_output_tokens": "default"}
    return resolve_request_plan("google/gemini-3.7-flash", block, {"google/gemini-3.7-flash": "gemini-3.7-flash"})


class _VersionedResponse(_FakeResponse):
    def __init__(self, version):
        self.version = version

    def read(self):
        data = json.loads(super().read())
        data["modelVersion"] = self.version
        return json.dumps(data).encode("utf-8")


class VertexAndModelGuardTests(unittest.TestCase):
    messages = [{"role": "user", "content": "Decide."}]

    def test_vertex_plan_uses_vertex_endpoint_and_native_model(self):
        plan = _plan("vertex").as_dict()
        self.assertEqual(plan["provider"], "vertex")
        self.assertEqual(plan["endpoint"], "https://aiplatform.googleapis.com/v1/publishers/google")
        self.assertEqual(plan["provider_model"], "gemini-3.7-flash")

    def test_vertex_request_goes_to_publisher_model_url(self):
        from src.utils import _create_vertex_client
        with patch.dict("os.environ", {"GEMINI_API_KEY": "k"}):
            client = _create_vertex_client()
        seen = {}

        def fake_urlopen(request, timeout=None):
            seen["url"] = request.full_url
            return _VersionedResponse("gemini-3.7-flash")

        with patch("src.utils.urllib.request.urlopen", side_effect=fake_urlopen):
            out = _call_gemini(client.client, "gemini-3.7-flash", 1.0, self.messages, 1, request_plan=_plan("vertex"))
        self.assertEqual(seen["url"], "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3.7-flash:generateContent")
        self.assertEqual(out["usage"]["response_model"], "gemini-3.7-flash")

    def test_silent_redirect_to_another_model_fails_the_call(self):
        from src.utils import LLMRequestError
        client = {"api_key": "k", "base_url": "https://generativelanguage.googleapis.com/v1beta"}
        with patch("src.utils.urllib.request.urlopen", return_value=_VersionedResponse("gemini-3.8-flash")):
            with self.assertRaises(LLMRequestError) as caught:
                _call_gemini(client, "gemini-3.7-flash", 1.0, self.messages, 1, request_plan=_plan("direct"))
        self.assertEqual(caught.exception.usage["outcome"], "model_mismatch")

    def test_dated_snapshot_of_requested_model_is_accepted(self):
        client = {"api_key": "k", "base_url": "https://generativelanguage.googleapis.com/v1beta"}
        with patch("src.utils.urllib.request.urlopen", return_value=_VersionedResponse("gemini-3.7-flash-001")):
            out = _call_gemini(client, "gemini-3.7-flash", 1.0, self.messages, 1, request_plan=_plan("direct"))
        self.assertEqual(out["usage"]["response_model"], "gemini-3.7-flash-001")
        self.assertNotEqual(out["usage"]["outcome"], "model_mismatch")

    def test_model_check_rejects_other_models_sharing_a_prefix(self):
        from src.utils import model_matches
        self.assertTrue(model_matches("gpt-5-nano", "gpt-5-nano-2025-08-07"))
        self.assertTrue(model_matches("claude-sonnet-4-5", "claude-sonnet-4-5-20250929"))
        self.assertTrue(model_matches("gemini-3.7-flash", "gemini-3.7-flash"))
        self.assertTrue(model_matches("gemini-3.7-flash", "gemini-3.7-flash-001"))
        self.assertFalse(model_matches("gpt-5", "gpt-5-nano"))
        self.assertFalse(model_matches("claude-opus-5", "claude-opus-5-5"))
        self.assertFalse(model_matches("gemini-3.7-flash", "gemini-3.7-flash-lite"))
        self.assertFalse(model_matches("gemini-3.7-flash", "gemini-3.8-flash"))
        self.assertFalse(model_matches("gemini-3.7-flash", None))


if __name__ == "__main__":
    unittest.main()
