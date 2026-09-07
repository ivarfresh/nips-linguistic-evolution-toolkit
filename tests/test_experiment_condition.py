import contextlib
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from games.trust_game import TrustGame
from src.experiment_condition import (
    ConditionMismatchError, build_condition, check_conditions, condition_from_run,
    digest, validate_output_provenance,
)
from src.llm_settings import LLMSettings, resolve_llm_settings
from src.myth_writer import MythWriter
from src.simulation import run_simulation
from src.utils import LLMClient, call_llm, llm_runtime_metadata
from analyses._shared import load_simulation_runs
from scripts.write_provenance import build_provenance
from scripts.hf_sync_completed_runs import has_llm_provenance


MODEL = "anthropic/claude-sonnet-4.5"
SETTINGS = LLMSettings("direct", "off", 0.8)


def make_game():
    return TrustGame(
        5, 3, system_prompt_template="Game rules",
        round1_investor_template="You are the sender. Send?",
        round1_trustee_template="You are the receiver. Return?",
        later_investor_template="Send?", later_trustee_template="Return?",
    )


def make_state():
    metadata = llm_runtime_metadata(LLMClient("anthropic", object()), MODEL, SETTINGS)
    condition = build_condition(make_game(), MythWriter("anything", "Directed myth", "Directed continuation"), metadata, {"task_order": ["myth", "game"]}, 0)
    metadata.update(experiment_condition=condition, condition_sha256=digest(condition))
    return {"agents": {}, "conversation_history": [], "game_data": {}, "task_order": ["myth", "game"], "run_metadata": metadata}


def write_run(path, data):
    path.write_text(json.dumps(data))
    return str(path)


def test_comparison_catches_myth_changes_even_when_api_settings_match(tmp_path):
    original = make_state()
    changed = copy.deepcopy(original)
    changed["run_metadata"]["experiment_condition"]["protocol"]["myth"]["round1_template"] = "Generic myth"
    changed["run_metadata"]["condition_sha256"] = digest(changed["run_metadata"]["experiment_condition"])
    paths = [write_run(tmp_path / "original.json", original), write_run(tmp_path / "changed.json", changed)]
    with pytest.raises(ConditionMismatchError, match="protocol.myth.round1_template"):
        load_simulation_runs(paths)
    assert len(load_simulation_runs(paths, allowed_differences={"protocol.myth.round1_template": "Intentional myth-direction ablation"})) == 2


def test_output_provenance_derives_differences_instead_of_trusting_a_summary(tmp_path):
    state = make_state()
    first = write_run(tmp_path / "first.json", state)
    changed = copy.deepcopy(state)
    changed["run_metadata"]["experiment_condition"]["protocol"]["game"]["decision_format"] = "json_only"
    changed["run_metadata"]["condition_sha256"] = digest(changed["run_metadata"]["experiment_condition"])
    second = write_run(tmp_path / "second.json", changed)
    allowed = {"protocol.game.decision_format": "Intentional format intervention"}
    document = build_provenance([first, second], allowed_differences=allowed)
    validate_output_provenance(document)
    document["observed_differences"] = []
    with pytest.raises(ConditionMismatchError, match="misstates"):
        validate_output_provenance(document)
    document["observed_differences"] = ["protocol.game.decision_format"]
    document["allowed_differences"] = {}
    with pytest.raises(ConditionMismatchError, match="Undeclared"):
        validate_output_provenance(document)


def test_legacy_requires_a_reason_and_still_checks_known_differences(tmp_path):
    state = make_state()
    state["run_metadata"].pop("experiment_condition")
    first = write_run(tmp_path / "old.json", state)
    with pytest.raises(ConditionMismatchError, match="Historical"):
        load_simulation_runs([first])
    assert load_simulation_runs([first], legacy_reason="Descriptive historical audit")
    state["run_metadata"]["myth_default_prompt_key"] = "changed"
    second = write_run(tmp_path / "old_changed.json", state)
    with pytest.raises(ConditionMismatchError, match="myth_default_prompt_key"):
        load_simulation_runs([first, second], legacy_reason="Descriptive historical audit")


def test_result_and_checkpoint_files_cannot_be_analysis_inputs(tmp_path):
    for name in ("run.results.json", "run.checkpoint.json", "run.error.json"):
        path = write_run(tmp_path / name, make_state())
        with pytest.raises(ConditionMismatchError, match="Not a final"):
            load_simulation_runs([path])
    path = write_run(tmp_path / "metadata.json", {"run_metadata": make_state()["run_metadata"]})
    with pytest.raises(ConditionMismatchError, match="full-state"):
        load_simulation_runs([path])


def test_provider_model_and_call_settings_are_verified():
    state = make_state()
    assert has_llm_provenance(state)
    state["run_metadata"]["provider_model"] = "another-model"
    assert not has_llm_provenance(state)
    state = make_state()
    state["agents"] = {"Agent_1": {"interaction_history": [{"response": {"usage": {"finish_reason": "end_turn"}}}]}}
    with pytest.raises(ConditionMismatchError, match="Per-call"):
        condition_from_run(state)
    assert not has_llm_provenance({"run_metadata": {"llm_provider": "anthropic", "provider_model": MODEL}})


def test_identical_request_plan_is_used_after_environment_changes(monkeypatch):
    calls = []
    response = SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking="A thought"), SimpleNamespace(type="text", text='{"send": 4}')], usage=SimpleNamespace(input_tokens=10, output_tokens=20), stop_reason="end_turn")
    def create(**params):
        calls.append(params)
        return response
    client = LLMClient("anthropic", SimpleNamespace(messages=SimpleNamespace(create=create)))
    metadata = llm_runtime_metadata(client, MODEL, SETTINGS)
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "999")
    monkeypatch.setenv("LLM_DIRECT_MODEL_ANTHROPIC_CLAUDE_SONNET_4_5", "different-model")
    result = call_llm(client, MODEL, 0.8, [{"role": "user", "content": "Decide"}], llm_settings=SETTINGS)
    assert calls[0]["model"] == metadata["provider_model"]
    assert calls[0]["max_tokens"] == 4096
    assert result["usage"]["request_settings"] == metadata["llm_settings_effective"]
    assert result["usage"]["reasoning_tokens"] is None
    assert result["usage"]["thinking_present"] is True
    with pytest.raises(ValueError, match="settings changed"):
        call_llm(client, MODEL, 0.8, [{"role": "user", "content": "Decide"}], llm_settings=LLMSettings("direct", "low", 0.8))
    assert len(calls) == 1


def test_resume_refuses_changed_settings_and_preserves_checkpoint(tmp_path):
    checkpoint = tmp_path / "run.checkpoint.json"
    count = 0
    def fake_call(*args, **kwargs):
        nonlocal count
        count += 1
        if count > 2:
            raise RuntimeError("Simulated interruption")
        return {"content": '{"send": 4}' if count == 1 else '{"return": 3}', "usage": {}, "reasoning": None}
    options = dict(model=MODEL, temperature=0.8, num_turns=2, num_agents=2, memory_capacity=3, agent_biases="", myth_writer=None, task_order=["game"], checkpoint_path=str(checkpoint), checkpoint_every=1, llm_settings=SETTINGS)
    with patch("src.simulation.create_llm_client", return_value=LLMClient("anthropic", object())), patch("src.agents.call_llm", side_effect=fake_call), patch("src.simulation.time.sleep"), contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(RuntimeError, match="interruption"):
            run_simulation(game=make_game(), **options)
        saved = checkpoint.read_bytes()
        changed_options = dict(options, llm_settings=LLMSettings("direct", "low", 0.8))
        with pytest.raises(ConditionMismatchError, match="llm.reasoning"):
            run_simulation(game=make_game(), resume_from=str(checkpoint), **changed_options)
        assert checkpoint.read_bytes() == saved
        changed_game = make_game()
        changed_game.later_investor_template = "An unintended new prompt"
        with pytest.raises(ConditionMismatchError, match="later_investor_template"):
            run_simulation(game=changed_game, resume_from=str(checkpoint), **options)
        with patch("src.agents.call_llm", side_effect=[{"content": '{"send": 4}', "usage": {}, "reasoning": None}, {"content": '{"return": 3}', "usage": {}, "reasoning": None}]):
            resumed = run_simulation(game=make_game(), resume_from=str(checkpoint), **options)
        assert len(resumed.conversation_history) == 2
        assert resumed.run_metadata["experiment_condition"]["llm"]["reasoning"] == "off"


def test_legacy_settings_need_explicit_migration_opt_in():
    with pytest.raises(ValueError, match="no `llm_settings`"):
        resolve_llm_settings({}, "historical", environ={})
    assert resolve_llm_settings({}, "historical", environ={}, allow_legacy=True) is None


def test_declared_format_comparison_rejects_accidental_myth_default():
    from experiments.run_noisy_batch import NoisyExperimentConfig
    from src.comparison_config import validate_config_comparisons
    config = NoisyExperimentConfig(str(Path(__file__).resolve().parent.parent / "config/experiments_noisy.yaml"))
    validate_config_comparisons(config, environ={})
    config.config["experiment_sets"]["fmt_controlled_v2_json_only_n5"]["myth_prompt_arms"] = [{"id": "generic", "default": "myth_writing_default", "later": "myth_writing_later_rounds"}]
    with pytest.raises(ConditionMismatchError, match="myth_writing"):
        validate_config_comparisons(config, environ={})


@pytest.mark.parametrize("module_name", ["analyses.cooperation_ratio_over_time", "analyses.resources_over_time_max"])
def test_plot_entrypoints_refuse_unmatched_inputs_before_writing(module_name, tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module(module_name)
    original = make_state()
    changed = copy.deepcopy(original)
    changed["run_metadata"]["experiment_condition"]["protocol"]["myth"]["round1_template"] = "Different myth"
    changed["run_metadata"]["condition_sha256"] = digest(changed["run_metadata"]["experiment_condition"])
    paths = [write_run(tmp_path / "first.json", original), write_run(tmp_path / "second.json", changed)]
    monkeypatch.setattr(module, "load_runs", lambda root: [{"source_path": path} for path in paths])
    monkeypatch.setattr("sys.argv", [module_name, "--out", str(tmp_path / "plots")])
    with pytest.raises(ConditionMismatchError, match="round1_template"):
        module.main()
    assert not (tmp_path / "plots").exists()


def test_missing_runner_rejects_existing_changed_condition(tmp_path):
    from scripts.run_noisy_missing import verify_existing_output
    from src.comparison_config import resolved_comparison_inputs

    combo = {"llm_settings": SETTINGS, "task_order": ["game"], "replicate_id": 0}
    state = make_state()
    state["run_metadata"]["comparison_inputs"] = resolved_comparison_inputs(combo, SETTINGS)
    path = write_run(tmp_path / "complete.json", state)
    verify_existing_output(path, combo)
    with pytest.raises(ConditionMismatchError, match="task_order"):
        verify_existing_output(path, dict(combo, task_order=["myth", "game"]))
    with pytest.raises(ConditionMismatchError, match="full-state"):
        verify_existing_output(write_run(tmp_path / "incomplete.json", {}), combo, allow_legacy=True)


def test_legacy_upload_opt_in_does_not_accept_corrupt_modern_provenance():
    state = make_state()
    state["run_metadata"]["condition_sha256"] = "wrong"
    assert not has_llm_provenance(state, allow_legacy=True)


def test_local_http_simulation_records_real_payloads_and_uploads_only_final(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from scripts.hf_sync_completed_runs import artifacts_for_completed_runs

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            decision = '{"send": 4}' if len(requests) % 2 else '{"return": 3}'
            body = json.dumps({
                "candidates": [{"content": {"parts": [{"text": decision}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5, "thoughtsTokenCount": 0},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = LLMClient("google", {"api_key": "local-test-not-a-secret", "base_url": f"http://127.0.0.1:{server.server_port}"})
    try:
        with patch("src.simulation.create_llm_client", return_value=client), contextlib.redirect_stdout(io.StringIO()):
            simulation = run_simulation(
                game=make_game(), model="google/gemini-3.6-flash", temperature=0.8,
                num_turns=2, num_agents=2, memory_capacity=3, agent_biases="",
                myth_writer=None, task_order=["game"],
                llm_settings=LLMSettings("direct", "minimal", None, max_output_tokens=1024),
                checkpoint_path=str(tmp_path / "run.checkpoint.json"), checkpoint_every=1,
            )
        final = tmp_path / "run.json"
        simulation.save_state(str(final))
        assert len(requests) == 4
        for request in requests:
            assert request["generationConfig"] == {"thinkingConfig": {"thinkingLevel": "minimal"}, "maxOutputTokens": 1024}
        state = load_simulation_runs([str(final)])[str(final)]
        condition_from_run(state)
        finals, artifacts = artifacts_for_completed_runs([final, tmp_path / "run.checkpoint.json"], data_root=tmp_path)
        assert finals == (final,)
        assert artifacts == (final,)
        document = build_provenance([final])
        validate_output_provenance(document)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
