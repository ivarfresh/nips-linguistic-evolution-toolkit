import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import hf_sync_completed_runs as hf_sync


FULL_STATE = {
    "agents": {},
    "conversation_history": [],
    "game_data": {},
    "task_order": ["game"],
    "run_metadata": {},
}


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_discovery_accepts_only_full_final_state_json(tmp_path):
    data_root = tmp_path / "data" / "json"
    final = write_json(data_root / "experiment" / "run.json", FULL_STATE)
    write_json(data_root / "experiment" / "run.results.json", FULL_STATE)
    write_json(data_root / "experiment" / "run.checkpoint.json", FULL_STATE)
    write_json(data_root / "experiment" / "run.checkpoint.json.error.json", FULL_STATE)
    write_json(
        data_root / "experiment" / "analysis.json",
        {"metadata": {}, "trials": []},
    )
    invalid = data_root / "experiment" / "truncated.json"
    invalid.write_text("{", encoding="utf-8")

    discovered = hf_sync.discover_completed_final_jsons(
        [data_root],
        data_root=data_root,
    )

    assert discovered == (final.resolve(),)


def test_plan_allows_only_artifacts_gated_by_a_final_json(tmp_path):
    data_root = tmp_path / "data" / "json"
    run_dir = data_root / "noise_experiments" / "example" / "model"
    final = write_json(run_dir / "run.json", FULL_STATE)
    results = write_json(run_dir / "run.results.json", {"game_data": {}})
    log = run_dir / "run.log"
    log.write_text("complete", encoding="utf-8")
    transcript = run_dir / "run.transcript.pdf"
    transcript.write_bytes(b"%PDF-complete")

    write_json(run_dir / "failed.results.json", {"game_data": {}})
    (run_dir / "failed.log").write_text("partial", encoding="utf-8")
    write_json(run_dir / "failed.checkpoint.json.error.json", FULL_STATE)

    plan = hf_sync.build_upload_plan(
        [final],
        repo_id="owner/dataset",
        data_root=data_root,
    )

    assert plan.final_paths == (final.resolve(),)
    assert set(plan.artifact_paths) == {
        final.resolve(),
        results.resolve(),
        log.resolve(),
        transcript.resolve(),
    }
    assert plan.folder_path == run_dir.resolve()
    assert plan.path_in_repo == "noise_experiments/example/model"
    assert set(plan.allow_patterns) == {
        "run.json",
        "run.results.json",
        "run.log",
        "run.transcript.pdf",
    }
    assert not any("failed" in path for path in plan.remote_paths)


def test_plan_rejects_symlinked_companion_outside_data_root(tmp_path):
    data_root = tmp_path / "data" / "json"
    run_dir = data_root / "experiment"
    final = write_json(run_dir / "run.json", FULL_STATE)
    outside_secret = tmp_path / "outside.env"
    outside_secret.write_text("HF_TOKEN=do-not-upload", encoding="utf-8")
    symlinked_log = run_dir / "run.log"
    symlinked_log.symlink_to(outside_secret)

    plan = hf_sync.build_upload_plan(
        [final],
        repo_id="owner/dataset",
        data_root=data_root,
    )

    assert plan.artifact_paths == (final.resolve(),)
    assert symlinked_log not in plan.artifact_paths
    assert outside_secret.resolve() not in plan.artifact_paths


def test_sync_uses_one_locked_exact_folder_upload(tmp_path):
    data_root = tmp_path / "data" / "json"
    final = write_json(data_root / "experiment" / "run.json", FULL_STATE)
    log = data_root / "experiment" / "run.log"
    log.write_text("complete", encoding="utf-8")
    calls = []

    class FakeApi:
        def repo_info(self, **kwargs):
            assert kwargs == {"repo_id": "owner/dataset", "repo_type": "dataset"}
            return SimpleNamespace(sha="remote-head", private=True)

        def upload_folder(self, **kwargs):
            calls.append(kwargs)

    lock_path = tmp_path / "sync.lock"
    plan = hf_sync.sync_completed_runs(
        [final],
        repo_id="owner/dataset",
        data_root=data_root,
        lock_path=lock_path,
        api=FakeApi(),
        label="example",
    )

    assert lock_path.exists()
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "owner/dataset"
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["folder_path"] == str(plan.folder_path)
    assert calls[0]["path_in_repo"] == "experiment"
    assert set(calls[0]["allow_patterns"]) == {"run.json", "run.log"}
    assert calls[0]["parent_commit"] == "remote-head"


@pytest.mark.parametrize("status_code", [409, 412])
def test_remote_conflict_refreshes_head_and_retries(tmp_path, status_code):
    data_root = tmp_path / "data" / "json"
    final = write_json(data_root / "experiment" / "run.json", FULL_STATE)

    class ConflictError(RuntimeError):
        def __init__(self, status_code):
            super().__init__(f"conflict {status_code}")
            self.response = SimpleNamespace(status_code=status_code)

    class FakeApi:
        def __init__(self):
            self.heads = iter(["old-head", "new-head"])
            self.uploads = []

        def repo_info(self, **kwargs):
            return SimpleNamespace(sha=next(self.heads), private=True)

        def upload_folder(self, **kwargs):
            self.uploads.append(kwargs)
            if len(self.uploads) == 1:
                raise ConflictError(status_code)

    api = FakeApi()
    hf_sync.sync_completed_runs(
        [final],
        repo_id="owner/dataset",
        data_root=data_root,
        lock_path=tmp_path / "sync.lock",
        api=api,
    )

    assert [call["parent_commit"] for call in api.uploads] == [
        "old-head",
        "new-head",
    ]


def test_non_conflict_upload_error_is_not_retried(tmp_path):
    data_root = tmp_path / "data" / "json"
    final = write_json(data_root / "experiment" / "run.json", FULL_STATE)

    class AuthenticationError(RuntimeError):
        response = SimpleNamespace(status_code=401)

    class FakeApi:
        def __init__(self):
            self.repo_info_calls = 0

        def repo_info(self, **kwargs):
            self.repo_info_calls += 1
            return SimpleNamespace(sha="remote-head", private=True)

        def upload_folder(self, **kwargs):
            raise AuthenticationError("denied")

    api = FakeApi()
    with pytest.raises(AuthenticationError, match="denied"):
        hf_sync.sync_completed_runs(
            [final],
            repo_id="owner/dataset",
            data_root=data_root,
            lock_path=tmp_path / "sync.lock",
            api=api,
        )

    assert api.repo_info_calls == 1


def test_sync_refuses_public_dataset_by_default(tmp_path):
    data_root = tmp_path / "data" / "json"
    final = write_json(data_root / "experiment" / "run.json", FULL_STATE)

    class FakeApi:
        def repo_info(self, **kwargs):
            return SimpleNamespace(sha="remote-head", private=False)

        def upload_folder(self, **kwargs):
            raise AssertionError("public dataset must not receive an upload")

    with pytest.raises(RuntimeError, match="refusing to upload to public dataset"):
        hf_sync.sync_completed_runs(
            [final],
            repo_id="owner/dataset",
            data_root=data_root,
            lock_path=tmp_path / "sync.lock",
            api=FakeApi(),
        )


def test_repo_env_loader_targets_project_dotenv(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(hf_sync, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("dotenv.load_dotenv", lambda path: calls.append(path))

    hf_sync._load_repo_env()

    assert calls == [tmp_path / ".env"]


def test_automatic_hook_is_noop_unless_enabled(monkeypatch):
    called = []
    monkeypatch.setattr(
        hf_sync,
        "sync_completed_runs",
        lambda *args, **kwargs: called.append((args, kwargs)),
    )

    result = hf_sync.maybe_sync_completed_runs(
        [],
        environ={"HF_DATASET_REPO": "owner/dataset"},
    )

    assert result is False
    assert called == []


def test_automatic_hook_warns_and_swallows_upload_failure(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise RuntimeError("authentication failed")

    monkeypatch.setattr(hf_sync, "sync_completed_runs", fail)

    result = hf_sync.maybe_sync_completed_runs(
        [],
        environ={
            "HF_DATASET_AUTO_UPLOAD": "1",
            "HF_DATASET_REPO": "owner/dataset",
        },
    )

    assert result is False
    assert "authentication failed" in capsys.readouterr().err


def test_dry_run_cli_needs_no_hugging_face_authentication(tmp_path, capsys):
    data_root = tmp_path / "data" / "json"
    final = write_json(data_root / "experiment" / "run.json", FULL_STATE)

    exit_status = hf_sync.main(
        [
            str(final),
            "--data-root",
            str(data_root),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert exit_status == 0
    assert "completed_runs=1 artifacts=1" in output
    assert "experiment/run.json" in output
