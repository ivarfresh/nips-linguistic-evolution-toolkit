from scripts import run_noisy_missing


def test_load_combinations_embeds_execution_provenance(monkeypatch, tmp_path):
    config_path = tmp_path / "experiments.yaml"
    config_path.write_text("experiment_sets: {}\n", encoding="utf-8")

    class FakeConfig:
        def __init__(self, path):
            assert path == str(config_path)

        def get_experiment_combinations(self, experiment_name):
            assert experiment_name == "example"
            return [{"index": 0}, {"index": 1}]

    provenance = {
        "execution_provenance_version": 1,
        "code_commit": "abc123",
        "code_dirty": False,
        "config_sha256": "feedface",
    }
    monkeypatch.setattr(run_noisy_missing, "NoisyExperimentConfig", FakeConfig)
    monkeypatch.setattr(
        run_noisy_missing,
        "execution_provenance",
        lambda path: provenance,
    )

    combinations = run_noisy_missing.load_combinations(
        "example",
        str(config_path),
    )

    assert [item["execution_provenance"] for item in combinations] == [
        provenance,
        provenance,
    ]
    assert combinations[0]["execution_provenance"] is not combinations[1][
        "execution_provenance"
    ]
