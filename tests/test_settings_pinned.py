"""Repo-level guards for the LLM regime (researchlog 2026-09-04).

A PR check can only see what is in git, so these tests pin three things:

1. every experiment set in config/*.yaml carries a valid ``llm_settings``
   block, except a frozen allowlist of legacy sets;
2. every launch script references pinned experiment sets and does not export
   the legacy per-vendor reasoning knobs, except a frozen allowlist;
3. every committed output directory under data/analysis/ and docs/figures/
   carries provenance.json, with every differing condition field explicitly
   declared and its outputs hashed, except unchanged historical directories.

The allowlists are frozen: do not add to them. Remove entries as things get
pinned.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from src.llm_settings import LLMSettingsError, parse_llm_settings_block
from src.experiment_condition import ConditionMismatchError, validate_output_provenance
from scripts.write_provenance import output_hashes

REPO = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent
CONFIGS = tuple(str(path.relative_to(REPO)) for path in sorted((REPO / "config").glob("*.yaml")))
BASELINE_BYTES = (TESTS / "fixtures" / "legacy_settings_baseline.json").read_bytes()
assert hashlib.sha256(BASELINE_BYTES).hexdigest() == "d0bc771713afcf626785ff159140b28e9f94d3bf9b3eb29087b658e8d21607db"
LEGACY_BASELINE = json.loads(BASELINE_BYTES)
LEGACY_ENV_KNOBS = (
    "OPENAI_REASONING_EFFORT",
    "GEMINI_THINKING_LEVEL",
    "OPENROUTER_REASONING_EFFORT",
)


def _allowlist(name: str) -> set[str]:
    lines = (TESTS / name).read_text(encoding="utf-8").splitlines()
    current = {line.strip() for line in lines if line.strip() and not line.startswith("#")}
    frozen = set(LEGACY_BASELINE["allowlists"][name])
    assert current <= frozen, f"{name}: new exemptions are forbidden: {sorted(current - frozen)}"
    return current


def _definition_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _unchanged_legacy(path):
    changed = subprocess.run(["git", "diff", "--quiet", "--", path], cwd=REPO).returncode
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "--", path], cwd=REPO, check=True, capture_output=True, text=True).stdout
    indexed = subprocess.run(["git", "ls-files", "--stage", "-z", "--", path], cwd=REPO, check=True, capture_output=True, text=True).stdout
    files = {}
    for record in indexed.split("\0"):
        if record:
            info, filename = record.split("\t", 1)
            _, blob, stage = info.split()
            if stage != "0":
                return False
            files[filename] = blob
    return not changed and not untracked and _definition_digest(files) == LEGACY_BASELINE["paths"].get(path)


def _experiment_sets():
    out = {}
    for cfg in CONFIGS:
        data = yaml.safe_load((REPO / cfg).read_text(encoding="utf-8"))
        out[cfg] = data.get("experiment_sets", {}) or {}
    return out


# ---------------------------------------------------------------------------
# 1. Experiment sets
# ---------------------------------------------------------------------------

def test_every_new_experiment_set_pins_llm_settings():
    legacy = _allowlist("legacy_unpinned_experiment_sets.txt")
    problems = []
    for cfg, sets in _experiment_sets().items():
        for name, body in sets.items():
            key = f"{cfg}:{name}"
            block = body.get("llm_settings") if isinstance(body, dict) else None
            if block is None:
                if key not in legacy:
                    problems.append(f"{key}: missing llm_settings")
                continue
            try:
                parse_llm_settings_block(block, key)
            except LLMSettingsError as exc:
                problems.append(str(exc))
    assert not problems, "\n".join(problems)


def test_legacy_experiment_set_allowlist_only_shrinks():
    # Entries may be removed once pinned; nothing new may be added.
    legacy = _allowlist("legacy_unpinned_experiment_sets.txt")
    existing = {
        f"{cfg}:{name}" for cfg, sets in _experiment_sets().items() for name in sets
    }
    stale = sorted(legacy - existing)
    assert not stale, "allowlist names sets that no longer exist: " + ", ".join(stale)
    pinned_but_listed = sorted(
        f"{cfg}:{name}"
        for cfg, sets in _experiment_sets().items()
        for name, body in sets.items()
        if isinstance(body, dict) and "llm_settings" in body and f"{cfg}:{name}" in legacy
    )
    assert not pinned_but_listed, (
        "remove these pinned sets from legacy_unpinned_experiment_sets.txt: "
        + ", ".join(pinned_but_listed)
    )


def test_modified_legacy_sets_must_pin_settings():
    legacy = _allowlist("legacy_unpinned_experiment_sets.txt")
    for config_path, sets in _experiment_sets().items():
        for name, body in sets.items():
            if f"{config_path}:{name}" in legacy:
                assert _definition_digest(body) == LEGACY_BASELINE["experiment_sets"][f"{config_path}:{name}"], f"{config_path}:{name}: modified historical sets need llm_settings"


# ---------------------------------------------------------------------------
# 2. Launch scripts
# ---------------------------------------------------------------------------

_SET_REF = re.compile(
    r"run_(?:noisy|trust_game)_batch\.py\s+(?:--\S+\s+\S+\s+)*([A-Za-z0-9_]+)"
)


def _launch_scripts():
    return sorted(
        p for pattern in ("launch_*.sh", "run_*.sh") for p in (REPO / "scripts").glob(pattern)
    )


def test_new_launch_scripts_are_pinned():
    legacy = _allowlist("legacy_unpinned_launch_scripts.txt")
    pinned_sets = {
        name
        for sets in _experiment_sets().values()
        for name, body in sets.items()
        if isinstance(body, dict) and "llm_settings" in body
    }
    problems = []
    for script in _launch_scripts():
        if script.name in legacy and _unchanged_legacy(str(script.relative_to(REPO))):
            continue
        text = script.read_text(encoding="utf-8")
        for knob in LEGACY_ENV_KNOBS:
            if re.search(rf"^\s*(export\s+)?{knob}=", text, re.M):
                problems.append(
                    f"{script.name}: exports {knob}; pin reasoning in llm_settings instead"
                )
        refs = _SET_REF.findall(text)
        for ref in refs:
            if ref not in pinned_sets:
                problems.append(
                    f"{script.name}: runs experiment set {ref!r} which has no llm_settings"
                )
        if not refs:
            # A launcher that never names a set cannot be checked; flag it.
            problems.append(f"{script.name}: no experiment set reference found")
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# 3. Committed outputs
# ---------------------------------------------------------------------------

def _output_dirs():
    for base in ("data/analysis", "docs/figures"):
        root = REPO / base
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir():
                yield f"{base}/{child.name}", child


def test_new_output_dirs_carry_provenance():
    legacy = _allowlist("legacy_unprovenanced_output_dirs.txt")
    problems = []
    for rel, path in _output_dirs():
        if rel in legacy and _unchanged_legacy(rel):
            continue
        prov = path / "provenance.json"
        if not prov.is_file():
            problems.append(f"{rel}: missing provenance.json (scripts/write_provenance.py)")
            continue
        try:
            doc = json.loads(prov.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"{rel}: provenance.json is not valid JSON ({exc})")
            continue
        try:
            validate_output_provenance(doc)
            if doc.get("outputs") != output_hashes(path):
                problems.append(f"{rel}: output content differs from provenance; regenerate it")
        except (ConditionMismatchError, TypeError, ValueError) as exc:
            problems.append(f"{rel}: {exc}")
    assert not problems, "\n".join(problems)


def test_output_dir_allowlist_only_shrinks():
    legacy = _allowlist("legacy_unprovenanced_output_dirs.txt")
    existing = {rel for rel, _ in _output_dirs()}
    stale = sorted(legacy - existing)
    assert not stale, "allowlist names directories that no longer exist: " + ", ".join(stale)


@pytest.mark.parametrize("name", [
    "legacy_unpinned_experiment_sets.txt",
    "legacy_unpinned_launch_scripts.txt",
    "legacy_unprovenanced_output_dirs.txt",
])
def test_allowlists_have_no_duplicates(name):
    lines = [
        line.strip()
        for line in (TESTS / name).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(lines) == len(set(lines))


def test_declared_comparisons_match_resolved_inputs():
    from experiments.run_noisy_batch import NoisyExperimentConfig
    from src.experiment_config import ExperimentConfig
    from src.comparison_config import validate_config_comparisons

    for config_path in CONFIGS:
        raw = yaml.safe_load((REPO / config_path).read_text())
        if not raw.get("comparison_sets"):
            continue
        loader = NoisyExperimentConfig if config_path.endswith("experiments_noisy.yaml") else ExperimentConfig
        validate_config_comparisons(loader(str(REPO / config_path)), environ={})


def test_allowlist_cannot_gain_an_exemption(tmp_path, monkeypatch):
    name = "legacy_unpinned_experiment_sets.txt"
    (tmp_path / name).write_text("config/experiments.yaml:brand_new_unpinned_set\n")
    monkeypatch.setitem(globals(), "TESTS", tmp_path)
    with pytest.raises(AssertionError, match="new exemptions"):
        _allowlist(name)


def test_new_launcher_cannot_bypass_checks_with_a_function_name(tmp_path, monkeypatch):
    script = tmp_path / "launch_unpinned.sh"
    script.write_text("python -c 'from src.simulation import run_simulation'\n")
    monkeypatch.setitem(globals(), "_launch_scripts", lambda: [script])
    with pytest.raises(AssertionError, match="no experiment set reference"):
        test_new_launch_scripts_are_pinned()
