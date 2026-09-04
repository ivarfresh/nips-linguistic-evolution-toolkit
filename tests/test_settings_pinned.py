"""Repo-level guards for the LLM regime (researchlog 2026-09-04).

A PR check can only see what is in git, so these tests pin three things:

1. every experiment set in config/*.yaml carries a valid ``llm_settings``
   block, except a frozen allowlist of legacy sets;
2. every launch script references pinned experiment sets and does not export
   the legacy per-vendor reasoning knobs, except a frozen allowlist;
3. every committed output directory under data/analysis/ and docs/figures/
   carries provenance.json, and no model inside it mixes regimes without an
   explicit acknowledgement, except a frozen allowlist.

The allowlists are frozen: do not add to them. Remove entries as things get
pinned.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from src.llm_settings import LLMSettingsError, parse_llm_settings_block

REPO = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent
CONFIGS = ("config/experiments.yaml", "config/experiments_noisy.yaml")
LEGACY_ENV_KNOBS = (
    "OPENAI_REASONING_EFFORT",
    "GEMINI_THINKING_LEVEL",
    "OPENROUTER_REASONING_EFFORT",
)


def _allowlist(name: str) -> set[str]:
    lines = (TESTS / name).read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


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
        if script.name in legacy:
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
        if not refs and "run_simulation" not in text:
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
        if rel in legacy:
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
        for key in ("provenance_version", "n_runs", "models", "mixed_settings_models"):
            if key not in doc:
                problems.append(f"{rel}: provenance.json lacks {key!r}")
        if doc.get("mixed_settings_models") and not doc.get("mixed_settings_acknowledged"):
            problems.append(
                f"{rel}: runs mix LLM settings within "
                + ", ".join(doc["mixed_settings_models"])
                + " and no mixed_settings_acknowledged reason is given"
            )
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
