#!/usr/bin/env python3
"""Write provenance.json for a committed analysis or figure directory.

Every new directory under data/analysis/ or docs/figures/ must say which runs
it was computed from and which LLM regime (provider, reasoning level,
temperature policy) those runs used, so a reader can tell whether a
cross-model figure compares matched conditions (researchlog 2026-09-04).
tests/test_settings_pinned.py enforces the file's presence and refuses mixed
regimes within one model unless ``mixed_settings_acknowledged`` is set.

Usage (from repo root):
  python scripts/write_provenance.py <out_dir> <run.json or dir> [more ...]
  python scripts/write_provenance.py docs/figures/my_fig data/json/some_set --acknowledge-mixed "why"

The output lists one entry per model with the distinct settings signatures
found and the run count per signature.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analyses._shared import llm_settings_signature  # noqa: E402
from scripts.hf_sync_completed_runs import _is_non_final_name  # noqa: E402

PROVENANCE_VERSION = 1


def iter_run_jsons(paths):
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for f in sorted(p.rglob("*.json")):
                if not _is_non_final_name(f):
                    yield f
        elif p.suffix == ".json" and not _is_non_final_name(p):
            yield p


def build_provenance(run_paths, acknowledge_mixed=None):
    by_model = {}
    n_runs = 0
    for path in run_paths:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or "run_metadata" not in data:
            continue
        sig = llm_settings_signature(data)
        model = str(sig["model"])
        key = json.dumps(
            {k: sig[k] for k in ("provider", "reasoning", "temperature")}, sort_keys=True
        )
        entry = by_model.setdefault(model, {})
        bucket = entry.setdefault(key, {"settings": json.loads(key), "runs": 0, "example": None})
        bucket["runs"] += 1
        bucket["example"] = bucket["example"] or os.path.relpath(path, REPO_ROOT)
        n_runs += 1
    models = {
        model: {"signatures": list(buckets.values())} for model, buckets in by_model.items()
    }
    mixed = sorted(m for m, b in by_model.items() if len(b) > 1)
    doc = {
        "provenance_version": PROVENANCE_VERSION,
        "written_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_runs": n_runs,
        "models": models,
        "mixed_settings_models": mixed,
    }
    if acknowledge_mixed:
        doc["mixed_settings_acknowledged"] = acknowledge_mixed
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out_dir")
    parser.add_argument("runs", nargs="+", help="run JSON files or directories")
    parser.add_argument(
        "--acknowledge-mixed",
        metavar="REASON",
        help="record that mixed regimes within a model are deliberate",
    )
    args = parser.parse_args()
    doc = build_provenance(iter_run_jsons(args.runs), args.acknowledge_mixed)
    if doc["n_runs"] == 0:
        sys.exit("No run JSONs found; nothing written.")
    out = Path(args.out_dir) / "provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({doc['n_runs']} runs, {len(doc['models'])} model(s))")
    if doc["mixed_settings_models"] and not args.acknowledge_mixed:
        print(
            "WARNING: mixed LLM settings within "
            + ", ".join(doc["mixed_settings_models"])
            + "; the pinned-settings test will fail unless you pass --acknowledge-mixed."
        )


if __name__ == "__main__":
    main()
