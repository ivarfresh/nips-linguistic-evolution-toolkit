#!/usr/bin/env python3
"""Write checked input conditions and content hashes beside analysis outputs."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.experiment_condition import (
    NON_FINAL_SUFFIXES, check_conditions, comparison_condition, read_final_run,
    validate_output_provenance,
)


def iter_run_jsons(paths):
    seen = set()
    for raw in paths:
        path = Path(raw)
        candidates = sorted(path.rglob("*.json")) if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.suffix == ".json" and not candidate.name.endswith(NON_FINAL_SUFFIXES):
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved


def output_hashes(directory):
    directory = Path(directory)
    return [
        {"path": str(path.relative_to(directory)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in {"provenance.json", ".DS_Store"}
    ]


def build_provenance(run_paths, *, allowed_differences=None, legacy_reason=None, output_dir=None):
    runs = []
    for path in run_paths:
        path = Path(path)
        data = read_final_run(path)
        runs.append({
            "path": os.path.relpath(path, REPO_ROOT),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "condition": comparison_condition(data, legacy_reason),
        })
    observed = check_conditions([run["condition"] for run in runs], allowed_differences)
    document = {
        "provenance_version": 2,
        "n_runs": len(runs),
        "runs": runs,
        "allowed_differences": allowed_differences or {},
        "observed_differences": observed,
        "outputs": output_hashes(output_dir) if output_dir else [],
    }
    if legacy_reason:
        document["legacy_reason"] = legacy_reason
    validate_output_provenance(document)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--allow-difference", action="append", default=[], metavar="FIELD=REASON")
    parser.add_argument("--legacy-reason", help="Explicitly document missing historical provenance")
    args = parser.parse_args()
    allowed = {}
    for value in args.allow_difference:
        field, separator, reason = value.partition("=")
        if not separator or not field or not reason.strip():
            parser.error("--allow-difference requires FIELD=REASON")
        allowed[field] = reason
    document = build_provenance(
        iter_run_jsons(args.runs), allowed_differences=allowed,
        legacy_reason=args.legacy_reason, output_dir=args.out_dir,
    )
    output = Path(args.out_dir) / "provenance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} ({document['n_runs']} runs)")


if __name__ == "__main__":
    main()
