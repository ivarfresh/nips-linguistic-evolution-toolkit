#!/usr/bin/env python3
"""Report completion status for a configured noisy experiment set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_noisy_missing import expected_output_path, load_combinations  # noqa: E402


def count_status(experiment_name: str, output_subdir: str, config: str | None) -> dict[str, object]:
    combinations = load_combinations(experiment_name, config)
    complete: list[Path] = []
    missing: list[Path] = []
    for index, combo in enumerate(combinations):
        path = expected_output_path(combo, experiment_name, index, output_subdir)
        if path.exists():
            complete.append(path)
        else:
            missing.append(path)

    root = PROJECT_ROOT / "data" / "json" / "noise_experiments" / output_subdir / experiment_name
    errors = sorted(root.rglob("*.error.json")) if root.exists() else []
    checkpoints = sorted(root.rglob("*.checkpoint.json")) if root.exists() else []

    return {
        "experiment": experiment_name,
        "output_subdir": output_subdir,
        "total": len(combinations),
        "complete": len(complete),
        "missing": len(missing),
        "errors": len(errors),
        "checkpoints": len(checkpoints),
        "missing_paths": missing,
        "error_paths": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_name")
    parser.add_argument("--output-subdir", default="v2")
    parser.add_argument("--config", default=None)
    parser.add_argument("--show-missing", type=int, default=8)
    parser.add_argument("--show-errors", type=int, default=8)
    args = parser.parse_args()

    status = count_status(args.experiment_name, args.output_subdir, args.config)
    print(
        f"{status['experiment']} ({status['output_subdir']}): "
        f"complete={status['complete']}/{status['total']} "
        f"missing={status['missing']} errors={status['errors']} "
        f"checkpoints={status['checkpoints']}"
    )

    missing_paths = status["missing_paths"]
    if args.show_missing and missing_paths:
        print("first missing:")
        for path in missing_paths[: args.show_missing]:
            print(f"  {path.relative_to(PROJECT_ROOT)}")

    error_paths = status["error_paths"]
    if args.show_errors and error_paths:
        print("error checkpoints:")
        for path in error_paths[: args.show_errors]:
            print(f"  {path.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
