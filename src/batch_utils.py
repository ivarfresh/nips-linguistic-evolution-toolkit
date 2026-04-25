"""Filesystem helpers shared between batch runners.

`ExperimentConfig` and `NoisyExperimentConfig` have genuinely different YAML
schemas (num_runs, game_params_list, etc.), so they stay separate. These two
path helpers, however, were byte-for-byte identical in both runners and belong
in one place.
"""

import os


def unique_json_path(path: str) -> str:
    """If `path` exists, return `..._2.json`, `..._3.json`, ... (first free slot)."""
    if not path.endswith(".json"):
        raise ValueError(f"Expected a .json path, got: {path}")
    if not os.path.exists(path):
        return path
    base = path[:-5]
    n = 2
    while True:
        candidate = f"{base}_{n}.json"
        if not os.path.exists(candidate):
            return candidate
        n += 1


def sanitize_for_filename(value: str) -> str:
    """Make a string safe-ish for filenames across OSes.

    Keeps alphanumerics, dash, underscore, and dot; maps everything else to
    underscore. Empty / None inputs become the literal strings "empty" / "none".
    """
    if value is None:
        return "none"
    value = str(value).strip()
    if value == "":
        return "empty"
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in value)
