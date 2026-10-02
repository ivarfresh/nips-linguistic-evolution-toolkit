"""Dataset switch for the ported search lens (September scripts in
docs/research/myth_predictors_20260930/search/). MP_DATASET=september|frontier (default frontier).

Every script reads the shared linguistic tables read-only and writes to search/<dataset>/, so the
September folders are never touched. Family names are data-driven: a (family, role) cell whose
outcome has no spread is reported as "not estimable: ceiling" instead of being dropped by name.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parents[3]  # worktree root
NAME = os.environ.get("MP_DATASET", "frontier")
CFG = {
    "september": {
        "data": WT / "data/analysis/linguistic_20260923",
        "n_myths": 8520,
        "rules": "myth_rules_september_z-ai__glm-5.2.csv",
        "families": ["Sonnet", "GPT", "Gemini"],
        "style": True,
    },
    "frontier": {
        "data": WT / "data/analysis/linguistic_frontier_20260930",
        "n_myths": 4520,
        "rules": "myth_rules_frontier_z-ai__glm-5.2.csv",
        "families": ["Opus", "Sol", "GeminiPro"],
        "style": False,  # no style classifier for the frontier families
    },
}[NAME]
D = CFG["data"]
OUT = HERE / NAME
OUT.mkdir(exist_ok=True)
SEPT_SEARCH = WT.parent / "myth-predictors/docs/research/myth_predictors_20260930/search"


def ceiling(y) -> bool:
    """No usable spread: sd below 0.02 (a tenth of a dollar on send/5) or >= 97% at one value."""
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    if len(y) < 10:
        return True
    _, c = np.unique(np.round(y, 3), return_counts=True)
    return y.std() < 0.02 or c.max() / len(y) >= 0.97
