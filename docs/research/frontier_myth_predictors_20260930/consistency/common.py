"""Dataset switch for the ported consistency lens (September scripts in
docs/research/myth_predictors_20260930/consistency/).

LINGUISTIC_DATASET=september|frontier (default september). Outputs go to <this folder>/<dataset>/.
Both corpora are read only.
"""
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parents[3]
NAME = os.environ.get("LINGUISTIC_DATASET", "september")
assert NAME in ("september", "frontier"), NAME
OUT = HERE / NAME
OUT.mkdir(exist_ok=True)

if NAME == "september":
    DATA = WT / "data/analysis/linguistic_20260923"
    RULES_GLM = DATA / "myth_rules_september_z-ai__glm-5.2.csv"
    RULES_DS = DATA / "myth_rules_september_deepseek__deepseek-v4-flash_sample1000.csv"
    # the September lens re-embedded locally (the shared cache has no fingerprint stamp)
    EMB_LOCAL = WT.parent / "myth-predictors/data/analysis/myth_predictors_20260930/consistency/embeddings_mpnet_local.npy"
    FAMILIES = ["Sonnet", "GPT", "Gemini"]
    CEILING_SEND = ["Gemini"]           # sends $5 in 99.7% of decisions
    COLORS = {"Sonnet": "#7b3294", "GPT": "#1b7837", "Gemini": "#d6604d"}
else:
    DATA = WT / "data/analysis/linguistic_frontier_20260930"
    RULES_GLM = DATA / "myth_rules_frontier_z-ai__glm-5.2.csv"
    RULES_DS = DATA / "myth_rules_frontier_deepseek__deepseek-v4-flash.csv"
    EMB_LOCAL = None
    FAMILIES = ["Opus", "Sol", "GeminiPro"]
    CEILING_SEND = ["GeminiPro"]        # sends $5 with sd 0 everywhere (testability_cells.csv)
    COLORS = {"Opus": "#7b3294", "Sol": "#1b7837", "GeminiPro": "#d6604d"}

SETTINGS = ["2-agent homogeneous", "8-agent homogeneous", "2-agent mixed", "8-agent mixed"]


def setting_of(df):
    return df["size"].astype(str) + "-agent " + np.where(df["mixed"], "mixed", "homogeneous")
