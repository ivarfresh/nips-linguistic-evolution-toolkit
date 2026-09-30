#!/usr/bin/env python3
"""Seconds-long check that spread.py's September outputs reproduce the committed tables
(docs/figures/linguistic_analysis_20260923): item-1 word adoption (reuse_summary.csv), moral
label uptake by task order (+4.8 / +5.9 / +0.7; moral_uptake_by_task_order.csv) and the
carryover models (moral_carryover_models.csv). Run after spread.py --dataset september."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from analyses.linguistic_uptake import run_level_summary  # noqa: E402
from analyses.moral_carryover import run_summary  # noqa: E402

REF = ROOT / "docs/figures/linguistic_analysis_20260923"
S = HERE / "september"

ours = run_level_summary(pd.read_csv(S / "word_uptake_children.csv"), ["size", "exposure"])
m = pd.read_csv(REF / "reuse_summary.csv").merge(ours, on=["size", "exposure"], suffixes=("_ref", "_ours"))
assert len(m) == 5
for c in ["n_runs", "adopt_parent_mean", "adopt_null_mean", "adopt_excess_mean", "cos_excess_mean"]:
    assert np.allclose(m[c + "_ref"], m[c + "_ours"]), c
print("word adoption, shown vs unseen (%):")
print((m[["size", "exposure", "n_runs_ours"]].assign(shown=100 * m["adopt_parent_mean_ours"], unseen=100 * m["adopt_null_mean_ours"])
       .round(1).to_string(index=False)))

up = pd.read_csv(S / "moral_uptake_children.csv")
ours = run_summary(up, ["setting", "exposure", "task_order"], ["moral_cos_excess", "same_label_excess"])
ref = pd.read_csv(REF / "moral_uptake_by_task_order.csv")
m = ref.merge(ours, on=["setting", "exposure", "task_order"], suffixes=("_ref", "_ours"))
assert len(m) == len(ref) == len(ours)
for c in ["n_runs", "same_label_excess_mean", "same_label_excess_p", "moral_cos_excess_mean"]:
    assert np.allclose(m[c + "_ref"], m[c + "_ours"]), c
k = m[m["setting"].str.startswith("8-agent") & (m["task_order"] == "myth_game")]
print("\n8-agent myth→game label uptake (points):")
print(k[["setting", "exposure", "n_runs_ours"]].assign(pts=100 * k["same_label_excess_mean_ours"], p=k["same_label_excess_p_ours"])
      .round(4).to_string(index=False))

key = ["fe", "setting", "role", "model", "predictor", "level"]
ref = pd.read_csv(REF / "moral_carryover_models.csv").dropna(subset=["coef"])
m = ref.merge(pd.read_csv(S / "carryover_models_pooled.csv"), on=key, suffixes=("_ref", "_ours"))
assert len(m) == len(ref) and np.allclose(m["coef_ref"], m["coef_ours"]) and np.allclose(m["p_ref"], m["p_ours"])
print(f"\ncarryover models: {len(m)}/{len(ref)} coefficients and p-values match")
print("ALL SEPTEMBER REPRODUCTION CHECKS PASS")
