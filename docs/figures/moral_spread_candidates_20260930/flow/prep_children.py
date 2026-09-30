"""Rebuild the moral-uptake child table with labels attached (shown, child, unseen).

Same exposures and unseen comparison myths as analyses/moral_carryover.py
summary_measures(); asserts that the per-run summary reproduces moral_uptake*.csv.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/moral-split")
sys.path.insert(0, str(REPO))
from analyses.linguistic_uptake import load_myths, null_candidates  # noqa: E402
from analyses.moral_carryover import load, run_summary  # noqa: E402

OUT = Path(__file__).resolve().parent
LABELS = ["be generous", "be fair", "be cautious"]


def children() -> pd.DataFrame:
    myths, _ = load("moral_labels_z-ai__glm-5.2.csv")
    has = myths["summary"].notna().to_numpy()
    rows = []
    for i, js in null_candidates(myths).items():
        p, nulls = js[0], [j for j in js[1:] if has[j]]
        if not (has[i] and has[p]) or not nulls:
            continue
        lab_i, lab_p = myths.at[i, "label"], myths.at[p, "label"]
        null_labs = [myths.at[j, "label"] for j in nulls if isinstance(myths.at[j, "label"], str)]
        rec = {"run_id": myths.at[i, "run_id"], "setting": myths.at[i, "setting"],
               "composition": myths.at[i, "composition"],
               "family": myths.at[i, "family"], "parent_family": myths.at[p, "family"],
               "round": myths.at[i, "round"], "task_order": myths.at[i, "task_order"],
               "child_label": lab_i, "shown_label": lab_p,
               "same_label_shown": float(lab_i == lab_p) if isinstance(lab_i, str) and isinstance(lab_p, str) else np.nan,
               "same_label_unseen": np.nanmean([float(lab_i == l) for l in null_labs]) if isinstance(lab_i, str) else np.nan}
        for lab in LABELS:  # share of the unseen comparison myths carrying each label
            rec[f"unseen_share_{lab}"] = np.mean([l == lab for l in null_labs]) if null_labs else np.nan
        rows.append(rec)
    d = pd.DataFrame(rows)
    d["same_label_excess"] = d["same_label_shown"] - d["same_label_unseen"]
    d["exposure"] = np.where(d["family"] == d["parent_family"], "same family", "other family")
    return d


if __name__ == "__main__":
    d = children()
    d.to_csv(OUT / "children_labels.csv", index=False)
    ref = pd.read_csv(REPO / "docs/figures/linguistic_analysis_20260923/moral_uptake_by_task_order.csv")
    mine = run_summary(d, ["setting", "exposure", "task_order"], ["same_label_excess"])
    m = ref.merge(mine, on=["setting", "exposure", "task_order"], suffixes=("_ref", ""))
    for c in ["n_runs", "same_label_excess_mean", "same_label_excess_p"]:
        assert np.allclose(m[f"{c}_ref"], m[c]), c
    ref2 = pd.read_csv(REPO / "docs/figures/linguistic_analysis_20260923/moral_uptake.csv")
    mine2 = run_summary(d, ["setting", "exposure"], ["same_label_shown", "same_label_unseen", "same_label_excess"])
    m2 = ref2.merge(mine2, on=["setting", "exposure"], suffixes=("_ref", ""))
    for c in ["same_label_shown_mean", "same_label_unseen_mean", "same_label_excess_mean"]:
        assert np.allclose(m2[f"{c}_ref"], m2[c]), c
    print("reproduces moral_uptake.csv and moral_uptake_by_task_order.csv;", len(d), "children")
