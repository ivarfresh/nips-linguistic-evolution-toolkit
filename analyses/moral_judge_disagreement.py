#!/usr/bin/env python3
"""Where the two moral judges disagree, split by task order.

Reads the existing labels (docs/figures/linguistic_analysis_20260923/moral_labels.csv:
GLM-5.2 and DeepSeek V4 Flash, Arabella Sinclair's three-moral rubric) and asks
whether disagreement is random noise (a bad judge) or concentrated on one
category boundary (overlapping categories). Everything is reported per task
order, never pooled.

Outputs (docs/figures/myth_text_predictiveness_20260930/):
  judge_agreement_by_task_order.csv   agreement, Cohen's kappa, ordinal (quadratic)
                                      kappa and label shares per task order x author
                                      family and per task order x setting
  judge_confusion_by_task_order.csv   GLM x DeepSeek counts per task order
No API calls.

  python3 analyses/moral_judge_disagreement.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from myth_text_predictiveness import write_provenance  # noqa: E402
LABELS = ROOT / "docs/figures/linguistic_analysis_20260923/moral_labels.csv"
OUT = ROOT / "docs/figures/myth_text_predictiveness_20260930"
A, B = "label_glm_5_2", "label_deepseek_v4_flash"
ORDER = {"be cautious": 0, "be fair": 1, "be generous": 2}


def summarise(d: pd.DataFrame) -> dict:
    a, b = d[A].map(ORDER), d[B].map(ORDER)
    dis = a != b
    return {"n": len(d), "agreement": (a == b).mean(), "kappa": cohen_kappa_score(a, b),
            "ordinal_kappa": cohen_kappa_score(a, b, weights="quadratic"),
            "share_of_disagreements_fair_vs_generous": ((a + b == 3) & dis).sum() / dis.sum() if dis.sum() else float("nan"),
            **{f"glm_{k.split()[1]}": (d[A] == k).mean() for k in ORDER},
            **{f"deepseek_{k.split()[1]}": (d[B] == k).mean() for k in ORDER}}


def main() -> None:
    m = pd.read_csv(LABELS).dropna(subset=[A, B])
    m["setting"] = m["size"].astype(str) + "-agent " + m["mixed"].map({True: "mixed", False: "homogeneous"})
    rows = []
    for t, d in m.groupby("task_order"):
        rows.append({"task_order": t, "subset": "all", **summarise(d)})
        for f, g in d.groupby("family"):
            rows.append({"task_order": t, "subset": f"author {f}", **summarise(g)})
        for s, g in d.groupby("setting"):
            rows.append({"task_order": t, "subset": s, **summarise(g)})
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "judge_agreement_by_task_order.csv", index=False)
    print(table.round(3).to_string(index=False))
    conf = (m.groupby(["task_order", A, B]).size().rename("n").reset_index()
            .rename(columns={A: "glm_5_2", B: "deepseek_v4_flash"}))
    conf.to_csv(OUT / "judge_confusion_by_task_order.csv", index=False)
    for t, d in m.groupby("task_order"):
        print(f"\n{t}\n{pd.crosstab(d[A], d[B], margins=True)}")
    write_provenance()


if __name__ == "__main__":
    main()
