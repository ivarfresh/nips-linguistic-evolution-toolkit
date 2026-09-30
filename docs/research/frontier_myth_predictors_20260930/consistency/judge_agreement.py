#!/usr/bin/env python3
"""GLM-5.2 vs DeepSeek consistency flag agreement (Cohen's kappa) per family. Writes <dataset>/judge_agreement.csv."""
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from common import OUT

m = pd.read_csv(OUT / "myth_features.csv")
m = m[m.valid].dropna(subset=["cons_judge", "cons_judge_ds"])
rows = []
for fam, g in list(m.groupby("family")) + [("all", m)]:
    rows.append({"family": fam, "n": len(g), "kappa": cohen_kappa_score(g.cons_judge, g.cons_judge_ds),
                 "glm_true": g.cons_judge.mean(), "deepseek_true": g.cons_judge_ds.mean(),
                 "raw_agree": (g.cons_judge == g.cons_judge_ds).mean()})
t = pd.DataFrame(rows).round(3); t.to_csv(OUT / "judge_agreement.csv", index=False); print(t.to_string(index=False))
