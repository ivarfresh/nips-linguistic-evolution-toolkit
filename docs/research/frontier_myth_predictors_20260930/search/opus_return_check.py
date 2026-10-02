#!/usr/bin/env python3
"""Robustness for the one frontier upper-bound gain (Opus return share, raw ridge +0.039): is it myth
content, or myth features standing in for the run's cell (setting x composition x task order), which
the September baseline only has as main effects? Refit with the full cell as a baseline factor, and
with the own-myth return rule alone. Writes frontier/opus_return_check.csv."""
import numpy as np
import pandas as pd

import search as S
from common import OUT

d = pd.read_csv(OUT / "decision_table.csv")
d = d[(d.split == "discovery") & (d.own_i >= 0) & (d.shown_i >= 0) & (d.role == "trustee")].dropna(subset=["coop"])
d["cell"] = d["size"].astype(str) + "|" + d["composition"] + "|" + d["task_order"]
rows = []
for fam in ["Opus", "Sol", "GeminiPro", None]:
    sub = d if fam is None else d[d.family == fam]
    cols, embs = S.feature_sets(sub)["ALL myth features"]
    rules = [c for c in cols if c.startswith("own_rule_return")]
    for base in ("September baseline", "+ cell (size x composition x task order)"):
        S.BASE_CAT = ["family", "partner_family", "size", "mixed", "task_order"] + (["cell"] if base != "September baseline" else [])
        for within in (False, True):
            F = S.Fitter(sub, "coop", within)
            p0, y, _ = F.oof([], [], "ridge")
            for name, (c, e) in {"ALL myth features": (cols, embs), "own return rule only": (rules, [])}.items():
                p1, _, _ = F.oof(c, e, "ridge")
                b = S.boot_gain(y, p0, p1, F.groups, n=500)
                rows.append({"stratum": fam or "pooled", "baseline": base, "version": "within-agent" if within else "raw",
                             "feature_set": name, "n": len(y), "n_runs": sub.run_id.nunique(), **b})
                print(rows[-1]["stratum"], base, rows[-1]["version"], name, round(b["gain"], 4), round(b["gain_lo"], 4),
                      round(b["gain_hi"], 4), flush=True)
pd.DataFrame(rows).to_csv(OUT / "opus_return_check.csv", index=False)

# Split by task order (added after review): does the Opus rule gain hold inside one task order?
rows2 = []
S.BASE_CAT = ["family", "partner_family", "size", "mixed", "task_order"]
for to in ["game_myth", "myth_game"]:
    sub = d[(d.family == "Opus") & (d.task_order == to)]
    rules = [c for c in S.feature_sets(sub)["ALL myth features"][0] if c.startswith("own_rule_return")]
    F = S.Fitter(sub, "coop", False)
    p0, y, _ = F.oof([], [], "ridge")
    p1, _, _ = F.oof(rules, [], "ridge")
    b = S.boot_gain(y, p0, p1, F.groups, n=500)
    rows2.append({"stratum": f"Opus {to}", "baseline": "September baseline", "version": "raw",
                  "feature_set": "own return rule only", "n": len(y), "n_runs": sub.run_id.nunique(), **b})
    print(rows2[-1]["stratum"], round(b["gain"], 4), round(b["gain_lo"], 4), round(b["gain_hi"], 4), flush=True)
pd.concat([pd.DataFrame(rows), pd.DataFrame(rows2)]).to_csv(OUT / "opus_return_check.csv", index=False)
