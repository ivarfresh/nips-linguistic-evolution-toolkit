#!/usr/bin/env python3
"""Continuous self-copying test (new; the keyword ratchet is not estimable where the keyword is ~absent).

For each valid myth at round r >= 2: consistency ~ own previous myth + shown previous myth (the one the author
read before writing) + unseen previous myths (mean over same-family myths from round r-1 that the author never
saw: 8-agent = other agents of the run, not self or the shown author; dyads = same cell, other runs, same
family), round FE + cell FE (composition x task order), SE clustered by run.
If own > unseen, the theme is carried by copying the author's own last myth, not by a population-wide
round trend. Per family x setting and per family over all settings. Writes <dataset>/ratchet_continuous.csv.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from common import FAMILIES, OUT
from fe import fe_ols, rows_from

m = pd.read_csv(OUT / "myth_features.csv")
m["cons_emb_z"] = m["cons_emb"] / m.loc[m.valid, "cons_emb"].std()
MEAS = ["cons_emb_z", "cons_judge", "cons_lex"]
m.loc[~m.valid, MEAS] = np.nan
m["cell"] = m["composition"] + "|" + m["task_order"]
idx = {(r, t, a): i for i, (r, t, a) in enumerate(zip(m.run_id, m["round"], m.agent))}
v = m[m.valid]
by_run_round = {k: g.index.to_list() for k, g in v.groupby(["run_id", "round", "family"])}
by_cell_round = {k: g.index.to_list() for k, g in v.groupby(["cell", "round", "family"])}
rows = []
for i, r in m[(m["round"] >= 2) & m.valid].iterrows():
    own = idx.get((r.run_id, r["round"] - 1, r.agent))
    shown = idx.get((r.run_id, r.exposed_round, r.exposed_author)) if isinstance(r.exposed_author, str) else None
    if own is None:
        continue
    if r["size"] == 8:
        c = [j for j in by_run_round.get((r.run_id, r["round"] - 1, r.family), [])
             if m.at[j, "agent"] not in (r.agent, r.exposed_author)]
    else:
        c = [j for j in by_cell_round.get((r.cell, r["round"] - 1, r.family), []) if m.at[j, "run_id"] != r.run_id]
    row = {"run_id": r.run_id, "cell": r.cell, "setting": r.setting, "family": r.family, "round": r["round"]}
    for k in MEAS:
        row[k] = r[k]
        row[f"own_prev_{k}"] = m.at[own, k]
        row[f"shown_prev_{k}"] = m.at[shown, k] if shown is not None else np.nan
        row[f"unseen_prev_{k}"] = m.loc[c, k].mean() if c else np.nan
    rows.append(row)
ch = pd.DataFrame(rows)
res = []
groups = [(st, fam, g) for (st, fam), g in ch.groupby(["setting", "family"])] + [("all settings", fam, g) for fam, g in ch.groupby("family")]
for st, fam, g in groups:
    for k in MEAS:
        if g[k].std() == 0 or g[k].mean() < 0.02 and k != "cons_emb_z":
            continue  # keyword/judge (almost) never used in this stratum
        xs = [f"own_prev_{k}", f"shown_prev_{k}", f"unseen_prev_{k}"]
        try:
            out = fe_ols(g, k, xs, absorb="cell", dummies=["round"])
        except Exception:
            continue
        rr = rows_from(out, xs, stratum=st, family=fam, measure=k, y_mean=g[k].mean())
        # own - unseen contrast
        s = g.dropna(subset=xs + [k]).copy()
        s["sum_ou"] = (s[xs[0]] + s[xs[2]]) / 2
        s["half"] = (s[xs[0]] - s[xs[2]]) / 2
        o2 = fe_ols(s, k, ["half", "sum_ou", xs[1]], absorb="cell", dummies=["round"])
        rr += [{**x, "term": "own minus unseen"} for x in rows_from(o2, ["half"], stratum=st, family=fam, measure=k, y_mean=g[k].mean())]
        res += rr
t = pd.DataFrame(res)
t["p_holm_stratum"] = np.nan
for _, ix in t.groupby(["stratum", "family"]).groups.items():
    t.loc[ix, "p_holm_stratum"] = multipletests(t.loc[ix, "p"], method="holm")[1]
t.round(4).to_csv(OUT / "ratchet_continuous.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(t[t.stratum == "all settings"].round(3).to_string(index=False))
