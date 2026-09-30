"""Checks on the two surviving signals: is 8-agent give_align a level/functional-form artefact,
and does the dyad giving-gap effect just track the send?"""
import numpy as np, pandas as pd
import norm_alignment_fast as F
g = pd.read_csv("norm_alignment_games_v2.csv")
g = g[g["has_myths_before"]]
rows = []
for size in [2, 8]:
    d = g[g["size"] == size].copy()
    d["inv_give_c"] = d["inv_give"].round().astype("Int64").astype(str)
    d["tru_give_c"] = d["tru_give"].round().astype("Int64").astype(str)
    d["inv_above"] = (d["inv_give"] - d["tru_give"]).clip(lower=0)
    d["tru_above"] = (d["tru_give"] - d["inv_give"]).clip(lower=0)
    for m in ["give_align", "rule_index", "same_label"]:
        for y in F.OUTCOMES:
            e = d.dropna(subset=[m]).copy(); e["zx"] = F.z(e[m])
            specs = {"primary (linear own send scores)": (["zx"] + F.LAGS + F.NUMS, F.CATS),
                     "own send scores as 0-10 categories": (["zx"] + F.LAGS, F.CATS + ["inv_give_c", "tru_give_c"])}
            if m == "give_align":
                specs["split: investor above / trustee above (raw points)"] = (["inv_above", "tru_above"] + F.LAGS + F.NUMS, F.CATS)
            if size == 2:
                specs["gap model + own sent_frac"] = None
            for spec, xc in specs.items():
                if xc is None:
                    if y != "giving_gap":
                        continue
                    xc = (["zx", "sent_frac"] + F.LAGS + F.NUMS, F.CATS)
                f = F.feols(e, y, xc[0], xc[1], F.fe_for(size, "cell"))
                for t in [c for c in xc[0][:2] if c in f["terms"]] if spec.startswith("split") else ["zx"]:
                    rows.append({"size": size, "measure": m, "outcome": y, "spec": spec, "term": t,
                                 **f["terms"][t], "n_games": f["n_games"]})
    rows.append({"size": size, "measure": "corr(giving_gap, sent_frac)", "coef": d[["giving_gap", "sent_frac"]].corr().iloc[0, 1]})
out = pd.DataFrame(rows)
out.to_csv("artefact_checks.csv", index=False)
pd.set_option("display.width", 250)
print(out.round(4).to_string(index=False))
