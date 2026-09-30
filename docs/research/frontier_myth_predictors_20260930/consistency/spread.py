#!/usr/bin/env python3
"""Part 3 (port; LINGUISTIC_DATASET=september|frontier): does consistency language spread from the myth an agent is shown?

For each child myth (round >= 2) we compare three 'parents':
  shown   the myth the child's author was actually shown (exposed_author, exposed_round)
  unseen  mean over comparable unseen same-family myths from the same round
          (linguistic_uptake.null_candidates: 8-agent = other agents in the run; dyads = same cell, other runs)
  future  the shown author's NEXT myth (written at the child's round, never visible to the child) --
          the Aug-28 control that 'consistency' failed on the homogeneous corpus.
Outcome: child mentions consistency (keyword), restricted to NEW use (child's own earlier myths never
used the keyword) and, separately, any use with the child's own previous myth as a control; also judge
and embedding versions. Linear probability, round FE, run-clustered. Writes spread_*.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

from fe import fe_ols, rows_from

from common import FAMILIES, NAME, OUT, WT  # noqa: E402  (sets nothing global; LINGUISTIC_DATASET picks the corpus)

sys.path.insert(0, str(WT))
os.environ.setdefault("LINGUISTIC_DATASET", NAME)
from analyses.linguistic_uptake import DATA as UPTAKE_DATA, load_myths, null_candidates  # noqa: E402

assert UPTAKE_DATA.name in ("linguistic_20260923", "linguistic_frontier_20260930") and (NAME == "frontier") == ("frontier" in UPTAKE_DATA.name)


def main():
    myths = load_myths()
    f = pd.read_csv(OUT / "myth_features.csv")
    assert (f["run_id"].to_numpy() == myths["run_id"].to_numpy()).all() and (f["round"].to_numpy() == myths["round"].to_numpy()).all()
    for c in ["cons_lex", "cons_judge", "cons_emb"]:
        myths[c] = f[c].to_numpy()
    myths["cons_emb_z"] = myths["cons_emb"] / myths["cons_emb"].std()
    cands = null_candidates(myths)
    idx = {(r, t, a): i for i, (r, t, a) in enumerate(zip(myths.run_id, myths["round"], myths.agent))}
    # own history of keyword use
    myths = myths.sort_values(["run_id", "agent", "round"])
    myths["ever_before"] = myths.groupby(["run_id", "agent"])["cons_lex"].transform(lambda s: s.shift(1).cummax()).fillna(0)
    for c in ["cons_lex", "cons_judge", "cons_emb_z"]:
        myths[f"own_prev_{c}"] = myths.groupby(["run_id", "agent"])[c].shift(1)
    myths = myths.sort_index()
    rows = []
    for i, js in cands.items():
        p, nulls = js[0], js[1:]
        c = myths.loc[i]
        fut = idx.get((c.run_id, c["round"], c.exposed_author))
        r = {"run_id": c.run_id, "size": c["size"], "mixed": c.mixed, "family": c.family, "round": c["round"],
             "task_order": c.task_order, "parent_family": myths.at[p, "family"], "ever_before": c.ever_before}
        for k in ["cons_lex", "cons_judge", "cons_emb_z"]:
            r[f"child_{k}"] = c[k]
            r[f"own_prev_{k}"] = c[f"own_prev_{k}"]
            r[f"shown_{k}"] = myths.at[p, k]
            r[f"unseen_{k}"] = myths.loc[nulls, k].mean()
            r[f"future_{k}"] = myths.at[fut, k] if fut is not None and myths.at[fut, "valid"] else np.nan
        rows.append(r)
    ch = pd.DataFrame(rows)
    ch["stratum"] = np.where(~ch["mixed"], ch["size"].astype(str) + "-agent homogeneous",
                             ch["size"].astype(str) + "-agent mixed, " +
                             np.where(ch["family"] == ch["parent_family"], "same family", "other family"))
    ch.to_csv(OUT / "spread_children.csv", index=False)
    res_rows, rate_rows = [], []
    strata = list(ch.groupby("stratum"))
    for pf in FAMILIES:  # September reported only the Sonnet parent row; frontier gets one per parent family
        strata.append((f"cross-family, {pf} parent -> other-family child", ch[(ch.parent_family == pf) & (ch.family != pf)]))
    strata += [(f"{st} | child {fam}", g) for (st, fam), g in ch.groupby(["stratum", "family"])]
    for name, g in strata:
        for k in ["cons_lex", "cons_judge", "cons_emb_z"]:
            sets = [("any use, own previous as control", g, [f"shown_{k}", f"unseen_{k}", f"future_{k}", f"own_prev_{k}"])]
            if k == "cons_lex":
                sets.append(("new use (never before)", g[g["ever_before"] == 0], [f"shown_{k}", f"unseen_{k}", f"future_{k}"]))
            for lab, s, xs in sets:
                s = s.dropna(subset=xs + [f"child_{k}"]).copy()
                if s["run_id"].nunique() < 5:
                    continue
                s["shown_minus_future"] = s[f"shown_{k}"] - s[f"future_{k}"]
                res = fe_ols(s, f"child_{k}", xs, absorb=None, dummies=["round", "family"] if ("mixed" in name or "cross" in name) and "| child" not in name else ["round"])
                res_rows += rows_from(res, xs[:3], stratum=name, measure=k, sample=lab)
                # headline spec: same model without the future myth
                xs0 = [x for x in xs if x != f"future_{k}"]
                r0 = fe_ols(s, f"child_{k}", xs0, absorb=None, dummies=["round", "family"] if ("mixed" in name or "cross" in name) and "| child" not in name else ["round"])
                res_rows += [{**x, "term": "shown (no future term)"} for x in rows_from(r0, [f"shown_{k}"], stratum=name, measure=k, sample=lab)]
                # shown - future contrast via reparametrisation: shown coefficient when future enters as (shown+future)
                s["sum_sf"] = s[f"shown_{k}"] + s[f"future_{k}"]
                xs2 = [f"shown_{k}", "sum_sf"] + [x for x in xs if x not in (f"shown_{k}", f"future_{k}")]
                s["half_diff"] = (s[f"shown_{k}"] - s[f"future_{k}"])
                r2 = fe_ols(s.assign(**{f"shown_{k}": s["half_diff"] / 2, "sum_sf": s["sum_sf"] / 2}), f"child_{k}", xs2, absorb=None,
                            dummies=["round", "family"] if ("mixed" in name or "cross" in name) and "| child" not in name else ["round"])
                res_rows += [{**x, "term": "shown minus future"} for x in rows_from(r2, [f"shown_{k}"], stratum=name, measure=k, sample=lab)]
                if k == "cons_lex" and lab.startswith("new"):
                    for par in ["shown", "future"]:
                        for v in [0, 1]:
                            q = s[s[f"{par}_{k}"] == v]
                            rate_rows.append({"stratum": name, "parent": par, "parent_uses_word": v,
                                              "child_new_use_rate": q[f"child_{k}"].mean(), "n": len(q)})
                    rate_rows.append({"stratum": name, "parent": "unseen (mean share)", "parent_uses_word": np.nan,
                                      "child_new_use_rate": s[f"child_{k}"].mean(), "n": len(s),
                                      "unseen_share": s[f"unseen_{k}"].mean(), "shown_share": s[f"shown_{k}"].mean()})
    out = pd.DataFrame(res_rows).round(4)
    out.to_csv(OUT / "spread_models.csv", index=False)
    pd.DataFrame(rate_rows).round(4).to_csv(OUT / "spread_new_use_rates.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    print(out[["stratum", "measure", "sample", "term", "coef", "ci_low", "ci_high", "p", "n", "n_runs"]].to_string(index=False))
    print(pd.DataFrame(rate_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
