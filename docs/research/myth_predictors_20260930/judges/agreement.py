#!/usr/bin/env python3
"""Judge agreement: GLM vs DeepSeek on labels, rule fields and the new 0-10 scores; and my blind codes vs each.

Outputs: agreement_labels.csv, agreement_rules.csv, agreement_giving.csv, blind_agreement.csv,
blind_confusion.csv, disagreement_samples.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
WT = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/moral-spread-viz")
D = WT / "data/analysis/linguistic_20260923"
KEY = ["run_id", "round", "agent"]
LAB = ["be generous", "be fair", "be cautious"]


def kappa(a, b, weights=None):
    ok = a.notna() & b.notna()
    return cohen_kappa_score(a[ok], b[ok], weights=weights), int(ok.sum())


def main():
    m = pd.read_csv(HERE / "measures.csv")
    rows = []
    for fam, g in [("all", m), *m.groupby("family")]:
        k, n = kappa(g.label_glm, g.label_ds)
        agree = (g.label_glm == g.label_ds)[g.label_glm.notna() & g.label_ds.notna()].mean()
        rows.append({"family": fam, "n": n, "kappa": k, "pct_agree": agree,
                     "share_generous_glm": (g.label_glm == "be generous").mean(),
                     "share_generous_ds": (g.label_ds == "be generous").mean()})
    for st, g in m.groupby(["size", "mixed"]):
        k, n = kappa(g.label_glm, g.label_ds)
        rows.append({"family": f"setting {st[0]}-agent {'mixed' if st[1] else 'homog'}", "n": n, "kappa": k,
                     "pct_agree": (g.label_glm == g.label_ds).mean()})
    pd.DataFrame(rows).round(3).to_csv(HERE / "agreement_labels.csv", index=False)

    # rules: GLM full vs DeepSeek 1000 sample
    g = pd.read_csv(D / "myth_rules_september_z-ai__glm-5.2.csv")
    d = pd.read_csv(D / "myth_rules_september_deepseek__deepseek-v4-flash_sample1000.csv")
    x = g.merge(d, on=KEY, suffixes=("_g", "_d"))
    x = x[(x.status_g == "ok") & (x.status_d == "ok")]
    rr = []
    ordmap = {"none": 0, "little": 1, "moderate": 2, "most": 3, "all": 4}
    for f in ["send_rule", "return_rule", "after_letdown"]:
        k, n = kappa(x[f + "_g"], x[f + "_d"])
        rr.append({"field": f, "n": n, "pct_agree": (x[f + "_g"] == x[f + "_d"]).mean(), "kappa": k})
    so = x[["send_rule_g", "send_rule_d"]].apply(lambda s: s.map(ordmap))
    k, n = kappa(so.send_rule_g, so.send_rule_d, weights="quadratic")
    rr.append({"field": "send_rule ordinal (quadratic-weighted, unspecified dropped)", "n": n, "kappa": k,
               "spearman": stats.spearmanr(so.dropna().send_rule_g, so.dropna().send_rule_d)[0]})
    both = x.send_amount_g.notna() & x.send_amount_d.notna()
    rr.append({"field": "send_amount (both named)", "n": int(both.sum()),
               "pearson": x.loc[both, "send_amount_g"].corr(x.loc[both, "send_amount_d"]),
               "pct_agree": (x.loc[both, "send_amount_g"] == x.loc[both, "send_amount_d"]).mean(),
               "named_by_glm_only": int((x.send_amount_g.notna() & x.send_amount_d.isna()).sum()),
               "named_by_ds_only": int((x.send_amount_g.isna() & x.send_amount_d.notna()).sum())})
    for f in ["test_first", "consistency", "noise_mentioned"]:
        k, n = kappa(x[f + "_g"].astype(str), x[f + "_d"].astype(str))
        rr.append({"field": f, "n": n, "pct_agree": (x[f + "_g"] == x[f + "_d"]).mean(), "kappa": k,
                   "rate_glm": x[f + "_g"].mean(), "rate_ds": x[f + "_d"].mean()})
    pd.DataFrame(rr).round(3).to_csv(HERE / "agreement_rules.csv", index=False)

    # giving scores
    if "give_send_glm" in m:
        gr = []
        for f in ["send", "return", "cond"]:
            a, b = m[f"give_{f}_glm"], m[f"give_{f}_ds"]
            ok = a.notna() & b.notna()
            for fam, sub in [("all", m[ok]), *m[ok].groupby("family")]:
                gr.append({"field": f, "family": fam, "n": len(sub),
                           "pearson": sub[f"give_{f}_glm"].corr(sub[f"give_{f}_ds"]),
                           "spearman": stats.spearmanr(sub[f"give_{f}_glm"], sub[f"give_{f}_ds"])[0],
                           "mean_glm": sub[f"give_{f}_glm"].mean(), "sd_glm": sub[f"give_{f}_glm"].std(),
                           "mean_ds": sub[f"give_{f}_ds"].mean(), "sd_ds": sub[f"give_{f}_ds"].std()})
        pd.DataFrame(gr).round(3).to_csv(HERE / "agreement_giving.csv", index=False)

    # blind codes
    codes = pd.read_csv(HERE / "blind/my_codes.csv")
    k1 = pd.read_csv(D / "human_coding_key.csv")[["item_id"] + KEY]
    k2 = pd.read_csv(HERE / "blind/new_items_key.csv")[["item_id"] + KEY]
    kk = pd.concat([k1, k2])
    b = codes.merge(kk, on="item_id").merge(m, on=KEY, how="left")
    b["sample"] = np.where(b.item_id.str.startswith("M"), "human sheet (stratified by GLM label)", "new (stratified by family x setting)")
    b["my_send_ord"] = b.my_send_rule.map(ordmap)
    b.to_csv(HERE / "blind_joined.csv", index=False)
    br = []
    for samp, sub in [("all", b), *b.groupby("sample")]:
        rec = {"sample": samp, "n": len(sub)}
        for j in ("glm", "ds"):
            k, n = kappa(sub.my_label, sub[f"label_{j}"])
            rec[f"label_kappa_me_{j}"], rec[f"label_agree_me_{j}"] = k, (sub.my_label == sub[f"label_{j}"]).mean()
        rec["label_kappa_glm_ds"] = kappa(sub.label_glm, sub.label_ds)[0]
        rec["send_rule_agree_me_glm"] = (sub.my_send_rule == sub.send_rule).mean()
        rec["send_rule_wkappa_me_glm"] = kappa(sub.my_send_ord, sub.rule_send_ord, "quadratic")[0]
        if "give_send_glm" in sub:
            for j in ("glm", "ds"):
                ok = sub[f"give_send_{j}"].notna()
                rec[f"give_send_r_me_{j}"] = sub.loc[ok, "my_give"].corr(sub.loc[ok, f"give_send_{j}"])
            rec["give_send_r_glm_ds"] = sub.give_send_glm.corr(sub.give_send_ds)
        rec["my_give_r_label_ord_glm"] = sub.my_give.corr(sub.label_ord_glm)
        rec["my_give_r_rule_prescribed"] = sub.my_give.corr(sub.rule_prescribed)
        rec["my_give_r_emb_axis_summary"] = sub.my_give.corr(sub.emb_axis_summary)
        rec["my_give_r_emb_axis_text"] = sub.my_give.corr(sub.emb_axis_text)
        br.append(rec)
    pd.DataFrame(br).round(3).to_csv(HERE / "blind_agreement.csv", index=False)
    conf = []
    for j in ("glm", "ds"):
        c = pd.crosstab(b.my_label, b[f"label_{j}"]).reindex(index=LAB, columns=LAB, fill_value=0)
        c.index = [f"me: {i}" for i in c.index]
        c.insert(0, "judge", j)
        conf.append(c)
    pd.concat(conf).to_csv(HERE / "blind_confusion.csv")
    # per-label precision/recall of each judge against my codes (me as reference)
    pr = []
    for j in ("glm", "ds"):
        for lab in LAB:
            tp = ((b[f"label_{j}"] == lab) & (b.my_label == lab)).sum()
            pr.append({"judge": j, "label": lab, "precision_vs_me": tp / max((b[f"label_{j}"] == lab).sum(), 1),
                       "recall_vs_me": tp / max((b.my_label == lab).sum(), 1), "n_me": int((b.my_label == lab).sum()),
                       "n_judge": int((b[f"label_{j}"] == lab).sum())})
    pd.DataFrame(pr).round(3).to_csv(HERE / "blind_precision_recall.csv", index=False)

    # disagreement samples (GLM fair vs DS generous is the main mass)
    dis = m[(m.label_glm == "be fair") & (m.label_ds == "be generous")]
    dis.sample(n=min(20, len(dis)), random_state=3)[KEY + ["family", "label_glm", "label_ds", "send_rule"] +
                                                    [c for c in m.columns if c.startswith("give_send")]].to_csv(
        HERE / "disagreement_samples.csv", index=False)
    print(pd.read_csv(HERE / "agreement_labels.csv").to_string())
    print(pd.read_csv(HERE / "agreement_rules.csv").to_string())
    print(pd.read_csv(HERE / "blind_agreement.csv").T.to_string())
    print(pd.read_csv(HERE / "blind_confusion.csv").to_string())
    print(pd.read_csv(HERE / "blind_precision_recall.csv").to_string())
    if (HERE / "agreement_giving.csv").exists():
        print(pd.read_csv(HERE / "agreement_giving.csv").to_string())


if __name__ == "__main__":
    main()
