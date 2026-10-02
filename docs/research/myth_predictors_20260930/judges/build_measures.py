#!/usr/bin/env python3
"""One row per September myth with every candidate measure of 'how much the myth recommends giving'.

label_ord_{glm,ds}   3-way label as ordinal: cautious 0, fair 1, generous 2
gen_{glm,ds}         label == 'be generous'
rule_send_ord        GLM rule extraction send_rule: none 0 .. all 4 (unspecified missing)
rule_prescribed      named amount if the amount check says endorsed, else send_rule band midpoint
                     (= prescribed_endorsed_or_band in myth_rules_analysis.py)
give_send_{glm,ds}, give_return_{glm,ds}, give_cond_{glm,ds}   new 0-10 judge pass
emb_axis_summary, emb_axis_text   projection of the one-sentence moral / the full myth on a
                     generous-minus-cautious axis built from fixed anchor sentences (not label centroids)
"""
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
WT = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/moral-spread-viz")
D = WT / "data/analysis/linguistic_20260923"
KEY = ["run_id", "round", "agent"]
ORD = {"be cautious": 0, "be fair": 1, "be generous": 2}
SEND_ORD = {"none": 0, "little": 1, "moderate": 2, "most": 3, "all": 4}
MIDPOINT = {"all": 5.0, "most": 4.25, "moderate": 2.75, "little": 1.25, "none": 0.0}
GEN_ANCHORS = [
    "Give everything you have freely, without holding anything back.",
    "Send all five and trust your partner completely.",
    "Generosity without reserve creates abundance for everyone.",
    "Give fully even to a stranger, before they have proven themselves.",
    "Keep giving generously even when your partner returns little.",
]
CAU_ANCHORS = [
    "Keep most of what you have and give only a small amount.",
    "Send only what you can afford to lose and protect yourself.",
    "Trust must be earned; test your partner with a little before giving more.",
    "Hold back your resources until the other proves reliable.",
    "After betrayal, stop giving and guard what remains.",
]


def main():
    m = pd.read_csv(D / "myths.csv")
    m = m.reset_index(drop=True)
    m["row"] = np.arange(len(m))
    glm = pd.read_csv(D / "moral_labels_z-ai__glm-5.2.csv")[KEY + ["label", "summary"]].rename(columns={"label": "label_glm"})
    ds = pd.read_csv(D / "moral_labels_deepseek__deepseek-v4-flash.csv")[KEY + ["label"]].rename(columns={"label": "label_ds"})
    m = m.merge(glm, on=KEY, how="left").merge(ds, on=KEY, how="left")
    for j in ("glm", "ds"):
        m[f"label_ord_{j}"] = m[f"label_{j}"].map(ORD)
        m[f"gen_{j}"] = (m[f"label_{j}"] == "be generous").astype(float).where(m[f"label_{j}"].notna())
    r = pd.read_csv(D / "myth_rules_september_z-ai__glm-5.2.csv")
    r = r[r.status == "ok"]
    chk = pd.read_csv(D / "myth_amount_check_september_z-ai__glm-5.2.csv")[KEY + ["amount_status"]]
    r = r.merge(chk, on=KEY, how="left")
    band = r.send_rule.map(MIDPOINT)
    r["rule_prescribed"] = r.send_amount.where(r.send_amount.notna() & (r.amount_status == "endorsed"), band)
    r["rule_send_ord"] = r.send_rule.map(SEND_ORD)
    r["rule_consistency"] = r.consistency.astype(float)
    m = m.merge(r[KEY + ["send_rule", "rule_send_ord", "rule_prescribed", "rule_consistency"]], on=KEY, how="left")
    for j, tag in (("glm", "z-ai__glm-5.2"), ("ds", "deepseek__deepseek-v4-flash")):
        p = HERE / f"giving_scores_{tag}.csv"
        if p.exists():
            g = pd.read_csv(p)[KEY + ["g_send", "g_return", "g_conditional"]]
            g.columns = KEY + [f"give_send_{j}", f"give_return_{j}", f"give_cond_{j}"]
            m = m.merge(g, on=KEY, how="left")
    assert (m["row"].to_numpy() == np.arange(len(m))).all()
    # embedding axis from fixed anchors
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("all-mpnet-base-v2")
    ga = st.encode(GEN_ANCHORS, normalize_embeddings=True).mean(0)
    ca = st.encode(CAU_ANCHORS, normalize_embeddings=True).mean(0)
    axis = (ga - ca) / np.linalg.norm(ga - ca)
    es = np.load(D / "embeddings_moral_summary_mpnet.npy")
    et = np.load(D / "embeddings_mpnet.npy")
    m["emb_axis_summary"] = np.where(m.summary.notna(), es @ axis, np.nan)
    m["emb_axis_text"] = np.where(m.n_words >= 20, et @ axis, np.nan)
    keep = KEY + ["size", "mixed", "composition", "task_order", "family", "partner_this_round", "exposed_author",
                  "exposed_family", "exposed_round", "n_words", "label_glm", "label_ds", "label_ord_glm", "label_ord_ds",
                  "gen_glm", "gen_ds", "send_rule", "rule_send_ord", "rule_prescribed", "rule_consistency",
                  "emb_axis_summary", "emb_axis_text"] + [c for c in m.columns if c.startswith("give_")]
    m[keep].to_csv(HERE / "measures.csv", index=False)
    print(m[keep].describe().T[["count", "mean", "std"]].round(3).to_string())


if __name__ == "__main__":
    main()
