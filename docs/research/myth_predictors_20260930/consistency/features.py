#!/usr/bin/env python3
"""Per-myth consistency features (read-only on the shared data dir).

Three measures per myth:
  cons_judge  GLM-5.2 myth_rules 'consistency' (explicitly praises steady/predictable giving)
  cons_lex    keyword hit, same regex as myth-rules analyses/myth_rules_analysis.py KEYWORDS
  cons_emb    judge-free embedding score: cosine to consistency anchor sentences minus cosine
              to matched cooperative anchors that say nothing about steadiness (all-mpnet-base-v2)
Writes myth_features.csv to this lens folder.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/moral-spread-viz")
DATA = WT / "data/analysis/linguistic_20260923"
OUT = Path(__file__).resolve().parent

LEX = r"\bconsisten(?:t|cy|tly)\b|\bsteady\b|\bsteadiness\b|\breliab"
LEX_STRICT = r"\bconsisten(?:t|cy|tly)\b"

CONS_ANCHORS = [
    "Be consistent: give the same amount every time, steadily and reliably.",
    "The wise giver was steady and predictable, never wavering from round to round.",
    "Trust grows from consistency; a reliable partner gives the same measure each season.",
    "Keep your giving steady and dependable, not wild or changing.",
    "Consistency over volatility: the steady hand earns trust.",
]
CTRL_ANCHORS = [
    "Be generous: give much, and trust will return to you.",
    "The wise giver shared freely and the receiver returned the gift fairly.",
    "Trust grows from generosity; a good partner gives and returns a fair share.",
    "Share your wealth with others and both will prosper.",
    "Generosity over greed: the open hand earns trust.",
]


def load_embeddings(texts):
    own = OUT / "embeddings_mpnet_local.npy"
    if own.exists():
        return np.load(own)
    emb = np.load(DATA / "embeddings_mpnet.npy")
    digest = hashlib.sha256(b"all-mpnet-base-v2")
    for t in texts:
        digest.update(b"\0" + t.encode())
    stamp = (DATA / "embeddings_mpnet.npy.sha256")
    ok = stamp.exists() and stamp.read_text().strip() == digest.hexdigest()
    print("embedding fingerprint matches myths.csv text:", ok)
    if not ok:
        # shared cache fingerprint does not match; re-embed into this lens folder only
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer("all-mpnet-base-v2").encode(list(texts), normalize_embeddings=True, batch_size=64)
        np.save(own, emb)
    return emb


def main():
    m = pd.read_csv(DATA / "myths.csv")
    m["text"] = m["text"].fillna("")
    emb = load_embeddings(m["text"].tolist())
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("all-mpnet-base-v2")
    a = st.encode(CONS_ANCHORS, normalize_embeddings=True).mean(0)
    c = st.encode(CTRL_ANCHORS, normalize_embeddings=True).mean(0)
    m["cons_emb"] = emb @ (a / np.linalg.norm(a)) - emb @ (c / np.linalg.norm(c))
    low = m["text"].str.lower()
    m["cons_lex"] = low.str.contains(LEX, regex=True).astype(float)
    m["cons_lex_strict"] = low.str.contains(LEX_STRICT, regex=True).astype(float)
    m["cons_lex_count"] = low.str.count(LEX) / m["n_words"].clip(lower=1) * 100
    r = pd.read_csv(DATA / "myth_rules_september_z-ai__glm-5.2.csv")
    r = r[["run_id", "round", "agent", "consistency", "send_rule", "send_amount"]]
    r["cons_judge"] = r["consistency"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    m = m.merge(r[["run_id", "round", "agent", "cons_judge", "send_rule"]], on=["run_id", "round", "agent"], how="left")
    ds = pd.read_csv(DATA / "myth_rules_september_deepseek__deepseek-v4-flash_sample1000.csv")
    ds["cons_judge_ds"] = ds["consistency"].astype(float)
    m = m.merge(ds[["run_id", "round", "agent", "cons_judge_ds"]], on=["run_id", "round", "agent"], how="left")
    m["setting"] = m["size"].astype(str) + "-agent " + np.where(m["mixed"], "mixed", "homogeneous")
    m["valid"] = m["n_words"] >= 20
    m.drop(columns=["text", "path"]).to_csv(OUT / "myth_features.csv", index=False)
    print(m[["cons_judge", "cons_judge_ds", "cons_lex", "cons_lex_strict", "cons_emb"]].corr().round(2))
    print(m.groupby("family")[["cons_judge", "cons_lex", "cons_emb"]].mean().round(3))


if __name__ == "__main__":
    main()
