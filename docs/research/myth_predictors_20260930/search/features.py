#!/usr/bin/env python3
"""Per-myth feature bank (8,520 myths). Reads the shared linguistic tables read-only.

Writes myth_features.csv (one row per myth, same order as myths.csv) to this folder.
Sentiment needs vaderSentiment:  uv run --with vaderSentiment --with pandas --with numpy python features.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/linguistic-mixed/data/analysis/linguistic_20260923")
OUT = Path(__file__).resolve().parent
KEY = ["run_id", "round", "agent"]

# Lexicons: counted as matches per 100 words. Multiplier words are kept apart from amounts,
# because "three / triple / fifteen" come from the 3x rule, not from a claim about what to send.
LEX = {
    "reciprocity": r"\b(recipro\w*|return\w*|repa(y|id)\w*|in kind|give back|gave back|mutual\w*|exchange\w*|both prosper\w*)\b",
    "trust": r"\b(trust\w*|faith\w*|believ\w*|confiden\w*|rely|relied|reliance)\b",
    "betrayal": r"\b(betray\w*|cheat\w*|deceiv\w*|decept\w*|greed\w*|hoard\w*|selfish\w*|stole|steal\w*|exploit\w*|broken promise\w*)\b",
    "punishment": r"\b(punish\w*|curse\w*|wrath|revenge|vengean\w*|retaliat\w*|withh[eo]ld\w*|withdr[ae]w\w*|penalt\w*)\b",
    "forgiveness": r"\b(forgiv\w*|forgave|mercy|merciful|second chance\w*|pardon\w*|grace)\b",
    "abundance": r"\b(abundan\w*|plent\w*|bount\w*|prosper\w*|flourish\w*|harvest\w*|rich\w*|overflow\w*|wealth\w*)\b",
    "scarcity": r"\b(scarc\w*|famine|drought|hunger\w*|starv\w*|barren|wither\w*|poor|poverty|dwindl\w*|empty)\b",
    "uncertainty": r"\b(noise|nois\w*|uncertain\w*|mist\w*|fog\w*|distort\w*|obscur\w*|unseen|hidden|veil\w*|shadow\w*|murk\w*|unclear|unknown)\b",
    "community": r"\b(communit\w*|village\w*|together|kin|people|tribe\w*|neighbou?r\w*|all of us|collective\w*|share[ds]?|sharing)\b",
    "future": r"\b(future|generation\w*|forever|eternal\w*|always|seasons?|years?|long after|endur\w*|lasting|patien\w*|over time)\b",
    "generosity": r"\b(generos\w*|generous\w*|gift\w*|give|gave|giving|offer\w*|open.handed)\b",
    "caution": r"\b(cauti\w*|careful\w*|wary|prudent\w*|guard\w*|test\w*|small first|little at first|measured)\b",
    "fairness": r"\b(fair\w*|equal\w*|half|halves|balance\w*|just|justice|even share\w*|proportion\w*)\b",
    "consistency": r"\b(consisten\w*|steady|steadfast\w*|reliab\w*|predictab\w*|constant\w*|pattern\w*)\b",
    "multiplier": r"\b(three|thrice|threefold|triple\w*|tripling|fifteen|3x|x3|15)\b",
    "amount_numbers": r"(\$\s?[0-5](\.\d+)?\b|\b[0-5](\.\d+)? (coins?|sacks?|stones?|seeds?|measures?|dollars?|gold|pieces?)\b|\b(one|two|four|five) (coins?|sacks?|stones?|seeds?|measures?|dollars?|pieces?|of (his|her|their) five)\b)",
    "give_all": r"\b(gave all|give all|giving all|everything|all (she|he|they) had|all of (his|her|their)|whole|entire\w*|every coin|every last)\b",
    "give_little": r"\b(a little|only a little|small portion|a few|pebbles?|a single|one coin|only one|nothing)\b",
}
SEND_RULE = {"none": 0, "little": 1, "moderate": 2, "most": 3, "all": 4}  # 'unspecified' -> NaN + flag


def main() -> None:
    myths = pd.read_csv(D / "myths.csv")
    assert len(myths) == 8520
    text = myths["text"].fillna("").astype(str)
    f = myths[KEY + ["family"]].copy()
    nw = myths["n_words"].clip(lower=1)
    f["n_words"] = myths["n_words"]
    low = text.str.lower()
    for k, pat in LEX.items():
        f[f"lex_{k}"] = low.str.count(pat) / nw * 100
    f["heading"] = text.str.lstrip().str.startswith("#").astype(float)
    f["n_sentences"] = text.str.count(r"[.!?](\s|$)")

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        s = [sia.polarity_scores(t) for t in text]
        f["sent_compound"] = [x["compound"] for x in s]
        f["sent_pos"] = [x["pos"] for x in s]
        f["sent_neg"] = [x["neg"] for x in s]
    except ImportError:
        raise SystemExit("run under uv with vaderSentiment")

    r = pd.read_csv(D / "myth_rules_september_z-ai__glm-5.2.csv")
    r = r[r["status"] == "ok"]
    rules = r[KEY].copy()
    rules["rule_send_ord"] = r["send_rule"].map(SEND_RULE)
    rules["rule_send_unspecified"] = (r["send_rule"] == "unspecified").astype(float)
    rules["rule_send_amount"] = r["send_amount"]
    rules["rule_send_amount_given"] = r["send_amount"].notna().astype(float)
    for v in ["half", "more_than_half", "at_least_sent", "match_partner", "little", "unspecified"]:
        rules[f"rule_return_{v}"] = (r["return_rule"] == v).astype(float)
    for v in ["keep_trusting", "reduce", "withdraw", "unspecified"]:
        rules[f"rule_letdown_{v}"] = (r["after_letdown"] == v).astype(float)
    for c in ["test_first", "consistency", "noise_mentioned"]:
        rules[f"rule_{c}"] = r[c].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    f = f.merge(rules, on=KEY, how="left")

    for tag, fn in [("glm", "moral_labels_z-ai__glm-5.2.csv"), ("ds", "moral_labels_deepseek__deepseek-v4-flash.csv")]:
        lab = pd.read_csv(D / fn)[KEY + ["label"]]
        f = f.merge(lab, on=KEY, how="left")
        for v in ["be generous", "be cautious"]:
            f[f"label_{tag}_{v.split()[1]}"] = np.where(f["label"].isna(), np.nan, (f["label"] == v).astype(float))
        f = f.drop(columns="label")

    st = pd.read_csv(D / "style_probabilities.csv")[KEY + ["p_Sonnet", "p_Gemini", "p_GPT"]]
    st.columns = KEY + ["style_p_Sonnet", "style_p_Gemini", "style_p_GPT"]
    f = f.merge(st, on=KEY, how="left")
    assert len(f) == 8520 and (f[KEY].values == myths[KEY].values).all()

    # Change vs the same agent's previous myth.
    emb = np.load(D / "embeddings_mpnet.npy")
    assert emb.shape[0] == 8520
    f["_i"] = np.arange(len(f))
    change_cols = ["rule_send_ord", "rule_send_amount", "label_glm_generous", "lex_generosity", "lex_betrayal",
                   "lex_caution", "sent_compound", "n_words"]
    for c in change_cols:
        f[f"chg_{c}"] = np.nan
    f["chg_cos_prev"] = np.nan
    for _, g in f.sort_values("round").groupby(["run_id", "agent"]):
        idx = g.index.to_numpy()
        for a, b in zip(idx[:-1], idx[1:]):
            for c in change_cols:
                f.at[b, f"chg_{c}"] = f.at[b, c] - f.at[a, c]
            f.at[b, "chg_cos_prev"] = float(emb[a] @ emb[b])
    f = f.drop(columns="_i")
    f.to_csv(OUT / "myth_features.csv", index=False)
    print(f.shape)
    print(f.describe().T[["mean", "std"]].round(3).to_string())


if __name__ == "__main__":
    main()
