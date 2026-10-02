#!/usr/bin/env python3
"""H1 norm alignment, games table (port of docs/research/myth_predictors_20260930/alignment/
norm_alignment.py, parameterised by dataset).

One row per game (investor decision) with both players' latest myths before it
(myth_game: same round; game_myth: previous round) and after it, and the norm
alignment measures between them:

  same_label   both myths carry the same GLM-5.2 moral label
  moral_cos    cosine of the one-sentence moral summaries (mpnet)
  rule_index   mean agreement over the rule fields both players specified
  give_align   1 - |send score A - send score B| / 10 (GLM 0-10 giving score)
  myth_cos     whole-myth cosine

The games-table logic is unchanged from September; only the input paths are
switched. Set ALIGN_DATASET=september|frontier (default frontier). The models are
in norm_alignment_fast.py.

Reads the shared tables read-only; writes the (large) games table to the gitignored
data/analysis/frontier_myth_predictors_20260930/alignment/<dataset>/.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path(__file__).resolve().parents[4]
SEPT_WT = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/myth-predictors")
DATASET = os.environ.get("ALIGN_DATASET", "frontier")
CFG = {
    "september": {
        "data": WT / "data/analysis/linguistic_20260923",
        "rules": "myth_rules_september_z-ai__glm-5.2.csv",
        "labels_ds": "moral_labels_deepseek__deepseek-v4-flash.csv",
        # September lens used the judges lens's measures table (give_send_glm, label_ds)
        "give": SEPT_WT / "data/analysis/myth_predictors_20260930/judges/measures.csv",
    },
    "frontier": {
        "data": WT / "data/analysis/linguistic_frontier_20260930",
        "rules": "myth_rules_frontier_z-ai__glm-5.2.csv",
        "labels_ds": "moral_labels_deepseek__deepseek-v4-flash.csv",
        "give": WT / "data/analysis/linguistic_frontier_20260930/giving_scores_z-ai__glm-5.2.csv",
    },
}[DATASET]
DATA = CFG["data"]
OUT = Path(__file__).resolve().parent / DATASET
BIG = WT / "data/analysis/frontier_myth_predictors_20260930/alignment" / DATASET
KEY = ["run_id", "round", "agent"]
MIN_WORDS = 20
SEND_ORD = {"none": 0, "little": 1, "moderate": 2, "most": 3, "all": 4}
LABEL_LEVEL = {"be cautious": 0, "be fair": 1, "be generous": 2}
OUTCOMES = ["sent_frac", "return_proportion", "giving_gap"]
MEASURES = ["same_label", "moral_cos", "rule_index", "myth_cos"]
FIELDS = ["f_send_rule", "f_return_rule", "f_after_letdown", "f_send_amount", "f_consistency", "f_test_first"]


def judge_measures() -> pd.DataFrame:
    """KEY + give_send_glm + label_ds, the same columns for both datasets."""
    if DATASET == "september":
        return pd.read_csv(CFG["give"])[KEY + ["give_send_glm", "label_ds"]]
    g = pd.read_csv(CFG["give"])[KEY + ["g_send"]].rename(columns={"g_send": "give_send_glm"})
    p = DATA / CFG["labels_ds"]
    if p.exists():
        ds = pd.read_csv(p)[KEY + ["label"]].rename(columns={"label": "label_ds"})
        ds["label_ds"] = ds["label_ds"].where(ds["label_ds"].isin(LABEL_LEVEL))
        g = g.merge(ds, on=KEY, how="left")
    else:
        g["label_ds"] = np.nan
    return g


def load_myths():
    myths = pd.read_csv(DATA / "myths.csv").reset_index(drop=True)
    emb = np.load(DATA / "embeddings_mpnet.npy")
    semb = np.load(DATA / "embeddings_moral_summary_mpnet.npy")
    assert len(emb) == len(semb) == len(myths)
    glm = pd.read_csv(DATA / "moral_labels_z-ai__glm-5.2.csv")[KEY + ["label", "summary"]]
    ds = judge_measures()[KEY + ["label_ds"]]
    rules = pd.read_csv(DATA / CFG["rules"])
    rules = rules[KEY + ["send_rule", "return_rule", "after_letdown", "send_amount", "test_first",
                         "consistency", "status"]]
    n = len(myths)
    myths = myths.merge(glm, on=KEY, how="left").merge(ds, on=KEY, how="left").merge(rules, on=KEY, how="left")
    assert len(myths) == n
    myths["row"] = np.arange(n)
    myths["valid"] = myths["n_words"] >= MIN_WORDS
    ok = myths["status"] == "ok"
    for c in ["send_rule", "return_rule", "after_letdown"]:
        myths[c] = myths[c].where(ok & (myths[c] != "unspecified"))
    for c in ["test_first", "consistency"]:
        myths[c] = myths[c].where(ok).map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    myths["send_amount"] = myths["send_amount"].where(ok)
    myths["send_ord"] = myths["send_rule"].map(SEND_ORD)
    myths["label_level"] = myths["label"].map(LABEL_LEVEL)
    return myths, emb, semb


def decisions() -> pd.DataFrame:
    dec = pd.read_csv(DATA / "decisions.csv").sort_values(["run_id", "agent", "round"])
    dec["coop_lag_any"] = dec.groupby(["run_id", "agent"])["coop"].shift(1)
    dec["coop_lag_same"] = dec.groupby(["run_id", "agent", "role"])["coop"].shift(1)
    return dec


def agree(a, b):
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a == b)


def pair_measures(myths, emb, semb, i, j) -> dict:
    """Norm and text alignment between myth rows i and j (either may be None)."""
    if i is None or j is None:
        return {}
    A, B = myths.iloc[i], myths.iloc[j]
    out = {"same_label": agree(A.label, B.label), "same_label_ds": agree(A.label_ds, B.label_ds),
           "judges_agree": float(A.label == A.label_ds and B.label == B.label_ds)
           if isinstance(A.label, str) and isinstance(B.label, str) else np.nan,
           "moral_cos": float(semb[i] @ semb[j]) if isinstance(A.summary, str) and isinstance(B.summary, str) else np.nan,
           "myth_cos": float(emb[i] @ emb[j]) if A.valid and B.valid else np.nan,
           "f_send_rule": agree(A.send_rule, B.send_rule), "f_return_rule": agree(A.return_rule, B.return_rule),
           "f_after_letdown": agree(A.after_letdown, B.after_letdown),
           "f_send_amount": np.nan if pd.isna(A.send_amount) or pd.isna(B.send_amount)
           else 1 - abs(A.send_amount - B.send_amount) / 5,
           "f_consistency": agree(A.consistency, B.consistency), "f_test_first": agree(A.test_first, B.test_first),
           "send_ord_dist": np.nan if pd.isna(A.send_ord) or pd.isna(B.send_ord) else abs(A.send_ord - B.send_ord)}
    f = [out[k] for k in FIELDS if not pd.isna(out[k])]
    out["rule_index"] = float(np.mean(f)) if len(f) >= 2 else np.nan
    out["rule_n_fields"] = len(f)
    return out


def games_table() -> pd.DataFrame:
    myths, emb, semb = load_myths()
    idx = {(r, t, a): k for k, (r, t, a) in enumerate(zip(myths["run_id"], myths["round"], myths["agent"]))}
    dec = decisions()
    lag = dec.set_index(KEY)[["coop_lag_any", "coop_lag_same"]]
    inv = dec[dec["role"] == "investor"].rename(columns={"agent": "investor", "partner": "trustee",
                                                         "family": "inv_family", "partner_family": "tru_family"})
    g = inv[["run_id", "size", "mixed", "composition", "task_order", "replicate_id", "round", "investor", "trustee",
             "inv_family", "tru_family", "sent", "return_proportion"]].copy().reset_index(drop=True)
    g["sent_frac"] = g["sent"] / 5
    g["giving_gap"] = (g["sent_frac"] - g["return_proportion"]).abs()
    g["pair_family"] = ["-".join(sorted(p)) for p in zip(g["inv_family"], g["tru_family"])]
    g["pair_type"] = np.where(g["inv_family"] == g["tru_family"], "same family", "cross family")
    g["setting"] = [f"{s}-agent {'mixed' if m else 'homogeneous'}" for s, m in zip(g["size"], g["mixed"])]
    g["pair"] = ["|".join(sorted(p)) for p in zip(g["investor"], g["trustee"])]
    g = g.sort_values(["run_id", "round"]).reset_index(drop=True)
    g["prior_meetings"] = g.groupby(["run_id", "pair"]).cumcount()
    for who in ["investor", "trustee"]:
        keys = list(zip(g["run_id"], g["round"], g[who]))
        g[f"{who[:3]}_lag_any"] = [lag["coop_lag_any"].get(k, np.nan) for k in keys]
        g[f"{who[:3]}_lag_same"] = [lag["coop_lag_same"].get(k, np.nan) for k in keys]

    rows_b, rows_a, lev = [], [], []
    for r in g.itertuples(index=False):
        rb = r.round if r.task_order == "myth_game" else r.round - 1
        ra = r.round + 1 if r.task_order == "myth_game" else r.round
        ib, jb = idx.get((r.run_id, rb, r.investor)), idx.get((r.run_id, rb, r.trustee))
        ia, ja = idx.get((r.run_id, ra, r.investor)), idx.get((r.run_id, ra, r.trustee))
        rows_b.append(pair_measures(myths, emb, semb, ib, jb))
        rows_a.append(pair_measures(myths, emb, semb, ia, ja))
        L = {}
        for who, k in (("inv", ib), ("tru", jb)):
            m = myths.iloc[k] if k is not None else None
            L[f"{who}_label"] = m.label if m is not None and isinstance(m.label, str) else np.nan
            L[f"{who}_label_level"] = m.label_level if m is not None else np.nan
            L[f"{who}_send_rule"] = (m.send_rule if isinstance(m.send_rule, str) else "unspecified") if m is not None else np.nan
            L[f"{who}_send_ord"] = m.send_ord if m is not None else np.nan
            L[f"{who}_return_rule"] = (m.return_rule if isinstance(m.return_rule, str) else "unspecified") if m is not None else np.nan
            L[f"{who}_myth_round"] = rb if k is not None else np.nan
        lev.append(L)
    before = pd.DataFrame(rows_b)
    after = pd.DataFrame(rows_a).add_suffix("_after")
    g = pd.concat([g, before, after, pd.DataFrame(lev)], axis=1)
    g["has_myths_before"] = g["inv_label"].notna() & g["tru_label"].notna()
    g["pair_label_level"] = (g["inv_label_level"] + g["tru_label_level"]) / 2
    g["pair_send_ord"] = (g["inv_send_ord"] + g["tru_send_ord"]) / 2
    g["label_pair"] = np.where(g["same_label"] == 1, "both " + g["inv_label"].astype(str).str.replace("be ", ""),
                               "different labels")
    for w in ["inv", "tru"]:
        g[f"{w}_lag_same_missing"] = g[f"{w}_lag_same"].isna().astype(float)
        g[f"{w}_lag_same_f"] = g[f"{w}_lag_same"].fillna(0)
    return g


def descriptives(g: pd.DataFrame) -> pd.DataFrame:
    d = g[g["has_myths_before"]]
    cols = MEASURES + ["give_align"] + FIELDS + ["sent_frac", "return_proportion", "giving_gap"]
    per_run = d.groupby(["setting", "composition", "task_order", "run_id"])[[c for c in cols if c in d]].mean()
    agg = per_run.groupby(["setting", "composition", "task_order"]).agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    n = d.groupby(["setting", "composition", "task_order"]).agg(n_games=("run_id", "size"), n_runs=("run_id", "nunique"))
    return n.join(agg).reset_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    BIG.mkdir(parents=True, exist_ok=True)
    g = games_table()
    g.to_csv(BIG / "norm_alignment_games.csv", index=False)
    print(f"{DATASET}: {len(g)} games, {int(g['has_myths_before'].sum())} with both myths before -> {BIG}")


if __name__ == "__main__":
    main()
