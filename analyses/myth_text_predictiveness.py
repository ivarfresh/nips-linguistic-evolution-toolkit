#!/usr/bin/env python3
"""Does the myth text predict cooperation at all, beyond what past play predicts?

A ceiling check for the moral labels. If the full myth text (a 768-number
sentence embedding) cannot predict the next game beyond the players' previous
game, no set of moral categories extracted from it can. If the text predicts
but the three moral labels do not, the labels are losing information.

Corpus: the September informed negative-only myths (analyses/linguistic_corpus.py,
8,520 myths, 8,519 of at least 20 words) and their all-mpnet-base-v2 embeddings (analyses/linguistic_uptake.py).
Every test is run separately per task order, never pooled:
  myth_game  round t: myth, then game t   -> the myth's next game is game t
  game_myth  round t: game t, then myth   -> the myth's next game is game t+1
and separately for the two decisions: amount sent (/5) by senders, and return
proportion by receivers.

Tests
  opening  myth_game round 1 only: the author's myth, written before any play,
           against the author's round-1 decision. Base = composition cell + family.
  author   the author's myth against the author's next decision, t >= 2 for
           myth_game. Base = cell, family, round, the author's own last decision
           in each role, and what the author's last partner did.
  reader   the myth the author was SHOWN before writing (its previous partner's
           myth) against the author's next decision. Same base; the reader's own
           myth is not controlled (total association of what was read).
Each test also runs "within agent": outcome and every feature demeaned per
(run, agent), so only an agent's round-to-round variation counts.

Text feature sets added to the base, one at a time:
  embedding       all-mpnet-base-v2 vector, 50 principal components
  tfidf           word and word-pair counts (TF-IDF), 100 SVD components; unlike the
                  embedding it sees specific words such as "five" or "half"
  label_glm       the three-moral label, GLM-5.2
  label_deepseek  the three-moral label, DeepSeek V4 Flash
  rules_pr4       PR #4's rule extraction (GLM-5.2): send amount and send, return and
                  letdown rules, test-first, consistency, noise. A targeted positive
                  control. Read from data/analysis/linguistic_20260923/
                  myth_rules_september_z-ai__glm-5.2.csv (analyses/myth_rule_judge.py, PR #4).
PCA and SVD are unsupervised (they never see the outcome) and are fitted once per test.

Score: out-of-sample R^2, 5-fold cross-validation grouped by run. Two-stage
ridge inside each fold: the base first, then the text features on the base's
training residuals with their own penalty (up to 1e7, i.e. switched off), so
an uninformative feature set scores about zero rather than below it. Folds grouped
by run (no run in both train and test), 10 repeats with reshuffled folds.
Reported: gain in R^2 over the base, mean (±std over repeats), and a 95% interval
from resampling whole runs (1,000 draws, out-of-fold predictions held fixed). A
gain is "clear" when that interval lies above zero. A permutation p (text shuffled
among myths of the same family, round and composition, 20 shuffles, each scored on
the same folds as its base) is kept as a secondary check; it separates signal from
noise poorly because stage 2 shrinks shuffled text to about zero.

Outputs: docs/figures/myth_text_predictiveness_20260930/{results.csv,README.md}
No API calls.

  python3 analyses/myth_text_predictiveness.py
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread per worker process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/analysis/linguistic_20260923"
LABELS = ROOT / "docs/figures/linguistic_analysis_20260923/moral_labels.csv"
OUT = ROOT / "docs/figures/myth_text_predictiveness_20260930"
KEY = ["run_id", "round", "agent"]
N_PC = 50
ALPHAS = np.logspace(-2, 4, 13)
TEXT_ALPHAS = np.logspace(-2, 7, 19)
MORALS = ["be generous", "be fair", "be cautious"]
RULES = DATA / "myth_rules_september_z-ai__glm-5.2.csv"
N_SVD = 100


def rule_features(myths: pd.DataFrame) -> np.ndarray:
    """PR #4 rule fields as numbers, one row per myths.csv row (zeros where missing)."""
    r = myths[KEY].merge(pd.read_csv(RULES), on=KEY, how="left")
    amt = r.send_amount
    parts = [pd.DataFrame({"send_amount": amt.fillna(amt.mean()) / 5, "send_amount_missing": amt.isna().astype(float)})]
    for c in ["send_rule", "return_rule", "after_letdown"]:
        parts.append(pd.get_dummies(r[c], prefix=c, dtype=float))
    for c in ["test_first", "consistency", "noise_mentioned"]:
        parts.append(r[c].astype(str).str.lower().eq("true").astype(float).rename(c))
    return pd.concat(parts, axis=1).to_numpy()


def lag_table(dec: pd.DataFrame) -> pd.DataFrame:
    """Per agent and round: the agent's own decision in each role and what its partner did."""
    d = dec.copy()
    d["own_send"] = np.where(d.role == "investor", d.sent / 5, np.nan)
    d["own_return"] = np.where(d.role == "trustee", d.return_proportion, np.nan)
    d["partner_send"] = np.where(d.role == "trustee", d.sent / 5, np.nan)  # what the partner sent me
    d["partner_return"] = np.where(d.role == "investor", d.return_proportion, np.nan)
    d = d.sort_values(KEY[:1] + ["agent", "round"])
    cols = ["own_send", "own_return", "partner_send", "partner_return"]
    # carry each quantity forward so a lag always holds the latest value in that role
    d[cols] = d.groupby(["run_id", "agent"])[cols].ffill()
    return d[KEY + ["role"] + cols]


def build(myths: pd.DataFrame, dec: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    myths = myths.reset_index().rename(columns={"index": "emb_row"})
    myths = myths[myths.n_words >= 20]
    myths = myths.merge(labels[KEY + ["label_glm_5_2", "label_deepseek_v4_flash"]], on=KEY, how="left")
    lags = lag_table(dec)
    # which game follows the myth, and which game is the last one before it
    myths["next_round"] = np.where(myths.task_order == "myth_game", myths["round"], myths["round"] + 1)
    myths["lag_round"] = myths["next_round"] - 1
    nxt = dec[KEY + ["role", "sent", "return_proportion"]].rename(columns={"round": "next_round"})
    df = myths.merge(nxt, on=["run_id", "next_round", "agent"], how="inner")
    df["y"] = np.where(df.role == "investor", df.sent / 5, df.return_proportion)
    lag = lags.drop(columns="role").rename(columns={"round": "lag_round"})
    lag.columns = [c if c in ("run_id", "agent", "lag_round") else "lag_" + c for c in lag.columns]
    df = df.merge(lag, on=["run_id", "agent", "lag_round"], how="left")
    # the myth the author was shown before writing (its previous partner's myth)
    shown = myths[KEY + ["emb_row", "label_glm_5_2", "label_deepseek_v4_flash"]].rename(columns={
        "round": "exposed_round", "agent": "exposed_author", "emb_row": "shown_emb_row",
        "label_glm_5_2": "shown_label_glm_5_2", "label_deepseek_v4_flash": "shown_label_deepseek_v4_flash"})
    df = df.merge(shown, on=["run_id", "exposed_round", "exposed_author"], how="left")
    df["cell"] = df["size"].astype(str) + "_" + df["composition"]
    return df.dropna(subset=["y"]).reset_index(drop=True)


def base_matrix(df: pd.DataFrame, test: str) -> np.ndarray:
    parts = [pd.get_dummies(df["cell"], dtype=float), pd.get_dummies(df["family"], dtype=float)]
    if test != "opening":
        parts.append(pd.get_dummies(df["round"].astype(str), prefix="r", dtype=float))
        for c in ["lag_own_send", "lag_own_return", "lag_partner_send", "lag_partner_return"]:
            parts.append(pd.DataFrame({c: df[c].fillna(df[c].mean()), c + "_missing": df[c].isna().astype(float)}))
    return pd.concat(parts, axis=1).to_numpy()


def label_matrix(col: pd.Series) -> np.ndarray:
    return np.column_stack([(col == m).astype(float) for m in MORALS[:2]])  # cautious = reference


def demean(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(x)
    return (frame - frame.groupby(groups).transform("mean")).to_numpy()


def cv_pred(base: np.ndarray, text: np.ndarray | None, y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    """Out-of-fold predictions of ridge on [base, text], folds grouped by run and set by `seed`."""
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    remap = dict(zip(uniq, perm))
    g = np.array([remap[v] for v in groups])
    pred = np.empty_like(y)
    for tr, te in GroupKFold(5).split(base, y, g):
        # stage 1: past play and design cells; stage 2: text on what stage 1 leaves over,
        # with its own penalty, so an uninformative text block is shrunk to zero
        # instead of dragging the stage-1 fit down
        sc = StandardScaler().fit(base[tr])
        m1 = RidgeCV(alphas=ALPHAS).fit(sc.transform(base[tr]), y[tr])
        pred[te] = m1.predict(sc.transform(base[te]))
        if text is not None:
            resid = y[tr] - m1.predict(sc.transform(base[tr]))
            st = StandardScaler().fit(text[tr])
            m2 = RidgeCV(alphas=TEXT_ALPHAS).fit(st.transform(text[tr]), resid)
            pred[te] += m2.predict(st.transform(text[te]))
    return pred


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def run_interval(y, base_preds, full_preds, groups, n=1000, seed=0) -> tuple[float, float]:
    """95% interval of the R^2 gain from resampling whole runs, averaged over fold shuffles.
    Out-of-fold predictions are held fixed (no refit), so the interval is somewhat narrow."""
    rng = np.random.default_rng(seed)
    runs = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in runs}
    gains = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(runs, len(runs))])
        gains.append(np.mean([r2(y[s], f[s]) - r2(y[s], b[s]) for b, f in zip(base_preds, full_preds)]))
    return tuple(np.quantile(gains, [0.025, 0.975]))


def shuffle_within(n: int, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = np.arange(n)
    out = idx.copy()
    for s in np.unique(strata):
        m = idx[strata == s]
        out[m] = rng.permutation(m)
    return out


def evaluate(df: pd.DataFrame, emb: np.ndarray, tfidf, rules: np.ndarray, test: str, within: bool, repeats: int,
             n_perm: int) -> list[dict]:
    rows_col = "shown_emb_row" if test == "reader" else "emb_row"
    df = df.dropna(subset=[rows_col]).reset_index(drop=True)
    if within:
        df = df[df.groupby(["run_id", "agent"]).y.transform("size") >= 3].reset_index(drop=True)
    if len(df) < 60 or df.run_id.nunique() < 10 or df.y.std() == 0:
        return []
    y = df.y.to_numpy(float)
    groups = df.run_id.to_numpy()
    base = base_matrix(df, test)
    idx = df[rows_col].astype(int).to_numpy()
    # PCA is unsupervised (never sees y), so fitting it once on all rows leaks no outcome
    pcs = PCA(N_PC, random_state=0).fit_transform(emb[idx])
    prefix = "shown_" if test == "reader" else ""
    feats = {"embedding": pcs,
             "tfidf": TruncatedSVD(N_SVD, random_state=0).fit_transform(tfidf[idx]),
             **({"rules_pr4": rules[idx]} if rules is not None else {}),
             "label_glm": label_matrix(df[prefix + "label_glm_5_2"]),
             "label_deepseek": label_matrix(df[prefix + "label_deepseek_v4_flash"])}
    agent_key = (df.run_id + "|" + df.agent).to_numpy()
    if within:
        y = y - pd.Series(y).groupby(agent_key).transform("mean").to_numpy()
        base = demean(base, agent_key)
    strata = (df.family + "|" + df["round"].astype(str) + "|" + df.cell).to_numpy()
    # one base fit per fold shuffle; every text fit and every permutation is scored
    # against the base on the SAME folds
    base_preds = [cv_pred(base, None, y, groups, s) for s in range(max(repeats, n_perm))]
    base_r2 = np.array([r2(y, b) for b in base_preds])
    out = []
    for name, x in feats.items():
        xw = demean(x, agent_key) if within else x
        full_preds = [cv_pred(base, xw, y, groups, s) for s in range(repeats)]
        gains = np.array([r2(y, f) for f in full_preds]) - base_r2[:repeats]
        lo, hi = run_interval(y, base_preds[:repeats], full_preds, groups)
        rng = np.random.default_rng(1)
        null = []
        for k in range(n_perm):
            p = shuffle_within(len(y), strata, rng)
            null.append(r2(y, cv_pred(base, xw[p], y, groups, k)) - base_r2[k])
        null = np.array(null)
        out.append({"feature": name, "n_decisions": len(y), "n_runs": int(df.run_id.nunique()),
                    "outcome_sd": float(df.y.std()),
                    "base_r2": float(base_r2[:repeats].mean()), "gain_r2": float(gains.mean()),
                    "gain_sd": float(gains.std()), "gain_ci_low": float(lo), "gain_ci_high": float(hi),
                    "null_gain_mean": float(null.mean()), "null_gain_p95": float(np.quantile(null, 0.95)),
                    "p_perm": float((1 + np.sum(null >= gains.mean())) / (1 + len(null)))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--perm", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    myths = pd.read_csv(DATA / "myths.csv")
    myths["text"] = myths.text.fillna("")
    emb = np.load(DATA / "embeddings_mpnet.npy")
    assert len(emb) == len(myths), "embeddings out of step with myths.csv"
    dec = pd.read_csv(DATA / "decisions.csv")
    labels = pd.read_csv(LABELS)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True).fit_transform(myths.text)
    # PR #4's extraction is optional: skip that feature set on a checkout without it
    rules = rule_features(myths) if RULES.exists() else None
    df = build(myths, dec, labels)
    df["decision"] = np.where(df.role == "investor", "send", "return")

    jobs = []
    for task_order in ["myth_game", "game_myth"]:
        for decision in ["send", "return"]:
            sub = df[(df.task_order == task_order) & (df.decision == decision)]
            for test in ["opening", "author", "reader"]:
                if test == "opening":
                    if task_order != "myth_game":
                        continue
                    t = sub[sub["round"] == 1]
                elif test == "author" and task_order == "myth_game":
                    t = sub[sub["round"] >= 2]
                else:
                    t = sub
                for family in ["all", "Sonnet", "GPT", "Gemini"]:
                    tf = t if family == "all" else t[t.family == family]
                    for within in ([False] if test == "opening" else [False, True]):
                        jobs.append(({"task_order": task_order, "decision": decision, "test": test,
                                      "author_family": family, "within_agent": within}, tf, test, within))
    print(f"{len(jobs)} tests on {args.jobs} processes", flush=True)
    done = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(evaluate)(tf, emb, tfidf, rules, test, within, args.repeats, args.perm)
        for _, tf, test, within in jobs)
    results = [{**meta, **r} for (meta, *_), rows in zip(jobs, done) for r in rows]
    for r in results:
        print(f"{r['task_order']:9s} {r['decision']:6s} {r['test']:7s} {r['author_family']:6s} "
              f"within={r['within_agent']!s:5s} {r['feature']:14s} n={r['n_decisions']:5d} "
              f"base={r['base_r2']:.3f} gain={r['gain_r2']:+.3f} (±{r['gain_sd']:.3f}) "
              f"[{r['gain_ci_low']:+.3f}, {r['gain_ci_high']:+.3f}] p={r['p_perm']:.2f}")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUT / "results.csv", index=False)
    print(f"-> {OUT / 'results.csv'}")


if __name__ == "__main__":
    main()
