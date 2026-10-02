#!/usr/bin/env python3
"""Open search on the DISCOVERY split only: how much held-out predictive gain do myth features
add over a behavioural baseline?

Ported from the September search lens (MP_DATASET=september|frontier). Strata: pooled, each
family, and each family x task order; a cell with no outcome spread gets a "not estimable:
ceiling" row instead of a fit. For each stratum x outcome (send/5, return proportion, change in
send) x learner (ridge, gradient boosting) x feature set, 5-fold CV grouped by run gives
out-of-fold predictions; the gain is R2(baseline + set) - R2(baseline) with the same learner.
CIs: paired bootstrap over runs (1,000) on the out-of-fold predictions. Within-agent version:
outcome and all columns demeaned inside agent-within-run (ridge only; Nickell-biased, read as
predictive only). Group permutation importance on the full model. Single-feature screen picks
the top-5 candidates, frozen to candidates_top5.csv before any confirmatory row is touched.

Outputs: gain_table.csv, group_importance.csv, single_feature_screen.csv, candidates_top5.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
from common import CFG, D, NAME, OUT, ceiling  # noqa: E402
RNG = np.random.default_rng(0)
NPC = 10

BASE_NUM = ["round", "lag_send", "lag_return", "lag_got_return", "lag_got_send", "mean_send_sofar",
            "mean_return_sofar", "cop_last3_send", "cop_last3_return", "received_now"]
BASE_CAT = ["family", "partner_family", "size", "mixed", "task_order"]
GROUP_PREFIX = {"rules": "rule_", "labels": "label_", "length": ("n_words", "n_sentences", "heading"),
                "lexicon": "lex_", "sentiment": "sent_", "style": "style_", "change": "chg_"}
EMB_GROUPS = {"text_emb": "embeddings_mpnet.npy", "moral_emb": "embeddings_moral_summary_mpnet.npy"}
HGB_SETS = {"ALL own myth", "ALL shown myth", "ALL myth features", "NOISE placebo (same width)"}
OUTCOMES = {"send": ("investor", "coop"), "return": ("trustee", "coop"), "dsend": ("investor", "dsend")}


def group_cols(df: pd.DataFrame, group: str, src: str) -> list[str]:
    pre = GROUP_PREFIX[group]
    names = [c[len(src) + 1:] for c in df.columns if c.startswith(src + "_")]
    if isinstance(pre, tuple):
        return [f"{src}_{n}" for n in names if n in pre]
    return [f"{src}_{n}" for n in names if n.startswith(pre)]


def base_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[BASE_NUM].copy()
    for c in BASE_CAT:
        X = X.join(pd.get_dummies(df[c].astype(str), prefix=c, dtype=float))
    X = X.join(pd.get_dummies(df["round"].astype(str), prefix="rd", dtype=float))
    return X


def impute(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train-mean imputation plus missing indicators for columns with any missing value."""
    mu = np.nanmean(Xtr, axis=0)
    mu = np.where(np.isnan(mu), 0, mu)
    miss = np.isnan(Xtr).any(0) | np.isnan(Xte).any(0)
    out = []
    for X in (Xtr, Xte):
        ind = np.isnan(X[:, miss]).astype(float)
        X = np.where(np.isnan(X), mu, X)
        out.append(np.hstack([X, ind]))
    src = np.concatenate([np.arange(Xtr.shape[1]), np.where(miss)[0]])  # source column of each output column
    return out[0], out[1], src


def demean(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    Xd = pd.DataFrame(X).groupby(g).transform("mean").to_numpy()
    return X - Xd


def ridge_path(X, y, alphas):
    ym = y.mean()
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    uty = U.T @ (y - ym)
    return [(Vt.T @ (s / (s ** 2 + a) * uty), ym) for a in alphas]


def ridge_loo(Xtr, ytr, Xte, groups, alphas=np.logspace(-1, 5, 13)):
    """Ridge on standardized columns with an intercept. Alpha is chosen by an inner 4-fold CV
    grouped by run, so the penalty is tuned for new runs, not for rows of runs already seen."""
    err = np.zeros(len(alphas))
    for i_tr, i_va in GroupKFold(4).split(Xtr, groups=groups):
        for k, (c, m) in enumerate(ridge_path(Xtr[i_tr], ytr[i_tr], alphas)):
            err[k] += np.sum((ytr[i_va] - Xtr[i_va] @ c - m) ** 2)
    c, m = ridge_path(Xtr, ytr, [alphas[int(np.argmin(err))]])[0]
    return Xte @ c + m


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


class Fitter:
    def __init__(self, df: pd.DataFrame, ycol: str, within: bool):
        self.df = df.reset_index(drop=True)
        self.y = self.df[ycol].to_numpy(float)
        self.within = within
        self.groups = self.df["run_id"].to_numpy()
        self.ra = self.df["run_agent"].to_numpy()
        self.folds = list(GroupKFold(5).split(self.df, groups=self.groups))
        self.B = base_matrix(self.df)
        self.emb = {k: np.load(D / f) for k, f in EMB_GROUPS.items()}

    def emb_pcs(self, name: str, src: str, tr: np.ndarray) -> np.ndarray:
        """PCA fitted on training rows' myths only, applied to all rows."""
        ii = self.df[f"{src}_i"].to_numpy()
        E = self.emb[name]
        ok = ii >= 0
        pca = PCA(NPC, random_state=0).fit(E[np.unique(ii[tr & ok])])
        Z = np.full((len(ii), NPC), np.nan)
        Z[ok] = pca.transform(E[ii[ok]])
        return Z

    def design(self, cols: list[str], embs: list[tuple[str, str]], tr: np.ndarray) -> tuple[np.ndarray, list]:
        parts = [self.B.to_numpy(float)] + ([self.df[cols].to_numpy(float)] if cols else [])
        names = list(self.B.columns) + cols
        for name, src in embs:
            parts.append(self.emb_pcs(name, src, tr))
            names += [f"{src}_{name}_pc{k}" for k in range(NPC)]
        return np.hstack(parts), names

    def oof(self, cols, embs, learner, perm_groups=None):
        pred = np.zeros(len(self.y))
        imp = {g: [] for g in (perm_groups or {})}
        for tr_i, te_i in self.folds:
            tr = np.zeros(len(self.y), bool)
            tr[tr_i] = True
            X, names = self.design(cols, embs, tr)
            y = self.y
            if learner == "ridge":
                Xtr, Xte, src = impute(X[tr_i], X[te_i])
                if self.within:
                    Xtr, Xte = demean(Xtr, self.ra[tr_i]), demean(Xte, self.ra[te_i])
                    ytr, yte = demean(y[tr_i, None], self.ra[tr_i])[:, 0], demean(y[te_i, None], self.ra[te_i])[:, 0]
                else:
                    ytr, yte = y[tr_i], y[te_i]
                sd = Xtr.std(0)
                keep = sd > 1e-9
                m = Xtr[:, keep].mean(0)
                Xtr, Xte = (Xtr[:, keep] - m) / sd[keep], (Xte[:, keep] - m) / sd[keep]
                # Two stages so myth columns cannot degrade the baseline through a shared penalty:
                # (1) ridge on the baseline columns, (2) a separately tuned ridge of the myth
                # columns on the stage-1 residual. Identical stage 1 in baseline and augmented runs.
                isb = src[keep] < self.B.shape[1]
                g = self.groups[tr_i]
                b_tr, b_te = ridge_loo(Xtr[:, isb], ytr, Xtr[:, isb], g), ridge_loo(Xtr[:, isb], ytr, Xte[:, isb], g)
                pred[te_i] = b_te
                if (~isb).any():
                    pred[te_i] += ridge_loo(Xtr[:, ~isb], ytr - b_tr, Xte[:, ~isb], g,
                                            alphas=np.logspace(-1, 7, 17))
                colmap = None
            else:
                ytr, yte = y[tr_i], y[te_i]
                model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                                      min_samples_leaf=20, l2_regularization=1.0,
                                                      random_state=0).fit(X[tr_i], ytr)
                Xte = X[te_i]
                pred[te_i] = model.predict(Xte)
                colmap = names
            if perm_groups:
                base_r2 = r2(yte, pred[te_i])
                for g, gnames in perm_groups.items():
                    if colmap is None:
                        continue
                    ci = [colmap.index(n) for n in gnames if n in colmap]
                    drops = []
                    for _ in range(5):
                        Xp = Xte.copy()
                        Xp[:, ci] = Xp[RNG.permutation(len(Xp))][:, ci]
                        drops.append(base_r2 - r2(yte, model.predict(Xp)))
                    imp[g].append(np.mean(drops))
        ytrue = self.y
        if self.within:
            ytrue = np.zeros_like(self.y)
            for _, te_i in self.folds:
                ytrue[te_i] = demean(self.y[te_i, None], self.ra[te_i])[:, 0]
        return pred, ytrue, imp


def boot_gain(y, p0, p1, groups, n=1000):
    runs = np.unique(groups)
    pos = {r: np.where(groups == r)[0] for r in runs}
    g0, g1 = r2(y, p0), r2(y, p1)
    gains, shares = [], []
    for _ in range(n):
        ii = np.concatenate([pos[r] for r in RNG.choice(runs, len(runs))])
        a, b = r2(y[ii], p0[ii]), r2(y[ii], p1[ii])
        gains.append(b - a)
        shares.append((b - a) / (1 - a))
    return {"r2_base": g0, "r2_aug": g1, "gain": g1 - g0, "gain_lo": np.percentile(gains, 2.5),
            "gain_hi": np.percentile(gains, 97.5), "resid_share": (g1 - g0) / (1 - g0),
            "resid_share_lo": np.percentile(shares, 2.5), "resid_share_hi": np.percentile(shares, 97.5)}


def feature_sets(df):
    sets = {}
    allcols, allembs = [], []
    for g in GROUP_PREFIX:
        cols = group_cols(df, g, "own") + (group_cols(df, g, "shown") if g != "change" else [])
        sets[g] = (cols, [])
        allcols += cols
    for g in EMB_GROUPS:
        sets[g] = ([], [(g, "own"), (g, "shown")])
        allembs += [(g, "own"), (g, "shown")]
    own_cols = [c for c in allcols if c.startswith("own_")]
    shown_cols = [c for c in allcols if c.startswith("shown_")]
    sets["ALL own myth"] = (own_cols, [e for e in allembs if e[1] == "own"])
    sets["ALL shown myth"] = (shown_cols, [e for e in allembs if e[1] == "shown"])
    sets["ALL myth features"] = (allcols, allembs)
    if any("_style_" in c for c in allcols):  # September only: the frontier bank has no style group
        sets["ALL myth features (no style)"] = ([c for c in allcols if "_style_" not in c], allembs)
    # Calibration: as many pure-noise columns as the full myth set has (after embedding PCs).
    width = len(allcols) + NPC * len(allembs)
    assert width <= 200, width
    sets["NOISE placebo (same width)"] = ([f"noise_{k}" for k in range(width)], [])
    return sets


def strata(d, role, ycol):
    """Pooled, per family, and (frontier) per family x task order. Yields (name, rows, status):
    status None = fit; otherwise the reason it is not fitted."""
    sub = d[d["role"] == role]
    yield "pooled", sub, None
    for fam in CFG["families"]:
        s = sub[sub["family"] == fam]
        yield fam, s, ("not estimable: ceiling" if ceiling(s[ycol]) else None)
        if NAME == "frontier":
            for to, st in s.groupby("task_order"):
                if st["run_id"].nunique() < 8:
                    yield f"{fam} {to}", st, "underpowered"
                    continue
                yield f"{fam} {to}", st, ("not estimable: ceiling" if ceiling(st[ycol]) else None)


def main():
    d = pd.read_csv(OUT / "decision_table.csv")
    d = d[(d["split"] == "discovery") & (d["own_i"] >= 0) & (d["shown_i"] >= 0)].copy()
    nz = np.random.default_rng(7).normal(size=(len(d), 200))
    d = pd.concat([d.reset_index(drop=True), pd.DataFrame(nz, columns=[f"noise_{k}" for k in range(200)])], axis=1)
    print("discovery decisions with own + shown myth:", d.groupby("role").size().to_dict(), "runs", d.run_id.nunique())
    sets = feature_sets(d)
    rows, imps = [], []
    for oname, (role, ycol) in OUTCOMES.items():
        dd = d.dropna(subset=[ycol])
        for sname, sub, status in strata(dd, role, ycol):
            if status:
                rows.append({"outcome": oname, "stratum": sname, "feature_set": "ALL myth features", "n": len(sub),
                             "n_runs": sub["run_id"].nunique(), "status": status,
                             "y_sd": sub[ycol].std()})
                continue
            if HEADLINE and not (oname == "send" and sname == "pooled"):
                continue
            for within in (False, True):
                if HEADLINE and within:
                    continue
                learners = ["ridge"] if within else ["ridge", "hgb"]
                for learner in (["ridge"] if HEADLINE else learners):
                    F = Fitter(sub, ycol, within)
                    p0, y, _ = F.oof([], [], learner)
                    for setname, (cols, embs) in sets.items():
                        if learner == "hgb" and setname not in HGB_SETS:
                            continue
                        if HEADLINE and not setname.startswith("ALL myth features"):
                            continue
                        perm = None
                        if setname == "ALL myth features" and learner == "hgb":
                            perm = {}
                            for g in GROUP_PREFIX:
                                for src in ("own", "shown"):
                                    perm[f"{src} {g}"] = group_cols(sub, g, src)
                            for g in EMB_GROUPS:
                                for src in ("own", "shown"):
                                    perm[f"{src} {g}"] = [f"{src}_{g}_pc{k}" for k in range(NPC)]
                        p1, _, imp = F.oof(cols, embs, learner, perm)
                        res = boot_gain(y, p0, p1, F.groups)
                        rows.append({"outcome": oname, "stratum": sname, "version": "within-agent" if within else "raw",
                                     "learner": learner, "feature_set": setname, "n": len(y),
                                     "n_runs": len(np.unique(F.groups)), "status": "fitted",
                                     "y_sd": float(np.std(F.y)), **res})
                        for g, v in imp.items():
                            imps.append({"outcome": oname, "stratum": sname, "group": g,
                                         "perm_drop_r2": np.mean(v) if v else np.nan})
                    print(oname, sname, within, learner, "done", flush=True)
                    pd.DataFrame(rows).to_csv(OUT / ("gain_table_headline.csv" if HEADLINE else "gain_table.csv"), index=False)
    if HEADLINE:
        return
    pd.DataFrame(imps).to_csv(OUT / "group_importance.csv", index=False)
    screen(d)


def screen(d):
    """Single-feature screen: ridge, baseline + one feature, pooled, raw and within-agent."""
    feats = sorted({c.split("_", 1)[1] for c in d.columns if c.startswith("own_") and c not in ("own_i",)})
    rows = []
    for oname in ("send", "return"):
        role, ycol = OUTCOMES[oname]
        sub = d[d["role"] == role].dropna(subset=[ycol])
        for within in (False, True):
            F = Fitter(sub, ycol, within)
            p0, y, _ = F.oof([], [], "ridge")
            for src in ("own", "shown"):
                for f in feats:
                    col = f"{src}_{f}"
                    if col not in sub or sub[col].notna().sum() < 100:
                        continue
                    p1, _, _ = F.oof([col], [], "ridge")
                    x = sub[col].to_numpy(float)
                    rows.append({"outcome": oname, "version": "within-agent" if within else "raw", "source": src,
                                 "feature": f, "gain": r2(y, p1) - r2(y, p0),
                                 "corr_resid": np.corrcoef(np.nan_to_num(x, nan=np.nanmean(x)), y - p0)[0, 1]})
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "single_feature_screen.csv", index=False)
    agg = s.groupby(["feature"]).agg(mean_gain=("gain", "mean"), max_gain=("gain", "max"),
                                     n_pos=("gain", lambda v: int((v > 0).sum())), n=("gain", "size")).reset_index()
    # Change features cannot be tested at R4 (no previous myth); keep them in the screen but rank
    # the candidates on features testable at both R3 and R4.
    agg = agg[~agg["feature"].str.startswith("chg_")].sort_values("mean_gain", ascending=False)
    agg.to_csv(OUT / "single_feature_screen_ranked.csv", index=False)
    # September froze its top-5 here; on frontier the same screen only nominates exploratory candidates.
    agg.head(5).to_csv(OUT / ("candidates_top5.csv" if NAME == "september" else "candidates_top5_exploratory.csv"),
                       index=False)
    print(agg.head(15).to_string())


HEADLINE = sys.argv[1:] == ["headline"]  # send / pooled / raw / ridge only (reproduction check)

if __name__ == "__main__":
    if sys.argv[1:] == ["screen"]:
        d = pd.read_csv(OUT / "decision_table.csv")
        d = d[(d["split"] == "discovery") & (d["own_i"] >= 0) & (d["shown_i"] >= 0)].copy()
        screen(d)
    else:
        main()
