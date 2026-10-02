#!/usr/bin/env python3
"""Confirmatory tests of September's FROZEN top-5 candidates (candidates_top5_september_frozen.csv)
on the held-out rungs, ported from the September search lens (MP_DATASET=september|frontier).

R4 (founding window): myth_game round-1 decisions. Within a family x composition cell the only
input that differs between senders is the agent's own myth (frontier check:
analyses/frontier_round1_identification.py).
   coop ~ z(own feature) + C(cell) [+ received_now for receivers], SE clustered by run.
R3 (clean exposure): 8-agent myth_game, rounds >= 2; shown myth = last round's partner's.
   Unseen placebo = mean feature of same-family myths from the same run and round (not the
   reader's, not the author's). Agent-within-run + round FE, own lagged moves, co-player's last
   3 games, author's move toward the reader last round. Test = shown - unseen.
   Samples: "never played before" (September's primary) and "author != current partner"
   (settled spec), both reported; Holm uses the September-matching sample.

Changes from the September script:
 - Family filters are data-driven: a family whose outcome has no spread in a rung is reported as
   "not estimable: ceiling" and left out of the pooled model (September dropped Gemini by name).
 - Strata: pooled, each family, and (frontier) each family x setting. Holm within each stratum.
 - Few clusters: every primary (Holm) test also gets a wild cluster restricted bootstrap-t p (Webb weights,
   1,999 draws); Holm is given on both p's. Fewer than 5 runs = "underpowered".
 - z-scoring guards against families with no variance in a feature.
Output: confirm_R3_R4.csv (all rows incl. non-estimable cells, with a status column).
`python confirm.py exploratory` instead tests this dataset's own screen picks
(candidates_top5_exploratory.csv, minus the frozen five) as a separate exploratory Holm family and
writes confirm_R3_R4_exploratory.csv.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from common import CFG, D, HERE, NAME, OUT, ceiling

warnings.filterwarnings("ignore")
LAGS = ["lag_send", "lag_return", "lag_got_return", "lag_got_send", "cop_last3_send", "cop_last3_return"]
CTRL = " + ".join(c + "_f" for c in LAGS)
WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
RNG = np.random.default_rng(11)


def fill(d, cols):
    for c in cols:
        d[c + "_miss"] = d[c].isna().astype(float)
        d[c + "_f"] = d[c].fillna(d[c].mean())
    return d


def fit(formula, data):
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data["run_id"])[0]})


def wild_p(formula, data, weights: dict, B=1999):
    """Wild cluster restricted bootstrap-t p for H0: sum_k weights[k] * beta_k = 0 (CR1 SEs)."""
    y, X = patsy.dmatrices(formula, data, return_type="dataframe")
    g = pd.factorize(data.loc[y.index, "run_id"])[0]
    G = g.max() + 1
    y, Xn = y.to_numpy()[:, 0], X.to_numpy()
    c = np.zeros(Xn.shape[1])
    for k, w in weights.items():
        c[list(X.columns).index(k)] = w
    P = np.linalg.pinv(Xn)
    Q = P @ P.T  # pinv(X'X)
    wv = P.T @ c  # c'beta = wv . y
    k = np.linalg.matrix_rank(Xn)
    fac = G / (G - 1) * (len(y) - 1) / (len(y) - k)

    Gm = np.zeros((G, len(y)))
    Gm[g, np.arange(len(y))] = 1.0
    Gw = Gm * wv  # cluster sums of wv * residual
    M = Xn @ P  # hat matrix

    def tstat(Y):
        U = Y - M @ Y
        return (wv @ Y) / np.sqrt(fac * ((Gw @ U) ** 2).sum(0))

    t0 = tstat(y[:, None])[0]
    beta = P @ y
    fit_r = Xn @ (beta - Q @ c * (c @ beta) / (c @ Q @ c))
    u_r = y - fit_r
    V = WEBB[RNG.integers(0, 6, size=(G, B))]
    ts = tstat(fit_r[:, None] + u_r[:, None] * V[g])
    return float(np.mean(np.abs(ts) >= abs(t0))), G


def zscore(feat, f):
    g = feat.groupby("family")[f]
    sd = g.transform("std")
    return ((feat[f] - g.transform("mean")) / sd.where(sd > 1e-9)).to_numpy()


def strata(s):
    yield "pooled", s
    for fam, sf in s.groupby("family"):
        yield fam, sf
        if NAME == "frontier":
            for st, ss in sf.groupby("setting"):
                yield f"{fam} | {st}", ss


def primary(base):
    return base["rung"] == "R4" or (base.get("sample") == "never played before" and base["estimate"].startswith("total"))


def test(rows, base, formula, data, a, b, ycol):
    """Fit one test and append a row with a status. a = feature term, b = placebo term or None."""
    rec = dict(base, n=len(data), n_runs=data["run_id"].nunique() if len(data) else 0)
    if len(data) == 0:
        rows.append(dict(rec, status="not testable by design"))
        return
    if data["run_id"].nunique() < 3:
        rows.append(dict(rec, status="underpowered"))
        return
    if ceiling(data[ycol]):
        rows.append(dict(rec, status="not estimable: ceiling", y_sd=data[ycol].std()))
        return
    if data[a].std() < 1e-9 or data[a].nunique() < 2:
        rows.append(dict(rec, status="not estimable: predictor has no spread"))
        return
    try:
        m = fit(formula, data)
        if b:
            w = m.t_test(f"{a} - {b} = 0")
            coef, p = float(np.asarray(w.effect).ravel()[0]), float(np.asarray(w.pvalue).ravel()[0])
            lo, hi = np.asarray(w.conf_int()).ravel()
            pw, G = wild_p(formula, data, {a: 1, b: -1}) if primary(base) else (np.nan, data["run_id"].nunique())
            extra = {"shown_coef": m.params[a], "shown_p": m.pvalues[a], "unseen_coef": m.params[b],
                     "unseen_p": m.pvalues[b]}
        else:
            coef, p = m.params[a], m.pvalues[a]
            lo, hi = m.conf_int().loc[a]
            pw, G = wild_p(formula, data, {a: 1}) if primary(base) else (np.nan, data["run_id"].nunique())
            extra = {}
    except Exception as e:  # noqa: BLE001
        rows.append(dict(rec, status=f"fit failed: {type(e).__name__}"))
        return
    rows.append(dict(rec, coef=coef, ci_low=lo, ci_high=hi, p=p, p_wild=pw, n=int(m.nobs),
                     y_sd=data[ycol].std(), x_sd=data[a].std(),
                     status="underpowered" if G < 5 else "fitted", **extra))


def main():
    frozen = pd.read_csv(HERE / "candidates_top5_september_frozen.csv")["feature"].tolist()
    cands = frozen
    if EXPLORATORY:
        cands = [f for f in pd.read_csv(OUT / "candidates_top5_exploratory.csv")["feature"] if f not in frozen]
    myths = pd.read_csv(D / "myths.csv")
    feat = pd.read_csv(OUT / "myth_features.csv")
    for f in cands:
        feat[f"z_{f}"] = zscore(feat, f)
    d = pd.read_csv(OUT / "decision_table.csv")
    dec_all = d
    for f in cands:
        zf = feat[f"z_{f}"].to_numpy()
        for src in ("own", "shown"):
            ii = d[f"{src}_i"].to_numpy()
            d[f"{src}_z_{f}"] = np.where(ii >= 0, zf[np.clip(ii, 0, None)], np.nan)
    d = fill(d, LAGS)
    d["run_agent"] = d["run_id"] + "|" + d["agent"]
    d["cell"] = d["size"].astype(str) + "|" + d["composition"] + "|" + d["family"]
    d["setting"] = d["size"].astype(str) + "-agent " + np.where(d["mixed"], "mixed", "homogeneous")
    rows = []

    # ---------------------------------------------------------------- R4
    r4 = d[d["split"] == "R4"]
    for oname, role in (("send", "investor"), ("return", "trustee")):
        s_all = r4[r4["role"] == role]
        ceil_fams = [fam for fam, g in s_all.groupby("family") if ceiling(g["coop"])]
        for f in cands:
            t = f"own_z_{f}"
            s = s_all.dropna(subset=[t, "coop"])
            form = f"coop ~ {t} + C(cell)" + (" + received_now" if role == "trustee" else "")
            for sname, ss in strata(s):
                if sname == "pooled":
                    ss = ss[~ss["family"].isin(ceil_fams)]
                test(rows, {"rung": "R4", "stratum": sname, "feature": f, "outcome": oname,
                            "sample": "senders" if role == "investor" else "receivers",
                            "estimate": "own myth, per within-family SD",
                            "pooled_excludes": ",".join(ceil_fams) if sname == "pooled" else ""},
                     form, ss, t, None, "coop")

    # ---------------------------------------------------------------- R3
    r3 = d[d["split"] == "R3"].copy()
    key = {(a, b, c): (p, co) for a, b, c, p, co in zip(dec_all.run_id, dec_all["round"], dec_all.agent,
                                                       dec_all.partner, dec_all.coop)}
    partners = dec_all.groupby(["run_id", "agent"]).apply(lambda g: dict(zip(g["round"], g["partner"]))).to_dict()
    auth_move, before = [], []
    for g in r3.itertuples():
        if not isinstance(g.shown_author, str):
            auth_move.append(np.nan); before.append(np.nan); continue
        pr = int(g.shown_round)
        p, co = key.get((g.run_id, pr, g.shown_author), (None, np.nan))
        auth_move.append(co if p == g.agent else np.nan)
        hist = partners.get((g.run_id, g.agent), {})
        before.append(float(any(hist.get(k) == g.shown_author for k in range(1, pr))))
    r3["author_move_prev"], r3["played_before"] = auth_move, before
    mi = myths.assign(i=np.arange(len(myths)))
    grp = {k: g for k, g in mi.groupby(["run_id", "round"])}
    for f in cands:
        zf = feat[f"z_{f}"].to_numpy()
        vals = []
        for g in r3.itertuples():
            if g.shown_i < 0:
                vals.append(np.nan); continue
            pool = grp.get((g.run_id, int(g.shown_round)))
            fam = myths.at[g.shown_i, "family"]
            c = pool[(pool["family"] == fam) & (~pool["agent"].isin([g.agent, g.shown_author]))]["i"].to_numpy()
            vals.append(np.nanmean(zf[c]) if len(c) else np.nan)
        r3[f"unseen_z_{f}"] = vals
    r3 = fill(r3, ["author_move_prev"])
    for oname, role, ycol in (("send", "investor", "coop"), ("return", "trustee", "coop"),
                              ("dsend", "investor", "dsend")):
        s_role = r3[r3["role"] == role]
        # a family whose send level has no spread cannot show send or change-in-send effects
        ceil_fams = [fam for fam, g in s_role.groupby("family") if ceiling(g["coop"])]
        for f in cands:
            for sample in ("never played before", "author != current partner"):
                s0 = s_role.dropna(subset=[ycol, f"shown_z_{f}", f"unseen_z_{f}"])
                s0 = s0[s0["played_before"] == 0] if sample == "never played before" else \
                    s0[s0["shown_author"] != s0["partner"]]
                ctrl = CTRL + " + author_move_prev_f + author_move_prev_miss" + \
                    (" + received_now" if role == "trustee" else "")
                for est, extra in (("total (shown - unseen)", ""),
                                   ("direct, own round-r myth held fixed", f" + own_z_{f}")):
                    s1 = s0.dropna(subset=[f"own_z_{f}"]) if extra else s0
                    form = f"{ycol} ~ shown_z_{f} + unseen_z_{f}{extra} + {ctrl} + C(run_agent) + C(round)"
                    for sname, ss in strata(s1):
                        base = {"rung": "R3", "stratum": sname, "feature": f, "outcome": oname, "sample": sample,
                                "estimate": est}
                        if sname == "pooled":
                            ss = ss[~ss["family"].isin(ceil_fams)]
                            base["pooled_excludes"] = ",".join(ceil_fams)
                        elif sname.split(" | ")[0] in ceil_fams:
                            rows.append(dict(base, n=len(ss), n_runs=ss["run_id"].nunique(),
                                             status="not estimable: ceiling"))
                            continue
                        test(rows, base, form, ss, f"shown_z_{f}", f"unseen_z_{f}", ycol)
    res = pd.DataFrame(rows)
    # Holm family per stratum: R4 send/return + R3 total (never-played-before sample) x 5 features;
    # cells that are not estimable are not counted. September's committed family = pooled, 25 tests.
    res["holm"] = (res["status"] == "fitted") & ((res["rung"] == "R4") | (
        (res["sample"] == "never played before") & res["estimate"].str.startswith("total")))
    for col, pcol in (("p_holm", "p"), ("p_holm_wild", "p_wild")):
        res[col] = np.nan
        for _, g in res[res["holm"]].groupby("stratum"):
            res.loc[g.index, col] = multipletests(g[pcol], method="holm")[1]
    res["n_holm_tests"] = res.groupby("stratum")["holm"].transform("sum")
    res.insert(0, "dataset", NAME)
    res.insert(1, "candidate_set", "exploratory (own screen)" if EXPLORATORY else "September frozen top-5")
    res.to_csv(OUT / ("confirm_R3_R4_exploratory.csv" if EXPLORATORY else "confirm_R3_R4.csv"), index=False)
    show = res[res["holm"] & (res["stratum"].isin(["pooled"] + CFG["families"]))]
    print(show[["rung", "stratum", "feature", "outcome", "coef", "ci_low", "ci_high", "p", "p_wild", "p_holm",
                "p_holm_wild", "n", "n_runs"]].round(4).to_string())
    print(res.groupby(["rung", "status"]).size())


EXPLORATORY = sys.argv[1:] == ["exploratory"]

if __name__ == "__main__":
    main()
