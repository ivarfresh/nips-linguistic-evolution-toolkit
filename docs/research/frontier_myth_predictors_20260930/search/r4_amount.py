#!/usr/bin/env python3
"""R4 follow-up in raw units (ported from the September search lens; MP_DATASET=september|frontier).
Per-SD coefficients depend on how spread out each corpus's stated amounts are, so the side-by-side
also needs: $ sent per $ stated (send ~ own stated amount + C(cell), SE by run, wild p), the
exact-match share, the send-rule slope for all senders, and the held-out R2 gain at round 1.
Outputs: r4_amount_models.csv, r4_predictability.csv, r4_amount_scatter.png"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import GroupKFold

import search as S
from common import CFG, NAME, OUT, ceiling
from confirm import wild_p

warnings.filterwarnings("ignore")


def fit(formula, data):
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data["run_id"])[0]})


def main():
    d = pd.read_csv(OUT / "decision_table.csv")
    r4 = d[(d["split"] == "R4") & (d["role"] == "investor")].copy()
    r4["cell"] = r4["size"].astype(str) + "|" + r4["composition"] + "|" + r4["family"]
    r4["send"] = r4["sent"]
    live = [f for f in CFG["families"] if not ceiling(r4.loc[r4.family == f, "send"] / 5)]
    groups = [("+".join(live), r4[r4.family.isin(live)])] + [(f, r4[r4.family == f]) for f in CFG["families"]]
    rows = []
    for fam, s in groups:
        a = s.dropna(subset=["own_rule_send_amount"])
        rec = {"dataset": NAME, "family": fam, "n_investors": len(s), "n_runs": s["run_id"].nunique(),
               "n_with_amount": len(a), "share_with_amount": len(a) / len(s) if len(s) else np.nan,
               "mean_send": s["send"].mean(), "sd_send": s["send"].std(),
               "stated_mean": a["own_rule_send_amount"].mean(), "stated_sd": a["own_rule_send_amount"].std(),
               "exact_match_share": float((np.isclose(a["send"], a["own_rule_send_amount"])).mean()) if len(a) else np.nan,
               "mean_abs_diff": float((a["send"] - a["own_rule_send_amount"]).abs().mean()) if len(a) else np.nan,
               "status": "not estimable: ceiling" if ceiling(s["send"] / 5) else "fitted"}
        if rec["status"] == "fitted" and len(a) > 10 and a["own_rule_send_amount"].nunique() > 1 and a["run_id"].nunique() > 5:
            m = fit("send ~ own_rule_send_amount + C(cell)", a)
            t = "own_rule_send_amount"
            rec.update({"slope_per_dollar": m.params[t], "slope_lo": m.conf_int().loc[t, 0],
                        "slope_hi": m.conf_int().loc[t, 1], "slope_p": m.pvalues[t],
                        "slope_p_wild": wild_p("send ~ own_rule_send_amount + C(cell)", a, {t: 1})[0],
                        "slope_n_runs": a["run_id"].nunique()})
        o = s.dropna(subset=["own_rule_send_ord"])
        if rec["status"] == "fitted" and len(o) > 10 and o["own_rule_send_ord"].nunique() > 1:
            m = fit("send ~ own_rule_send_ord + C(cell)", o)
            t = "own_rule_send_ord"
            rec.update({"n_rule": len(o), "rule_ord_slope": m.params[t], "rule_ord_lo": m.conf_int().loc[t, 0],
                        "rule_ord_hi": m.conf_int().loc[t, 1], "rule_ord_p": m.pvalues[t],
                        "rule_ord_p_wild": wild_p("send ~ own_rule_send_ord + C(cell)", o, {t: 1})[0]})
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT / "r4_amount_models.csv", index=False)
    print(pd.DataFrame(rows).round(3).T.to_string())

    # Held-out predictability at R4 (families with send spread only).
    s = r4[r4.family.isin(live)].reset_index(drop=True)
    y = s["send"].to_numpy(float) / 5
    cellX = pd.get_dummies(s["cell"], dtype=float).to_numpy()
    own_cols = [c for c in s.columns if c.startswith("own_") and c not in ("own_i",) and not c.startswith("own_chg_")]
    folds = list(GroupKFold(5).split(s, groups=s["run_id"]))

    def cv(Xm):
        p = np.zeros(len(y))
        for tr, te in folds:
            Xtr, Xte = Xm[tr], Xm[te]
            mu = np.nanmean(Xtr, 0)
            mu = np.where(np.isnan(mu), 0, mu)
            miss = np.isnan(Xtr).any(0) | np.isnan(Xte).any(0)
            Xtr = np.hstack([np.where(np.isnan(Xtr), mu, Xtr), np.isnan(Xtr[:, miss]).astype(float)])
            Xte = np.hstack([np.where(np.isnan(Xte), mu, Xte), np.isnan(Xte[:, miss]).astype(float)])
            sd = Xtr.std(0)
            k = sd > 1e-9
            m = Xtr[:, k].mean(0)
            p[te] = S.ridge_loo((Xtr[:, k] - m) / sd[k], y[tr], (Xte[:, k] - m) / sd[k], s["run_id"].to_numpy()[tr],
                                alphas=np.logspace(-2, 6, 17))
        return p

    out = []
    p0 = cv(cellX)
    for name, cols in [("stated amount + send rule", ["own_rule_send_amount", "own_rule_send_ord", "own_rule_send_amount_given"]),
                       ("all own-myth features (no embeddings)", own_cols)]:
        p1 = cv(np.hstack([cellX, s[cols].to_numpy(float)]))
        res = S.boot_gain(y, p0, p1, s["run_id"].to_numpy())
        out.append({"dataset": NAME, "families": "+".join(live), "model": name, "n": len(y),
                    "n_runs": s["run_id"].nunique(), **res})
    pr = pd.DataFrame(out)
    pr.to_csv(OUT / "r4_predictability.csv", index=False)
    print(pr.round(4).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    colors = {"Sonnet": "#7570b3", "GPT": "#d95f02", "Gemini": "#1b9e77",
              "Opus": "#7570b3", "Sol": "#d95f02", "GeminiPro": "#1b9e77"}
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for fam, g in r4.dropna(subset=["own_rule_send_amount"]).groupby("family"):
        jit = rng.uniform(-0.12, 0.12, (2, len(g)))
        ax.scatter(g["own_rule_send_amount"] + jit[0], g["send"] + jit[1], s=22, alpha=0.7, color=colors[fam],
                   label=f"{fam} (n={len(g)})")
    ax.plot([0, 5], [0, 5], color="grey", lw=1, ls="--")
    ax.set_xlabel("amount the agent's own round-1 myth says to send ($)")
    ax.set_ylabel("amount it then sends in round 1 ($)")
    ax.set_title(f"{NAME.title()}: round 1, myth before game", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "r4_amount_scatter.png", dpi=180)


if __name__ == "__main__":
    main()
