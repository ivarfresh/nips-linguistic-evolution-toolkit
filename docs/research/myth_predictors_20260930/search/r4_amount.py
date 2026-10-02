#!/usr/bin/env python3
"""Follow-up on the one confirmed candidate: at R4 (round-1 myth_game, the myth is the only input
that differs between agents), does the send amount the agent's own myth states set its send?

1. Raw-unit slope: send ($) ~ myth amount ($) + C(cell), SE by run; by family.
2. Exact-match rate: send == stated amount.
3. All round-1 investors (not only myths that state a number): send ~ send_rule ordinal.
4. How much round-1 send variance the own myth explains, held out (ridge, CV grouped by run):
   cell-only baseline vs cell + all own-myth features vs cell + stated amount / send rule.
Outputs: r4_amount_models.csv, r4_predictability.csv, r4_amount_scatter.png
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import search as S

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent


def fit(formula, data):
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data["run_id"])[0]})


def main():
    d = pd.read_csv(OUT / "decision_table.csv")
    r4 = d[(d["split"] == "R4") & (d["role"] == "investor")].copy()
    r4["cell"] = r4["size"].astype(str) + "|" + r4["composition"] + "|" + r4["family"]
    r4["send"] = r4["sent"]
    rows = []
    for fam, s in [("Sonnet+GPT", r4[r4["family"] != "Gemini"]), ("Sonnet", r4[r4["family"] == "Sonnet"]),
                   ("GPT", r4[r4["family"] == "GPT"]), ("Gemini", r4[r4["family"] == "Gemini"])]:
        a = s.dropna(subset=["own_rule_send_amount"])
        rec = {"family": fam, "n_investors": len(s), "n_with_amount": len(a),
               "share_with_amount": len(a) / len(s) if len(s) else np.nan,
               "mean_send": s["send"].mean(), "sd_send": s["send"].std(),
               "exact_match_share": float((np.isclose(a["send"], a["own_rule_send_amount"])).mean()) if len(a) else np.nan,
               "mean_abs_diff": float((a["send"] - a["own_rule_send_amount"]).abs().mean()) if len(a) else np.nan}
        if len(a) > 10 and a["send"].std() > 0 and a["run_id"].nunique() > 5:
            m = fit("send ~ own_rule_send_amount + C(cell)", a)
            t = "own_rule_send_amount"
            rec.update({"slope_per_dollar": m.params[t], "slope_lo": m.conf_int().loc[t, 0],
                        "slope_hi": m.conf_int().loc[t, 1], "slope_p": m.pvalues[t]})
        o = s.dropna(subset=["own_rule_send_ord"])
        if len(o) > 10 and o["send"].std() > 0:
            m = fit("send ~ own_rule_send_ord + C(cell)", o)
            t = "own_rule_send_ord"
            rec.update({"n_rule": len(o), "rule_ord_slope": m.params[t], "rule_ord_lo": m.conf_int().loc[t, 0],
                        "rule_ord_hi": m.conf_int().loc[t, 1], "rule_ord_p": m.pvalues[t]})
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT / "r4_amount_models.csv", index=False)
    print(pd.DataFrame(rows).round(3).T.to_string())

    # Held-out predictability at R4 (Sonnet + GPT investors; Gemini has no variance).
    s = r4[r4["family"] != "Gemini"].reset_index(drop=True)
    s["run_agent"] = s["run_id"] + "|" + s["agent"]
    y = s["send"].to_numpy(float) / 5
    cellX = pd.get_dummies(s["cell"], dtype=float).to_numpy()
    own_cols = [c for c in s.columns if c.startswith("own_") and c not in ("own_i",) and not c.startswith("own_chg_")]
    from sklearn.model_selection import GroupKFold
    folds = list(GroupKFold(5).split(s, groups=s["run_id"]))
    out = []

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

    rng = np.random.default_rng(0)
    p0 = cv(cellX)
    for name, cols in [("stated amount + send rule", ["own_rule_send_amount", "own_rule_send_ord", "own_rule_send_amount_given"]),
                       ("all own-myth features (no embeddings)", own_cols)]:
        p1 = cv(np.hstack([cellX, s[cols].to_numpy(float)]))
        res = S.boot_gain(y, p0, p1, s["run_id"].to_numpy())
        out.append({"model": name, "n": len(y), "n_runs": s["run_id"].nunique(), **res})
    pr = pd.DataFrame(out)
    pr.to_csv(OUT / "r4_predictability.csv", index=False)
    print(pr.round(4).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    cols = {"Sonnet": "#6a3d9a", "GPT": "#1b9e77", "Gemini": "#d95f02"}
    for fam, g in r4.dropna(subset=["own_rule_send_amount"]).groupby("family"):
        jit = rng.uniform(-0.12, 0.12, (2, len(g)))
        ax.scatter(g["own_rule_send_amount"] + jit[0], g["send"] + jit[1], s=22, alpha=0.7, color=cols[fam],
                   label=f"{fam} (n={len(g)})")
    ax.plot([0, 5], [0, 5], color="grey", lw=1, ls="--")
    ax.set_xlabel("amount the agent's own round-1 myth says to send ($)")
    ax.set_ylabel("amount it then sends in round 1 ($)")
    ax.set_title("Round 1, myth before game: agents send what their myth says", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "r4_amount_scatter.png", dpi=180)


if __name__ == "__main__":
    main()
