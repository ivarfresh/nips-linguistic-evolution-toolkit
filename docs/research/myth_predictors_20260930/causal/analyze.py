#!/usr/bin/env python3
"""R5 lens, step 2: which donor-content dimension orders the transplant ladder?

Unit of analysis = donor text (30 seeded texts, 5 per cell). Each text was run three times on three
apparatus (historical 8-agent Phase 3/5/6, Sept-16 8-agent rerun, Sept-17 dyad rerun), so donor-level
signal can be separated from run noise. Writes CSVs + one PNG next to this script.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
APP = ["historical_8", "rerun_8", "rerun_2"]
CELLS = ["s_end_minus", "s_filler", "s_end_plus_gpt", "s_start", "s_end_plus_gemini", "s_end_plus"]
LABEL = {"baseline": "no text", "s_end_minus": "late low-coop Sonnet", "s_filler": "Wikipedia filler",
         "s_end_plus_gpt": "late GPT", "s_start": "early Sonnet", "s_end_plus_gemini": "late Gemini",
         "s_end_plus": "late high-coop Sonnet"}
OUTCOMES = ["send_mean", "send_r1", "joint_frac", "return_share_told"]
RNG = np.random.default_rng(20260930)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(HERE / "runs.csv")
    don = pd.read_csv(HERE / "donors.csv")
    mor = pd.read_csv(HERE / "donor_moral_labels.csv")
    don = don.merge(mor[["seed_type", "rep", "label_z", "label_deepseek"]], on=["seed_type", "rep"], how="left")
    don["moral_generous_glm"] = (don["label_z"] == "be generous").astype(float)
    don["moral_generous_ds"] = (don["label_deepseek"] == "be generous").astype(float)
    don["moral_cautious_glm"] = (don["label_z"] == "be cautious").astype(float)
    don["rule_send_all"] = (don["send_rule"] == "all").astype(float)
    don["rule_consistency"] = don["consistency"].astype(float)
    don["rule_return_half_or_more"] = don["return_rule"].isin(["half", "more_than_half", "at_least_sent"]).astype(float)
    don["rule_letdown_reduce"] = don["after_letdown"].isin(["reduce", "withdraw"]).astype(float)
    don["rule_test_first"] = don["test_first"].astype(float)
    don["amount_endorsed"] = (don["amount_status"] == "endorsed").astype(float)
    don["late_source"] = (don["source_round"] > 1).astype(float)
    return runs, don


# ---------------------------------------------------------------- 0. reproduce the existing result
def reproduce(runs: pd.DataFrame, don: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for app in APP:
        d = runs[runs.apparatus == app].merge(don, on=["seed_type", "rep"])
        d = d[d["prescribed"].notna()]
        fit = smf.ols("send_mean ~ prescribed + C(seed_type)", d).fit()
        rows.append({"apparatus": app, "n": len(d), "coef_per_dollar": fit.params["prescribed"],
                     "ci_lo": fit.conf_int().loc["prescribed", 0], "ci_hi": fit.conf_int().loc["prescribed", 1],
                     "p": fit.pvalues["prescribed"]})
    return pd.DataFrame(rows).round(3)


# ---------------------------------------------------------------- 1. is there donor signal within cells?
def within_cell_replication(runs: pd.DataFrame) -> pd.DataFrame:
    """Cell-demean each apparatus's outcome, then ask whether a donor's within-cell rank replicates across
    apparatus. Permutation p: shuffle donor labels within cell independently per apparatus."""
    rows = []
    seeded = runs[runs.seed_type.isin(CELLS)].copy()
    for y in OUTCOMES:
        seeded[f"{y}_dm"] = seeded[y] - seeded.groupby(["apparatus", "seed_type"])[y].transform("mean")
        wide = seeded.pivot_table(index=["seed_type", "rep"], columns="apparatus", values=f"{y}_dm")
        for a, b in [("historical_8", "rerun_8"), ("historical_8", "rerun_2"), ("rerun_8", "rerun_2")]:
            ok = wide[[a, b]].dropna()
            var_cells = int((ok.groupby(level=0)[[a, b]].std() > 1e-9).all(axis=1).sum())
            rho = stats.spearmanr(ok[a], ok[b])[0] if ok[a].std() > 0 and ok[b].std() > 0 else np.nan
            # permutation within cell
            null = []
            for _ in range(5000):
                pb = ok[b].groupby(level=0).transform(lambda s: RNG.permutation(s.values))
                null.append(stats.spearmanr(ok[a], pb)[0])
            null = np.array(null)
            p = (np.sum(np.abs(null) >= abs(rho)) + 1) / (len(null) + 1) if np.isfinite(rho) else np.nan
            rows.append({"outcome": y, "pair": f"{a} vs {b}", "n_donors": len(ok), "cells_with_variance": var_cells,
                         "spearman_within_cell": rho, "perm_p": p})
        # ICC(1) across the three runs of each donor, on cell-demeaned z-scores
        z = seeded.copy()
        z["zz"] = z.groupby("apparatus")[f"{y}_dm"].transform(lambda s: s / s.std() if s.std() > 0 else s * 0)
        g = z.groupby(["seed_type", "rep"])["zz"]
        k = g.size().mean()
        msb = k * g.mean().var(ddof=1)
        msw = (z["zz"] - g.transform("mean")).pow(2).sum() / (len(z) - g.ngroups)
        icc = (msb - msw) / (msb + (k - 1) * msw)
        null = []
        for _ in range(2000):
            zp = z.copy()
            zp["zz"] = zp.groupby(["apparatus", "seed_type"])["zz"].transform(lambda s: RNG.permutation(s.values))
            gp = zp.groupby(["seed_type", "rep"])["zz"]
            mb = k * gp.mean().var(ddof=1)
            mw = (zp["zz"] - gp.transform("mean")).pow(2).sum() / (len(zp) - gp.ngroups)
            null.append((mb - mw) / (mb + (k - 1) * mw))
        rows.append({"outcome": y, "pair": "ICC(1) across 3 apparatus", "n_donors": g.ngroups,
                     "cells_with_variance": np.nan, "spearman_within_cell": icc,
                     "perm_p": (np.sum(np.array(null) >= icc) + 1) / (len(null) + 1)})
    return pd.DataFrame(rows).round(4)


def donor_scores(runs: pd.DataFrame) -> pd.DataFrame:
    """Per donor: raw outcome per apparatus + a combined score = mean over apparatus of the within-apparatus
    z-score (z computed across all 35 runs of that apparatus, so cell differences are kept)."""
    out = []
    for y in OUTCOMES:
        w = runs.pivot_table(index=["seed_type", "rep"], columns="apparatus", values=y)
        z = (w - w.mean()) / w.std()
        w.columns = [f"{y}__{c}" for c in w.columns]
        w[f"{y}__combined_z"] = z.mean(axis=1)
        out.append(w)
    return pd.concat(out, axis=1).reset_index()


# ---------------------------------------------------------------- 2. feature horse race
FEATURES = {
    # judge-extracted rule (GLM-5.2)
    "prescribed": "stated send ($; named amount, else rule band)",
    "prescribed_endorsed": "stated send, narrated amounts replaced by band",
    "rule_send_all": "send rule = all",
    "rule_consistency": "consistency norm (judge)",
    "rule_return_half_or_more": "return rule >= half",
    "rule_letdown_reduce": "after letdown: reduce/withdraw",
    "rule_test_first": "test small first",
    "amount_endorsed": "named amount endorsed (not narrated)",
    # moral label (Arabella rubric)
    "moral_generous_glm": "moral = be generous (GLM)",
    "moral_generous_ds": "moral = be generous (DeepSeek)",
    # lexicon (judge-free)
    "coop_pct": "cooperative words %",
    "uncoop_pct": "uncooperative words %",
    "coop_minus_uncoop": "coop minus uncoop words %",
    "kw_consistency": "consistency keywords (count)",
    "kw_measured": "measured/prudent keywords (count)",
    "num_per100": "number words per 100",
    "digit_per100": "digits per 100",
    "five_mentions": "mentions of 5 / all / everything",
    "game_terms_per100": "game terms per 100 (send/return/triple)",
    "n_words": "length (words)",
    # provenance
    "joint_at_source": "joint resources of the source run",
    "late_source": "written in round 10 (vs round 1)",
}


def spearman(x, y):
    ok = pd.notna(x) & pd.notna(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, int(len(x))
    r, p = stats.spearmanr(x, y)
    return r, p, int(len(x))


def horse_race(don: pd.DataFrame, sc: pd.DataFrame, outcome: str) -> pd.DataFrame:
    d = don[don.seed_type.isin(CELLS)].merge(sc, on=["seed_type", "rep"])
    y = d[outcome]
    rows = []
    for f, lab in FEATURES.items():
        x = d[f].astype(float)
        # cell level (6 seeded cells; NaN-skipping cell means)
        cm = d.groupby("seed_type")[[f, outcome]].mean()
        r_cell, p_cell, n_cell = spearman(cm[f], cm[outcome])
        # pooled donor level
        r_pool, p_pool, n_pool = spearman(x, y)
        # within cell: demean feature and outcome by cell, only cells where the feature varies
        vary = d.groupby("seed_type")[f].transform(lambda s: s.dropna().nunique() > 1)
        dd = d[vary & x.notna()]
        xd = dd[f] - dd.groupby("seed_type")[f].transform("mean")
        yd = dd[outcome] - dd.groupby("seed_type")[outcome].transform("mean")
        r_w, p_w, n_w = spearman(xd, yd)
        cells_var = int(d.loc[vary, "seed_type"].nunique())
        # permutation p for within (shuffle feature within cell)
        if np.isfinite(r_w):
            null = []
            xv, gv = dd[f].to_numpy(float), dd["seed_type"].to_numpy()
            idx = [np.where(gv == c)[0] for c in np.unique(gv)]
            xdm = xd.to_numpy(float)
            for _ in range(1000):
                xp = xdm.copy()
                for ix in idx:
                    xp[ix] = xdm[RNG.permutation(ix)]
                null.append(stats.spearmanr(xp, yd)[0])
            p_wperm = (np.sum(np.abs(np.array(null)) >= abs(r_w)) + 1) / (len(null) + 1)
        else:
            p_wperm = np.nan
        # leave-one-out: drop the "send nothing" donor; drop each cell in turn (pooled rho)
        nz = ~((d.seed_type == "s_end_minus") & (d.rep == 1))
        r_no_zero = spearman(x[nz], y[nz])[0]
        loo = [spearman(x[d.seed_type != c], y[d.seed_type != c])[0] for c in CELLS]
        dd_nz = dd[~((dd.seed_type == "s_end_minus") & (dd.rep == 1))]
        r_w_nz = spearman(dd_nz[f] - dd_nz.groupby("seed_type")[f].transform("mean"),
                          dd_nz[outcome] - dd_nz.groupby("seed_type")[outcome].transform("mean"))[0]
        rows.append({"feature": f, "description": lab, "outcome": outcome,
                     "cell_rho": r_cell, "cell_p": p_cell, "cell_n": n_cell,
                     "pooled_rho": r_pool, "pooled_p": p_pool, "pooled_n": n_pool,
                     "pooled_rho_drop_send_nothing": r_no_zero,
                     "pooled_rho_leave_cell_out_min": np.nanmin(loo), "pooled_rho_leave_cell_out_max": np.nanmax(loo),
                     "within_rho": r_w, "within_perm_p": p_wperm, "within_n": n_w, "within_cells_with_variance": cells_var,
                     "within_rho_drop_send_nothing": r_w_nz})
    return pd.DataFrame(rows)


def holm(p: pd.Series) -> pd.Series:
    ok = p.dropna().sort_values()
    m = len(ok)
    adj = np.minimum(1, np.maximum.accumulate([(m - i) * v for i, v in enumerate(ok.values)]))
    return pd.Series(adj, index=ok.index).reindex(p.index)


# ---------------------------------------------------------------- 3. cell table + contrasts
def cell_table(runs: pd.DataFrame, don: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["apparatus", "seed_type"])
    t = g[["send_mean", "send_r1", "joint_frac", "return_share_told", "r1_cite_myth"]].agg(["mean", "std"]).round(3)
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    t = t.reset_index()
    f = don.groupby("seed_type").agg(prescribed_mean=("prescribed", "mean"),
                                     n_named_amount=("send_amount", lambda s: int(s.notna().sum())),
                                     n_endorsed=("amount_status", lambda s: int((s == "endorsed").sum())),
                                     n_consistency=("rule_consistency", "sum"),
                                     n_generous_glm=("moral_generous_glm", "sum"),
                                     coop_pct=("coop_pct", "mean"), kw_consistency=("kw_consistency", "mean"),
                                     joint_at_source_mean=("joint_at_source", "mean"),
                                     joint_at_source_sd=("joint_at_source", "std")).round(2).reset_index()
    return t.merge(f, on="seed_type", how="left")


def plot(don: pd.DataFrame, sc: pd.DataFrame) -> None:
    d = don[don.seed_type.isin(CELLS)].merge(sc, on=["seed_type", "rep"])
    cols = {"s_end_minus": "#b2182b", "s_filler": "#999999", "s_end_plus_gpt": "#d95f02", "s_start": "#7570b3",
            "s_end_plus_gemini": "#1b9e77", "s_end_plus": "#3f007d"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, app in zip(axes, APP):
        for c in CELLS:
            x = d[d.seed_type == c]
            xs = x["prescribed"].fillna(-0.7) + RNG.uniform(-0.12, 0.12, len(x))
            ax.scatter(xs, x[f"send_mean__{app}"], color=cols[c], s=46, edgecolor="white", label=LABEL[c])
        ax.set_xticks([-0.7, 0, 2, 3, 4.25, 5])
        ax.set_xticklabels(["none", "0", "2", "3", "most", "5"])
        ax.set_xlabel("send the donor text states ($)")
        ax.set_title(app.replace("_", " ") + (" agents" if app != "rerun_2" else " (dyad)"))
    axes[0].set_ylabel("host mean send, rounds 1-10 ($)")
    axes[-1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(HERE / "donor_amount_vs_host_send.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    runs, don = load()
    rep = reproduce(runs, don)
    rep.to_csv(HERE / "reproduce_myth_rules.csv", index=False)
    print("reproduce (myth_rules_20260928 reports +0.41 8-agent, +0.34 dyad):\n", rep.to_string())

    wf = HERE / "within_cell_replication.csv"
    wr = pd.read_csv(wf) if wf.exists() else within_cell_replication(runs)
    wr.to_csv(wf, index=False)
    print("\nwithin-cell replication across apparatus:\n", wr.to_string())

    sc = donor_scores(runs)
    sc.to_csv(HERE / "donor_scores.csv", index=False)
    ct = cell_table(runs, don)
    ct.to_csv(HERE / "cell_table.csv", index=False)

    races = []
    for outcome in ["send_mean__combined_z", "send_r1__combined_z", "send_mean__rerun_2", "send_mean__rerun_8",
                    "send_mean__historical_8", "return_share_told__combined_z"]:
        races.append(horse_race(don, sc, outcome))
    hr = pd.concat(races, ignore_index=True)
    for col in ["cell_p", "pooled_p", "within_perm_p"]:
        hr[f"{col}_holm"] = holm(hr[col])
    hr = hr.round(4)
    hr.to_csv(HERE / "feature_horse_race.csv", index=False)
    n_tests = int(hr[["cell_p", "pooled_p", "within_perm_p"]].notna().sum().sum())
    print(f"\n{n_tests} tests in the horse race (Holm applied separately per test family of "
          f"{hr['cell_p'].notna().sum()}/{hr['pooled_p'].notna().sum()}/{hr['within_perm_p'].notna().sum()})")
    prim = hr[hr.outcome == "send_mean__combined_z"].sort_values("pooled_rho", ascending=False)
    print(prim[["feature", "cell_rho", "cell_p", "pooled_rho", "pooled_p", "pooled_rho_drop_send_nothing",
                "pooled_rho_leave_cell_out_min", "pooled_rho_leave_cell_out_max", "within_rho", "within_perm_p",
                "within_n", "within_cells_with_variance"]].to_string())

    # donor-level table for the contrast cells
    keep = ["seed_type", "rep", "writer", "source_round", "joint_at_source", "send_rule", "send_amount",
            "amount_status", "prescribed", "return_rule", "after_letdown", "consistency", "label_z", "label_deepseek",
            "coop_pct", "kw_consistency", "five_mentions", "n_words"]
    dt = don[keep].merge(sc[["seed_type", "rep", "send_mean__historical_8", "send_mean__rerun_8", "send_mean__rerun_2",
                             "send_r1__rerun_2", "send_mean__combined_z"]], on=["seed_type", "rep"])
    dt.round(3).to_csv(HERE / "donor_table.csv", index=False)
    print(dt[dt.seed_type.isin(["s_end_plus_gpt", "s_end_plus_gemini", "s_end_plus", "s_start"])]
          .drop(columns=["writer"]).round(2).to_string())
    plot(don, sc)


if __name__ == "__main__":
    main()
