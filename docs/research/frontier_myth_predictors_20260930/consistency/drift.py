#!/usr/bin/env python3
"""Part 1 (port; LINGUISTIC_DATASET=september|frontier): is there a drift toward 'consistency' in the myths, and is it a description of play?

Outputs (this folder):
  drift_by_round.csv        mean (sd over runs) of each measure by setting x family x task_order x round
  drift_r1_vs_late.csv      round 1 vs rounds 8-10, with run-level paired Wilcoxon
  drift_describes_play.csv  does a myth's consistency track how steady the author's own play just was?
  drift_by_round.png
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from common import COLORS, DATA, FAMILIES, NAME, OUT, SETTINGS

MEAS = ["cons_judge", "cons_judge_ds", "cons_lex", "cons_lex_strict", "cons_emb"]


def load():
    m = pd.read_csv(OUT / "myth_features.csv")
    return m[m["valid"]]


def by_round(m):
    run = m.groupby(["setting", "family", "task_order", "round", "run_id"])[MEAS].mean().reset_index()
    g = run.groupby(["setting", "family", "task_order", "round"])
    out = g[MEAS].mean().add_suffix("_mean").join(g[MEAS].std().add_suffix("_sd"))
    out["n_runs"] = g.size()
    return out.reset_index().round(4)


def r1_vs_late(m):
    m = m.assign(phase=np.where(m["round"] == 1, "r1", np.where(m["round"] >= 8, "late", None))).dropna(subset=["phase"])
    run = m.groupby(["setting", "family", "task_order", "run_id", "phase"])[MEAS].mean().unstack("phase")
    rows = []
    for (st, fam, to), g in run.groupby(level=[0, 1, 2]):
        for k in MEAS:
            x = g[k].dropna()
            if len(x) < 3:
                continue
            d = x["late"] - x["r1"]
            p = stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0
            rows.append({"setting": st, "family": fam, "task_order": to, "measure": k,
                         "r1_mean": x["r1"].mean(), "r1_sd": x["r1"].std(),
                         "late_mean": x["late"].mean(), "late_sd": x["late"].std(),
                         "diff": d.mean(), "p_wilcoxon": p, "n_runs": len(x), "k_rose": int((d > 0).sum()),
                         "k_fell": int((d < 0).sum())})
    return pd.DataFrame(rows).round(4)


def rise_pooled(m):
    """Round 1 vs rounds 8-10 pooled over task orders (exact Wilcoxon on 5 runs cannot go below 0.0625):
    per (setting, family) and per family over all settings; run-level paired diffs, k of n runs rising."""
    m = m.assign(phase=np.where(m["round"] == 1, "r1", np.where(m["round"] >= 8, "late", None))).dropna(subset=["phase"])
    run = m.groupby(["setting", "family", "run_id", "phase"])[MEAS].mean().unstack("phase").reset_index()
    rows = []
    groups = [(st, fam, g) for (st, fam), g in run.groupby(["setting", "family"])]
    groups += [("all settings", fam, g) for fam, g in run.groupby("family")]
    for st, fam, g in groups:
        for k in MEAS:
            x = g[k].dropna()
            if len(x) < 3:
                continue
            d = x["late"] - x["r1"]
            p = stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0
            rows.append({"setting": st, "family": fam, "measure": k, "r1_mean": x["r1"].mean(), "r1_sd": x["r1"].std(),
                         "late_mean": x["late"].mean(), "late_sd": x["late"].std(), "diff": d.mean(),
                         "k_rose": int((d > 0).sum()), "k_fell": int((d < 0).sum()), "n_runs": len(x), "p_wilcoxon": p})
    t = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    t["p_holm_stratum"] = np.nan
    for _, idx in t.groupby(["setting", "family"]).groups.items():
        t.loc[idx, "p_holm_stratum"] = multipletests(t.loc[idx, "p_wilcoxon"], method="holm")[1]
    return t.round(4)


def describes_play(m):
    """Does the author's recent play stability predict the consistency of the myth written right after it?
    Agent-within-run FE + round FE; predictor = mean |change| of own coop over the last two same-role moves."""
    dec = pd.read_csv(DATA / "decisions.csv").sort_values(["run_id", "agent", "round"])
    dec["dcoop"] = (dec["coop"] - dec.groupby(["run_id", "agent", "role"])["coop"].shift(1)).abs()
    # instability over the last up-to-two moves ending at game round t
    dec["instab"] = dec.groupby(["run_id", "agent"])["dcoop"].transform(lambda s: s.rolling(2, min_periods=1).mean())
    dec["level"] = dec.groupby(["run_id", "agent"])["coop"].transform(lambda s: s.rolling(2, min_periods=1).mean())
    # a myth written after game round t: game_myth myth t; myth_game myth t+1
    dec["myth_round"] = np.where(dec["task_order"] == "game_myth", dec["round"], dec["round"] + 1)
    d = m.merge(dec[["run_id", "agent", "myth_round", "instab", "level"]].rename(columns={"myth_round": "round"}),
                on=["run_id", "agent", "round"], how="inner").dropna(subset=["instab", "level"])
    d["run_agent"] = d["run_id"] + "|" + d["agent"]
    d["prev_cons_lex"] = d.sort_values("round").groupby("run_agent")["cons_lex"].shift(1)
    from fe import fe_ols, rows_from
    rows = []
    for fam, g in list(d.groupby("family")) + [("all", d)]:
        for k in ["cons_lex", "cons_judge", "cons_emb"]:
            res = fe_ols(g, k, ["instab", "level"], absorb="run_agent", dummies=["round"])
            rows += rows_from(res, ["instab", "level"], family=fam, measure=k)
    return pd.DataFrame(rows).round(4)


def plot(t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams, cols = FAMILIES, COLORS
    meas = [("cons_lex", "keyword share (consistent/steady/reliab*)"), ("cons_judge", "GLM judge: praises consistency"),
            ("cons_emb", "embedding: consistency - generosity anchors")]
    settings = SETTINGS
    fig, ax = plt.subplots(len(meas), len(settings), figsize=(16, 10), sharex=True)
    for i, (k, lab) in enumerate(meas):
        for j, st in enumerate(settings):
            a = ax[i, j]
            for fam in fams:
                for to, ls in [("myth_game", "-"), ("game_myth", ":")]:
                    s = t[(t.setting == st) & (t.family == fam) & (t.task_order == to)]
                    if len(s):
                        a.plot(s["round"], s[k + "_mean"], ls, color=cols[fam], label=f"{fam} {to}")
            if i == 0:
                a.set_title(st)
            if j == 0:
                a.set_ylabel(lab, fontsize=8)
            if i == len(meas) - 1:
                a.set_xlabel("round")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"{NAME}: consistency language by round (solid myth->game, dotted game->myth; mean over runs)")
    fig.tight_layout()
    fig.savefig(OUT / "drift_by_round.png", dpi=150)


def main():
    m = load()
    t = by_round(m)
    t.to_csv(OUT / "drift_by_round.csv", index=False)
    r = r1_vs_late(m)
    r.to_csv(OUT / "drift_r1_vs_late.csv", index=False)
    rise_pooled(m).to_csv(OUT / "drift_rise_pooled.csv", index=False)
    dp = describes_play(m)
    dp.to_csv(OUT / "drift_describes_play.csv", index=False)
    plot(t)
    pd.set_option("display.width", 250)
    print(r[r.measure.isin(["cons_lex", "cons_judge", "cons_emb"])].to_string())
    print(dp.to_string())


if __name__ == "__main__":
    main()
