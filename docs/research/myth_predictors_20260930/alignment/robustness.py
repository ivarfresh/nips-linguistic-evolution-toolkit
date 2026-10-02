"""Follow-up robustness for the norm-alignment lens (advisor review):
 a) misalignment split with 0-10 categorical own-score controls (identified),
 b) DeepSeek same_label with DeepSeek level controls,
 c) R4 8-agent with run x pair-family FE,
 d) primary 30 tests with missing lags filled (0 + indicator),
 e) Holm within each secondary family,
 f) summary numbers table."""
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
import norm_alignment_fast as F

g = pd.read_csv("norm_alignment_games_v2.csv")
ds = pd.read_csv(F.JUDGES).set_index(["run_id", "round", "agent"])["label_ds"]
rb = np.where(g["task_order"] == "myth_game", g["round"], g["round"] - 1)
g["inv_label_ds"] = [ds.get(k, np.nan) for k in zip(g["run_id"], rb, g["investor"])]
g["tru_label_ds"] = [ds.get(k, np.nan) for k in zip(g["run_id"], rb, g["trustee"])]
for w in ["inv", "tru"]:
    g[f"{w}_give_c"] = g[f"{w}_give"].round().astype("Int64").astype(str)
    g[f"{w}_lag_any_missing"] = g[f"{w}_lag_any"].isna().astype(float)
    g[f"{w}_lag_any_f"] = g[f"{w}_lag_any"].fillna(0)
gb = g[g["has_myths_before"]]
rows = []

# a) split
for size in [2, 8]:
    d = gb[gb["size"] == size].copy()
    d["inv_above"] = (d["inv_give"] - d["tru_give"]).clip(lower=0)
    d["tru_above"] = (d["tru_give"] - d["inv_give"]).clip(lower=0)
    for y in F.OUTCOMES:
        f = F.feols(d, y, ["inv_above", "tru_above"] + F.LAGS, F.CATS + ["inv_give_c", "tru_give_c"], F.fe_for(size, "cell"))
        for t in ["inv_above", "tru_above"]:
            rows.append({"check": "a) misalignment split, categorical own scores (per point of send score)", "stratum": f"{size}-agent",
                         "outcome": y, "term": t, **f["terms"][t], "n_games": f["n_games"]})

# b) DeepSeek with DeepSeek levels
cats_ds = ["inv_label_ds", "tru_label_ds", "inv_send_rule", "tru_send_rule"]
for size in [2, 8]:
    d = gb[gb["size"] == size].copy()
    d["zx"] = F.z(d["same_label_ds"])
    for y in F.OUTCOMES:
        f = F.feols(d, y, ["zx"] + F.LAGS + F.NUMS, cats_ds, F.fe_for(size, "cell"))
        rows.append({"check": "b) same_label by DeepSeek, DeepSeek level controls", "stratum": f"{size}-agent",
                     "outcome": y, "term": "same_label_ds", **f["terms"]["zx"], "n_games": f["n_games"]})

# c) R4 8-agent with run x pair-family
d0 = g[(g["task_order"] == "myth_game") & (g["round"] == 1) & g["has_myths_before"] & (g["size"] == 8)]
for m in F.MEASURES:
    for y in F.OUTCOMES:
        e = d0.dropna(subset=[m]).copy(); e["zx"] = F.z(e[m])
        f = F.feols(e, y, ["zx"] + F.NUMS, [], ["cell"])
        if f is None or "zx" not in f["terms"]:
            continue
        rows.append({"check": "c) R4 8-agent round 1, run x pair-family FE + own send scores", "stratum": "8-agent",
                     "measure": m, "outcome": y, "term": "zx", **f["terms"]["zx"], "n_games": f["n_games"]})

# d) primary with filled lags
lags_f = ["inv_lag_any_f", "inv_lag_any_missing", "tru_lag_any_f", "tru_lag_any_missing",
          "inv_lag_same_f", "inv_lag_same_missing", "tru_lag_same_f", "tru_lag_same_missing"]
ptab = []
for size in [2, 8]:
    d = gb[gb["size"] == size]
    for y in F.OUTCOMES:
        for m in F.MEASURES:
            e = d.dropna(subset=[m]).copy(); e["zx"] = F.z(e[m])
            f = F.feols(e, y, ["zx"] + lags_f + F.NUMS, F.CATS, F.fe_for(size, "cell"))
            ptab.append({"check": "d) primary with missing lags filled (0 + indicator)", "stratum": f"{size}-agent",
                         "measure": m, "outcome": y, "term": "zx", **f["terms"]["zx"], "n_games": f["n_games"]})
ptab = pd.DataFrame(ptab); ptab["p_holm"] = multipletests(ptab["p"], method="holm")[1]
out = pd.concat([pd.DataFrame(rows), ptab])
out.to_csv("robustness.csv", index=False)

# e) Holm within secondary families
bg = pd.read_csv("before_game_models.csv")
sec = (~bg["primary"]) & ~bg["spec"].str.contains("coef of") & bg["p"].notna()
bg.loc[sec, "p_holm_secondary"] = multipletests(bg.loc[sec, "p"], method="holm")[1]
bg.to_csv("before_game_models.csv", index=False)
for fn in ["interaction_models.csv", "founding_window_R4.csv", "reverse_models.csv"]:
    t = pd.read_csv(fn); ok = t["p"].notna()
    t.loc[ok, "p_holm_family"] = multipletests(t.loc[ok, "p"], method="holm")[1]
    t.to_csv(fn, index=False)

# f) summary numbers
S = []
per_run = gb.groupby(["size", "run_id"])[F.MEASURES + ["sent_frac", "return_proportion"]].mean()
for size, pr in per_run.groupby(level=0):
    for c in pr.columns:
        S.append({"item": f"per-run mean {c}", "size": size, "mean": pr[c].mean(), "sd_over_runs": pr[c].std(), "n_runs": len(pr)})
for size, d in gb.groupby("size"):
    S.append({"item": "share of games with identical send scores (give_align=1)", "size": size, "mean": (d["give_align"] == 1).mean(), "n_games": d["give_align"].notna().sum()})
    S.append({"item": "corr(giving_gap, sent_frac)", "size": size, "mean": d[["giving_gap", "sent_frac"]].corr().iloc[0, 1]})
    S.append({"item": "share of 8-agent games that are first meetings", "size": size, "mean": (d["prior_meetings"] == 0).mean()})
S.append({"item": "primary tests (Holm family)", "mean": int(bg["primary"].sum())})
S.append({"item": "secondary before-game tests", "mean": int(sec.sum())})
st = bg[(~bg["primary"]) & (bg["fe"] == "cell") & (bg["spec"] == "alone + levels + lags") & ~bg["stratum"].isin(["2-agent", "8-agent"])]
S.append({"item": "stratum tests (setting/order/meeting/family) with p<0.05", "mean": int((st["p"] < 0.05).sum()), "n_games": len(st)})
for fn in ["interaction_models.csv", "founding_window_R4.csv", "reverse_models.csv"]:
    t = pd.read_csv(fn)
    S.append({"item": f"{fn}: tests / p<0.05 / Holm<0.05", "mean": len(t), "sd_over_runs": int((t["p"] < 0.05).sum()), "n_runs": int((t["p_holm_family"] < 0.05).sum())})
S.append({"item": "secondary before-game tests Holm<0.05", "mean": int((bg["p_holm_secondary"] < 0.05).sum())})
pd.DataFrame(S).to_csv("summary_numbers.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(out.drop(columns=["se"]).round(4).to_string(index=False))
print(pd.DataFrame(S).round(3).to_string(index=False))
print(bg[bg["p_holm_secondary"] < 0.05][["stratum", "measure", "outcome", "spec", "fe", "coef", "p", "p_holm_secondary"]].to_string())
rv = pd.read_csv("reverse_models.csv"); print(rv[rv["p_holm_family"] < 0.05].round(4).to_string())
