"""Future-myth placebo (lead's request): the shown author's NEXT myth (round r, not visible to the reader
when it decides or writes) as a placebo next to the shown myth; plus the shown author != current partner subset."""
import numpy as np, pandas as pd, re
from pathlib import Path
import statsmodels.api as sm
OUT = Path(__file__).resolve().parent
P = pd.read_pickle(OUT / "panel.pkl"); P["send"] = P["sent"]
P["runround"] = P["run_id"] + "|" + P["round"].astype(str)
M = pd.read_pickle(OUT / "myths_features.pkl").set_index(["run_id", "round", "agent"])["judge_amount"]
rows = []

def demean(A, g):
    c = pd.factorize(g)[0]; s = np.zeros((c.max() + 1, A.shape[1])); np.add.at(s, c, A)
    return A - (s / np.bincount(c)[:, None])[c]

def fit(df, y, xs, term, label, fam):
    df = df.dropna(subset=[y] + xs)
    if df[y].std() < 1e-9 or len(df) < 30:
        return
    A = demean(df[[y] + xs].to_numpy(float), df["runround"].to_numpy())
    r = sm.OLS(A[:, 0], A[:, 1:]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["run_id"])[0]})
    j = xs.index(term); ci = r.conf_int()[j]
    rows.append(dict(test=label, family=fam, term=term, n=len(df), n_runs=df.run_id.nunique(), coef=r.params[j],
                     ci_low=ci[0], ci_high=ci[1], p=r.pvalues[j]))

base = P[(P["size"] == 8) & (P.task_order == "myth_game") & P.shown_judge_amount.notna()].copy()
# the shown author's next myth: written in round shown_round+1 (= decision round), same time as the reader's; never shown to it
base["future_amt"] = [M.get((r.run_id, r.shown_round + 1, r.shown_author), np.nan) for r in base.itertuples()]
inv = base[base.role == "investor"].copy()
mt = base.drop_duplicates(["run_id", "agent", "own_round"]).copy()
for fam in ("all", "Sonnet", "GPT"):
    i = inv if fam == "all" else inv[inv.family == fam]
    m = mt if fam == "all" else mt[mt.family == fam]
    xs = ["shown_judge_amount", "future_amt", "unseen_judge_amount", "lag_sent", "own_prev_judge_amount", "b_coop_prev", "c_coop_prev"]
    for t in ("shown_judge_amount", "future_amt"):
        fit(i, "send", xs, t, "-> next send (future-myth placebo in model)", fam)
        fit(i[i.shown_author != i.partner], "send", xs, t, "-> next send, shown author != current partner", fam)
    xm = ["shown_judge_amount", "future_amt", "unseen_judge_amount", "own_prev_judge_amount", "lag_coop_any", "b_coop_prev"]
    for t in ("shown_judge_amount", "future_amt"):
        fit(m, "own_judge_amount", xm, t, "-> reader's next MYTH amount (future-myth placebo in model)", fam)
out = pd.DataFrame(rows); out.to_csv(OUT / "future_placebo.csv", index=False)
print("shown author == current partner share:", (inv.shown_author == inv.partner).mean().round(3),
      "| shown author = last round's partner:", inv.b_played_prev_round.mean())
print(out.round(3).to_string())
