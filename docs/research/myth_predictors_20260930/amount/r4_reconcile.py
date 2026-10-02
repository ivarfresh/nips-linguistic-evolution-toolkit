"""R4 reconciliation with the search lens: round-1 myth->game send per $ stated in the own myth,
named amounts only vs named-else-band-midpoint (Sonnet + GPT, composition FE, SE clustered by run)."""
import pandas as pd, statsmodels.formula.api as smf
from pathlib import Path
OUT = Path(__file__).resolve().parent
P = pd.read_pickle(OUT / "panel.pkl"); M = pd.read_pickle(OUT / "myths_features.pkl")
r = P[(P.role == "investor") & (P.task_order == "myth_game") & (P["round"] == 1) & (P.family != "Gemini")]
r = r.merge(M[["run_id", "round", "agent", "send_amount"]], on=["run_id", "round", "agent"])
rows = []
for fam in ("all", "Sonnet", "GPT"):
    for lab, col in (("named amount only", "send_amount"), ("named amount, else send-rule band midpoint", "own_judge_amount")):
        d = (r if fam == "all" else r[r.family == fam]).dropna(subset=[col])
        f = smf.ols(f"sent ~ {col} + C(composition)", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.run_id)[0]})
        ci = f.conf_int().loc[col]
        rows.append(dict(family=fam, measure=lab, n=len(d), n_runs=d.run_id.nunique(), coef=f.params[col],
                         ci_low=ci[0], ci_high=ci[1], p=f.pvalues[col]))
out = pd.DataFrame(rows); out.to_csv(OUT / "r4_reconcile.csv", index=False); print(out.round(3).to_string())
