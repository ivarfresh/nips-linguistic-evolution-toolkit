"""Round-1 myth_game sends by what the agent's own round-1 myth names (mean (±sd) over runs)."""
import numpy as np, pandas as pd
from pathlib import Path
OUT = Path(__file__).resolve().parent
P = pd.read_pickle(OUT / "panel.pkl")
r = P[(P.role == "investor") & (P.task_order == "myth_game") & (P["round"] == 1) & P.family.isin(["Sonnet", "GPT"])].copy()
r["myth_names"] = np.where(r.own_judge_amount >= 4.9, "names all 5", np.where(r.own_judge_amount.notna(), "names less than 5", "nothing"))
per_run = r.groupby(["family", "myth_names", "run_id"])["sent"].mean().reset_index()
out = per_run.groupby(["family", "myth_names"])["sent"].agg(mean="mean", sd="std", n_runs="count").reset_index()
out["n_decisions"] = r.groupby(["family", "myth_names"]).size().values
out.round(2).to_csv(OUT / "round1_send_by_own_myth.csv", index=False)
print(out.round(2).to_string())
