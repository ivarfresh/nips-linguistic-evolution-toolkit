import numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
D=Path('/private/tmp/claude-502/-Users-ivar-Desktop-Research-AI-projects-LLM-evolution-nips-linguistic-evolution-toolkit/ec36f89d-6a64-494f-a76c-34534ff1d678/scratchpad/predictors/amount')
P=pd.read_pickle(D/'panel.pkl'); P["runround"]=P.run_id+"|"+P["round"].astype(str)
M=pd.read_pickle(D/'myths_features.pkl').set_index(["run_id","round","agent"])["judge_amount"]
def demean(A,g):
    c=pd.factorize(g)[0]; s=np.zeros((c.max()+1,A.shape[1])); np.add.at(s,c,A); return A-(s/np.bincount(c)[:,None])[c]
def fit(df,y,xs,fe):
    df=df.dropna(subset=[y]+xs); A=demean(df[[y]+xs].to_numpy(float),df[fe].to_numpy())
    r=sm.OLS(A[:,0],A[:,1:]).fit(cov_type="cluster",cov_kwds={"groups":pd.factorize(df.run_id)[0]})
    return {x:(round(r.params[i],3),round(r.conf_int()[i][0],3),round(r.conf_int()[i][1],3),round(r.pvalues[i],4)) for i,x in enumerate(xs[:3])}, len(df)
base=P[(P["size"]==8)&(P.task_order=="myth_game")&P.shown_judge_amount.notna()].copy()
base["future_amt"]=[M.get((r.run_id,r.shown_round+1,r.shown_author),np.nan) for r in base.itertuples()]
mt=base.drop_duplicates(["run_id","agent","own_round"]).copy()
print("own_round==shown_round+1 share:",(mt.own_round==mt.shown_round+1).mean())
# random other agent's same-round myth (not reader, not shown author)
rng=np.random.default_rng(0)
agents=M.reset_index()
bygrp=agents.groupby(["run_id","round"])
def rand_other(r):
    try: g=bygrp.get_group((r.run_id,r.shown_round+1))
    except KeyError: return np.nan
    g=g[(g.agent!=r.agent)&(g.agent!=r.shown_author)]
    return g.judge_amount.sample(1,random_state=int(rng.integers(1e9))).iloc[0] if len(g) else np.nan
mt["other_amt"]=[rand_other(r) for r in mt.itertuples()]
ctrl=["unseen_judge_amount","own_prev_judge_amount","lag_coop_any","b_coop_prev"]
print("orig (runround FE):",fit(mt,"own_judge_amount",["shown_judge_amount","future_amt"]+ctrl,"runround"))
print("no future:",fit(mt,"own_judge_amount",["shown_judge_amount","unseen_judge_amount"]+ctrl[1:],"runround"))
print("random-other placebo instead of future:",fit(mt,"own_judge_amount",["shown_judge_amount","other_amt"]+ctrl,"runround"))
print("orig with run FE only:",fit(mt,"own_judge_amount",["shown_judge_amount","future_amt"]+ctrl,"run_id"))
print("corr shown,future:",mt[["shown_judge_amount","future_amt"]].corr().iloc[0,1].round(3))
