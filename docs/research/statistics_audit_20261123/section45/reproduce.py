#!/usr/bin/env python3
"""Offline reruns for paper Section 4.5. Outputs stay in this audit directory.

Usage: python reproduce.py word|moral|regression|map|amount|descriptive
Original scripts are imported; their output roots are redirected before calling.
No simulation, judge, embedding generation, or paid API is invoked.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ['LINGUISTIC_DATASET'] = 'september_n10'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/statistics_audit_20261123_mpl')
import numpy as np
import pandas as pd
from scipy import stats


def local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def word():
    from analyses import word_adoption_placebo as wp
    from analyses import linguistic_uptake as lu
    wp.OUT = OUT / 'word_placebo'
    wp.PAIRS = wp.OUT
    # Third audit specialist handles full final-file provenance. Do not rewrite it here.
    wp.linguistic_provenance.main = lambda out: None
    wp.main()
    c = pd.read_csv(lu.DATA / 'uptake_children.csv')
    lu.run_level_summary(c, ['size', 'exposure']).to_csv(OUT/'word_shown_unseen.csv', index=False)
    lu.run_level_summary(c, ['size', 'exposure', 'task_order']).to_csv(OUT/'word_by_order.csv', index=False)
    # Direct independent aggregation and scipy signed-rank, with explicit sample counts.
    rows = []
    for k, g in c.groupby(['size','exposure']):
        p = g.groupby('run_id')[['adopt_parent','adopt_null']].mean()
        d = p.adopt_parent-p.adopt_null
        rows.append(dict(size=k[0],exposure=k[1],n_pairs=len(g),n_runs=len(p),
            mean_shown=p.adopt_parent.mean(),sd_shown=p.adopt_parent.std(),
            mean_unseen=p.adopt_null.mean(),sd_unseen=p.adopt_null.std(),
            mean_difference=d.mean(),sd_difference=d.std(),
            ratio=p.adopt_parent.mean()/p.adopt_null.mean(),positive_runs=int((d>0).sum()),
            wilcoxon_statistic=stats.wilcoxon(d).statistic,p_raw=stats.wilcoxon(d).pvalue))
    pd.DataFrame(rows).to_csv(OUT/'word_independent.csv',index=False)


def moral():
    from analyses import moral_placebo as mp
    mp.OUT = OUT / 'moral_placebo'
    mp.PAIRS = mp.OUT
    mp.linguistic_provenance.main = lambda out: None
    mp.main()


def markers():
    from analyses import linguistic_uptake as lu
    myths=lu.load_myths()
    X,vocab=lu.marker_matrix(myths)
    H=lu.history_matrix(myths,X)
    spec,words,rates=lu.marker_tests(myths,X,vocab,H,lu.null_candidates(myths),1000,np.random.default_rng(20260923))
    spec.to_csv(OUT/'marker_specific_uptake.csv',index=False)
    words.to_csv(OUT/'marker_words.csv',index=False)
    # Original script consumes RNG only in marker_tests before generating this table.
    print(spec.to_string(index=False))


def regression():
    import statsmodels.formula.api as smf
    from analyses import moral_carryover as mc
    from analyses import plot_moral_send_return_simple as plot
    rows, metadata = [], []
    for ds in ['september_n10','frontier']:
        mc.DATA = ROOT / ('data/analysis/linguistic_n10_20261001' if ds=='september_n10' else 'data/analysis/linguistic_frontier_20260930')
        # load_myths function belongs to linguistic_uptake, so redirect that module as well.
        from analyses import linguistic_uptake as lu
        lu.DATA = mc.DATA
        myths, dec = mc.load('moral_labels_z-ai__glm-5.2.csv')
        d = mc.decision_table(myths,dec)
        d['run_agent'] = d.run_id + '|' + d.agent
        for role in ['investor','trustee']:
            cols=['coop','own_label','shown_label','coop_lag_same_role','run_id','run_agent','round','partner_family']
            sub=d[d.role==role][cols].dropna().copy()
            # pandas3 extension strings are unsupported by this installed patsy; object conversion
            # changes storage only, not values/design. This is an audit-only compatibility step.
            for c in sub.select_dtypes(include=['str','string']).columns:
                sub[c] = sub[c].astype(object)
            formula="coop ~ C(own_label, Treatment('be fair')) + C(shown_label, Treatment('be fair')) + coop_lag_same_role + C(run_agent) + C(round)"
            fit=smf.ols(formula,sub).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(sub.run_id)[0]})
            for term in fit.params.index:
                if 'label' not in term and term!='coop_lag_same_role':
                    continue
                ci=fit.conf_int().loc[term]
                rows.append(dict(dataset=ds,role=role,term=term,coef=fit.params[term],se=fit.bse[term],ci_low=ci.iloc[0],ci_high=ci.iloc[1],p=fit.pvalues[term],n_rows=len(sub),n_runs=sub.run_id.nunique(),n_agents=sub.run_agent.nunique(),df_resid=fit.df_resid,use_t=fit.use_t))
            metadata.append(dict(dataset=ds,role=role,formula=formula,n_rows=len(sub),n_runs=sub.run_id.nunique(),n_agents=sub.run_agent.nunique(),n_params=len(fit.params),rank=int(fit.model.rank)))
        ceiling = 'Gemini' if ds=='september_n10' else 'GeminiPro'
        plotted=d[d.family!=ceiling].dropna(subset=['shown_label','coop']).copy()
        plotted['coop_vs_own_avg']=plotted.coop-plotted.groupby(['run_id','agent','role']).coop.transform('mean')
        pd.concat([plot.per_round(plotted,v) for v in ['coop','coop_vs_own_avg']],ignore_index=True).to_csv(OUT/f'figure6_{ds}.csv',index=False)
        grouped=plotted.groupby(['role','shown_label']).coop.agg(['mean','std','count']).reset_index()
        grouped.to_csv(OUT/f'figure6_overall_{ds}.csv',index=False)
        metadata.append(dict(dataset=ds,plotted_runs=plotted.run_id.nunique(),plotted_decisions=len(plotted)))
    pd.DataFrame(rows).to_csv(OUT/'moral_regression.csv',index=False)
    (OUT/'moral_regression_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


def map_analysis():
    from analyses import linguistic_datasets as ld
    from analyses import myth_map_significance as mm
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.decomposition import PCA
    ds=ld.get('september_n10')
    myths,emb=mm.load(ds)
    myths['group']=[mm.group_of(r,ds.families) for r in myths.itertuples()]
    rows=[]
    for order,g in myths[myths['round']==1].groupby('task_order'):
        pred=cross_val_predict(LogisticRegression(max_iter=2000),emb[g.index],g.family,groups=g.run_id,cv=GroupKFold(5))
        ncorrect=int((pred==g.family).sum())
        rows.append(dict(task_order=order,n_myths=len(g),n_runs=g.run_id.nunique(),n_correct=ncorrect,accuracy=ncorrect/len(g)))
    pd.DataFrame(rows).to_csv(OUT/'classifier.csv',index=False)
    rng=np.random.default_rng(mm.SEED)
    curves,boots=mm.distance_curves(myths,emb,ds,rng)
    curves.to_csv(OUT/'distance_curves.csv',index=False)
    out=[]
    for (order,size,group,pair),draws in boots.items():
        g=curves[(curves.task_order==order)&(curves['size']==size)&(curves.group==group)&(curves.pair==pair)]
        a=g[g['round']==1].distance.iloc[0]; b=g[g['round']==10].distance.iloc[0]
        ci=np.nanpercentile(draws[:,-1]-draws[:,0],[2.5,97.5])
        out.append(dict(task_order=order,size=size,group=group,pair=pair,round1=a,round10=b,delta=b-a,ci_low=ci[0],ci_high=ci[1],n_runs=g.n_runs.iloc[0]))
    pd.DataFrame(out).to_csv(OUT/'distance_change.csv',index=False)
    # Reproduce all supplement inference from embedding arrays too.
    mm.tests(myths,emb,curves,boots,ds,rng).to_csv(OUT/'map_significance.csv',index=False)
    xdir,ydir=mm.chosen_axes(myths,emb,ds)
    pca=PCA(2,svd_solver='full').fit(emb)
    pd.DataFrame([dict(view='PCA',total_variance=float(pca.explained_variance_ratio_.sum()),**mm.share_kept(myths,emb,list(pca.components_))),
                  dict(view='Chosen axes',**mm.share_kept(myths,emb,[xdir,ydir]))]).to_csv(OUT/'embedding_variance.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(out).to_string(index=False))


def amount():
    import statsmodels.formula.api as smf
    common=local_module('plan_common_audit',ROOT/'docs/research/frontier_myth_predictors_20260930/plan/common.py')
    m,p=common.load_panel('september')
    base="send ~ shown_judge_amount + unseen_judge_amount + lag_sent + own_prev_judge_amount + b_coop_prev + c_coop_prev + C(runround)"
    myth="own_judge_amount ~ shown_judge_amount + unseen_judge_amount + own_prev_judge_amount + lag_coop_any + b_coop_prev + C(runround)"
    r3=p[(p['size']==8)&(p.task_order=='myth_game')&p.shown_judge_amount.notna()]
    inv=r3[(r3.role=='investor')&(r3.shown_author!=r3.partner)]
    mt=r3.drop_duplicates(['run_id','agent','own_round'])
    rows=[dict(check='read amount to next send',**common.fit(inv,base,'shown_judge_amount')),
          dict(check='read amount to own next amount',**common.fit(mt,myth,'shown_judge_amount'))]
    for name,g,formula in [('read amount to next send',inv,base),('read amount to own next amount',mt,myth)]:
        lhs,rhs=formula.split('~')
        cols=[lhs.strip()]+[c.strip() for c in rhs.split('+') if not c.strip().startswith('C(')]+['runround','run_id']
        sub=g.dropna(subset=cols).copy()
        for c in sub.select_dtypes(include=['str','string']).columns:
            sub[c]=sub[c].astype(object)
        fit=smf.ols(formula,sub).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(sub.run_id)[0]},use_t=True)
        ci=fit.conf_int().loc['shown_judge_amount']
        rows.append(dict(check=name+' FULL DUMMY FE AND t',n=len(sub),n_runs=sub.run_id.nunique(),coef=fit.params['shown_judge_amount'],se=fit.bse['shown_judge_amount'],ci_low=ci.iloc[0],ci_high=ci.iloc[1],p=fit.pvalues['shown_judge_amount'],resid_df=fit.df_resid,n_params=len(fit.params)))
    pd.DataFrame(rows).to_csv(OUT/'amount_models.csv',index=False)
    # Myth amount vs previous OWN SEND, matching the original search lens population.
    d=p[(p.task_order=='myth_game')&(p.role=='investor')&p.family.isin(['Sonnet','GPT'])&(p['round']>=2)].dropna(subset=['own_send_amount','lag_sent']).copy()
    rest=[]
    for name,g in [('all',d),*d.groupby('family')]:
        hit=np.isclose(g.own_send_amount,g.lag_sent)
        rest.append(dict(family=name,n_rows=len(g),n_runs=g.run_id.nunique(),exact=int(hit.sum()),share=float(hit.mean())))
    pd.DataFrame(rest).to_csv(OUT/'later_restatement.csv',index=False)
    (OUT/'amount_columns.txt').write_text('\n'.join(p.columns))
    print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(rest).to_string(index=False))


def descriptive():
    from sklearn.metrics import cohen_kappa_score
    data=ROOT/'data/analysis/linguistic_n10_20261001'
    m=pd.read_csv(data/'myths.csv'); lab=pd.read_csv(data/'moral_labels_z-ai__glm-5.2.csv')
    m=m.merge(lab[['run_id','round','agent','label']],on=['run_id','round','agent'],how='left')
    g=m[(m.family=='Gemini')&(m['size']==8)&(~m.mixed)]
    out=[]
    for order,x in [('pooled',g),*g.groupby('task_order')]:
        for rnd,h in x[x['round'].isin([1,10])].groupby('round'):
            v=h.groupby('run_id').label.apply(lambda s:(s=='be generous').mean())
            out.append(dict(dataset='september_n10',judge='glm',order=order,round=rnd,n_myths=len(h),n_runs=len(v),generous=int((h.label=='be generous').sum()),share=(h.label=='be generous').mean(),run_mean=v.mean(),run_sd=v.std()))
    old=ROOT/'data/analysis/linguistic_20260923'
    orig=pd.read_csv(old/'myths.csv')
    ds=pd.read_csv(old/'moral_labels_deepseek__deepseek-v4-flash.csv')
    a=orig.merge(ds[['run_id','round','agent','label']],on=['run_id','round','agent'],how='left')
    g=a[(a.family=='Gemini')&(a['size']==8)&(~a.mixed)]
    for order,x in [('pooled',g),*g.groupby('task_order')]:
        for rnd,h in x[x['round'].isin([1,10])].groupby('round'):
            v=h.groupby('run_id').label.apply(lambda s:(s=='be generous').mean())
            out.append(dict(dataset='september',judge='deepseek',order=order,round=rnd,n_myths=len(h),n_runs=len(v),generous=int((h.label=='be generous').sum()),share=(h.label=='be generous').mean(),run_mean=v.mean(),run_sd=v.std()))
    pd.DataFrame(out).to_csv(OUT/'gemini_morals.csv',index=False)
    supp=[]
    for rnd,h in m[(m['size']==8)&m['round'].isin([1,10])].groupby('round'):
        for label in ['be generous','be fair','be cautious']:
            v=h.groupby('run_id').label.apply(lambda x:(x==label).mean())
            supp.append(dict(test='population moral share',family='all',round=rnd,label=label,n=len(h),count=int((h.label==label).sum()),share=float((h.label==label).mean()),run_mean=float(v.mean()),run_sd=float(v.std())))
    for fam,h in m.groupby('family'):
        v=h.groupby('run_id').label.apply(lambda x:(x=='be cautious').mean())
        supp.append(dict(test='all-corpus cautious share',family=fam,round='all',label='be cautious',n=len(h),count=int((h.label=='be cautious').sum()),share=float((h.label=='be cautious').mean()),run_mean=float(v.mean()),run_sd=float(v.std())))
    for fam,h in m[m['size']==8].groupby('family'):
        v=h.groupby('run_id').label.apply(lambda x:(x=='be cautious').mean())
        supp.append(dict(test='population-only cautious share',family=fam,round='all',label='be cautious',n=len(h),count=int((h.label=='be cautious').sum()),share=float((h.label=='be cautious').mean()),run_mean=float(v.mean()),run_sd=float(v.std())))
    pd.DataFrame(supp).to_csv(OUT/'supplement_descriptive.csv',index=False)
    corpus=m.groupby(['size','mixed']).agg(n_runs=('run_id','nunique'),n_myths=('text','size'),n_labeled=('label','count')).reset_index()
    corpus.to_csv(OUT/'corpus_counts.csv',index=False)
    dec=pd.read_csv(data/'decisions.csv'); v=dec[(dec.family=='Gemini')&(dec['size']==8)&(~dec.mixed)&(dec.role=='investor')].sent
    (OUT/'gemini_send.json').write_text(json.dumps(dict(n=len(v),min=float(v.min()),max=float(v.max()),mean=float(v.mean()),sd=float(v.std()))))
    glm=pd.read_csv(old/'moral_labels_z-ai__glm-5.2.csv')
    joined=glm.merge(ds,on=['run_id','round','agent'],suffixes=('_glm','_ds'))
    joined=joined.dropna(subset=['label_glm','label_ds'])
    (OUT/'judge_agreement.json').write_text(json.dumps(dict(n=len(joined),agreement=float((joined.label_glm==joined.label_ds).mean()),kappa=float(cohen_kappa_score(joined.label_glm,joined.label_ds))),indent=2))
    print(pd.DataFrame(out).to_string(index=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('part',choices=['word','moral','markers','regression','map','amount','descriptive']);args=ap.parse_args()
    {'word':word,'moral':moral,'markers':markers,'regression':regression,'map':map_analysis,'amount':amount,'descriptive':descriptive}[args.part]()
