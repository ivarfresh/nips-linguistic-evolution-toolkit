"""Read-only corpus verification and budget preparation. Makes no model calls."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/research/narrative_evolution_20261003'
SOURCES=[
 ('september_original','docs/figures/negative_only_crossmodel_reasoning_rerun_20260909/provenance.json','runs'),
 ('september_n10_extension','data/json/noise_experiments/table1_n10_extension_20261001/completion_receipt.json','finals'),
 ('september_mixed_dyads','data/json/noise_experiments/mixed_model_dyads_20260917/completion_receipt.json','finals'),
 ('september_mixed_populations','data/json/noise_experiments/mixed_model_populations_20260918/completion_receipt.json','finals'),
 ('september_noise_extension','data/json/noise_experiments/figure2_no_defectors_20260915/completion_receipt.json','finals'),
 ('september_range2_bridge','data/json/noise_experiments/noise_strength_bridge_20260916/completion_receipt.json','finals'),
 ('frontier_main','data/json/noise_experiments/frontier_rerun_20260918/main_reasoning_on_receipt.json','finals'),
 ('frontier_main_mixed','data/json/noise_experiments/frontier_mixed_main_20260928/all_receipt.json','finals'),
 ('frontier_update_opus','data/json/noise_experiments/frontier_rerun_20260918/main_opus55_receipt.json','finals'),
 ('frontier_update_sol','data/json/noise_experiments/frontier_rerun_20260918/main_sol6_receipt.json','finals'),
 ('frontier_update_mixed','data/json/noise_experiments/frontier_mixed_populations_20260928/completion_receipt.json','finals'),
 ('frontier_defectors','data/json/noise_experiments/frontier_defector_pilot_20261001/all_receipt.json','finals'),
]
META=['replicate_id','num_agents','noise_config','noise_semantics','history_policy',
      'chat_memory_mode','prompt_regime','condition_sha256','myth_prompt_arm_id',
      'defector_agent_ids','defector_action_policy','defector_myth_policy','defector_role_visible_to_self',
      'random_defection_probability','random_defection_unit','pairing_seed','noise_seed','defector_seed']

def main():
    runs=[];seen=set();receipts={};totals=Counter();strata={};issues=[]
    for cohort,receipt,key in SOURCES:
        raw=(ROOT/receipt).read_bytes();receipts[receipt]=hashlib.sha256(raw).hexdigest()
        counts=Counter()
        for entry in json.loads(raw)[key]:
            path=Path(entry['path'])
            path=path if path.is_absolute() else ROOT/path
            raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            assert digest==entry['sha256'],path
            assert digest not in seen,('duplicate',path)
            seen.add(digest);run=json.loads(raw)
            assert len(run['conversation_history'])==10 and [x['round'] for x in run['conversation_history']]==list(range(1,11)),path
            assert 'agents' in run and 'game_data' in run and 'run_metadata' in run,path
            counts['all_completed_runs']+=1
            order='_'.join(run['task_order'])
            meta=run['run_metadata'];authors={};chars=0;myths=0;unreadable=0
            for row in run['conversation_history']:
                for agent,text in (row.get('myths') or {}).items():
                    myths+=1;authors.setdefault(agent,[]).append(row['round'])
                    if isinstance(text,str):chars+=len(text)
                    bad=not isinstance(text,str) or not text.strip() or text.strip() in ['{}','[]','null','Myth: {}','Myth: []']
                    if bad:
                        unreadable+=1;issues.append(dict(path=str(path.relative_to(ROOT)),agent=agent,round=row['round'],reason='Empty/non-text/empty-object response; not a negative feature label.'))
            if myths:
                assert all(rounds==list(range(1,11)) for rounds in authors.values()),path
                counts.update(myth_bearing_runs=1,trajectories=len(authors),myth_entries=myths,text_characters=chars,obvious_unreadable=unreadable)
            models={a:run['agents'][a].get('model') for a in authors}
            role={a:run['agents'][a].get('population_role') for a in authors}
            runs.append(dict(cohort=cohort,path=str(path.relative_to(ROOT)),sha256=digest,task_order=order,
                myth_entries=myths,trajectories=len(authors),text_characters=chars,obvious_unreadable=unreadable,
                models=models,population_roles=role,metadata={k:meta.get(k) for k in META}))
        strata[cohort]=dict(counts);totals.update(counts)
        print(cohort,dict(counts),flush=True)
    assert totals['all_completed_runs']==1109 and totals['myth_entries']==38720 and totals['trajectories']==3872,totals
    with urlopen('https://openrouter.ai/api/v1/models',timeout=30) as r:
        model=next(x for x in json.load(r)['data'] if x['id']=='openai/gpt-6.1-sol')
    pin=float(model['pricing']['prompt']);pout=float(model['pricing']['completion'])
    n=totals['trajectories'];text_tokens=totals['text_characters']/4
    # Planning assumptions, not measured token counts; completion includes reasoning.
    extraction=(text_tokens+1800*n)*pin+n*8000*pout
    attribution=(5*text_tokens+1200*n)*pin+n*4000*pout
    calibration=48*((text_tokens/n+1800)*pin+8000*pout)
    budget=dict(model=model['id'],reasoning='high',workers=8,extraction_trajectories=n,
        source_comparison_calls_upper=n,additional_calibration_calls=48,
        extraction_est_usd=extraction,source_est_usd=attribution,calibration_est_usd=calibration,
        repair_allowance_fraction=.25,total_est_usd=1.25*(extraction+attribution+calibration),
        requested_cap_usd=800,paid_calls_made=0,
        assumptions='chars/4 input; 1800/1200 instruction tokens per stage; 8000/4000 total output+reasoning tokens; source stage allows own+peer+3 unseen trajectories; no caching discount; 25% repair allowance. Re-estimate from calibration before scaling.',
        price_source='https://openrouter.ai/api/v1/models',pricing=model['pricing'],verified_at=datetime.now(timezone.utc).isoformat())
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'corpus_manifest.json').write_text(json.dumps(dict(receipt_sha256=receipts,totals=dict(totals),strata=strata,runs=runs,obvious_unreadable=issues),indent=2)+'\n')
    (OUT/'budget_preflight.json').write_text(json.dumps(budget,indent=2)+'\n')
    print(json.dumps(dict(totals=dict(totals),estimated_usd=budget['total_est_usd'],paid_calls_made=0),indent=2))

if __name__=='__main__':main()
