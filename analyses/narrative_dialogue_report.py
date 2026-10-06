"""Descriptive dialogue screen reports; no behavioral or transmission claims."""
from collections import Counter, defaultdict
import json
import random
from narrative_dialogue_batch import OUT, BASE, read, save, digest
from narrative_feature_screen_evaluate import status

def main():
    if not (OUT/'index.json').exists():raise SystemExit('Prepare the frozen batch first')
    index=read(OUT/'index.json');corpus=read(BASE/'corpus_manifest.json')
    labels={};invalid=[]
    for path in sorted(OUT.glob('labels_*.json')):
        for row in read(path):
            if row['errors']:invalid.append({'id':row['id'],'errors':row['errors']});continue
            if row['id'] in labels:raise ValueError('Duplicate valid myth')
            labels[row['id']]=row['parsed']
    groups=defaultdict(Counter);overall=Counter();candidate=[];conditions={}
    for item in index:
        value=status(labels[item['id']],'challenge_dialogue') if item['id'] in labels else 'missing'
        overall[value]+=1
        run=corpus['runs'][item['run_index']];meta=run['metadata'];group_size=meta['num_agents']
        factors={k:meta.get(k) for k in ['noise_config','noise_semantics','history_policy','chat_memory_mode',
            'prompt_regime','myth_prompt_arm_id','defector_agent_ids','defector_action_policy','defector_myth_policy',
            'defector_role_visible_to_self','random_defection_probability','random_defection_unit']}
        factors['population_composition']=dict(sorted(Counter(run['models'].values()).items()))
        condition=digest(json.dumps(factors,sort_keys=True).encode())[:12];conditions[condition]=factors
        key=(item['cohort'],item['model'],item['task_order'],group_size,item['scripted_defector'],condition)
        groups[key][value]+=1;groups[key]['total']+=1
        if value!='missing':candidate.append({**item,'status':value,'reading':labels[item['id']]})
    table=[dict(cohort=k[0],model=k[1],task_order=k[2],group_size=k[3],scripted_defector=k[4],condition=k[5],
                **{s:c[s] for s in ['present','unclear','absent','missing','total']}) for k,c in sorted(groups.items())]
    state=read(OUT/'state.json') if (OUT/'state.json').exists() else {'waves':[]}
    plan=read(OUT/'plan.json')
    terminal=len(state['waves'])==len(plan['waves']) and all(x.get('collected') for x in state['waves'])
    result=dict(expected_myths=len(index),valid_labeled_myths=len(labels),counts=dict(overall),
        all_waves_collected=terminal,complete_coverage=overall['missing']==0,
        invalid_rows=invalid,groups=table,conditions=conditions,
        caveat='AI-screened fictional challenge dialogue only. Not human-validated; not strategy evolution, abstract reasoning quality, or transmission. Unknown and missing are not negatives.')
    save(OUT/'summary.json',result)
    lines=['Dialogue screen — '+('COMPLETE COVERAGE' if result['complete_coverage'] else 'PARTIAL; DO NOT USE AS FULL-CORPUS RESULT'),'',
        f"Expected myths: {len(index)}; valid labeled: {len(labels)}.",
        f"Present: {overall['present']}; uncertain: {overall['unclear']}; absent: {overall['absent']}; missing: {overall['missing']}.",
        '',result['caveat'],'','Counts keep cohorts, orders, group sizes, model composition, noise/history/prompt/defection conditions and author roles separate.',
        'Condition identifiers map to the complete factor definitions in summary.json.',
        'cohort | model | order | group size | scripted defector | condition | present | uncertain | absent | missing | total']
    for row in table:
        lines.append(' | '.join(str(row[k]) for k in ['cohort','model','task_order','group_size','scripted_defector','condition','present','unclear','absent','missing','total']))
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    completed_requests=sum((w.get('counts') or {}).get('completed',0) for w in state['waves'])
    submitted_requests=sum(w['end']-w['start'] for w in state['waves'])
    phase='Machine coding collected; source review still required' if result['complete_coverage'] else 'Collected with missing/invalid labels; review required' if terminal else 'Batch processing in progress'
    status_lines=[phase,'','Scope: fictional challenge dialogue only. Other feature counts and rule evolution are held.',
        f"Requests submitted: {submitted_requests} / {plan['requests']}.",
        f"Requests completed at provider: {completed_requests}.",
        f"Myths downloaded and technically validated: {len(labels)} / {len(index)}.",
        'Completed requests become available for local validation when their Batch wave finishes.',
        '',f"Estimated total including pilots and allowance: ${plan['estimate_with_25pct_allowance_and_pilots_usd']:.2f}.",
        'Accounting guard: $100; pending waves reserve their maximum before more can launch.',
        'The local one-time runner collects waves and writes REPORT.txt and summary.json.',
        'Batch jobs continue remotely if the local runner stops; resume with the saved state.',
        '', 'These are exploratory machine labels, not human-validated findings or evidence of transmission.']
    (OUT/'RUN_STATUS.txt').write_text('\n'.join(status_lines)+'\n')
    # Fixed procedure, source-check sample is deliberately not prevalence sampling.
    rng=random.Random(2026100306);rng.shuffle(candidate);selected=[];used=set()
    for value in ['present','unclear','absent']:
        for _ in range(6):
            choices=[x for x in candidate if x['status']==value and x['run_sha256'] not in used]
            if not choices:break
            counts=Counter(x['model'] for x in selected)
            x=min(choices,key=lambda r:counts[r['model']]);selected.append(x);used.add(x['run_sha256'])
    save(OUT/'source_check_packet.json',selected)
    print(json.dumps({k:result[k] for k in ['expected_myths','valid_labeled_myths','counts','complete_coverage','all_waves_collected']}))

if __name__=='__main__':main()
