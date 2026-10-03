"""Bounded Batch execution for the sole stable pilot feature: challenge dialogue.

No API submission without --step. Each step collects completed waves, then
reserves worst-case tokens before dispatching more. --prepare is free/local.
"""
import argparse
import fcntl
import json
import random
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import narrative_feature_screen as screen
from narrative_feature_screen import BASE, ROOT, MODEL, read, save, digest

PILOT=BASE/'feature_screen'
OUT=PILOT/'full_dialogue'
WAVE=800
PACK=6
MAX_OUTPUT=6000
CAP=100.0

def budget_allows(pilot_cost, waves, next_bound, cap=CAP):
    accounted=pilot_cost+sum(w.get('accounted_upper_usd',w['bound']) for w in waves)
    return accounted+next_bound<=cap

def prepare():
    target=OUT/'plan.json'
    if target.exists():return read(target)
    evaluation=read(PILOT/'dialogue/evaluation.json')
    if not evaluation['complete'] or not evaluation['scores'][0]['numeric_gate_pass']:
        raise ValueError('Dialogue regression gate has not passed')
    rubric=read(PILOT/'dialogue/rubric.json')
    if rubric['features']!=['challenge_dialogue']:raise ValueError('Unexpected features')
    corpus=read(BASE/'corpus_manifest.json');rows=[]
    for n,run in enumerate(corpus['runs']):
        if not run['models']:continue
        raw=(ROOT/run['path']).read_bytes()
        if digest(raw)!=run['sha256']:raise ValueError('Source changed')
        data=json.loads(raw)
        for history in data['conversation_history']:
            for agent,text in sorted(history['myths'].items()):
                rows.append(dict(id=f'M{len(rows)+1:05}',run_index=n,run_sha256=run['sha256'],agent=agent,
                    round=history['round'],model=run['models'][agent],cohort=run['cohort'],task_order=run['task_order'],
                    scripted_defector=agent in (run['metadata']['defector_agent_ids'] or []),
                    text=text,text_sha256=digest(text.encode())))
    if len(rows)!=38720:raise ValueError('Corpus scope changed')
    random.Random(2026100305).shuffle(rows)
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'index.json',rows)
    requests=[]
    for i in range(0,len(rows),PACK):
        chunk=rows[i:i+PACK]
        body=dict(model=MODEL,reasoning_effort='high',max_completion_tokens=MAX_OUTPUT,
            response_format={'type':'json_object'},messages=[{'role':'system','content':rubric['prompt']},
            {'role':'user','content':json.dumps([{'id':x['id'],'text':x['text']} for x in chunk],ensure_ascii=False)}])
        bound=(len(json.dumps(body,ensure_ascii=False).encode())+2000)*1.25e-6+MAX_OUTPUT*5e-6
        requests.append(dict(custom_id=f'D{i//PACK:05}',method='POST',url='/v1/chat/completions',body=body,
            ids=[x['id'] for x in chunk],bound=bound))
    save(OUT/'requests.json',requests)
    waves=[]
    for i in range(0,len(requests),WAVE):
        chunk=requests[i:i+WAVE];path=OUT/f'input_{i//WAVE:02}.jsonl'
        with path.open('w') as f:
            for x in chunk:f.write(json.dumps({k:x[k] for k in ['custom_id','method','url','body']},ensure_ascii=False)+'\n')
        waves.append(dict(number=i//WAVE,start=i,end=i+len(chunk),path=str(path.relative_to(ROOT)),
            sha256=digest(path.read_bytes()),bound=sum(x['bound'] for x in chunk)))
    pilot_ledger=[json.loads(x) for x in (PILOT/'budget.jsonl').read_text().splitlines()]
    costs={x['rid']:x['bound'] for x in pilot_ledger if x['event']=='reserve'}
    for x in pilot_ledger:
        if x['event']=='settle':costs[x['rid']]=x['cost_upper_usd']
    point=evaluation['rough_batch_full_cost_usd'];estimate=point*1.25+sum(costs.values())
    plan=dict(model=MODEL,reasoning='high',myths=len(rows),requests=len(requests),waves=waves,
        max_completion_tokens=MAX_OUTPUT,cap_usd=CAP,pilot_accounted_upper_usd=sum(costs.values()),
        batch_point_estimate_usd=point,estimate_with_25pct_allowance_and_pilots_usd=estimate,
        total_worst_case_request_bound_usd=sum(x['bound'] for x in requests),
        rubric_sha256=rubric['sha256'],source_manifest_sha256=digest((BASE/'corpus_manifest.json').read_bytes()),
        caveat='Exploratory machine-coded dialogue presence only. Pilot is enriched; estimate is not an invoice. Form/stance are not quantitative endpoints.')
    save(target,plan);return plan

def step(plan):
    if plan['estimate_with_25pct_allowance_and_pilots_usd']>100:
        raise SystemExit('Estimate exceeds $100: explicit approval required; no Batch submitted.')
    guard=(OUT/'process.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    load_dotenv(ROOT/'.env');client=OpenAI(base_url='https://api.openai.com/v1',max_retries=0,timeout=120)
    state_path=OUT/'state.json';state=read(state_path) if state_path.exists() else {'waves':[]}
    requests=read(OUT/'requests.json');byrequest={x['custom_id']:x for x in requests}
    index={x['id']:x for x in read(OUT/'index.json')};screen.FEATURES=['challenge_dialogue']
    for wave in state['waves']:
        if wave.get('collected'):continue
        if not wave.get('batch_id'):
            matches=[b for b in client.batches.list(limit=100) if (b.metadata or {}).get('screen_wave')==wave['tag']]
            if len(matches)!=1:raise RuntimeError('Unresolved Batch create; preserve reservation and inspect before retry')
            wave['batch_id']=matches[0].id;save(state_path,state)
        batch=client.batches.retrieve(wave['batch_id']);wave['status']=batch.status
        wave['counts']=batch.request_counts.model_dump() if batch.request_counts else None
        save(state_path,state)
        if batch.status not in ['completed','failed','expired','cancelled']:continue
        if batch.status=='failed' and not batch.output_file_id:
            wave['collected']=True;wave['accounted_upper_usd']=0.;wave['errors']=batch.errors.model_dump() if batch.errors else None
            save(state_path,state);continue
        output=[]
        if batch.output_file_id:
            raw=client.files.content(batch.output_file_id).text
            output=[json.loads(line) for line in raw.splitlines()]
            save(OUT/f"output_{wave['number']:02}.json",output)
        if batch.error_file_id:
            raw=client.files.content(batch.error_file_id).text
            save(OUT/f"errors_{wave['number']:02}.json",[json.loads(line) for line in raw.splitlines()])
        accounted={x['custom_id']:x['bound'] for x in requests[wave['start']:wave['end']]}
        labels=[]
        for record in output:
            rid=record['custom_id']; req=byrequest[rid]
            response=record.get('response') or {};body=response.get('body') or {};usage=body.get('usage')
            if usage:
                accounted[rid]=usage['prompt_tokens']*1.25e-6+usage['completion_tokens']*5e-6
            if response.get('status_code')!=200 or not body.get('choices'):continue
            choice=body['choices'][0]
            try:parsed=json.loads(choice['message']['content'])
            except (ValueError,TypeError):continue
            if choice['finish_reason']!='stop':continue
            if sorted(x.get('id','') for x in parsed.get('myths',[]))!=sorted(req['ids']):continue
            for row in parsed['myths']:
                errors=screen.validate({'myths':[row]},[index[row['id']]])
                labels.append(dict(id=row['id'],request_id=rid,parsed=row,errors=errors))
        save(OUT/f"labels_{wave['number']:02}.json",labels)
        wave['collected']=True;wave['accounted_upper_usd']=sum(accounted.values())
        wave['valid_myths']=sum(not x['errors'] for x in labels)
        save(state_path,state)
    if any(w.get('status')=='failed' for w in state['waves']):
        print('A Batch wave failed; inspect errors before further submissions.',flush=True)
    else:
        used={w['number'] for w in state['waves']}
        for definition in plan['waves']:
            if definition['number'] in used:continue
            if not budget_allows(plan['pilot_accounted_upper_usd'],state['waves'],definition['bound']):break
            path=ROOT/definition['path']
            if digest(path.read_bytes())!=definition['sha256']:raise ValueError('Batch input changed')
            with path.open('rb') as f:uploaded=client.files.create(file=f,purpose='batch')
            wave=dict(definition,tag=uuid.uuid4().hex,file_id=uploaded.id,status='submitting')
            state['waves'].append(wave);save(state_path,state)
            batch=client.batches.create(input_file_id=uploaded.id,endpoint='/v1/chat/completions',completion_window='24h',
                metadata={'screen_wave':wave['tag'],'analysis':'myth-dialogue-20261003'})
            wave.update(batch_id=batch.id,status=batch.status);save(state_path,state)
            print(f"Submitted wave {wave['number']} ({wave['end']-wave['start']} requests), status={batch.status}",flush=True)
    summary=dict(submitted_waves=len(state['waves']),total_waves=len(plan['waves']),
        collected_waves=sum(bool(w.get('collected')) for w in state['waves']),
        valid_myths=sum(w.get('valid_myths',0) for w in state['waves']),
        accounted_including_pending_upper_usd=plan['pilot_accounted_upper_usd']+sum(w.get('accounted_upper_usd',w['bound']) for w in state['waves']),
        statuses=[{'number':w['number'],'status':w['status'],'counts':w.get('counts')} for w in state['waves']])
    save(OUT/'progress.json',summary);print(json.dumps(summary),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--step',action='store_true');ap.add_argument('--watch',action='store_true');args=ap.parse_args()
    plan=prepare()
    prefix='CONTINUATION' if (OUT/'state.json').exists() else 'PREFLIGHT'
    command_flag='--watch' if args.watch else '--step'
    print(f"{prefix}: MODEL={MODEL} REASONING=high N={plan['requests']} MYTHS={plan['myths']} WORKERS=provider-managed WAVE_SIZE={WAVE} EST_COST=${plan['estimate_with_25pct_allowance_and_pilots_usd']:.2f} CAP=$100 CMD=python3 analyses/narrative_dialogue_batch.py {command_flag}",flush=True)
    if args.step:step(plan)
    if args.watch:
        from narrative_dialogue_report import main as report
        failures=0
        for _ in range(7200):
            try:
                step(plan);report();failures=0
            except Exception as exc:
                failures+=1
                print(f'Batch continuation error {type(exc).__name__}; attempt {failures}/10',flush=True)
                if failures>=10:raise
                time.sleep(min(3*2**failures,60));continue
            progress=read(OUT/'progress.json');state=read(OUT/'state.json')
            if any(w.get('status')=='failed' for w in state['waves']):
                raise SystemExit('Batch wave failed; preserved evidence, no further automatic submission.')
            if progress['collected_waves']==len(plan['waves']):
                print('All waves collected. Check summary.json for invalid/missing labels before claiming complete coverage.',flush=True)
                return
            active=any(not w.get('collected') for w in state['waves'])
            if not active:
                raise SystemExit('No active wave and work remains: budget guard requires review; no further spend.')
            time.sleep(45)
        raise SystemExit('Bounded Batch watcher ended; saved state supports resumption.')

if __name__=='__main__':main()
