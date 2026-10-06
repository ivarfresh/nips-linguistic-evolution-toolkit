"""Bounded steps 4–5 audit; no API calls or experimental data mutation."""
import hashlib
import json
import random
import re
import statistics as st
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/research/strategy_clause_audit_20261002'
DATA = ROOT/'data/analysis/strategy_clause_audit_20261002'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def tokens(t): return re.findall(r"[a-z0-9]+", re.sub(r'^\s*Myth:\s*','',t,flags=re.I).lower())
def grams(t):
    words=tokens(t)
    return {tuple(words[i:i+5]) for i in range(len(words)-4)}
def match(q,t):
    a,b=tokens(q),tokens(t)
    m=SequenceMatcher(None,a,b,autojunk=False).find_longest_match()
    return dict(longest_tokens=m.size,matched_phrase=' '.join(a[m.a:m.a+m.size]),
                normalized_full_quote=bool(a) and m.size==len(a))
def mean(xs): return st.mean(xs) if xs else None
def stats(xs): return dict(n=len(xs),mean=mean(xs),sd=st.stdev(xs) if len(xs)>1 else None)
def load_final(s,cache):
    path=ROOT/s['path']
    if s['path'] not in cache: cache[s['path']]=(sha(path),read(path))
    digest,data=cache[s['path']]
    if 'final_sha256' in s: assert digest==s['final_sha256']
    return data


def main():
    manifest=read(OUT/'manifest.json')
    assert manifest['plan_sha256']==sha(OUT/'PLAN.md')
    for name,digest in manifest['input_sha256'].items(): assert sha(ROOT/name)==digest
    samples={s['id']:s for s in manifest['sample']}
    codes={c['id']:c for i in range(1,4) for c in read(OUT/f'coding_{i}.json')}
    assert len(samples)==len(codes)==48
    cache={}; packets={}; lexical=[]; ledger=[]; checks=Counter()
    for ident,s in samples.items():
        p=read(DATA/f'{ident}_context.json'); assert p['sample']==s
        packets[ident]=p; final=load_final(s,cache)
        assert len(final['conversation_history'])==10
        previous_grams=set()
        for row in p['rounds']:
            n=row['round']; h=final['conversation_history'][n-1]
            assert row['own_text'].strip()==h['myths'][s['agent']].strip()
            checks['own_myths']+=1
            game=next(g for g in h['dyads'] if s['agent'] in g['agents']) if h.get('dyads') else h
            assert all(row['agent_game'][k]==game.get(k) for k in row['agent_game'])
            game_calls=[x for x in final['agents'][s['agent']]['interaction_history'] if x.get('metadata',{}).get('task')=='game' and x.get('metadata',{}).get('round')==n]
            assert row['decision_prompts']==[x['messages_sent'][-1]['content'] for x in game_calls]
            checks['game_contexts']+=1
            donor=row['exposed']; unseen=row['unseen']
            if donor:
                exp=h['myth_exposures'][s['agent']]
                assert exp['original_author_id']==donor['agent'] and exp['source_round']==int(donor['round']) and not exp.get('substitution_applied')
                assert donor['text'].strip()==final['conversation_history'][int(donor['round'])-1]['myths'][donor['agent']].strip()
                calls=[x for x in final['agents'][s['agent']]['interaction_history'] if x.get('metadata',{}).get('task')=='myth' and x.get('metadata',{}).get('round')==n]
                accepted=[x for x in calls if row['own_text'].strip() in (x['response'].get('content','') if isinstance(x['response'],dict) else x['response'])]
                assert accepted and all(donor['text'] in x['messages_sent'][-1]['content'] for x in accepted)
                checks['accepted_exposures']+=1
                uf=load_final(unseen,cache)
                assert unseen['run_id']!=s['run_id'] and unseen['same_composition']
                for key in ['family','round','size','mixed','task_order','composition']: assert str(donor[key])==str(unseen[key]),(ident,n,key)
                assert unseen['text'].strip()==uf['conversation_history'][int(unseen['round'])-1]['myths'][unseen['agent']].strip()
                checks['unseen_final_texts']+=1
                novel=grams(row['own_text'])-previous_grams
                e=len(novel & grams(donor['text'])); u=len(novel & grams(unseen['text']))
                lexical.append(dict(id=ident,run_id=s['run_id'],tier=s['tier'],family=s['family'],order=s['task_order'],round=n,
                                    novel_fivegrams=len(novel),exposed_matches=e,unseen_matches=u,
                                    delta=(e-u)/len(novel) if novel else None,
                                    interpretation='Lexical novelty relative to all earlier own myths, not semantic novelty.'))
            previous_grams |= grams(row['own_text'])
        for e in codes[ident]['events']:
            n=e['round']; row=p['rounds'][n-1]; older=p['rounds'][:n-1]
            assert e['before_quote'] in p['rounds'][e['before_round']-1]['own_text']
            assert e['after_quote'] in row['own_text']
            for k,source in [('exposed_quote','exposed'),('unseen_quote','unseen')]:
                if e[k]: assert e[k] in row[source]['text']
            checks['selected_events']+=1
            q=e['after_quote']; prior_set=set().union(*(grams(x['own_text']) for x in older))
            novel=grams(q)-prior_set
            old_matches=[dict(round=x['round'],**match(q,x['own_text'])) for x in older]
            last_game=n if s['task_order']=='game_myth' else n-1
            future=[x for x in p['rounds'] if x['round']>last_game]
            next_roles={role:next((dict(round=x['round'],game=x['agent_game'],decision_prompts=x['decision_prompts'])
                                  for x in future if x['agent_game'][role]==s['agent']),None)
                        for role in ['investor','trustee']}
            ledger.append(dict(id=ident,event_round=n,run_id=s['run_id'],tier=s['tier'],family=s['family'],order=s['task_order'],
                original_event=e,original_assessment_status='Inherited exploratory machine reading; not independent re-coding.',
                previous_own_match=match(q,older[-1]['own_text']),
                all_earlier_own_matches=old_matches,
                exposed_match=match(q,row['exposed']['text']),unseen_match=match(q,row['unseen']['text']),
                novel_quote_fivegrams=len(novel),exposed_novel_quote_matches=len(novel & grams(row['exposed']['text'])),
                unseen_novel_quote_matches=len(novel & grams(row['unseen']['text'])),
                previous_own_text=older[-1]['own_text'],exposed_text=row['exposed']['text'],unseen_text=row['unseen']['text'],
                all_pre_write_games=[dict(round=x['round'],**x['agent_game']) for x in p['rounds'][:last_game]],
                last_pre_write_game_round=last_game,same_round_game_can_be_source=s['task_order']=='game_myth',
                first_subsequent_decisions_by_role=next_roles,
                timing_caution='Later role-specific decisions may follow intervening myths; availability is not an isolated effect window.',
                source_caution='Observed temporal availability permits competing sources; it does not identify which caused the clause.'))
    assert dict(checks)==dict(own_myths=480,game_contexts=480,accepted_exposures=432,unseen_final_texts=432,selected_events=44)

    # Text-only manual eligibility decisions, including explicit exclusion reasons.
    forgiveness={('T24',6):'Failure unspecified; no measurable poor-return threshold; recognition/partner switching is a competing issue.',
                 ('T27',2):'General trust in a noise-distorted gift; no explicit new sender response to a personally observed shortfall.',
                 ('T31',7):'Explicit continued full sending, but skewed/diminished measure has no numerical shortfall threshold.',
                 ('T33',3):'Clear send-again-once advice, but repaid poorly is undefined; no invented below-half/break-even mapping.',
                 ('T36',6):'Carry rule to next river after a failed partnership; no measurable single-noise-shortfall rule.',
                 ('T38',7):'Thin harvest/winter undefined; forgiving weather twice already present in R6.'}
    reduction={('T19',8):'Conditional four after insufficient/proven reciprocity; no exact repeat count.',
               ('T20',5):'Recalibration after a decline; no count or exact sender reduction rule.',
               ('T32',8):'Reduction to one is a narrated example; repeated betrayal has no fixed count.',
               ('T34',6):'Season/withdrawal undefined in duration and poor-return threshold.',
               ('T37',8):'Three consecutive dry responses specified, but dry undefined numerically; cannot score hidden true returns as perceived defection.',
               ('T38',8):'Narrated four tiny SENDS, not four poor RETURNS; thimble undefined.',
               ('T39',6):'Third thin return stated, but bare cost/thin threshold inconsistent with the 5.5 illustrative return; no unique numeric test.',
               ('T40',3):'Season length and shortfall threshold unspecified.',
               ('T41',3):'Deliberate greed requires intent not observable from transfer counts.',
               ('T46',3):'Habitual hoarding has neither a repeat count nor a numeric threshold.'}
    amounts={('T03',6):5.,('T07',9):5.,('T21',4):3.,('T24',4):3.,('T25',3):5.,('T36',3):5.}
    screens=[]
    for event in ledger:
        key=(event['id'],event['event_round']); tests=[]
        if key in forgiveness: tests.append(dict(test='forgiveness',eligible=False,reason=forgiveness[key]))
        if key in reduction: tests.append(dict(test='repeated_defection',eligible=False,reason=reduction[key]))
        if key in amounts: tests.append(dict(test='sender_amount',candidate=True))
        if not tests: tests.append(dict(test='none',eligible=False,reason='Receiver rule, conditional generosity, or narrated amount outside the three bounded tests.'))
        screens.append(dict(id=key[0],round=key[1],tests=tests))
    # Select timing windows and prior decisions before extracting next-send values.
    eligibility=[]
    for (ident,n),target in amounts.items():
        p=packets[ident]; s=p['sample']; own=s['agent']
        game_first=s['task_order']=='game_myth'
        active_round=n+1 if game_first else n
        prior=[r for r in p['rounds'] if (r['round']<=n if game_first else r['round']<n) and r['agent_game']['investor']==own]
        current=p['rounds'][active_round-1] if active_round<=10 else None
        eligible=bool(current and current['agent_game']['investor']==own and prior)
        eligibility.append(dict(id=ident,event_round=n,run_id=s['run_id'],family=s['family'],order=s['task_order'],target=target,
                                active_game_round=active_round,prior_sender_round=prior[-1]['round'] if prior else None,
                                eligible=eligible,reason='Within the fixed pre-next-myth window.' if eligible else 'No sender decision before the next own myth (or missing prior send).'))
    eligible_runs={x['run_id'] for x in eligibility if x['eligible']}
    gates=dict(forgiveness_eligible_runs=0,repeated_defection_eligible_runs=0,amount_eligible_runs=len(eligible_runs),minimum_distinct_runs=10,
               inference_allowed=False,reason='Strict antecedents undefined for conditional tests; amount cases fall below run/variation gate.')
    assert len(eligible_runs)<10
    # These are within-case descriptions only, never a significance test.
    for x in eligibility:
        if not x['eligible']:continue
        p=packets[x['id']]; before=p['rounds'][x['prior_sender_round']-1]; after=p['rounds'][x['active_game_round']-1]
        x.update(prior_send=before['agent_game']['sent'],next_send=after['agent_game']['sent'])
        assert all(v is not None and 0<=v<=5 for v in [x['prior_send'],x['next_send']])
        x.update(distance_before=abs(x['prior_send']-x['target']),distance_after=abs(x['next_send']-x['target']),
                 send_change=x['next_send']-x['prior_send'],already_at_target=abs(x['prior_send']-x['target'])<1e-8,
                 decision_prompt=after['decision_prompts'])

    by_run=defaultdict(list)
    for x in lexical:
        if x['delta'] is not None:by_run[x['run_id']].append(x['delta'])
    run_means=[st.mean(v) for v in by_run.values()]
    rng=random.Random(20261003)
    boot=sorted(st.mean(rng.choices(run_means,k=len(run_means))) for _ in range(2000))
    lexical_summary=dict(run_weighted=stats(run_means),descriptive_cluster_bootstrap_95=[boot[49],boot[1949]],
                         weighting='Equal weight per run; each run averages its sampled agents/rounds. Paired exposure-minus-unseen target-novel-fivegram fraction.',
                         zero_novel_targets=sum(x['novel_fivegrams']==0 for x in lexical),
                         limitation='Bootstrap captures variation in this selected sample, not causal/randomized or population-sampling uncertainty.')
    strata={}
    for tier in ['september','frontier']:
        for order in ['game_myth','myth_game']:
            group=defaultdict(list)
            for x in lexical:
                if x['tier']==tier and x['order']==order and x['delta'] is not None:group[x['run_id']].append(x['delta'])
            strata[tier+'/'+order]=stats([st.mean(v) for v in group.values()])
    lexical_summary['strata']=strata
    result=dict(plan_sha256=sha(OUT/'FOLLOWUP_PLAN_20261003.md'),script_sha256=sha(Path(__file__)),
                checked_final_sha256={path:value[0] for path,value in sorted(cache.items())},
                input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'manifest.json',*[OUT/f'coding_{i}.json' for i in range(1,4)]]},
                validation=dict(checks),n_runs=45,source_assessments=Counter(e['original_event']['source_assessment'] for e in ledger),
                events=ledger,lexical_pairs=lexical,lexical_summary=lexical_summary,behavior_screens=screens,behavior_eligibility=eligibility,behavior_gates=gates)
    (OUT/'followup_results_20261003.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['STEPS 4–5: BOUNDED ATTRIBUTION AND BEHAVIOR FOLLOW-UP','',
           'Exploratory; human reliability unestablished. No new runs or model calls.',
           'All 44 earlier selected events audited, not an exhaustive change census.',
           '480 own myths and game contexts; 432 exposures and matched unseen texts verified against finals.',
           '', 'SOURCE LEDGER (semantic verdicts inherited from earlier machine readings)',
           'Exact phrase diagnostics below supplement, not replace, those semantic judgments.','']
    for e in ledger:
        o=e['original_event']; best=max(e['all_earlier_own_matches'],key=lambda x:x['longest_tokens'])
        lines.extend([f"{e['id']} R{e['event_round']} — {o['source_assessment']}",
                      'New clause: '+o['after_quote'], 'Earlier own: '+o['before_quote'],
                      'Shown peer: '+o['exposed_quote'], 'Unseen comparison: '+o['unseen_quote'],
                      'Original reasoning (unvalidated): '+o['source_reason'],
                      f"Consecutive tokens matching earlier own / peer / unseen: {best['longest_tokens']} (R{best['round']}) / {e['exposed_match']['longest_tokens']} / {e['unseen_match']['longest_tokens']}.",
                      f"New clause fivegrams found in peer / unseen: {e['exposed_novel_quote_matches']} / {e['unseen_novel_quote_matches']}.",
                      f"Latest preceding game: R{e['last_pre_write_game_round']}. Same-round game precedes writing: {e['same_round_game_can_be_source']}.",
                      'Game experience remains a competing source; complete pre-writing game timeline is in JSON.',''])
    lines+=['LEXICAL DIAGNOSTIC — NOT A NORM-TRANSMISSION TEST',json.dumps(lexical_summary,indent=2),'',
            'BEHAVIORAL ELIGIBILITY — EVERY RECORDED EVENT SCREENED',json.dumps(gates,indent=2),'']
    for screen in screens:lines.append(f"{screen['id']} R{screen['round']}: "+json.dumps(screen['tests']))
    lines+=['','AMOUNT CASES — TIMING SELECTED BEFORE NEXT-SEND SUMMARY']
    for x in eligibility:lines.append(json.dumps({k:v for k,v in x.items() if k!='decision_prompt'}))
    lines+=['','No conditional behavior effect, null effect, or causal claim is estimated.',
            'Low eligibility is a measurement/timing limitation, not evidence of no enactment.',
            'All original labels, raw texts, prompts, and published analyses remain unchanged.']
    (OUT/'followup_source_ledger_20261003.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(validation=dict(checks),source_assessments=result['source_assessments'],lexical=lexical_summary,
                         gates=gates,amount_cases=[{k:v for k,v in x.items() if k!='decision_prompt'} for x in eligibility]),indent=2))


if __name__=='__main__':main()
