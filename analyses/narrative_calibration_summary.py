"""Free, reproducible calibration diagnostics; not population prevalence."""
import argparse, json
from collections import Counter
from narrative_calibration import OUT, FEATURES, OPS, read, validate, charged_cost

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--repair',action='store_true');args=ap.parse_args()
    base=OUT
    if args.repair:OUT=base/'repair'
    sample=read(OUT/'sample.json');pairs=[];invalid=[];outputs=[]
    for item in sample:
        pair=[]
        for rep in ['A','B']:
            path=OUT/f"{item['id']}_{rep}.json"
            if not path.exists():continue
            row=read(path);outputs.append(row)
            errors=validate(row['parsed'],item) if row['parsed'] else ['invalid JSON']
            if row['finish_reason']!='stop':errors.append('finish_reason')
            if args.repair and row['parsed'] and any(t.get('basis') not in ['explicit_advice','endorsed_custom'] for t in row['parsed'].get('transitions',[])):errors.append('unsupported transition basis')
            if errors:invalid.append({'id':item['id'],'rep':rep,'errors':errors})
            else:pair.append(row)
        if len(pair)==2:pairs.append((item,pair))
    def agreement(kind,labels,key):
        rows=[]
        for label in labels:
            counts=Counter()
            for item,pair in pairs:
                for a,b in zip(pair[0]['parsed']['rounds'],pair[1]['parsed']['rounds']):
                    yes=[any((key(x) if callable(key) else x[key])==label for x in r[kind]) for r in [a,b]]
                    counts['both' if all(yes) else 'a_only' if yes[0] else 'b_only' if yes[1] else 'neither']+=1
            n=sum(counts.values());positive=2*counts['both']+counts['a_only']+counts['b_only']
            rows.append(dict(label=label,**{k:counts[k] for k in ['both','a_only','b_only','neither']},n=n,
                positive_agreement=2*counts['both']/positive if positive else None,
                raw_agreement=(counts['both']+counts['neither'])/n if n else None))
        return rows
    events=[]
    for item,pair in pairs:
        a,b=[{(t['round'],t['type']) for t in row['parsed']['transitions']} for row in pair]
        events.append(dict(id=item['id'],challenge=item['challenge'],both=len(a&b),a_only=len(a-b),b_only=len(b-a),
            a_events=sorted(a),b_events=sorted(b)))
    ledger=[json.loads(x) for x in (base/'attempts.jsonl').read_text().splitlines()]
    settled={x['reservation_id']:x for x in ledger if x['event']=='settle'}
    unresolved=[x for x in ledger if x['event']=='reserve' and x['reservation_id'] not in settled]
    pricing=read(OUT/'config.json')['pricing'];pin=float(pricing['prompt']);pout=float(pricing['completion'])
    costs=[charged_cost(x['usage'],pin,pout) for x in outputs]
    by_cohort={}
    manifest=read(base.parent/'corpus_manifest.json')
    corpus_counts=Counter()
    for run in manifest['runs']:corpus_counts[run['cohort']]+=len(run['models'])
    for item in sample:
        rows=[x for x in outputs if x['id']==item['id'] and x['usage'].get('cost') is not None]
        if rows and not item['challenge']:by_cohort.setdefault(item['cohort'],[]).extend(charged_cost(x['usage'],pin,pout) for x in rows)
    projection=sum(corpus_counts[c]*sum(vals)/len(vals) for c,vals in by_cohort.items())
    result=dict(trajectories=len(sample),outputs=len(outputs),valid=len(outputs)-len(invalid),invalid=invalid,pending=2*len(sample)-len(outputs),
        paired_valid_trajectories=len(pairs),provider_reported_cost=sum(costs),unresolved_reservations=unresolved,
        accounted_cost=sum(x['charged_or_reserved'] for x in ledger if x['event'] in ['settle','cost_correction'])+sum(x['bound'] for x in unresolved),
        tokens={k:sum(x['usage'].get(k,0) for x in outputs) for k in ['prompt_tokens','completion_tokens','total_tokens']},
        reasoning_tokens=sum((x['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens',0) for x in outputs),
        finish_reasons=dict(Counter(x['finish_reason'] for x in outputs)),
        features=agreement('features',FEATURES,'f'),operators=agreement('operators',OPS,'op'),transitions=events,
        feature_mode_endorsement=agreement('features',[(f,m,e) for f in FEATURES for m in ['advice','narration','hypothetical','counterfactual'] for e in ['endorsed','rejected','unclear']],lambda x:(x['f'],x['mode'],x['endorsement'])),
        transition_metric_caveat='Round/type event-label overlap, not semantic agreement on the same rule.',
        projection_missing_cohorts=sorted(set(corpus_counts)-set(by_cohort)),
        projected_full_single_pass_cohort_weighted_usd=projection if set(corpus_counts)==set(by_cohort) else None,
        partial_covered_cohorts_projection_usd=projection,
        projection_caveat='Tiny purposively balanced calibration; broad-coverage subset only; one pass, no attribution or validation/repair overhead. Not a quote or confidence interval.')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
