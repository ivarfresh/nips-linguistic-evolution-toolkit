"""Evaluate the frozen, source-first pilot reference; no API calls."""
import argparse
from collections import Counter
import narrative_feature_screen as screen
from narrative_feature_screen import OUT, BASE, FEATURES, read, save, validate

def status(row, feature):
    if not row['readable']:
        return 'missing'
    if any(feature in e['yes'] for e in row['evidence']):
        return 'present'
    if any(feature in e['unclear'] for e in row['evidence']):
        return 'unclear'
    if feature in row['absent']:
        return 'absent'
    return 'missing'

def main():
    global OUT,FEATURES
    ap=argparse.ArgumentParser();ap.add_argument('--validated-subset',action='store_true');ap.add_argument('--dialogue-only',action='store_true');ap.add_argument('--presence-only',action='store_true');args=ap.parse_args()
    original=OUT
    if args.validated_subset or args.dialogue_only:
        OUT=OUT/('dialogue' if args.dialogue_only else 'subset');FEATURES=read(OUT/'rubric.json')['features'];screen.FEATURES=FEATURES
    sample=read(OUT/'sample.json'); reference=read(original/'source_review.json')
    ref={x['id']:x for x in reference['cases']}; byid={x['id']:x for x in sample}
    inverse={v:k for k,v in reference['key'].items()}
    results=[]; predictions={}; errors=[]; ignored=[]
    for path in sorted((OUT/'pilot').glob('F*.json')):
        record=read(path); results.append(record)
        if record['parsed']:
            rows=[byid[x['id']] for x in record['parsed']['myths'] if x['id'] in byid]
            failures=validate(record['parsed'],rows)
        else:failures=['invalid JSON']
        if record['finish_reason']!='stop':failures.append('non-stop completion')
        if args.presence_only:
            ignored.extend({'file':path.name,'error':e} for e in failures if e.endswith(':form/stance'))
            failures=[e for e in failures if not e.endswith(':form/stance')]
        if failures:
            errors.append({'file':path.name,'errors':failures});continue
        for row in record['parsed']['myths']:
            if row['id'] in predictions:raise ValueError('duplicate output id')
            predictions[row['id']]=row
    scores=[]; differences=[]
    for feature in FEATURES:
        code=inverse[feature];counts=Counter();pairs=Counter()
        for rid,source in ref.items():
            expected='present' if code in source['yes'] else 'unclear' if code in source['unclear'] else 'absent'
            counts['reference_'+expected]+=1
            actual=status(predictions[rid],feature) if rid in predictions else 'missing'
            pairs[expected+'__'+actual]+=1
            if expected!=actual:differences.append(dict(id=rid,feature=feature,reference=expected,predicted=actual))
        tp=pairs['present__present'];fp=pairs['absent__present']
        fn=counts['reference_present']-tp
        precision=tp/(tp+fp) if tp+fp else None
        sensitivity=tp/(tp+fn) if tp+fn else None
        agreement=sum(pairs[s+'__'+s] for s in ['present','absent','unclear'])/len(ref)
        enough=counts['reference_present']>=3 and counts['reference_absent']>=3
        numeric_pass=bool(enough and precision is not None and sensitivity is not None and precision>=.85 and sensitivity>=.85 and agreement>=.85)
        scores.append(dict(feature=feature,counts=dict(counts),confusion=dict(pairs),precision=precision,
            sensitivity=sensitivity,three_state_agreement=agreement,numeric_gate_pass=numeric_pass,
            semantic_review_required=True))
    costs=sum(x['estimated_cost_usd'] for x in results)
    completion=sum(x['usage']['completion_tokens'] for x in results)
    prompt=sum(x['usage']['prompt_tokens'] for x in results)
    n=sum(len(x['parsed']['myths']) if x['parsed'] else 0 for x in results)
    # Provisional extrapolation from deliberately enriched sample; not a binding quote.
    corpus_n=read(BASE/'corpus_manifest.json')['totals']['myth_entries']
    projection=costs/len(sample)*corpus_n if len(results)==6 else None
    result=dict(sample_myths=len(sample),valid_myths=len(predictions),calls=len(results),technical_errors=errors,ignored_form_errors=ignored,
        complete=len(predictions)==len(sample) and not errors,scores=scores,differences=differences,
        estimated_pilot_cost_usd=costs,prompt_tokens=prompt,completion_tokens=completion,
        rough_standard_full_cost_usd=projection,rough_batch_full_cost_usd=projection/2 if projection else None,
        projection_caveat=f'{len(FEATURES)} selected categories, same six-myth packing; challenge-enriched sample, no repairs or inference uncertainty included.',
        interpretation='Source-first AI reference, not human reliability. Numeric pass alone does not authorize scaling.')
    save(OUT/('evaluation_presence_only.json' if args.presence_only else 'evaluation.json'),result)
    print('Valid myths',len(predictions),'of',len(sample),'cost estimate',round(costs,4))
    for x in scores:
        print(x['feature'], 'numeric_pass='+str(x['numeric_gate_pass']), 'precision='+str(x['precision']), 'sensitivity='+str(x['sensitivity']), 'agreement='+str(round(x['three_state_agreement'],3)))
    print(f'Rough full Batch cost, {len(FEATURES)} selected categories:',projection/2 if projection else 'pending')

if __name__=='__main__':main()
