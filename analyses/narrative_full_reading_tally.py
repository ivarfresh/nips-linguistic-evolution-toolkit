"""Verify quotes and tally the full-text 70% close reading (no model calls).

Inputs:
  docs/research/narrative_evolution_20261003/full_reading_70/selection.json  (informed-noise 70% sample)
  docs/research/narrative_evolution_20261003/full_reading_70/codes/*.jsonl    (reader codes)
  docs/research/narrative_evolution_20261003/full_ladder_reading/blind_*.jsonl (blind defector reading)
  <scratch>/ladderread/{blind_key,trajectory_index}.json via --blind-src
Writes full_reading_70/summary.json and prints the headline tables.

    python analyses/narrative_full_reading_tally.py --blind-src <dir>
"""
import argparse
import collections
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'docs/research/narrative_evolution_20261003/full_reading_70'
BLIND = ROOT / 'docs/research/narrative_evolution_20261003/full_ladder_reading'
PRESCRIBED = {'graded_never_zero', 'total_exclusion', 'measure_for_measure', 'withdraw_unspecified'}
SHORT = {'claude-sonnet-4.5': 'Sonnet 4.5', 'gemini-3.7-flash': 'Gemini 3.7 Flash', 'gpt-5-nano': 'GPT-5 Nano',
         'claude-opus-5': 'Opus 5', 'gemini-3.1-pro-preview': 'Gemini 3.1 Pro', 'gpt-5.6-sol': 'GPT-5.6 Sol',
         'claude-opus-5-5': 'Opus 5.5', 'gpt-6-sol': 'GPT-6 Sol'}


def norm(s):
    s = str(s)
    for a, b in (('“', '"'), ('”', '"'), ('’', "'"), ('‘', "'"), ('—', '-'), ('–', '-')):
        s = s.replace(a, b)
    return re.sub(r'\s+', ' ', s).strip()


_cache = {}


def round_text(path, agent, rd):
    if path not in _cache:
        _cache[path] = json.load(open(ROOT / path))
    for row in _cache[path]['conversation_history']:
        if row['round'] == rd:
            return norm((row.get('myths') or {}).get(agent, ''))
    return ''


def verified(rec, path, agent):
    ok = []
    for q in rec.get('quotes') or []:
        t = norm(q.get('text', '')).strip('"')
        if t and t in round_text(path, agent, q.get('round')):
            ok.append(q)
    return ok


def load(pattern):
    out = []
    for f in sorted(glob.glob(str(pattern))):
        for line in open(f):
            if line.strip():
                out.append(json.loads(line))
    return out


def rate(rows, pred):
    n = len(rows)
    k = sum(1 for r in rows if pred(r))
    return dict(k=k, n=n, share=round(k / n, 3) if n else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blind-src', required=True)
    args = ap.parse_args()
    sel = json.load(open(DIR / 'selection.json'))
    info = {}
    for r in sel['runs']:
        for a in r['agents']:
            info[a['trajectory_id']] = dict(
                path=r['path'], agent=a['agent'], run=r['run_id'], cohort=r['cohort'],
                era='frontier' if r['cohort'].startswith('frontier') else 'september',
                order={'game_myth': 'Game→Myth', 'myth_game': 'Myth→Game'}[r['task_order']],
                size='dyad' if r['num_agents'] == 2 else '8-agent',
                mix='mixed' if '+' in r['composition'] else 'homogeneous',
                defection='scripted defectors' if r['defectors'] else ('random defection' if r['random_defection'] else 'none'),
                model=SHORT.get(a['model'], a['model']), defector=a['defector'])

    rows, quotes_ok, quotes_all = [], 0, 0
    for rec in load(DIR / 'codes/*.jsonl'):
        tid = rec.get('tid')
        if tid not in info:
            continue
        i = info[tid]
        ok = verified(rec, i['path'], i['agent'])
        quotes_ok += len(ok)
        quotes_all += len(rec.get('quotes') or [])
        supported = {q.get('code') for q in ok}
        s = rec.get('sanction', 'none')
        rows.append(dict(i, tid=tid,
                         sanction=s if s in ('none',) or s in supported or 'sanction' in supported else 'unsupported',
                         prescribed=s in PRESCRIBED and (s in supported or 'sanction' in supported),
                         narrated=bool(rec.get('punishment_narrated')),
                         way_back=rec.get('way_back'), forgive_count=bool(rec.get('forgive_count')),
                         villain=bool(rec.get('villain_voiced')), first_sanction_round=rec.get('first_sanction_round'),
                         rule_change=rec.get('rule_change'), styles=set(rec.get('styles') or []),
                         themes=set(rec.get('themes') or []), self_narration=rec.get('self_narration', 'none')))
    assert len(rows) == sum(len(r['agents']) for r in sel['runs']), (len(rows),)

    std = [r for r in rows if not r['defector']]
    report = dict(trajectories=len(rows), runs=len(sel['runs']), quotes=dict(ok=quotes_ok, all=quotes_all))

    def table(group_key, subset, metrics):
        out = {}
        groups = collections.defaultdict(list)
        for r in subset:
            groups[group_key(r)].append(r)
        for g in sorted(groups):
            out[g] = {m: rate(groups[g], f) for m, f in metrics.items()}
        return out

    sanction_metrics = {
        'prescribes a sanction': lambda r: r['prescribed'],
        'graded, never to zero': lambda r: r['sanction'] == 'graded_never_zero',
        'withdraw (unspecified)': lambda r: r['sanction'] == 'withdraw_unspecified',
        'total exclusion': lambda r: r['sanction'] == 'total_exclusion',
        'measure for measure': lambda r: r['sanction'] == 'measure_for_measure',
        'punishment only rejected': lambda r: r['sanction'] == 'rejected_only',
        'way back when sanctioning': lambda r: r['prescribed'] and r['way_back'] == 'yes',
        'counted forgiveness': lambda r: r['forgive_count'],
        'punishment narrated in story': lambda r: r['narrated'],
        'villain voices punishment': lambda r: r['villain'],
    }
    change_metrics = {c: (lambda c: lambda r: r['rule_change'] == c)(c) for c in
                      ['stable', 'refines', 'frozen', 'softens', 'hardens', 'dissolved', 'reverses']}
    style_names = sorted({s for r in rows for s in r['styles']})
    theme_names = sorted({t for r in rows for t in r['themes']})
    report['sanction_by_model'] = table(lambda r: (r['era'], r['model']), std, sanction_metrics)
    report['sanction_by_condition'] = {
        k: table(f, std, sanction_metrics) for k, f in {
            'era x order': lambda r: (r['era'], r['order']),
            'era x size': lambda r: (r['era'], r['size']),
            'era x mix': lambda r: (r['era'], r['mix']),
            'era x defection': lambda r: (r['era'], r['defection']),
        }.items()}
    report['rule_change_by_model'] = table(lambda r: (r['era'], r['model']), std, change_metrics)
    report['styles_by_model'] = table(lambda r: (r['era'], r['model']), std,
                                      {s: (lambda s: lambda r: s in r['styles'])(s) for s in style_names})
    report['themes_by_model'] = table(lambda r: (r['era'], r['model']), std,
                                      {t: (lambda t: lambda r: t in r['themes'])(t) for t in theme_names})
    report['themes_by_condition'] = table(lambda r: (r['era'], r['order'], r['size']), std,
                                          {t: (lambda t: lambda r: t in r['themes'])(t) for t in theme_names})
    rounds = collections.defaultdict(list)
    for r in std:
        if r['prescribed'] and isinstance(r['first_sanction_round'], int):
            rounds[(r['era'], r['order'])].append(r['first_sanction_round'])
    report['first_sanction_round'] = {f'{k[0]} | {k[1]}': dict(n=len(v), mean=round(sum(v) / len(v), 2),
                                                               sd=round((sum((x - sum(v) / len(v)) ** 2 for x in v) / max(1, len(v) - 1)) ** .5, 2))
                                      for k, v in sorted(rounds.items())}
    defrows = [r for r in rows if r['defection'] == 'scripted defectors']
    report['self_narration_unblinded'] = table(
        lambda r: (r['era'], r['model'], 'DEFECTOR' if r['defector'] else 'standard'), defrows,
        {'withholding protagonist (any)': lambda r: r['self_narration'] != 'none',
         'confession or both': lambda r: r['self_narration'] in ('confession', 'both'),
         'justification or both': lambda r: r['self_narration'] in ('justification', 'both'),
         'fearful, corrected': lambda r: r['self_narration'] == 'fearful_corrected'})

    # shared rules in 8-agent runs
    runlines = {o['run']: o for o in load(DIR / 'codes/*.jsonl') if 'run' in o}
    eight = [r for r in sel['runs'] if r['num_agents'] > 2]
    sh = collections.defaultdict(list)
    for r in eight:
        o = runlines.get(r['run_id'])
        key = ('frontier' if r['cohort'].startswith('frontier') else 'september',
               'mixed' if '+' in r['composition'] else r['composition'].split('/')[-1])
        sh[key].append(bool(o and o.get('shared_rule')))
    report['shared_rule_8agent'] = {f'{k[0]} | {SHORT.get(k[1], k[1])}': dict(k=sum(v), n=len(v), coded=sum(1 for r in eight if r['run_id'] in runlines))
                                    for k, v in sorted(sh.items())}
    report['shared_rule_examples'] = [dict(run=k, phrase=o.get('phrase'), carriers=len(o.get('carriers') or []))
                                      for k, o in runlines.items() if o.get('shared_rule')][:60]

    # blind defector reading (all 240 defectors + 90 controls)
    key = json.load(open(Path(args.blind_src) / 'blind_key.json'))
    index = json.load(open(Path(args.blind_src) / 'trajectory_index.json'))
    brows = collections.defaultdict(list)
    bq = [0, 0]
    for rec in load(BLIND / 'blind_*.jsonl'):
        t = index[key[rec['bid']]]
        ok = verified(rec, t['path'], t['agent'])
        bq[0] += len(ok)
        bq[1] += len(rec.get('quotes') or [])
        sn = rec.get('self_narration', 'none')
        if sn != 'none' and not ok:
            sn = 'unsupported'
        brows[(SHORT.get(t['model'], t['model']), 'DEFECTOR' if t['defector'] else 'control')].append(sn)
    report['blind'] = dict(quotes=dict(ok=bq[0], all=bq[1]), by_model={
        f'{m} | {g}': dict(n=len(v), withholding_protagonist=sum(x not in ('none', 'unsupported') for x in v),
                           **collections.Counter(v)) for (m, g), v in sorted(brows.items())})
    def keyfix(o):
        if isinstance(o, dict):
            return {(' | '.join(k) if isinstance(k, tuple) else k): keyfix(v) for k, v in o.items()}
        return o
    json.dump(keyfix(report), open(DIR / 'summary.json', 'w'), indent=1, default=list)

    def show(title, t, cols):
        print(f'\n## {title}')
        for g, m in t.items():
            print(' | '.join([' / '.join(g) if isinstance(g, tuple) else g] +
                             [f"{c}: {m[c]['k']}/{m[c]['n']} ({m[c]['share']:.0%})" for c in cols]))
    print(json.dumps(dict(trajectories=len(rows), quotes=report['quotes'], blind_quotes=report['blind']['quotes'])))
    show('Sanctions by model (standard agents)', report['sanction_by_model'],
         ['prescribes a sanction', 'graded, never to zero', 'total exclusion', 'way back when sanctioning', 'punishment narrated in story', 'villain voices punishment'])
    for k, t in report['sanction_by_condition'].items():
        show(f'Sanctions by {k}', t, ['prescribes a sanction', 'graded, never to zero', 'total exclusion', 'punishment narrated in story'])
    show('Rule change by model', report['rule_change_by_model'], list(change_metrics))
    print('\n## First sanction round', json.dumps(report['first_sanction_round']))
    show('Self-narration, defector runs (unblinded readers)', report['self_narration_unblinded'],
         ['withholding protagonist (any)', 'confession or both', 'justification or both', 'fearful, corrected'])
    print('\n## Blind reading', json.dumps(report['blind'], indent=0))
    print('\n## Shared rule in 8-agent runs', json.dumps(report['shared_rule_8agent']))


if __name__ == '__main__':
    main()
