"""Verify quotes and compare Claude game reasoning across game-only and myth runs (no model calls).

Reads docs/research/narrative_evolution_20261003/game_reasoning/codes/*.jsonl, the blind key
(--key), and the full-text myth codes; checks every quote against the cited round's game
response; writes game_reasoning/summary.json and prints the comparison tables.

    python analyses/narrative_game_reasoning_tally.py --key <dir>/key.json
"""
import argparse
import collections
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'docs/research/narrative_evolution_20261003/game_reasoning'
MYTH = ROOT / 'docs/research/narrative_evolution_20261003/full_reading_70'
PRESCRIBED = {'graded_never_zero', 'total_exclusion', 'measure_for_measure', 'withdraw_unspecified'}
SHORT = {'claude-sonnet-4.5': 'Sonnet 4.5', 'claude-opus-5': 'Opus 5', 'claude-opus-5-5': 'Opus 5.5'}
ORDER = {'game': 'game only', 'game_myth': 'Game→Myth', 'myth_game': 'Myth→Game'}


def norm(s):
    s = str(s)
    for a, b in (('“', '"'), ('”', '"'), ('’', "'"), ('‘', "'"), ('—', '-'), ('–', '-')):
        s = s.replace(a, b)
    return re.sub(r'\s+', ' ', s).strip()


_cache = {}


def game_text(path, agent, rd):
    if path not in _cache:
        _cache[path] = json.load(open(ROOT / path))
    parts = []
    for h in _cache[path]['agents'][agent]['interaction_history']:
        md = h.get('metadata') or {}
        if md.get('task') == 'game' and md.get('round') == rd:
            parts.append(str((h.get('response') or {}).get('content') or ''))
    return norm(' '.join(parts))


def load(pattern):
    rows = []
    for f in sorted(glob.glob(str(pattern))):
        for line in open(f):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', required=True)
    args = ap.parse_args()
    key = json.load(open(args.key))
    recs = {r['gid']: r for r in load(DIR / 'codes/*.jsonl') if 'gid' in r}
    missing = sorted(set(key) - set(recs))
    myth = {r['tid']: r for r in load(MYTH / 'codes/*.jsonl') if 'tid' in r}

    rows, qok, qall = [], 0, 0
    for gid, rec in recs.items():
        k = key.get(gid)
        if not k:
            continue
        ok = []
        for q in rec.get('quotes') or []:
            t = norm(q.get('text', '')).strip('"')
            if t and t in game_text(k['path'], k['agent'], q.get('round')):
                ok.append(q)
        qok += len(ok)
        qall += len(rec.get('quotes') or [])
        sup = {q.get('code') for q in ok}
        s = rec.get('sanction', 'none')
        themes = {t for t in rec.get('themes') or [] if t in sup or 'themes' in sup}
        m = myth.get(k['myth_tid']) if k.get('myth_tid') else None
        rows.append(dict(model=SHORT[k['model']], order=ORDER[k['task_order']],
                         size='dyad' if k['num_agents'] == 2 else '8-agent',
                         mix='mixed' if '+' in k['composition'] else 'homogeneous',
                         defection='scripted defectors' if k['defector_run'] else ('random' if k['random_defection'] else 'none'),
                         text=bool(rec.get('text_present')),
                         sanction=s if s == 'none' or s in sup or 'sanction' in sup else 'unsupported',
                         prescribed=s in PRESCRIBED and (s in sup or 'sanction' in sup),
                         applied=bool(rec.get('applied_sanction')) and ('applied_sanction' in sup or bool(ok)),
                         way_back=rec.get('way_back'), forgive=bool(rec.get('forgive_count')) and bool(ok),
                         rule_change=rec.get('rule_change'), themes=themes,
                         myth_sanction=(m or {}).get('sanction'), myth_themes=set((m or {}).get('themes') or [])))

    def pct(v, f):
        return dict(k=sum(1 for r in v if f(r)), n=len(v))
    metrics = {
        'reasoning text present': lambda r: r['text'],
        'states a sanction': lambda r: r['prescribed'],
        'graded, never to zero': lambda r: r['sanction'] == 'graded_never_zero',
        'total exclusion': lambda r: r['sanction'] == 'total_exclusion',
        'measure for measure': lambda r: r['sanction'] == 'measure_for_measure',
        'withdraw (unspecified)': lambda r: r['sanction'] == 'withdraw_unspecified',
        'applies a sanction in a move': lambda r: r['applied'],
        'counted forgiveness': lambda r: r['forgive'],
    }
    themes = sorted({t for r in rows for t in r['themes']})
    for t in themes:
        metrics[f'theme: {t}'] = (lambda t: lambda r: t in r['themes'])(t)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['model'], r['order'])].append(r)
        groups[(r['model'], r['order'], r['size'], r['defection'])].append(r)
    table = {' | '.join(g): {m: pct(v, f) for m, f in metrics.items()} for g, v in sorted(groups.items())}

    # within-agent link: does the myth's sanction kind reappear in the same agent's game reasoning?
    link = collections.defaultdict(collections.Counter)
    for r in rows:
        if r['order'] == 'game only' or r['myth_sanction'] is None:
            continue
        ms = r['myth_sanction'] if r['myth_sanction'] in PRESCRIBED else 'no sanction in myth'
        link[(r['model'], ms)]['n'] += 1
        link[(r['model'], ms)]['game states a sanction'] += r['prescribed']
        link[(r['model'], ms)]['same kind in game'] += r['sanction'] == r['myth_sanction']
    report = dict(units=len(rows), expected=len(key), missing=len(missing), quotes=dict(ok=qok, all=qall),
                  by_condition=table,
                  myth_to_game={' | '.join(k): dict(v) for k, v in sorted(link.items())})
    DIR.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(DIR / 'summary.json', 'w'), indent=1)
    print(json.dumps({k: report[k] for k in ('units', 'expected', 'missing', 'quotes')}))
    show = ['reasoning text present', 'states a sanction', 'graded, never to zero', 'total exclusion',
            'measure for measure', 'applies a sanction in a move', 'counted forgiveness'] + [f'theme: {t}' for t in themes]
    for g, m in table.items():
        if g.count('|') == 1:
            print(f"\n## {g} (n={m['states a sanction']['n']})")
            print('; '.join(f"{c}: {100 * m[c]['k'] / max(1, m[c]['n']):.0f}%" for c in show))
    print('\n## Myth sanction -> same agent\'s game reasoning')
    for k, v in report['myth_to_game'].items():
        print(k, v)


if __name__ == '__main__':
    main()
