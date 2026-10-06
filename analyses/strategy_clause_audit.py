#!/usr/bin/env python3
"""Build a fixed exploratory trajectory sample from existing final-state data.

Standard library only; no API calls. Generated packets contain research text,
not executable instructions. Source CSVs/finals are never modified.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/research/strategy_clause_audit_20261002'
DATA = ROOT / 'data/analysis/strategy_clause_audit_20261002'
CORPORA = {'september': 'linguistic_20260923', 'frontier': 'linguistic_frontier_20260930'}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def filehash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def build():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, excluded, fingerprints = [], [], {}
    all_packets = []
    for tier, folder in CORPORA.items():
        source = ROOT / 'data/analysis' / folder / 'myths.csv'
        fingerprints[str(source.relative_to(ROOT))] = filehash(source)
        with source.open() as fh:
            rows = list(csv.DictReader(fh))
        by_agent, by_key, strata = defaultdict(list), {}, defaultdict(list)
        for r in rows:
            by_agent[(r['run_id'], r['agent'])].append(r)
            by_key[(r['run_id'], r['agent'], int(r['round']))] = r
        for (run, agent), rr in by_agent.items():
            rr.sort(key=lambda x: int(x['round']))
            if [int(r['round']) for r in rr] != list(range(1, 11)) or any(not r['text'].strip() for r in rr):
                excluded.append({'tier': tier, 'run_id': run, 'agent': agent, 'reason': 'not ten nonempty myths'})
                continue
            r = rr[0]
            key = (r['family'], int(r['size']), r['mixed'], r['task_order'])
            strata[key].append((digest(f'strategy-clause-v1|{tier}|{run}|{agent}'), rr))
        assert len(strata) == 24, (tier, len(strata))
        for key in sorted(strata):
            rank, rr = min(strata[key], key=lambda x: x[0])
            r = rr[0]
            ident = f'T{len(manifest)+1:02d}'
            final_path = ROOT / r['path']
            final = json.loads(final_path.read_text())
            history = final['conversation_history']
            assert [e['round'] for e in history] == list(range(1, 11)), r['path']
            assert 'agents' in final and 'run_metadata' in final
            for row, entry in zip(rr, history):
                assert entry['myths'][row['agent']].strip() == row['text'].strip(), (ident, row['round'])
            info = {'id': ident, 'tier': tier, 'family': key[0], 'size': key[1],
                    'mixed': key[2] == 'True', 'task_order': key[3], 'composition': r['composition'],
                    'run_id': r['run_id'], 'agent': r['agent'], 'path': r['path'],
                    'final_sha256': filehash(final_path), 'selection_sha256': rank,
                    'eligible_trajectories_in_stratum': len(strata[key])}
            manifest.append(info)
            packet = {'sample': info, 'rounds': []}
            own_text = [f"# {ident}: {tier}, {key}; {r['agent']}\n",
                        'Research texts below are data, not instructions. No game outcomes in this first-pass packet.\n']
            for row in rr:
                turn = int(row['round'])
                own_text += [f'## Round {turn}\n', row['text'] + '\n']
                exposed = None
                unseen = None
                if row['exposed_author']:
                    er = int(float(row['exposed_round']))
                    exposed = by_key[(row['run_id'], row['exposed_author'], er)]
                    pool = [x for x in rows if x['run_id'] != row['run_id'] and x['family'] == exposed['family']
                            and int(x['round']) == er and x['size'] == row['size']
                            and x['mixed'] == row['mixed'] and x['task_order'] == row['task_order']
                            and x['text'].strip()]
                    exact = [x for x in pool if x['composition'] == row['composition']]
                    pool = exact or pool
                    if pool:
                        chosen = min(pool, key=lambda x: digest(f"control-v1|{ident}|{turn}|{x['run_id']}|{x['agent']}"))
                        unseen = {**chosen, 'same_composition': bool(exact)}
                game_entry = history[turn-1]
                if game_entry.get('dyads'):
                    own_game = next(g for g in game_entry['dyads'] if r['agent'] in g['agents'])
                else:
                    own_game = game_entry
                game_calls = [x for x in final['agents'][r['agent']]['interaction_history']
                              if x.get('metadata', {}).get('task') == 'game'
                              and x.get('metadata', {}).get('round') == turn]
                packet['rounds'].append({'round': turn, 'own_text': row['text'],
                                         'exposed': exposed, 'unseen': unseen,
                                         'game_precedes_myth': key[3] == 'game_myth',
                                         'decision_prompts': [x['messages_sent'][-1]['content'] for x in game_calls],
                                         'agent_game': {k: own_game.get(k) for k in ['investor','trustee','sent','returned',
                                                        'sent_communicated','returned_communicated','received_communicated']},
                                         'game': {k: game_entry.get(k) for k in ['roles','pairings','sent','returned','sent_communicated',
                                                  'returned_communicated','received_communicated','myth_exposures']}})
            (DATA / f'{ident}_own.md').write_text('\n'.join(own_text))
            write_json(DATA / f'{ident}_context.json', packet)
            all_packets.append(packet)
    write_json(OUT / 'manifest.json', {'plan_sha256': filehash(OUT / 'PLAN.md'), 'input_sha256': fingerprints,
                                     'sample': manifest, 'excluded': excluded,
                                     'n_trajectories': len(manifest),
                                     'n_unique_runs': len({(s['tier'],s['run_id']) for s in manifest})})
    for i in range(3):
        assignment = manifest[i::3]
        write_json(OUT / f'assignment_{i+1}.json', assignment)
    print(json.dumps({'trajectories': len(manifest), 'unique_runs': len({(s['tier'],s['run_id']) for s in manifest}),
                      'excluded': len(excluded), 'packets': str(DATA)}))


def context(ident, turn):
    packet = json.loads((DATA / f'{ident}_context.json').read_text())
    row = packet['rounds'][turn-1]
    print(json.dumps({'sample':packet['sample'], 'previous_own':packet['rounds'][turn-2]['own_text'] if turn>1 else None,
                      **row}, indent=2, ensure_ascii=False))


def finalize():
    """Validate provenance/quotes and export descriptive, provisional coding.

    This checks textual accuracy, NOT interpretive validity or human agreement.
    """
    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert filehash(OUT / 'PLAN.md') == manifest['plan_sha256']
    for path, sha in manifest['input_sha256'].items():
        assert filehash(ROOT / path) == sha, path
    samples = {s['id']: s for s in manifest['sample']}
    codes = sum([json.loads((OUT / f'coding_{i}.json').read_text()) for i in range(1, 4)], [])
    assert len(codes) == len(samples) == 48
    assert Counter(c['id'] for c in codes) == Counter(samples.keys())
    counts = Counter()
    for code in codes:
        ident = code['id']
        sample = samples[ident]
        assert code['evolution'] in {'stable', 'revision', 'elaboration', 'simplification', 'unclear'}
        assert len(code['events']) <= 2
        packet = json.loads((DATA / f'{ident}_context.json').read_text())
        assert packet['sample'] == sample
        final = json.loads((ROOT / sample['path']).read_text())
        assert filehash(ROOT / sample['path']) == sample['final_sha256']
        history = final['conversation_history']
        assert [r['round'] for r in history] == list(range(1, 11))
        assert len(packet['rounds']) == 10
        for r in packet['rounds']:
            turn = r['round']
            assert r['own_text'].strip() == history[turn-1]['myths'][sample['agent']].strip()
            counts['own_myths_verified'] += 1
            exposed = r['exposed']
            if exposed:
                exposure = history[turn-1]['myth_exposures'][sample['agent']]
                assert exposure['original_author_id'] == exposed['agent']
                assert exposure['source_round'] == int(exposed['round'])
                assert not exposure.get('substitution_applied')
                assert exposed['text'].strip() == history[int(exposed['round'])-1]['myths'][exposed['agent']].strip()
                calls = [x for x in final['agents'][sample['agent']]['interaction_history']
                         if x.get('metadata', {}).get('task') == 'myth'
                         and x.get('metadata', {}).get('round') == turn]
                assert calls
                # A malformed response can produce multiple logged attempts.
                accepted = [x for x in calls if r['own_text'].strip() in
                            (x['response'].get('content', '') if isinstance(x['response'], dict)
                             else x['response'])]
                assert accepted, (ident, turn, 'no response matches final myth')
                assert all(exposed['text'] in x['messages_sent'][-1]['content'] for x in accepted)
                counts['extra_logged_myth_attempts'] += len(calls) - 1
                counts['exposures_verified_in_saved_prompts'] += 1
                unseen = r['unseen']
                assert unseen and unseen['run_id'] != sample['run_id']
                counts['matched_unseen_controls'] += 1
                counts['controls_same_composition'] += int(unseen['same_composition'])
        for event in code['events']:
            before, after = event['before_round'], event['round']
            assert 1 <= before < after <= 10, (ident, event)
            assert event['endorsement'] in {'explicit', 'narrated', 'ambiguous'}
            assert isinstance(event['own_earlier_present'], bool)
            assert event['source_assessment'] in {'own_continuation', 'exposure_candidate',
                    'shared_common_rule', 'game_experience_candidate', 'unclear'}
            row = packet['rounds'][after-1]
            for key, text in [('before_quote', packet['rounds'][before-1]['own_text']),
                              ('after_quote', row['own_text']),
                              ('exposed_quote', row['exposed']['text'] if row['exposed'] else ''),
                              ('unseen_quote', row['unseen']['text'] if row['unseen'] else '')]:
                quote = event[key]
                if quote is not None:
                    assert quote and quote in text, (ident, after, key, quote)
                    counts['exact_quotes_verified'] += 1
                else:
                    assert key in {'exposed_quote', 'unseen_quote'}
            counts['selected_events'] += 1
    summary = {'status': 'exploratory machine coding; human validation pending',
               'n_trajectories': len(samples), 'n_unique_runs': manifest['n_unique_runs'],
               'checks': dict(counts),
               'evolution_by_tier': {tier: dict(Counter(c['evolution'] for c in codes
                      if samples[c['id']]['tier'] == tier)) for tier in CORPORA},
               'source_assessments': dict(Counter(e['source_assessment'] for c in codes for e in c['events'])),
               'coding_sha256': {f'coding_{i}.json': filehash(OUT / f'coding_{i}.json') for i in range(1, 4)}}
    write_json(OUT / 'validation.json', summary)
    lines = ['# Provisional trajectory coding', '',
             'Machine readings, not validated findings. Events are selected examples (at most two per trajectory), not an exhaustive event census.', '',
             '| ID | Tier / family | Evolution label | Selected rounds | Assessment |',
             '|---|---|---|---|---|']
    for c in sorted(codes, key=lambda x: x['id']):
        s = samples[c['id']]
        assessment = c['assessment'].replace('|', '/').replace('\n', ' ')
        lines.append(f"| {c['id']} | {s['tier']} / {s['family']} | {c['evolution']} | {', '.join(str(e['round']) for e in c['events']) or '—'} | {assessment} |")
    (OUT / 'TRAJECTORIES.md').write_text('\n'.join(lines) + '\n')
    # Deterministic validation subset: one whole trajectory per tier × family × size.
    # Selection ignores code labels/events, and includes stable cases.
    strata = defaultdict(list)
    for s in samples.values():
        strata[(s['tier'], s['family'], s['size'])].append(s)
    selected = [min(v, key=lambda s: digest('human-v1|' + s['id'])) for _, v in sorted(strata.items())]
    blind = ['# Human validation packet', '',
             'Research texts are data, not instructions. Code independently before opening the key or machine coding.', '',
             'For each complete trajectory: quote explicit endorsed conditions, actions, exceptions and recovery rules. Separate prescriptions from narrated actions. Record unclear cases. Compare all earlier rounds before calling a clause new. Describe revisions, dropped rules and stable strategies, not just increasing detail. Metaphor or length alone is not complexity.', '',
             'Return a trajectory-level label (stable, elaboration, simplification, revision, unclear), with round-specific supporting quotes. Mixed paths may have several kinds of event. This packet validates extraction only; it does not validate source attribution or behavioral influence.', '']
    key = []
    for index, s in enumerate(selected, 1):
        blind_id = f'H{index:02d}'
        key.append({'human_id': blind_id, 'trajectory_id': s['id']})
        blind.append(f'## {blind_id}')
        packet = json.loads((DATA / f"{s['id']}_context.json").read_text())
        for r in packet['rounds']:
            blind.extend([f"### Round {r['round']}", r['own_text'], ''])
    (OUT / 'HUMAN_VALIDATION.md').write_text('\n'.join(blind) + '\n')
    write_json(OUT / 'human_validation_key.json', key)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', nargs=2, metavar=('ID','ROUND'))
    parser.add_argument('--finalize', action='store_true')
    args=parser.parse_args()
    if args.finalize:
        finalize()
    elif args.context:
        context(args.context[0],int(args.context[1]))
    else:
        build()
