"""Pack Claude agents' written game reasoning into blind reading batches (no model calls).

Scope (informed negative noise, range 1, only):
- every game-only run with a Claude agent (Sonnet 4.5, Opus 5, Opus 5.5);
- every Claude agent in the 70% myth-run sample already read in full
  (docs/research/narrative_evolution_20261003/full_reading_70/selection.json).
Scripted defectors are skipped (their moves are forced and carry no text).
Each unit is one agent's game responses for rounds 1-10, shuffled and given a
blind ID so readers cannot tell game-only from myth runs. key.json maps back.

    python analyses/narrative_game_reasoning_prepare.py --out <dir> --batches 10
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs/research/narrative_evolution_20261003/corpus_manifest.json'
SELECTION = ROOT / 'docs/research/narrative_evolution_20261003/full_reading_70/selection.json'
SEED = 2026100572


def informed(meta):
    n = meta.get('noise_config') or {}
    return n.get('direction') == 'negative' and n.get('inform_agents') is True and n.get('range') == 1.0


def game_turns(state, agent):
    out = []
    for h in state['agents'][agent]['interaction_history']:
        md = h.get('metadata') or {}
        if md.get('task') != 'game':
            continue
        resp = h.get('response') or {}
        if resp.get('response_source') == 'forced_zero':
            continue
        out.append((md.get('round'), md.get('role_label') or md.get('role'), str(resp.get('content') or '').strip()))
    return sorted(out, key=lambda x: x[0] or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--batches', type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(MANIFEST))['runs']
    sel = json.load(open(SELECTION))
    myth_runs = {r['path']: r for r in sel['runs']}

    units = []
    for r in manifest:
        if not informed(r['metadata']):
            continue
        if r['task_order'] == 'game':
            run_id, chosen = None, None
        elif r['path'] in myth_runs:
            run_id, chosen = myth_runs[r['path']]['run_id'], myth_runs[r['path']]
        else:
            continue
        raw = (ROOT / r['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == r['sha256'], r['path']
        state = json.loads(raw)
        defectors = set(r['metadata'].get('defector_agent_ids') or [])
        comp = r['path'].split('/')[5]
        for agent, v in state['agents'].items():
            model = v['model'].split('/')[-1]
            if 'claude' not in model or agent in defectors:
                continue
            turns = game_turns(state, agent)
            if not turns:
                continue
            units.append(dict(path=r['path'], agent=agent, model=model, task_order=r['task_order'],
                              cohort=r['cohort'], num_agents=r['metadata']['num_agents'], composition=comp,
                              defector_run=bool(defectors),
                              random_defection=r['metadata'].get('random_defection_probability') or 0,
                              myth_tid=f"{run_id}-{agent}" if run_id else None, turns=turns))

    rng = random.Random(SEED)
    rng.shuffle(units)
    key = {}
    bodies = []
    for i, u in enumerate(units):
        gid = f"G{i:04d}"
        key[gid] = {k: v for k, v in u.items() if k != 'turns'}
        lines = [f"===== {gid} ====="]
        for rd, role, text in u['turns']:
            lines.append(f"[R{rd} | {role}]")
            lines.append(text)
        bodies.append('\n'.join(lines) + '\n\n')
    loads = [[0, b, []] for b in range(args.batches)]
    for body in sorted(bodies, key=len, reverse=True):
        loads.sort(key=lambda x: x[0])
        loads[0][0] += len(body)
        loads[0][2].append(body)
    for total, b, chunk in sorted(loads, key=lambda x: x[1]):
        (out / f"game_{b:02d}.txt").write_text(''.join(sorted(chunk)))
        print(b, len(chunk), f"{total:,}")
    json.dump(key, open(out / 'key.json', 'w'), indent=0)
    from collections import Counter
    print('units', len(units), Counter((u['model'], u['task_order']) for u in units))


if __name__ == '__main__':
    main()
