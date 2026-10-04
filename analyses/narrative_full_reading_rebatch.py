"""Restrict the 70% full-text selection to informed negative noise (range 1) and rebatch the uncoded runs.

Keeps the original run IDs (F###) so trajectories already coded stay valid.
No model calls.

    python analyses/narrative_full_reading_rebatch.py --out <dir> --batches N
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path

import narrative_full_reading_prepare as prep

ROOT = prep.ROOT
DIR = ROOT / 'docs/research/narrative_evolution_20261003/full_reading_70'


def informed(meta):
    n = meta.get('noise_config') or {}
    return n.get('direction') == 'negative' and n.get('inform_agents') is True and n.get('range') == 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--batches', type=int, required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sel = json.load(open(DIR / 'selection_all_noise.json'))
    manifest = {r['path']: r for r in json.load(open(prep.MANIFEST))['runs']}
    keep = [r for r in sel['runs'] if informed(manifest[r['path']]['metadata'])]
    coded = set()
    for f in glob.glob(str(DIR / 'codes/full_*_part*.jsonl')):
        for line in open(f):
            try:
                coded.add(json.loads(line).get('tid'))
            except json.JSONDecodeError:
                pass
    todo = [r for r in keep if not all(a['trajectory_id'] in coded for a in r['agents'])]
    bodies = []
    for r in todo:
        raw = (ROOT / r['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == r['sha256']
        state = json.loads(raw)
        lines = [f"===== RUN {r['run_id']} =====",
                 f"set: {r['cohort']} | task order: {r['task_order']} | agents: {r['num_agents']} | composition: {r['composition']}",
                 f"defector agents (forced to send/return $0, not told): {r['defectors'] or 'none'} | random defection probability: {r['random_defection']}",
                 f"file: {r['path']}"]
        for a in r['agents']:
            if a['trajectory_id'] in coded:
                continue
            lines.append(f"\n----- TRAJECTORY {a['trajectory_id']} | {a['model']} | {'SCRIPTED DEFECTOR' if a['defector'] else 'standard'} -----")
            play = prep.play_lines(state, a['agent'])
            for row in state['conversation_history']:
                text = (row.get('myths') or {}).get(a['agent'])
                if text is None:
                    continue
                lines.append(f"[R{row['round']}] (play this round: {play.get(row['round'], 'n/a')})")
                lines.append(str(text).strip())
        bodies.append((r['run_id'], '\n'.join(lines) + '\n'))
    loads = [[0, b, []] for b in range(args.batches)]
    for rid, body in sorted(bodies, key=lambda x: -len(x[1])):
        loads.sort(key=lambda x: x[0])
        loads[0][0] += len(body)
        loads[0][2].append((rid, body))
    for total, b, chunk in sorted(loads, key=lambda x: x[1]):
        (out / f"batch_{b:02d}.txt").write_text(''.join(body for _, body in sorted(chunk)))
        print(b, len(chunk), 'runs', f"{total:,}")
    sel2 = dict(sel, runs=keep, runs_chosen=len(keep), trajectories_chosen=sum(len(r['agents']) for r in keep),
                filter='informed negative noise, range 1 (excludes figure2 no-noise/uninformed and range-2 bridge)')
    sel2.pop('batches', None)
    json.dump(sel2, open(DIR / 'selection.json', 'w'), indent=1)
    print('kept runs', len(keep), 'trajectories', sel2['trajectories_chosen'], 'already coded', len(coded & {a['trajectory_id'] for r in keep for a in r['agents']}),
          'to read', sum(len(r['agents']) for r in todo) - len(coded & {a['trajectory_id'] for r in todo for a in r['agents']}),
          'chars', f"{sum(len(b) for _, b in bodies):,}")


if __name__ == '__main__':
    main()
