"""Pack reading batches for the full-corpus close reading (no model calls).

Two products, written to --out:
1. ladder_XX.txt: the closing two sentences of every round for all 3,872
   myth trajectories (whole runs kept together), for rule coding.
2. blind_XX.txt: full myth text, round 1-10, for all 240 scripted defectors plus
   one randomly chosen standard agent from each defector run, with the
   defector flag hidden from the reader. blind_key.json maps blind IDs back.

    python analyses/narrative_ladder_reading_prepare.py --out <dir> --ladder-batches 16 --blind-batches 5
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs/research/narrative_evolution_20261003/corpus_manifest.json'
SEED = 2026100471


def closing(text, n=2):
    s = re.sub(r'\s+', ' ', str(text).strip())
    parts = re.split(r'(?<=[.!?"”*])\s+(?=[A-Z"“*])', s)
    return ' '.join(parts[-n:])


def balance(items, n, size):
    loads = [[0, b, []] for b in range(n)]
    for it in sorted(items, key=lambda x: -size(x)):
        loads.sort(key=lambda x: x[0])
        loads[0][0] += size(it)
        loads[0][2].append(it)
    return sorted(loads, key=lambda x: x[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--ladder-batches', type=int, default=16)
    ap.add_argument('--blind-batches', type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    runs = sorted((r for r in json.load(open(MANIFEST))['runs'] if r['myth_entries']), key=lambda r: r['path'])
    ladder_runs, blind_items, index = [], [], {}
    for i, r in enumerate(runs):
        raw = (ROOT / r['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == r['sha256'], r['path']
        state = json.loads(raw)
        meta = r['metadata']
        run_id = f"R{i:03d}"
        defectors = set(meta.get('defector_agent_ids') or [])
        comp = r['path'].split('/')[5]
        authors = sorted({a for row in state['conversation_history'] for a in (row.get('myths') or {})},
                         key=lambda a: int(a.split('_')[-1]))
        myths = {a: [(row['round'], (row.get('myths') or {}).get(a)) for row in state['conversation_history']
                     if (row.get('myths') or {}).get(a) is not None] for a in authors}
        lines = [f"===== RUN {run_id} | set {r['cohort']} | {r['task_order']} | {meta['num_agents']} agents | "
                 f"{comp} | scripted defectors: {sorted(defectors) or 'none'} | "
                 f"random defection: {meta.get('random_defection_probability') or 0} ====="]
        for a in authors:
            model = state['agents'][a]['model'].split('/')[-1]
            tid = f"{run_id}-{a}"
            index[tid] = dict(run_id=run_id, path=r['path'], cohort=r['cohort'], task_order=r['task_order'],
                              num_agents=meta['num_agents'], composition=comp, agent=a, model=model,
                              defector=a in defectors, defector_run=bool(defectors),
                              random_defection=meta.get('random_defection_probability') or 0)
            lines.append(f"--- {tid} | {model}{' | SCRIPTED DEFECTOR' if a in defectors else ''}")
            lines += [f"R{rd}: {closing(t)}" for rd, t in myths[a]]
        ladder_runs.append(('\n'.join(lines) + '\n\n'))
        if defectors:
            standard = [a for a in authors if a not in defectors]
            for a in sorted(defectors) + [rng.choice(standard)]:
                blind_items.append(dict(tid=f"{run_id}-{a}", myths=myths[a]))

    for total, b, chunk in balance(ladder_runs, args.ladder_batches, len):
        (out / f"ladder_{b:02d}.txt").write_text(''.join(sorted(chunk)))
        print('ladder', b, chunk.__len__(), 'runs', f"{total:,} chars")

    rng.shuffle(blind_items)
    key = {}
    for j, it in enumerate(blind_items):
        bid = f"B{j:03d}"
        key[bid] = it['tid']
        it['bid'] = bid
        it['body'] = f"===== TRAJECTORY {bid} =====\n" + '\n'.join(f"[R{rd}]\n{str(t).strip()}" for rd, t in it['myths']) + '\n\n'
    for total, b, chunk in balance(blind_items, args.blind_batches, lambda x: len(x['body'])):
        chunk.sort(key=lambda x: x['bid'])
        (out / f"blind_{b:02d}.txt").write_text(''.join(x['body'] for x in chunk))
        print('blind', b, len(chunk), 'trajectories', f"{total:,} chars")

    json.dump(index, open(out / 'trajectory_index.json', 'w'), indent=0)
    json.dump(key, open(out / 'blind_key.json', 'w'), indent=0)
    print('trajectories', len(index), 'blind', len(blind_items),
          'defectors', sum(v['defector'] for v in index.values()))


if __name__ == '__main__':
    main()
