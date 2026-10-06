"""Select 70% of myth-bearing runs per condition cell and pack full trajectories into reader batches.

No model calls. Reads the hash-checked corpus manifest from
docs/research/narrative_evolution_20261003/corpus_manifest.json, samples 70% of
runs (largest-remainder allocation) across cells = cohort x model composition x task order x group size
x noise x history policy x defector design, keeps every agent of a chosen run, and writes balanced
plain-text batches (whole runs only, conditions mixed across readers).

    python analyses/narrative_full_reading_prepare.py --batches 30 --out <dir>
"""
import argparse
import collections
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs/research/narrative_evolution_20261003/corpus_manifest.json'
SELECTION = ROOT / 'docs/research/narrative_evolution_20261003/full_reading_70/selection.json'
SEED = 2026100470
SHARE = 0.70


def composition(path):
    # data/json/noise_experiments/<set>/<run_group>/<model_dir>/<order>/...
    return path.split('/')[5]


def cell(run):
    # condition_sha256 differs per replicate (it hashes seeds), so group on the design fields instead
    m = run['metadata']
    return (run['cohort'], composition(run['path']), run['task_order'], m['num_agents'],
            json.dumps(m.get('noise_config'), sort_keys=True), m.get('history_policy'),
            len(m.get('defector_agent_ids') or []), m.get('random_defection_probability') or 0)


def play_lines(state, agent):
    out = {}
    for turn in state['conversation_history']:
        for d in turn.get('dyads', []):
            if d['investor'] == agent:
                out[turn['round']] = f"as sender: sent {d.get('sent_decision')}; saw {d.get('returned_communicated', d.get('returned'))} returned"
            elif d['trustee'] == agent:
                out[turn['round']] = f"as receiver: saw {d.get('received_communicated', d.get('received'))} arrive; returned {d.get('returned_decision')}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', type=int, default=30)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = [r for r in json.load(open(MANIFEST))['runs'] if r['myth_entries']]
    cells = collections.defaultdict(list)
    for r in runs:
        cells[cell(r)].append(r)
    # Largest-remainder allocation: 70% of all runs overall, proportional per cell, at least 1 per cell.
    keys = sorted(cells, key=str)
    target = round(SHARE * len(runs))
    quota = {k: max(1, math.floor(SHARE * len(cells[k]))) for k in keys}
    by_remainder = sorted(keys, key=lambda k: -(SHARE * len(cells[k]) - math.floor(SHARE * len(cells[k]))))
    i = 0
    while sum(quota.values()) < target:
        k = by_remainder[i % len(keys)]
        if quota[k] < len(cells[k]):
            quota[k] += 1
        i += 1
    rng = random.Random(SEED)
    chosen = []
    for key in keys:
        group = sorted(cells[key], key=lambda r: r['path'])
        chosen += rng.sample(group, quota[key])

    records = []
    for i, r in enumerate(sorted(chosen, key=lambda r: r['path'])):
        raw = (ROOT / r['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == r['sha256'], r['path']
        state = json.loads(raw)
        meta = r['metadata']
        defectors = set(meta.get('defector_agent_ids') or [])
        run_id = f"F{i:03d}"
        authors = sorted({a for row in state['conversation_history'] for a in (row.get('myths') or {})},
                         key=lambda a: int(a.split('_')[-1]))
        lines = [f"===== RUN {run_id} =====",
                 f"set: {r['cohort']} | task order: {r['task_order']} | agents: {meta['num_agents']} | "
                 f"composition: {composition(r['path'])}",
                 f"defector agents (forced to send/return $0, not told): {sorted(defectors) or 'none'} | "
                 f"random defection probability: {meta.get('random_defection_probability') or 0}",
                 f"file: {r['path']}"]
        agents = []
        for a in authors:
            model = state['agents'][a]['model'].split('/')[-1]
            tid = f"{run_id}-{a}"
            agents.append(dict(trajectory_id=tid, agent=a, model=model, defector=a in defectors))
            lines.append(f"\n----- TRAJECTORY {tid} | {model} | {'SCRIPTED DEFECTOR' if a in defectors else 'standard'} -----")
            play = play_lines(state, a)
            for row in state['conversation_history']:
                text = (row.get('myths') or {}).get(a)
                if text is None:
                    continue
                lines.append(f"[R{row['round']}] (play this round: {play.get(row['round'], 'n/a')})")
                lines.append(str(text).strip())
        body = '\n'.join(lines) + '\n'
        records.append(dict(run_id=run_id, path=r['path'], sha256=r['sha256'], cohort=r['cohort'],
                            task_order=r['task_order'], num_agents=meta['num_agents'],
                            composition=composition(r['path']), condition_sha256=meta['condition_sha256'],
                            defectors=sorted(defectors), random_defection=meta.get('random_defection_probability') or 0,
                            agents=agents, chars=len(body), body=body))

    # Balance whole runs across batches; shuffle first so each reader sees mixed conditions.
    order = records[:]
    random.Random(SEED + 1).shuffle(order)
    order.sort(key=lambda r: -r['chars'])
    loads = [[0, b, []] for b in range(args.batches)]
    for r in order:
        loads.sort(key=lambda x: x[0])
        loads[0][0] += r['chars']
        loads[0][2].append(r)
    loads.sort(key=lambda x: x[1])
    batches = []
    for total, b, rs in loads:
        rs.sort(key=lambda r: r['run_id'])
        name = f"batch_{b:02d}.txt"
        (out / name).write_text(''.join(r['body'] for r in rs))
        batches.append(dict(batch=b, file=str(out / name), runs=[r['run_id'] for r in rs],
                            trajectories=sum(len(r['agents']) for r in rs), chars=total))

    for r in records:
        r.pop('body')
    SELECTION.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(seed=SEED, share=SHARE, cell_key='cohort x composition x task_order x num_agents x noise_config x history_policy x n_defectors x random_defection',
                   runs_total=len(runs), runs_chosen=len(records),
                   trajectories_chosen=sum(len(r['agents']) for r in records),
                   batches=[{k: v for k, v in b.items() if k != 'file'} for b in batches], runs=records),
              open(SELECTION, 'w'), indent=1)
    json.dump(batches, open(out / 'batches.json', 'w'), indent=1)
    print(f"runs {len(records)}/{len(runs)}, trajectories {sum(len(r['agents']) for r in records)}, "
          f"chars {sum(r['chars'] for r in records):,}")
    for b in batches:
        print(b['batch'], b['trajectories'], f"{b['chars']:,}")


if __name__ == '__main__':
    main()
