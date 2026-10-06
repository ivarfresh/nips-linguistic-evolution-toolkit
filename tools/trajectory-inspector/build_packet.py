"""Export only blinded human-packet text; never load the identity key or machine codes."""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / 'docs/research/strategy_clause_audit_20261002/HUMAN_VALIDATION.md'
text = SOURCE.read_text()
trajectories = []
parts = re.split(r'^## (H\d{2})\s*$', text, flags=re.M)
for i in range(1, len(parts), 2):
    rounds = re.split(r'^### Round (\d+)\s*$', parts[i+1], flags=re.M)
    trajectory = {'id': parts[i], 'rounds': [
        {'round': int(rounds[j]), 'text': rounds[j+1].strip()}
        for j in range(1, len(rounds), 2)]}
    assert [r['round'] for r in trajectory['rounds']] == list(range(1, 11))
    trajectories.append(trajectory)
assert [t['id'] for t in trajectories] == [f'H{i:02d}' for i in range(1, 13)]
packet = {'id': 'strategy-clauses-20261002-human-v1', 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'trajectories': trajectories}
target = HERE / 'dist/packet.js'
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text('window.PACKET = ' + json.dumps(packet, ensure_ascii=False).replace('<', '\\u003c') + ';\n')
print(f'Exported {len(trajectories)} blinded trajectories / 120 rounds.')
