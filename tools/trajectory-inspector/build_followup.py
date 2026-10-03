"""Publish a read-only, unblinded view; exclude local paths and raw call histories."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'docs/research/strategy_clause_audit_20261002'
out = json.loads((BASE / 'followup_results_20261003.json').read_text())
manifest = json.loads((BASE / 'manifest.json').read_text())
trajectories = []
for s in manifest['sample']:
    p = json.loads((ROOT / 'data/analysis/strategy_clause_audit_20261002' / f"{s['id']}_context.json").read_text())
    trajectories.append({**{k:s[k] for k in ['id','tier','family','task_order','size','mixed','agent']},
        'rounds':[{'round':r['round'],'own':r['own_text'],'game':r['agent_game'],
                   **{k:({'text':r[k]['text'],'agent':r[k]['agent'],'round':r[k]['round'],'family':r[k]['family']} if r[k] else None) for k in ['exposed','unseen']}}
                  for r in p['rounds']]})
events = [{k:e[k] for k in ['id','event_round','original_event','previous_own_match','all_earlier_own_matches','exposed_match','unseen_match','last_pre_write_game_round','same_round_game_can_be_source']} for e in out['events']]
payload={'trajectories':trajectories,'events':events,'screens':out['behavior_screens'],
         'amounts':[{k:v for k,v in a.items() if k not in ['run_id','decision_prompt']} for a in out['behavior_eligibility']],
         'lexical':out['lexical_summary']}
target = Path(__file__).parent/'dist/followup-data.js'
target.write_text('window.FOLLOWUP = '+json.dumps(payload,ensure_ascii=False).replace('</','<\\/')+';\n')
print(f'Exported {len(trajectories)} trajectories, {len(events)} selected events; original notes untouched.')
