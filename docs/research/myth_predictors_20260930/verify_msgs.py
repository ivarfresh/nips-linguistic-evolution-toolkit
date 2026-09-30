import json,sys,difflib
F=sys.argv[1]
d=json.load(open(F)); A=d['agents']
calls={}
for aid,ag in A.items():
    for ih in ag['interaction_history']:
        md=ih['metadata']
        if md['round']==1 and md['task']!='myth':
            calls[aid]=ih
        if md['round']==1 and md['task']=='myth':
            calls.setdefault(aid+'_myth',ih)
print('agent-level fields:',{a:(A[a]['model'],A[a]['temperature'],A[a]['memory_capacity'],A[a]['initial_bias'],A[a]['population_role'],A[a]['display_name']) for a in A})
print('system prompts identical across agents:',len({A[a]['system_prompt'] for a in A})==1)
inv=[a for a in calls if not a.endswith('_myth') and calls[a]['metadata']['role']=='investor']
tru=[a for a in calls if not a.endswith('_myth') and calls[a]['metadata']['role']=='trustee']
print('investors',inv,'trustees',tru)
ex=calls[inv[0]]
print('call keys',list(ex.keys())); print('metadata keys', list(ex['metadata'].keys()))
for k in ex:
    if k not in ('messages_sent','response','prompt','metadata','timestamp','interaction_index','agent_id'): print(k, repr(ex[k])[:300])
def show(a):
    ms=calls[a]['messages_sent']
    return [(m['role'], m['content']) for m in ms]
base=show(inv[0])
print('n messages', len(base), [r for r,_ in base])
for a in inv[1:]+tru:
    o=show(a)
    print('\n==',a, calls[a]['metadata']['role'],'n msgs',len(o), [r for r,_ in o])
    for i,((r1,c1),(r2,c2)) in enumerate(zip(base,o)):
        if c1!=c2:
            sm=difflib.SequenceMatcher(None,c1,c2)
            ops=[op for op in sm.get_opcodes() if op[0]!='equal']
            print(f' msg{i} {r1}: differs; {len(ops)} ops; first diffs:')
            for op in ops[:4]:
                print('   ',op[0],repr(c1[op[1]:op[2]][:150]),'->',repr(c2[op[3]:op[4]][:150]))
# print full investor round-1 game messages for inspection
print('\n##### FULL', inv[0])
for r,c in base: print('---',r); print(c[:2500])
