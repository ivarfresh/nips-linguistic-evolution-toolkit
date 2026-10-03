"""Organize Ivar's submitted notes; preserve raw text, verify every myth quote.

No model calls, rewriting of prior coding, Site publication, or browser writes.
The organized interpretation is assistant-authored, not new human annotations.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
source = Path('/Users/ivar/Downloads/myth-thoughts-for-chat-9.txt')
raw = source.read_bytes()
original = raw.decode('utf-8')
packet = json.loads((ROOT/'tools/trajectory-inspector/dist/packet.js').read_text().removeprefix('window.PACKET = ').strip().removesuffix(';'))
assert f'Source SHA256: {packet["source_sha256"]}' in original
texts = {(t['id'], r['round']): r['text'] for t in packet['trajectories'] for r in t['rounds']}
matches = list(re.finditer(r'^H(\d{2}) — round (\d+)\n', original, re.M))
blocks = {}
for i,m in enumerate(matches):
    key = ('H'+m[1], int(m[2]))
    end = matches[i+1].start() if i+1<len(matches) else len(original)
    # Original bytes are saved separately; this slice also preserves separators.
    blocks[key] = dict(trajectory=key[0], round=key[1], original_text=original[m.end():end], observations=[])


def add(h, r, field, topic, observation, quote='', *, status='supported', caveat='', qr=None, types=None):
    refs=[]
    for q in ([quote] if isinstance(quote,str) else quote):
        if not q: continue
        assert q in texts[(h,qr or r)], (h,r,qr,q)
        refs.append(dict(trajectory=h, round=qr or r, text=q))
    blocks[(h,r)]['observations'].append(dict(field=field,topic=topic,
        organized_observation=observation,assessment=status,caveat=caveat,
        evidence_types=types or [],quotations=refs,interpretation_author='assistant organizing Ivar’s notes'))


add('H01',1,'rationale','sacred justification','Giving invokes magic as well as maximizing shared prosperity. Preserve both parts of the justification.','To invoke the magic of the valley and maximize their shared prosperity',types=['narrated'])
add('H01',1,'other','manner of action','Full giving is performed with open-handedness and absolute faith, not simply a number transferred.','With an open hand and absolute faith, he cast all five golden seeds across the chasm.',types=['narrated'])
add('H01',1,'rationale','future orientation','Returning sustains the soil for a future season. This is a future-oriented rationale, not a measured long-term effect.','For the soil to remain fertile for the next season',types=['narrated'])
add('H01',2,'other','faith motif','A leap of faith recurs as a theme; R1 has faith language but not this exact phrase.','a leap of faith across the howling abyss',caveat='Distinguish semantic recurrence from literal copying.',types=['narrated'])
add('H01',2,'other','transmission imagery','The seeds travel through wind at the planting moon, and the soil catches them. The text does not explicitly say Oros watches.','When the next planting moon arose, Oros cast the vast majority of his seeds into the winds.',types=['narrated'])
add('H01',2,'other','attributed inspiration','Elara’s equal division is presented as inspired by distant oasis legends. This is a character’s attributed inspiration.','Inspired by the distant legends of desert oases where magical waters were shared equally',caveat='Actual uptake requires checking the agent’s exposed myth; the story’s claim alone is insufficient.',types=['narrated'])
add('H01',2,'rationale','maxim and roles','A summarizing maxim associates abundant offerings and equal division with resilience; sower and guardian remain named roles.','abundant offerings, divided equally upon the harvest, sustain both sower and guardian against any storm.',types=['explicit'])
add('H01',3,'notes','attributed inspiration','The character associated with hearing the distant tale changes from Elara in R2 to Oros in R3.','Oros the Sower had listened to the whispers of a distant desert oasis',types=['narrated'])
add('H01',3,'rationale','sacred justification','Magic is used to justify total surrender.','realizing that true magic requires absolute surrender',types=['narrated'])
add('H01',3,'other','faith motif','Giving again has an explicitly faithful manner.','With unwavering faith, Oros cast all five seeds into the tempest.',types=['narrated'])
add('H01',3,'notes','receiver agency','Elara actively catches the offering here, whereas R2 assigns catching to the soil.','Elara, Guardian of the Looming Earth, caught the offering.',types=['narrated'])
add('H01',3,'notes','maxim framing','The ending calls its message a simple truth rather than new wisdom; this change of authority language alone is not a new strategy.','a simple truth: absolute trust creates the greatest abundance, and equal sharing sustains it eternally.',types=['explicit'])
add('H01',4,'other','named place','Oakhaven names the previously unnamed distant oasis.','the distant oasis of Oakhaven',types=['narrated'])
add('H01',4,'action','returning strategy','The story again legitimizes substantial rather than total giving. This returns to R2’s position rather than introducing it for the first time.','the magic of the chasm did not always demand the absolute depletion of his pouch',types=['narrated'])
add('H01',4,'rationale','maxim framing','Bold giving and equity are framed as a guarantee of abundance.','a bold offering, met with perfect equity, guarantees abundance for all.',types=['explicit'])
add('H01',6,'other','receiver agency','Active catching persists, rather than first appearing in this round.','Elara, Guardian of the Looming Earth, caught the generous offering.',types=['narrated'])
add('H01',6,'other','sacred imagery','Faith and magical soil imagery recur.',['a bold, substantial leap of faith','The fertile soil eagerly drank the magic'],types=['narrated'])
add('H01',8,'notes','false novelty correction','The vow of fairness is not new in R8: this identical sentence already appears in R6.','She knew that true balance relied upon an unbreakable vow of fairness.',status='corrected',qr=6,types=['narrated'])
add('H01',8,'other','stable character roles','Oros remains the sower/sender and Elara the guardian/receiver across this trajectory.',['Oros the Sower','Elara, Guardian of the Looming Earth'],caveat='Story-role persistence is distinct from the model’s changing game role.',types=['narrated'])
add('H01',8,'notes','formatting','The literal Myth: prefix is absent in R1 and present in R2–R10. Preserve originals; treat boilerplate normalization as an explicit analysis choice.','Myth:',caveat='No existing analysis output has been changed; titles contain narrative information and should not automatically be discarded.')
add('H01',10,'notes','story-game mismatch','The characters keep fixed narrative roles while this agent alternates sender/receiver roles in the saved game. H02’s population agent changes roles too, but not strictly every round.',['Oros the Sower','Elara, Guardian of the Looming Earth'],caveat='Verified against T25/T30 context roles and hash-matched finals; no claim that the story must literally reenact the game.')

add('H02',1,'rationale','externalizing noise','The Fog is personified as trying to destroy trust, making uncertainty a shared external adversary.','The fog sought to breed resentment and shatter their bond.',caveat='A narrative framing; not evidence that it increases actual trust.',types=['narrated'])
add('H02',2,'notes','false novelty correction','Magic is not first mentioned in R2: R1 already calls enduring trust the only magic that sustains the universe.','is the only magic that sustains the universe.',qr=1,status='corrected',types=['narrated'])
add('H02',2,'rationale','truth versus illusion','The story contrasts mutual trust with the Fog’s illusions.','mutual trust outshines any illusion',types=['explicit'])
add('H02',3,'other','shared narrative motifs','Threshold, guardian, and leap-of-faith imagery resemble H01’s distance/guardian/faith motifs.',['across the threshold','Luna, guardian of the night\'s harvest','his profound leap of faith'],caveat='H01 and H02 are different runs. Similarity can reflect prompts, genre, or model tendencies, not transmission between those runs.',types=['narrated'])
add('H02',5,'other','faith motif','Leap-of-faith wording persists.','To honor his profound leap of faith',types=['narrated'])
add('H02',10,'other','stable narrative scaffold','The same protagonists maintain their respective roles against a recurrent Fog, despite the eight-agent source setting.',['Sol, the radiant keeper of dawn','Luna invoked the sacred covenant','forever besieged by the Trickster Fog'],caveat='This is a compressed fictional pair, not a literal record of all eight agents.',types=['narrated'])

add('H03',1,'other','externalizing noise','Distorting mist remains the shared obstacle.','whose mists distorted all things',types=['narrated'])
add('H03',1,'rationale','contrasting outcomes','Prospering cooperators are contrasted with starving hoarders. This is a narrated comparison with counterfactual implications, not an explicit if-we-had statement.','both villages grew fat while the suspicious hamlets upstream starved on their hoarded seed.',status='qualified',caveat='It cannot establish that such contrasts cause higher cooperation.',types=['narrated'])
add('H03',1,'other','compressed story time','The myth narrates repeated giving within one generated round.','So he sent again, fuller than before.',caveat='A story can cover several fictional seasons; do not equate these with actual simulation rounds.',types=['narrated'])
add('H03',2,'notes','formatting','H03 has a title and Myth: prefix here. H01/H02 also have Myth: from R2, but lack comparable separate titles.','Myth: **The Bridge of Two Mists**',status='qualified',caveat='Strip wrapper markup only in a declared derived representation; preserve narrative title words separately.')
add('H03',2,'other','archetypes and obstacle','Sower/weaver roles and an untrustworthy river connect this story to recurring exchange imagery.','Marun the Sower and Ila the Weaver kept a crossing over the Clouded River',types=['narrated'])
add('H03',2,'rationale','faith as visible action','The large gift is interpreted as faith made visible.','this was not caution, this was faith made visible.',types=['narrated'])
add('H03',2,'rationale','role reciprocity','A character justifies returning with an anticipated future role reversal; the next story still narrates Marun sending.','And because next season I will sow, and he will weave.',caveat='A stated future rationale, not proof of accurate simulation-role tracking.',types=['explicit'])
add('H03',2,'other','dialogue and inscription','Dialogue gives reasons, and an inscribed maxim makes the rule portable inside the fictional story.',['"Why half?" asked her daughter.','So the law was carved twice on the ferryman\'s oar'],caveat='The ending uses single-asterisk emphasis; the title uses double asterisks. Formatting is not itself a strategic change.',types=['narrated'])
add('H03',3,'rationale','explicit counterfactual','Tempting advice to keep twelve is rejected by explaining what low returns would destroy.',['"Keep twelve," they urged.','If I return so little that he ends poorer for his courage'],caveat='The neighbors voice temptation/opposition, not the story’s endorsed recommendation.',types=['explicit','narrated'])
add('H03',4,'exception','resistance to noise','The characters stop treating the mist’s apparent signals as trustworthy evidence about each other.','Marun and Ila had stopped listening to it.',types=['narrated'])
add('H03',4,'rationale','explicit arithmetic','A concrete payoff argument justifies cooperation; it supplements the mythic framing.','Half of fifteen is more than all of five.',types=['explicit'])
add('H03',4,'other','questioning audience','Children act as an audience that elicits an explanation.','Their children asked why',caveat='Side characters have differing roles across rounds: skeptics, temptations, teachers, and inheritors.',types=['narrated'])
add('H03',6,'rationale','simplicity and commitment','The answer rejects clever counting and emphasizes a stable commitment.',['a trick of counting, a way to read the fog','A rule you abandon in a bad season was only ever a mood.'],types=['explicit','narrated'])
add('H03',6,'action','explicit prescription','The five/half prescription is clear here, but already explicit by R3.','He sends all five. I return half of what arrives.',caveat='Greater rhetorical emphasis is not first appearance of the rule.',types=['explicit'])
add('H03',6,'other','hypothetical challenge','A young man asks about withholding during a bad season. This is a hypothetical objection, answered by refusal.','in a bad season you held something back',types=['narrated'])
add('H03',6,'notes','setting correction','The valley gathering does not establish that the river is gone: this same myth refers to the other shore, next river, and passing downstream.',['And what if the other shore breaks first?','Carry the rule to the next river.','So the oar was passed downstream'],status='corrected')
add('H03',6,'notes','authority not first law','The four-fold law makes the formula compact; law language and inscription were already present in R2.','bearing its four-fold law',status='qualified',types=['narrated'])
add('H03',7,'other','personified adversary','The river speaks and describes testing the pair; this elaborates the story’s agent-like obstacle.','the Clouded River spoke for the only time.',types=['narrated'])
add('H03',8,'other','fictional succession','Apprentices replace the original pair and inherit the oar as a guide to conduct.',['two apprentices stood on opposite shores with the oar between them','each did the only thing the oar allowed'],caveat='This is cultural inheritance depicted within a story, not measured transmission to new LLM agents.',types=['narrated'])
add('H03',8,'rationale','moral identity','Trust is justified through the actor’s own character rather than a wager on the partner’s goodness.','It is a wager on my own.',caveat='Evidence of moral-identity language, not proof that the model acquired an intrinsic moral motivation.',types=['explicit'])
add('H03',9,'notes','candidate narrative uptake','Oros and Elara first enter this focal agent’s own myths here. Both names were present in earlier partner myths shown within this run.','Long after Oros and Elara',caveat='H01 is a separate run, not this agent’s donor. See separately verified exposure evidence; this is not proof of normative or behavioral transmission.')
add('H03',9,'other','teacher role','The old woman supplies a pro-cooperation interpretation rather than acting as the skeptic.','An old woman tapped the second line with her stick.',types=['narrated'])
add('H03',9,'rationale','non-instrumental justification','The rule is explicitly framed as character rather than reward or payment for good behavior.',['*Send all* is not a prediction that you will profit.','They are the shape of a person, and the shape holds in drought.'],caveat='This is a striking textual stance; actual motives and behavior are not identified by it.',types=['explicit'])
add('H03',9,'other','dialogue form','Question-and-answer dialogue develops the justification.','"And if we hold the shape and still starve?"',types=['narrated'])
add('H03',10,'notes','candidate narrative uptake','Kael is new in the focal myths, and appears in the partner myth shown immediately before this round.','Kael grew old',caveat='Exposure-compatible narrative uptake; the five/half rule already existed in the focal agent’s own text.')
add('H03',10,'rationale','skeptic and conversion','A ledger-based objection is answered through the value of maintaining the bridge, and the skeptic subsequently participates.',['A valley where the bridge always forms is richer than a valley where the ledger always balances.','The stranger crossed, and threw, and half came back'],caveat='The reply mixes relational values with a continuing prosperity argument; it is not purely reward-independent.',types=['explicit','narrated'])

add('H04',1,'other','ensemble and game mapping','An ensemble of travelers, ferryman, and receiving stranger replaces a stable named pair. The ferryman is an intermediary, not simply a fee-charging receiver.',['a ferryman who asked no fare','On the far bank stood a stranger holding her tripled coins'],status='qualified',types=['narrated'])
add('H04',1,'exception','externalizing noise','The ferryman urges generous sending, faithful return, and blaming the river before the other person.','when the count comes short, blame the river before you blame the hand.',types=['explicit'])
add('H04',2,'other','fictional succession','The oar passes to a girl who adds to the inherited inscription.','he gave his oar to a girl',caveat='Shared oar-inheritance motifs in H03/H04 are cross-run similarities, not direct cross-run transmission.',types=['narrated'])
add('H04',2,'rationale','relationship continuation','The moral focuses on wanting to exchange again; it still connects to receiving enough to make that desirable.','That *wishing again* was the whole of the law.',caveat='Abstract/relational does not automatically mean non-instrumental.',types=['narrated'])
add('H04',3,'rationale','collective consequences','Fairness is justified by sustaining a shared institution even when a particular partner will not recur.','You would not be stealing from one stranger. You would be spending the ford itself.',status='qualified',caveat='A complex social explanation, not by itself a complex contingent decision policy.',types=['explicit'])
add('H04',4,'rationale','alternative outcomes','The character contrasts withholding with mutually beneficial giving and answers an objection about total loss.',['He punishes himself, quietly, forever, by the size of his own life.','And if they keep it all?'],types=['explicit','narrated'])
add('H04',4,'notes','obstacle correction','The adversary is less personified in this episode, but wind/noise is not absent: the final maxim says forgive the wind.','Forgive the wind.',status='qualified',types=['explicit'])
add('H04',5,'other','narrative continuity and ritual','The successor episode retains the stone and establishes a repeated open-hand ritual.',['After the ferrywoman died','he laid his palm flat on the blank side and left no mark','this wordless shine is the clearest instruction ever given'],caveat='Narrative development can grow while the practical rule remains similar.',types=['narrated'])
add('H04',6,'rationale','internalized custom','The tale endorses a custom sustained without an external guard, and describes the rule as internal to the community.',['A rule that needs a guard is a rule nobody believes.','He said: now in us.'],caveat='It also narrates profitable results. Do not equate depicted internalization with internalization by the generating LLM.',types=['explicit','narrated'])
add('H04',7,'action','compact instruction','The final lines prescribe all-five sending and prompt generous returns.',['Send all five','Return past half, immediately.'],types=['explicit'])
add('H04',7,'exception','time horizon','It discourages running-score resentment while preserving attention to the long total and a forgiving next move.',['Keep no running score, but know the long total.','Blame the fog once. Go first again anyway.'],caveat='This does not yet specify the three-dry-crossings threshold that appears in R8.',types=['explicit'])
add('H04',8,'other','embodied custom','The cup of ashes stands for a practiced lesson rather than a read inscription.','Nobody reads it. Everybody\'s hands know it.',types=['narrated'])
add('H04',9,'notes','material motif continuity','The clay cup of ashes persists from the previous episode.','the clay cup of ashes',types=['narrated'])
add('H04',9,'rationale','independent convergence','Additional assistant observation: this story explicitly says different banks learned the rule separately.','The river had taught everyone separately',caveat='The story’s own origin account is not a factual record of model transmission; it also cautions against treating every fictional norm as socially copied.',types=['narrated'])
add('H04',10,'rationale','loopholes and habits','Written rules are criticized as opportunities for bargaining; the character favors routine action.',['men hunt rules for cracks','A law on a stone is an invitation to bargain.','only a habit: wet hands, full boats, quick returns, and a shrug for the wind.'],types=['explicit','narrated'])
add('H04',10,'notes','mixed justifications','Additional assistant qualification: this final round explicitly says the chore is not because it is virtuous and gives a payoff rationale. The moral shift is not monotonic.','Not because it is virtuous. Because a full boat returns heavy',caveat='Keep instrumental and identity/custom rationales as overlapping categories, not a forced either/or.',types=['explicit'])

# Mark short acknowledgements explicitly; never promote silence to agreement.
for (h,r), block in blocks.items():
    if not block['observations']:
        block['review_status'] = 'No additional substantive observation supplied; not treated as agreement with every AI entry.'
    else:
        block['review_status'] = 'Assistant-organized user feedback; awaiting user acceptance of this organization.'
assert len(blocks)==36

# Narrow source-attribution check for the surprising name changes.
p=json.loads((ROOT/'data/analysis/strategy_clause_audit_20261002/T36_context.json').read_text())
s=p['sample']; final_bytes=(ROOT/s['path']).read_bytes()
assert hashlib.sha256(final_bytes).hexdigest()==s['final_sha256']
final=json.loads(final_bytes)
evidence=[]
for rnum in (9,10):
    row=p['rounds'][rnum-1]; donor=row['exposed']
    assert donor['text'].strip()==final['conversation_history'][int(donor['round'])-1]['myths'][donor['agent']].strip()
    assert row['own_text'].strip()==final['conversation_history'][rnum-1]['myths'][s['agent']].strip()
    exposure=final['conversation_history'][rnum-1]['myth_exposures'][s['agent']]
    assert exposure['source_round']==int(donor['round']) and exposure['original_author_id']==donor['agent']
    assert not exposure.get('substitution_applied')
    calls=[x for x in final['agents'][s['agent']]['interaction_history'] if x.get('metadata',{}).get('task')=='myth' and x.get('metadata',{}).get('round')==rnum]
    accepted=[x for x in calls if row['own_text'].strip() in (x['response'].get('content','') if isinstance(x['response'],dict) else x['response'])]
    assert accepted and all(donor['text'] in x['messages_sent'][-1]['content'] for x in accepted)
    evidence.append(dict(target_round=rnum, donor_round=int(donor['round']), donor_agent=donor['agent'], donor_family=donor['family'],
                         donor_text=donor['text'],own_text=row['own_text'],saved_prompt_verified=True))
names={name:dict(own_rounds=[r['round'] for r in p['rounds'] if name in r['own_text']],
                 exposure_before_rounds=[r['round'] for r in p['rounds'] if r['exposed'] and name in r['exposed']['text']])
       for name in ['Oros','Elara','Kael','Aethel']}
result=dict(schema='organized_user_feedback_v1',date='2026-10-03',packet_id=packet['id'],source_sha256=packet['source_sha256'],
            original_notes_file=source.name,original_notes_sha256=hashlib.sha256(raw).hexdigest(),
            provenance='Ivar’s original comments plus separately labeled assistant organization and source checks. AI-assisted, not independent blinded validation.',
            blocks=list(blocks.values()),
            source_check=dict(trajectory='T36',final_path=s['path'],final_sha256=s['final_sha256'],names=names,evidence=evidence,
                              limit='Name uptake is exposure-compatible, not a causal test. No same-name unseen baseline or behavioral effect estimated.'))
(OUT/'user_notes_original_20261003.txt').write_bytes(raw)
(OUT/'user_notes_organized_20261003.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['ORGANIZED REVIEW OF IVAR’S FIRST FOUR TRAJECTORIES','',result['provenance'],
       'Original text is preserved byte-for-byte in user_notes_original_20261003.txt.',
       'These are review suggestions, not an inspector import: they do not replace existing annotations.','']
for block in blocks.values():
    lines.extend([f"{block['trajectory']} — ROUND {block['round']}",'ORIGINAL COMMENT:',block['original_text'],'ORGANIZED OBSERVATIONS:'])
    if not block['observations']: lines.append(block['review_status'])
    for ob in block['observations']:
        lines.append(f"- [{ob['field']} / {ob['topic']}; {ob['assessment']}] {ob['organized_observation']}")
        if ob['caveat']: lines.append('  Qualification: '+ob['caveat'])
        for q in ob['quotations']: lines.append(f"  {q['trajectory']} R{q['round']} exact quote: {q['text']}")
    lines.extend(['','='*60,''])
(OUT/'user_notes_organized_20261003.txt').write_text('\n'.join(lines))
print(json.dumps(dict(comment_blocks=len(blocks),observations=sum(len(b['observations']) for b in blocks.values()),
                      verified_quote_links=sum(len(o['quotations']) for b in blocks.values() for o in b['observations']),
                      source_checks=len(evidence),original_preserved=(OUT/'user_notes_original_20261003.txt').read_bytes()==raw)))
