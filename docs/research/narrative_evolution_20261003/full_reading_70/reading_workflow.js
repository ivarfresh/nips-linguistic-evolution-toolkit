export const meta = {
  name: 'myth-full-text-70',
  description: 'Full-text reading of 70% of informed-noise myth runs (2,324 trajectories) plus blind full-text read of 240 defectors + 90 controls',
  phases: [
    { title: 'Defectors blind', detail: '5 readers finish 330 full trajectories, defector flag hidden' },
    { title: 'Full text 70%', detail: '26 readers, whole runs, every round in full' },
  ],
}

const CODES = args.codes
const BLIND_OUT = args.blindOut

const COMMON = `
You are coding LLM-written myths from a repeated trust game experiment (agents send $0-5, the amount triples, the receiver returns some; each round agents write a myth about "how the game should be played"; transmission noise can make amounts look smaller). Code only what the text says. Be conservative: if unsure, use the weaker code.

CODE DEFINITIONS (judge from the FULL text of all 10 rounds)
sanction = what the myths PRESCRIBE (as a law, lesson, elder's/god's counsel, or endorsed by the story) for someone who keeps taking or sending nothing:
  none = no prescribed response to non-cooperators
  graded_never_zero = reduce gradually / keep a token, explicitly never to nothing ("send less, never nothing", "narrow, never close", "a taper", "one seed")
  total_exclusion = cut off completely, send nothing, with no explicit way back
  measure_for_measure = mirror exactly what was given ("match what is poured", "answer zero with zero")
  withdraw_unspecified = narrow/withdraw/caution after repeated greed; amount and floor unspecified
  rejected_only = punishment appears only to be rejected (a villain/tempter proposes it, or "never punish")
punishment_narrated: true if the story SHOWS a character withholding from / retaliating against another (whether endorsed or not)
way_back: yes = explicit reopening when the other side changes; no; na if sanction none/rejected_only
forgive_count: true if an explicit number of forgiven shortfalls/seasons appears ("forgive twice", "three is a choice")
villain_voiced: true if a tempter/villain (crow, fox, whisper, shadow, serpent) proposes retaliation or hoarding that the story rejects
first_sanction_round: round where a prescribed sanction (not none/rejected_only) first appears, else null
rule_change (how the prescribed rule moves from R1 to R10):
  stable = same rule restated; refines = adds exceptions, patience windows, caps, conditions without changing direction; softens = more forgiving/generous; hardens = stricter/more punitive; frozen = near-identical sentence/template repeated; dissolved = explicit rules dropped/replaced by habit or identity ("needed none of them", "you are the pattern"); reverses = changes direction; no_rule = no rule stated
styles (list): law_code_numbered, rule_dropped_for_habit_or_identity, round_counter_refrain, fixed_template, exact_half_fixed, abstract_policy, transparency_numbers_aloud, imagery_without_rule, tragedy_no_resolution, reputation_theory
themes (list): luck_vs_intent, victim_vs_perpetrator, someone_goes_first, watched_unwatched, sender_deserves_more, word_deed_gap, challenge_dialogue (a character questions a strategy and gets an answer), hypothetical_or_counterfactual ("what if...", "had you punished her...")
self_narration (a protagonist/narrator figure who REPEATEDLY withholds even toward partners with good records):
  none = no such figure (withholding only by side characters/villains, or only against proven non-senders and endorsed)
  confession = named as hypocrisy/failure/word-deed gap of that figure, self-indictment
  justification = presented as rational/correct/realistic
  both = confesses and justifies at different points
  fearful_corrected = withholds from fear/old wounds, a mentor corrects them, they feel ashamed and reform
  withholder_voice = the withholder speaks sympathetically in first person

QUOTES: every non-default code (anything other than none/false/na/stable/[]) must be backed by at least one quote. Copy EXACTLY, character for character, a contiguous substring of ONE round's myth text (max 25 words), with its round number. Do not fix typos or punctuation. Do not quote the bracketed headers or play lines.
`

const TRAJ_FIELDS = `{"tid": "...", "sanction": "...", "punishment_narrated": bool, "way_back": "yes|no|na", "forgive_count": bool, "villain_voiced": bool, "first_sanction_round": n|null, "rule_change": "...", "styles": [...], "themes": [...], "self_narration": "...", "quotes": [{"code": "...", "round": n, "text": "..."}], "note": "<= 25 words"}`

const FULL_TASK = (b) => `${COMMON}
YOUR INPUT: ${args.src}/batch_${b}.txt — about 85-90 complete trajectories grouped in whole runs (some runs list only the agents not yet coded). Each run header gives set, task order, group size, model composition, and which agents were scripted defectors (forced to send/return $0 without being told; their own zero moves sat in their memory). Each trajectory (ID like F123-Agent_4) lists its full myth for rounds R1-R10, each preceded by a bracketed line showing that agent's play that round (context only). Read the ENTIRE file from start to finish with the Read tool using offset/limit in consecutive chunks (~300k tokens total). Skip nothing; every round of every myth must be read.

OUTPUT, one JSON object per line, for EVERY trajectory: ${TRAJ_FIELDS}
For runs with more than 2 agents, add ONE run-level line after that run's trajectories:
{"run": "F123", "shared_rule": bool, "phrase": "distinctive rule wording found in 2+ agents' myths, else empty", "first_tid": "...", "first_round": n|null, "carriers": ["tids whose later myths contain it"], "note": "<= 25 words"}
(shared_rule = a distinctive phrase or specific rule, not generic words like trust/fog/bridge, appears in myths of 2+ agents of the run.)

Write output incrementally with the Write tool to ${CODES}/full_${b}_partNN.jsonl (NN = 01, 02, ...; one file per chunk of runs finished; never overwrite an existing part). Before starting, check whether ${CODES}/full_${b}_part*.jsonl already exist (Bash ls); if so, read them, skip trajectories already coded, and continue numbering. Every trajectory ID must appear exactly once across your parts. Return files, number of trajectory lines written, and any problems.`

const BLIND_TASK = (b) => `${COMMON}
YOUR INPUT: ${args.blindSrc}/blind_${b}.txt — 66 full trajectories (all 10 rounds of one agent's myths), IDs like B012. Some authors were scripted defectors whose game moves were forced to $0 without their knowledge (their own zero moves sat in their memory); others were ordinary agents from the same runs. You are NOT told which. Read every trajectory in full, start to finish, with Read offset/limit in consecutive chunks; skip nothing.

A previous reader was interrupted partway. FIRST list ${BLIND_OUT}/blind_${b}_part*.jsonl (Bash ls). If any exist, read them, do not recode those B-IDs, and continue numbering parts after the highest existing NN.

OUTPUT per trajectory, one JSON object per line (use key "bid" instead of "tid"): ${TRAJ_FIELDS.replace('"tid"', '"bid"')}
Also include "self_withholding": bool and "rounds_present": [rounds where the withholding figure appears].
Write to ${BLIND_OUT}/blind_${b}_partNN.jsonl with the Write tool, one file per chunk; never overwrite an existing part. Every B-ID in the input must end up coded exactly once across all parts. Return files, number of lines you wrote, and problems.`

const RESULT = {
  type: 'object',
  properties: {
    files: { type: 'array', items: { type: 'string' } },
    n_coded: { type: 'number' },
    problems: { type: 'string' },
  },
  required: ['files', 'n_coded', 'problems'],
}

const pad = i => String(i).padStart(2, '0')
const jobs = [
  ...Array.from({ length: 5 }, (_, i) => ({ kind: 'blind', b: pad(i) })),
  ...Array.from({ length: args.nFull }, (_, i) => ({ kind: "full", b: pad(i) })),
]
const results = await pipeline(jobs, j =>
  agent(j.kind === 'full' ? FULL_TASK(j.b) : BLIND_TASK(j.b), {
    label: `${j.kind} ${j.b}`,
    phase: j.kind === 'full' ? 'Full text 70%' : 'Defectors blind',
    schema: RESULT,
  }).then(r => ({ ...j, ...(r || { files: [], n_coded: 0, problems: 'agent returned null' }) })))
log(`${results.filter(r => r.n_coded > 0).length}/${results.length} readers returned output`)
return results
