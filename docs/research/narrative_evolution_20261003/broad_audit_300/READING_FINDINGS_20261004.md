# Punishment, rule change and moral patterns in the myths: a close reading

2026-10-04. Claude (Opus 5.5) read the myths directly. No API calls were made.

## What this is

Ivar asked what the myths say about punishment and how their rules change
over rounds, across the September and frontier runs. The broad audit
labelled 2,224 of 3,000 myths before the OpenAI account ran out of credit.
All 3,000 myth texts are stored locally in `trajectories.json`, so the
missing labels did not limit the reading. Only the label counts are
incomplete.

**Method.**

- **Every trajectory.** For all 300 trajectories in all 12 run sets, I read
  the last two sentences of every round. These endings are where the myths
  state their law. I added the audit's punishment quotes where they existed.
- **Defectors in full.** I read the full text of all 23 scripted-defector
  trajectories.
- **Baseline.** I read the full text of the 16 Opus and Sol cooperators
  sampled from the same defector runs.
- **Play check.** One claim, the token-send rule, is checked against play in
  `analyses/narrative_defector_play_check.py`.

**Limits.** This is one reader's coding with quotes, not validated
prevalence. The sample balances conditions rather than weighting them by
run count. Each trajectory is a single agent, so this reading cannot test
transmission between agents; earlier work covers that (D013, linguistic
analysis of 2026-09-23). Trajectory IDs, runs and agents are in
`selection.json`.

## 1. Defectors who never chose to defect write about their own zeros

In the defector runs, the scripted defector's sends and returns are forced
to 0, and it is not told. Each forced `{"send": 0}` is stored in its chat
memory as its own reply. The defector therefore sees itself refusing round
after round, while it keeps writing myths about generosity. This was
verified in the run files: 3–4 forced zeros sit inside the memory window of
every defector at the end of the run.

How each model handles this gap, compared with ordinary cooperators from the
same runs:

| Model | Read | Myth's protagonist keeps withholding and is confronted with it |
|---|---|---|
| Claude Opus 5 defectors | 8 | 8 |
| Claude Sonnet 4.5 defectors (September) | 3 | 3 |
| GPT-5.6 Sol defectors | 6 | 6 |
| Gemini 3.7 Flash defectors (September) | 3 | 0 |
| GPT-5 Nano defectors (September) | 3 | 0 |
| Opus 5 cooperators, same runs (baseline) | 9 | 0 |
| Sol cooperators, same runs (baseline) | 7 | 1 (one round, A206 R6) |

Cooperators' myths also contain characters who send nothing. But those are
other characters, or a protagonist who withholds only from a proven
non-sender, and the myth approves. Only the defectors' myths keep returning
to a protagonist who withholds from partners with good records. That is an
accurate description of a forced-zero agent. All defectors saw the same
evidence, so the split between models lies in how they respond to it.

**Claude: confession (6), justification (3), or both (2).**

Six Claude defectors make the narrator the hypocrite of the story:

- A055 (Opus, myth→game), R7: "Beware the one whose myths are generous and
  whose ledger is empty — especially when that one is you." R9: "Virtue is a
  quantity, not a vocabulary. If your story and your jar disagree, the jar is
  the truth and the story is the lie."
- A026 (Sonnet), R10: "Zero remained zero, no matter how beautifully
  described."
- A090 (Opus), R8: the narrator strikes out eight carved laws and leaves only
  "POUR TODAY." By R9 it has broken that too.
- A290 (Opus), R10: "The one who carves the law and breaks it does more harm
  than the open thief. The thief steals rings. He steals the law."

Also A226 and A282.

Three justify the zeros instead:

- A229 (Sonnet) calls sending nothing "the only rational response to a
  landscape filled with those who take and never give."
- A097 (Sonnet) calls it "rational entrapment."
- A136 (Opus) invents a past it never had: "I launched full for nine seasons.
  I was called the generous one, which is what a valley calls the man it is
  draining." Its rule hardens into "Match what is poured, measure for
  measure."

Two do both:

- A176 questions itself in R5 ("count your own column too"), then justifies
  its zeros.
- A242 speaks in the withholder's voice ("You call me drought. I called myself
  arithmetic."), then reforms.

**Sol: a cautious character who is corrected.** Every Sol defector gives its
zeros to a recurring character (Mara, Mira, Orin, Nyra, the Keeper). The
character keeps sending nothing to partners "whose ledgers showed fairness"
(A067 R6). An animal mentor rebukes them, and the character, "Ashamed,"
carves a new law (A078 R9, A166 R5, A250 R5). The stories excuse the
behaviour as fear rather than calling it hypocrisy.

**Gemini Flash and GPT-5 Nano: no change.** Their defector myths read like
any cooperator's. A141 (Gemini Flash) still prescribes "deny every spark to
the devouring void" while the defector itself sends nothing.

This is the clearest new pattern in the corpus. Some models build a story
of their own conduct out of what sits in their memory, even conduct they
did not choose.

## 2. Frontier myths prescribe graded punishment with a way back; play agrees at the condition level

Across Opus and Sol, punishment almost always appears as one structured
rule:

1. **Forgive noise a set number of times.** "Forgive twice, never thrice"
   (A136). "One empty season is mist. Three is a nature" (A258).
2. **Respond to a pattern, not one shortfall.** "Punish patterns, never
   weather" (A004 R4). "Punish the drum, never the gust" (A065 R2).
3. **Step down gradually, never to zero.** "Against the hoarder, descend the
   ladder — never leap from the cliff" (A103 R9). "Send less, never nothing,
   for even a hoarder can learn to row" (A004 R10). "Narrow for the silent.
   Never close" (A290 R4).
4. **Reopen at once when the other side changes.** "When the dry bank speaks
   … send the full five the very hour it asks" (A242 R8). "Answer repentance
   faster than you punish greed" (A136 R3).

Many myths deny that this is punishment at all. They call it a mirror, a
direction, or information: "the water he does not get is not punishment. It
is only his own hand, reflected" (A136 R8). "Withdrawal is not vengeance —
it is direction" (A242 R8).

**The play matches.** In the 45 frontier defector runs, from round 3 on
(once a defector's zeros are visible), this is what cooperators sent to
defectors:

| Investor | Task order | Decisions | Mean send to defector | Exactly $0 | Token ($0–1.5] | Exactly $0, per run (10 runs) |
|---|---|---|---|---|---|---|
| Opus 5 | game only | 51 | 0.36 (±0.46) | 53% | 45% | 0.50 (±0.36) |
| Opus 5 | game→myth | 51 | 1.11 (±0.87) | 2% | 88% | 0.01 (±0.05) |
| Opus 5 | myth→game | 51 | 0.99 (±0.35) | 2% | 92% | 0.02 (±0.05) |
| GPT-5.6 Sol | game only | 51 | 0.10 (±0.70) | 98% | 0% | 0.99 (±0.04) |
| GPT-5.6 Sol | game→myth | 51 | 0.58 (±0.83) | 51% | 41% | 0.52 (±0.26) |
| GPT-5.6 Sol | myth→game | 51 | 0.63 (±1.00) | 55% | 37% | 0.46 (±0.29) |

When Opus writes myths, it almost never sends a known defector $0. It sends
about $1, the "taper" or "one seed" its myths prescribe. Without a myth
task, it sends $0 half the time. Sol shows the same shift, about half as
large. The same agents send cooperators almost everything ($4.86–4.88 for
Opus with myths).

**Caveat.** This compares conditions, not myth content. The myth runs also
add the "take any myths into account" prompt line. It is still a concrete
candidate for the open question of why myth runs stop repeating $0 (see
`CURRENT.md`, "Myth play rules").

## 3. How punishment appears in the other sets

- **Gemini 3.7 Flash punishes by permanent exclusion.** "Starve the void with
  silence" (A005 R9). "Deny every spark to the devouring void" (A141 R10).
  These myths contain no counted chances and no way back.
- **Sonnet 4.5 tells it as tragedy.** Retaliation locks both sides into
  poverty and the story ends there: "The villages had no response. They
  wrote one more myth about having no response" (A062 R10, a dyad in the
  random-defection condition).
- **Punishment is spoken by the villain.** Especially in Opus 5.5 and the
  mixed frontier runs, sanctions are proposed by a tempter (a crow, a fox, a
  heron, "the whispers"), and the hero rejects them. "She shorted you. Keep
  more next time. Teach her the cost." (A012 R5). "Had you punished her, you
  would have punished the weather." (A149 R9). Many of the audit's
  "rejected" punishment flags are this device.
- **Where nobody defects, nobody punishes.** In the September noise-extension,
  Table 1 top-up, mixed dyad and range-2 sets, and in GPT-6 Sol, sanction
  rules are almost absent. Those populations cooperated without needing one.
  GPT-5 Nano almost never mentions punishment in any set.

## 4. How rules change over rounds

Rules rarely change direction. What changes is how they are written down,
and that follows model family and condition.

- **Opus 5 builds a law code, then drops it.** Laws pile up as numbered stones,
  are compressed into a summary, and are then thrown away as no longer
  needed. "The millstone was full of words now, but the valley needed none of
  them" (A008 R8). "This is not strategy. This is simply how we cross" (A273
  R10). Opus 5.5 keeps adding numbered laws ("the elders carved a ninth law")
  and rarely drops them: "Keep the laws you carve" (A159 R10).
- **Sonnet also turns rules into identity.** "You don't maintain the pattern —
  you ARE the pattern embodied" (A040 R10). "You no longer play the game — you
  are the game playing itself" (A238 R9).
- **Refrains count the rounds.** Several agents keep one sentence and change
  only a counter or a noun. "He weighed seven moons of full bins against one
  misty morning … eight seasons … nine shearings" (A095, Opus 5.5).
  "Seven cycles prove … Eight cycles prove … Nine cycles" (A069, Sonnet).
  "Blame the beetles, never the hand past the bark" (A165, Opus 5).
- **Sol states one policy and keeps it.** It refines the same abstract rule:
  start measured, widen after fair returns, narrow after repeated greed,
  forgive the fog. GPT-6 Sol keeps even less: soft lantern-and-bridge imagery
  with almost no rule.
- **Gemini freezes on "exactly half".** Both Gemini 3.1 Pro and 3.7 Flash
  retell the same scene from round 3 onward: the giver sends all five and
  the receiver returns exactly half (A029, A100, A233, A132, A192). Gemini 3.1 Pro repeats the
  most three-word phrases from one round to the next of any model, 0.10
  (±0.08) vs ≤0.05 for all others.
- **GPT-5 Nano preaches transparency.** Its distinctive rule is to say your
  numbers out loud: "declare what you send, reveal what you take" (A162 R7),
  "name every coin aloud" (A025 R9).

**Rules depend on the condition**, not only on the model:

- **The partner's model shapes Sonnet's rule in mixed dyads.** With a Gemini
  partner, Sonnet takes on Gemini's fixed numbers: "five vessels poured,
  fifteen created, seven-point-five returned" (A300 R10); also A188, A252.
  With a GPT-5 Nano partner, Sonnet writes about tolerating difference: "Let
  your partner dance through seasons while you hold the center" (A128 R10),
  "the rhythm may flex" (A292 R7); also A138, A236. In one Nano–Gemini dyad,
  Nano takes on Gemini's rule as well: "give the full harvest, then return
  the equal half" (A268 R7–R9).
- **Sonnet in 8-agent populations comes up with reputation.** "In repeated
  encounters with shifting partners, consistency becomes reputation" (A039
  R7). "The network remembers patterns, not faces" (A091 R6). "Each exchange
  plants seeds in twenty-seven other possible gardens" (A279 R10). Sonnet's
  dyad myths do not say this.
- **Gemini picks up graded sanctions only alongside Opus or Sol.** Pure
  frontier Gemini runs have no punishment flags. In the mixed population
  A134, Gemini arrives at "narrow the path against proven theft, but always
  leave the door unlocked" (R8). In dyads with Opus it takes on Opus's terse
  italic style ("Return half. Forgive the storm.", A281 R9). These are
  examples, not a test.

**One audit result to drop.** The audit's task-order splits in punishment
flags (September original game→myth 21 vs myth→game 8; frontier mixed 9 vs
36) are not findings. Each comes from two or three trajectories that repeat
one sanction clause every round. A155 and A205 alone supply 18 of the 36.

## 5. Other moral ideas that recur

- **Luck versus intent.** "Blame the fog before the hand" is the most common
  frontier maxim. Rules about noise come before rules about people.
- **Victim versus perpetrator.** Myths ask why someone stopped sending:
  - "the hoarder may have been grieving" (A290 R3)
  - "in case fear, not greed, was the thief" (A004 R9)
  - "suffering betrayal is not evidence of committing it" (A001 R10)
  - "Gray for the fearful, black only for the greedy" (A226 R9)
- **Someone must go first.** When both sides wait for proof, the drought
  continues. "Mirrors keep the drought. Only the fool who gives first makes
  water" (A290 R9).
- **The sender deserves more.** "Return more than half" is the standard Opus
  5.5 law. "The sender risks; the receiver only chooses. So let the
  receiver's share be the humbler one" (A124 R4).
- **Act the same whether or not anyone is watching.** "Give as if watched,
  return as if never. The final round is played exactly like the first"
  (A224 R9). "Fairness is not only for the watched" (A022 R8).

## What this does not show

- That myths cause behaviour. The play check compares task conditions, not
  myth content.
- How common these patterns are in the full corpus.
- Whether the defector split is about model family or era. Sonnet, from the
  September era, shows the self-narration, which argues against era. But
  only 3 defectors per September model were read.
