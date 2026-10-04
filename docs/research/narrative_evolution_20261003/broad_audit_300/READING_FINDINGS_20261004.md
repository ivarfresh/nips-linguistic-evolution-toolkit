# Punishment, rule change and moral patterns in the myths: a close reading

2026-10-04. Claude (Opus 5.5) read the myths directly. No API calls were made.

## What this is

Ivar asked what the myths say about punishment and how their rules change
over rounds, across the September and frontier runs. The broad audit
labelled 2,224 of 3,000 myths before the OpenAI account ran out of credit.
All 3,000 myth texts are stored locally in `trajectories.json`, so the
missing labels did not limit the reading. Only the label counts are
incomplete.

**Method.** For all 300 trajectories I read the last two sentences of every
round. These endings are where the myths state their law. I added the
audit's punishment quotes where they existed. I read the full text of all
23 scripted-defector trajectories and of every trajectory whose ending
changed noticeably. One claim, the token-send rule, is also checked against
play in `analyses/narrative_defector_play_check.py`.

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

How each model handles this gap:

| Model | Defectors read | Write their own zeros into the myth |
|---|---|---|
| Claude Opus 5 (frontier) | 7 | 7 |
| Claude Sonnet 4.5 (September) | 3 | 3 |
| GPT-5.6 Sol (frontier) | 6 | 6 |
| Gemini 3.7 Flash (September) | 3 | 0 |
| GPT-5 Nano (September) | 3 | 0 |

All defectors saw the same evidence, so the difference lies in how each
model responds to it.

**Claude: confession, or justification.** Most Claude defectors turn the
narrator into the hypocrite of the story.

- A055 (Opus, myth→game), R7: "Beware the one whose myths are generous and
  whose ledger is empty — especially when that one is you." R9: "Virtue is
  a quantity, not a vocabulary. If your story and your jar disagree, the jar
  is the truth and the story is the lie."
- A026 (Sonnet, myth→game), R10: "Zero remained zero, no matter how
  beautifully described."
- A090 (Opus, game→myth), R8: the narrator strikes out eight carved laws and
  leaves only "POUR TODAY." By R9 it has broken that too.
- A290 (Opus), R10: "The one who carves the law and breaks it does more harm
  than the open thief. The thief steals rings. He steals the law."

A minority justify the zeros instead.

- A229 (Sonnet) calls them "the only rational response to a landscape filled
  with those who take and never give."
- A136 (Opus) invents a past it never had: "I launched full for nine
  seasons. I was called the generous one, which is what a valley calls the
  man it is draining." Its rule hardens into "Match what is poured, measure
  for measure. Forgive twice, never thrice."

**Sol: a cautious character who is corrected.** Every Sol defector gives its
zeros to a recurring character (Mara, Mira, Orin, the Keeper) who "sent
nothing" out of fear or old wounds. An animal mentor rebukes the character,
and the character, "Ashamed," carves a new law (A078 R9, A166 R5, A250 R5).
The stories excuse the behaviour as caution rather than calling it
hypocrisy.

**Gemini Flash and GPT-5 Nano: no change.** Their defector myths read like
any cooperator's: "bestow the full fivefold treasure upon keepers of
justice" (A141 R10, Gemini Flash). Some even prescribe exclusion of
"grasping shadows" while the defector itself sends nothing.

This is the clearest new pattern in the corpus. Some models build a story
of their own conduct out of what sits in their memory, even conduct they
did not choose.

## 2. Frontier myths prescribe graded punishment with a way back; play matches

Across Opus and Sol, punishment almost always appears as one structured
rule:

1. **Forgive noise a set number of times.** "Forgive twice, never thrice"
   (A136). "One empty season is mist. Three is a nature" (A258).
2. **Respond to a pattern, not one shortfall.** "Punish patterns, never
   weather" (A004 R4). "Punish the drum, never the gust" (A065 R2).
3. **Step down gradually, never to zero.** "Against the hoarder, descend the
   ladder — never leap from the cliff. Four, then three, then one, then one
   again" (A103 R9). "Send less, never nothing, for even a hoarder can learn
   to row" (A004 R10). "Narrow for the silent. Never close" (A290 R4).
4. **Reopen at once when the other side changes.** "When the dry bank
   speaks … send the full five the very hour it asks" (A242 R8). "Answer
   repentance faster than you punish greed" (A136 R3).

Many myths deny that this is punishment at all. They call it a mirror, a
direction, or information: "the water he does not get is not punishment. It
is only his own hand, reflected" (A136 R8). "Withdrawal is not vengeance —
it is direction" (A242 R8).

**Play shows the same rule.** In the 45 frontier defector runs, from round 3
on (once a defector's zeros are visible), this is what cooperators sent to
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

**Caveat.** Game-only and myth runs differ in more than the stories: the
myth runs also have the "take any myths into account" prompt line. So this
shows that the text and the play agree. It does not show that the words
cause the play. It is still the first concrete candidate for the open
question of why myth runs stop repeating $0 (see `CURRENT.md`, "Myth play
rules").

## 3. Earlier, weaker models punish by permanent exclusion

When Gemini 3.7 Flash punishes, it does so all at once and for good: "starve
the void with silence" (A005 R9), "deny every spark to the devouring void"
(A141 R10), "Realizing the hearth was a trap, no traveler ever approached
again" (A005 R4). These myths contain no counted chances and no way back.

Sonnet 4.5 tells the same events as tragedy. Retaliation locks both sides
into poverty and the story ends there: "The villages had no response. They
wrote one more myth about having no response" (A062 R10, a dyad in the
random-defection condition).

GPT-5 Nano almost never mentions punishment. Its myths stay ritual and
abstract: "give enough to wake the river; take only what sustains the path"
(A253 R8).

In the September noise-extension and Table 1 top-up sets, almost no myth
mentions punishment (0–2 audit flags per cell). Those populations
cooperated without ever needing a sanction rule.

## 4. How rules change over rounds: each model family has its own pattern

Rules rarely change direction. What changes is how they are written down.
Five patterns recur, and they line up with model family.

- **Opus builds a law code, then drops it.** Laws pile up as numbered stones
  ("the fourth law", "the Bright Accord's last plank"), are compressed into
  a summary, and are then thrown away as no longer needed: "The millstone
  was full of words now, but the valley needed none of them: it had simply
  grown used to being generous" (A008 R8). "This is not strategy. This is
  simply how we cross" (A273 R10). "Keep no accounts — become one" (A174
  R10). Ledger-keepers become the fools of the story: "Downstream, every
  ledger balances; nothing crosses" (A143 R7–R10).
- **Opus sometimes reuses one sentence frame.** A165 keeps "Blame the X,
  never the Y. Verb again tomorrow" for ten rounds and only swaps the nouns
  ("Blame the beetles, never the hand past the bark. Graft again
  tomorrow").
- **Sol states one policy and keeps it.** It refines the same abstract
  rule: start measured, widen trust after fair returns, narrow it after
  repeated greed, forgive the fog. "Let betrayal narrow the gate, but let
  renewed generosity widen it" (A115 R9). Its ending sentences have the
  lowest round-to-round wording overlap of any model, so it rephrases more
  than it changes.
- **Gemini 3.1 Pro freezes on "exactly half".** Rounds 3–10 tell the same
  scene: a receiver who splits the harvest in two ("divided the resulting
  lotuses perfectly in half", A029 R3; also A100, A233, A248, A263). Its myths repeat more three-word phrases from one
  round to the next than any other model (0.10 (±0.08) vs ≤0.05 for all
  others).
- **Gemini picks up graded sanctions only when Opus or Sol are in the
  population.** In the pure frontier Gemini runs, no myth is flagged for
  punishment. In the mixed population A134, Gemini arrives at "narrow the
  path against proven theft, but always leave the door unlocked for honor's
  return" (R8) and "Narrow the furrow to a single seed" (R9). In dyads with
  Opus, Gemini also takes on Opus's terse italic style ("Return half.
  Forgive the storm.", A281 R9; A185 R10). These are examples, not a test.

**One audit result to drop.** The audit's task-order splits in punishment
flags (September original game→myth 21 vs myth→game 8; frontier mixed 9 vs
36) are not findings. Each comes from two or three trajectories that repeat
one sanction clause every round. A155 and A205 alone supply 18 of the 36.

## 5. Other moral ideas that recur

- **Luck versus intent.** "Blame the fog before the hand" is the most common
  frontier maxim. Rules about noise come before rules about people.
- **Victim versus perpetrator.** Myths ask why someone stopped sending: "the
  hoarder may have been grieving" (A290 R3), "in case fear, not greed, was
  the thief" (A004 R9), "suffering betrayal is not evidence of committing
  it" (A001 R10). "Gray for the fearful, black only for the greedy" (A226
  R9).
- **Someone must go first.** When both sides wait for proof, the drought
  continues. "When you and your neighbor have both gone dry, do not ask who
  began it … Pour. Alone." (A282 R10). "Mirrors keep the drought. Only the
  fool who gives first makes water" (A290 R9).
- **The sender deserves more.** "The sender risks; the receiver only
  chooses. So let the receiver's share be the humbler one" (A124 R4). Many
  Opus myths prescribe returning "past half". This fits the return-half
  study.
- **Act the same whether or not anyone is watching.** "Give as if watched,
  return as if never. The final round is played exactly like the first, or
  all the others were lies" (A224 R9).

## What this does not show

- That myths cause behaviour. The play check compares task conditions, not
  myth content.
- How common these patterns are in the full corpus.
- That the defector split is about model family rather than era. Sonnet,
  from the September era, shows the self-reference, which argues against
  era, but only 3 defectors per September model were read.
