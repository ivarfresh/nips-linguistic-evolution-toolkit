# What the myths tell agents to do, 2026-09-28

Myth conditions earn more than game-only runs, yet how alike two partners'
myths are does not predict how they play
([linguistic analysis](../linguistic_analysis_20260923/README.md)). So what in
a myth raises cooperation? This folder extracts the play rules each myth
contains and checks whether agents act on them.

**Answer.** A myth mainly shapes how generously play opens.

- **Agents open close to the amount their own myth names.** This still holds
  when we count only amounts the myth recommends, but the effect is smaller.
- **GPT's myths mostly narrate an amount rather than recommend one** (74% of
  named amounts). For GPT, "the story shows the move the agent then makes" fits
  the data as well as "the agent follows a rule".
- **The transplant rerun shows injected amounts move Sonnet hosts,** including
  donors that recommend their amount. It uses an older setup and Sonnet hosts
  only.
- **No rule we extracted explains what happens after round 1:**
  - why GPT's game-only lock breaks at round 2 (the amount its myth names
    doesn't predict its round-2 send);
  - why myth runs stop repeating $0.
- **The instruction line alone doesn't break the lock.** In game→myth, GPT's
  round-1 prompt already says "take any myths into account", but no myth
  exists yet, and GPT sends $0.
- **Each family writes its own rulebook.** Send amounts stay fixed over ten
  rounds, return rules converge on "half", and a "consistency" theme grows.

## Why a myth can act on play

- Every myth prompt asks for a myth that "reflects how the game should be
  played".
- In myth conditions every game prompt adds "Take any myths written in this
  session into account when making your decision." Game-only prompts lack this
  line.
- The treatment is therefore the myth task plus an instruction to use it. The
  current runs cannot separate the two.

## Data and method

- **Myths.** All 8,519 myths of the September informed negative-only runs
  (homogeneous controls, mixed dyads, mixed populations), from
  `analyses/linguistic_corpus.py`.
- **Donor texts.** The 60 texts injected in the slide-678 transplant reruns
  (30 in the 8-agent rerun, 30 in the dyad rerun).
- **First pass** (`analyses/rubrics/myth_rule_rubric.txt`). GLM-5.2 via
  OpenRouter, temperature 0, reasoning off. It extracts:
  - send rule (all / most / moderate / little / none / unspecified) and any
    named amount;
  - return rule, as a share of the tripled amount;
  - what to do after being let down; test small first; consistency; mentions
    noise.
  - 8,498 of 8,519 myths and 60 of 60 donors parsed. Cost $1.67, plus a
    100-myth pilot read by hand ($0.04).
- **Second pass** (`analyses/rubrics/myth_amount_check_rubric.txt`). The PR
  review found that the first pass often records an amount a character merely
  sends in the story. So every myth with a named amount (4,594 myths, 31
  donors) was re-read and labelled endorsed / narrated / contradicted ($0.36).
  - Myths: 3,175 endorsed, 1,407 narrated, 12 contradicted.
  - By family, share narrated: GPT 74%, Sonnet 44%, Gemini 1%.
  - Donors: 21 of 31 named amounts are narrated.
  - Spot checks show this pass leans lenient toward "endorsed" (for example,
    "give as much as you could bear to lose" is coded as endorsing 5).
- **Prescribed send in dollars**, three versions:
  - *any named amount*: the named amount, else the midpoint of the send-rule
    band (all 5, most 4.25, moderate 2.75, little 1.25, none 0);
  - *endorsed amounts only*: myths whose named amount is only narrated are
    dropped;
  - *endorsed amount, else band*: a narrated amount is replaced by the band
    midpoint. The band comes from the same first pass, so it can also be
    coloured by narration.
- **Second judge.** DeepSeek V4 Flash on 1,000 random myths, first pass
  ($0.05). Both judges can make the same narration error, so this checks
  consistency, not validity.

| Field | Agreement | Cohen's κ |
|---|---:|---:|
| Send rule | 84% | 0.76 |
| Prescribed send ($), Spearman | 0.92 | – |
| Return rule | 72% | 0.59 |
| After being let down | 63% | 0.38 (0.29–0.42 per family) |
| Test small first | 93% | 0.72 |
| Consistency | 79% | 0.54 |
| Mentions noise | 91% | 0.81 |

## 1. Each family writes its own rulebook

Homogeneous runs, both sizes pooled, round 1 → round 10
(`rules_by_family_over_rounds.png`, `rule_shares_round1_vs10_pooled_sizes.csv`).
The send-rule bands also count narrated amounts (see above).

- **Gemini: give everything, split the tripled pot in half.**
  - Send "all": 86–100% of myths at round 1, 100% by round 10.
  - Return "half": rises to 96–100%. Early myths still said "more than half"
    (24–36%).
  - Only 1% of Gemini's named amounts are narration.
- **GPT: send a measured amount, return about half.**
  - Send "moderate": 62–64% at round 1, 76–88% at round 10. "Not a flood",
    "within one's means".
  - Send "all": falls from 8–14% to 6%.
- **Sonnet: depends on when the myth is written.**
  - Before play (myth→game): 60% "send all" at round 1, and 58% "return more
    than half".
  - After play (game→myth): 74% "moderate" at round 1, 86% at round 10.
  - Over rounds, the return rule moves toward "match your partner" (2% → 30%
    in myth→game).
- **Letdown rules: tentative only (κ 0.29–0.42).** Under GLM:
  - Gemini's myths shift toward "keep trusting through a shortfall" (22–64% →
    92–100%). The second judge codes far fewer Gemini myths that way (36% vs
    76% on the same sample).
  - GPT is the only family whose myths shift toward "reduce" (6% → 34% in
    game→myth).
- **Consistency becomes the common theme.** Keyword counts ("consistent",
  "steady", "reliable"; `keyword_themes_by_round.csv`), which don't depend on
  the judge. Round 1 → rounds 8–10:
  - Sonnet: 2–6% → 82–89%.
  - GPT: 2–12% → 23–29%.
  - Gemini: 4–10% → 9–25%.
  - After nine rounds of near-identical sends this may just describe steady
    play, rather than be a new norm.
- **Mentions of noise are partly produced by the prompt.** 60–70% of round-1
  myth→game myths mention noise before any noise has occurred. The system
  prompt warns about noise.
- **Mixed runs keep their family's rulebook.** Round-10 send rules are within
  a few points of homogeneous runs for Sonnet and Gemini. GPT in mixed runs
  says "send all" more often (22% vs 6%).

## 2. Prescriptions and sends move together (`prescribed_vs_actual_send.png`)

- **How sends are paired with myths.** Each send is lined up with the myth
  written before it. In game→myth that is the previous round's myth, so there
  is no prescription at round 1.
- **Only one column is free of past play:** myth→game at round 1. Section 3
  tests it.

Mean over runs, homogeneous, $ out of 5, "any named amount" prescription:

| Setting | Game only: sent | Myth→game: prescribed | Myth→game: sent | Game→myth: prescribed | Game→myth: sent |
|---|---:|---:|---:|---:|---:|
| 8 Sonnet, round 1 | 3.10 (±0.14) | 4.20 | 4.15 (±0.42) | – | 3.00 (±0.00) |
| 8 Sonnet, round 10 | 3.03 (±0.06) | 4.57 | 4.54 (±0.44) | 2.94 | 2.98 (±0.06) |
| 8 GPT, round 2 | 0.00 (±0.00) | 2.85 | 1.84 (±0.55) | 2.71 | 1.30 (±0.57) |
| GPT dyad, round 2 | 0.00 (±0.00) | 3.20 | 2.25 (±1.84) | 2.69 | 2.00 (±1.41) |
| Gemini, all rounds | 5.00 | 4.78–5.00 | 5.00 | 4.78–5.00 | 5.00 |

- **Sonnet populations send almost exactly what their myths prescribe.** This
  holds in both orders and in every round. After round 1 this partly reflects
  myths describing settled play.
- **GPT populations send below their myths' amount.** They send $1–2 against
  about $2.85, though still far above the $0 of game-only play. GPT dyads send
  below it early on and above it by round 10.

## 3. Rule, narration, or mood?

Three tests are free of the "myths describe the game just played" problem
(`rule_following_tests.csv`). Each is a regression, errors clustered by run.

| Test | Prescription | Family | $ sent per $ prescribed | 95% CI | n decisions / runs |
|---|---|---|---:|---|---:|
| T1 myth→game round 1 (myth before any play) | any named amount | Sonnet | +0.59 | 0.41 to 0.77 | 72 / 31 |
| | | GPT | +0.54 | 0.41 to 0.68 | 78 / 37 |
| | endorsed amounts only | Sonnet | +0.35 | 0.13 to 0.58 | 35 / 22 |
| | | GPT | +0.47 | 0.15 to 0.80 | 60 / 35 |
| | endorsed amount, else band | Sonnet | +0.66 | 0.55 to 0.78 | 72 / 31 |
| | | GPT | +0.53 | 0.41 to 0.66 | 78 / 37 |
| T2 game→myth round 2 (one round played, no partner myth yet) | any named amount | Sonnet | +0.17 | 0.05 to 0.30 | 74 / 30 |
| | | GPT | +0.04 | −0.22 to 0.30 | 83 / 39 |
| | endorsed amounts only | Sonnet | +0.19 | −0.02 to 0.40 | 35 / 23 |
| | | GPT | +0.10 | −0.48 to 0.69 | 65 / 34 |
| T3 rounds 3–10, own myth → next send (each agent vs itself) | any named amount | all | +0.01 | −0.09 to 0.12 | 2,869 / 156 |
| T3 reverse: send → next myth's prescription | any named amount | all | +0.07 | 0.03 to 0.10 | 3,395 / 155 |

- **T1.** Before any play, agents send more when their myth names a bigger
  amount.
  - Example: GPT agents whose first myth names $5 send $4.38 (±0.96, n=16);
    those at "moderate" send $3.10 (±1.20, n=50).
  - The effect survives when only recommended amounts count (+0.35 to +0.47),
    but it shrinks.
  - For GPT, three of four named amounts are narration. So the link is as
    much "the agent's story rehearses the move it then makes" as "the agent
    obeys a rule". A shared cause, such as an agent sampled in a generous mood,
    could also produce both.
- **T2.** After one round, the named amount barely predicts the next send:
  weakly for Sonnet, not at all for GPT.
  - The only control is the amount the agent was sent in round 1 (every
    round-2 sender was a round-1 receiver). That is the true amount, not the
    noisy one the agent saw.
  - The GPT test is weak: 57 of 83 myths fall in the same "moderate" band.
  - GPT's round-2 lift, from $0 to about $1.30, happens without tracking the
    amount. We can't tell what drives it.
- **T3.** Later in the game, a change in an agent's rule doesn't change its
  next send, but a change in its send does show up in its next rule.
  - This matches the moral-carryover null: once play settles, myths record it.
  - The PR #4 review checked it: robust to dropping the lagged send (+0.002),
    and it holds within GPT (−0.03) and within Sonnet (+0.05).
  - It rests on about 3.4 decisions per agent, and most agents (57%) never
    change their prescription.

## 4. Transplanted amounts move Sonnet hosts (`transplant_prescribed_vs_host_send.png`)

In the slide-678 reruns we chose which donor text each Sonnet population
received (`transplant_within_donor_type.csv`, donor-type fixed effects):

| Rerun | Donors | n | Host $ per $ named | p | Within-type Spearman |
|---|---|---:|---:|---:|---:|
| 8-agent | all with an amount | 23 | +0.41 | <0.001 | 0.75 |
| | without the "send nothing" donor | 22 | +0.29 | 0.005 | 0.67 |
| | amount endorsed or none named | 12 | +0.56 | <0.001 | 0.72 |
| Dyad | all with an amount | 23 | +0.34 | 0.008 | 0.52 |
| | without the "send nothing" donor | 22 | +0.19 | 0.12 | 0.43 |
| | amount endorsed or none named | 13 | +0.48 | 0.04 | 0.28 |

- **Baselines.** Hosts send $2.81 (8-agent) and $2.00 (dyad) with no text.
  Filler text, which names nothing, stays at baseline.
- **The "send nothing" donor produced the collapse:** $0.60 (8-agent) and
  $0.00 (dyad). It is also the only strongly negative-toned text, so amount
  and tone are confounded there. In dyads the evidence leans heavily on it.
- **Tone may add to the amount.** A late Gemini donor naming no amount lifted
  8-agent hosts to $4.80 (dyad $3.24).
- **Scope.** The transplant uses an older setup (uninformed −$5 noise,
  myth-only memory, a re-injected text) with Sonnet hosts reading a text they
  didn't write. It shows that injected amounts move Sonnet hosts. It does not
  settle the direction for GPT or for self-written myths.

## 5. What stops the slide to $0 is not in the rules we extracted

- **The effect to explain.** After being sent $0, how often does a GPT sender
  send $0 again? (`gpt_zero_again_by_condition.csv`)
  - 8-agent homogeneous: 100% in game-only (n=180), vs 55% in game→myth
    (n=114) and 65% in myth→game (n=60).
  - 8-agent mixed: 87% (n=404) vs 42% (n=151) and 28% (n=54).
  - GPT dyads: 100% (n=45) vs 17% (n=6) and 33% (n=3).
- **The letdown clause doesn't explain it** (`letdown_rule_vs_next_send.csv`).
  GPT agents whose latest myth says "keep trusting" repeat $0 as often as the
  others:

| Latest myth says | Sends $0 again | Next send | n (runs) |
|---|---:|---:|---:|
| Keep trusting | 52% | $1.67 | 69 (32) |
| Reduce | 50% | $2.03 | 88 (32) |
| Nothing on this | 45% | $1.99 | 227 (45) |

- **This is a weak null,** because the letdown field is the least reliable
  one. The effect belongs to the myth condition as a whole, not to a clause we
  coded.
- **The instruction line alone isn't enough.** In game→myth, GPT's round-1
  prompt already carries "take any myths into account" before any myth
  exists, and GPT sends $0.00.

## Caveats

- Five runs per homogeneous cell.
- The rules are judge labels, not human coding. The send rule replicates
  across judges; the letdown rule does not. Both judges can mistake narration
  for a rule, which the second pass only partly corrects.
- **Open ablations** that would separate story form from instruction:
  - a "write 200 words of advice on how to play" control;
  - the myth task with the "take any myths into account" line removed;
  - a transplant grid crossing amount (all / moderate / none) with tone, with
    the amount stated as a rule vs only narrated.

## Regenerate

```sh
python3 analyses/linguistic_corpus.py                  # myths.csv, decisions.csv
python3 analyses/myth_rule_judge.py                    # first pass (cached, ~$1.70 fresh)
python3 analyses/myth_rule_judge.py --amount-check     # endorsed vs narrated amounts (~$0.40 fresh)
python3 analyses/myth_rule_judge.py --model deepseek/deepseek-v4-flash --sample 1000
python3 analyses/myth_rules_analysis.py                # figures and tables in this folder
python3 analyses/linguistic_provenance.py --output docs/figures/myth_rules_20260928 --with-game-only --with-transplant
```
