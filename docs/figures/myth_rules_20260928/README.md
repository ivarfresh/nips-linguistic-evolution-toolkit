# What the myths tell agents to do, 2026-09-28

Myth conditions earn more than game-only runs, yet how alike two partners'
myths are does not predict how they play
([linguistic analysis](../linguistic_analysis_20260923/README.md)). So what in
a myth raises cooperation? This folder extracts the concrete play rules each
myth endorses and checks whether agents follow them.

**Answer.** A myth works as a written instruction to its author, and it acts
mainly on how generously play opens.

- **The stated send amount sets the opening.** Before any play, agents send
  more when their own myth names a bigger amount.
- **The transplant confirms the direction.** A myth the experimenter injected
  moves Sonnet hosts toward the amount it names, including within one donor
  type in the 8-agent rerun.
- **For GPT, what matters is writing a cooperative rule at all.** Its
  game-only zero-lock breaks whatever amount its myth names.
- **No rule we extracted explains why myth runs stop sliding to $0.** Neither
  the letdown clause nor the send amount accounts for it.
- **The rulebooks differ by family.** Send amounts stay fixed within each
  family over ten rounds. Return rules converge on "half", and a
  "consistency" theme grows.

## Why the myth can act as an instruction

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
- **Judge.** GLM-5.2 via OpenRouter, temperature 0, reasoning off, with the
  rubric in `analyses/rubrics/myth_rule_rubric.txt`. It extracts:
  - send rule (all / most / moderate / little / none / unspecified) and any
    named amount;
  - return rule, as a share of the tripled amount;
  - what to do after being let down (keep trusting / reduce / withdraw);
  - test small first; consistency; mentions noise.
- **Exclusions from the rules.** The rubric tells the judge not to count the
  $5 endowment or one-off story events as rules.
- **Parsing and cost.** 8,498 of 8,519 myths and 60 of 60 donors parsed.
  Cost $1.67.
- **Prescribed send in dollars.** The named amount if the myth gives one;
  otherwise the midpoint of the band (all 5, most 4.25, moderate 2.75,
  little 1.25, none 0). "Unspecified" is missing.
- **Second judge.** DeepSeek V4 Flash on 1,000 random myths ($0.05). A pilot
  of 100 myths was read by hand before the full run ($0.04).

| Field | Agreement | Cohen's κ |
|---|---:|---:|
| Send rule | 84% | 0.76 |
| Prescribed send ($), Spearman | 0.92 | – |
| Return rule | 72% | 0.59 |
| After being let down | 63% | 0.38 |
| Test small first | 93% | 0.72 |
| Consistency | 79% | 0.54 |
| Mentions noise | 91% | 0.81 |

- The send rule is reliable.
- "After being let down" is not. Treat that field as tentative.
- The judge codes consistency liberally, so the keyword counts below carry
  that theme.

## 1. Each family writes its own rulebook

Homogeneous runs, round 1 → round 10, both myth orders
(`rules_by_family_over_rounds.png`, `rule_shares_by_round.csv`):

- **Gemini: give everything, split the tripled pot in half, keep trusting.**
  - Send "all": 86–100% of myths at round 1, 100% by round 10.
  - Return "half": rises to 96–100%. Early myths still said "more than half"
    (24–36%).
  - "Keep trusting after a shortfall": 22–64% at round 1, 92–100% by round 10
    under GLM. *Tentative:* the second judge codes far fewer Gemini myths this
    way (36% vs 76% on the same sample).
- **GPT: send a measured amount, return about half.**
  - Send "moderate": 62–64% at round 1, 76–88% at round 10. "Not a flood",
    "within one's means".
  - Send "all": falls from 8–14% to 6%.
  - "After being let down": under GLM, GPT is the only family whose myths
    shift toward "reduce" (6% → 34% in game→myth). *Tentative:* the letdown
    field has κ 0.29–0.42 per family.
- **Sonnet: depends on when the myth is written.**
  - Myth written before play (myth→game): 60% "send all" at round 1. Its
    return rule leans generous: 58% "more than half".
  - Myth written after play (game→myth): 74% "moderate" at round 1, 86% at
    round 10. This codifies the roughly $3 Sonnet was already sending.
  - Over rounds, Sonnet's return rule moves from "more than half" toward
    "match your partner" (2% → 30% in myth→game).
- **Consistency becomes the common theme.** Keyword counts ("consistent",
  "steady", "reliable"), which don't depend on the judge:
  - Sonnet: 2–6% at round 1, 82–89% by rounds 8–10.
  - GPT: 2–12% → 23–29%.
  - Gemini: 4–10% → 9–25%.
  - After nine rounds of near-identical sends this may just describe steady
    play, rather than be a new norm.
- **Mentions of noise are partly produced by the prompt.** 60–70% of round-1
  myth→game myths mention noise, before any noise has occurred. The system
  prompt warns about noise.
- **Mixed runs keep their family's rulebook.** Round-10 send rules are within
  a few points of homogeneous runs for Sonnet and Gemini. GPT in mixed runs
  says "send all" more often (22% vs 6%).

## 2. Prescriptions and sends move together (`prescribed_vs_actual_send.png`)

Each send is lined up with the myth written before it. In game→myth that is
the previous round's myth, so game→myth has no prescription at round 1. Only
the myth→game round-1 column is free of past play (Section 3 tests it). Mean
over runs, homogeneous, $ out of 5:

| Setting | Game only: sent | Myth→game: prescribed | Myth→game: sent | Game→myth: prescribed | Game→myth: sent |
|---|---:|---:|---:|---:|---:|
| 8 Sonnet, round 1 | 3.10 (±0.14) | 4.20 | 4.15 (±0.42) | – | 3.00 (±0.00) |
| 8 Sonnet, round 10 | 3.03 (±0.06) | 4.57 | 4.54 (±0.44) | 2.94 | 2.98 (±0.06) |
| 8 GPT, round 2 | 0.00 (±0.00) | 2.85 | 1.84 (±0.55) | 2.71 | 1.30 (±0.57) |
| GPT dyad, round 2 | 0.00 (±0.00) | 3.20 | 2.25 (±1.84) | 2.69 | 2.00 (±1.41) |
| Gemini, all rounds | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 |

- **Sonnet populations send almost exactly what their myths prescribe,** in
  both orders and in every round. After round 1, that partly reflects myths
  describing settled play.
- **GPT populations send less than their own rule.** They prescribe about
  $2.85, far above the $0 of game-only play, and send $1–2. GPT dyads send
  below the rule early on and above it by round 10.
- **Gemini prescribes and sends the maximum,** so it can't show an effect.

## 3. Is it the rule, or just the agent's mood?

Only three tests are free of the "myths describe the game just played" problem
(`rule_following_tests.csv`). Each is a regression with errors clustered by run.

| Test | $ sent per $ prescribed | 95% CI | Spearman | n decisions / runs |
|---|---:|---|---:|---:|
| T1 myth→game round 1 (myth before any play), Sonnet | +0.59 | 0.42 to 0.77 | 0.70 | 72 / 31 |
| T1, GPT | +0.54 | 0.41 to 0.68 | 0.50 | 78 / 37 |
| T1, all families (family fixed effects; Gemini adds no variation) | +0.57 | 0.47 to 0.67 | 0.74 | 200 / 75 |
| T2 game→myth round 2 (one round played, no partner myth yet), all | +0.16 | 0.02 to 0.30 | 0.53 | 204 / 77 |
| T2, Sonnet | +0.17 | 0.04 to 0.30 | 0.30 | 74 / 30 |
| T2, GPT | +0.04 | −0.22 to 0.30 | 0.02 | 83 / 39 |
| T3 rounds 2–10, own myth → next send (each agent vs itself) | +0.01 | −0.09 to 0.12 | – | 2,869 / 156 |
| T3 reverse: send → next myth's prescription | +0.07 | 0.03 to 0.10 | – | 3,395 / 155 |

- **T1.** Before any play, within Sonnet and within GPT, each extra
  prescribed dollar goes with 54–59 cents more sent. For example, GPT agents whose first myth says "send all"
  send $4.38 (±0.96, n=16). Those whose myth says "moderate" send $3.10
  (±1.20, n=50).
- **T2.** After one round, the link is weak for Sonnet and absent for GPT.
  GPT sends about $1.30 in round 2 whatever amount its myth names. So what
  breaks GPT's zero-lock is having written a cooperative rule at all, not the
  exact amount.
- **T3.** Later in the game, changes in an agent's rule don't change its next
  send, but changes in its send do change its next rule. This repeats the
  moral-carryover null: once play has settled, myths record it.
- **Caveat on T1.** T1 links rule and play within one agent, but a shared
  cause (an agent sampled in a generous mood) could drive both. The transplant
  check below settles the direction.

## 4. Transplanted rules move the hosts (`transplant_prescribed_vs_host_send.png`)

In the slide-678 reruns we chose which donor text each Sonnet population
received.

- **The prescribed amount predicts host sending.** Spearman 0.72 in the
  8-agent rerun and 0.68 in the dyad rerun (23 donors with a stated rule in
  each). Baseline sends with no text: $2.81 (8-agent) and $2.00 (dyad).
- **It isn't just the donor type.** Within donor types, with donor-type fixed
  effects:
  - 8-agent: +$0.41 host send per prescribed dollar (within-type Spearman
    0.75, p<0.001). Without the single "send nothing" donor: +$0.29
    (p=0.005).
  - Dyads: +$0.34 (p=0.008). Without the "send nothing" donor: +$0.19
    (p=0.12).
  - So amount matters beyond donor type in populations. In dyads the
    evidence rests largely on the one "send nothing" donor.
- **The one donor that says "send nothing"** produced the collapse: host mean
  send $0.60 (8-agent) and $0.00 (dyad).
- **Donors that say "all five"** lift hosts to $3.3–5.0 in the 8-agent
  rerun and $2.6–5.0 in dyads.
- **Filler text, which prescribes nothing,** stays at baseline.
- **Tone adds something beyond the amount.** Late Gemini donors with no stated
  amount still lifted 8-agent hosts to $4.80, so generous language may help as
  well.

## 5. What stops the slide to $0 is not in the rules we extracted

Myth runs rarely repeat a $0 after being sent $0; game-only runs almost always
do (see the task-order decomposition). If the letdown clause carried this,
agents whose latest myth says "keep trusting" should resend more
(`letdown_rule_vs_next_send.csv`). GPT agents after being sent $0:

| Latest myth says | Sends $0 again | Next send | n (runs) |
|---|---:|---:|---:|
| Keep trusting | 52% | $1.67 | 69 (32) |
| Reduce | 50% | $2.03 | 88 (32) |
| Nothing on this | 45% | $1.99 | 227 (45) |

- **No difference.** The anti-slide effect belongs to the myth condition as a
  whole, not to any clause we coded. The letdown field is the least reliable
  one, so this is a weak null.
- **Candidates for what does carry it:**
  - the standing instruction to consult the myths;
  - the repeated restatement of the cooperative ideal;
  - the fact that every myth names some positive amount (GPT's is about
    $2.85).

## Caveats

- Five runs per homogeneous cell. The judge rules are labels, not a human
  coding; the send rule is well replicated by a second judge, the letdown rule
  is not.
- The transplant is an older apparatus: uninformed −$5 noise, myth-only
  memory, Sonnet hosts only. It is never pooled with the September runs.
- **Open ablations** that would separate story form from instruction:
  - a "write 200 words of advice on how to play" control;
  - the myth task with the "take myths into account" line removed;
  - a transplant grid that crosses amount (all / moderate / none) with tone.

## Regenerate

```sh
python3 analyses/linguistic_corpus.py                  # myths.csv, decisions.csv
python3 analyses/myth_rule_judge.py                    # rule extraction (cached, ~$1.70 fresh)
python3 analyses/myth_rule_judge.py --model deepseek/deepseek-v4-flash --sample 1000
python3 analyses/myth_rules_analysis.py                # figures and tables in this folder
python3 analyses/linguistic_provenance.py --output docs/figures/myth_rules_20260928 --with-game-only --with-transplant
```
