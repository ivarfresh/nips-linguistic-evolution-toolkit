# Frozen protocol: Gemini hidden-defector punishment confirmation

Frozen on 2026-08-21 after the three-population mechanism pilot passed and
before any confirmation calls.

## Confirmatory question

Does Gemini 3.1 Flash-Lite independently replicate selective, restrained use
of the current costly-deduction mechanism in complete Myth→Game populations
with hidden mechanical defectors and signed transfer noise?

## Design

Run 10 new eight-agent, ten-round populations with 25% hidden mechanical
defectors (replicate IDs 70–79). The code, model, prompts, temperature,
balanced anonymous schedule, memory regime, myth policy, and economic rules
are unchanged from the pilot:

- every agent writes myths normally;
- mechanical defectors always send, return, and deduct zero without game-
  decision LLM calls;
- ordinary senders receive two optional deduction points after observing a
  return; each spent point costs one and removes up to three from the receiver;
- agents are informed only that communication noise may slightly change the
  amounts they see;
- `U(−1,+1)` noise is applied to sends and returns after the true decisions;
  and
- nine memory exchanges retain three prior Myth→Game→deduction/notice rounds.

There is no no-defector arm. Each treatment population contains both target
classes: its ordinary senders have eight defector-receiver and 22 ordinary-
receiver deduction opportunities. The within-population target contrast is
therefore the efficient direct replication of the selected mechanism. The
three pilot populations, all controlled calibrations, the smoke, and GPT runs
are excluded.

## Frozen outcomes and decision rule

The population is the inferential unit (`n=10`). For each population compute:

1. mean deduction points toward defector minus ordinary receivers; and
2. probability of any deduction toward defector minus ordinary receivers.

Test each contrast against zero with a two-sided one-sample t-test and apply
Holm correction across the two co-primary tests. Also report two-sided 95%
confidence intervals and standardized paired effects.

The mechanism is independently confirmed only if all conditions hold:

- both target contrasts are positive with Holm-adjusted `p<.05`;
- the mean-spending contrast is at least +.50 points;
- the any-deduction contrast is at least +.25;
- at least 50% of pooled defector-target opportunities receive a deduction;
  and
- no more than 25% of pooled opportunities with a visible return ratio of at
  least .5 receive a deduction.

The effect-size and restraint thresholds are unchanged from the pilot. The
Holm-controlled population-level tests are new confirmatory requirements.
Failure of any condition is a failed or unresolved replication, not grounds
for threshold or prompt revision.

Only after the mechanism decision is fixed, report descriptive ordinary-agent
sending, returning, deduction trajectories, and myth cooperation/fairness,
threat, explicit-half, and deduction language. There is no causal condition
contrast for those secondary outcomes in this treatment-only design.

## Acceptance gate

Require exactly:

- 10 completed populations, 100 population-rounds, and 400 dyads;
- 2,400 accepted interaction records: 1,700 Gemini responses, 300 forced
  defector decisions, and 400 scripted deduction notices, apart from recorded
  retries;
- 800 exact signed-noise checks applied after true decisions;
- 300 total deduction opportunities for ordinary senders, split across target
  classes exactly as implied by the frozen seeded schedules and defector
  assignments;
- every defector game and deduction action zero and every defector myth call
  normal;
- exact deduction costs, target losses, payoff floors, and balances;
- correct pairing/noise seeds, target counts, memory horizon, provider/model,
  current prompt variant, and embedded clean Git/config provenance;
- no role, policy, true-transfer, identity, or other hidden-state leakage; and
- no unrecovered provider or response-boundary errors.

If an accepted run lacks clean embedded provenance, exclude it and rerun that
replicate from the committed protocol. Do not inspect outcomes until all ten
populations pass jointly.

## Structural-count amendment (2026-08-22)

This amendment was made after all runs passed the general integrity audit but
before any deduction, cooperation, or myth outcome was read. The prewritten
analyzer stopped at its structural gate because the original acceptance list
incorrectly assumed every population would contain exactly eight defector-
receiver and 22 ordinary-receiver opportunities for ordinary senders.

Balanced pair scheduling guarantees 30 ordinary-sender opportunities per
population; it does not guarantee that the two defectors occupy the receiver
role exactly eight times. Reading only agent types and scheduled roles showed
the following deterministic split for the already frozen seeds:

| Replicate | Defector receiver | Ordinary receiver | Total |
|---:|---:|---:|---:|
| 70 | 9 | 21 | 30 |
| 71 | 10 | 20 | 30 |
| 72 | 8 | 22 | 30 |
| 73 | 8 | 22 | 30 |
| 74 | 9 | 21 | 30 |
| 75 | 8 | 22 | 30 |
| 76 | 9 | 21 | 30 |
| 77 | 8 | 22 | 30 |
| 78 | 9 | 21 | 30 |
| 79 | 8 | 22 | 30 |
| **Total** | **86** | **214** | **300** |

The analyzer now requires exactly this schedule-derived map. No run is added,
removed, or reweighted; the two co-primary population-level contrasts remain
within-population target-type means, so unequal opportunity counts do not
change their definition. All statistical tests, Holm correction, effect-size
thresholds, restraint thresholds, and secondary-outcome lock remain exactly
as frozen above.
