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
- 80 defector-target and 220 ordinary-target deduction opportunities for
  ordinary senders;
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
