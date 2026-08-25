# Frozen screen: Gemini 3.7 Flash with defectors and punishment

Frozen on 2026-08-23 after the baseline ceiling screen and controlled
deduction calibration, before any population outcome in this design was
generated.

## Questions

1. Does exposure to 25% hidden mechanical defectors create behavioral headroom
   in otherwise ceiling-limited Gemini 3.7 populations?
2. Does the model's selective controlled punishment policy survive anonymous
   rotation, signed communication noise, myths, and accumulated memory?
3. Does the defector-dependent reduction in return generosity found in Gemini
   3.1 Flash-Lite appear directionally in Gemini 3.7?

## Design

Run three matched population quadruplets (replicates 93–95) crossing:

| Deduction stage | Hidden defectors |
|---|---:|
| Unavailable | 0% |
| Unavailable | 25% |
| Available | 0% |
| Available | 25% |

All 12 populations use Gemini 3.7 Flash through the direct Google API at
medium thinking, eight agents, ten Myth→Game rounds, balanced anonymous
rotation, normal myth writing, three complete rounds of individual-memory
content, and informed signed `U(-1,+1)` noise applied after true decisions.
Within replicate, all four cells share schedules and noise seeds; defector cells
share the same two hidden defector IDs.

Mechanical defectors always send, return, and deduct zero without game-
decision calls but write myths normally. Available-stage ordinary senders have
two optional points after observing the return; each spent point costs one and
removes up to three from the receiver. Memory capacities are six exchanges off
and nine on, preserving the same three-round content horizon.

## Acceptance gate

Before outcomes, require exactly:

- 12 populations, 120 rounds, 480 dyads, and 960 myths;
- 2,400 accepted interactions: 2,010 Gemini calls, 150 forced defector
  actions, and 240 scripted deduction notices;
- 960 exact post-decision noise checks;
- correct model/provider, medium thinking, omitted temperature, recorded
  300-second timeout, action policies, prompts, memory, and accounting;
- matched schedules/seeds across all cells and matched defector IDs;
- clean single-commit/config provenance; and
- no unrecovered provider, response-boundary, identity, role, policy, or true-
  transfer leakage.

## Frozen screen metrics and escalation

The population is the analysis unit. With only three quadruplets, intervals and
p-values are descriptive.

**Live targeting gate.** Within available 25%-defector populations, require:

- defector-minus-ordinary mean deduction intensity at least +1 point;
- defector-minus-ordinary probability of any deduction at least +.50;
- a positive intensity difference in all three populations; and
- zero deductions after any visible return ratio of at least one half.

**Return-crowding gate.** Compute available minus unavailable ordinary-receiver
return ratios. A defector-cell signal requires an estimate at most `-.025` and
negative differences in at least two of three pairs. A moderation signal also
requires the availability-by-defector interaction at most `-.025` with the
same direction in at least two pairs.

If live targeting passes and both return gates pass, expand the complete 2x2 to
ten new-seed quadruplets. If targeting and the defector-cell return gate pass
but moderation does not, add only matched available/unavailable defector cells.
Otherwise do not scale this population design.

Also report sending, exact return choices, round trajectories, ordinary- and
defector-authored myth language, and list-price cost. These do not add gates.
Do not compare final balances across deduction availability because deduction
budgets change payoffs mechanically.
