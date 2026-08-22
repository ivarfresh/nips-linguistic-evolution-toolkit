# Frozen protocol: Gemini punishment-availability factorial confirmation

Frozen on 2026-08-22 after the matched exploratory screen and before any
factorial calls.

## Confirmatory questions

1. Does the negative punishment-availability effect on ordinary receiver
   return ratios independently replicate in populations containing 25% hidden
   mechanical defectors?
2. Is that availability effect moderated by defector presence, or is it a
   general consequence of introducing the deduction institution?

## Design

Run a fully new-seed 2×2 with ten matched population quadruplets (replicate IDs
80–89):

| Deduction stage | Hidden defectors |
|---|---:|
| Unavailable | 0% |
| Unavailable | 25% |
| Available | 0% |
| Available | 25% |

All 40 populations use Gemini 3.1 Flash-Lite, eight agents, ten Myth→Game
rounds, temperature .8, balanced anonymous rotation, normal myth writing,
informed signed `U(−1,+1)` noise applied after true sends and returns, and a
three-round individual-memory content horizon. Within replicate, all four
cells share pairing and noise seeds and realize the same role/partner schedule.
The two defector cells also share the same two hidden defector IDs.

Mechanical defectors always send, return, and deduct zero without game-
decision calls but write myths normally. In available cells, ordinary senders
receive two optional deduction points after the return; each spent point costs
one and removes up to three from the receiver. The unavailable cells contain
no deduction rules or post-game exchange. Memory capacity is six off and nine
on, retaining three complete rounds in both arms.

## Frozen primary analysis

The matched replicate quadruplet is the inferential unit (`n=10`). The outcome
is each population's mean actual return divided by actual received among
ordinary receivers with a positive receipt. Compute for each replicate:

- **direct replication:** available minus unavailable within the 25%-defector
  condition; and
- **interaction:** that defector-cell availability effect minus the available-
  minus-unavailable effect in the 0%-defector condition.

Test both contrasts against zero using two-sided one-sample t-tests. Report 95%
confidence intervals and standardized paired effects, and apply Holm correction
across the two primary tests.

The exploratory negative return effect is independently confirmed only if its
defector-cell estimate is at most `−.025` and Holm-adjusted `p<.05`. Defector
moderation is established only if the interaction magnitude is at least `.025`
and Holm-adjusted `p<.05`; its sign determines whether defectors strengthen or
weaken the institutional effect. Failure of either minimum-effect or testing
criterion is a failed/unresolved result, not grounds to revise thresholds.

Also report, without adding primary tests:

- the availability simple effect in the no-defector cells and the equal-weight
  availability main effect across defector levels;
- condition means and round trajectories;
- the same 2×2 contrasts for ordinary-agent sending;
- exact/rounded return-choice distributions as a post-hoc mechanism diagnostic;
- ordinary-authored myth cooperation/fairness, threat/defection, explicit
  half/equal-return, and punishment/deduction language; and
- selective deduction targeting in available defector cells as a validated
  manipulation check.

Do not compare final-balance levels across availability arms because the
deduction budget changes payoffs mechanically.

## Acceptance gate

Before any outcome analysis, require exactly:

- 40 populations, 400 population-rounds, 1,600 dyads, and 3,200 myths;
- 8,000 accepted interaction records: 6,700 Gemini responses, exactly 500
  forced defector actions, and 800 scripted deduction notices, apart from
  recorded recovered retries;
- 3,200 exact signed-noise checks applied after true decisions;
- cell totals consistent with the design: 1,600 accepted interactions in each
  unavailable cell and 2,400 in each available cell;
- all defector game/deduction actions zero and all defector myth calls normal;
- exact deduction costs, target losses, payoff floors, and cumulative balances;
- correct model/provider, current deduction wording, memory horizon, and clean
  embedded Git/config provenance;
- identical realized schedules and exogenous seeds across all four cells per
  replicate, plus matched defector IDs across the two defector cells; and
- no unrecovered provider/response-boundary error or role, identity, policy,
  true-transfer, or other hidden-state leakage.

If a run lacks clean provenance or fails, exclude it and rerun that same cell
and replicate. Do not inspect outcomes until all 40 accepted populations pass
jointly. All pilot, exploratory, GPT, and earlier-seed runs remain excluded
from the confirmatory estimates.
