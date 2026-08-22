# Frozen protocol: exploratory matched punishment-availability follow-up

Frozen on 2026-08-22 after the punishment-on Gemini confirmation outcomes
were known and before any punishment-off calls.

## Question

Within eight-agent Myth→Game populations containing 25% hidden mechanical
defectors, does making the validated costly-deduction stage available change
ordinary-agent sending, returning, or myth language?

This is explicitly exploratory. The ten punishment-on populations (replicates
70–79) already exist and were inspected before this follow-up was designed.
The matched comparison can screen a causal availability effect efficiently,
but any positive finding requires an independent new-seed factorial.

## Design

Run only the ten punishment-unavailable populations for replicate IDs 70–79,
using the unchanged non-punishment 25%-defector configuration. Pair each with
its completed punishment-on population. The arms match on:

- Gemini 3.1 Flash-Lite, temperature .8;
- eight agents, ten rounds, Myth→Game order, ordinary myth writing by all;
- two hidden mechanical defectors that always send and return zero;
- balanced anonymous schedules, defector assignments, pairing seeds, and
  signed-noise seeds;
- informed `U(−1,+1)` send/return noise applied after true decisions; and
- three prior complete rounds of individual chat memory.

The off arm uses memory capacity six for Myth→Game; the on arm uses nine for
Myth→Game→deduction/notice. Both represent the same three-round content
horizon. The additional post-game exchange and economic consequences are part
of the punishment-availability treatment.

## Frozen analysis

The population is the paired unit (`n=10`). Compute punishment-on minus
punishment-off differences for two co-primary behavioral outcomes:

1. mean proportion sent by ordinary agents; and
2. mean return ratio among ordinary receivers with a positive amount received.

Report two-sided paired t-tests, 95% confidence intervals, standardized paired
effects, and Holm-adjusted p-values across the two co-primary outcomes. Also
report round trajectories and the count of pairs in each direction.

Secondary paired outcomes are ordinary-authored myth cooperation/fairness
density, threat/defection density, explicit half/equal-return frequency, and
punishment/deduction density. Report estimates and 95% intervals descriptively;
do not treat them as confirmatory. Do not compare final balances: the on arm
adds two deduction points to every sender's post-game budget, so its level is
mechanically different.

The screen merits a fully independent new-seed 2×2 availability × defector
factorial if either co-primary outcome has Holm `p<.05`, or if the sending
effect has magnitude at least .03 with at least eight of ten paired differences
in the same direction. This escalation rule is a design decision, not a claim
that the selected effect is confirmed.

## Acceptance gate

Before outcome analysis, require the new off arm to have exactly:

- 10 populations, 100 population-rounds, and 400 dyads;
- 1,600 accepted interaction records: 1,400 Gemini responses and exactly 200
  forced defector game decisions, apart from recorded recovered retries;
- no deduction decisions or receiver notices;
- 800 exact signed-noise checks applied after true decisions;
- every defector game action zero and every defector myth call normal;
- correct memory horizon, model/provider, prompt, and clean embedded Git/config
  provenance; and
- no unrecovered provider errors, response-boundary failures, role/identity
  leakage, true-transfer leakage, or balance inconsistencies.

Jointly audit the existing on arm again. For every replicate require identical
realized role/pair schedules, defector IDs, pairing seeds, and noise seeds
across availability arms. If a new run lacks clean provenance, exclude it and
rerun the same replicate before opening outcomes.
