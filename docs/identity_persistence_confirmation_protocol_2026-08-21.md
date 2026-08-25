# Frozen protocol: independent identity-persistence confirmation

Frozen before generating any outcomes for replicate IDs 15–24 on 2026-08-21.

## Confirmatory question

Does making partner identities persistent across rounds reduce cooperation
relative to otherwise similar round-local pair labels?

## Design

Two new Game-only arms, each with ten independent GPT-5 Nano populations:

- **Stable IDs:** the agent and current co-player receive stable `Member A`–
  `Member H` codes that remain fixed across rounds.
- **Round-local pair IDs:** the same prompt position instead uses `Member Self`
  and `Member Other`, explicitly reassigned each round and unable to identify
  agents over time.

Both prompts contain four matched identity-context bullets, explicitly hide
display names, and state that no population-wide interaction history is shown.
No ledger, dossier, or synthetic self-history block appears in either arm.
Agents retain their own previous three game exchanges in private chat memory.

All other fields are identical: eight agents, ten rounds, balanced rotating
dyads, informed signed `U(−1,+1)` noise after send and return decisions,
temperature .8, and matched pairing/noise seeds `202608215`–`202608224`.
These seeds were not used in the exploratory identity screen.

## Primary outcome and test

Primary contrast:

`Stable IDs − round-local pair IDs`

on average final balance per agent. The independent population is the unit;
the analysis uses ten matched replicate differences, a two-sided paired t test,
a 95% t interval, and paired Cohen's `dz`.

Directional hypothesis: persistent IDs reduce final balance by lowering the
proportion sent.

## Secondary outcomes

- mean proportion sent (behaviorally equivalent to final balance);
- mean return ratio;
- round-one proportion sent, which captures anticipation/framing before any
  partner-specific experience can exist; and
- per-round sending trajectories.

Secondary tests are reported as unadjusted descriptive diagnostics and do not
alter the primary verdict.

## Decision rule

The identity-persistence hypothesis is confirmed only if the two-sided primary
95% CI excludes zero in the predicted negative direction. Otherwise it is not
confirmed, regardless of exploratory effect size or secondary outcomes.

## Acceptance gate

All 20 runs must pass jointly: complete rounds and decisions, strict response
boundaries, memory counts, noise bounds, matched schedules, correct identity
context, and absence of stable Member/Agent IDs from round-local prompts. No
outcomes will be analyzed before this gate passes.
