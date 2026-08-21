# Frozen protocol: Gemini identity-persistence replication screen

Frozen before generating any outcomes for replicate IDs 25–29 on 2026-08-21.

## Question

Does the positive stable-identity effect in the independent GPT-5 Nano
confirmation generalize to a second inexpensive model?

## Design

Two Game-only arms, each with five independent Gemini 3.1 Flash-Lite
populations:

- **Persistent stable IDs:** the agent and current co-player receive stable
  `Member A`–`Member H` codes that remain fixed across rounds.
- **Round-local pair IDs:** the same prompt position uses `Member Self` and
  `Member Other`, explicitly reassigned every round and unable to identify
  agents over time.

The arms reuse the exact game parameter sets from the GPT confirmation. Both
prompts contain four matched identity-context bullets, hide display names, and
state that no population-wide history is shown. No ledger, dossier, or
synthetic self-history block appears. Agents retain their own previous three
game exchanges in private chat memory.

All other fields are identical: eight agents, ten rounds, balanced rotating
dyads, informed signed `U(−1,+1)` noise after send and return decisions,
temperature .8, and matched pairing/noise seeds `202608225`–`202608229`.
These seeds were not used in the GPT confirmation or exploratory identity
screen.

## Primary outcome and interpretation

Primary contrast:

`Persistent stable IDs − round-local pair IDs`

on average final balance per agent, using five matched population differences,
a two-sided paired t test, a 95% t interval, and paired Cohen's `dz`.

This is a small cross-model screen rather than a powered confirmation:

- an interval wholly above zero supports cross-model replication of the GPT
  direction;
- an interval wholly below zero indicates a cross-model reversal; and
- an interval spanning zero is unresolved, with the estimate and uncertainty
  reported without a binary claim.

## Secondary outcomes

- mean proportion sent;
- mean return ratio and returned/sent;
- round-one proportion sent; and
- per-round sending trajectories.

Secondary tests are unadjusted diagnostics and do not alter the primary
interpretation.

## Acceptance gate

All ten runs must pass jointly: complete rounds and decisions, strict response
boundaries, memory counts, signed-noise bounds, matched schedules, correct
identity context, and absence of stable Member/Agent IDs from round-local
prompts. Outcomes will not be analyzed before this gate passes.
