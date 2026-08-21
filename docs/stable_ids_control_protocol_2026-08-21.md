# Frozen protocol: stable-ID/no-ledger mechanism control

Frozen before generating stable-ID outcomes on 2026-08-21.

## Question

The public-ledger Game-only arm sent less from round 1, before ledger records
existed. Is that early decrease attributable to persistent neutral identities,
or to the public-monitoring/ledger component layered on top of those identities?

## New arm

Five matched GPT-5 Nano Game-only populations receive:

- a stable neutral ID (`Member A`–`Member H`);
- the current co-player's stable neutral ID in every game prompt; and
- an explicit statement that no population-wide interaction history is shown.

They are not told that the population observes a ledger and they receive no
population or co-player history block. Their private conversation memory still
contains their own previous three game exchanges.

Everything else matches the public-ledger and private-memory Game-only cells:
eight agents, ten rounds, balanced rotating dyads, hidden display names,
informed signed `U(−1,+1)` noise after both decisions, and pairing/noise seeds
`202608200`–`202608204`.

## Planned tests

Primary diagnostic:

`public ledger − stable IDs | round-1 proportion sent`.

Round 1 has no behavioral records in either arm, so this contrast isolates the
public-monitoring announcement from persistent pseudonyms. It remains a prompt-
framing contrast and does not separately identify every wording component.

Secondary exploratory contrasts:

- stable IDs minus private memory on round-1 proportion sent;
- stable IDs minus private memory on average final balance;
- public ledger minus stable IDs on average final balance; and
- round trajectories across private memory, stable IDs, and public ledger.

All tests use matched replicate differences, two-sided paired t tests, and 95%
t intervals. No multiplicity-adjusted confirmatory claim will be made at
`n=5`; the arm is a mechanism screen.

## Acceptance gate

Every new run must pass complete-call, response-boundary, memory-count, noise,
and schedule checks. The audit must also verify the exact own/current-partner
ID mappings and the absence of injected population-history rows in every game
prompt.
