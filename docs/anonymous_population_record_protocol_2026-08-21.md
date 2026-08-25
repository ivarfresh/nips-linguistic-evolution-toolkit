# Frozen protocol: anonymous population-record mechanism screen

Frozen before generating outcomes on 2026-08-21.

## Question

Does population-wide social information affect cooperation even when agents
cannot use it to track individual reputations?

## New arm

Five matched GPT-5 Nano Game-only populations see all communicated/noisy sends
and returns from the previous three population rounds. Historical dyads are
shown only as `Pair 1` through `Pair 4`, relabeled independently in each round.
No stable agent ID appears, and the record explicitly states that it cannot
identify the current co-player or track individuals.

Everything else matches the existing Game-only cells: eight agents, ten
rounds, anonymous balanced rotating dyads, three-round private chat memory,
informed signed `U(−1,+1)` noise after both decisions, and matched pairing/noise
seeds `202608200`–`202608204`.

## Comparisons

Primary exploratory contrast:

`anonymous population record − anonymous private memory`

on average final balance per agent. This estimates the bundled effect of
population social information and the knowledge that the anonymous record is
shared, without stable reputation tracking.

Secondary contrasts:

- anonymous record minus stable IDs/no ledger;
- stable-ID public ledger minus anonymous record, which is the closest screen
  of persistent reputation tracking conditional on population records;
- the same comparisons for mean proportion sent and return ratio; and
- round-one differences, before records exist, as prompt-framing diagnostics.

All tests use the independent population as the unit, matched replicate
differences, two-sided paired t tests, and 95% t intervals. Results remain
exploratory at `n=5`, with no confirmatory multiplicity claim.

## Acceptance gate

Every run must pass complete-call, response-boundary, memory-count, noise, and
schedule checks. The audit must reconstruct every anonymous record from saved
communicated transfers and confirm that no stable Member or hidden agent ID is
present in the current record prompt.
