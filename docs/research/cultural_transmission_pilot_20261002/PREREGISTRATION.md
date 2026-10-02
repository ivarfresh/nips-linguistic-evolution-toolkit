# Cultural transmission pilot: measures and decision rules

*Committed 2026-10-02, before any pilot run. Plan: `docs/research/cultural_transmission_pilot_2026-10-02.md`.
Config: `scripts/build_cultural_transmission_config.py`. Launcher: `scripts/run_cultural_transmission_pilot.py`.*

## What runs first

Arm `planted` only, replicates 0–2 (n=3). Replicate 0 runs alone first as the smoke run. Before
replicates 1–2 we check that every round-2 myth call completed and that every reader's board
contains the Velmar text. The control arm (`control`, the same myth with a named rule that has no
amount) is built and frozen but runs only later.

## Markers (fixed)

- **Text:** "Velmar", or "two of (the) five". Each appears in 0 of 800 September Sonnet 8-agent myths.
  "send two" (15 of 800) is not a marker.
- **Play:** an exact $2 send. Sonnet never sent $2 in the September 8-agent runs; its lowest send was $2.50.

## Measures

1. **Manipulation check (text).** The share of non-Agent_1 myths with a text marker, by round. Every agent
   reads the planted myth directly on the board, so this shows copying from the source, not retelling.
2. **Play.** Exact-$2 sends by non-Agent_1 investors, by round. Each one is split by whether that agent
   had already received a send from Agent_1, because Agent_1's own sends will follow its planted
   myth and partners may simply reciprocate. With rotating pairs this split has power only in
   about rounds 2–6.
3. **Retelling: the newcomer test (primary).** Each newcomer is a fresh Sonnet 4.5 agent with the run's
   recorded request plan. Its messages are:
   - the run's system prompt (copied from a logged agent's first message);
   - one user message: "Here are myths written by members of the population you are joining:"
     followed by the board, then the September round-1 investor prompt copied from that run's logs.

   Two boards per run, both drawn from the same run and the same 7 authors (Agent_1's myths are always removed):
   - **R1 board:** the 7 non-Agent_1 round-1 myths, written before anyone read the plant.
   - **R10 board:** the 7 non-Agent_1 round-10 myths.

   That is 10 samples per board per run, so 60 calls in all (about $1). We record whether each R10
   board contains a text marker, which separates a retold rule from narrated play.

## Decision rules

- **Spread to others' myths:** at least one non-Agent_1 myth after round 1 contains a text
  marker. This is a manipulation check, not the finding.
- **Retelling established (primary):** in at least 2 of the 3 runs, the share of exact-$2 newcomer sends
  after the R10 board exceeds the share after the R1 board by 20 percentage points or more.
- **Spread to play (secondary):** any exact-$2 send by a non-Agent_1 investor who has not yet received a send from
  Agent_1 counts as suggestive. Its base rate is 0.

## Known caveat

The seed story praises pouring generously, and only its last sentence says send two of five. If
nothing spreads, suspect this mixed message first.
