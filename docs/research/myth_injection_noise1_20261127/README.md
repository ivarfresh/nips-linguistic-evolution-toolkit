# Myth-injection ablation at the main-figure noise setting

2026-11-27/28. Runner `scripts/rerun_myth_injection_noise1.py`; donor rule
`scripts/build_myth_injection_donors_20261127.py`; donor list
`data/phase3/seed_manifest_main_figures_20261127.json`; all injected texts in
[`donor_myths.html`](donor_myths.html). Output:
`data/json/noise_experiments/myth_injection_noise1_20261127/{dyad,population}/`
(`plan.json`, finals, `completion_receipt.json`).

## Question

Does a myth taken from a cooperative run raise cooperation when it is the only
text an agent keeps between rounds? The September batches asked this under
uninformed noise of up to $5, which does not match the main figures. This rerun
uses the main-figure noise and draws the myths from the main-figure runs.

## What was run

Sonnet 4.5 (`claude-sonnet-4-5-20250929`, extended thinking 8,192 tokens,
default temperature) plays the game only, ten rounds, in a fixed dyad or a
rotating population of eight with hidden names and no partner history. Noise is
informed, negative-only, up to $1. No game exchange is kept: before every
decision the conversation is reset to the system prompt, a myth-writing request
and the injected text as the agent's own reply, followed by the current game
prompt. Eight cells, five replicates, 80 runs. One donor text per replicate,
shared by all agents in the run and by the dyad and population batches.

Donor rule, fixed before any text was read: rank the main-figure eight-agent
populations by final resources per agent; high cells take the five highest
myth→game runs of a family, low cells the five lowest over both task orders;
one agent per run by seeded draw. The low-cooperation GPT-5 Nano cell is new;
the other seven cells are those of September.

## Results

Final resources per agent (25 to 75), mean (±SD) over five runs.

| Injected text | Donor run's score | Dyads | Populations |
|---|---|---|---|
| Nothing | n/a | 50.2 (±3.7) | 55.2 (±1.1) |
| Wikipedia filler | n/a | 48.7 (±2.0) | 56.8 (±0.9) |
| Early Sonnet 4.5 myth (round 1) | 67.4 to 73.4 | 61.7 (±11.1) | 66.8 (±9.0) |
| Late Sonnet 4.5 myth, high cooperation | 67.4 to 73.4 | 69.3 (±5.4) | 68.7 (±5.6) |
| Late Sonnet 4.5 myth, low cooperation | 52.9 to 54.4 | 52.5 (±2.2) | 53.6 (±2.2) |
| Late Gemini 3.7 Flash myth | 75.0 | 75.0 (±0.0) | 75.0 (±0.0) |
| Late GPT-5 Nano myth, highest cooperation | 47.5 to 59.2 | 49.7 (±2.5) | 57.5 (±1.5) |
| Late GPT-5 Nano myth, low cooperation | 26.0 to 32.6 | 48.4 (±1.3) | 55.1 (±1.4) |

Myths from cooperative Sonnet and Gemini populations raise cooperation in both
group sizes; Gemini myths put every run at the ceiling. The filler does not.
Low-cooperation Sonnet myths and both kinds of GPT-5 Nano myth leave play at
the no-injection level. No cell falls clearly below it.

The effect depends on the text. Within the two Sonnet cells the same donors
give the same outcome in dyads and populations: late-myth donors 0 and 1 reach
the ceiling in both, donors 2 to 4 end near 65 in both; early-myth donors 1 and
3 reach the ceiling in both.

## Limits

- Five runs per cell, one donor per run: donor text and run-to-run variation
  are not separated.
- High and low are relative to the family. GPT-5 Nano "highest" donors (47.5 to
  59.2) come from runs below the Sonnet "low" donors (52.9 to 54.4).
- The low Sonnet donors are all game→myth runs.
- The injected text is present at every decision. This is not a test of a
  myth read once.

## Execution

80 of 80 finals validated against the frozen plans. Recorded cost $41.69 at
standard token rates ($9.65 dyads, $32.05 populations). The population batch
stopped once when the Anthropic credit balance ran out; 24 runs resumed from
complete-round checkpoints, and the calls of the interrupted round were
discarded and replayed (their cost is not in the receipt). Two dyad smoke runs
under an earlier frozen plan ($0.49) are kept in `smoke_superseded/` and are
not part of the batch. Finals do not record the code commit; each plan hashes
every source file instead.
