# D012 — Myth-pressure pilot: word budget × council on Sonnet 4.5 dyads

- Recorded / last verified: 2026-09-28 / 2026-09-28
- Decision status: agreed (pilot only). Stage 2 (eight-agent populations with
  hidden co-player history, transplant and newcomer tests) is proposed, not agreed.
- Scope: 20 Sonnet 4.5 myth-first dyads: word budget (loose 200 / tight
  200→100→60→40→25→20) × council between rounds (off / 2 exchanges) × replicates
  0–4, on the September informed negative-only protocol. A new condition, not a
  change to any existing set.
- Decision authority: Ivar, 2026-09-28 Claude session ("yes please", in reply to the
  assistant's proposal to write the design up, build it on a branch and run the
  dyad pilot). He did not choose between the council and budget-only options, and
  the 2 × 2 design covers both.
- Implementation status: implemented and configured on branch
  `run/myth-pressure-pilot-20260928`. No runs yet.

## Decision and rationale

Test whether pressure turns the myth channel into a shared code, using the
conditions GlossoGen (arXiv 2609.01491) found necessary for LLM language emergence:
a cost on talking and an unbudgeted channel to agree conventions.

- **Explicit** (Ivar, 2026-09-20): the paper would be much stronger if rising
  cooperation went together with language evolution.
- **Inferred** (assistant): a trial reanalysis of the September runs
  (2026-09-22, scratch, not committed) found myth language drifts but does not
  track cooperation. That motivates adding pressure rather than mining existing
  data further.
- Dyads first because they are cheap and Sonnet 4.5 has cooperation headroom there
  (57.8 (±8.2) resources out of 75). Opus 5 dyads sit at the 75 ceiling.

Design and pre-registered criteria:
`docs/research/myth_pressure_pilot_2026-09-28.md`.

## Evidence

- Primary source: Ivar's instructions in the 2026-09-20 to 2026-09-28 Claude
  session (user instruction).
- Implementation: `src/myth_writer.py` (budget, delivery, council prompt),
  `src/simulation.py` (`_run_council`), `experiments/run_noisy_batch.py`
  (`myth_pressure` resolution), `src/experiment_condition.py` (recorded only when
  configured), `tests/test_myth_pressure.py`,
  `scripts/build_myth_pressure_config.py`, `scripts/run_myth_pressure_pilot.py`.
- Runs: none yet.

## Chronology and supersession

- 2026-09-20: GlossoGen read; three reanalysis options proposed; Ivar picks the
  reanalysis of existing runs.
- 2026-09-22: trial reanalysis (scratch) finds no link between language change and
  cooperation.
- 2026-09-28: Ivar approves the pressure pilot.

## Unresolved / next evidence

- Pilot results against criteria 1–3.
- Whether the council's instruction not to discuss amounts holds (criterion 3).
- Stage 2 needs a new decision.
