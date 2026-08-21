# Frozen protocol: public ledger × Myth→Game extension

Frozen before generating the public-ledger Myth→Game outcomes on 2026-08-21.

## Question

Does writing a myth immediately before each game decision increase cooperation
under population-wide public monitoring?

## Design

- Model: `openai/gpt-5-nano`.
- Population: 8 agents in balanced rotating dyads.
- Duration: 10 population rounds.
- New cell: Myth→Game with five independent populations, replicate IDs 0–4.
- Matched comparison: the completed public-ledger Game-only populations with
  the same pairing/noise seeds (`202608200`–`202608204`).
- Public history: all communicated/noisy sends and returns from the previous
  three population rounds, with stable `Member A`–`Member H` pseudonyms and an
  explicit mapping of the current co-player.
- Private memory: the previous three task-rounds. With two tasks per round,
  `memory_capacity: 6` retains the same three-round horizon as the game-only
  cell's capacity of 3.
- Noise: informed, signed `U(−1,+1)`, independently applied after send and
  return decisions using the matched recorded noise seed.
- Myth instruction: the established memory-primary myth templates and exactly
  one game-prompt instruction to take session myths into account.

The public ledger is added to current game prompts. Myth calls do not receive a
fresh ledger block, although their private conversation memory can contain
ledger-bearing game prompts from previous rounds. This is the intended
memory-primary task sequence.

## Planned analysis

The independent population is the analysis unit.

Primary exploratory contrast:

`Myth→Game − Game only | public population ledger`

on average final balance per agent, using the five matched replicate
differences, a two-sided paired t test, and a 95% t interval. Proportion sent is
the behaviorally equivalent cooperation measure because sending creates total
surplus.

Secondary analyses:

- the same contrast for mean return ratio;
- the difference-in-differences against private memory; and
- the corresponding interaction against the current-partner dossier.

The secondary interactions reuse already observed comparison cells and are
therefore not independent confirmation. All analyses remain exploratory at
`n=5` and one model.

## Acceptance gate

No substantive analysis will be reported unless all five runs pass the
expanded joint audit: complete rounds/calls, strict myth and game response
boundaries, exact prompt-memory counts, exact public-ledger reconstruction,
noise bounds, no hidden true-value leakage, and identical realized schedules
across matched seeds. Recovered bounded retries are permitted only when they
are explicit in the audit trail and excluded from accepted memory.
