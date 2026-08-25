# Frozen protocol: exploratory punishment-availability factorial

Frozen on 2026-08-21 before running either punishment-unavailable cell. The
punishment-available cells were already run and inspected, so this extension
is explicitly exploratory rather than confirmatory.

## Question

Does making a costly deduction stage available change ordinary GPT-5 Nano
agents' cooperation, and is that change different when an eight-agent
population contains two hidden mechanical defectors?

## Design

Complete a 2×2 design:

| | 0% defectors | 25% hidden defectors |
|---|---|---|
| Deduction unavailable | new run | new run |
| Deduction available | existing run | existing run |

All four cells use eight-agent rotating Myth→Game populations, GPT-5 Nano,
ten rounds, balanced anonymous pairings, signed informed `U(−1,+1)` transfer
noise applied after each decision, ordinary myth writing by every agent, and a
three-complete-round memory-content horizon. Mechanical defectors write myths
normally but always send and return zero without game-decision LLM calls. In
the available cell, they also mechanically deduct zero.

The available cells are the accepted runs in
`data/json/noise_experiments/defector_punishment_gpt_n5_20260821/`, generated
under the previously frozen costly-deduction protocol. The new unavailable
cells use the same replicate IDs 61–65, pairing/noise seed base, and defector
assignment procedure. Capacity is six exchanges without the deduction stage
and nine with it: both values retain three complete prior rounds, because the
available condition has one additional deduction-decision or receiver-notice
exchange per agent-round.

The deduction institution necessarily changes the system rules, gives senders
a two-point post-return opportunity endowment, creates an extra decision or
notice, and can deter behavior before any deduction occurs. These are parts of
the treatment, not separable components of this comparison.

## Frozen estimands

For each replicate and cell, compute population means among ordinary agents:

1. sender proportion sent (`sent / 5`); and
2. receiver return ratio (`returned / received`) when actual received amount
   is positive.

For each outcome, report:

- punishment available minus unavailable at 0% defectors;
- punishment available minus unavailable at 25% defectors;
- 25% minus 0% defectors when punishment is unavailable;
- 25% minus 0% defectors when punishment is available; and
- the difference-in-differences:
  `(available25 − unavailable25) − (available0 − unavailable0)`.

Use replicate-level paired differences, two-sided one-sample t intervals and
tests, and show every population pair. No contrast is confirmatory and no
multiple-testing-adjusted discovery claim will be made.

Secondary descriptive outcomes are round trajectories and ordinary-authored
myth cooperation/fairness, threat/defection, and half-return-rule language.
Do not compare raw final balances across punishment availability: the
available condition gives senders a fixed two-point opportunity endowment.

## Acceptance gate

Before looking at new outcomes, require:

- 10/10 new populations complete all rounds, dyads, game decisions, and myths;
- exactly 1,500 LLM calls plus 100 forced defector game decisions across the
  new cells, apart from explicitly recorded recovered retries;
- 800 exact signed-noise checks, within bounds and applied only after the
  corresponding true decision;
- all defector game actions mechanically zero and all myth calls normal;
- no defector label, forced policy, hidden true transfer, or other privileged
  information in ordinary prompts;
- correct task order, model, memory mode/capacity, pairing regime, and prompt
  additions;
- identical realized schedules across all four cells sharing a replicate ID,
  with matching defector assignments between the two 25% cells; and
- no unrecovered provider or response-boundary failures.

The prior available-stage smoke (replicate 60), all previous defector runs,
and any failed/incomplete output remain excluded.

## Decision rule

The experiment is informative if it distinguishes among three broad patterns:

- **deterrent/helpful:** availability raises sending or returning;
- **crowding-out/harmful:** availability lowers sending or returning; or
- **behaviorally inert:** availability contrasts remain small and uncertain
  despite widespread realized deductions.

A different availability effect with defectors than without them would be an
interaction hypothesis for independent new-seed confirmation, not a confirmed
finding from this reuse of already inspected available-stage cells.
