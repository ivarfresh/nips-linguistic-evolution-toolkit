# Frozen protocol: GPT-5 Nano history-visibility confirmation

## Motivation

The exploratory matched-seed `2 × 3` screen (`n=5` populations/cell) suggested
that an explicit dossier about the current co-player lowers sending in Game
only, while Myth→Game removes that penalty. Because the history-by-order
interaction was selected after inspecting the screen, confirmation uses new
independent protocol seeds and a reduced `2 × 2` design.

## Frozen design

- Model: `openai/gpt-5-nano`, temperature `0.8`.
- Population: 8 agents, balanced rotating anonymous dyads, 10 rounds.
- Noise: informed signed uniform `U(−1,+1)`, applied after decisions to both
  communicated sends and returns.
- History factor:
  - private interaction memory only (`history_policy: none`); or
  - the same memory plus the current co-player's previous three games
    (`history_policy: self_and_coplayer`, self window 0, co-player window 3).
- Task-order factor: Game only versus Myth→Game.
- Memory: capacity 3 interactions for Game only and 6 for Myth→Game, preserving
  approximately three completed rounds in both conditions.
- Independent confirmation replicates: IDs 5–14 (`n=10` populations/cell), with
  pairing/noise seeds 202608205–202608214 matched across all four cells.
- Runtime boundaries: malformed game decisions and decision-as-myth responses
  are rejected before memory, retained as explicit audit events, and retried
  under the frozen bounded recovery policy.

## Frozen analysis

The independent eight-agent population is the unit of analysis.

Primary outcome: average final balance per agent. In this trust game it is a
welfare measure algebraically determined by sending.

Primary test: for each matched replicate, compute

`(dossier Myth→Game − private-memory Myth→Game)
− (dossier Game-only − private-memory Game-only)`

and test the mean paired difference against zero with a two-sided one-sample
t test and 95% t interval. The predicted sign is positive.

Secondary outcomes are mean proportion sent and mean return ratio. Proportion
sent is expected to mirror final balance exactly; return ratio is descriptive
and does not determine joint welfare. The two simple dossier effects and the
two within-history Myth→Game effects will be reported with confidence intervals
but will not replace the primary interaction test.

The `n=5` discovery seeds will not enter the primary confirmatory test. A pooled
`n=15` estimate may be shown separately and labelled exploratory/descriptive.

## Completion gate

Inference begins only after all 40 final JSONs pass the expanded protocol audit
jointly, including complete rounds/calls, valid accepted task outputs, matched
realized schedules, memory/history placement, and all transfer-noise bounds.
Any stopped replicate is rerun from a clean initial state under the same
replicate ID; partial-round checkpoints are not treated as completed data.
