# Frozen protocol: cross-model population-size figure

## Purpose

Complete the missing corrected two-agent task-order cells needed for a
descriptive comparison of Claude Sonnet 4.5, GPT-5 Nano, and Gemini 3.7 Flash
across the repeated-dyad and rotating eight-agent population regimes.

Existing corrected Claude runs and the existing GPT/Gemini eight-agent screens
remain unchanged. Older GPT and Gemini dyad files are excluded because they do
not use the corrected dyad prompt, transfer-noise finalization, and memory
protocol.

## New runs

- GPT-5 Nano: five paired replicates of Game only, Game-to-Myth, and
  Myth-to-Game (`replicate_id` 0-4).
- Gemini 3.7 Flash: three paired replicates of the same conditions
  (`replicate_id` 90-92), using the direct Google API and medium thinking.
- Two agents, ten rounds, no defectors, and no deduction/punishment stage.

All cells use the corrected dyad implementation, temperature 0.8 where the
provider accepts it, informed signed `U(-1,+1)` communication noise applied
after both transfer decisions, and memory-primary history containing three
complete rounds. Two-task cells retain the shared myth-to-decision instruction
and the corrected minimal later-round dyad game prompts.

The replicate IDs reproduce the signed-noise seeds used by the corresponding
eight-agent model screens. This aligns exogenous noise draws, but the two-agent
and eight-agent arms remain different interaction regimes: one repeated dyad
versus anonymous rotating partners with current-partner history.

## Integrity gates

Before using outcomes:

- all 24 new populations must complete ten rounds;
- all 480 game decisions and 320 myths must be present;
- all 480 communicated transfers must equal the post-decision signed-noise
  transform, within rounding tolerance;
- all task-order conditions must contain the frozen replicate IDs;
- model, provider, thinking setting, prompt keys, memory horizon, absence of
  defectors/punishment, code commit, config hash, and clean-worktree provenance
  must match this protocol; and
- there must be no unrecovered response-boundary or provider failures.

## Planned figure

The analysis unit is the independent run. Plot average final cumulative
balance per agent with run-level observations and 95% t intervals, using one
panel per model, task order on the x-axis, and population regime as the visual
group. Use one shared y-axis and report the unequal replicate counts directly.

This is a descriptive cross-model figure. It is not a powered test of model by
population-size interactions, especially for Gemini (`n=3` per cell).
