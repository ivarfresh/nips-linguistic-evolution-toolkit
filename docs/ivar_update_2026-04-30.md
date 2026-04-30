# LLM Linguistic Evolution: Noise Runs Update

Date: 2026-04-30

This is a short status update on the trust-game/noise simulation work after the
latest implementation fixes and direct-provider runs.

## What We Changed

We cleaned up the noisy trust-game implementation so that the model's chosen
amount and the actual environmental transfer are tracked separately.

For each transfer:

- The agent makes a decision, e.g. `sent_decision` or `returned_decision`.
- Directional uniform noise is added to the actual ledger amount.
- The actual amount is clamped to the valid range.
- Balances and later prompts use the actual ledger amount.
- The raw model decision, applied noise, noise draw, actual amount, and balances
  are all logged.

The main directional noise conditions are:

- Positive uniform noise: `decision + U(0, k)`, clamped to the available amount.
- Negative uniform noise: `decision + U(-k, 0)`, clamped to the valid range.
- Noise applies to both sent and returned amounts.
- We ran informed and uninformed variants, where informed agents are told that
  transfer amounts may be environmentally perturbed.

We also added a resumable batch runner, `scripts/run_noisy_missing.py`, which
computes the expected output files, skips completed runs, and only launches
missing jobs. This made the larger batches much easier to resume after failures.

Finally, we added direct provider routing for OpenAI, Anthropic, and Gemini, so
we no longer need OpenRouter for these runs.

## Completed Runs

The completed v4 direct-provider result sets are in:

`data/json/noise_experiments/v4_direct_provider/`

Completed non-Gemini batches:

| Experiment | Model | Final runs |
|---|---:|---:|
| `noise_positive_mem3_gpt5_nano` | GPT-5 nano | 90/90 |
| `noise_negative_mem3_gpt5_nano` | GPT-5 nano | 90/90 |
| `noise_bootstrap_mem3` | GPT-5 nano | 90/90 |
| `noise_positive_mem3_claude_sonnet_45` | Claude Sonnet 4.5 | 90/90 |
| `noise_negative_mem3_claude_sonnet_45` | Claude Sonnet 4.5 | 90/90 |

Gemini was smoke-tested successfully through the direct provider route, but the
full batch hit the daily request quota after 12 completed runs. We are leaving
Gemini paused for now.

There is also a partial deterministic-max GPT-5 nano run from an earlier
diagnostic path. It is not part of the main directional-uniform design and has
not been treated as a primary result.

## Headline Findings So Far

These are preliminary because the matched no-noise v4 baseline is still the
first planned follow-up.

Claude appears highly sensitive to the environmental ledger. Under positive
uniform noise, Claude is near the cooperative ceiling: total earnings are around
148-149 out of a 150 maximum, and pre-noise return decisions are close to full
reciprocation. Under negative uniform noise, Claude collapses toward much lower
actual transfers and much lower return behavior.

GPT-5 nano also responds strongly to positive noise on the investor side, but
its trustee-side reciprocity remains much weaker than Claude's. In the positive
noise runs, GPT-5 nano often sends near the maximum after the ledger is boosted,
but its pre-noise return decisions remain substantially below Claude's.

The informed noise note does not seem to change the main qualitative pattern.
Direction of noise and model identity matter much more than whether the agents
are explicitly told that transfers may be perturbed.

The bootstrap condition is useful diagnostically, but it should not be confused
with natural cooperation. It forces returned amounts to the maximum, so it shows
that GPT-5 nano can be held in a cooperative-looking ledger environment, while
its own return decisions remain lower.

The `k=5` directional noise conditions are strong manipulation checks. Because
they often hit the floor or ceiling, the next sensitivity question is whether
the same qualitative pattern appears with smaller perturbations.

## Important Caveat

The older baseline files are not perfectly matched to the v4 noise runs. Many
older baselines are 15-turn runs, while the current v4 noise batches are
10-turn, memory-3 runs. We should therefore avoid over-interpreting noise
effects until we run a matched v4 no-noise baseline.

## Prepared Next Runs

The next run definitions are already prepared in `config/experiments_noisy.yaml`,
with exact commands documented in `docs/v4_next_runs.md`.

Recommended order:

1. Run matched no-noise v4 baseline.
   - Experiment set: `baseline_v4_mem3_direct`
   - Models: GPT-5 nano and Claude Sonnet 4.5
   - Conditions: `game`, `game_myth`, `myth_game`
   - Runs: 15 per condition
   - Total jobs: 90

2. Run neutral-framing pilot.
   - Experiment set: `neutral_framing_v4_pilot`
   - Replaces "trust game", "investor", "trustee", "send", and "return" framing
     with a more neutral ROLE A / ROLE B allocation-task frame.
   - The parser-facing JSON keys remain `send` and `return`, but the visible
     prompts avoid those labels.
   - Total jobs: 18

3. If the neutral-framing pilot shifts behavior, run the full neutral-framing set.
   - Experiment set: `neutral_framing_v4_mem3`
   - Total jobs: 90

4. Run smaller directional noise sensitivity checks.
   - Experiment set: `noise_directional_k1_mem3`
   - Experiment set: `noise_directional_k2_mem3`
   - Each is 360 jobs across model, task order, direction, and informed/uninformed
     variants.

5. Return to Gemini only if quota constraints become manageable.

## Relevant Commits

- `f707473` Add resumable noisy batch runner
- `1374148` Add direct Gemini provider routing
- `6dd5833` Add Claude negative uniform noise set
- `04bf1bc` Add GPT-5 nano positive uniform noise set
- `ce93cdd` Add v4 direct provider noise results
- `89f0f29` Prepare matched baseline and framing runs
