# Overnight Missing-Condition Sweep Summary

Generated Monday 2026-05-04, after the Sunday-night GPT-5.5 / Gemini sweep.

## Bottom Line

Gemini-3.1-pro-preview completed cleanly and is useful as boundary/robustness evidence, but it does not reproduce the GPT-5-Nano bootstrap destabilisation result. In the Gemini bootstrap cells, myth exposure is higher than game-only, not lower: standard prompt means are 62.6 for game only, 67.6 for game-then-myth, and 71.8 for myth-then-game.

GPT-5.5 hit OpenAI `insufficient_quota`. The available GPT-5.5 data are mostly the positive-noise cells, which are saturated at ceiling. They should not affect the abstract spine.

## Completion Status

| model | set | final JSONs | status |
|---|---:|---:|---|
| Gemini-3.1-pro-preview | positive | 90/90 | complete |
| Gemini-3.1-pro-preview | negative | 90/90 | complete |
| Gemini-3.1-pro-preview | bootstrap | 90/90 | complete |
| GPT-5.5 | positive | 77/90 | quota-blocked; 13 failures |
| GPT-5.5 | negative | 1/90 | quota-blocked; 89 failures |
| GPT-5.5 | bootstrap | 1/90 | quota-blocked; 89 failures |

The GPT-5.5 counts include the launch-prep smoke files. In practical terms, the sweep produced 76 new GPT-5.5 final runs before quota, all in the positive-noise set.

## Main Results

### Gemini-3.1-pro-preview

Positive noise is saturated. All cells are around 74-75 cumulative reward, so this condition gives little room for myth effects.

Negative noise stays low. The game-only mean is 28.8 in the standard prompt and 26.7 when agents are informed that transfer amounts may be perturbed. Myth exposure does not create a clear recovery.

Bootstrap is the most informative Gemini condition. Unlike GPT-5-Nano, Gemini does not show a drop when myths are introduced. In the standard prompt cells, means are:

| task order | mean cumulative reward |
|---|---:|
| game only | 62.6 +/- 10.5 |
| game then myth | 67.6 +/- 7.5 |
| myth then game | 71.8 +/- 3.8 |

When agents are informed about the noise, Gemini shows the same broad ordering but slightly lower levels: 60.1, 64.8, and 69.6. This looks more like model/noise heterogeneity than a direct replication of the GPT-5-Nano mechanism.

### GPT-5.5

The usable GPT-5.5 data are almost entirely positive noise. Those cells are at ceiling:

| task order | mean cumulative reward |
|---|---:|
| game only | 75.0 |
| game then myth | 74.9 |
| myth then game | 75.0 |

Negative and bootstrap GPT-5.5 cells are not interpretable because each has only one final JSON after the smoke test. The remaining jobs failed with OpenAI `insufficient_quota`.

## Interpretation For The Paper

The overnight runs support a bounded version of the story: the sharp bootstrap destabilisation/stabilisation pattern remains a GPT-5-Nano result, while Gemini provides evidence that the effect is not universal across frontier models. That is useful, but it argues against broad wording such as "myths destabilise" or "myths rescue cooperation" without specifying the model/noise regime.

For the abstract, the safest wording is still mechanism-led and conditional: under GPT-5-Nano bootstrap noise, implicit myth-writing destabilises cooperation, while adding coherent text under the partner-story field stabilises play back toward the game-only trajectory. The Gemini result should be described as a boundary condition unless the team wants to make heterogeneity the story.

## Figures

- `fig1_completion_status.png`: run completion and quota block.
- `fig2_gemini_trajectories.png`: Gemini trajectory grid across positive, negative, and bootstrap noise.
- `fig3_bootstrap_model_comparison.png`: bootstrap endpoint comparison showing GPT-5-Nano destabilisation versus Gemini non-replication.
- `fig4_positive_ceiling_comparison.png`: positive-noise ceiling comparison including partial GPT-5.5.
- `fig5_gpt55_positive_trajectories.png`: the 77 GPT-5.5 positive-noise final runs split by standard vs noise-informed prompt.
