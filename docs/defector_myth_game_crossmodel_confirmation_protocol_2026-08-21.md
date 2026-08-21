# Frozen protocol: independent cross-model defector confirmation

Frozen before generating any outcomes for replicate IDs 35–44 on 2026-08-21.
The preceding replicate IDs 30–34 are an exploratory pilot and are excluded
from all confirmatory tests below.

## Confirmatory questions

Across GPT-5 Nano and Gemini 3.1 Flash-Lite, do two hidden mechanical
defectors:

1. reduce the proportion sent by ordinary agents; and
2. reduce cooperation/fairness language in ordinary agents' myths?

## Design

The experiment independently repeats the frozen pilot design with ten new
matched populations per model/defector level (`40` populations total):

- models: GPT-5 Nano and Gemini 3.1 Flash-Lite;
- no-defector control versus two of eight (`25%`) mechanical defectors;
- Myth→Game for ten rounds;
- defectors always send/return zero without game LLM calls, retain the scripted
  exchange in memory, write myths normally, and are not labeled in prompts;
- balanced anonymous rotating dyads, no synthetic history recap, approximately
  three rounds of private game-and-myth memory, informed signed `U(−1,+1)`
  transfer noise, temperature .8, and unified prompts; and
- replicate IDs 35–44, yielding matched pairing/noise seeds
  `202608156`–`202608165` and the same defector assignments across models.

Within each model and replicate, control and treatment differ only in
`defector_ratio`.

## Two confirmatory outcomes

For every model/replicate, form treatment-minus-control differences:

1. **Behavioral spillover:** mean proportion of `$5` sent by standard agents
   only. Controls average all eight standard agents; treatments average the six
   non-scripted agents, matching the frozen pilot definition.
2. **Cultural spillover:** population-mean cooperation/fairness lexicon density
   among standard-authored myths, using the exact frozen pilot stems per 100
   words: `cooperat`, `trust`, `reciproc`, `share`, `generos`, `fair`,
   `return`, `mutual`, `together`, `gift`, and `help`.

For each outcome, combine the two model strata with equal model weight. If
`d_m` is the mean paired difference within model `m`, the cross-model estimate
is `(d_GPT + d_Gemini)/2`. Its standard error is
`sqrt(s_GPT^2/n_GPT + s_Gemini^2/n_Gemini)/2`, with a Welch-Satterthwaite
degree of freedom. This estimates the average effect across the two selected
models without treating their baselines as interchangeable.

Both hypotheses are directional negative, but tests are two-sided. Apply Holm
correction across the two confirmatory p values. A claim is confirmed only if
its effect is negative and its Holm-adjusted p value is below .05. Report raw
95% model-stratified intervals, adjusted p values, and model-specific paired
estimates regardless of outcome.

## Predeclared secondary diagnostics

- model-specific treatment effects for both outcomes;
- round-one and per-round standard-agent sending;
- standard-agent return ratios and balances, clearly separating mechanical
  balance loss from behavioral spillover;
- fixed defection/threat density and explicit half/equal-split prevalence in
  standard-authored myths;
- Gemini defector-minus-standard cooperation and threat density in rounds
  2–10, testing the pilot's model-specific cultural-imprinting signal; and
- the corresponding GPT author-type contrasts.

Secondary p values are unadjusted diagnostics and do not alter the two
confirmatory verdicts. The fixed lexicons and half-rule expressions are exactly
those implemented in `scripts/analyze_defector_myth_game_crossmodel_n5.py`.

## Acceptance gate

All 40 populations must pass jointly before outcome analysis:

- complete rounds, dyads, decisions, and myths;
- strict response boundaries and memory counts;
- exact signed-noise bounds;
- matched schedules within replicate and across models;
- exactly 20 forced game responses per treatment population, no LLM game call
  for defectors, and normal LLM myth calls for them;
- no forced response in controls; and
- no defector label or agent identity leakage in prompts.
