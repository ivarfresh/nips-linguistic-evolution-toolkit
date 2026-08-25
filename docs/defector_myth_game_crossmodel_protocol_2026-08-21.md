# Frozen protocol: cheap-model Myth→Game defector stress test

Frozen before generating any outcomes for replicate IDs 30–34 on 2026-08-21.

## Questions

1. Do two mechanical zero-giving agents reduce cooperative sending among the
   six ordinary agents in an eight-agent rotating population?
2. Does the treatment change the myths ordinary agents write, and do the
   mechanically constrained agents produce culturally distinctive myths?
3. Does a defector stressor create behavioral headroom in Gemini 3.1
   Flash-Lite, which otherwise showed complete maximum sending?

## Design

A matched `2 models × 2 defector levels` Myth→Game screen:

- models: GPT-5 Nano and Gemini 3.1 Flash-Lite;
- control: no defectors;
- treatment: two of eight agents (`25%`) are assigned reproducibly from the
  replicate ID;
- five independent populations per model/level, for 20 populations total.

In treatment populations, defectors make no game-decision LLM calls and always
emit `{"send": 0}` or `{"return": 0}`. Their scripted prompt and response are
still appended to private memory. Their status is not disclosed in prompts,
and other agents are never told who they are. Defectors use the ordinary LLM
myth-writing path, so their cultural output can react to their experience even
though their game actions are mechanically fixed.

All populations use ten rounds, balanced rotating dyads, hidden names, no
synthetic history recap, approximately three rounds of private game-and-myth
chat memory, informed signed `U(−1,+1)` transfer noise after both decisions,
temperature .8, and the existing unified prompts. Replicate IDs 30–34 yield
matched pairing/noise seeds `202608151`–`202608155`; the same schedules and
defector assignments are used across models.

The no-defector cells are necessary causal controls, not an additional
substantive non-defector study. Within each model, control and treatment differ
only in `defector_ratio`.

## Primary behavioral outcome

For each population, calculate the mean proportion of the `$5` endowment sent
by **standard agents only**. The primary model-specific contrast is:

`25% defectors − 0% defectors`

using five paired population differences, a two-sided paired t interval and
paired Cohen's `dz`. Negative estimates indicate behavioral spillover beyond
the mechanically imposed zero actions. Because this is an `n=5` screen, exact
estimates and intervals take priority over binary significance.

The cross-model directional criterion is descriptive: both model-specific
estimates below zero support a general defector-spillover hypothesis; opposite
directions imply model dependence; intervals spanning zero remain unresolved.

## Secondary behavioral diagnostics

- standard-agent sending to standard versus defector receivers;
- standard-agent return ratios when facing standard versus defector senders;
- standard-agent final balances and whole-population final balance;
- round-one and per-round standard-agent sending trajectories; and
- whether Gemini leaves its previous maximum-send ceiling under treatment.

These tests are unadjusted diagnostics.

## Frozen myth-text measures

Myths are analyzed separately for standard and defector authors. To avoid a
post-outcome or self-judge rubric, the initial analysis uses fixed transparent
text measures aggregated to the population level:

- cooperation/fairness lexicon density and myth-level presence: stems covering
  `cooperat`, `trust`, `reciproc`, `share`, `generos`, `fair`, `return`,
  `mutual`, `together`, `gift`, and `help`;
- defection/threat/punishment density and presence: stems covering `defect`,
  `betray`, `exploit`, `withhold`, `punish`, `retaliat`, `revenge`, `threat`,
  `distrust`, `selfish`, and `zero`;
- an explicit half/equal-split rule using fixed expressions linking
  `return/give/share/split` to `half`, `equal`, or `fifty percent`; and
- word count.

Primary myth contrast: treatment-standard minus control-standard population
mean cooperation/fairness density. Threat/punishment density and explicit-
half prevalence are secondary. Treatment defectors versus treatment standard
authors is a descriptive within-population comparison with no control analogue.

No LLM judge scores are part of this frozen pilot. Selected examples may be
quoted only after the quantitative text summary and must be labeled
illustrative.

## Acceptance gate

All 20 populations must pass jointly before outcomes are analyzed:

- complete rounds, dyads, decisions, and myths;
- strict game/myth response boundaries and memory counts;
- exact signed-noise bounds;
- matched schedules within each replicate and across models;
- exactly 20 forced game responses per treatment population, with no LLM game
  call for defectors and normal LLM myth calls for them;
- no forced response in controls; and
- no hidden defector label or agent identity leakage in prompts.
