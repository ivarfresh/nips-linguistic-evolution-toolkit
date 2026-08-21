# Frozen protocol: do defector-authored myths transmit their cultural signal?

Frozen before running replicate IDs 45–49 on 2026-08-21.

## Question

The independent defector confirmation showed that Gemini 3.1 Flash-Lite
defectors write less cooperation-oriented and more threat-oriented myths after
being mechanically forced to send or return zero. It did not show that this
signal changes ordinary agents.

This mechanism screen asks: **does exposing ordinary agents to those
defector-authored myths change the ordinary agents' next myth and game
decision?**

## Design

Run ten new Gemini populations: five matched replicates in each of two arms.
Both arms contain the same two of eight hidden mechanical defectors, with the
same defector assignments, balanced pairing schedules, and signed-noise draws.
They differ only in defector-myth circulation:

1. **Normal circulation:** an ordinary agent paired with a defector in round
   `r` receives that defector's myth in its round `r+1` myth prompt, as in the
   preceding experiments.
2. **Standard substitute:** at exactly those exposure opportunities, the
   ordinary agent instead receives a real ordinary-authored myth from the same
   prior population-round. The substitute author is selected
   deterministically, is neither the target nor the original defector author,
   and is not identified in the prompt.

Defectors write their own myths normally in both arms. Their game responses
remain mechanically fixed at zero without LLM calls. The intervention does not
change the number of myths, myth calls, game calls, memory slots, or prompt
structure. It changes only the source text presented at defector-to-standard
cultural transmission events.

All runs use Myth→Game for ten rounds, Gemini 3.1 Flash-Lite, eight anonymous
rotating agents, memory-primary capacity six, no synthetic game-history recap,
the shared myth decision instruction, and informed signed `U(−1,+1)` transfer
noise. Replicate IDs 45–49 yield paired protocol seeds 202608166–202608170.

## Frozen outcomes

For every standard target in rounds 2–10 whose prior-round partner was a
defector, compare the standard-substitute arm with normal circulation:

1. **Immediate cultural transmission:** cooperation/fairness lexicon density
   in the target's current myth, using the frozen stems from the preceding
   pilot and confirmation. Prediction: positive.
2. **Immediate behavioral transmission:** proportion sent in the same round by
   exposed targets who are assigned sender. Prediction: positive.

Aggregate each outcome within replicate and use matched-replicate differences.
Report two-sided 95% paired intervals and p values; apply Holm correction across
the two tests. This is an `n=5` mechanism screen, so effect sizes and intervals
remain more important than a binary significance label.

## Secondary diagnostics

- threat/defection density and explicit half/equal-split prevalence in the
  directly exposed standard-authored myths;
- return ratios for directly exposed targets assigned receiver;
- population-wide ordinary-agent sending and myth-language trajectories;
- the number and timing of substituted exposure events; and
- whether the distinctive Gemini defector-author language pattern remains
  present in both arms.

## Acceptance gate

All ten populations must pass jointly before outcome analysis:

- complete rounds, dyads, decisions, and myths;
- exact forced-zero counts and normal defector myth LLM calls;
- exact signed-noise bounds and matched schedules;
- an exposure record for every agent in rounds 2–10;
- in the substitute arm, every and only defector-to-standard exposure is
  replaced by a same-round ordinary-authored myth;
- the recorded presented myth is present in the target's prompt and the
  defector-authored source myth is absent;
- no author identity, defector label, or circulation-policy leakage in prompts;
  and
- no unrecovered provider or response-boundary failures.

## Scope of inference

This design identifies the effect of a defector-authored myth relative to a
natural ordinary-authored replacement, not the effect of myth exposure versus
no myth. A positive substitution effect would show that the distinctive
defector content transmits to an immediately exposed ordinary agent. A null
effect would indicate that the cultural imprint is largely localized to the
mechanically constrained authors under the present anonymous-rotation design.
