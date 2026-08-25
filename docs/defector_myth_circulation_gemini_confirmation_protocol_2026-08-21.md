# Frozen protocol: independent confirmation of threat-language transmission

Frozen before running replicate IDs 50–59 on 2026-08-21. Replicate IDs 45–49
from the preceding mechanism screen are excluded from every confirmatory test.

## Confirmatory question

When an ordinary Gemini agent has just interacted with a hidden mechanical
defector, does replacing the defector-authored myth with a natural
ordinary-authored myth reduce defection/threat language in the ordinary agent's
immediately subsequent myth?

## Design

Independently repeat the frozen circulation design with ten new matched
populations per arm (`20` populations total), using Gemini 3.1 Flash-Lite.
Both arms contain the same two of eight hidden mechanical defectors. They share
defector assignments, balanced anonymous pairing schedules, and signed-noise
draws within replicate. They differ only at defector-to-ordinary myth exposure
events:

- **normal:** present the prior partner's actual defector-authored myth; or
- **standard substitute:** present a real ordinary-authored myth from the same
  prior population-round, selected deterministically without source labels.

Defectors always send/return zero without game LLM calls and write myths
normally in both arms. All runs use Myth→Game for ten rounds, memory-primary
capacity six, no synthetic game-history recap, the shared myth decision
instruction, informed signed `U(−1,+1)` transfer noise, and replicate IDs
50–59 (paired protocol seeds 202608171–202608180).

## Sole confirmatory outcome

For every ordinary target in rounds 2–10 whose prior-round partner was a
defector, compute fixed defection/threat lexicon density in the target's current
myth. Average within population and form matched-replicate differences:

`standard substitute − normal circulation`.

The fixed lexicon is unchanged from the pilot: stems for defection, betrayal,
exploitation, withholding, punishment, retaliation, revenge, threat, distrust,
selfishness, and zero, divided by total myth words and multiplied by 100.

The directional prediction is negative, but use a two-sided paired t test and
95% paired interval. The effect is confirmed only if the estimate and entire
interval are negative and `p<.05`. With one primary outcome, no multiplicity
adjustment is required.

## Frozen secondary diagnostics

- cooperation/fairness density in the same directly exposed target myths;
- same-round sending and return ratios among directly exposed targets;
- presented-text cooperation and threat density as a manipulation check;
- population-wide ordinary-agent sending and ordinary-authored myth language;
- defector-author language in both arms; and
- exact term-frequency breakdowns for interpretability.

Secondary tests are unadjusted diagnostics and cannot change the confirmatory
verdict.

## Acceptance gate

All 20 populations must pass jointly before outcomes are inspected:

- complete rounds, dyads, decisions, and myths;
- exactly 20 forced-zero game responses per population and normal LLM-authored
  defector myths;
- exact signed-noise bounds, matched schedules, and matched exposure
  opportunities;
- exposure records for every agent in rounds 2–10;
- every and only defector-to-ordinary exposure replaced in the substitute arm;
- recorded presented text present and the original defector text absent from
  each substituted current prompt;
- no author identity, defector label, or circulation-policy leakage; and
- no unrecovered provider or response-boundary errors.

## Scope

This confirms or rejects one-step lexical transmission in Gemini under the
current anonymous Myth→Game setting. It does not test whether culture changes
behavior under a game with more behavioral headroom, nor whether the result
generalizes to a model whose defector-authored myths lack the source signal.
