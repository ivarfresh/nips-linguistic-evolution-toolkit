# Frozen protocol: Gemini 3.7 Flash deduction calibration

Frozen on 2026-08-23 after the baseline ceiling screen and before any Gemini
3.7 deduction response was requested.

## Question

Does Gemini 3.7 Flash use the current costly-deduction mechanism selectively
as a response to low visible returns, as Gemini 3.1 Flash-Lite did, or does the
model adopt a different policy?

## Design

Repeat the exact current-wording controlled calibration previously used for
GPT-5 Nano and Gemini 3.1 Flash-Lite. Hold fixed the system rules, cooperative
myth, full `$5` send, `$15` receipt, two-point deduction budget, 1:3 sanction
effect, response format, and Myth→Game decision instruction. Manipulate only
the visible return: 0%, 10%, 25%, 50%, or 75% of the `$15` receipt.

Run ten independent calls per return level (`N=50`) in one frozen randomized
order. Use `google/gemini-3.7-flash` through the direct Google API at medium
thinking, omit its unsupported legacy temperature parameter, and record a
300-second request timeout. Do not test alternative wording.

## Frozen selectivity gate

Apply the same gate as in the earlier model calibrations:

1. mean spending at low returns (0%/10%) is at least 0.5 points above mean
   spending at high returns (50%/75%);
2. no more than 25% of high-return calls spend any point; and
3. mean spending is non-increasing across return levels, allowing at most one
   adjacent sampling reversal no larger than 0.2 points.

Report cell means and frequencies, linear and Spearman trends, low-minus-high
separation, retries, token usage, and estimated cost. Comparisons with the
historical GPT and Flash-Lite calibrations are descriptive model contrasts.
Gemini 3.7 is eligible for a small punishment population test only if all three
gate components pass. This calibration cannot establish cooperation or welfare
effects.

## Integrity

Require exactly 50 accepted integer decisions, ten per return level, identical
messages within cells, exact controlled states, no unrecovered error, retention
of all retries, and complete provider/model/Git/config/thinking/timeout
provenance. Refuse to launch from a dirty worktree.
