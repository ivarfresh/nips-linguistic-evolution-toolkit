# Frozen protocol: Gemini cross-model deduction calibration

Frozen on 2026-08-21 after the GPT-5 Nano calibration and before any Gemini
calls.

## Question

Is the failure of selective deduction under the current neutral wording
specific to GPT-5 Nano, or does it also occur in Gemini 3.1 Flash-Lite?

## Design

Repeat the exact **current-wording** controlled calibration with Gemini 3.1
Flash-Lite through the direct Google endpoint. Hold fixed the system rules,
fixed cooperative myth, full $5 sender decision, deduction budget, 1:3 effect,
response format, Myth→Game instruction, and five visible returns (0%, 10%,
25%, 50%, and 75% of $15 received). Run 10 independent calls per return level
(`N=50`) in a fixed randomized order.

Do not include the rejected cost-salient variant and do not change any economic
or normative wording. This is a cross-model test of the same mechanism, not a
new prompt search.

## Frozen diagnostics and decision

Apply the same selectivity gate as the GPT calibration:

1. low-return (0%/10%) mean spending at least 0.5 points above high-return
   (50%/75%) spending;
2. no more than 25% of high-return calls spend any point; and
3. mean spending non-increasing across return levels, allowing at most one
   adjacent sampling reversal of 0.2 points or less.

Report cell means/frequencies, linear and Spearman trends, low-minus-high
separation, and the Gemini-minus-GPT return-level slope interaction. Gemini is
eligible for a small population pilot only if it passes all three gate
components. If it fails, set aside this separate-point-budget mechanism across
both cheap models and redesign the institution before further live runs.

## Integrity

Require 50 accepted integer decisions, exactly 10 per return level, identical
messages within cells, exact controlled states, no unrecovered errors,
retention of all retries, and complete provider/model/Git/config provenance.
Do not infer cooperation or welfare effects from this calibration.
