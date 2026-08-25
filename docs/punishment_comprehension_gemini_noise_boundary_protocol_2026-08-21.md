# Frozen protocol: Gemini near-zero noise boundary

Frozen on 2026-08-21 after the broad Gemini calibration and before any
near-zero calls.

## Question

Where does Gemini 3.1 Flash-Lite switch from spending two deduction points to
spending zero within the visible return range that signed noise can generate
from a mechanical defector's true $0 return?

## Design

Use the unchanged current deduction wording, direct Gemini endpoint, fixed
cooperative myth, full $5 send, $15 received, 1:3 deduction rule, and exact
message history from the preceding calibration. Vary only the communicated
return:

- `$0.00` (0% of received);
- `$0.25` (1.67%);
- `$0.50` (3.33%);
- `$0.75` (5%); and
- `$1.00` (6.67%).

Run 10 independent calls per state (`N=50`) in a fixed randomized order. These
amounts span the full `[0,$1]` visible band produced when `U(−1,+1)` noise is
added to a true zero and then clamped at zero.

## Frozen outputs and decision

Report mean points and any-deduction probability at each amount. Define the
empirical switch interval as the adjacent tested amounts bracketing the first
drop from majority deduction (`>50%`) to minority deduction (`<50%`). If an
exact 50% cell occurs, report it as unresolved at that point.

Estimate the implied probability that a true-zero defector is punished under
the live noise process by integrating a piecewise-linear interpolation of the
five observed deduction probabilities over:

- point mass .5 at visible $0 (all negative noise draws clamp to zero); and
- uniform density .5 over visible `(0,$1]` (positive noise draws).

This is a calibrated descriptive estimate, not a behavioral outcome. Proceed
to a three-pair Gemini population pilot if:

1. the implied true-zero punishment probability is at least 50%;
2. the broad calibration's zero high-return punishment result remains intact;
   and
3. all integrity checks pass.

## Integrity

Require exactly 50 accepted integer decisions, 10 per amount, identical
messages within cells, exact controlled payoffs, no unrecovered errors,
retention of retries, and full provider/model/Git/config provenance.
