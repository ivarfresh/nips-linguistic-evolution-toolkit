# Frozen protocol: deduction-stage mechanism comprehension

Frozen on 2026-08-21 before any live calibration calls.

## Question

Does GPT-5 Nano use the deduction action as a selective response to a low
observed return, or does the mere presence of the action elicit routine
spending? Can a cost-salient but norm-neutral clarification improve that
mapping?

This is a prompt/mechanism calibration, not a cooperation experiment. It is
intended to decide whether another population simulation is scientifically
worthwhile.

## Controlled design

Use the same system rules, Myth→Game decision link, sender decision, response
format, deduction budget, and 1:3 effect as the accepted population runs. Each
independent call receives the same fixed cooperative myth and the same prior
sender action (`{"send": 5}`). The sender then sees one of five controlled
returns from the $15 received by its co-player:

- 0%;
- 10%;
- 25%;
- 50%; or
- 75%.

Cross return level with two wording conditions:

- **Current:** the exact accepted deduction-stage wording.
- **Cost-salient:** adds only that spending is optional, zero is valid, there
  is no requirement to use the budget, and each spent point costs the sender
  $1 relative to keeping it. These are already true under the rules; the
  variant adds no moral recommendation about when to punish.

Run 10 independent calls in each of the 10 cells (`N=100`) through the direct
OpenAI endpoint at the existing GPT-5 Nano settings. Randomize call order with
a fixed seed. Validate the exact `{"deduct": 0|1|2}` boundary; retry a malformed
decision up to three times while retaining every attempt.

## Frozen diagnostics and gate

For each wording, report by return level:

- mean deduction points;
- probability of any deduction; and
- probability of spending both points.

Report the slope and Spearman association between return ratio and deduction,
plus the difference between low-return states (0% and 10%) and high-return
states (50% and 75%). Also report the wording × return-level interaction.

A wording passes the operational selectivity gate only if:

1. mean spending in the low-return states is at least 0.5 points higher than
   in the high-return states;
2. no more than 25% of high-return calls spend any point; and
3. mean spending is non-increasing across the ordered return levels, allowing
   at most one adjacent sampling reversal of 0.2 points or less.

The cost-salient wording is eligible for a new population pilot only if it
passes the gate and improves low-minus-high separation over the current prompt
by at least 0.5 points. Otherwise, do not tune further toward a desired result;
redesign the economic mechanism or set aside explicit punishment for this
model.

## Integrity

Require:

- exactly 100 accepted decisions with 10 per cell;
- all accepted decisions integer and in `[0,2]`;
- identical messages within a return×wording cell;
- the two wording conditions differing only in the frozen clarification;
- exact controlled returns/payoffs and the same fixed myth/send history;
- complete raw response, usage, provider/model, call-order, Git commit, clean
  worktree, and config-hash metadata; and
- no unrecovered provider or response-boundary failures.

Do not infer an effect on cooperation or welfare from this calibration.
