# Strategy clause audit — frozen exploratory plan, 2026-10-02

Authorized by Ivar in this task after the discussion of conditional/paraphrased
rules: "oke great, do the analysis now please". This is an existing-data,
measurement-feasibility audit, not a new simulation or a confirmatory test.

## Sample fixed before reading trajectories

Use the original September corpus (not the later n=10 extension) and the main
frontier corpus: Sonnet/GPT/Gemini and Opus/Sol/GeminiPro, respectively.
One complete ten-round agent trajectory per tier × family × size (2/8) ×
mixed (false/true) × order (game_myth/myth_game): 48 trajectories.
Select the smallest SHA256 of `strategy-clause-v1|tier|run_id|agent` within
each cell, among trajectories with ten nonempty myths. Selection uses identifiers
and completeness only, never payoffs, labels or thematic content. Record all
candidate counts, omissions, corpus hashes and final-state hashes. Multiple
sampled agents can share a run; they are not independent replications.

## What is coded

Read each full trajectory and describe the round-1 and round-10 strategies.
Record at most two clearest structural changes per trajectory (not necessarily
more complexity): addition, removal, conditionalization, threshold change,
forgiveness/recovery, attribution to noise, or persistence with no clear change.
Every event must give exact quotes and round numbers. Separate endorsed rules
from narrated actions, generic virtues, and ambiguous metaphor. Do not infer a
numeric threshold or a multi-step policy where the text does not state one.
Quote both before and after where claiming transformation.

For each candidate event examine: the immediately previous own myth, all earlier
own myths (to avoid calling reappearance novel), the actually exposed myth,
and one deterministically selected unseen myth from a different run matched on
tier, author family, round, size, mixing and task order (same composition if
available; explicitly flag fallback). An unseen comparison is not a randomized
control. Gameplay and observed reputation remain competing sources.

## Behaviour and interpretation

Predefine only two potential behavioral checks: reduced sending after repeated
poor reciprocation; continued sending after a single explicitly noise-attributed
shortfall. First check whether clauses and logged *visible* histories permit
unambiguous eligible events. Do not substitute true transfers for communicated
observations. Do not fit exploratory pooled significance tests to the selected
48 trajectories or treat stories as independent units. If identification or
sample size is inadequate, report eligibility and illustrative consistency only.
Do not interpret a complex story as evidence that its author follows it.

## Validation and stop rule

Three GPT-6-Sol readers code disjoint samples. A second reading checks selected
claims, and a script verifies quote substrings and source round identities.
These are machine readings, not human validation or inter-rater reliability.
Create a blinded human coding packet and key. Human validation is a prerequisite
for scaling a new judge across the whole corpus. Stop after this feasibility
audit; no paid judge batch, new simulation, or runtime change is authorized by
this plan. Report inconclusive/absent changes alongside interesting ones.

The paper motivating the question is Vallinder & Hughes (2024),
https://arxiv.org/abs/2412.10270; its selected generational inheritance design
differs from this study's ongoing local myth exchange.
