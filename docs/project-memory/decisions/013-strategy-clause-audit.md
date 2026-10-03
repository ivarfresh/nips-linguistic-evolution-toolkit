# D013 — Bounded strategy-clause audit

- Recorded / last verified: 2026-10-02 / 2026-10-03
- Decision status: agreed
- Scope: 48 complete trajectories from original September and main-frontier corpora; excludes extensions, frontier-update models and pilots.
- Decision authority: Ivar's explicit instruction in the current Codex task, 2026-10-02: "oke great, do the analysis now please", following the staged existing-data proposal preserved in the audit plan.
- Implementation status: sampled completed finals; exploratory machine coding completed, human validation pending.

## Decision and rationale

**Explicit:** inspect existing complete trajectories for evolving conditional or
paraphrased rules, rather than launching a changed experimental design. The
[frozen plan](../../research/strategy_clause_audit_20261002/PLAN.md) limits the
work to a feasibility audit and makes human validation a gate before scaling.
This is a narrow user-authorized exception to the earlier meeting's no-further-
exploration scope; it does not authorize new simulations or alter paper claims.

## Evidence

- User instruction: current task, 2026-10-02; exact wording preserved above and in the plan. The plan/report are assistant-authored records, not an external team decision.
- Completed final-state manifest: [paths and SHA256 hashes](../../research/strategy_clause_audit_20261002/manifest.json), 48 trajectories from 45 unique runs.
- Implementation: [stdlib audit script](../../../analyses/strategy_clause_audit.py); no experimental runtime or source-data changes.
- Result: [report](../../research/strategy_clause_audit_20261002/README.md), [validation](../../research/strategy_clause_audit_20261002/validation.json), and [bounded independent review](../../research/strategy_clause_audit_20261002/REVIEW.md).

All 480 sampled myths and 432 actual exposures were checked against final-state
data and accepted saved prompts. The 44 selected events are not a census.
T34 and T40 provide candidate peer-to-peer clause uptake; T40 retains a
compensation clause for two further myths, then omits it. The audit does not
establish behavioral enactment, multi-hop transmission, or a mechanism for the
cooperation effect. No pooled behavioral test was justified by the selected
eligibility checks. Machine labels and counts await human validation.

## Chronology and supersession

2026-10-02: user authorized this bounded analysis; outcome-independent sample
fixed before reading; disjoint machine readings and selected independent
checks completed. This narrows a blanket no-rule-transmission interpretation
but does not supersede earlier registered nulls or causal evidence boundaries.

## Unresolved / next evidence

2026-10-03 inspector update (explicit user request in the same task): evidence
entries can repeat within a field, carry both advice and narration labels, and
link multiple quotations with source rounds. Rationale/expected consequence is
available separately from actions. This changes the human annotation interface,
not the frozen machine coding or corpus. [Inspector source and migration](../../../tools/trajectory-inspector/README.md)
use export schema v2; v1 draft text, progress, and shared quotes are preserved,
without inventing field-specific quote links. Machine/human comparisons must
account for the different entry schemas.

Validation provenance: H01 round 1 was discussed and given an example coding
by the assistant in this task before the user finished reviewing. That example
is guided practice, not independent human validation; do not report a subsequent
whole-H01 assessment as wholly blinded to assistant interpretation. No human
annotations have been ingested or judged complete by this update.

2026-10-03 successor (explicit user request, same task): "Can you just rate it
please :) i think tis fine for now. Lets just complete all the steps and then
ill check later if i agree". All twelve packet trajectories / 120 rounds now
have [AI draft assessments](../../research/strategy_clause_audit_20261002/ai_reading_packet.json),
with separate observation types, exact quotations, uncertainty notes, and
trajectory summaries. The inspector presents these separately from existing
browser-local human notes. All AI read flags are empty and completion flags
false; they do not stand in for a human review. The quote validator checks 923
linked quote instances across 841 entries (including absence notes), not semantic
validity. H07 round 3 is explicitly uncodeable because the packet contains an
empty response object. This new reading uses a broader notion of refinement
(including communication procedures and fractional specificity); it does not
silently replace the frozen earlier machine labels. Prior audit summaries were
available to the AI reader. No model batch, simulation, or behavioral analysis
was launched. Subsequent user checking of these visible drafts is AI-assisted
review, not independent blinded validation for any of the twelve trajectories.

2026-10-03 user feedback received: Ivar supplied 36 round-comment blocks across
H01–H04 and authorized organizing them. Original wording is preserved byte-for-byte
in [the submitted notes](../../research/strategy_clause_audit_20261002/user_notes_original_20261003.txt);
[organized feedback](../../research/strategy_clause_audit_20261002/user_notes_organized_20261003.json)
contains 69 assistant-organized observations and 93 exact quote links, including
explicit corrections and qualifications. These are not new independent human
labels or an agreement statistic. The [synthesis](../../research/strategy_clause_audit_20261002/user_review_findings_20261003.txt)
distinguishes action rules from sacred/identity/relational justification and
fictional transmission. A targeted T36 check verifies partner names in accepted
saved prompts before their appearance in own R9/R10 output (Oros/Elara and Kael).
This is exposure-compatible narrative borrowing, not cross-run H01-to-H03
communication, causal rule uptake, or behavioral transmission. Original audit
labels and analyses remain unchanged. Broader narrative coding is proposed,
not authorized scaling; independent reliability and behavioral effects remain open.

2026-10-03 bounded follow-up (explicit user instruction: "yes just go full
autonomously just do it now please"): steps 4–5 completed for the existing
sample, with no paid calls or simulations. The [follow-up plan](../../research/strategy_clause_audit_20261002/FOLLOWUP_PLAN_20261003.md)
was fixed before new summaries, not preregistered or fully outcome-blind.
[Results](../../research/strategy_clause_audit_20261002/followup_findings_20261003.txt)
and [reproduction script](../../../analyses/strategy_clause_followup.py) verify
480 own myths/game contexts and 432 exposures/unseen texts. Novel five-word
overlap favors actual exposure by 1.08 percentage points (run SD 2.61;
descriptive run-bootstrap interval 0.43–1.89, 45 runs), not a semantic or causal
effect. All 44 earlier selected events have source/timing ledgers; original
semantic labels remain inherited, unvalidated machine readings. Six forgiveness
and ten reduction candidates fail strict measurable-trigger gates; of six
unconditional amount candidates, only two have an immediate sender opportunity,
and both already sent the named amount. No pooled behavioral inference is
justified. Non-identifiability is not evidence of no enactment. This completes
the bounded follow-up, not exhaustive coding or independent human reliability.

Human validation of extraction and highlighted examples; separate blinded
source-attribution checks if pursued; behavioral identification with clearly
observed antecedents. The [12-trajectory packet](../../research/strategy_clause_audit_20261002/HUMAN_VALIDATION.md)
is prepared; independently completed human coding is not yet established. No scaling or new experiment is authorized by
this record.
