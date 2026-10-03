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
is prepared; independently completed human coding is not yet established. The
bounded authorization above did not permit scaling or new experiments; the
dated successor below expands only the existing-data analysis scope.

## 2026-10-03 successor — expanded narrative and operator analysis

**Explicit:** Ivar requested all September mid-tier and frontier runs, including
the frontier defector pilot, and approved the team with an additional dedicated
rule-evolution agent (simple-to-conditional and other operator changes). This
supersedes the 48-trajectory scope restriction, not the lack of independent
human reliability. Outputs must remain exploratory machine-coded findings.

Four specialist audits completed: corpus, narrative schema, semantic rule
evolution, and transmission/behavior safeguards. The [expanded plan](../../research/narrative_evolution_20261003/PLAN.md)
preserves distinctions among narration, endorsement, hypothetical and genuine
counterfactual reasoning; clause operators are not grammar counts. No new
experiment, replay or paper-claim change is authorized.

The [verified corpus manifest](../../research/narrative_evolution_20261003/corpus_manifest.json)
contains 1,109 unique hash-matched ten-round finals, of which 736 contain 38,720
myth entries in 3,872 trajectories. Scope includes canonical September-profile
matrices/extensions, main and update frontier sets, and ordinary frontier
defectors. Separate board/saboteur/transplant/replay/pressure/probe protocols are
excluded explicitly. Defector-authored myths remain separate; forced game
actions cannot support model-enactment claims. All entries are nonempty, but
codability still requires checking; simple empty-object screening is insufficient.

Implementation status: free preparation and manifest validation complete;
calibration and full paid coding NOT started. The [budget estimate](../../research/narrative_evolution_20261003/budget_preflight.json)
is approximately $795.50 under explicit token/output/repair assumptions using
provider-verified GPT-6.1 Sol pricing, high reasoning, eight workers. An $800
cap (or calibration-only $10) has been offered; explicit cost approval is
pending. Do not start paid calls until approved. Re-estimate after calibration
and pass semantic-quality gates before full scaling. No new prevalence or
evolution findings yet exist for this expanded corpus.

## 2026-10-03 successor — $10 calibration only

**Explicit user decision:** after asking why the full plan was expensive, Ivar
accepted the proposed calibration-only alternative ("yeah do that instead").
This authorizes a cumulative $10 cap, not the proposed $800 full batch.
Use compact quotation-backed narrative/operator labels for 24 complete
trajectories, two same-model readings each. Sixteen broadly cover the corpus;
eight are named purposive challenges from the earlier audit. This is not a
prevalence sample or independent human validation. No full-corpus or source-
attribution pass, new simulation, or paper-claim change is authorized here.

Implementation: [runner](../../../analyses/narrative_calibration.py),
[fixed sample](../../research/narrative_evolution_20261003/calibration/sample.json)
and [resolved configuration](../../research/narrative_evolution_20261003/calibration/config.json).
The preflight estimate is $5.10 for 48 calls, high reasoning, eight workers.
Budget reservations are persisted before dispatch, unresolved calls count
conservatively against the cap, and non-stop completions are invalid. Raw
responses and quote-validation failures remain auditable. Results pending.

**Completed calibration, later 2026-10-03:** the original version stopped after
16 returned readings (7 technically valid, 9 truncated); 32 planned readings
have no usable saved output. A separate repaired rubric tested C18/C21/C22/C24
twice with high reasoning and 24,000 completion tokens. All eight completed;
seven passed exact-quote/schema checks. One C21_B quotation is invalid.
The [report](../../research/narrative_evolution_20261003/calibration/REPORT.txt)
and [source-check key](../../research/narrative_evolution_20261003/calibration/semantic_qa.txt)
record the boundary: clear piecewise expansion, conditional reduction/recovery,
and baseline punishment examples survive; automatic counts of meaningful
evolution fail because both readings sometimes call clarification/restatement
novel. Rich hypothetical dialogue can defend an unchanged simple rule.
Punishment is ambiguous when the cause of a "dry" hand is unspecified.

Confirmed response-reported cost is $3.0420116; 16 interrupted requests without
receipts retain $2.502 in conservative reservations, total $5.5440116. BYOK
upstream inference costs are included, not mistaken for the router's zero fee.
No calls remain running, no full batch launched, and the failed original
configuration is blocked from accidental resumption. No reliable cheaper
full-corpus price or independent human reliability estimate is established.
Recommended next measurement separates changed decisions, clarification and
ambiguous candidates before scaling; this is not authority to launch it.

## 2026-10-03 successor — simplify, validate, scale passing features

**Explicit user decision:** "ok do it" accepted the proposed simpler Sol screen,
fresh source checks including negatives, then scaling only passing categories.
This supersedes the calibration-only restriction for a quality-gated feature
screen, not for the rejected rule-evolution or source-attribution pipeline.

The [frozen screen plan](../../research/narrative_evolution_20261003/feature_screen/PLAN.txt)
uses 36 fresh myths from 36 previously unreviewed runs: 24 cohort/order cases
and 12 lexical challenges. A source-first AI reference was frozen before API
labels; this is not independent human validation or a prevalence sample.
All 36 full-screen outputs completed with exact quotations. Only conditional
reduction and challenge dialogue passed the preset numerical thresholds.
The narrower two-category regression made reduction more uncertain (three-state
agreement 27/36), so it does not scale. Five joined presentation-enum values
were invalid; separate presence-only diagnostics preserve those raw failures.

Dialogue-only regression completed with all 36 technically valid outputs:
11/12 reference positives detected, no definite false positives, 33/36 three-
state matches. Form/stance are not quantitative endpoints for whole dialogues
containing both an objection and a reply. The full authorized screen therefore
counts only fictional challenge-dialogue presence/uncertainty, not punishment,
rule emergence, transmission or behavioral influence.

Direct OpenAI Batch execution has started for 38,720 myths / 6,454 requests.
The [batch plan](../../research/narrative_evolution_20261003/feature_screen/full_dialogue/plan.json)
estimates $62.50 including pilots and 25% allowance; worst-case pending token
reservations keep submissions within a $100 accounting guard. High reasoning
is unchanged; full requests allow 6,000 completion tokens (pilot maximum usage
1,773), with non-stop completions invalid rather than treated as negative.
Waves may require multiple Batch completion windows. Results are pending; no
full-corpus finding is established by submission. Generated payloads remain local
and reproducible from the pinned manifest, seed and code.
