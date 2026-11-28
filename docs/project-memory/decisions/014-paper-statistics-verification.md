# D014 — Verify paper Sections 4.4 and 4.5

- Recorded / last verified: 2026-11-23 / 2026-11-23
- Decision status: agreed (audit authorization); manuscript revisions and rebut-or-withdraw unresolved.
- Scope: Sections 4.4/4.5 and directly supporting methods, figures and supplement; existing data only.
- Decision authority: Ivar's session instruction to verify the full statistics and Holm correction and supply all code; subsequent explicit approval of a four-agent audit team.
- Implementation status: implemented; coefficients/tests rerun and completed final-state sources verified. No manuscript or original analysis code changed.

## Decision and rationale

**Explicit:** Ivar requested numerical verification and the full code trail.
The audit reproduces the headline numerical claims and all 444 checked stored
Holm adjustments. This settles the arithmetic and source traceability, not the
causal mechanism, human validity of judge labels, or a decision to rebut.

**Measured qualifications:** the supplement's opening-slope interval is labeled
as resampling runs but actually uses run-clustered normal inference. The opening
adjustment adds run-level round-1 transfer to an OLS model of later transfer;
$0.4163→$0.1712 reproduces in 60 single-model runs, but the reduction itself was
not tested and conditioning on the opening does not identify mediation.

The moral-matching p-values are raw; .015369 becomes .061477 under an illustrative
four-headline-test Holm family (.030739 under two tests). The writing-amount
coefficient .073887 has saved Holm p=.017972 for 25 selected primary tests,
.055714 for 62 related tests, and .142879 for 178 estimable tests. No separate
dated preregistration of the 25-test selection was located. Both reading-amount
models use a named amount or a rule-category midpoint, including narrated amounts.
The later 71% is 375/527 eligible sender occasions, not all later myths.

**Inferred interpretation:** the sections do not test contradictory hypotheses:
specific founding amounts before play and broad moral labels during play are
different predictors and times. The strong word-adoption results remain under
their declared Holm families; null behavioral associations do not prove no effect.

## Evidence

- Primary instruction: Ivar, current session, 2026-11-23; authored request and team approval.
- [Consolidated completed audit](../../research/statistics_audit_20261123/REPORT.txt), with three specialist reports and full-precision numerical outputs.
- [All 444 independently verified corrections](../../research/statistics_audit_20261123/holm_provenance/all_corrections.csv).
- [Source-final hash/completion ledger](../../research/statistics_audit_20261123/holm_provenance/manifest_source_hashes_and_completion.csv): 771 distinct completed finals, 2,247 manifest references, no mismatches; 126 registered output hashes also match.
- [Raw-to-cache reconstruction checks](../../research/statistics_audit_20261123/holm_provenance/raw_to_cache_checks.csv): September, n=10 and frontier myths/decisions match the original finals and condition metadata.
- [Source code index](../../research/statistics_audit_20261123/CODE_INDEX.txt) and [hashed source manifest](../../research/statistics_audit_20261123/source_manifest.json).
- Manuscript source: fetched Overleaf origin/main `cb6336c39afcad6f3fc193502113f3ff2f827c19` (2026-11-20); scoped results/methods/supplement files identical to local checkout. Source snapshots accompany the audit bundle.

## Chronology and supersession

- 2026-11-23 meeting: Sections 4.4/4.5 statistics and the opening adjustment were unresolved; see the meeting entry in CURRENT.md.
- 2026-11-23 authorized audit: arithmetic and provenance verified; the qualifications above replace the earlier blanket “under verification” status for these calculations. The paper decision, interpretation and edits remain open.
- 2026-11-24 emails: Ivar chose not to continue at AAMAS, citing among other things that Section 4.4 needed more time or removal; Edward agreed and Aron withdrew the submission. 2026-11-27 meeting ([Gemini-written notes](https://docs.google.com/document/d/17hQG5LqDBCbwG1QmMimmTdB53ieSpe2MLPH8hNk4W2o/edit)): the team agreed to delete Section 4.4 from the draft. The rebut-or-withdraw question in the status line above is thereby closed; this audit's numbers are unchanged.
- Earlier broad descriptions such as “the $0.07 result fails Holm” require a named family; both passing and failing stored corrections exist. No original exploratory result file was rewritten.

## Unresolved / next evidence

Authors must choose consistent wording and interval methods and identify correction
families explicitly. Human validation of labels, causal mediation, and preregistered
status of exploratory selections are not established by this audit. Historical
worktree paths in some scripts/manifests are stale; matching data exist and the
audit wrappers reproduce the results. No new experiment or paid call is authorized.
