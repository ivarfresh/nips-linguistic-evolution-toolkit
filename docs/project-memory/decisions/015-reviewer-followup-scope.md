# D015 — Narrow the reviewer follow-up

- Recorded / last verified: 2026-11-27 / 2026-11-27
- Decision status: agreed revision scope.
- Scope: retain two new myth conditions and use existing evidence for the other reviewer points. Skip other unrun controls for this revision.
- Decision authority: Ivar's 2026-11-27 conversation, relaying Edward Hughes's recommendation and directing manuscript TODOs; team meeting 2026-11-27 (Ivar, Aron, Edward) per Gemini-written notes.
- Implementation status: manuscript TODOs synced to Overleaf in Git revision `97b991f`. New conditions are pending design and execution.

## Decision and rationale

**Explicit.** Ed recommended conditions where agents write without reading and read without writing. Ivar kept those two in scope and asked to mark where existing results belong in the manuscript. **Inferred.** Together the two conditions help distinguish the effects of authoring text and seeing a live partner's text in the bundled myth task.

## Evidence

- [Overleaf manuscript](https://www.overleaf.com/project/6aa7e70288731d4b17434525), Git revision `97b991f`. TODOs appear in Methods, Results, Limitations, and the supplementary results.
- [Fixed-text transplant reruns](../../research/slide678_rerun_2026-09-16.md) supply inherited text to game-only agents. The live-partner read-only condition would change the source and timing of the text.
- [Corrected symmetric-noise runs](../../corrected_v2_confirmatory_overview_2026-08-13.md), [dyad regression audit](../../research/statistics_audit_20261123/section44/REPORT.txt), [founding-myth edit replay](../../research/myth_replay_probe_20261001/README.md), and [self-inspiration ablation](../../research/self_anchor_replay_20261002/README.md) give existing evidence or inputs for reanalysis.
- [Team meeting notes, 2026-11-27](https://docs.google.com/document/d/17hQG5LqDBCbwG1QmMimmTdB53ieSpe2MLPH8hNk4W2o/edit) (Gemini-written notes; not quotation authority) list the two conditions under agreed decisions and the symmetric-noise and neutral-writing-prompt controls under shelved ones. Ivar's agenda email of the same day proposed skipping symmetric noise (the existing Sonnet 4.5 symmetric runs had reasoning off, the main results have it on) and the memory-length ablation; the notes record no decision on the memory-length ablation.

## Chronology and supersession

The November 23 review discussion identified several possible controls. On November 27, Ivar narrowed the new-condition scope to the two writing and reading controls. The November 23 statistics audit still governs the interpretation of the existing estimates. The paper was withdrawn from AAMAS on November 24 (emails), so the scope now applies to the arXiv version, which the November 27 meeting notes target for before Christmas. The same notes record the agreement to delete Section 4.4 and to restrict the self-inspiration-sentence ablation to dyads ([D007](007-prompt-regimes.md)).

## Unresolved / next evidence

The exact task schedules, source of live myths in the read-only arm, matched controls, model profiles, sample sizes, and run budget remain open. Manuscript TODOs record where results can be added once the controls are run.
