# D007 — Keep myth prompts and game-side directives as separate interventions

- Recorded / last verified: 2026-09-15 / 2026-09-15
- Decision status: normative arm skipped for the July comparison; later game-side linkage changed across dated designs.
- Scope: descriptive/normative myth prompts and instructions to consider myths during game decisions.
- Decision authority: Aron documents prompt variants; Ed directs that the normative comparison can be skipped; later research-log decisions differ by experiment.
- Implementation status: multiple historical prompt regimes exist; no single label certifies rendered prompt equivalence.

## Decision and rationale

Treat a descriptive myth-writing request, a normative “how the game should be
played” request, and a game-side instruction to use myths as different causal
interventions. **Explicit rationale:** leading or game-linked language can
instruct the outcome being measured. July's decision to omit the normative arm
does not imply that all game-side myth reminders were removed from later runs.

## Evidence

- [May prompt chronology](../../experiment_design_reference.md#early-design-original-questions-and-changes) distinguishes the three interventions.
- [Slide 642 source record](../../experiment_design_reference.md#source-slide-642) and neighboring source slides preserve the compared prompts.
- [July 13 source records](../../experiment_design_reference.md#source-slide-676), Edward Hughes threads `AAAB_9gqYVg` and `AAAB_9gqYVs`: latest-prompt comparison requested and normative directive skipped.
- [Fork PR #52](https://github.com/ivarfresh/nips-linguistic-evolution-toolkit/pull/52) (config, launcher, completed-final receipt, 2026-11-27): Sonnet 4.5 dyad runs whose later-round myth prompt omits the self-inspiration sentence; the launcher proves that prompt is the only input differing from the paired September runs.
- [Unresolved design item 5](../../experiment_design_reference.md#superseded-choices-and-unresolved-questions) records July 23 guidance against `myth_decision_link` and a different August 12 choice.

## Chronology and supersession

Later directions are scoped to their comparisons. They do not retroactively
turn earlier directive, normative, and descriptive runs into the same
treatment. Match the exact rendered messages before comparing or resuming a
condition.

2026-11-27 (user instruction, Ivar): the self-inspiration sentence was deleted in a
separate ten-run Sonnet 4.5 dyad condition (`no_self_line_dyad_*_claude_n5`). It is a
distinct prompt regime: self-copying is 8.4 points lower than in the paired September
runs, so its myths must not be pooled with theirs. The September wording remains the
paper's condition; no change of default was decided.

2026-11-27 (team meeting, [Gemini-written notes](https://docs.google.com/document/d/17hQG5LqDBCbwG1QmMimmTdB53ieSpe2MLPH8hNk4W2o/edit)): the team agreed to run this
reviewer-requested ablation for dyads only, not for eight-agent populations or all task orders.

## Unresolved / next evidence

The held 180-run proposal does not yet specify which prompt/linkage regime it
inherits. Resolve this explicitly rather than relying on a condition name.

