# NeurIPS submission runbook: Sunday May 3 to Thursday May 7

Last updated: Sunday 2026-05-03, 23:07 CEST.

## Deadlines

Use these as the working deadlines:

- Abstract: Tuesday 2026-05-05, 13:59 CEST. Internally treat as 14:00 CEST.
- Full paper and supplementary materials: Thursday 2026-05-07, 13:59 CEST. Internally treat as 14:00 CEST.

These correspond to the official NeurIPS 2026 main-track deadlines of May 4 and May 6 Anywhere on Earth:
https://neurips.cc/Conferences/2026/CallForPapers

## Current State

The Sunday-night team update is done and packaged:

- Team text: `analysis/team_update_2026-05-03/TEAM_UPDATE_TO_SEND.md`
- Packet README: `analysis/team_update_2026-05-03/README.md`
- Recommended figures:
  - `fig2_bootstrap_mechanism_game_myth.png`
  - `fig3_boundary_conditions_game_myth.png`
  - `fig4_neutral_framing_trajectories.png`

No Claude runs were launched tonight. The GPT-5.5 / Gemini missing-condition sweep is running in the background under `nlet_missing_conditions_20260503`. As of 23:07 CEST, it was still in the first Gemini-positive completion set, with 60/90 final files and no new quota/auth failures.

The paper story should remain mechanism-led:

> Inter-agent language is not a uniform cooperation switch. Most myth-writing effects are null or small. The sharp result is conditional: under GPT-5-Nano bootstrap noise, the implicit myth channel destabilises cooperation, while putting coherent text under the partner-story prompt header restores cooperation to the game-only baseline. The filler result means this is not partner-specific narrative content; it is a channel-opening / coherence-anchor effect.

## Critical Path

There are only four decisions that matter before the abstract:

1. **Story lock, Monday 11:00-12:00.** Choose mechanism-led framing. Heterogeneity/model-boundary results support the story; they do not become the story.
2. **Experiment lock, Monday 17:00.** Decide what, if anything, from prompt variants / Claude / GPT-5.5 / Gemini is allowed into the main text. Everything else moves to appendix or notes.
3. **Abstract lock, Tuesday 12:00.** After this, only factual fixes. The abstract must not depend on unfinished analyses.
4. **Paper consistency lock, Wednesday evening.** Results, discussion, figures, captions, and appendix must say the same thing with the same scope.

The default plan is therefore:

- Send the Sunday team update.
- Let the existing model-expansion sweep run.
- Use Monday's meeting to make a small number of explicit decisions.
- Run only meeting-approved follow-ups.
- Freeze the main claim Monday 17:00 even if tempting extra results are still pending.
- Spend Tuesday/Wednesday making the paper coherent rather than making it bigger.

## Non-Negotiables

- Abstract/story consistency beats more experiments.
- New results only enter the main text if they cleanly sharpen the mechanism by Monday 17:00.
- Ambiguous prompt-variant or model-expansion results go to appendix, briefing notes, or post-submission follow-up.
- No Claude before the Monday 11:00 meeting unless the team explicitly asks for it.
- No new experiments after Wednesday morning except emergency backfills for already-claimed results.

## Decision Tree

### Model-Expansion Sweep

If GPT-5.5 / Gemini finish cleanly by Monday morning:

- Rebuild summaries.
- Use them as robustness/boundary evidence.
- Include in abstract only if they directly replicate or sharpen the mechanism.

If they are incomplete by Monday morning:

- Bring status only.
- Do not wait for them to write the abstract.

If Gemini quota fails:

- Stop Gemini.
- Continue GPT-5.5 if the launcher reaches it cleanly.
- Report Gemini as incomplete.

### Claude

If the team says "single-model mechanism is fine":

- Run no Claude before abstract.
- Mention Claude only as existing model-boundary evidence.

If the team says "reviewers will ask whether Claude also does this":

- Confirm Anthropic credits first.
- Run Claude Sonnet 4.5 x bootstrap baseline only.
- Continue to partner-story/filler controls only if Claude baseline shows the relevant bootstrap myth destabilisation.

If Claude baseline is null or saturated:

- Stop. Treat as boundary evidence, not a failed replication.

### Prompt Variants

If Ivar thinks the current prompt worry is central:

- Run only the small GPT-5-Nano x bootstrap battery.
- Use the results only if clean by Monday 17:00.

If Ivar thinks the prompt worry can be appendix/future-work:

- Do not run prompt variants before abstract.
- Add one limitations sentence about system-prompt/game-knowledge dependence.

If prompt variants are messy:

- Omit from main paper.
- Do not let them destabilise the abstract unless they directly falsify the mechanism wording.

## Monday Meeting Packet

The Monday 11:00 meeting should fit on one page:

1. **One-sentence claim.** Under GPT-5-Nano bootstrap noise, implicit myth-writing destabilises cooperation, but explicitly placing coherent text under the partner-story header restores the game-only trajectory; filler works, so the mechanism is channel-opening/coherence, not partner-specific narrative content.
2. **Figures.** Use Fig. 2 bootstrap mechanism, Fig. 3 boundary conditions, Fig. 4 neutral framing.
3. **Boundary notes.** No-noise null; positive_5 saturated; negative_5 not restored; neutral framing opens headroom for GPT-5-Nano; Claude moves less.
4. **Running experiments.** GPT-5.5 / Gemini sweep still running or completed, depending on morning status. No Claude run yet.
5. **Decisions needed.** Story, Claude, prompt variants, review owners.

## Team Asks

Send or ask these directly:

- **Mario:** "Can you sanity-check whether the abstract should lead with the mechanism result rather than the heterogeneous-effects framing, and return concrete title/abstract wording by Monday 17:00?"
- **Edward:** "Does 'channel-opening / coherence anchor' sound like a defensible mechanism, or like a prompt artifact? What wording would make the claim precise without overclaiming?"
- **Ivar:** "Do the bootstrap controls and boundary tests address the Apr 24 concerns enough for submission, and should we spend Monday afternoon on the no-game-knowledge / less-constrained prompt variants?"
- **Alexandra / Arabella / Jane:** "After Monday evening's reframe, can you do a clarity/accuracy pass only, focusing on claims that feel too strong or unexplained?"

## Sunday Night, May 3

Primary goal: send a sensible team update before bed.

Checklist:

- Send or paste `analysis/team_update_2026-05-03/TEAM_UPDATE_TO_SEND.md`.
- Attach `fig2_bootstrap_mechanism_game_myth.png` and `fig3_boundary_conditions_game_myth.png`.
- Attach `fig4_neutral_framing_trajectories.png` if keeping the neutral-framing robustness paragraph.
- Mention that GPT-5.5 / Gemini model-expansion runs are still running and that no Claude was launched tonight.
- Do not spend more time polishing the team update once it is sent.

Optional before sleep:

- Check `/tmp/nlet-runs/missing-conditions/_summary.log`.
- If the sweep completes a set, do nothing unless there is an obvious fatal quota/auth error.

## Monday May 4, 08:00-10:30

Goal: prepare a compact meeting packet, not a full rewrite.

Checklist:

- Check sweep status:

```bash
tail -n 80 /tmp/nlet-runs/missing-conditions/_summary.log
find data/json/noise_experiments/v4_direct_provider -name "*.json" -not -name "*checkpoint*" -not -name "*results*" -not -name "*error*" -newer /tmp/nlet-runs/missing-conditions/_start.marker | wc -l
```

- If the model sweep has completed, rebuild the core summary:

```bash
python3 projects/neurips-2026-llm-ling-evo/analysis/build_cell_summary.py
python3 projects/neurips-2026-llm-ling-evo/analysis/print_control_matrix.py --model gpt-5.5
python3 projects/neurips-2026-llm-ling-evo/analysis/print_control_matrix.py --model gemini-3.1-pro-preview
```

- If the sweep is still running, bring only a status note.
- Prepare a one-page meeting packet:
  - one-sentence abstract spine;
  - the three team-update figures;
  - current run status;
  - decisions needed below.

## Monday May 4, 11:00 Team Meeting

Use the Sunday update as the agenda. The meeting should decide these items.

### 1. Story

Decision: mechanism-led vs heterogeneity-led.

Recommendation: mechanism-led.

Ask Mario: Does the abstract read as a clean mechanism claim rather than a grab-bag of heterogeneous effects? Please return final abstract framing by Monday 17:00.

Ask Edward: Does "channel-opening / coherence anchor" sound defensible, or does it read like a prompt artifact? What would make the wording more precise?

Ask Ivar: Does the current mechanism interpretation respect the noise design and the Apr 24 prompt concerns? Are there any fatal confounds we should address before abstract submission?

### 2. Claude

Decision: whether Claude is necessary before submission.

Recommendation: optional; run only if the team thinks single-model mechanism evidence is a review risk.

If approved:

- First run Claude Sonnet 4.5 x bootstrap baseline: 90 runs.
- If Claude shows bootstrap myth destabilisation, add partner-story and filler controls: up to 120 more runs.
- If Claude is null or saturated, stop and report the mechanism as GPT-5-Nano-established with model-boundary evidence.

Ask the team: Is the review risk worse if we keep the mechanism single-model, or if we add a rushed Claude branch that may be null?

### 3. Prompt Variants

Decision: whether Ivar's less-constrained / no-game-knowledge prompt variants belong in the submission.

Recommendation: small GPT-5-Nano x bootstrap-only battery, frozen by Monday 17:00.

If approved:

- Unconstrained myth prompt: 60 runs.
- Myth-first blind system prompt: 30 runs.
- Myth-first blind plus unconstrained: 30 runs.

Inclusion rule:

- Clean and interpretable by Monday 17:00: add to appendix or one sentence in limitations.
- Ambiguous: omit from main paper.
- Contradictory: use only as an honest boundary note if it changes an abstract claim.

### 4. Review Roles

Proposed asks:

- Mario: abstract and title by Monday 17:00.
- Edward: mechanism sanity check by Monday 17:00.
- Ivar: experiment/prioritisation advice immediately after meeting.
- Alexandra, Arabella, Jane: clarity pass after Monday evening reframe; blocking clarity/accuracy only.

## Monday May 4, 12:00-17:00

Goal: execute only meeting-approved experiments and freeze experimental scope.

Default path:

- Let GPT-5.5 / Gemini sweep finish.
- Build `analysis/MORNING_BRIEFING_MISSING_CONDITIONS.md`.
- Treat model-expansion results as robustness/boundary evidence unless unexpectedly decisive.

Conditional path A: prompt variants approved.

- Implement only the minimal prompt/flow changes needed for the three GPT-5-Nano x bootstrap variant cells.
- Smoke-test one seed per new set.
- Run with conservative worker count.
- Freeze by 17:00.

Conditional path B: Claude approved.

- Confirm Anthropic credit health first.
- Run bootstrap baseline before any Claude controls.
- Stop immediately if the baseline is null/saturated.

Decision at 17:00:

- Main-text claims are frozen.
- Abstract wording is frozen except for coauthor clarity edits.
- Any incomplete experiment becomes appendix/post-submission material.

## Monday May 4, 17:00-22:00

Goal: align abstract, results, discussion, and figures around one story.

Writing tasks:

- Rewrite the abstract around the mechanism spine.
- Update results section wording to avoid "partner myth specifically" and "rescue" unless carefully defined.
- Reframe Chwe/common-knowledge as partly consistent with visibility, not evidence for partner-specific narrative content.
- Add or update the bootstrap mechanism figure and neutral-framing appendix/reference as needed.
- Ensure the neutral-framing claim is precise: GPT-5-Nano drops under ROLE A/B and myth-present conditions recover some of that loss; Claude moves less and remains mostly myth-null.

End-of-day checks:

- No claim in abstract depends on an unfinished experiment.
- Figure captions state model, noise condition, and n.
- The text does not use unexplained internal condition labels; every condition is named in plain English before any shorthand appears.

## Tuesday May 5, 08:00-12:00

Goal: abstract submission package.

Checklist:

- Incorporate Mario / Edward comments.
- Verify title, abstract, keywords, authors, and OpenReview profiles.
- Make sure the abstract does not overclaim cross-model generality.
- If model-expansion results finished overnight, mention them only if they directly support a sentence already in the abstract.
- Stop abstract editing by 12:00 CEST unless there is a factual error.

Recommended abstract claim shape:

1. Broad question: whether inter-agent language is load-bearing in LLM cooperation.
2. Empirical pattern: effects are conditional, mostly null/small across cells.
3. Mechanism result: GPT-5-Nano bootstrap destabilisation is reversed by partner-story-labelled coherent text.
4. Interpretation: channel-opening / coherence anchor; not partner-specific narrative content.
5. Boundary: model/framing/noise regime matter.

Submit abstract by 12:00 if possible. The hard internal buffer ends at 13:30 CEST.

## Tuesday May 5, Afternoon and Evening

Goal: full-paper consistency pass.

Work items:

- Regenerate analyses if new data were accepted into the paper.
- Update results tables and captions.
- Compress overlong results/discussion prose.
- Check that limitations explicitly say the mechanism is GPT-5-Nano-established unless Claude/Gemini evidence is clean.
- Update appendix for omitted/ambiguous prompt variants if useful.

Do not start new experiments Tuesday evening unless they are tiny emergency backfills for a claim already in the paper.

## Wednesday May 6

Goal: final paper package, no new science.

Morning:

- Rebuild analysis artifacts.
- Rebuild Overleaf bundle:

```bash
python3 projects/neurips-2026-llm-ling-evo/analysis/rebuild_overleaf.py
```

- Render PDF.
- Check page count, figures, tables, citations, anonymity, author list, acknowledgments, supplementary references, and OpenReview requirements.

Afternoon:

- Coauthor pass: blocking accuracy/clarity only.
- Resolve comments directly in source.
- No wording churn unless it fixes a real issue.

Evening:

- Final read-through against abstract.
- Confirm no stale "partner-myth-specific" language remains.
- Confirm all figure captions are self-contained.
- Confirm data availability / code statement is acceptable for anonymous review.

## Thursday May 7, 08:00-14:00

Goal: submit with buffer.

08:00-10:00:

- Final render.
- Check PDF visually.
- Verify supplementary ZIP if used.

10:00-11:30:

- Submission checklist:
  - PDF uploads cleanly.
  - Page limits satisfied.
  - Supplementary material uploaded if included.
  - Author/coauthor registration complete.
  - Conflict/profile metadata complete.
  - Abstract matches submitted abstract unless platform allows/needs updates.
  - Anonymous formatting checked.

12:00-12:30:

- Submit full paper.

12:30-14:00:

- Buffer for upload failures, metadata issues, or coauthor registration problems.

## Experiment Triage Rules

GPT-5.5 / Gemini missing-condition sweep:

- Use as robustness/boundary evidence.
- Do not rebuild abstract around it unless it strongly replicates the mechanism.
- If Gemini quota blocks, report as incomplete; do not chase.

Claude:

- Post-meeting only.
- Baseline first.
- Controls only if baseline shows the relevant bootstrap destabilisation.

Prompt variants:

- GPT-5-Nano x bootstrap only.
- Freeze Monday 17:00.
- Ambiguous results do not enter the main story.

## Main Risks

- Overclaiming common-knowledge / Chwe: solve by saying "partly consistent with visibility/common-knowledge, not partner-specific narrative content."
- Overclaiming cross-model generality: solve by making GPT-5-Nano the mechanism model and other models boundary evidence.
- Experiment creep: solve by freezing Monday 17:00.
- Abstract-paper mismatch: solve by making Tuesday/Wednesday consistency checks explicit.
- Page pressure: solve by moving prompt variants/model expansion to appendix unless they are essential.

## One-Sentence North Star

The submission should argue that inter-agent language can be load-bearing, but only under specific model/noise/framing regimes, and the cleanest mechanism is not the semantic content of the partner's myth but the opening of a coherent partner-story channel in a destabilised game context.
