# Manuscript diff — what to change given this week's analyses

Compares current manuscript state (as of 2026-04-30) against findings in
`HEADLINE_TABLES.md` and `RESULTS_SCAFFOLD.md`. Suggestions are scoped to
the NeurIPS submission deadline (May 4 / May 6 AoE). Per the project
pipeline, edits below should be made by humans (`b_draft.md`) or
ms-writer (`c_final.md`); I cannot make them directly.

**TL;DR:** The headline framing in §0 already matches the data well —
no major rewrites needed. §2 has one stale sentence to strike. §3
exists as a *partial* `c_final.md` and **needs significant expansion**
(missing §3.5; lag-1, reason-coding, embedding-convergence findings
absent). §4 outline needs three targeted updates to track what the data
actually showed. Five concrete additions and two deletions detailed
below.

---

## §0 Introduction (`_0-introduction.c_final.md`) — KEEP

The current framing is well-aligned with the data:

> "Myth-writing does not behave like a reliable cooperation switch. It
> sometimes shifts payoffs, but the sign and magnitude depend on the
> model and channel condition; its more plausible signal is
> consolidation and linguistic convergence within dyads."

This matches the headline findings exactly: 3/21 cells lift+consolidation,
1 consolidation, 2 lift, 10 null, 4 harmful/destabilizing.  No change.

**One small consideration:** the closing sentence — "this first design
does not yet show a robust myth-induced cooperation lift" — is technically
true but undersells the variance-reduction signal. Consider replacing
"does not yet show a robust myth-induced cooperation lift" with "does
not yet show a *uniform* lift, but does reveal a regime-conditional one".
This is a 5-word change, not a rewrite — judgment call whether it's
worth touching at this stage.

---

## §1 Background (`_1-background.c_final.md`) — KEEP

No changes prompted by this week's analyses. The myth-as-coordination /
narrative-as-compression framing remains the right scaffold.

---

## §2 Methods (`_2-method.c_final.md`) — ONE DELETION

### §3.5, last sentence: STRIKE

Currently:

> "Embedding-based and LLM-judge similarity are deferred to the
> follow-up analysis."

This is **no longer true**: `build_embedding_convergence.py` produces
`all-mpnet-base-v2` between-agent cosine convergence with bootstrap CIs
on the slope, on the v4_direct_provider corpus. Embedding-based is in
scope. (LLM-judge similarity remains deferred.)

**Suggested replacement:** *"Embedding-based between-agent convergence
uses sentence-transformers/all-mpnet-base-v2 with normalized cosine on
each agent pair per round; LLM-judge similarity is deferred to the
follow-up analysis."*

### §3.4, GPT-5-Nano floor-lock framing: CONSIDER REVISITING

Current §3.4 says nothing about GPT-5-Nano emitting only the bare JSON
action without reasoning prose. The §3 c_final mentions GPT-5-Nano "is
not simply floor-locked in the current direct-provider setup" but does
not flag the prose-emission asymmetry.

**Suggested addition (one sentence to §3.4):** *"In the current
direct-provider configuration, GPT-5-Nano emits only the JSON action
field with no surrounding reasoning prose, while Claude-Sonnet-4.5
emits extensive prose alongside the action — a difference that
constrains which models we can probe for cross-task linguistic carryover
(see §4.5)."*

---

## §3 Results (`_3-results.c_final.md`) — SIGNIFICANT EXPANSION

The current `c_final.md` has 4 subsections (§4.1–§4.4), but the outline
has 5 (§4.1–§4.5).  **§4.5 is missing entirely.**  The current §4.4 also
underweights the linguistic findings.  Concrete additions:

### Add to §4.1 / §4.2: explicit cell-level numbers

The current prose says things like "Claude under positive noise shows a
small positive `game -> myth` delta" without naming the number.  With
locked CIs in `cell_summaries/deltas.csv`, this can become:

- "Claude × positive (informed) × game→myth: Δmean +0.59 [+0.22, +0.96],
  variance ratio 0.31 [0.05, 0.66] — classified lift+consolidation
  (mean ↑, variance ↓)."
- "GPT-5-Nano × positive (informed) × game→myth: Δmean +2.23 [+1.06,
  +3.52], variance ratio 0.08 [0.01, 0.24] — the strongest
  lift+consolidation cell in the corpus."
- "GPT-5-Nano × bootstrap (informed) × game→myth: Δmean −9.67 [−14.67,
  −5.00], variance ratio 7.5 [3.5, 15.8] — destabilizing."

### Add to §4.2: a 3×3 classification breakdown sentence

> "Of 21 (model × noise × myth-order) cells, 3 are classified
> lift+consolidation, 1 pure consolidation, 2 pure lift, 10 null, 2
> harmful, and 2 destabilizing (bootstrap CI on Δmean and variance
> ratio).  Lift+consolidation lives entirely in upward-noise cells;
> destabilization lives entirely in GPT-5-Nano × bootstrap. The myth
> effect is **conditional on the noise regime**, not universal."

This sentence carries most of the §4.2 information and is more honest
than "the central pattern is not a uniform lift".

**Recommended figure to insert here:** the decision-table heatmap at
`analysis/figures/decision_table.png` — a single image that visualises
the 21-cell breakdown.  Drop into `manuscript/figures/` as
`fig_decision_table.png` and reference as `{#fig-decision}`.

### Replace §4.4 (lexical Jaccard) with a stronger §4.4

The current §4.4 reports between-agent lexical Jaccard, which is a weak
proxy and the prose acknowledges as much.  Two stronger findings should
appear here:

#### Stronger finding 1: between-agent embedding cosine convergence

From `embedding_summary.csv`: 18 of 21 cells show statistically positive
slope of cosine over rounds (CI excludes 0 above).  Strongest:

- GPT-5-Nano × negative_5 × myth→game: slope +0.027/round [0.016, 0.038],
  cosine 0.42 → 0.74 across 10 rounds.
- GPT-5-Nano × bootstrap × myth→game: slope +0.023/round [0.013, 0.032].
- Claude × positive (informed) × myth→game: slope +0.021/round
  [0.013, 0.028].

This is much sharper than Jaccard and supports the "linguistic
convergence is robust regardless of behavioural outcome" line that the
discussion will need.

#### Stronger finding 2: lag-1 cross-agent cooperativity-language correlations

From `lag_summary.csv`: positive in *every* cell.  Mean per-dyad
max-direction Pearson r 0.27–0.58, with 27–73% of dyads showing |r|>0.5.
Strongest:

- GPT-5-Nano × positive (informed) × myth→game: lag1_max mean 0.58
  [0.43, 0.71], 73% of dyads above 0.5.

This **replicates the project's pilot Claude r ≈ 0.72 finding at corpus
scale** and should be presented that way: not as a striking case study
but as the typical pattern.

### Add §4.5 (currently missing entirely): Coupling between game behaviour and myth content

This subsection is in the outline but absent from `c_final`.  Two
findings to include, both from this week's analyses:

#### Reason-coding asymmetry (`reason_coding_summary.csv`)

- **Claude:** 78–82% of game-response prose contains a content word
  (length ≥5, non-stopword) that appears in the agent's own preceding
  myth chain.  Mean ~5–7 unique vocabulary overlaps per reason. **100%
  of runs** have at least one such reference.  Theme-lexicon hits
  (story / spirit / elder / sacred / ancestor / ritual / etc.) reach
  10–33% of reasons.
- **GPT-5-Nano:** 0% across every cell — **but only because it emits no
  visible reasoning prose at all** (`game_responses[ag].content` is the
  bare JSON action).  The cross-task channel is methodologically
  undecidable for this model from the visible output.

#### Coinage non-leakage (`neologism_summary.csv`)

The pilot observation that made-up myth-coinages reappear in agents'
game-reasoning text **does not replicate at corpus scale**.  Across
600+ chains, share of runs with any myth-coinage in any reason text =
0.00, in every cell.  Coinages do persist *within* myth chains —
Claude's "Lumina", "Aelara", "Arachnis" appear in all 10 rounds of
specific chains — but they stay inside the myth.  Linguistic carryover,
where it exists (Claude's reasons), travels through *thematic content*
rather than through coined vocabulary.

### Suggested §4.5 framing line

The Tier-2 outline framed §4.5 as "linguistic-behavioural coupling is
real but weak in aggregate, and concentrated in particular dyads".  The
data tell a sharper story:

> "Linguistic carryover from myth to game-reasoning is dramatic for
> Claude (~80% of rounds) and methodologically undecidable for
> GPT-5-Nano (no visible reasoning prose).  The carryover travels
> through *thematic vocabulary*, not through invented neologisms — the
> pilot observation that coined myth-words leak into game reasoning
> does not replicate at corpus scale."

---

## §4 Discussion (`_4-discussion.a_outline.md`) — THREE TARGETED UPDATES

The outline has good bones.  Three things should change before
prose-ification:

### §5.1: the negative-result framing is too monolithic

Currently:

> "in most cells, adding myth-writing does not produce a statistically
> robust lift in mean cumulative reward. This is the result, not a
> failure of the experiment."

Update to acknowledge the **bidirectional** finding:

> "In ~half of cells, adding myth-writing has no detectable effect on
> mean cumulative reward.  In a clearly identifiable subset (upward-
> noise cells across both models), it produces a small mean lift
> combined with a substantial variance reduction.  In another
> identifiable subset (GPT-5-Nano × bootstrap noise), it actively
> harms or destabilises cooperation.  The effect is real, sign-
> conditional on the noise regime, and asymmetric across models."

### §5.2: "strategy consolidation as the primary effect" is too strong

The `[CHECK]` flag on the original outline is doing real work —
consolidation is supported in 4/21 cells (3 lift+consolidation +
1 pure), not as a universal effect.  Reframe as "consolidation under
favourable noise; destabilization under reciprocity-flipping noise" —
two patterns, not one.

Concretely: keep most of the §5.2 prose but add a paragraph contrasting
the bootstrap-noise harm pattern.  The mechanism speculation (the
"coordination commitment device" / "self-consistent narrative anchor")
is fine — but it needs a complement explaining why bootstrap noise
*reverses* the effect.  Suggested mechanism: bootstrap noise replaces
returned amounts with full-reciprocation reports, so agents observe a
signal that systematically *contradicts* what their game-math would
predict.  The myth then provides additional structured priors that
compete with the noisy reciprocation signal — increasing within-dyad
disagreement rather than locking in a coherent strategy.

### §5.3: cross-model variation framing — narrow scope honestly

Currently §5.3 leans on three-model comparison (Claude / Gemini /
GPT-5-Nano).  But the v4 / submission scope is two models (Claude +
GPT-5-Nano), with Gemini explicitly out of scope per §3.4.

Two options:

a. Pull Gemini cells from the prior baseline data and include them in
   §3.1 (cross-model regimes baseline) only.  Then §5.3 can keep the
   three-model framing.
b. Narrow §5.3 to two-model contrast and use the embedding-convergence
   universality (18/21 cells positive slope) as a *within-paper* check
   that the linguistic phenomenon is not model-specific.

Option (b) is cleaner for the timeline.  Option (a) is stronger for
the paper.

### §5.4: upgrade the "case study" framing of cross-task influence

Currently §5.4 says:

> "even through the constrained channel, dyads show measurable
> linguistic convergence (§4.4) and at least one case of strong lag-1
> cross-agent correlation in cooperativity language (§4.5)."

The "at least one case" understates what the data show.  Replace with:

> "Across the corpus, lag-1 cross-agent correlations in cooperativity
> language are positive in every cell (mean per-dyad max-direction
> Pearson r 0.27–0.58; 27–73% of dyads show |r|>0.5).  The pilot's
> Claude r ≈ 0.72 case sits at the upper tail of a clearly non-zero
> distribution, not as an anomaly.  Combined with the embedding-cosine
> convergence finding (§4.4) and the Claude reason-vocabulary carryover
> (§4.5), the linguistic channel is doing measurable work even when
> it does not lift mean cooperation."

### §5.5 Limitations: add one sentence

Add to the limitations list:

> "**Reasoning-prose visibility.** GPT-5-Nano emits only the structured
> JSON action with no surrounding reasoning prose.  The cross-task
> linguistic channel is therefore observable only for models that emit
> reasoning text alongside the action (Claude in our corpus).  This is
> a methodological constraint, not a substantive claim about
> GPT-5-Nano's internal use of the myth content."

### §5.6 Future directions: add one item

Add as a bullet (after the existing "direct partner-myth injection"
item):

> "**Forced reasoning prose for action-only models.** Re-run GPT-5-Nano
> with a system prompt that requires a reasoning-prose preamble before
> the JSON action, and re-code reason text against own-myth vocabulary.
> Tests whether the cross-task channel is genuinely absent for that
> model or merely invisible in the current configuration."

---

## §5 Appendix (`_5-appendix.c_final.md`) — NOT REVIEWED

Appendix can absorb the full per-cell tables from `cell_summary.csv`,
`deltas.csv`, `lag_summary.csv`, `embedding_summary.csv`,
`reason_coding_summary.csv`, and `neologism_summary.csv` if length
allows.  Recommend including at minimum:

1. The 3×3 classification table for all 21 cells (Δmean + variance
   ratio + classification, full bootstrap CIs).
2. The lag-1 correlation summary (one row per cell).
3. The embedding-convergence slope table.

These are short — together, ~50 rows of small-font data.

---

## Concrete edits in priority order

1. **`_3-results.c_final.md`** — needs the most work.  Add specific
   numbers, add §4.5 (entirely missing), upgrade §4.4 (Jaccard →
   embedding cosine + lag-1).  This is the single biggest gap. **Run
   ms-writer on a curated `_3-results.b_draft.md`** that pulls from
   `RESULTS_SCAFFOLD.md`.
2. **`_4-discussion.b_draft.md`** — currently empty.  Curate from the
   outline + the four updates above.
3. **`_2-method.c_final.md` §3.5** — strike the "embedding deferred"
   sentence; add the optional reasoning-prose-visibility note to §3.4.
4. **`_0-introduction.c_final.md`** — optional 5-word tweak; not
   required for submission.
5. **`_5-appendix.c_final.md`** — add per-cell tables if length allows
   (post-§3 prose pass).

---

## Files to point ms-writer at

When invoking ms-writer for §3 final pass:

- Source: a freshly curated `_3-results.b_draft.md` populated from
  `analysis/RESULTS_SCAFFOLD.md`.
- Reference numbers: `analysis/cell_summaries/deltas.csv`,
  `cell_summary.csv`, `lag_summary.csv`, `embedding_summary.csv`,
  `reason_coding_summary.csv`, `neologism_summary.csv`.
- Reference figures: `analysis/figures/decision_table.png`,
  `analysis/figures/trajectories.png` (move into `manuscript/figures/`
  before render and rename to fit existing `fig2_*` / `fig3_*` /
  `fig4_*` scheme).

When invoking ms-writer for §4 final pass:

- Source: a curated `_4-discussion.b_draft.md` reflecting the four
  outline updates above.
- Cross-link results section: §3 must be locked first so §4 can cite
  specific cells correctly.
