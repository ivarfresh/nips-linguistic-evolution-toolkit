# LLM linguistic evolution — week update for the team

*Aron, 2026-04-30. NeurIPS 2026 deadline: abstract Mon May 4 AoE, full
paper Wed May 6 AoE.*

## What I did this week

1. **Fixed the noise setup.** The Apr 28 meeting flagged that noise was
   being applied inconsistently across decision variables and that
   cumulative-balance logging confused agent intent with the perturbed
   ledger. The fix is in: noise now applies uniformly to send/return
   decisions for all models, the perturbed ledger is what gets logged
   as the cumulative balance, and the agent's pre-perturbation decision
   is logged separately as `*_decision`.

2. **Ran a corrected set of experiments** under the new pipeline,
   scoped to **Claude Sonnet 4.5 and GPT-5-Nano** through the
   direct-provider routes (Anthropic and OpenAI APIs, not via
   OpenRouter). N=15 seeds per (model × noise × task-order) cell;
   neutral persona, `anything` myth topic, T=10 rounds, mem-3.

3. **Added a positive (upward) noise condition** alongside the existing
   negative-only and bootstrap. This was the symmetry check the Apr 28
   meeting asked for: if downward perturbation depresses cooperation
   and bootstrap rescues GPT-5-Nano, what does upward perturbation do?

4. **Ran the full Tier-1 + Tier-2 analysis stack** on the corrected
   data: cell-level mean/median/IQR with bootstrap CIs, a 3×3
   classification of myth effects (mean shift × variance shift),
   between-agent embedding cosine convergence, lag-1 cross-agent
   cooperativity correlations, neologism persistence, and lexical
   reason-field coding. All scripts live in
   `projects/neurips-2026-llm-ling-evo/analysis/build_*.py` and are
   deterministic at RNG seed 42.

## What I found

### The myth effect is conditional, not uniform

Of 21 (model × noise × myth-order) cells, the bootstrap-CI
classification gives:

- **3 lift+consolidation** (mean ↑, variance ↓, both significant) —
  strongest myth signal
- **1 pure consolidation** (variance ↓, mean flat)
- **2 pure lift** (mean ↑, variance flat)
- **10 null** (no detectable effect)
- **2 harmful** (mean ↓, variance flat)
- **2 destabilizing** (mean ↓, variance ↑)
- 1 missing (insufficient data)

**All lift+consolidation cells live in the upward-noise (positive ±$5)
regime.** The strongest is GPT-5-Nano × positive (informed) ×
`game→myth`: Δmean +2.23 [+1.06, +3.52], variance ratio 0.08 [0.01,
0.24] — a 12× variance reduction.

**All destabilization cells live in GPT-5-Nano × bootstrap noise.**
Adding myth to a dyad already operating under reciprocity-flipping
noise depresses cumulative reward by 7–10 points and increases
across-seed variance 5–8×. The mechanism: bootstrap noise replaces the
returned amount with full reciprocation every round, so the perceived
signal contradicts the agent's own game-math; adding myth introduces a
second structured prior that competes rather than reinforces.

### Linguistic dynamics are robust regardless of behavioural outcome

Three layers of evidence from the linguistic side, all independent:

- **Between-agent embedding cosine convergence**: positive slope over
  rounds in **18 of 21 cells**. Strongest cell: GPT-5-Nano × negative_5
  × `myth→game`, slope +0.027/round, cosine 0.42 → 0.74 across 10
  rounds.
- **Lag-1 cross-agent cooperativity correlations**: positive in
  **every** cell. Mean per-dyad max-direction Pearson r ranges
  0.27–0.58, with **27–73% of dyads showing |r| > 0.5**. The pilot
  Claude r ≈ 0.72 finding sits at the upper tail of a clearly non-zero
  distribution; it is representative, not anomalous.
- **Reason-field carryover** (Claude only): Claude's game-response
  prose contains content words from its own preceding myth in **78–82%
  of rounds**, with 100% of runs having at least one such reference.
  GPT-5-Nano emits no reasoning prose — bare JSON action only — so the
  cross-task channel is methodologically undecidable for that model in
  this configuration.

### Two pilot observations did not replicate at corpus scale

- **Coinage leakage from myths to reasons**: the pilot example of
  invented words like "kyrexladilokrater" appearing in game-reasoning
  text **did not replicate**. Across 600+ chains, share of runs where
  any myth-coinage reappears in any agent's reason text = 0.00, in
  every cell. Coinages do persist *within* myth chains (Claude's
  "Lumina", "Aelara", "Arachnis" appear in all 10 rounds of specific
  chains), but they stay inside the myth.
- **GPT-5-Nano floor-locking under no-noise**: in the older OpenRouter
  configuration, GPT-5-Nano looked floor-locked at mutual defection.
  Under direct-provider, the no-noise baseline is mean cumulative
  balance ~70/150 — moderate cooperation. Cross-model gap to Claude
  (~85/150) is real but not a chasm.

## What this means for the paper

The §0 narrow-claim framing — "language is not mere decoration but does
not lift cooperation uniformly" — survives the corrected data, but the
story is sharper than "consolidation". Two patterns: **myth consolidates
under favourable noise; myth destabilises under reciprocity-flipping
noise**. This is more interesting than a uniform-effect or pure-null
result and is honestly defensible.

I've drafted candidate prose for §3 (Results) and §4 (Discussion)
reflecting these findings, with all numbers locked from the
reproducible CSVs. Drafts at:

- `projects/neurips-2026-llm-ling-evo/analysis/_3-results.b_draft.CANDIDATE.md`
- `projects/neurips-2026-llm-ling-evo/analysis/_4-discussion.b_draft.CANDIDATE.md`

Plus a section-by-section diff against current manuscript:

- `projects/neurips-2026-llm-ling-evo/analysis/MANUSCRIPT_DIFF.md`

And two abstract candidates (200 / 230 words):

- `projects/neurips-2026-llm-ling-evo/analysis/ABSTRACT_CANDIDATE.md`

## What's left for the deadline

| | Owner | By |
|---|---|---|
| Abstract — pick A or B from candidates, iterate | Aron + Mario | Sat May 2 |
| Curate §3 / §4 b_drafts → `manuscript/` (human-firewall step) | Aron | Sun May 3 |
| Run ms-writer for §3 and §4 → c_final | Aron | Mon May 4 |
| Strike the "embedding deferred" sentence in §2.5 | Aron | Mon May 4 |
| Render + rebuild overleaf-upload | Aron / automated | Mon May 4 |
| Submit abstract | Aron | Mon May 4 AoE |
| Iterate on §3 / §4 / §5 reviewer comments | Edward, Alexandra, Arabella | Tue May 5 |
| Render + submit full paper | Aron | Wed May 6 AoE |

## What I'd appreciate help with

- **Mario:** abstract iteration. Two candidates in `ABSTRACT_CANDIDATE.md`
  — Candidate A is 198 words and tighter; B is 232 words with method
  specifics.
- **Edward / Alexandra / Arabella:** when §3 / §4 land in the
  `b_draft.md` files, please drop inline `/comment[…]` annotations —
  Overleaf is up to date as of Apr 30 evening if you'd rather edit
  there.
- **Ivar:** a Gemini re-run under the corrected noise pipeline would
  let §3.1 / §3.2 carry the original three-model contrast. Out of
  scope for this submission but useful for the EMNLP follow-up.

## Reproducibility

All cell-level numbers in the analysis bundle regenerate
deterministically from the `data/json/noise_experiments/v4_direct_provider/`
runs via `analysis/build_*.py` scripts (RNG seed 42 throughout). If you
want to verify a specific number, the corresponding CSV in
`analysis/cell_summaries/` has the per-cell raw values; the script
that generated it sits one level up.
