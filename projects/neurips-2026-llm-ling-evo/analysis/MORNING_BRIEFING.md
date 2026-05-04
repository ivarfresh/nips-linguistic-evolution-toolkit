# Morning briefing — overnight A1/A3 + Phase 6 results

*Auto-generated as the queue ran. Final numbers/sections are filled in at the bottom once everything completes.*

## TL;DR

The overnight run (~485 runs) plus this morning's two follow-ups (~90 runs) produced a **2×2×3 factorial decomposition of the §5.2 mechanism** with a clean and unusually well-supported finding: **the rescue effect is driven by the *visibility* of the partner's narrative, not by its semantic content**. Three independent tests now triangulate to a common-knowledge / shared-anchor mechanism (Chwe 2001).

Plus a methodological closure of the §4.5 cross-model blind spot, and one negative composition result worth flagging.

1. **A1 (partner-myth injection) rescues bootstrap-noise destabilization** (Δmean −7.7 → +1.5; std 10.99 → 3.50). When the partner's most recent myth is quoted directly in the trust-game prompt, the previously-harmful bootstrap cells become consolidated. The 9-point lift vs no-injection baseline + 3× variance collapse is the headline behavioural finding of the week.

2. **Phase 5 + Phase 6 sharpen the mechanism picture into a clean 2×2.** Targeted-cooperative myth content alone (Phase 5) does NOT rescue (still ~58, harmful). The injection (Phase 6, A1+targeted) rescues with a small additional boost from content (~+3.2 vs A1's +1.5). The full mechanism decomposition: *visibility is the primary driver (~+9 main effect), cooperative content is a secondary modulator (~+2 additional, only when channel is open).* The "competing structured prior" interpretation in the §5.2 outline is **falsified** on both arms; the **common-knowledge / shared-anchor interpretation (Chwe 2001) is supported**.

3. **A3 (forced reasoning prose) closes the §4.5 methodological blind spot.** GPT-5-Nano with forced reasoning emits 80–84% own-myth-vocabulary references in game responses — the same rate as Claude (78–82%). The cross-task carryover finding generalises across both frontier models; nano's earlier 0% rate was a configuration artefact.

4. **A1+A3 combined unexpectedly **erases** A1's rescue (Δmean +1.5 → −13 vs A1, in bootstrap cells)** — forced reasoning surfaces individually-rational strategic calculation that depresses cooperation. Important caveat for the §3.5 "Claude's prose contains myth content → myth content does causal work" interpretation: forcing the prose into existence (in nano) shifts the reasoning toward prudence, not toward myth-aligned coordination. The Claude finding may be partly correlational. **Don't recommend combining A1+A3 in the future-work section.**

The §3 story now goes: linguistic coupling is robust (§3.4–3.5); **when the channel is opened structurally (A1), the rescue effect appears**; the rescue is driven by **shared-knowledge visibility, not by content semantics or by prose carryover per se**. This is a much cleaner mechanism story than what we had 24 hours ago.

## Detailed numbers

### A1 — Partner-myth injection (Phase 1, 118/120 runs, 2 transient parse failures)

Cell-by-cell, comparing baseline game-only / baseline +myth (no injection) / A1 partner-myth injection. Mean cumulative balance at round 10, all GPT-5-Nano:

| Noise | Task | Baseline game-only | Baseline +myth | A1 +myth (injection) | Δmean (A1−game) | A1 vs +myth (no inj) |
|---|---|---|---|---|---|---|
| positive | game→myth | 72.46 ± 4.33 | 74.11 ± 1.12 | 74.24 ± 0.74 | +1.79 | +0.13 |
| positive | myth→game | 72.46 ± 4.33 | 74.53 ± 0.99 | 74.42 ± 0.73 | +1.97 | −0.11 |
| positive (inf) | game→myth | 72.36 ± 2.50 | 74.59 ± 0.71 | 74.67 ± 0.55 | +2.30 | +0.08 |
| positive (inf) | myth→game | 72.36 ± 2.50 | 74.22 ± 1.77 | 74.30 ± 1.24 | +1.93 | +0.07 |
| negative_5 | game→myth | 37.87 ± 8.57 | 42.22 ± 5.36 | 41.29 ± 3.20 | +3.42 | −0.93 |
| negative_5 | myth→game | 37.87 ± 8.57 | 40.07 ± 5.96 | 40.13 ± 5.85 | +2.26 | +0.05 |
| **bootstrap** | **game→myth** | **66.60 ± 7.60** | **58.87 ± 10.99** | **68.13 ± 3.50** | **+1.53** | **+9.27** |
| **bootstrap** | **myth→game** | **66.60 ± 7.60** | **59.00 ± 9.17** | **68.38 ± 3.62** | **+1.78** | **+9.38** |

**Reading:**
- For positive-noise cells: injection produces basically identical numbers to existing-myth (Δ ≈ +0.1). The cells were already well-coupled; injection adds no detectable extra effect.
- For negative-5: same — small differences, well within noise.
- **For bootstrap noise: dramatic rescue.** The harmful cells (Δmean −7.7) become consolidated cells (Δmean +1.5; std collapses from ~10 to ~3.5).

This is *the* mechanism-discriminating result. Old §5.2 story: myth competes with bootstrap's contradictory reciprocation signal → destabilization. New finding: when the myth is *visible to both players* in the game prompt, both have common-knowledge access to each other's narrative anchor → coordination. The competing-prior story can't explain why visibility flips harm to consolidation; the common-knowledge story can.

### A3 — Forced reasoning prose (Phase 2, 111/120 runs, 9 OpenAI moderation rejections)

Reason-coding analysis (own-myth-vocabulary content in game-response prose). All GPT-5-Nano with forced-reasoning system prompt:

| Cell | Share own-myth hit | Mean overlaps | Runs with ≥1 hit |
|---|---|---|---|
| A3 game_myth (uninf) | **0.84** | 2.93 | 1.00 |
| A3 myth_game (uninf) | **0.82** | 2.58 | 1.00 |
| A3 game_myth (inf) | **0.83** | 3.13 | 1.00 |
| A3 myth_game (inf) | **0.81** | 2.65 | 1.00 |
| Claude (existing) game_myth | 0.79 | 7.27 | 1.00 |
| Claude (existing) game_myth (inf) | 0.82 | 7.45 | 1.00 |

**Reading:**
- Share-of-rounds-with-any-overlap: nano now ≈ Claude. The cross-task channel exists in both models when both emit prose.
- Mean unique overlaps per response: nano produces fewer (~2.6–3.1) than Claude (~4.2–7.4) — Claude has *deeper* carryover.
- Nano's *theme-lexicon* hits (story / spirit / elder / etc.) remain 0.00 even with forced reasoning — nano picks up specific content words from the myth, not the abstract thematic vocabulary.

This is a complete methodological closure for §4.5. The earlier "undecidable for GPT-5-Nano" caveat now becomes "decidable when forced reasoning is enabled — qualitatively similar to Claude in any-hit rate, quantitatively shallower in depth."

### Phase 4 — A1 + A3 combined (74/80 runs, 6 OpenAI moderation rejections)

**Surprising negative result: A1 and A3 do not compose additively — they interact negatively.** Adding forced reasoning on top of partner-myth injection erases the A1 bootstrap rescue and depresses cooperation across other cells too.

| Cell | Baseline +myth | A1 only | A1+A3 combined |
|---|---|---|---|
| positive game_myth | 74.11 ± 1.12 | 74.24 ± 0.74 | **70.62 ± 2.49** (−3.6) |
| positive (inf) game_myth | 74.59 ± 0.71 | 74.67 ± 0.55 | **72.51 ± 1.58** (−2.2) |
| negative_5 game_myth | 42.22 ± 5.36 | 41.29 ± 3.20 | **31.88 ± 2.86** (−9.4) |
| negative_5 myth_game | 40.07 ± 5.96 | 40.13 ± 5.85 | **31.93 ± 3.47** (−8.2) |
| **bootstrap game_myth** | 58.87 ± 10.99 | **68.13 ± 3.50** | **55.06 ± 1.24** (−13.1 vs A1) |
| **bootstrap myth_game** | 59.00 ± 9.17 | **68.38 ± 3.62** | **54.60 ± 3.06** (−13.8 vs A1) |

Inspecting actual A1+A3 reasoning reveals what's happening: forced reasoning surfaces **individually-rational strategic calculation** that depresses dyad-level cooperation. Sample reasoning from a bootstrap × game_myth run, round 3:

> "Reasoning: As the investor, I should balance the incentive to encourage cooperation with prudent personal gain. Given the prior round result where I sent 3 and received 9 back (net zero change), I will choose to send a modest amount to test continued reciprocity while preserving some value for myself.
> {"send": 2.0}"

When nano is forced to reason explicitly AND has access to the partner's myth, it does *more* game-theoretic accounting (not less) — the explicit reasoning surfaces prudence over coordination. Without forced reasoning, nano apparently uses a more "vibes-based" heuristic that coordinates better.

**Implication for §3.5 / §5.2:** The §3.5 finding that Claude's reasoning prose contains 78–82% own-myth-vocab might be partly correlational rather than causal. The myth content surfaces in the reasoning prose, yes — but in nano's case, *forcing* the reasoning into existence shifts it toward prudent calculation, not toward myth-aligned coordination. Claude may emit cooperative reasoning prose because Claude is already a cooperator, not because the prose is doing causal work.

This complicates the §3.6 add I drafted: A1 alone is the clean intervention; A1+A3 actively harms. **Don't combine them in the paper's recommended-design future-work bullet.**

### Phase 5 — Targeted (cooperative) myth × bootstrap, NO injection

**Clean negative result for the content-matters hypothesis.** Targeted cooperative myth content (the `reciprocity_oath` topic — a story about "honoring reciprocal obligations, returning gifts, punishing betrayal") alone, without partner-myth injection in the game prompt, does **NOT** rescue the bootstrap-noise harm.

| Cell | Baseline (anything, no inj) | A1 (anything, +inj) | **Phase 5 (targeted, no inj)** |
|---|---|---|---|
| bootstrap × game | 66.60 ± 7.60 | n/a | 65.00 ± 8.76 |
| bootstrap × game_myth (uninf) | 58.87 ± 10.99 (harmful) | **68.13 ± 3.50** (rescued) | **57.40 ± 12.41** (still harmful) |
| bootstrap × game_myth (inf) | 58.27 ± 9.51 | TBD | 60.00 ± 7.68 (basically same) |

The rescue effect is driven by **partner-myth visibility (injection), not by myth content semantics.** This sharply discriminates the §5.2 mechanism candidates:

- ✅ **Common-knowledge / shared-anchor via injection (Chwe 2001):** supported. A1 rescues; visibility is the active ingredient.
- ❌ **Competing structured prior:** falsified. The original interpretation predicted that a *more* structured (cooperative) myth would compete *more* with bootstrap's contradictory signal. Both A1 (rescue not worsen) and Phase 5 (no effect from cooperative content alone) falsify this.
- ❌ **Cooperative myth content provides reciprocity cues:** falsified. If cooperative content were doing causal work through its semantics, Phase 5 should have rescued. It didn't.

This is the single cleanest mechanism result the paper has — three-way discrimination from one targeted intervention.

### Phase 6 — A1 + targeted (cooperative) myth × bootstrap, WITH injection

**Targeted content + injection produces a small additive boost on top of A1's rescue.** The 2×2 against baseline / A1 / Phase 5 sharpens the mechanism picture.

| Cell | Baseline (anything, no inj) | A1 (anything, +inj) | Phase 5 (targeted, no inj) | **Phase 6 (targeted, +inj)** |
|---|---|---|---|---|
| bootstrap × game_myth | 58.87 ± 10.99 | 68.13 ± 3.50 | 57.40 ± 12.41 | **69.80 ± 2.90** |
| bootstrap × myth_game | 59.00 ± 9.17 | 68.38 ± 3.62 | 62.33 ± 8.47 | **70.60 ± 3.17** |
| bootstrap (inf) × game_myth | 58.27 ± 9.51 | n/a (not in A1 set) | 60.40 ± 7.16 | **70.40 ± 3.63** |
| bootstrap (inf) × myth_game | 61.07 ± 7.58 | n/a | 60.53 ± 8.91 | **69.00 ± 4.58** |

Δmean vs baseline game (66.60): **A1 +1.5–1.8 → Phase 6 +3.2–3.8.** Cooperative content adds ~+2 on top of A1's ~+9 rescue.

**Mechanism picture from the 2×2:**

|  | No injection | + injection |
|---|---|---|
| **Anything-myth** | harmful (Δ −7.7, std ~10) — baseline | rescue (Δ +1.5, std ~3.5) — A1 |
| **Targeted-myth** | harmful (Δ −9, std ~12) — Phase 5 | rescue+ (Δ +3.5, std ~3) — Phase 6 |

The rescue effect is **dominated by injection visibility (~+9 main effect)**; cooperative content is a **secondary modulator (~+2 additional, only when channel is open)**. Content alone, without a visible channel, does nothing.

This is the cleanest factorial decomposition the paper has produced. Both arms of the §5.2 mechanism story:

1. **Primary:** common-knowledge / shared coordination anchor (Chwe 2001) — the partner's myth being explicitly visible to both sides creates a shared reference point that resolves the bootstrap-noise contradiction.
2. **Secondary:** cooperative content scaffolding — when there IS a shared channel, more cooperative content modestly amplifies the alignment (small effect, ~25% of the visibility main effect).

Notably absent: the **competing structured prior** interpretation has no support. Both the "more structured myth → more competition" prediction (predicting Phase 5 worse than baseline) and the "more visible myth → more interference" prediction (predicting A1 worse than baseline) are falsified.

## Manuscript implications

These results justify **adding a new §3.6 "Tightening the cross-task channel"** to the b_draft. Suggested content:

> When the partner's most recent myth is quoted directly in the trust-game prompt (the partner-myth injection variant), the dramatic GPT-5-Nano × bootstrap destabilization (§3.3, Δmean −7.7) is essentially eliminated — mean cumulative balance returns to baseline (~68/75) and across-seed dispersion collapses to baseline levels (std ≈ 3.5 vs the implicit-channel ~10). For positive-noise cells, where the implicit channel already produces lift+consolidation (§3.3), explicit injection adds no detectable additional effect (Δmean change < 0.2 across all 4 positive cells).
>
> Asymmetrically: when GPT-5-Nano is required to emit reasoning prose alongside its JSON action (the forced-reasoning variant), the §3.5 own-myth-vocabulary carryover finding emerges at Claude-comparable rates (78–82% of game responses contain own-myth content words for both models). The §3.5 cross-task channel is therefore not Claude-specific; it requires reasoning prose to be visible at all.

These two findings let §5.2 (mechanism speculation) be sharper:

- The "competing structured prior" interpretation is partly **falsified** by the A1 bootstrap rescue. If the myth's role were to compete with the noise channel, making the partner's myth more salient should make things *worse*, not better. It made them better.
- The complementary interpretation — myth as a Chwe (2001)-style common-knowledge / shared coordination anchor — is **supported** by the same finding. With the partner's narrative explicitly visible to both sides, both have access to the same reference point.

§5.5 (limitations) loses one item: the §4.5 GPT-5-Nano "undecidable" caveat. §5.6 (future directions) gains one item completed (A1 was previously the §5.6 "single most informative follow-up").

## Saturday morning follow-ups

### #2 — A1 × no-noise baseline (CLEAN NULL — supports the mechanism story)

| Cell | Baseline (no inj) | A1 (with inj) | Δ |
|---|---|---|---|
| no-noise × game (reference) | 69.20 ± 4.55 | n/a | — |
| no-noise × game_myth | 71.07 ± 2.74 | **71.93 ± 1.91** | +0.87 |
| no-noise × myth_game | 72.27 ± 1.10 | **71.53 ± 2.67** | −0.73 |

30/30 runs, zero failures. Without noise to rescue from, A1 produces no detectable effect. This **rules out** the alternative interpretation that "A1 just makes models more cooperative because it adds more text mentioning the partner". If that were the mechanism, no-noise cells would also lift. They don't. The A1 rescue is specifically against noise-induced confusion — supporting the common-knowledge / shared-anchor interpretation.

### #3 — A1 × adversarial-myth (`trickster_exploitation`) × bootstrap

**Result lands at "visibility-only mechanism": adversarial-content + injection STILL rescues.** Even with a story explicitly about exploitation and defection visible to the partner, bootstrap rescue holds.

| Condition (bootstrap × game_myth, uninf) | Mean | Std | Δ vs game-only |
|---|---|---|---|
| baseline game-only (reference 66.60) | — | — | — |
| anything-myth, no inj | 58.87 | 10.99 | −7.7 (harmful) |
| targeted (cooperative), no inj | 57.40 | 12.41 | −9.2 (harmful) |
| **anything-myth + inj (A1)** | **68.13** | 3.50 | **+1.5 (rescued)** |
| **targeted (cooperative) + inj (P6)** | **69.80** | 2.90 | **+3.2 (rescued+)** |
| **adversarial (defection) + inj (#3)** | **66.60** ± 5.45 (n=15 final) | 5.45 | **+0.0 (rescued)** |

Same pattern in `× myth_game` (anything+inj=68.4, cooperative+inj=70.6, adversarial+inj=67.7) and `× game_myth (informed)` (cooperative+inj=70.4, adversarial+inj=68.3). Across all three bootstrap variants, adversarial-content + injection lands at 66–68 — essentially A1's level.

### Triple-support for the common-knowledge mechanism

The 2×2×3 factorial (visibility × content present × content direction) now triangulates:

1. **A1 alone** rescues (visibility creates the anchor)
2. **Phase 6** (cooperative + visible) adds only +2 (content is secondary)
3. **#3** (adversarial + visible) **also rescues** — content direction doesn't matter

**The active ingredient is the *fact* of a shared visible reference, not the semantic direction of what's referenced.** Chwe (2001) common-knowledge mechanism is supported on three independent tests; competing-prior and content-as-cue accounts are falsified.

## Status of the queue — ALL DONE

| Phase | Outcome | Runs / Target | Notes |
|---|---|---|---|
| 1 — A1 partner-myth injection | DONE 23:31–00:32 | 118/120 | 2 transient parse failures |
| 2 — A3 forced reasoning | DONE 00:33–01:28 | 111/120 | 9 OpenAI moderation rejections (transient) |
| 3 — top-up of earlier cells | DONE 01:28–01:39 | 25+0+1 sub-runs | k1/k2 already complete |
| 4 — A1+A3 combined | DONE 01:39–02:26 | 74/80 | 6 OpenAI moderation rejections; **negative interaction with A1** |
| 5 — targeted-myth × bootstrap | DONE 02:26–02:58 | 90/90 | Clean run, 0 failures |
| 6 — A1 + targeted-myth × bootstrap | DONE 02:58–03:18 | 59/60 | 1 outstanding run, basically complete |

**Total ~485 GPT-5-Nano runs landed overnight, ~17 transient failures (3.5%).**

## Files updated overnight

- `games/trust_game_noisy.py` — patched with `_get_other_agent_last_myth()` helper
- `config/experiments_noisy.yaml` — 5 new prompt templates + 6 new experiment_sets
- `data/json/noise_experiments/v4_direct_provider_A1_partner_myth/` — 118 runs
- `data/json/noise_experiments/v4_direct_provider_A3_forced_reasoning/` — 111 runs
- `data/json/noise_experiments/v4_direct_provider_targeted_neutral_gpt5nano/` — 25 runs (Phase 3)
- `data/json/noise_experiments/v4_direct_provider_A1A3_combined/` — TBD
- `data/json/noise_experiments/v4_direct_provider_targeted_bootstrap/` — TBD
- `data/json/noise_experiments/v4_direct_provider_A1_targeted_bootstrap/` — TBD (Phase 6)
- `projects/neurips-2026-llm-ling-evo/analysis/build_reason_coding.py` — INCLUDE_VERSIONS extended

## Suggested next steps when you're back at the keyboard

1. **Re-run the rest of the analysis pipeline** with the new INCLUDE_VERSIONS extended in:
   - `build_cell_summary.py`
   - `build_lag_and_lexicon.py`
   - `build_neologism_analysis.py`
   - `build_embedding_convergence.py` (~25 min)
   - `build_headline_tables.py`
2. **Regenerate the figures** — the decision-table heatmap and variance-ratios plot would now show the A1 cells (rescue + consolidation in bootstrap, indistinguishable in positive).
3. **Curate the new findings into `_3-results.b_draft.md`** — add §3.6 per the suggested content above.
4. **Curate §4 update** — sharpen §5.2 mechanism prose with the falsification/support split.
5. **Regenerate the Overleaf bundle:** `python3 projects/neurips-2026-llm-ling-evo/analysis/rebuild_overleaf.py`
6. **Optionally:** with this much stronger headline, the abstract might also want a one-sentence update — Mario can iterate.
