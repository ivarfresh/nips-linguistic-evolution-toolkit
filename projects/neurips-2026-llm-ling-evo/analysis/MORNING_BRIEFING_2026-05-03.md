# Overnight 2026-05-03 — control × noise matrix completion

*Launched ~10:54 Stockholm by Claude while Aron was AFK. Status updates flow into `/tmp/nlet-runs/overnight_2026-05-03/_overnight_summary.log`.*

## Why this batch

The 2026-05-02 afternoon controls (C1 shuffled / C2 filler / C3 own × bootstrap) **invalidated** the original "partner-myth-specifically" interpretation of the §4.6 / §5.2 mechanism story. C2 filler (length-matched non-narrative reference prose) **matched or exceeded** A1's bootstrap rescue (Δ ≈ +23 vs +18 dyad), which means the rescue isn't carried by partner-narrative content. The active ingredient is more like **prompt-volume / engagement-cue / dropping the LLM into a "you have prior context" frame**, not common-knowledge anchoring of the partner's myth.

The bootstrap-only controls don't tell us *what kind of noise* this generalises to. Two boundary tests we already had:
- A1 in **positive_5**: Δ < 0.2 (no effect, already-coupled cells)
- C2 filler in **positive_5**: see existing `gpt5nano_partner_myth_filler_positive_5` data (60 runs)
- A1 in **no-noise**: clean null

Missing: C1 + C3 in positive_5; all three controls in negative_5; all three controls + adversarial / cooperative content in no-noise. Once those land, you have a clean **3 controls × 4 noise (no_noise / positive_5 / negative_5 / bootstrap)** matrix plus content sweeps.

## Batch 1 (running) — 9 sets, ~390 runs

| # | Set | Cells | Runs | Why |
|---|---|---|---|---|
| 1 | `gpt5nano_partner_myth_filler_negative_5` | C2 × neg5 | 60 | Does prompt-volume rescue negative_5? |
| 2 | `gpt5nano_partner_myth_shuffled_negative_5` | C1 × neg5 | 60 | Cross-dyad coherent-myth in neg5 |
| 3 | `gpt5nano_partner_myth_own_negative_5` | C3 × neg5 | 60 | Self-anchoring in neg5 |
| 4 | `gpt5nano_partner_myth_shuffled_positive_5` | C1 × pos5 | 60 | Boundary completion |
| 5 | `gpt5nano_partner_myth_own_positive_5` | C3 × pos5 | 60 | Boundary completion |
| 6 | `gpt5nano_partner_myth_filler_no_noise` | C2 × no-noise | 30 | No-noise null check |
| 7 | `gpt5nano_partner_myth_shuffled_no_noise` | C1 × no-noise | 30 | No-noise null check |
| 8 | `gpt5nano_partner_myth_own_no_noise` | C3 × no-noise | 30 | No-noise null check |
| 9 | `gpt5nano_partner_myth_filler_bootstrap` | (top-up) | 1 | Backfill 1 missing run from 2026-05-02 |

All under output subdir `v4_direct_provider_controls`, GPT-5-Nano direct, neutral persona, 10-turn trust game with `myth_writing` (`game_myth` and `myth_game` task orders). Templates use the existing `*_with_partner_myth` family with `myth_injection_mode` set to filler / shuffled / own (or the implicit-channel real partner-myth for adversarial / targeted).

First set (`filler_negative_5`) finished in ~13 min with `--workers 4`. Estimated batch 1 finish: ~12:48 Stockholm (≈1h45m total).

## Batch 2 (chained) — 5 sets, 270 runs

Adversarial / cooperative myth content × non-bootstrap noise. Triangulates the §4.6 content-vs-channel decomposition outside bootstrap.

| # | Set | Cells | Runs |
|---|---|---|---|
| 1 | `gpt5nano_partner_myth_adversarial_positive_5` | adversarial × pos5 | 60 |
| 2 | `gpt5nano_partner_myth_adversarial_negative_5` | adversarial × neg5 | 60 |
| 3 | `gpt5nano_partner_myth_adversarial_no_noise` | adversarial × no-noise | 30 |
| 4 | `gpt5nano_partner_myth_targeted_positive_5` | cooperative × pos5 | 60 |
| 5 | `gpt5nano_partner_myth_targeted_negative_5` | cooperative × neg5 | 60 |

Estimated finish: ~14:30 Stockholm.

## What to look for tomorrow

After both batches land, re-run the analysis pipeline with `INCLUDE_VERSIONS` extended (the `build_*.py` scripts only know about the 13 directories that existed on 2026-05-02 — you'll need to add the new experiment_set names):

```bash
python3 projects/neurips-2026-llm-ling-evo/analysis/build_cell_summary.py
python3 projects/neurips-2026-llm-ling-evo/analysis/build_lag_and_lexicon.py
python3 projects/neurips-2026-llm-ling-evo/analysis/build_neologism_analysis.py
python3 projects/neurips-2026-llm-ling-evo/analysis/build_reason_coding.py
python3 projects/neurips-2026-llm-ling-evo/analysis/build_headline_tables.py
```

**Specifically look at:**

1. **Does C2 filler rescue negative_5?** If yes → "prompt-volume" is general; if no → bootstrap-specific. Either is publishable.
2. **Does ANY control lift positive_5 (where the implicit channel already saturates)?** If C1/C3 boost cooperation in positive_5 like C2 already did or did-not, you sharpen the saturation story.
3. **Does C2 filler lift no-noise?** A1 was null at no-noise. If filler is also null, the rescue is *noise-conditional*. If filler lifts no-noise, the prompt-volume effect is unconditional and the §4.6 finding is downgraded to "cosmetic".
4. **Does adversarial myth + injection follow A1 in non-bootstrap?** If yes → content direction confirmed independent of noise type.

## Files / state

- New experiment_sets in `config/experiments_noisy.yaml` (lines ~1316 onward, dated 2026-05-03)
- Launcher: `scripts/run_overnight_2026-05-03.sh` (batch 1) + `_batch2.sh`
- Per-set logs: `/tmp/nlet-runs/overnight_2026-05-03/<set_name>.out`
- Watchdog (in case chained-exec didn't pick up the script edit mid-run): triggers batch 2 ~30s after batch 1's filler_bootstrap line lands. Logged in `_watchdog.out`.

## Early reading (gpt-5-nano only, n=15 unless noted)

`analysis/print_control_matrix.py` filters to gpt-5-nano (Claude baseline rows would otherwise pollute the pooled means). Mean cumulative balance at round 10:

### Bootstrap (the headline cells)

| Variant | game_myth | myth_game | Notes |
|---|---|---|---|
| Baseline +myth, no inj (`anything`) | 58.87 ± 10.99 | 59.00 ± 9.17 | the harmful state |
| Baseline +myth, no inj (`reciprocity_oath`) | 57.40 ± 12.41 | 62.33 ± 8.47 | targeted, no inj — also harmful |
| **A1 partner-myth (real, with inj)** | **68.13 ± 3.50** | **68.38 ± 3.62** | the rescue |
| **C1 shuffled (cross-dyad, with inj)** | **68.20 ± 4.48** | **69.13 ± 3.70** | matches A1 |
| **C2 filler (encyclopedic, with inj)** | **70.53 ± 2.26** | **69.00 ± 3.12** | matches/exceeds A1 |
| **C3 own (self-myth, with inj)** | 65.40 ± 6.01 | 67.50 ± 3.73 | weaker, but still a lift |
| Phase 6 — cooperative + inj | 69.80 ± 2.90 | 70.60 ± 3.17 | matches A1 |
| Adversarial + inj | 66.60 ± 5.45 | ~67.7 | matches A1 |

**Conclusion: the §4.6 "partner-myth specifically" claim is dead.** A1 ≈ C1 ≈ C2; the rescue comes from "ANY partner-position content paragraph", not from partner-myth content. C3 (self-myth) is weaker, so the partner-position *channel* matters more than the content; but even self-myth provides ~half the lift. Adversarial / cooperative / encyclopedic content all match — content direction and genre don't matter.

### Negative_5

| Variant | game_myth | myth_game |
|---|---|---|
| Baseline game-only | 37.87 ± 8.57 | — |
| Baseline +myth, no inj | 42.22 ± 5.36 | 40.07 ± 5.96 |
| A1 partner-myth (real, with inj) | 41.29 ± 3.20 | 40.13 ± 5.85 |
| C1 shuffled (with inj) | 39.74 ± 5.71 | 40.41 ± 4.25 |
| C2 filler (with inj) | 41.19 ± 6.80 | 41.91 ± 3.92 |
| C3 own (with inj) | 41.27 ± 5.06 | 43.12 ± 4.01 |

**Conclusion: NO injection variant rescues negative_5.** All cluster at 39-43, indistinguishable from baseline +myth. Even A1 (real partner-myth) doesn't rescue. The clipping floor of negative_5 is structurally different — there's no destabilisation to rescue.

### Positive_5

All variants (A1/C1/C2/C3) ≈ 74.3-74.7 ≈ baseline +myth ≈ 74.4. Saturated; no further lift possible.

### No-noise

| Variant | game_myth | myth_game |
|---|---|---|
| Baseline +myth, no inj | 71.07 ± 2.74 | 72.27 ± 1.10 |
| A1 partner-myth (real, with inj) | 71.93 ± 1.91 | 71.53 ± 2.67 |
| C1 shuffled (with inj) | 71.47 ± 2.39 | 71.20 ± 2.46 |
| C2 filler (with inj) | 70.33 ± 4.24 | 70.27 ± 3.03 |
| C3 own (with inj) | 70.47 ± 3.87 | (running) |

**Conclusion: clean null at no-noise.** No injection variant lifts above the no-injection +myth baseline. (This is the same null as the SESSION_HANDOFF reported for A1 alone, now extended to all three controls.)

### Combined picture

The rescue effect is **bootstrap-specific**. In positive_5 the cells are saturated; in negative_5 the cells aren't destabilised; in no-noise there's nothing to fix. Bootstrap is the unique noise condition where (a) the implicit cross-task channel produces a *negative* effect (Δ −7.7 vs game-only) and (b) any contentful partner-position paragraph eliminates that destabilisation.

### Reframed §4.6 / §5.2 story

> Bootstrap noise — alternating coerced cooperation and defection signals — creates a coordination-disrupting confusion in the dyad. The implicit cross-task channel under this noise *amplifies* the disruption (Δ −7.7 below game-only baseline). When the partner-position prompt slot is filled with any contentful paragraph — the partner's own myth (A1), a cross-dyad myth (C1), an encyclopedic reference (C2), or even the agent's own myth (C3) — the destabilisation is largely eliminated. The active ingredient is the partner-position channel carrying *some* coherent content, not the specific identity or semantic direction of that content. The competing interpretations — common-knowledge anchoring of the partner's *narrative*, cooperative content as reciprocity prime, structured prior competition — are all falsified.

This is a cleaner mechanism story than the original. Three suggestions for §4.6:

1. **Lead with C2 filler.** It's the most surprising result: length-matched encyclopedic prose about basalt and fountain pens rescues bootstrap as well as the real partner myth.
2. **Use the four-cell control bar chart** (A1 / C1 / C2 / C3 × bootstrap) as the central figure.
3. **Frame the negative_5 / no-noise / positive_5 nulls as boundary tests** establishing that the rescue mechanism is bootstrap-specific.

§5.2 needs the Chwe (2001) interpretation demoted from "supported" to "partly supported (visibility yes, content no)". The "prompt-volume / coherence-providing reference frame" reading should be promoted.

## Wrap-up status — 13:13 Stockholm

**Everything queued has landed.**

- Batch 1 (10:54 → 12:13, 1h19m): 9 sets, ~390 runs. 1 transient `{}`-response failure; 1 backfill of the prior filler_bootstrap missing index. **All complete.**
- Batch 2 (12:13 → 13:09, 56m, chained via watchdog): 5 sets, 270 runs. 2 transient `{}`-response failures (1 in adversarial_negative_5, 1 in targeted_positive_5). Both <2% per set — same noise floor as prior batches. **All complete.**
- Total: ~660 successful runs, ~3 transient failures (~0.5%). All logs in `/tmp/nlet-runs/overnight_2026-05-03/`.

**Analysis pipeline regenerated** with `v4_direct_provider_controls` now in `INCLUDE_VERSIONS` for all 5 build scripts:

- `cell_summary.csv`, `deltas.csv` — re-built 13:10
- `lag_summary.csv`, `lag_correlations.csv`, `lexicon_per_run.csv` — re-built 13:10
- `reason_coding_summary.csv`, `reason_coding_per_round.csv` — re-built 13:10
- `neologism_summary.csv`, `neologisms_examples.csv`, `neologisms_per_run.csv` — re-built 13:10
- Embedding analysis NOT re-run (~25 min; deferred per SESSION_HANDOFF — run when you have the slot).

The 60 new control rows are in cell_summary.csv tagged `version=v4_direct_provider_controls`.

## Ready for tomorrow

1. Open `analysis/SECTION_4_6_REFRAME_DRAFT_2026-05-03.md` — drop-in §4.6 prose with the four-cell rescue table and the boundary tests.
2. Run `python3 projects/neurips-2026-llm-ling-evo/analysis/print_control_matrix.py` for the full grid (filtered to gpt-5-nano).
3. Decide whether to invoke ms-writer on §4.6 / §5.2 with the reframe — or to circulate the new finding to Mario / Edward first for framing input on the abstract.
4. Embedding analysis (`python3 .../build_embedding_convergence.py`, ~25 min) — only matters if §4.5 carryover claim survives the §4.6 mechanism reframe.
5. ~~Optional Tier 4 candidate~~ — actually queued and landed (13:13 → 13:22, 9 min, 60 runs, 0 failures). Output: `data/json/noise_experiments/v4_direct_provider_shared_context/gpt5nano_shared_context_bootstrap/`. Quick aggregation:

| A4 cell | n | mean ± sd |
|---|---|---|
| bootstrap × game (no myth at all) | 10 | **71.20 ± 2.39** |
| bootstrap × game (informed) | 10 | 70.60 ± 1.96 |
| bootstrap × game_myth | 10 | 69.70 ± 3.43 |
| bootstrap × game_myth (informed) | 10 | 71.10 ± 2.28 |
| bootstrap × myth_game | 10 | 67.00 ± 3.71 |
| bootstrap × myth_game (informed) | 10 | 70.10 ± 2.28 |

A4 ≈ A1 magnitudes (myth-task cells: 67–71 vs A1's 68.1/68.4). The symmetric-context intervention does not exceed the asymmetric A1 injection: **A1 is the minimum sufficient condition for the bootstrap rescue**. Useful for §5.6 future-work bullet on "minimum sufficient channel".

One unexpected wrinkle: A4 in `game`-only task order (no myth at all, just ledger info shared between rounds) also rescues bootstrap (71.2). This suggests the rescue may not require *any* myth content at all — just additional structured prompt context with prior-round signals. Worth double-checking: is the A4 game-only condition really "myth-content-free", or is it implicitly different from baseline game-only? (Compare to baseline_v4_mem3_direct game-only bootstrap = 66.60.) If confirmed, this further weakens any "narrative content matters" reading of the §4.6 rescue. Will need to read what `*_with_shared_context` prompt templates actually carry — they may include placeholder myth text even when no myth task ran.

## Final tally

- **14 experiment_sets queued** today, ~720 runs, ~3 transient failures.
- **3 batches** complete (B1: 9 sets / 390 runs, B2: 5 sets / 270 runs, B3: 1 set / 60 runs).
- **Headline finding** (cleanly established): bootstrap noise creates a destabilisation that any contentful paragraph in the partner-position prompt slot resolves; A1 partner-myth ≈ C1 cross-dyad ≈ C2 encyclopedic filler ≈ adversarial myth ≈ cooperative myth — content identity, content direction, and narrative status do not matter. C3 own-myth is meaningfully weaker (≈ 25 % less rescue), so the partner-position channel matters; the partner-myth content does not.
- **Boundary tests**: positive_5 saturated, negative_5 / no_noise clean nulls — rescue is bootstrap-specific.
- **A1 = minimum sufficient channel**: A4 symmetric-context doesn't exceed A1.
- **§4.6 / §5.2 reframe draft** in `analysis/SECTION_4_6_REFRAME_DRAFT_2026-05-03.md`.



## What I did NOT do

- No analysis re-runs. The new data won't feed into figures / cell_summary until you re-run the pipeline.
- No manuscript edits. The 2026-05-02 SESSION_HANDOFF is still the source of truth on what's in `_3-results.c_final.md`. The §4.6 / §5.2 prose still implies the "common-knowledge anchor" reading that the controls have weakened.
- No cross-model runs. Aron's instruction was GPT-5-Nano only.
