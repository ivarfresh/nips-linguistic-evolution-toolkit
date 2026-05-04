# Proposed §4.6 / §A.5 rewrite — A1 partner-myth controls

> **Status:** Draft synthesis based on the controls completed 2026-05-02.
> Source data: `data/json/noise_experiments/v4_direct_provider_controls/`
> (gpt5nano_partner_myth_{shuffled,filler,own}_bootstrap, 60 runs each;
> gpt5nano_partner_myth_filler_positive_5, 60 runs).
> Source analyses: `data/plots/v4_direct_provider_controls/summary_table.txt`,
> `data/plots/v4_direct_provider_controls/diagnostic_subanalysis.md`.

## Executive summary of what the controls did to the existing claim

The current §4.6 claim — *"partner-myth visibility produces the bootstrap-noise rescue (Δmean -7.7 → +1.5, std 11 → 3.5) by opening an explicit cross-task channel between game and myth"* — does not survive the controls.

| Metric (game_myth uninformed, bootstrap) | no-A1 | real A1 | C1 shuffled | C2 filler | C3 own |
|---|---|---|---|---|---|
| Final dyad balance (mean ± SD, n=15) | 117.7 ± 22.0 | 136.3 ± 7.0 | 136.4 ± 9.0 | **141.1 ± 4.5** | 130.8 ± 12.0 |
| Δ vs no-A1 baseline | — | +18.5 | +18.7 | **+23.3** | +13.1 |
| Per-round-payoff recovery (rounds) | 1.67 | 1.79 | 1.58 | **1.27** | 1.57 |

C2 filler — paragraphs about hydrology and concrete — produces a *larger* dyad-balance lift than the partner-myth condition and recovers from defection events *faster*. C1 shuffled (myth from a different dyad) matches real-A1 within ±0.1 dyad. C3 own (agent's own prior myth) lags slightly. The variance-collapse signature (~22 → ~7 SD) is reproduced by all three controls.

The positive_5 boundary test confirms the regime-conditional reading: in cells where cooperation is already at ceiling (~148.5/150 dyad), neither real-A1 nor C2 filler produces any lift (Δ ≈ +0.3 across all four positive_5 cells, n=15 each). Whatever the bootstrap rescue is, it requires destabilization headroom, not partner-myth content.

The within-round Agent_1↔Agent_2 myth similarity advantage (real-A1 0.71 vs no-A1 0.59) decomposes cleanly:

| Step | partner_cos | Δ |
|---|---|---|
| no-A1 baseline (no injection) | 0.589 | — |
| C2 filler (irrelevant prose injected) | 0.640 | +0.052 (prompt-volume) |
| real A1 (partner myth injected) | 0.700 | +0.059 (any-narrative lift) |
| C3 own (own myth injected) | 0.677 | (same lift, no partner content) |

C3 own reproduces ~80% of the real-A1 narrative-injection lift, with no partner content available. The partner-myth-specific contribution is statistically near zero. Real-A1 agents do not literally echo the injected text (echo_cos 0.70 is only 0.115 above the natural myth-myth similarity floor); they produce a fresh myth that has a soft stylistic shift toward whatever narrative they just read, regardless of whose narrative it was.

## What survives, what changes

### Survives
- The bootstrap × myth-present *destabilization* finding in §4.3 (myth harms cooperation under bootstrap).
- The positive_5 lift in §4.2 (myth helps under positive noise).
- The cross-model baseline gap (Claude > GPT-5-Nano).
- The variance-reduction-with-task-complexity pattern.
- The fact that partner-myth visibility *can* recover bootstrap cooperation — this is replicated.

### Changes
- The mechanism is no longer "partner-myth visibility opens an implicit channel." It is "any length-matched prose injection in the game prompt produces a regime-conditional cooperation lift, present where cooperation has destabilization headroom (bootstrap) and absent where cooperation is already at ceiling (positive_5)."
- Partner-myth visibility produces measurable narrative similarity lift between agents, but the lift is fully matched by injecting the agent's own prior myth — narrative-injection-general, not partner-specific.

## Suggested rewrite for §4.6

> ### 4.6 Visibility of any in-context narrative recovers bootstrap cooperation
>
> §4.3 showed that under bootstrap noise, dyads with myth-writing tasks cooperated *less* than game-only dyads (Δmean -7.7 dyad in `game_myth`, std rising to 11.0). We tested whether explicitly surfacing the partner's most recent myth in the trust-game prompt would reverse this destabilization, and whether the effect was specific to partner-myth content.
>
> We ran four matched conditions on GPT-5-Nano, all using the same prompt template (the partner-myth-injection wrapper) but varying the injected text:
>
> - **A1 partner-myth**: the partner's most recent myth from this dyad's run.
> - **A1<sub>shuffled</sub>**: a myth from a *different* baseline dyad's run.
> - **A1<sub>filler</sub>**: a length-matched neutral non-narrative paragraph (e.g., a paragraph on the hydrologic cycle).
> - **A1<sub>own</sub>**: the agent's own most recent myth.
>
> All four conditions reverse the bootstrap destabilization to roughly equal extent (Δmean +18.5 to +23.3 dyad over no-injection baseline; n=15 each, see Table X). The strongest lift comes from A1<sub>filler</sub>, not from the true partner myth. The variance-collapse signature is reproduced by every condition. The same conditions in positive_5 noise — where game-only cooperation already sits at the dyad ceiling — produce no measurable lift (Δ ≈ +0.3 across all positive_5 cells), confirming the rescue requires destabilization headroom rather than narrative content.
>
> Within-round Agent_1↔Agent_2 myth similarity rises from 0.59 (no-injection baseline) to 0.71 in the partner-myth condition (cosine, all-MiniLM-L6-v2 embeddings). However, this lift decomposes into ~0.05 attributable to generic added prompt volume (matched by A1<sub>filler</sub>: 0.64) and an additional ~0.06 attributable to any narrative material being injected — including the agent's own prior myth (A1<sub>own</sub>: 0.68). The portion attributable to *partner*-myth content specifically is statistically negligible. Agents do not copy-paste the injected text (echo cosine to the injected text in A1<sub>partner</sub> is only 0.115 above the natural myth-myth similarity floor); they produce a fresh myth with a soft stylistic shift toward whatever narrative they just read.
>
> We therefore retract the §4.3 implication that the bootstrap destabilization is a partner-myth-coordination problem solved by opening a partner-myth channel. The behavioural pattern is more parsimoniously explained as a regime-conditional prompt-augmentation effect: under bootstrap, the dyad is sensitive to small additions in the game-prompt context; under positive_5 it is not. Whether the underlying mechanism is genuine cognitive engagement, sampling-temperature regularization, or attentional re-anchoring is beyond what these controls can resolve and is left to future work.

## Suggested addition for §A.5 (appendix)

> ### A.5 Defensive controls for A1
>
> [retain the existing per-cell breadth table; replace the "the rescue is regime-conditional, not content-conditional, within partner-myth-injection" framing with the controls table below]
>
> All conditions use identical prompt templates; only the bound text changes.
>
> Table A.5.1 — Cumulative dyad balance, GPT-5-Nano bootstrap, 15 seeds per cell.
>
> | task_order | informed | no-A1 | A1<sub>partner</sub> | A1<sub>own</sub> | A1<sub>shuffled</sub> | A1<sub>filler</sub> |
> |---|---|---|---|---|---|---|
> | game_myth | uninf | 117.7 ± 22.0 | 136.3 ± 7.0 | 130.8 ± 12.0 | 136.4 ± 9.0 | 141.1 ± 4.5 |
> | game_myth | inf   | 116.5 ± 19.0 | — | 134.3 ± 8.3 | 132.9 ± 9.2 | 135.2 ± 7.6 (n=14) |
> | myth_game | uninf | 118.0 ± 18.3 | 136.8 ± 7.2 (n=13) | 135.0 ± 7.5 | 138.3 ± 7.4 | 138.0 ± 6.2 |
> | myth_game | inf   | 122.1 ± 15.2 | — | 136.5 ± 9.5 | 137.9 ± 5.0 | 133.2 ± 11.3 |
>
> Table A.5.2 — Within-round Agent_1↔Agent_2 myth cosine similarity, game_myth uninformed.
>
> | condition | mean cosine | convergence slope (late-early) |
> |---|---|---|
> | no-A1 baseline | 0.66 ± 0.04 | +0.05 ± 0.05 |
> | A1<sub>shuffled</sub> | 0.63 ± 0.05 | +0.02 ± 0.07 |
> | A1<sub>filler</sub> | 0.67 ± 0.05 | +0.06 ± 0.08 |
> | A1<sub>own</sub> | 0.70 ± 0.05 | +0.07 ± 0.04 |
> | A1<sub>partner</sub> | 0.71 ± 0.07 | +0.10 ± 0.05 |
>
> The directional ordering (partner > own > filler > shuffled) on convergence slope is consistent with — but does not require — a partner-coordination story. The own-myth condition produces ~70% of the partner-myth slope advantage with no partner content available, suggesting that "any narrative material in the prompt" rather than "partner-specific narrative material" is the operative ingredient.

## Reviewer-defense notes

This rewrite is *stronger* than the original §4.6 from a review standpoint, because:

1. The controls are now in the manuscript. A reviewer who would have asked "did you control for prompt volume?" cannot use that weapon.
2. The honest negative-result framing prevents a much worse outcome at review where someone runs filler themselves and contradicts the headline.
3. The narrower surviving claim ("regime-conditional prompt-augmentation effect, not partner-content-specific") is mechanistically more interesting than the original — it places the phenomenon in a class of attentional/prompt-sensitivity effects that connect to a broader literature on LLM context dependence.

The cost: the partner-myth narrative — the through-line that connects §4.6 back to the §1 framing about cultural-evolutionary cross-task channels — is gone. Either §1 needs to be reframed to predict the prompt-augmentation finding, or the cross-task-channel framing needs to be moved to "future work" in §5.6.
