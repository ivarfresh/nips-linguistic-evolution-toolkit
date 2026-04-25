### 2026-04-24 — Pre-NeurIPS measurement plan

#### Framing
Main research question: *do myths increase cooperation in multi-agent LLM settings*. Existing TODOs: ensure 15 runs/condition, try better models, add prompt variations. This entry logs additional measurement ideas surfaced in discussion, ranked for the NeurIPS window (internal deadline 2026-04-27).

#### Directions to add
- **Signal vs. noise 2×2**: per (model × noise) cell, tabulate mean-shift × variance-shift — separates useful signal / pure noise / strategy-reinforcement / harmful signal. Reuses existing runs.
- **Reason-field coding**: classify each game `reason` as myth-referencing / game-math / neither. Highest evidence-per-hour; existing data.
- **Content structure of myths**: Moral Foundations tagging (LLM-judge), named-entity persistence curves, motif-cluster mutation-selection under partner rewrite, compression-over-rounds entropy curves (quantifies the prior "concrete → abstract" observation).
- **Causal ablations**: content-scrambled myth (cheapest, pre-submission if compute allows); Jabberwocky structure-vs-semantics variant; forced-valence injected myths; myth-removal after round K; non-narrative cross-task control — all post-NeurIPS except scrambled.
- **Mediation**: ETI-style (Abdurahman et al.) warmth/competence inference per round; test whether myth shifts inferred traits and whether that mediates cooperation-lift.

#### Priority (pre-NeurIPS)
1. Reason-field coding + ETI trait inference on existing myths.
2. 2×2 mean/variance decision table.
3. Ghafouri-Ferrara convergence / selective-survival measures on existing myth corpus.
4. Content-scrambled myth ablation if compute headroom.

#### Papers pulled
- **Robinson & Burden, 2503.04840** — framing sensitivity; template for forced-valence injection and non-narrative control.
- **Ghafouri & Ferrara, 2602.17674** — AI-to-AI telephone chains; convergence + selective survival + competitive filtering measures. Most load-bearing for myth-evolution section.
- **Lupyan & Agüera y Arcas, 2601.11432** — Jabberwocky substitution; sharper content-scramble ablation design.
- **Abdurahman et al., 2604.19278** — Explicit Trait Inference; off-the-shelf mediator for partner-perception story.
