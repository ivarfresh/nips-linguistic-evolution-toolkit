# Abstract candidates — NeurIPS 2026

Two candidates at slightly different word counts (NeurIPS abstracts are
typically capped at 200–250 words, depending on the year's call). Pick
one and iterate; both built from `analysis/HEADLINE_TABLES.md`.

---

## Candidate A — 198 words (tight)

LLM agents increasingly converse with other LLMs while making
consequential decisions, but the natural-language exchange is rarely
the measurement target — the talk is treated as scaffolding around
numeric outcomes. We ask whether inter-agent language is *load-bearing*
on cooperation, or merely accompanies decisions already fixed by model
and prompt. We pair two same-base-model agents in a role-swapping
repeated trust game, optionally interleaved with iterated myth-writing,
and vary action-channel noise to manufacture behavioural headroom.
Across 21 (model × noise × task-order) cells with 15 seeds each, the
myth effect is conditional rather than uniform: under upward-biased
noise, adding myth-writing produces a small mean lift and substantial
variance reduction in three Claude/GPT-5-Nano cells (variance ratios
0.05–0.31); under reciprocity-flipping bootstrap noise, it actively
destabilises GPT-5-Nano cooperation (mean drops 7–10 reward units, four
cells); the remaining cells are null. The linguistic channel is
nevertheless doing measurable work: between-agent embedding cosine
converges over rounds in 18/21 cells, and Claude's game-response prose
references its own myth vocabulary in 78–82% of rounds. Inter-agent
language is real, asymmetric across models, and sign-conditional on the
noise regime — not mere decoration, but not a uniform cooperation
switch either.

---

## Candidate B — 232 words (with method specifics)

LLM agents increasingly converse with other LLMs while making
consequential decisions, but the natural-language exchange is rarely
the measurement target — the talk is treated as scaffolding around
numeric outcomes. We ask whether inter-agent language is *load-bearing*
on cooperation, or merely accompanies decisions already fixed by model
and prompt. We introduce a controlled dyadic paradigm pairing two
same-base-model agents in a role-swapping repeated trust game,
optionally interleaved with iterated myth-writing in which each agent
sees its own and its partner's previous round-text. We vary task order
(game-only, game→myth, myth→game), action-channel noise (uniform ±$5,
bootstrap reciprocity-flipping), and informedness across two frontier
models (Claude Sonnet 4.5, GPT-5-Nano), N=15 seeds per cell. Of 21
(model × noise × myth-order) cells, six show a positive myth effect
(three lift+consolidation with variance ratios 0.05–0.31), four show
destabilisation or harm (all GPT-5-Nano × bootstrap-noise; mean drops
7–10 reward units), and ten are null. Linguistic dynamics are robust
regardless of behavioural outcome: between-agent embedding cosine
converges in 18/21 cells, lag-1 cross-agent cooperativity correlations
are positive in every cell, and Claude's game-response prose references
its own myth vocabulary in 78–82% of rounds — a finding undecidable for
GPT-5-Nano, which emits no reasoning prose. Inter-agent language is
real, asymmetric across models, and sign-conditional on the noise
regime.

---

## Notes on framing choices

- Both candidates avoid the word "consolidation" in the headline because
  it's only one of two patterns the data show; the data also includes
  destabilisation. Honest framing.
- "Sign-conditional on the noise regime" is the active phrase carrying
  the load. Mario can sharpen.
- The asymmetry-across-models point (Claude reasoning prose; nano JSON
  only) is too important to leave for §3.5. It belongs in the abstract
  because it's both a finding and a methodological constraint.
- Numbers cited are bootstrap-derived (RNG seed 42); reproducible from
  `analysis/cell_summaries/deltas.csv`.
