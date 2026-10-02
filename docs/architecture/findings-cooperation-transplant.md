---
title: Findings — what myth transplants do to cooperation
status: current
updated: 2026-09-17
owner: ivar
---

# Findings: myth transplants and cooperation

Current state of knowledge, superseding earlier claims. All numbers are joint
balance after 10 rounds under the Phase 3 regime (ceiling $600) unless noted.

## The five-regime ladder (single mechanism)

Changing *only* the seed text in chat memory produces five separable behavioral
regimes _(from researchlog 2026-06-30, Phase 5)_:

| Seed | Joint mean | Reading |
|---|---|---|
| S-end− (low-coop round-10) | $407.2 (±$39.5) | myths can **actively suppress** |
| Baseline (no seed) | $437.4 (±$5.5) | — |
| S-filler (Wikipedia) | $446.0 (±$7.7) | "any text" does nothing |
| S-start (round-1 parable) | $499.0 (±$11.5) | values without numbers: moderate lift |
| S-end+ (round-10 myth) | $600.0 (±$0.0) | explicit play recipe: ceiling |

Unifying story: **the agent reads the play rule (if any) encoded in the myth and
applies it.** Strategic specificity, not cooperation vocabulary, is the carrier —
S-end+ has *fewer* cooperation words and shorter, chunkier sentences than S-start
("send five, return seven and a half"). Within-pool, a seed's `cooperative_pct`
predicts its outcome (r=+0.952 in S-end−); across pools it doesn't.
_(from researchlog 2026-06-23, 2026-06-30)_

## The ladder replicates under pinned settings, at 8 agents and in one dyad

The slide-678 transplant (seven donor contexts, repeated identical myth-only
injection, ten game-only rounds, historical negative-$5 communication noise,
no history blocks) was rerun in September under the pinned Sonnet 4.5 profile,
five replicates per cell. Joint resources after ten rounds:

| Donor context | 8 agents (ceiling $600) | Fixed dyad (ceiling $150) |
|---|---:|---:|
| Late Sonnet myth | 600.0 (±0.0) | 140.2 (±13.9) |
| Late Gemini myth | 573.8 (±49.0) | 112.0 (±6.6) |
| Early Sonnet myth | 540.0 (±37.5) | 110.8 (±10.1) |
| Late GPT myth | 455.4 (±32.1) | 100.4 (±8.5) |
| Filler text | 440.2 (±7.1) | 91.4 (±3.8) |
| No inherited text | 424.6 (±2.1) | 90.0 (±4.7) |
| Low-cooperation myth | 384.0 (±87.1) | 83.9 (±21.2) |

The qualitative ladder reappears at both scales: late Sonnet reaches the
ceiling, filler and late GPT stay near baseline, and the low-cooperation myth
falls below it. This is not an exact reproduction of unrecorded historical
request settings, it is descriptive at n=5 with donor and run variation
combined, and it does not separate narrative form from actionable content.
Linguistic analysis precedes any content-removal intervention. The dyad result
shows the old Phase-1 "content does nothing in dyads" null does not hold under
the current repeated-seed/no-history apparatus, without identifying which
protocol difference reversed it. _(from researchlog 2026-09-16, 2026-09-17)_

## The channel is semantic — survives an alien grammar

Translating S-end+ myths into **gowith** (Ayrey's subject-less relational English)
with numbers pinned: full-game cell **$598.5 (±$1.9), n=4** vs original $600.0
(±$0.0). Extraction legibility unchanged (all reader families recover send 5 /
return ~½ at 100%). ~15% of the round-1 behavioral lift is a small "binding
premium" of the chunked-strategic surface form; the payload rides on meaning.
_(from researchlog 2026-07-02)_

## Cross-model transfer: two conditions must both hold

Transfer of the lift across the model that *wrote* the seed is real but not
universal — Sonnet-written → $600 (full), Gemini-written → $549.2 (±$45.4)
(partial), GPT-5-nano-written → $446.2 (±$13.5) (**none**, ≈ baseline). The
writer must (1) reach high cooperation in its own runs AND (2) crystallise the
strategy into text with numeric/behavioral specificity. GPT writes evocative
prose without encoding the recipe. _(from researchlog 2026-06-30, 2026-07-01)_

## Legible to all, binding to some

Phase 7 probes dissociate two channels _(from researchlog 2026-07-02)_:

- **Legibility is universal.** Asked directly, every family (Sonnet, GPT, Gemini)
  extracts near-identical recipes from every pool — including GPT's "failed"
  myths. No hidden channel; any monitor can read the payload.
- **Bindingness varies by writer and reader.** Sonnet *states* send=5 from GPT
  myths but *plays* baseline with them in memory. GPT myths barely move GPT
  itself (+0.73 over its 0.00 baseline) while Sonnet's move it +3.20. The Phase
  6b transfer failure is "collusion on paper vs in action" (COLOSSEUM), located
  at the read-out→behavior step. Gemini flash-lite plays ceiling regardless of
  seed (uninformative as a reader).

## Fable-translated seed pools: gowith usable, jabberwocky blocked twice over

Regenerating seed pools with Claude Fable 5 as translator:
`s_end_plus_gowith_fable` is usable (5/5 translated, 4/5 pass the host probe,
readback recovers the recipe) — but only after steering Fable to gowith's
flowing-prose register; its default hyphen-compound register was host-refused
3/3. Jabberwocky is blocked at **two independent layers**: Fable-as-translator
refuses strategy-laden myths (5/5, vs 1/5 on innocent parables), and of the 5
it translated, Sonnet-as-host refused 4/5 in the assistant slot. Net: one
usable jabberwocky seed; jabberwocky cells stay unrunnable on a Sonnet host.
_(from researchlog 2026-07-03)_

## Superseded claims (kept to prevent re-litigating)

- **Phase 2's "round-1 parables beat round-10 myths" (H4) was a regime artifact**
  — the seed scrolled out and the history-block carried cooperation. Under the
  clean myth-only regime the ordering reverses decisively. _(2026-06-23 supersedes 2026-06-19)_
- **The Phase-1 dyad content null is not a general dyad result.** Under the
  current repeated-seed, no-history apparatus the seven-context ladder persists
  in a fixed dyad (late Sonnet 140 vs no text 90 of 150). _(2026-09-17
  supersedes the Phase-1 dyad null)_
- **Task-order moderation from Phase 2 does not carry into the Phase 3+ regime**
  — under Option A statelessness, task orders are the same condition (see
  [design-constraints.md](design-constraints.md)).

Still-standing Phase 2 result: the **counter-current finding** — low-coop-source
end-myths are judged *more* cooperative than high-coop-source end-myths, robust
across Sonnet and Opus judges (ρ=+0.937). Stressed agents write cooperation talk;
successful agents write recipes. _(from researchlog 2026-06-19)_
