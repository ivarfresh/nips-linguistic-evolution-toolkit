---
title: Research positioning — novelty claims and planned arms
status: current
updated: 2026-07-02
owner: ivar
---

# Research positioning

What we can claim, against which prior work, and the arms in flight. Full
literature report: `docs/lit_review_eval_awareness_stego.md`; per-paper notes:
`projects/neurips-2026-llm-ling-evo/references/notes/eval-awareness-stego/`.

## Novelty verdicts on the planned experiments _(from researchlog 2026-07-02)_

- **Emergent steganography under monitoring pressure — strong claim is scooped.**
  "Monitor pressure drives encoding underground while behavior persists" is shown
  by Baker et al. 2025 (2503.11926, obfuscated reward hacking), Ming 2026
  (2603.15655, MARL), Mathew et al. 2024 (2410.03768). **Our claimable version:**
  the same dynamic in a *cultural-transmission-chain* setting, over a
  *natural-language narrative (myth) channel*, sustaining a *prosocial
  cooperation lift* — that combination is unclaimed. Cite the three as priors we
  extend. A null (no obfuscation emerges) is credible and publishable —
  Jiralerspong 2026's intent–capability gap.
- **Decoder asymmetry — open**, but pre-empt "isn't this subliminal learning?"
  (Cloud et al. 2025, 2507.14805: same-base-model transfer) and use Anwar et al.
  2026's (2602.23163) action-based "steganographic gap" metric. Phase 7 answered
  the legibility half: extraction is universal; the asymmetry is behavioral.
- **Paraphrase defense — method canonical** (Roger & Greenblatt 2023); Gisler
  2026 predicts survival. Gowith (harsher than paraphrase) already survived, so
  this rung is confirmatory only.
- **Eval-awareness framing ablation — partially done** (Robinson & Burden 2025
  in PD). Position as extension; adopt Devbunova 2026's 2×2 context × format
  design so the effect isn't format sensitivity. Note our agents are *told*
  they're in an experiment (`config/experiments.yaml` system prompt) — the
  question is graded salience, not binary awareness.
- **Purpose-inference probes — method off-the-shelf** (Needham 2025, SAD);
  correlating inferred purpose with cooperation is the new bit.

Five load-bearing citations: Motwani 2024 (2402.07510) · Baker 2025 (2503.11926)
· Cloud 2025 (2507.14805) · Needham 2025 (2505.23836) · Mathew 2024 (2410.03768).

## Arms in flight (2026-07-02)

1. **Silent-monitor arm** (done — induction null): 10 rounds of silent,
   unattributable confiscation produced zero covert adaptation — detection flat
   at 100% from round 2, in-game cooperation persisted ($553 ±$32 pre-penalty vs
   control $585 ±$21), round-10 myths fully legible. The balance-only signal
   (truthful payoffs beside a stagnant total, visible in-prompt) was ignored.
   Frames as: covert encoding does not emerge from selection pressure alone on
   this timescale; attribution is the missing ingredient. Next cell:
   **visible-penalty variant** — agents told their myth was flagged — making
   silent-vs-visible the paper's two-cell contrast ("evasion requires knowing
   you're watched"), and bridging to the eval-awareness arm.
   _(from researchlog 2026-07-02)_
2. **Eval-awareness transcript mining** (done): spontaneous eval-awareness is
   real but thin — 399/280,658 texts (~1.4/1,000), almost entirely in game
   reasoning, never in myth text; Sonnet rarest-but-sharpest (reconstructs the
   study hypothesis and sometimes adjusts play); noisy payoffs elicit ~2× the
   baseline rate (confound for noisy cells). Verdict: fund a **small targeted**
   framing arm (Sonnet + gpt-5-nano, noisy + clean control, behavioral delta),
   not a broad sweep. Data: `data/phase8/eval_awareness_mining.json`.
   _(from researchlog 2026-07-02)_
