---
title: Findings — hidden defectors and costly punishment
status: current
updated: 2026-10-02
owner: aron
---

# Findings: mechanical defectors and the deduction institution

The defector/punishment thread (2026-08-12 → 2026-08-23, frontier defector
populations 2026-10-01 → 2026-10-02): hidden mechanical
defectors (`defector_action_policy: forced_zero` — scripted zero sends/returns,
no LLM game calls, myths still LLM-written, treatment label hidden) crossed
with an optional sender-side costly deduction stage (2 points, 1:3
cost-to-target, payoff floor zero).

## Defectors: direct losses, no confirmed cascade

25% hidden defectors impose large mechanical balance losses (−$19 to −$22 per
ordinary agent) but **no confirmed population-wide behavioral or cultural
collapse**: at n=10/cell in both GPT-5 Nano and Gemini Flash-Lite, ordinary
sending (−.013, p=.258) and ordinary-myth cooperation language (p=.300) do not
move. _(from researchlog 2026-08-21)_

**Gemini shows a replicated behavior→culture imprint:** defector *authors*
(forced to defect, writing myths freely) diverge culturally in rounds 2–10 —
−.57 cooperation terms/100 words, +.06 threat terms — absent in round 1 and
absent in GPT. Forced behavior becomes culturally visible in Gemini only.
_(from researchlog 2026-08-21)_

**Circulation transmits tone, not behavior.** Swapping defector-authored myths
for ordinary ones at cultural exposures changes presented cooperation language
but not game behavior; the one-step threat-transmission signal from the screen
(23-vs-5 threat matches) failed independent replication (26 vs 22, p=.624) —
a false positive or seed-contingent pattern. _(from researchlog 2026-08-21)_

## Punishment is model-dependent: GPT spends, Gemini sanctions

- **GPT-5 Nano fails the mechanism.** Deductions are near-universal (90.9% of
  opportunities) and untargeted (defector-minus-ordinary contrast ≈0);
  availability yields no reliable cooperative benefit in a 2×2 with defectors.
  Controlled calibration rejects both wordings: current wording deducts in
  100% of high-return cases; cost-salient wording halves spending but removes
  return sensitivity entirely. GPT treats deduction as a salient action, not a
  calibrated sanction — do not tune prose toward a desired result.
  _(from researchlog 2026-08-21)_
- **Gemini Flash-Lite passes.** Calibration: both points at $0 returns, zero
  at any positive return (cross-model return-slope interaction −1.244,
  p=.0057). Live-noise threshold sits between $.25 and $.50 visible return, so
  a true-zero defector is punished ~69% of the time under ±$1 noise. In
  populations, targeting is decisive and replicated: ~81–86% of
  hidden-defector opportunities punished vs ~2–3% of ordinary receivers, with
  zero deductions after any visibly ≥half return. This is selective response
  to observed defection, not role recognition. _(from researchlog 2026-08-21,
  2026-08-22)_
- **Gemini 3.7 Flash generalizes the sanction, more graded**: full deduction
  at 0–10% returns, partial at 25%, zero at ≥50%; live targeting perfect
  (25/25 defectors, 0/65 ordinary). _(from researchlog 2026-08-23)_

## Punishment's downstream effect: crowding, defector-dependent, version-specific

In Gemini Flash-Lite, making deductions *available* *lowers* ordinary return
ratios — confirmed twice: matched arms −.0491 (Holm p=.0099) and the frozen
new-seed 2×2 −.0455 in defector populations (Holm p=.0121), with the
availability×defector interaction −.0373 (Holm p=.0373) and a near-zero
no-defector simple effect. The effect is absent in round 1 and grows to −.137
by round 10; returns shift toward exactly-half and away from generosity —
punishment anchors the minimally fair rule (motivational crowding-out).
Ordinary myths under the institution gain punishment/threat/betrayal language,
more so with defectors present. _(from researchlog 2026-08-22, 2026-08-23)_

**Not in Gemini 3.7:** the crowding effect does not reproduce (+.0026,
CI [−.0016,+.0069]) — 3.7's rigid "return exactly half of receipt" rule
(280/370 decisions exact to the cent) leaves no motivational room. Selective
punishment generalizes across Gemini versions; crowding does not. Frozen
decision: do not scale the 3.7 population design. _(from researchlog 2026-08-23)_

## Cross-model defector set (Claude / GPT-5 Nano / Gemini 3.7): provisional

The negative-only cross-model defector series (2026-08-25; 2 and 8 agents ×
game / game→myth / myth→game × 0 / 25 / 50% defectors, n=5) showed Gemini 3.7
Flash ceiling-locking without forced defection, GPT-5 Nano beating Claude
Sonnet 4.5 on collective returns in some conditions, and Claude degrading
notably once defectors are added. _(from researchlog 2026-09-01)_

**These are model-plus-settings observations, not an isolated model-only ranking.**
The 270 checked final files record direct Anthropic/OpenAI/Google routes, 90 per
model. Claude's cap of 4096 and Gemini's medium thinking/omitted temperature are
recorded. GPT's exact effort is not: minimal was a code-default inference.
Different visible reply formats also change the context retained by
memory-primary. Those differences motivate robustness checks, not blanket
invalidation or a claim that low reasoning explains the ranking.

The review found no identified silent numeric-default explanation in accepted
slide-set decisions. Missing stop reasons and incomplete attempted-run evidence
prevent a universal no-truncation or clean-data certificate. Historical model
aliases also require dataset-specific interpretation.

The later Claude format-study means reproduce, but its arms also changed myth
instructions/self-context and retries. It does not establish that prose explains
the cross-model gap or the effect of adding defectors. See the
[reassessment and descriptive table](../api-audit-reassessment-2026-09-08.md)
before citing either comparison. _(from researchlog 2026-09-08)_

## Frontier defector populations: myths lift Opus 5, not Sol

Eight-agent populations on the September informed negative-only protocol,
with two permanent forced-zero defectors in fixed seats (Agent_4, Agent_8;
one per family in the mix), unaware and still writing myths. Gemini 3.1 Pro is
left out because it is ceiling-locked. Ordinary-agent final resources, mean
(±sd) over 5 runs (45/45 audited finals):

| Population | Game only | Game → Myth | Myth → Game |
|---|---:|---:|---:|
| 4 Opus 5 + 4 GPT-5.6 Sol | 49.1 (±6.3) | 54.6 (±1.6) | 56.8 (±2.1) |
| 8 Opus 5 | 48.4 (±2.3) | 56.5 (±1.6) | 57.3 (±1.2) |
| 8 Sol | 53.4 (±5.5) | 50.5 (±5.1) | 55.9 (±2.1) |

- Myths help Opus 5 clearly (seed-paired myth minus game +8.1 / +8.9 for
  game→myth / myth→game, Welch p < 0.001 both), the mix less surely (+5.5,
  p 0.12; +7.6, p 0.052, all five pairs positive) and Sol not at all (−2.9,
  p 0.41; +2.4, p 0.40).
- The mix sits at the average of its parts in every task order (mix minus the
  Opus/Sol average −1.8 [−7.8, +3.2], +1.1 [−1.3, +3.6], +0.2 [−1.7, +2.1]).
  The mid-tier pattern of mixes beating their parts with myths does not appear.
- In game-only play all-Sol groups end above all-Opus groups (53.4 against
  48.4, Welch p 0.11), so Sol may cope better with defectors; untested at n=5.
- **Myths make agents easier to exploit.** Across the 45 finals, ordinary
  agents send a defector $1.0–1.9 per decision with myths against $0.6–0.9
  without, in every group and task order.
- **A shared myth board changes nothing here** (pilot, n=3): when every agent
  reads every myth written so far instead of its last partner's, ordinary
  resources move by −0.6, −0.5 and +0.5 against seed-matched partner-myth
  runs, and within-round myth similarity stays at 0.31–0.40. With partner myths
  ordinary agents already send $4.81 of $5 to each other, so there is no room.
  The remaining 7 board runs are paused.
- **Saboteur myths barely move cooperation** (pilot, n=3 per channel): when
  the two defectors are privately told to persuade others to send less, and
  keep writing anti-trust myths through round 10, ordinary resources fall in 4
  of 6 seed-matched replicates, by at most 1.4. Ordinary agents send each other
  slightly less (4.83 → 4.65 partner myth, 4.79 → 4.71 board) and the
  defectors more clearly less (1.50 → 1.20, 1.48 → 1.33; only 27 sends per
  cell). The board does not amplify the saboteurs.

Caveats: n=3–5; defector losses are partly mechanical; in single-model groups
both defectors are of that family; fixed seats differ from the September
cells, which drew defectors per replicate, so the mid-tier comparison is not
like-for-like. There is no no-defector Opus + Sol control. Write-up and draft
paper text: `docs/figures/frontier_defector_populations_20261002/README.md`;
launchers `scripts/run_frontier_defector_populations.py`,
`scripts/run_frontier_saboteur_pilot.py`.
_(from researchlog 2026-10-01, 2026-10-02)_

## Standing design gates

- Baseline Gemini cells (both versions) are **ceiling-limited** — 3.7 sent the
  full $5 in all 360 sender decisions, making task-order/identity contrasts
  uninformative there. Punishment calibrations and defector stress tests are
  the variance-bearing designs for Gemini. _(from researchlog 2026-08-21, 2026-08-23)_
- Model eligibility for population pilots runs through the controlled
  calibration gate (selectivity: low-minus-high separation, zero high-return
  punishment, monotonicity) before any paid population cells.
  _(from researchlog 2026-08-21)_
