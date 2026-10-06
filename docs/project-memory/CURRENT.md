# Current research state

Last updated: 2026-10-04 (broad audit partial; API credit exhaustion, source-checked examples available).

## Research question

The broad question is whether narratives created from social experience can
carry information that changes cooperation, including when that information
reaches agents outside the original pair. The current evidence is strongest
for model- and condition-dependent behavioral effects, with promising older
content-transplant evidence. Reliable emergence and persistence of population
culture remains unresolved. See the [design reference](../experiment_design_reference.md)
and [data audit](../research/mixed_future_data_audit.md).

## Current operational state

- **Expanded narrative/rule-evolution analysis (2026-10-03, Ivar):** scope now
  includes canonical September-profile matrices/extensions, main and update
  frontier sets, and ordinary frontier defectors: 736 myth-bearing runs,
  3,872 trajectories, 38,720 entries verified against completed-final hashes.
  Four specialist audits and the [plan](../research/narrative_evolution_20261003/PLAN.md)
  are ready. Separate board/saboteur/transplant/replay protocols are excluded.
  Ivar chose calibration only, capped at $10, instead of the roughly $795.50
  full analysis. The 24-trajectory calibration stopped early: 16 readings
  returned, nine truncated. A four-case repaired test returned 8/8 complete
  readings, 7/8 exact-quote valid. Core examples survived but semantic review
  found shared false novelty labels: automatic evolution totals are not ready.
  Confirmed response costs $3.04; conservative accounting $5.54 including
  interrupted calls without receipts. See the [calibration report](../research/narrative_evolution_20261003/calibration/REPORT.txt).
  **Later successor, Ivar: "ok do it":** simpler fresh 36-myth calibration and
  regression permit only challenge-dialogue screening. A 38,720-myth direct
  Batch pass was launched, estimated $62.50 including allowance/pilots with
  a $100 accounting guard. See the [batch plan](../research/narrative_evolution_20261003/feature_screen/full_dialogue/plan.json).
  **Latest successor, Ivar: "yes run that now please", then "please keep it at $30 ceiling":**
  broad exploratory audit of 300 complete trajectories / 3,000 myths from
  300 distinct runs, covering all eight themes with punishment prioritized.
  Screening estimate $23.07 ($28.84 with 25% allowance); $26 screen guard and
  $30 total guard including targeted whole-trajectory readings. Batch execution
  started; no findings yet. [Plan](../research/narrative_evolution_20261003/broad_audit_300/plan.json).
  Dialogue-only local submitter stopped; STOP_SUBMISSIONS prevents further
  waves. Its two accepted remote batches remain separate existing charges.
  Exploratory candidates are authorized despite imperfect calibration;
  validated prevalence, automatic evolution totals and causal attribution are not established.
  **2026-10-04 inspection:** 2,224/3,000 technically valid labels; 54 saved
  requests report account credit exhaustion. No targeted follow-up receipt yet.
  Partial source readings verify changing punishment/forgiveness prescriptions
  in A004 and role-sensitive recovery in A001, not behavioral transmission.
  [Partial findings](../research/narrative_evolution_20261003/broad_audit_300/PARTIAL_FINDINGS_20261004.txt).
  Human reliability remains unestablished; no complete-corpus findings yet.
  See [D013](decisions/013-strategy-clause-audit.md).
  **2026-10-04 close reading (Claude, no API spend):** scripted defectors write
  their own forced zeros into their myths (17/17 Claude and Sol, 0/6 Gemini Flash and Nano, 1/16 cooperators);
  frontier myths prescribe graded sanctions with a way back, and with a myth task Opus almost
  never sends a defector $0 (0.01–0.02 of sends per run vs 0.50 game-only).
  [Reading findings](../research/narrative_evolution_20261003/broad_audit_300/READING_FINDINGS_20261004.md).
  **Full-text 70% reading (2026-10-04, informed noise only, 2,324 trajectories + blind 240 defectors):**
  sanctions mostly Opus 5 (88%, graded) / GPT-5.6 Sol (87%) / Sonnet (48%); defectors raise them
  (frontier graded, September 29% exclusion); blind self-narration Sonnet 60/60, Sol 30/30, Opus 21/30,
  Gemini Flash 7/60, Nano 0/60. [Results](../research/narrative_evolution_20261003/full_reading_70/RESULTS.md).

- **Myth-pressure pilot (2026-09-28/29, 20/20 audited finals, $27.07):** Sonnet 4.5
  myth-first dyads, word budget (loose / tight) × council between rounds (off / on).
  No shared code emerged. Without a council Sonnet ignores the budget; with one, half
  the agents write to it. What gets through is a plain English rule of play, never an
  invented code. Resources 57–61 in every arm (n=5 each). Stage 2 does not proceed on this
  design. See [the results](../figures/myth_pressure_pilot_20260928/README.md) and
  [D012](decisions/012-myth-pressure-pilot.md).
- **Paper scope (team meeting 2026-09-29):** Aron has drafted every section
  except the results and submits the abstract by 1 October (checking whether it
  can be edited afterwards). Agreed: the results section follows one narrative
  (one noise setting, weaker models before frontier models, one set of
  linguistic analyses); other plots and ablations go to the appendix. Edward
  scoped Ivar's remaining work to: significance tests on the word-adoption
  plots; send and return over rounds split by myth moral, for the agent's own
  myth and the myths it was shown, per experiment type (all splits to the
  appendix, the strongest in the main text, no cherry-picking); plots of the
  observation that cooperation rises after an agent reads its own first myth,
  before it sees a partner's myth (not yet shown); and, if time allows,
  frontier-model defector runs. No further exploratory analyses. Ivar's own
  notes after the meeting set the central question for the analyses as "what
  drives the cooperation?", with the analyses before the defector runs: keep
  the left and middle panels of the moral carry-over plot, stratify by the
  agent's own last myth and the other agent's last myth, test which morals an
  agent takes over from the myths it last saw into its own next myth, and
  split everything per simulation and task order rather than pooling. Edward and
  Mario review the full paper the week before 8 October. Ivar suggested
  dropping the frontier update set for time; this was not decided.
- **Submission target (team meeting 2026-09-22):** AAMAS, abstract 1 October
  2026, full paper 8 October 2026. The mixed-model dyad and eight-agent figures
  lead the paper; frontier results, cooperation round traces and a linguistic
  analysis of the mixed runs follow; noise strength, defector count and a
  possible message-board variant are ablations. Open figure fix: the dyad
  figure duplicates Sonnet/Gemini and must show six unique pairings. A frontier
  mixed-model population ran on 2026-09-28 (see below and
  [D010](decisions/010-mixed-model-population-sizes.md)); frontier mixed dyads
  (Edward's 2026-09-22 proposal) were not run.
- The **linguistic analysis of the September runs completed** on 2026-09-23
  (156 myth-bearing homogeneous and mixed runs, 8,519 myths; no new runs,
  $8.18 judge labelling). Agents take up words from the myth they are shown,
  across model families too (signature-word uptake beats a permutation null in
  all 10 family × size cells). In populations, moral stances spread only
  within a family (dyad matches are confounded by shared games). Myth
  alignment does not predict cooperation, and with agent-within-run fixed
  effects neither an agent's own moral nor the shown myth's moral predicts its
  next move; generous games are followed by generous myths. The human
  validation of the moral labels is not yet done. See
  [the results README](../figures/linguistic_analysis_20260923/README.md).
- **Myth play rules (2026-09-28, judge-extracted, $2.08):** myth prompts ask
  "how the game should be played" and myth-condition game prompts add "take
  any myths into account" (absent in game-only). Before any play, agents open
  closer to the amount their own myth names (+$0.35–0.47 per $ counting only
  endorsed amounts), but 74% of GPT's named amounts are narration, so rule
  following and story rehearsal are not separated. Transplanted amounts move
  Sonnet hosts. No extracted rule explains GPT's round-2 lift or why myth runs
  stop repeating $0. See
  [the results README](../figures/myth_rules_20260928/README.md).
- **Main frontier set (decided 2026-09-28, Ivar):** the 2026-09-18 Opus 5 / Gemini 3.1 Pro /
  GPT-5.6 Sol runs are the main frontier model simulation (the set Ed referred to); this
  supersedes the 2026-09-24 switch to Opus 5.5. Main-frontier mixed dyads (Opus 5/Sol,
  Opus 5/Gemini, Gemini/Sol; 54 runs) and an 8-agent 2 Gemini + 3 Opus 5 + 3 Sol
  population (15 runs) completed 69/69 ($73.18): Sol sends more with other families, Gemini
  sends less only in game-only dyads, game-only dyads with Sol track the round-1 sender
  (74.4 vs 56.2; confounded with seed block), and a myth task brings every mixed pair to the
  ceiling (see
  [D010](decisions/010-mixed-model-population-sizes.md) and
  [the results README](../figures/frontier_main_mixed_20260928/README.md)).
- **Frontier update set (2026-09-28; newer-model check, not the main result):** GPT-6 Sol (effort high, 30/30), Opus 5.5 completed to
  five replicates (30/30) and a three-family eight-agent mixed run (2 Gemini 3.1 Pro +
  3 GPT-6 Sol + 3 Opus 5.5, 15/15; no frontier mixed dyads), $73.22 in total. GPT-6 Sol
  is ceiling-locked, unlike GPT-5.6 Sol; the mixed population is at the ceiling, and only
  Opus 5.5 moves (game-only send 4.79 among other families vs 4.47 among its own; it
  gets to $5 sooner; no sign of partner-specific sending).
  Opus 5.5 is the only frontier model below the ceiling, so frontier runs can show
  pull-up but not contagion of defection. See [D011](decisions/011-frontier-model-rerun.md),
  [D010](decisions/010-mixed-model-population-sizes.md) and
  [the update README](../figures/frontier_update_20260928/README.md).
- **Opus 5.5 check (2026-09-23):** 18 audited runs (replicates 0–2 of the frontier
  matrix, same profile as Opus 5, $31.79). Every cell is within 2 points of Opus 5 on
  the same seeds, with the same task-order ordering; the 2-agent game is more spread
  out (61/75/73). **Decided 2026-09-24 (Ivar): Opus 5.5 is now the frontier Claude
  model** (superseded 2026-09-28: the Opus 5 set is the main frontier result; see
  [D011](decisions/011-frontier-model-rerun.md)).
- The **frontier-model rerun completed 90/90 validated finals** on 2026-09-18:
  the September no-defector matrix on Claude Opus 5 (adaptive thinking, effort
  high), Gemini 3.1 Pro Preview (thinking high) and GPT-5.6 Sol (effort high),
  five replicates per cell, $142.55. Opus 5 sits 13–19 points above Sonnet 4.5
  in five of six cells (6 points in 8-agent myth→game, where Sonnet was already at 69) and shows game < game→myth < myth→game at both sizes (Sonnet showed it in dyads only); Gemini 3.1
  Pro is ceiling-locked like 3.7 Flash; Sol replaces Nano's game-only zero-lock
  with a partial collapse in 2 of 10 game-only runs that the myth task removes.
  Thinking regime changed with the model (Claude 4.7+ rejects the fixed budget).
  See [D011](decisions/011-frontier-model-rerun.md) and
  [the results README](../figures/frontier_rerun_20260918/README.md).
  The gap is a sending gap (resources = 25 + 10 × mean send): higher openings
  and forgiveness of apparent losses (Sol +0.39 vs Nano −0.51 after a visible
  payoff below $5), not reasoning depth; see the
  [gap report](../research/frontier_gap_investigation_2026-09-18.md).
- The **mixed-model dyad stage completed 54/54 validated finals** on
  2026-09-17/18 (Sonnet/GPT, Sonnet/Gemini and Gemini/GPT; game, game→myth,
  myth→game; six replicates with the first sender alternating by family).
  Every family's sending tracks its partner: Sonnet follows Gemini up and GPT
  down, GPT's game-only zero-lock persists against both partners, and Gemini
  stops sending after its first unreciprocated send against GPT (five of six
  runs; one sends twice; corrected 2026-09-22 from a reading that pooled
  replicate cohorts by global round). A myth task breaks
  the lock in every composition. See
  [the results README](../figures/mixed_model_dyads_20260917/README.md) and
  [D010](decisions/010-mixed-model-population-sizes.md).
- The exact two-agent counterpart to the slide-678 rerun completed 35/35
  validated finals with the same seven donor contexts and pinned Sonnet host
  profile. The qualitative ladder largely persists in one fixed dyad: late
  Sonnet is strongest, filler is near baseline, and low-cooperation myths are
  below baseline. See [D008](decisions/008-myth-transplant-isolation.md).
- The seven-cell slide-678 transplant rerun completed 35/35 validated finals
  using existing donor texts, historical myth-only/negative-$5 apparatus and
  pinned September Sonnet settings. The qualitative seed ladder reappeared;
  late Sonnet reached the ceiling while filler and late GPT stayed near baseline.
  Linguistic analysis still precedes later content deletion; see [D008](decisions/008-myth-transplant-isolation.md).
- The **180 no-defector extension runs completed** after explicit user approval
  and design reconciliation. They add no-noise and uninformed negative-only noise
  at the September profiles; retain the 90 existing informed-noise controls.
  See [D005](decisions/005-hold-180-runs.md) for the superseded hold and approval.
- All 270 full-state sources were hash-checked and condition-validated for
  [the resource and delta figures](../figures/figure2_noise_comparison_20260916/README.md).
  Two Claude runs were resampled after role-key errors; GPT unfinished runs
  were rerun after credits were replenished. Preserve those caveats.
- The targeted informed `U(-2, 0)` dyad bridge completed 20/20 clean finals:
  five paired game-only/myth-first replicates for Claude and Gemini. See the
  [range-2 bridge report](../figures/noise_strength_bridge_20260916/README.md).
- No-defector comparisons are fixed dyads versus rotating populations with
  partner history, not a pure manipulation of agent count.

## Current design understanding

- Ordinary checked eight-agent runs hide display names, retain own game/myth
  exchanges through memory-primary chat context, provide the current
  co-player's three-game history separately, and transmit the previous
  round partner's myth locally. These statements are scoped to the checked
  conditions, not every historical run. See [D001](decisions/001-history-and-memory.md)
  and [D002](decisions/002-anonymity-and-local-transmission.md).
- Negative-only informed communication noise is recorded in the later
  comparison design. Earlier replacement, bidirectional, environmental, and
  communication-noise runs remain distinct conditions. See
  [D003](decisions/003-noise-semantics.md).
- Later runs use explicit model-specific request profiles. These profiles do
  not reveal missing historical API settings or make provider reasoning labels
  equivalent. See [D004](decisions/004-model-request-profiles.md).
- The mixed-model dyad stage (54 runs) and the eight-agent contagion ladder
  (90 runs, 1/2/4 Gemini among GPT and 1/2/4 GPT among Sonnet, unchanged
  September population protocol) are both complete. In game-only play a lone
  Gemini is exploited by a GPT population and a lone GPT cooperates among
  Sonnets, while two or more GPTs drag Sonnet populations down; contagion of
  cooperation needs four Geminis, or a myth channel, where one Gemini suffices.
  See [the ladder README](../figures/mixed_model_populations_20260918/README.md).
  Task orders are `game`, `game_myth` and `myth_game`; no shared-prose arm. See
  [D010](decisions/010-mixed-model-population-sizes.md).

## Result boundaries

- Strategy-clause audit (2026-10-02, user-authorized bounded exploration):
  48 complete trajectories / 480 myths from 45 original September and main-frontier
  runs. Machine readings find candidate peer uptake of specific conditional
  advice, including compensation for an already-cheated partner, and both
  additions and dropped clauses. This is not proof of behavioral transmission
  or the mechanism behind increased cooperation. The 44 selected events are
  not a census; human validation remains pending. No new simulations or paid
  judge batch. The human inspector now accepts repeated, individually quoted
  observations (v2). At the user's subsequent request all 12 packet trajectories
  / 120 rounds have quotation-backed AI draft readings in a separate inspector
  tab. Human notes/completion remain separate and untouched. Checking these
  drafts is AI-assisted review, not independent blinded human validation. The
  earlier frozen audit labels are unchanged. Ivar's H01–H04 feedback is now
  preserved and organized separately: it highlights narrative justification,
  identity and inherited custom beyond action rules. A targeted saved-prompt
  check supports exposure-compatible name borrowing within H03's own dyad,
  not transmission from the separate H01 run or a behavioral effect. The subsequent
  bounded steps 4–5 follow-up verifies sources and timing for all 44 selected
  events. Actual-exposure novel-five-word overlap exceeds matched unseen overlap
  by 1.08 percentage points (run SD 2.61; descriptive bootstrap 0.43–1.89,
  45 runs), not a norm/causal estimate. Conditional triggers are insufficiently
  measurable; only two of six explicit-amount cases have an immediate sender
  opportunity, and both already sent that amount. Behavioral enactment remains
  unidentified, not disproven. See the
  [follow-up](../research/strategy_clause_audit_20261002/followup_findings_20261003.txt),
  [D013](decisions/013-strategy-clause-audit.md) and
  [the report](../research/strategy_clause_audit_20261002/README.md).
- Myth effect mechanism (2026-09-30, existing runs only): the myth adds its own
  per-round upward push at fixed partner and own history (Sonnet myth→game
  +$0.14/round), which keeps the gap from fading. There is no sign that myths
  make agents more responsive: Sonnet follows its partner about equally in all
  task orders. Whether myths reduce responsiveness is unresolved; the
  forced-defection test has only 9 events per cell, with hints of weaker
  reaction in game→myth. Ceiling models (Opus 5, Geminis) cannot be tested.
  See [the partner-responsiveness README](../figures/partner_responsiveness_20260930/README.md).
- Myth text and cooperation (2026-09-30, no API calls): the only myth feature that
  predicts play beyond past play is the send amount a myth names, and only for the
  opening send in myth→game (R² +0.17, run-resampled 0.10–0.24; clear in Sonnet,
  borderline in GPT). After round 1 no feature forecasts an agent's round-to-round
  change, in either task order, and a shown myth forecasts nothing for its reader;
  across agents in myth→game, PR #4 rules add a little to returns (Sonnet +0.05 R²). The
  two moral judges' disagreement is concentrated on the fair/generous boundary. See
  [the README](../figures/myth_text_predictiveness_20260930/README.md).
- Cultural transmission pilot (2026-10-02, planted arm, n=3, $18.42; independently reviewed in
  `ASTRA_REVIEW.md`): a named send rule ("the Velmar Rule: send two of five") planted in one
  Sonnet agent's round-1 myth and shown to all 8 agents on a shared board failed the registered
  tests: 0 of 189 other myths use its name or "two of five", 0 of 96 post-board sends are $2,
  and the newcomer retelling rule failed in all 3 runs. This rules out this rule in this
  setup, not transmission in general: paraphrased "send two" echoes exist (uncoded), and the
  board is removed from memory before game decisions. Newcomers sent $0.72 less after
  round-10 boards (not registered, no control). See
  [the README](../research/cultural_transmission_pilot_20261002/README.md).
- Self-anchoring instruction (2026-10-02, replay of 70 Sonnet myth calls, $5.84):
  deleting "Use the myth you wrote in the previous round as inspiration, but adapt
  it in your own way" leaves self-copying unchanged (32%) and slightly lowers
  borrowing from the shown partner myth (−2.0 pts over the unseen-myth baseline,
  −3.5 to −0.5, 20 runs). Self-anchoring is not caused by the instruction; the own myth in chat
  memory is the likelier source (not tested: the replay never removed it). See [the README](../research/self_anchor_replay_20261002/README.md).
- The pinned-profile slide-678 rerun is descriptive at n=5 donor/run replicates
  per cell. It shows strong context-dependent behavioral differences under the
  historical transplant apparatus, but does not isolate narrative form from
  actionable content or separately estimate donor and run variation.
- The matched dyad rerun is also descriptive at n=5. It shows content-dependent
  differences under the current repeated-seed/no-history apparatus, so the old
  Phase-1 content null is not a general dyad result. It does not isolate which
  historical protocol difference accounts for the reversal.
- The audited September no-defector result is model-dependent: myth-first
  increases ordinary-agent resources for Claude and GPT in the checked setup;
  Gemini is already at the ceiling. With forced defection, differences shrink
  or become uncertain. See the [independently reproduced table](../research/mixed_future_data_audit.md#latest-figures-independently-reproduced).
- In the current dyad protocol, Claude's observed difference of medians was
  +20.0 at range 2 versus +3.5 at range 1; all five range-2 paired effects were
  positive. The pattern is consistent with a noise-strength explanation, but
  the five-replicate direct range contrast is too uncertain to establish one.
  Gemini stayed ceiling-locked at 75 throughout.
- Historical myth-transplant results show that different injected texts can
  produce different behavior under an older apparatus. They do not yet prove
  emergent population culture in the current pipeline.
- Do not resurrect interpretations listed in
  [Findings that must not be resurrected](../research/mixed_future_data_audit.md#findings-that-must-not-be-resurrected),
  including the pre-August-12 dyad collapse, the raw 88% meme-inheritance
  claim, or a population-wide cultural collapse caused by defectors.
  The dyad boundary is tracked in [D006](decisions/006-pre-august-dyad-results-invalid.md).
- Descriptive/normative myth prompts and game-side directives are distinct
  interventions ([D007](decisions/007-prompt-regimes.md)); transplant controls
  are a separate causal design ([D008](decisions/008-myth-transplant-isolation.md));
  permanent and random forced defection are distinct stress treatments
  ([D009](decisions/009-defector-treatments.md)).

## Open questions

- Which myth information causes behavioral change, beyond extra text,
  reflection, or explicit strategic advice?
- Does useful information survive transmission to newcomers who never met the
  source agents?
- Which task-order effects survive a comparison where decision-time inputs are
  genuinely different?

Research suggestions are saved separately in
[research_proposals_from_design_review_2026-09-15.md](../research/research_proposals_from_design_review_2026-09-15.md).
They are proposals, not decisions.
