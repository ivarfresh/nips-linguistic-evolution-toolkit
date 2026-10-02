---
title: Findings — task order, myths, and the cultural ratchet
status: current
updated: 2026-10-02
owner: ivar
---

# Findings: task order and myth-mediated cooperation

The endogenous-myth thread (agents write their own myths between game rounds),
distinct from the seeded-transplant thread in
[findings-cooperation-transplant.md](findings-cooperation-transplant.md).
Canonical numbers come from the corrected v2 confirmatory dataset unless noted.

## Myth→Game > Game-only > Game→Myth — the core, replicated result

Writing myths **before** play raises cooperation; writing them **after** play
does not. Confirmed at n=10/cell over all six corrected cells (2/8 agents ×
game / game→myth / myth→game; 60 runs, strict audit clean): final balance
64.86 / 62.94 / 69.83 per agent in repeated dyads and 59.26 / 60.51 / 66.33 in
8-agent rotating populations. Both 8-agent myth→game contrasts survive Holm
correction; game→myth beats game-only in **neither** regime, and its
population-regime interaction is not detected (diff-in-diff +3.16, p=.218).
_(from researchlog 2026-08-13)_

The ordering is model-general: a corrected GPT-5 Nano replication gives
62.64 / 62.34 / 66.26 (Myth→Game vs Game only +3.63, Holm p=.003).
_(from researchlog 2026-08-21)_ Earlier Sonnet triplets show the same ordering
in plain, uninformed-noise, and informed-noise regimes (trust ratio ~0.84–0.89
myth-first vs ~0.65–0.70 game-first); telling agents about the noise barely
moves anything. _(from researchlog 2026-07-17, 2026-07-22)_ The plain 2026-07-17 triplet has a documented provider-change caveat:
the researchlog records OpenRouter reruns for game and game→myth, and their saved
reasoning-text signatures differ from myth→game. Exact native effort is not
recovered by those signatures. Do not promote no-text records from other
triplets to proof of a direct/non-thinking regime. The
[reassessment](../api-audit-reassessment-2026-09-08.md) lists the bounded evidence;
this caveat does not by itself invalidate within-condition observations.
_(from researchlog 2026-09-08)_

## Cross-model September series: the myth effect is model-dependent

The pinned-profile negative-only rerun (270 runs: Claude Sonnet 4.5, GPT-5 Nano,
Gemini 3.7 Flash × 2/8 agents × three task orders × no/25%/50% forced
defection × 5 replicates; September 8 profiles, no truncations) and its 180-run
no-defector extension (no noise and uninformed negative-only noise added to the
90 informed controls) are the current cross-model dataset. Read them per model:

- **GPT-5 Nano locks at zero in game-only play.** At high reasoning it opens
  round 1 with `send 0` in every game-only run (5/5 at both 2 and 8 agents)
  and the population never recovers: zero-receipt rate 1.00 for all ten
  rounds. The zeros are genuine model output (1.7–3k reasoning tokens, bare
  JSON), not parsing defaults. With a myth task in the sequence GPT does send
  (zero-receipt 0.12 / 0.08 in dyads, 0.59 / 0.40 at 8 agents for game→myth /
  myth→game), so myth exchange supplies the cooperative signal the game-only
  start lacks, consistent with the founding-myth account. GPT's return
  proportion on positive receipts is 0.31–0.33 against Claude 0.41–0.50 and
  Gemini 0.45–0.47. Game-only GPT cells therefore contribute no return data.
  Not verified: whether the round-1 zero is specific to the high-reasoning
  profile; the 2026-09-08 cost replay of archived prompts with cooperative
  history got non-zero sends at the same profile. _(from researchlog 2026-09-10)_
- **Claude and GPT gain from myth-first; Gemini is already saturated.** The
  September 14 census reproduced all 405 plotted observations from the 270
  finals: substantial no-defector myth-first gains for GPT and Claude, a
  ceiling-locked Gemini baseline, and shrinking or uncertain differences under
  forced defection. _(from researchlog 2026-09-14)_ The extension's 180 finals
  passed the request-profile and completion audit; two Claude population
  myth-first runs were resampled after a `return` key appeared in a sender
  response, a selection to disclose in any analysis. _(from researchlog 2026-09-16)_
- **Noise strength modulates Claude's myth effect.** Under informed `U(-2, 0)`
  noise, Claude's game-only versus myth-first difference of medians was +20.0
  against +3.5 in the matched `U(-1, 0)` controls (five paired blocks, all
  positive); Gemini stayed at the $75 ceiling in every run. The paired
  range-2-minus-range-1 effect was 9.58 (±17.05), so n=5 is consistent with a
  noise-strength explanation without establishing it. _(from researchlog 2026-09-16)_

## Frontier models: richer by sending more, same task-order ordering

The September no-defector matrix (2 and 8 agents × three task orders × five
replicates) was rerun on each provider's current flagship with reasoning on:
Opus 5, Gemini 3.1 Pro and GPT-5.6 Sol at effort high (90/90 audited finals).
This Opus 5 / Gemini 3.1 Pro / Sol set is the **main frontier simulation**.
Final resources per agent (max 75), mean (±sd over agents), n=5 runs per cell,
with the September models for scale:

| Cell | Opus 5 | Sonnet 4.5 | Gemini 3.1 Pro | Sol (high) | GPT-5 Nano |
|---|---:|---:|---:|---:|---:|
| 2 agents, game | 67.8 (±3.9) | 50.4 (±5.5) | 73.8 (±3.5) | 57.0 (±17.8) | 25.0 (±0.0) |
| 2 agents, game→myth | 73.5 (±3.7) | 55.6 (±6.2) | 73.8 (±2.7) | 57.1 (±7.5) | 53.7 (±11.5) |
| 2 agents, myth→game | 75.0 (±1.7) | 57.8 (±8.3) | 75.0 (±0.9) | 70.5 (±6.3) | 59.6 (±12.4) |
| 8 agents, game | 68.4 (±2.6) | 55.2 (±2.2) | 74.7 (±2.2) | 63.1 (±14.7) | 25.0 (±0.0) |
| 8 agents, game→myth | 73.0 (±2.4) | 54.1 (±2.1) | 74.5 (±2.1) | 67.0 (±8.6) | 38.0 (±13.2) |
| 8 agents, myth→game | 74.7 (±2.0) | 68.7 (±5.5) | 75.0 (±1.8) | 73.6 (±2.5) | 45.0 (±12.0) |

- Opus 5 sits 13–19 points above Sonnet 4.5 in five of six cells (6 in
  8-agent myth→game) and shows game < game→myth < myth→game at both sizes,
  which Sonnet showed in dyads only. Gemini 3.1 Pro is at the ceiling
  everywhere, like 3.7 Flash. Sol replaces Nano's game-only zero-lock with a
  partial collapse in 2 of 10 game-only runs that the myth task removes. The
  myth-first advantage persists on Opus 5 (+6 to +7) and Sol (+11 to +14).
- **Why they end richer.** Final resources equal 25 + 10 × mean send in every
  run (returns only move money within a pair), so the frontier gap is entirely
  a sending gap. Opus 5 opens at 3–4, raises its send by 0.35 after a
  profitable round (Sonnet −0.06) and sends about 1.5 more than Sonnet at the
  same partner input (after a fair return: 4.46 against 2.50). Sonnet's stated "50% baseline" policy reads the
  noise-lowered numbers as falling trust, so a matching player drifts down
  (dyads 3.0 → 2.3). Sol opens at 2.5–3 (Nano: 0 in 50/50) and raises its send
  by 0.44 after an apparent loss (Nano −0.58). Reasoning is not the driver:
  Opus used a median of 0 thinking tokens per game call (Sonnet 444). The gap
  is opening prior, escalation and forgiveness. Descriptive at n=5; tier versus
  generation is untested.
- **Newer-model check (not the main set).** Opus 5.5 stays within 2.3 points of
  Opus 5 in every cell with the same ordering, while thinking 3–9× more per
  call. GPT-6 Sol at effort high is ceiling-locked ($5 in 732 of 750
  decisions). No frontier model defects on its own, so frontier runs show
  pull-up, not contagion of defection.

Write-ups: `docs/figures/frontier_rerun_20260918/README.md`,
`docs/research/frontier_gap_investigation_2026-09-18.md`,
`docs/figures/frontier_update_20260928/README.md`; decision records D010, D011.
_(from researchlog 2026-09-18, 2026-09-23, 2026-09-28)_

## Mixed-model groups: each model adapts to its partners

Mixed groups put different model families in one game, everything else equal
to the September informed-noise controls; model names are never shown.
Total dyad resources after ten rounds (max 150), mean (±sd), n=6 per mixed
cell, with the September homogeneous dyads for scale:

| Composition | game | game→myth | myth→game |
|---|---:|---:|---:|
| Sonnet + GPT | 64.0 (±14.3) | 116.9 (±26.7) | 120.2 (±18.1) |
| Gemini + GPT | 63.3 (±8.2) | 141.3 (±5.9) | 147.2 (±2.9) |
| GPT + GPT | 50.0 (±0.0) | 107.4 (±20.9) | 119.3 (±23.6) |
| Sonnet + Sonnet | 100.8 (±8.0) | 111.2 (±9.3) | 115.6 (±16.4) |
| Sonnet + Gemini | 140.2 (±6.2) | 146.0 (±1.3) | 148.4 (±1.4) |
| Gemini + Gemini | 150.0 (±0.0) | 150.0 (±0.0) | 150.0 (±0.0) |

- **A mixed score is adaptation, not an average of fixed styles.** Sonnet
  sends $2.5 to another Sonnet, $4.1 to Gemini and $1.1 to GPT in game-only
  dyads, while its return proportion stays at 0.38–0.50. GPT sends about $0
  in game-only dyads whatever the partner, but once a myth task unlocks it, it
  sends $4.1–4.7 to Gemini, $3.3–3.6 to Sonnet and $2.9–3.5 to another GPT,
  and returns about 0.40 to Gemini against 0.32 to GPT.
- **Gemini 3.7 Flash is conditional.** Its homogeneous $5 is sustained by
  reciprocation. Against GPT in game-only play it sends $5 on its first own
  turn, gets nothing back, and then sends $0 (five of six runs; one run sent
  $5 twice). Sonnet against GPT opens with $2–4 and stops after its first to
  fourth own turn. GPT's game-only zero-lock is not broken by either
  cooperative partner.
- **Eight-agent ladder** (1/2/4 Gemini among GPT, 1/2/4 GPT among Sonnet;
  per-agent resources, max 75). One Gemini among seven GPTs is exploited (the
  Gemini agent ends at 20.6, the population at 26.1 against 25.0 for 8 GPT);
  GPT only starts sending to Gemini at four Geminis ($2.69), where the
  population reaches 53.7. With a myth task one Gemini lifts the population to
  52.2 / 58.9 (8 GPT: 38.0 / 45.0). One GPT among seven Sonnets cooperates
  (sends $3.21; population 53.5 against 55.2), but two GPTs pull the
  population to 43.7 and four to 40.0, with Sonnet-to-Sonnet sends falling
  from $3.02 to $2.22. With myths every mixture stays flat (55–57 game→myth,
  63–69 myth→game). Effects run through sending; only GPT's return proportion
  ever moves.
- **Mixed versus the average of its parts (paper Table 1), n=10.** After a
  216-run extension that changed only the replicate id, 16 of 27 cells differ
  from the average of their homogeneous parts at Welch p < 0.05 (11 after
  Holm). Sonnet + GPT game-only is now clearly below (−8.7). 1 Gemini + 7 GPT
  game→myth shrinks from +9.6 to +4.4 and is no longer significant; 1 GPT +
  7 Sonnet and 4 GPT + 4 Sonnet game→myth also lose significance, while
  4 Gemini + 4 GPT game-only and 4 GPT + 4 Sonnet myth→game gain it.
- **Frontier mixes.** Opus 5 + Sol, Opus 5 + Gemini 3.1 Pro and Gemini + Sol
  dyads, plus a 2 Gemini + 3 Opus 5 + 3 Sol population (n=5–6, descriptive):
  Sol sends more with either partner than among its own kind (8-agent
  game-only 4.73 against 3.81). Gemini drops only in game-only dyads (3.83
  with Sol, 4.15 with Opus 5, against 4.88). In game-only dyads with Sol the
  outcome tracks who opened (partner opened: 74.4; Sol opened at $2.50–3:
  56.2), but first sender and seed block are confounded. With myth→game every
  mixed pair ends at 74.3–75.0. In the newer-model mix (2 Gemini + 3 GPT-6
  Sol + 3 Opus 5.5) everything sits at the ceiling; only Opus 5.5 moves (4.79
  among other families against 4.47 among its own kind, game-only).

Trace timing is read by each agent's own turn: in a dyad each agent sends
every other round and the first sender alternates across replicates, so
traces by global round mix first and second turns. Write-ups:
`docs/figures/mixed_model_dyads_20260917/`, `mixed_model_populations_20260918/`,
`mixed_model_family_split_20260922/`, `mixed_vs_average_n10_20261001/`,
`frontier_main_mixed_20260928/` (all under `docs/figures/`).
_(from researchlog 2026-09-17, 2026-09-18, 2026-09-22, 2026-09-28, 2026-10-01)_

## Mechanism: the cultural ratchet cuts both ways

- **Founding myths seed the opening.** Myth-first populations open generous and
  hit the $5 send ceiling by ~round 4; game-first populations cold-open at
  ~3.0 (structurally identical round-1 context to game-only) and climb slowly.
  The plateau follows the myths, not memory depth: a game-only run with matched
  memory rounds escalates normally. Directive myths written after play codify
  the cautious status quo and anchor it. _(from researchlog 2026-07-20)_
- **Washout is a historical observation with a provenance caveat.** In a 20-round 8-agent washout
  (n=5/arm, directional), per-round sends never converge (myth-first ~4.9 vs
  game-first ~4.4 at round 20) and the cumulative-balance lead grows from
  +11.4/agent at round 10 to +17.9 at round 20. In dyads, sends do converge
  and the lead plateaus (~+6). _(from researchlog 2026-08-19)_
  The 30 checked washout finals contain 23 OpenRouter and seven Anthropic
  top-level labels. Two Anthropic-labelled population myth→game finals contain
  reasoning text in rounds 1–10 but not 11–20. A single run-level label cannot
  establish their full request history; inspect that transition before treating
  the longer-horizon contrast as clean evidence of persistence. This is a
  recorded signature change, not proof of an exact provider switch.
  _(from researchlog 2026-09-08)_
- **Transmission fidelity moderates the effect in both directions.** A lineage
  pointer ("Base it on the myths you wrote earlier this session") raises
  self-myth similarity 0.709→0.839 without collapsing population diversity,
  and amplifies whatever the myths crystallize: myth-first balance rises
  ($66.91→$70.69) while game-first stays anchored ($59.95→$59.00) — a
  fidelity knob on the culture→behavior link, with the decision side
  uninstructed. _(from researchlog 2026-07-23)_

## What in a myth reaches play: the agent's own plan

The myth works mainly as the writer's own plan for its next move. Most of the
effect sits in the opening send.

- **Your own first myth sets your opening send.** On the 156 September myth
  runs, Sonnet and GPT senders in round 1 of myth→game send $0.67 per $1 their
  own myth names (0.55–0.79; 62 senders, 37 runs) and match it exactly 71% of
  the time. Within a cell the myth is the only input that differs between
  send calls. The link fades by rounds 6–10. On the frontier set it is
  tighter: 50 of 52 round-1 senders send exactly the named amount ($0.85 per
  $1; Sol 31/31, Opus 5 19/21, Gemini 3.1 Pro always $5). Counting only
  amounts the myth endorses, rather than merely narrates, gives +$0.35–0.47
  per $1; 74% of GPT's named amounts are narration, so "the story rehearses
  the move" fits as well as "the rule is followed".
- **The link is causal.** Replaying logged September decisions with only the
  amount in a fixed rule sentence edited ($1/$2/$3/$5), the next send moves
  $0.93 per $1 in the agent's own first myth (0.88–0.99) and $0.68 (0.55–0.80)
  in its own later myth (Sonnet and GPT pooled, both pre-registered rules
  confirmed). A partner's myth the agent read moves it $0.23 (0.12–0.34):
  real but small, and inconclusive under the fixed rule; it is a lower bound
  because the agent's own reaction myth stays unedited. Gemini 3.7 Flash
  sends $5 whatever its myth says. An amount told inside the story moves
  Sonnet less ($0.57 and $0.24 per $1).
- **Named amounts predict; moral labels do not.** Before any play the amount a
  myth names adds R² 0.17 [0.10, 0.24] for the opening send (Sonnet myths
  saying "three of five" open at exactly 3.00, 12 of 12), while the three-way
  moral label is flat. Judges disagree mostly on fair versus generous
  (80–83% of disagreements), where myths often endorse both. After round 1 a
  change in an agent's myth forecasts no change in its next move in either
  task order; about 150 myth features give no held-out gain over past moves.
  The one later trace is a stable return habit across agents in myth→game
  (extracted rules add Sonnet +0.05 R² for returns). There is no evidence
  that partners whose myths agree cooperate more. In 2 GPT + 6 Sonnet runs,
  pairs whose myths disagree on how much to send give more (−0.039 per SD,
  Holm 0.050), the opposite of the hypothesis; it is a level effect and
  disappears with agent fixed effects.
- **Myths add their own push; they do not make agents more responsive.** With
  partner history and own previous send held fixed, myth→game sends are
  higher every round (Sonnet +$0.14 [0.06, 0.28], Sol +$0.20, Nano +$1.44;
  game→myth Sol +$0.29, Nano +$1.59, Opus +$0.11), and the Sonnet gap does not
  fade over ten rounds. Sonnet follows its partner about equally in every task
  order ($0.54 / $0.39 / $0.48 per $1, correlational) and cites the partner as
  often with a myth as without. Its myth lessons split about 60% "reciprocate",
  40% "give regardless" (Sonnet). The forced-defection test is too small to say
  whether myths blunt reactions to defection.
- **The game-side prompt line alone does not unlock GPT.** Myth prompts ask
  how the game should be played, and myth-condition game prompts add "take any
  myths into account"; adding that line to game-only play does not break
  GPT's zero-lock, and no extracted rule explains GPT's round-2 lift.
- **Self-copying is not caused by the "use your previous myth as inspiration"
  line.** Replaying 70 logged Sonnet myth calls with and without it leaves
  self-copying unchanged (32.4% vs 32.1%) and slightly lowers borrowing from
  the shown partner myth (−2.0 points, −3.5 to −0.5). The own myth in chat
  memory is the likelier source (inferred, not tested).

Write-ups: `docs/research/myth_predictors_20260930/`,
`docs/research/frontier_myth_predictors_20260930/`,
`docs/research/myth_replay_probe_20261001/`,
`docs/research/self_anchor_replay_20261002/`, and under `docs/figures/`:
`myth_rules_20260928/`, `myth_text_predictiveness_20260930/`,
`partner_responsiveness_20260930/`.
_(from researchlog 2026-09-28, 2026-09-30, 2026-10-01, 2026-10-02)_

## Textual-evolution evidence (words spread; planted content has not been shown to spread)

The first three bullets are deterministic no-LLM analyses over the 60
corrected runs; the rest use the September and frontier myth corpora.

- **Capsule genealogy:** 5,000 capsules, 32,700 verified parent→child edges.
  Exact fact inheritance is far stronger in dyads (43–61%) than 8-agent cells
  (11–15%); in dyadic myth→game, 30% of event references reach beyond the
  literal context window (8-agent: 2.5%). Norm inheritance ~90% in every
  myth-bearing cell. _(from researchlog 2026-08-19)_
- **Meme evolution:** myth-bearing conditions carry more ideas per capsule
  (~2.7 vs ~1.8). Proportional reciprocity dominates (75% capsule prevalence);
  25–31% of retained memes shift variant, mostly between fixed-percentage and
  responsive-reward forms. Two decoded private-belief packets (myth-carried
  noisy perceptions surfacing in a partner's game reasoning) exist but are
  rare. _(from researchlog 2026-08-19)_
- **Meme transmission is mostly base rate.** Against a degree-preserving
  rewiring null with exposure contrasts, the raw ~88% "inheritance" reduces to
  single-digit percentage-point excess; the largest regex families fail a
  future-myth control (a not-yet-visible myth predicts adoption as well as the
  seen one), and `noise_adaptation` is prompt elicitation. Under blinded
  LLM-judge labels (5,000 capsules) no family survives cleanly. Two weak regex
  candidates remain (sustainable_equilibrium, proportional_reciprocity).
  _(from researchlog 2026-08-28)_
- **Words cross families; morals spread only within a family, and weakly.**
  In the September homogeneous and mixed runs (156 runs, 8,519 myths), agents
  adopt new words from the myth they were shown 1.3–2.5× the rate from a
  comparable unseen myth. Mixed runs break the shared-model confound: a
  partner-family signature word is adopted more when the shown myth used it,
  beating a permutation null in all 10 family × size cells (strong
  observational evidence, not a causal test). An outnumbered GPT drifts toward
  Sonnet's style (0.26 as 1 of 8, 0.04 as 4 of 8). On the frontier set words
  copy more than in September, across families about 4×.
  Moral stances (Arabella Sinclair's 3-label rubric) move toward the shown
  myth only in 8-agent myth→game runs, where the shown myth predates any
  shared game: +5.9 points for mixed same-family pairs (future-myth placebo
  +2.3), resting mainly on Sonnet (+9.6, p 0.007; GPT +3.1, p 0.11). The
  homogeneous +4.8 is only suggestive (placebo +3.9, p 0.046). Across families
  the effect is +0.7 (p 0.40); because "across" always means GPT with another
  family, this says nothing detectably crosses to or from GPT. No single run
  shows the effect. On the frontier set the within-family effect passes its
  placebo (+6.8 against +0.6) and nothing crosses families. Dyad matches are
  larger but confounded by the partners' shared games.
- **Morals follow play.** With agent-within-run fixed effects neither an
  agent's own moral nor the shown myth's moral predicts its next move; a more
  cooperative game is followed by a generous myth (+0.16 per unit of send
  fraction). The amount a shown myth names does not move the reader's next
  send (−0.00, −0.10 to 0.10, author not the current partner) but echoes into
  the reader's next myth (September +$0.07, not Holm-significant; frontier
  +$0.13 per $1, Holm-significant). Among GPTs the moral mix drifts from "be
  generous" to "be fair" (4 Gemini + 4 GPT: 75% → 20%; 1 GPT + 7 Sonnet:
  46% → 14%). Open anomaly: Gemini's generous share falls among GPTs even as
  GPT's sends to it rise to $5; Gemini 3.1 Pro also drifts generous → fair at
  a constant $5 send, which favours a judge-reading explanation.
- **Consistency drift is mostly self-copying.** The share of Sonnet myths
  using consistency keywords rises from 0.05 (±0.07) to 0.86 (±0.14) over a
  run (8-agent homogeneous myth→game). Mostly this is self-copying: a Sonnet
  agent uses the word 72% of the time if its previous myth did, 34% if not.
  Within agents it has no detectable effect on send or return levels. The
  embedding version of the rise holds only as a consistency-minus-generosity
  score, and the embedding self-copying test is not specific to consistency.
  Two Sonnet round-1 results survive the consistency tests' Holm correction
  (keyword consistency goes with later cooperation, +0.095 / +0.110); they
  are still unconfirmed.
- **Myth map.** In a 2-D map of September myth embeddings, round-1 myths
  cluster by family (silhouette 0.30–0.41 in every cell) and families keep
  their own morals (GPT opens "be fair", 139/182 myth-first; Gemini "be
  generous", 73/97). Single-model runs stay apart through round 10. In mixed
  dyads partners' myths end 0.04–0.11 cosine closer than different-family myths
  from other runs of the same pairing and round, in all six cells, with no gap
  at round 1. Closer language does not predict cooperation.
- **A planted rule does not spread (pilot, no control arm).** In three Sonnet
  4.5 8-agent myth→game runs with a shared myth board, Agent_1's round-1 myth
  carried a "send two of five" rule that every agent's myth prompt then held.
  No other agent repeated the rule's exact words (0 of 189 myths), though
  paraphrased "send two" echoes exist and are not yet coded, and no agent sent
  $2 after the board appeared (0 of 96 sends; Sonnet's September minimum is
  $2.50). Agent_1 sent $2 in round 1, then $3–4, and by rounds 4–7 recast the
  rule as an opening move. The pre-registered retelling rule was met in no
  run. Fresh newcomers shown the other agents' round-1 or round-10 myths sent
  $2 in none of 60 decisions; the round-10 test is 30 decisions drawn from
  three fixed boards, not independent tests. They sent less after round-10
  boards than after round-1 boards ($4.27 → $3.55; −$0.72 (±0.24), sign test
  p 0.25), but this was not pre-registered and needs the control arm. An
  independent review rebuilt all 216 board prompts and confirmed the plant
  was delivered exactly. Design limit: the board is
  replaced by a one-line note before game decisions, so read myths reach play
  only through the agent's own rewritten myth. The seed story also praises
  generosity, and a $2 rule costs payoff.
- **A word budget yields maxims, not a code.** Cutting the myth a Sonnet 4.5
  partner receives to its first N words (down to 20), with or without a
  council between rounds, did not produce a shared code (20 dyads, criteria
  fixed in advance). Only 19% of tight-budget myths were written for the cut
  (bar: 80%); the council, not the budget, moved length. No invented word was
  shared. What got through was an English rule of play ("send three of five
  faithfully, return proportionally"). Resources were 57–61 in every arm.
  A redesign needs a real cost for going over budget or information that must
  be passed on.

Genealogy and meme counts establish visible textual inheritance only. Across
families, lexical transmission is strongly supported (observational, not causal-grade). For play, the causal evidence
is about the agent's **own** myth: editing the amount in it moves the next send
almost one for one. A read myth's amount moves the reader's send by at most a
small amount in replay ($0.23 per $1) and not detectably in the observational
data, and a rule planted in one agent's myth did not spread in the pilot.
Write-ups: `docs/figures/linguistic_analysis_20260923/README.md`,
`docs/figures/myth_convergence_map_20261002/README.md`,
`docs/research/cultural_transmission_pilot_20261002/` (with `ASTRA_REVIEW.md`),
`docs/figures/myth_pressure_pilot_20260928/README.md` (D012).
_(from researchlog 2026-09-23, 2026-09-29, 2026-09-30, 2026-10-02)_

## Return behavior: the send/return gap is a denominator artifact

Plots normalize send by the $5 endowment but return by the *tripled* receipt.
In dollars, receivers returned more than was sent in 98.5% of the 1,500
corrected confirmatory dyad-rounds; returned/sent ≈150% everywhere. Receivers
anchor on "return half of the tripled pool" (~50–54% in all cells); myths move
sends, not returns. The old zero-send explanation is retracted.
_(from researchlog 2026-08-20)_

## Superseded claims

- **"60–77% edge transmission" of memes (2026-08-19)** was base rate; honest
  excess over the rewiring null is single-digit pp and no family survives
  under judge labels. _(2026-08-28 supersedes 2026-08-19)_
- **"game_myth vs game is a population-size effect" (2026-08-10)** ran on the
  broken dyad transfer-noise path (post-round-1 receivers saw ~$0) and is
  superseded by the corrected confirmatory result above. _(2026-08-13
  supersedes 2026-08-10)_
- **Noise-buffering by population size** (dyads collapse under noise, 8-agent
  populations don't; 2026-07-17) predates the dyad transfer fix; the corrected
  dataset shows dyads outperforming 8-agent cells on balance. Treat the old
  dyad-collapse numbers as artifact-contaminated.
- **Mixed-dyad stopping times read by global round (2026-09-17/22)** mixed
  replicates' first and second turns. Read by each agent's own turn, Gemini
  stops after one unreciprocated send and Sonnet after its first to fourth own
  turn (see the mixed-model section). Means and the adaptation result are
  unchanged. _(2026-09-22 correction)_
- **Opus 5.5 as the frontier Claude model (2026-09-24)** was reversed: the
  Opus 5 / Gemini 3.1 Pro / GPT-5.6 Sol set is the main frontier simulation,
  and the Opus 5.5 / GPT-6 Sol runs are a newer-model check. _(2026-09-28
  supersedes 2026-09-24)_
- **September moral and amount transmission wording (2026-09-23)** is narrowed
  as stated in the textual-evolution section: cross-family word uptake is
  observational, the homogeneous moral uptake is suggestive only, and a shown
  myth's amount has no detectable effect on the reader's next send.
  _(2026-09-23 and 2026-09-30 corrections)_
