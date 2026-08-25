# Claims → Evidence Map (draft for Mon 17 Aug meeting)

Draft prepared 2026-08-14 by Claude for Aron — to be corrected/confirmed at the meeting.

Sources: `researchlog.md` (dated Result entries), `docs/corrected_v2_confirmatory_overview_2026-08-13.md` (authoritative for all dyad/population task-order numbers), `docs/verified-facts.md`. All balances are per-agent means unless noted; Phase 3–7 numbers are joint balances (ceiling $600).

## A. Claims we can currently make

**A1. Writing myths before play (myth→game) raises cooperation in the 8-agent rotating population.** Corrected confirmatory rerun (2026-08-13, "Corrected informed-noise confirmatory rerun completed"; overview doc), n=10 runs/cell, Sonnet 4.5, informed ±$1 noise: myth→game − game = **+7.07 [4.86, 9.27], p<.001** and myth→game − game→myth = **+5.82 [3.40, 8.24], p<.001**, both surviving six-test Holm correction. Balances 66.33±2.51 vs 59.26±2.16 (game) vs 60.51±2.64 (game→myth); trust ratio 0.827±0.050 vs 0.685±0.043 vs 0.710±0.053. Replicated in three earlier n=5 regimes (2026-07-17 plain: trust 0.892/0.716/0.653; 2026-07-17 uninformed noise: 0.851/0.731/0.692; 2026-07-22 informed: 0.838/0.711/0.699). **Confidence: solid.** Caveats: single host model (Sonnet 4.5); cells not seed-paired (Welch tests, not the intended paired design); raw JSON still gitignored, not yet archived.

**A2. The task-order effect operates through how much senders entrust, not through receivers' reciprocity.** Overview doc, 2026-08-13: return ratios sit near one half in every corrected cell (0.497–0.537); the balance ordering is mirrored by trust ratio. **Solid** (descriptive, consistent across all six cells).

**A3. Game→myth does not beat game-only, in either regime, and no population-regime interaction is detected.** Corrected rerun: dyad game→myth − game = −1.91 [−6.76, 2.93], p=.407; 8-agent +1.25 [−1.03, 3.52], p=.263; difference-in-differences +3.16, 95% CI [−2.04, 8.36], p=.218 (myth→game DiD also null: +2.10 [−2.47, 6.66], p=.350). **Solid null.** This replaces the invalidated 2026-08-10 dyad claim (see C).

**A4. Myth→game is also highest in repeated dyads — but only suggestively.** Corrected rerun: +4.97 [0.82, 9.12], unadjusted p=.0227, Holm p=.0798 (does not survive). Balance 69.83±5.47 vs 64.86±2.67. **Suggestive.** High between-run dyad variance.

**A5. Higher transmission fidelity amplifies what the myths carry ("fidelity knob").** Lineage-pointer A/B (2026-07-23, "lineage pointer restores fidelity…"), 8-agent informed, n=5 vs 5: self-myth similarity 0.709±0.020 → 0.839±0.014, **p<0.001**, without diversity collapse (population sim 0.692→0.719). Behavioral amplification: trust 0.838±0.033 → 0.914±0.072, p=0.109; balance 66.91→70.69; and lineage triplet (2026-07-23) shows myth→game rises while game→myth doesn't (59.95→59.00). **Fidelity effect solid; behavioral amplification suggestive** (p=.109, n=5; runs pre-date the v2 protocol but are 8-agent, i.e. on the clean transfer path).

**A6. Memory-primary is the only memory design whose linguistic layer couples to game events.** Noise pilot (2026-07-17): under turbulent play, myth self-similarity 0.676±0.051 (memory-primary) vs 0.826±0.035 (hybrid) and 0.850±0.061 (stateless), p=0.002 vs both; game behavior channel-invariant. **Solid** (methods-level; justifies the design choice).

**A7. Seed myth content, not "any text," determines the cooperation regime — and the carrier is the encoded play recipe.** Phase 5 (2026-06-30), Phase-3 seeded regime, n=5/cell, joint balance: s_end− $407.2±39.5 (active suppression) < baseline $437.4±5.5 ≈ s_filler $446.0±7.7 < s_start $499.0±11.5 < s_end+ $600.0±0.0 (ceiling). Within the s_end− pool, seed `cooperative_pct` predicts outcome at Pearson **r=+0.952** (n=5); across pools vocab does not predict — strategic specificity does. **Solid** within its regime. Caveat: seeded chat-memory regime + `noisy_negative_5` payoff noise, game-only — a different apparatus from the A1 task-order runs.

**A8. Cross-model transfer of the cooperation lift is real but not universal — it requires the source model to have written a recipe.** Phase 6/6b (2026-06-30 / 07-01), n=5/cell into Sonnet 4.5: Sonnet seeds $600.0±0.0 (ceiling), Gemini seeds $549.2±45.4 (partial, +$112 vs baseline), GPT seeds $446.2±13.5 (≈ baseline, no transfer). Structural surface metrics don't explain the GPT failure. **Solid descriptively;** the two-condition explanation (source cooperation × numeric specificity) is interpretive.

**A9. Myths are legible to all model families but binding only to some.** Phase 7 decoder study (2026-07-02): all 3 reader families extract the same recipe from s_end+ (send 5.00, rf ~0.57); behaviorally, Sonnet round-1 send on GPT seeds 3.13±0.35 ≈ baseline 3.10±0.32 vs s_end+ 5.00±0.00 — states the rule but doesn't act on it. **Suggestive-to-solid** (3 samples/cell; cheap round-1 probe validated against full cells).

**A10. The recipe survives a radical grammar transplant (gowith).** Phase 7 (2026-07-02): extraction unchanged (100%, send 5.00±0.00); behavioral probe retains ~85% of lift (4.67±0.62 vs 5.00 original vs 3.10 baseline); full game cell **$598.5±1.9, n=4** vs baseline $437.4±5.5. **Solid**, caveat: rep 4 refusal-censored (Sonnet refusal fires on content × style × exact-context conjunction); jabberwocky untestable on a Sonnet host.

**A11. A silent, unattributable monitor induces no covert adaptation in 10 rounds.** Phase 8 (2026-07-02), n=5: monitor detection 1.00±0.00 from round 2 on (retrospective control also 100%, 40/40); cooperation persisted (pre-penalty $553±32 vs control $585±21); round-10 myths remain 100% extractable. **Solid as an *induction* null** (not a capability null; no attribution signal was available).

**A12. Spontaneous eval-awareness exists but is thin, and myths almost never break frame.** Free mining over 4,898 files (2026-07-02): 399 genuine texts (~1.4/1,000), 3.26/1,000 in game reasoning vs **0.02/1,000 in myths**; noisy-payoff cells elicit more (6.2–8.3 vs 3.9/1,000 — a confound to disclose for noisy cells). **Solid observational.**

**A13. Protocol integrity is verified, not assumed.** 60/60 corrected runs passed the strict audit (5,000 interactions, 5,000 LLM calls, 3,000 noise checks, zero violations); earlier context audit: 25 runs / 3,600 calls, zero issues, audit validated by planted-corruption negative controls. Execution commit `9a972c36…` recorded per run. **Supporting-methods claim.**

## B. Claims we'd like to make but can't yet

**B1. "The dyad/population difference is caused by X" (agent count vs partner rotation vs anonymity vs reputation block).** All four differ together in the current comparison — the overview doc explicitly restricts us to a *population-regime* comparison. Closing sims are implemented and smoke-ready but unrun: fixed-pair 2×3 (`noise_population_isolation_v2_game` family) and named-vs-anonymous rotating 8-agent (`noise8i_identity_v2_game`). Cost: not stated in log [verify]; prior 6-cell n=10 batch scale suggests non-trivial.

**B2. Defection / punishment dynamics (does the culture respond to defectors?).** Scripted `forced_zero` 2-of-8 defector treatment implemented and smoke-tested (2026-08-12), paired 0%-vs-25% cells across all three task orders — **no substantive batch run**. This was a core team ask since 2026-06-15. Cost: not stated [verify].

**B3. Myth-first advantage is larger in populations than dyads (interaction claim).** DiD +2.10 [−2.47, 6.66], p=.350 — descriptively true, not established. Needs a larger (ideally seed-paired) batch, mainly more dyad replicates (dyad variance is the bottleneck).

**B4. Fidelity causally amplifies the culture→behavior effect.** p=0.109 at n=5 (A5). A powered lineage A/B (n≈10–15/arm) at ~$14/5-run cell would close it.

**B5. Monitoring pressure drives strategy encoding underground in a cultural-transmission chain.** Narrowed novelty claim after the lit review (2026-07-02): the *silent* arm is a null; the **visible-penalty variant** ("your myth was flagged…") is the missing cell that makes silent-vs-visible a clean contrast (comparable scale to the $13 silent cell).

**B6. Told-framing changes behavior (eval-awareness ablation).** Observational side done (A12); the funded-but-unrun piece is the small targeted 2×2 framing ablation (Sonnet 4.5 + gpt-5-nano, noisy + clean).

**B7. The seed-transmission arc (A7–A10) holds under the corrected memory-primary/v2 regime.** [Observation, not from the log:] Phases 3–7 ran in the seeded chat-memory, `noisy_negative_5`, game-only apparatus; the confirmatory task-order results use the v2 informed-noise memory-primary apparatus. No bridging cell exists. Either frame the two arcs as separate experiments (defensible) or budget one bridging cell.

## C. Superseded / dead results — do not cite

- **2026-08-10 "game_myth vs game is a population-size effect" (dyad +7.19 advantage).** Broken dyad transfer path: post-round-1 receivers saw communicated transfers of $0.21–$0.28 (should be $3.77–$4.44). Corrected estimate: −1.91. Explicitly superseded 2026-08-12/13.
- **2026-07-17 "population buffers noise" / dyad noise collapse (trust ~0.4, balances $42–47).** Same broken dyad path; corrected dyads are healthy (game-only 64.86, trust 0.797). The 2-vs-8 separation and its anonymity mechanism are "not established" (2026-08-12 entry). Note: the log entry itself carries no supersession banner — add one.
- **June-baseline (pre-2026-07-17) double-memory numbers** (e.g. return ratio 0.607, myth self-sim 0.823): rounds appeared 2–3× in context. Not comparable 1:1 with post-fix runs.
- **Phase 2's H4 ("round-1 parables beat round-10 myths").** Regime artifact (seed scrolled out); reversed in Phase 3 under the clean re-injection regime.
- **Phase 4 task-order cells under Option A.** Statelessness collapsed the three "conditions" into one; all task-order claims must come from Option-B/memory-primary designs.
- The game→myth "plateau/deficit" framing (2026-07-20 mechanism entry): in the corrected batch game→myth ≈ game (+1.25, n.s.), not below it — soften "the ratchet cuts both ways" accordingly. The mem3 memory-depth control itself still stands.

## D. Open decisions for Monday

1. **Paper spine.** Two candidate arcs: (i) *task-order / population-regime* arc (A1–A4, corrected confirmatory — the statistically strongest result) and (ii) *myth-as-transmissible-strategy* arc (A7–A10 + A5: seeds, cross-model transfer, legible-vs-binding, grammar transplant, fidelity). Decide: one as spine + other as secondary, or both — and how to handle the regime split (B7).
2. **Must-have vs nice-to-have sims before the Sep 19 abstract.** Candidates: defector batch (B2 — new phenomenon, high narrative value), fixed-pair/identity isolation (B1 — converts "regime" to "mechanism"), powered dyad/lineage replications (B3/B4), visible-penalty monitor (B5). Rank and cut; the abstract should only commit to claims already in section A.
3. **Framing discipline.** Agree to say "population-regime comparison," never "population-size effect," and "suggestive" for the dyad myth→game contrast, in all drafts.
4. **Lineage prompt status.** Canonical for any new runs, or keep standard-vs-lineage as the low/high-fidelity contrast (the contrast is free — both cells exist)?
5. **Data + release logistics.** Archive the 60 corrected raw JSONs (+ audit output) durably before writing (currently gitignored); Zenodo plan for the public repo; anonymity check for ICLR.
6. **Design decision still open (verified-facts):** keep the one-round gossip-lag myth diffusion (current) vs current-partner myths — must be settled before the methods section is written.
7. **Who runs what / who writes what.** Runner assignments for the chosen sims; writing split (figure-first drafting was the agreed approach on 2026-06-15). Team: Aron, Ivar, Mario, Arabella, Ed, Alex — assign at meeting.
