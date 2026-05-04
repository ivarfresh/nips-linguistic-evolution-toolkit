# 5. Discussion

*Stage: draft (b) candidate, generated from the §4 outline plus this week's analyses (`analysis/build_*` on v4_direct_provider). Six subsections, ~1100 words at draft stage; ms-writer will tighten to the final ~700–900 needed for the NeurIPS body. Format follows §1 / §3 b_draft: numbered paragraphs (¶N) with topic sentence first, supporting prose retained from the §4 outline. Numbers cite §4 paragraphs and `analysis/cell_summaries/*.csv`. The discussion mirrors the question-arc of §1, with §5.1 returning to the central thesis and §5.2–§5.4 tackling each finding in order.*

## 5.1 What we found and what we didn't

**¶1.** *The three findings, restated honestly.* The corpus shows three layers: (i) cross-model variation in baseline cooperation is moderate but persistent under matched conditions, with Claude consistently above GPT-5-Nano (§4.1); (ii) the myth-writing effect on game payoffs is heterogeneous and conditional on the noise regime — six cells positive (three lift+consolidation, one pure consolidation, two pure lift), four cells negative (two harmful, two destabilizing), ten cells null out of 21 measured (§4.3); and (iii) linguistic dynamics inside the myth chain are robust regardless of behavioural outcome — between-agent embedding cosine converges in 18/21 cells, lag-1 cross-agent cooperativity correlations are positive in every cell, and Claude's game-response prose threads its own myth vocabulary into roughly four-fifths of rounds (§4.4–4.5).

**¶2.** *State the negative result honestly.* In about half of cells, adding myth-writing has no detectable effect on mean cumulative reward. This is the result, not a failure of the experiment. The first version of this paradigm does not show a uniform myth-induced cooperation lift, and we report this as the central empirical fact rather than burying it.

**¶3.** *Reframe the §1 thesis.* The §1 question was whether language between LLM agents is *load-bearing* on cooperation. The answer the data give is "yes, but conditionally and asymmetrically across models". Where the noise regime gives the dyad headroom and the perceived signal points in a useful direction (positive uniform noise), myth-writing produces a small mean lift and a substantial variance reduction — a 3× to 20× drop in across-seed dispersion in the strongest cells (§4.3 ¶9). Where the noise regime systematically contradicts the action signal (bootstrap reciprocity reports), myth-writing actively harms cooperation. In the rest of the cells the effect is null. The narrow form of the §1 thesis survives — language is not mere decoration — but the broader form (a uniform cooperation lift) is rejected by the data we have.

## 5.2 The active mechanism is common-knowledge visibility, not content semantics

**¶4.** *The §4.3 mean-and-variance pattern admits multiple mechanism candidates, and the §4.6 factorial decomposes between them.* Three candidates motivated the design of the §4.6 follow-up: (i) myth as a *competing structured prior* — adding narrative content competes with the noise channel's signal; (ii) myth as a *cooperative-content cue* — cooperative narrative semantics provide direct cooperation guidance; (iii) myth as a *Chwe (2001)-style common-knowledge anchor* — both agents seeing the same narrative creates a shared reference that resolves the noise channel's contradictions. The three predict different patterns under the visibility × content factorial: (i) predicts that more visible or more structured myth makes things worse; (ii) predicts that cooperative content rescues whether or not it is shared; (iii) predicts that visibility rescues regardless of content direction.

**¶5.** *The data support (iii) and falsify (i) and (ii).* In bootstrap × `game→myth`, the no-injection baseline cells are harmful regardless of content direction (mean ~58, std ~11 with neutral content; mean ~57, std ~12 with cooperative content). All three injection cells are rescued (mean 67–70, std 3–5), with cooperative content giving a marginal +2 boost over neutral and adversarial content giving a marginal −2 — a content modulation an order of magnitude smaller than the visibility main effect. The competing-prior story (i) cannot explain why a more salient myth (visible) reduces the harm rather than amplifying it. The cooperative-cue story (ii) cannot explain why cooperative content alone, without visibility, leaves the harm intact. The common-knowledge anchor (iii) accounts for both arms.

**¶6.** *Two further controls anchor the reading.* (a) The same A1 injection produces no detectable effect at no-noise baseline ($N = 30$, Δmean −0.8 to +0.9) — ruling out the alternative that injection trivially lifts cooperation by adding more partner-related text to the prompt. The intervention is regime-conditional: it acts where the implicit channel was failing, not as a general "more language helps" effect. (b) Adding forced reasoning prose on top of injection (the A1+A3 combination) erases the rescue: GPT-5-Nano emits explicit cost–benefit reasoning that depresses cooperation back below the no-injection baseline. This complicates the §4.5 "Claude's prose contains myth content → myth content does causal work" framing: when reasoning prose is forced into existence in nano, the prose surfaces strategic prudence rather than narrative coordination, suggesting Claude's higher cooperation may be partly the substrate (Claude-as-cooperator) rather than the carryover.

**¶7.** *Mechanism, stated as a falsifiable claim.* Under bootstrap-style noise, agents observe a perceived reciprocation signal that systematically contradicts their own game-math. With both agents' previous narrative writing made common-knowledge in the game prompt, both have access to the same shared reference and resolve the contradiction by coordinating on it. The shared reference is **the active ingredient**; its semantic direction is a small modulator. This is consistent with the cultural-evolution literature on narrative as a coordination device [@chwe2001; @boyd2011] — narratives stabilise cooperation by being publicly shared (the common-knowledge channel), not primarily by being prescriptively cooperative (the content channel). With $T = 10$ and a 2×3 factorial × four bootstrap variants we cannot fully decompose anchoring-in-self versus anchoring-on-partner; that requires the partner-vs-own injection contrast (§5.6).

## 5.3 Cross-model variation, narrowed honestly

**¶8.** *Two-model contrast in this paper.* Pilot work suggested a Claude / Gemini / GPT-5-Nano three-way contrast with substantively different cooperation regimes. The submitted scope is narrower: the v4 direct-provider corpus contains Claude and GPT-5-Nano; the prior Gemini cells are not yet re-run under the corrected noise pipeline (§3.4). The cross-model claim is therefore a two-model claim in this version: Claude and GPT-5-Nano are distinguishable in mean cooperation level (~14 points of dyad reward at no-noise, §4.1), but both reach similar near-ceiling cooperation under positive noise.

**¶9.** *Position against existing work.* Cross-model variation in social-game behaviour is a recurring finding [@fontana2024; @leng2023; @serapio2023]. Our contribution is more specific: variation persists *under matched experimental conditions* (same prompts, same seeds, same noise mechanism), and it shows up not only in mean cooperation but in the very different shapes of the linguistic channel — Claude emits extensive game-response prose threaded with own-myth vocabulary; GPT-5-Nano emits only the structured action with no surrounding prose at all (§4.5 ¶16–17). This output-format asymmetry is itself a cross-model variation worth naming.

**¶10.** *Methodological recommendation.* Future LLM-game work should report all findings with model-stratified statistics; pooling across frontier models would have hidden the bootstrap-noise harm pattern (which is GPT-5-Nano-only) and the reasoning-prose presence (which is Claude-only).

## 5.4 Cross-task influence: language is in the loop, measurably

**¶11.** *The constrained-channel design choice.* The paper deliberately does not inject the partner's myth into the game prompt (§3.3); the partner's myth reaches the game-reasoning model only via the agent's own previous myth-writing prompt — a thin and easily-overshadowed channel. This makes any positive cross-task signal informative against the strong-language hypothesis: even through the constrained channel, do we see linguistic content shaping game behaviour?

**¶12.** *Three layers of evidence say yes, modestly.* (i) Between-agent embedding cosine converges over rounds in 18/21 cells (§4.4 ¶13) — the myth-writing chain is doing systematic work, regardless of behavioural outcome. (ii) Lag-1 cross-agent correlations in cooperativity-lexicon scores are positive in every cell, with mean per-dyad max-direction Pearson r 0.27–0.58 and 27–73% of dyads showing |r| > 0.5 (§4.4 ¶14). The pilot Claude r ≈ 0.72 case sits at the upper tail of a clearly non-zero distribution; it is not anomalous. (iii) Claude's game-response prose contains content words from its own preceding myth in 78–82% of rounds, with 100% of runs having at least one such reference (§4.5 ¶16). Three different measurement approaches — semantic embedding, narrow lexical scoring, thematic carryover — all detect non-trivial linguistic activity.

**¶13.** *And one negative result that matters.* Coinages from myths do *not* reappear in the agents' game `reason` text in any cell (§4.4 ¶15). The pilot observation that invented myth-words leak across the cross-task channel does not replicate at corpus scale. Carryover, where it exists, travels through *thematic vocabulary* rather than through coined neologisms. This refines the proposed mechanism rather than rejecting it: the myth shapes the agent's subsequent reasoning through its general semantic content, not through specific coinages that act as cross-task tokens.

**¶14.** *Two readings of the modest cross-task effect.* (i) The weak myth → game effect tells us "language is unimportant to LLM cooperation" — the strong reading. (ii) The weak effect tells us "the channel through which language could influence cooperation was deliberately constrained, and even so, multiple linguistic signals leak through" — the design-constrained reading. The data favour (ii). The single most informative follow-up is the partner-myth-injection design described in §5.6.

## 5.5 Limitations

**¶15.** *Short horizon.* $T = 10$ rounds is short relative to typical iterated-learning paradigms. Some predicted dynamics (compression, structural emergence) require many more iterations and probably generational replacement; we explicitly do not test those.

**¶16.** *Two-agent dyad.* No population dynamics, no partner-switching, no opportunity for reputation effects or assortment. The dyad is the minimal interesting unit, not the natural unit.

**¶17.** *Single base model per dyad.* Cross-model dyads (e.g. Claude × GPT-5-Nano) are not tested; the cross-model variation result tells us about same-model behaviour only.

**¶18.** *Single persona, single myth topic.* All v4 cells use the neutral persona and the unconstrained `anything` myth topic. Persona × myth-content interactions are infrastructure-ready but unmeasured here.

**¶19.** *Noise as artifice.* Bootstrap and negative-only noise mechanisms are designed to manufacture headroom; they don't correspond to any naturally-occurring noise process. Findings under those conditions are about LLM responses to a specific intervention, not about LLM cooperation in some neutral baseline.

**¶20.** *Linguistic measures.* Dictionary-based cooperativity counts have no negation handling and no stemming. Embedding-cosine convergence is reproducible but does not pick out *what* is converging. The thematic-carryover metric in §4.5 uses a deliberately conservative own-myth-vocabulary overlap proxy rather than an LLM-judge classification; the latter is deferred to the EMNLP follow-up.

**¶21.** *Reasoning-prose visibility.* GPT-5-Nano emits only the structured JSON action with no surrounding reasoning prose in this configuration. The cross-task linguistic channel (§4.5) is therefore observable only for models that emit reasoning text alongside the action — Claude in our corpus. This is a methodological constraint, not a substantive claim about GPT-5-Nano's internal use of myth content.

**¶22.** *No human baseline.* We compare LLM dyads to other LLM dyads, not to human pairs in the same trust-game protocol. Some claims (e.g. about model-personality magnitudes, about narrative-as-coordination-device) would benefit from human anchors.

**¶23.** *Channel design constrains conclusions.* See §5.4 ¶11: the partner's myth is not directly injected into the game prompt, which deliberately weakens the cross-task channel.

## 5.6 Future directions

**¶24.** *Direct partner-myth injection into the game prompt.* The single most informative follow-up: rerun the design with the partner's most recent myth quoted in the trust-game prompt, and compare. Strongest test of the language → cooperation hypothesis.

**¶25.** *Forced reasoning prose for action-only models.* Re-run GPT-5-Nano with a system prompt that requires a reasoning-prose preamble before the JSON action, and re-code reason text against own-myth vocabulary. Tests whether the cross-task linguistic channel (§4.5) is genuinely absent for that model or merely invisible in the current configuration.

**¶26.** *Longer horizons + generational replacement.* Move from 10 rounds in a 2-agent dyad to an N-agent population with selection across generations [@vallinder2024], and measure whether the consolidation signal compounds into level shifts over time. Out of scope here; flagged for the population follow-up.

**¶27.** *Cross-model dyads and a Gemini re-run.* Holding the experimental design fixed, vary which models are paired. The Gemini re-run under the corrected noise pipeline would also let §5.3 carry a three-model contrast rather than a two-model one.

**¶28.** *Persona × myth interaction.* The infrastructure is ready (§3.4). One factorial pass would test whether myth content interacts with persona to produce cooperation effects neither produces alone.

**¶29.** *Alternative games with more headroom.* Trust game is locked in for both models even under several of our noise interventions. Public goods games and multi-agent commons games [@piatti2024] may have more behavioural latitude for the myth channel to surface.

**¶30.** *Pre-registered confirmatory replication of the consolidation finding.* Because the variance-reduction interpretation emerged partly from inspection of cell-level results, a clean confirmatory study with pre-registered cells, $N$, and analysis path would substantially strengthen the claim.

---

*Notes for ms-writer:*

- *Three places where §5 cites §3 ¶ numbers (e.g. "§4.3 ¶9", "§4.5 ¶16"); these will need to be re-resolved against the final §3 c_final paragraph numbering.*
- *§5.3 ¶8 acknowledges the two-model scope honestly; if Aron pulls Gemini from `data/json/baseline/` for §3.1 cross-model framing, §5.3 should expand to a three-model contrast (one-day expansion path noted in `analysis/MANUSCRIPT_DIFF.md`).*
- *The mechanism speculation in §5.2 ¶5 is the most speculative paragraph; ms-writer should keep it labelled clearly as proposed-mechanism rather than an established interpretation.*
- *Word count target: NeurIPS body ~700–900 for §5. Current draft is ~1500; ms-writer should compress the limitations section (¶15–23) most aggressively — these are list items that compress to one or two sentences each in the final, often via a single comma-separated list rather than per-paragraph bullets.*
