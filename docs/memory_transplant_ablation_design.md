# Memory-Transplant Ablation — Design Doc

**Are myths sufficient causal carriers of cooperation?**

Owner: Ivar · Workstream: "Ablations of agents who have lost their memories"
Status: Phase 1 — locked · Scope: dyadic only (society = future work)

Fixed parameters (Phase 1): model **Claude Sonnet 4.5** · **N = 2** agents · **10 rounds** per run · both agents seeded **identically** · **5 reps per seeded cell** · negative-noise condition (`noisy_negative_5`) · seeds harvested from existing v4 direct-provider Sonnet 4.5 runs.

---

## 1. Background and motivation

The broader project studies whether shared **narrative** functions as a medium of normative transmission in collectives of LLM agents. A repeated **trust game** supplies a clean, quantitative behavioral readout (cooperation = amounts sent and returned); collaborative **myth-writing** is the candidate cultural channel running alongside it. The headline question is whether the stories agents tell each other do *causal work* in establishing and sustaining cooperation, or are merely **epiphenomenal** — narration of behavior the game dynamics already produce.

The base result from the main runs ("myth condition cooperates more than no-myth condition") is correlational at the level of *condition*. It is compatible with four mutually distinct explanations:

1. **Causal seed** — the myth seeds/stabilizes a cooperative norm. *(the interesting claim)*
2. **Readout** — cooperation comes from game dynamics; agents just narrate it.
3. **Warm-up** — any extra generation raises cooperation; content is irrelevant.
4. **Content-specific** — it is specifically *prosocial content* that matters, not narrative-as-such.

Ablations exist to cut between these. This document specifies the **sufficiency / transplant** ablation, the strongest single lever for explanation (1).

Sibling designs in the ablation family (documented separately):
- **Necessity / lesion** — run normally, remove myths at round K; does cooperation decay?
- **Specificity / content** — manipulate what is in the channel (scrambled / valence-flipped / filler); which property carries the effect?

---

## 2. Why the noise condition (and only the noise condition)

The 2026-02-26 runs showed Claude defaults near-ceiling cooperation (~0.95 trust ratio) with no noise. A ceiling-locked baseline cannot discriminate between conditions — every seed type would look identical near the max. The **negative-noise condition** (`noisy_negative_5`, uniform negative noise on both sent and returned, range 5) is the only setup in which Sonnet 4.5 shows a usable cooperation range.

**Pre-run check (passed).** Of the existing v4 negative-noise Sonnet 4.5 runs:
- Joint cumulative balance spans **50.6 – 81.4** (n = 60 myth-bearing runs across `game_myth` and `myth_game`, both informed and uninformed pooled). Effective ceiling under −5 noise ≈ 100. Runs sit comfortably off the ceiling.
- Mean **62.9 ± 6.4**; median 62.5.
- Top 25% threshold: ≥ 67.3 (15 runs).
- Bottom 25% threshold: ≤ 58.5 (16 runs).
- Top vs bottom means: ~71 vs ~56 → ~15-point gap, ~2.3 std separation.

Both source pools comfortably exceed the 5 distinct runs needed.

---

## 3. Core idea and logic

In a normal run, when an agent decides how much to send in round *t*, its memory holds two things: (a) the **record of game outcomes so far** ("I sent 5, they returned 6…") and (b) the **myths** it and its partner have written. This ablation strips (a) and keeps (b). The game is always played — that is what produces the cooperation score — but **what the agent knows when it plays** is varied.

The transplant logic: take a myth *out of one run* and make it the *entire inheritance* of naive agents in a new run.

- If cooperation **reproduces** → the myth was a *sufficient carrier* of the cooperative disposition, independent of the behavioral history that produced it. Strong support for (1).
- If cooperation **collapses to baseline** → the myth was *insufficient / epiphenomenal*. Support for (2).

This is a causal claim that correlational analysis of the main runs cannot deliver.

---

## 4. Research questions and hypotheses

**RQ1 (sufficiency).** Can a transplanted myth, as an agent's sole inheritance, induce cooperation in naive agents?

**RQ2 (content-specificity).** Is the effect carried by *cooperative narrative content*, or by mere presence of text / extra generation?

**RQ3 (cultural refinement).** Do myths *accumulate* cooperative carrying-capacity over a run — i.e., is a late-run myth a better carrier than an early-run myth?

**RQ4 (self-sufficiency vs trajectory-nudge).** Does the myth generate cooperation on its own, round after round, or does it merely put a normally-playing system on a better trajectory? *(the M1 / M2 fork, §6)*

| # | Hypothesis | Predicted pattern | Adjudicates |
|---|------------|-------------------|-------------|
| H1 | Myths are sufficient carriers | myth seed > no-seed baseline | (1) vs (2) |
| H2 | Effect is content-specific, not warm-up | Real-myth seed > length-matched filler seed | (4) vs (3) |
| H3 | It is *cooperative* content specifically | End-myth from high-coop run > end-myth from low-coop run | (4) vs (2) |
| H4 | Myths refine over a run | End-myth (high-coop) > start-myth | RQ3 |
| H5 | Graded dose-response | Independently-scored myth cooperativeness correlates with resulting game cooperation | strengthens (4) |

A **null on H1** (real myth ≈ no-seed) is itself a clean, publishable result: it pushes the project toward the *readout* interpretation.

---

## 5. Conditions and factors

Two active factors (Seed × Memory regime). Society is fixed at dyadic for Phase 1.

### Factor A — Seed (what the agent inherits)
- **S-start** — a myth from the *start* (round 1) of a prior run. Drawn from `myth_game` round-1 myths only (no game history baked in).
- **S-end+** — a myth from the *end* (round 10) of a *high-cooperation* prior run (top 25%, ≥ 67.3 joint).
- **S-end−** — a myth from the *end* (round 10) of a *low-cooperation* prior run (bottom 25%, ≤ 58.5 joint). **Control for selection / surface-feature confounds.**
- **S-filler** — length-matched non-narrative text in the seed slot. **Control for warm-up / extra-tokens.**
- **S-none** — no seed; standard repeated game with empty initial memory. **Baseline.**

The research log specifies S-start and S-end+ only. The other three are load-bearing: without **S-end−**, a positive end-myth result cannot be distinguished from "end-of-run myths are simply longer and better-written" (selection confound); without **S-filler**, it cannot be distinguished from warm-up (3 vs 4).

### Factor B — Memory regime — running both, M1 primary
- **M1 — myth-only / frozen.** Memory is wiped to the seed slot each round. The agent never accumulates a record of game outcomes. Tests whether the myth is a **self-sufficient generator** of cooperation. *(primary, more decisive)*
- **M2 — myth-prior / seeded-start.** Memory is seeded at construction; normal `memory_capacity=3` truncation runs; game history accumulates. The seed naturally scrolls out of working memory after ~3 rounds. Tests whether the myth puts a normally-playing system on a **better trajectory**. *(ecological comparison)*

### Factor C — Society — fixed at dyadic for Phase 1
- **Soc-dyad — N = 2.** Society / re-pairing is deferred (§13).

### Replication
**5 reps per myth-seeded cell, each rep a different myth drawn from a different source run.** Drawing 5 distinct myths from 5 distinct source runs avoids pseudo-replication. S-filler / S-none reps are simple replicates (filler re-sampled per run; S-none is identical replicates with different decoding seeds — variance comes from temperature).

---

## 6. Memory implementation — the load-bearing detail

The 2026-02-26 advisor flag: M1 is uninterpretable if the seed scrolls out of context. This section pins down exactly how the seed stays present.

### The truncation rule (already in `src/agents.py:58–60`)
```python
if len(self.messages) > self.memory_capacity * 2 + 1:
    self.messages = [self.messages[0]] + self.messages[-(self.memory_capacity * 2):]
```

`messages[0]` (the system prompt slot) is preserved automatically.

### Memory-injection (not system-prompt injection)

The seed enters the agent as a **fake prior assistant message**, not as part of the system prompt. Rationale: in the source runs the myths existed as assistant messages in memory. Elevating them to system level changes the model's semantic register (system = standing instructions; assistant message = something I once wrote). A system-prompt seed would confound "myth carries cooperation" with "elevating myth to system rule made it more authoritative."

Initial `messages` at construction (for any seeded condition):
```
[
  {"role": "system",    "content": <normal game system prompt for this condition>},
  {"role": "user",      "content": <generic myth elicitation: "Write a short story.">},
  {"role": "assistant", "content": <SEED MYTH TEXT (or length-matched filler)>},
]
```

The fake user prompt is the **`myth_writing_default`** template from `config/experiments.yaml` with `myth_topic="anything"` — the actual prompt the source-run agents saw at round 1. It is neutral, the `"anything"` topic permits any myth content (including round-10-style cooperation-themed myths), and the cost over a stripped 4-word prompt is zero.

**It is a controlled fixture, not a provenance claim.** For S-end+ / S-end− seeds the source-run round-10 elicitation was actually `myth_writing_later_rounds` (which references prior myths the seeded agent does not have, so it cannot be reproduced faithfully). Using `myth_writing_default` uniformly across all seed types means the conversational frame `[user asks for myth] → [assistant produces myth]` is plausible and identical across cells. The seed *myth text* is the experimental signal — the eliciting prompt is just the message structure that makes the assistant message land in memory naturally.

### Critical: the round-N user prompt itself leaks game history

The configured trust-game prompts `trust_game_later_investor` and `trust_game_later_trustee` embed last-round game state (`{last_round_sent}`, `{last_round_returned}`, `{last_round_*_payoff}`, `{agent_balance}`) **inside the user prompt text**. So wiping `self.messages` between rounds is necessary but not sufficient — under those templates the agent would still see 1 round of game history every round, baked into the prompt itself. M1 would not actually be myth-only.

**Fix:** under M1, use **round-1 templates every round** (`trust_game_round1_investor` for the investor, `trust_game_round1_trustee` for the trustee). These reference only the current round's state — they do not contain any `last_round_*` fields. The trustee's round-1 template legitimately needs `{sent}` (the current round's investor send) because the trustee can't act without knowing what they were sent; that is current-round state, not history. The investor's round-1 template references only `{endowment}`.

Concretely, under M1 the round-N user prompt for an investor reads `"Round 1: You have $5. How much do you send? (0-5)"` — same string every round. The agent loses sense of round progression, which is exactly the intended condition (no memory of progression).

A custom M1 template like `"Round N. You have $5. How much do you send? (0-5)"` is acceptable as an alternative — preserves round number, still no history. **Pick one and pin it.**

### Mode-specific behavior

| Mode | Initial messages | Truncation behavior | Round-N prompt template |
|---|---|---|---|
| **M1** (myth-only) | `[system, fake-user, seed-myth]` | Before each `respond()`: `self.messages = self.messages[:3]` | **round-1 templates every round** (no history fields) |
| **M2** (seeded-start) | `[system, fake-user, seed-myth]` | Normal `memory_capacity=3`; seed scrolls out after a few rounds | Normal: round-1 in round 1, later templates from round 2+ |
| **S-none, M2** | `[system]` only | Normal | Normal templates (standard baseline) |
| **S-none, M1** | — | *Excluded.* Degenerate — agent has no memory of anything. | — |

### Implementation footprint

- `src/agents.py` — `Agent.__init__` accepts `memory_mode: Literal["normal", "m1"] = "normal"`. In `respond()`, before the existing truncation block, add:
  ```python
  if self.memory_mode == "m1":
      self.messages = self.messages[:3]  # keep system + seed exchange
  ```
- `src/simulation.py` — pass `memory_mode` through; accept `seed_myth: Optional[str]` and a fake user prompt; if present, append the user/assistant pair after the system prompt at agent construction (currently lines 173–174).
- `games/trust_game.py` — when an agent's `memory_mode == "m1"`, dispatch to the round-1 templates for every round, never the `_later_*` templates.
- New `experiments/run_ablation.py` — loads harvested seeds from `data/json/.../` source runs, dispatches the cell sweep with appropriate `memory_mode` and `seed_myth` per cell.
- New `scripts/harvest_seeds.py` — pulls 5 myths per pool per the rules in §11, writes a manifest.
- Analysis script extensions for trajectory + threshold testing.

Agent + simulation + game changes: ~50 LoC. Runner + harvest + analysis: 150–300 LoC. Total ~200–350 LoC.

### Fake user prompt

`myth_writing_default` template with `myth_topic="anything"`. See the rationale block above (Memory-injection section) for the full justification — this is a controlled fixture used uniformly across all seed types, not a claim of faithful provenance for S-end seeds.

### Mandatory verification (gate to launching the full sweep)

On a single M1 × S-end+ smoke run (one rep, 10 rounds), the `call_LLM` debug log (added 2026-05-11) must show **three** things for round 10. **Inspect the full message list as sent to the API, not just the most recent user prompt** — the failure mode is "seed scrolled out of memory", which only shows up by reading the whole list.

1. The full message list at the API call is exactly `[system, fake-user, seed-myth-assistant, round-10-user]` — length 4, no more.
2. The seed myth text appears verbatim in `messages[2]` (the assistant slot).
3. The round-10 user prompt (`messages[3]`) is the **round-1 template** for the agent's current role — no `last_round_*` substrings.

If all three → mechanism sound, launch. If any fails → mechanism is broken, stop.

---

## 7. Run plan (Phase 1, locked)

| Memory regime | Seeds swept | Cells | Runs (×5 reps) |
|---|---|---|---|
| **M1** (myth-only) | S-start, S-end+, S-end−, S-filler | 4 | 20 |
| **M2** (seeded-start) | S-start, S-end+, S-end−, S-filler, **S-none** | 5 | 25 |
| **Total** | | **9** | **45 runs** |

Each run: Sonnet 4.5, N=2, 10 rounds, both agents seeded identically, negative-noise (`noisy_negative_5`), uninformed (matches the source-pool selection; see §11 for justification), **`task_order = ["game"]`** (no myth-writing during host play — the seed is the entire myth memory the agent ever has; under M1 any newly-written myths would be wiped on the next round anyway).

Plus: independent cooperativeness scoring of every distinct seed myth used (≈ 15 distinct myths total: 5 start + 5 end+ + 5 end−) for the H5 calibration analysis.

**Estimated cost:** 45 runs × ~20 LLM calls/run (10 rounds × 2 agents, game-only) × Sonnet 4.5 (≈ $0.003 input + $0.015 output per ~1k tokens) ≈ **$5–10 host runs**. LLM-judge scoring of 15 seed myths: an additional **~$0.50–1**. **Total: under $12**.

---

## 8. Measures (dependent variables)

**Primary — cooperation.**
- Amount **sent** and amount **returned** per round (direct readout of the cooperative act). Track per-round series, not just final.
- **Mean cumulative balance** (joint / total). Valid as a cooperation proxy *because agents are symmetric and identically seeded*: with the multiplier, only mutual cooperation grows the joint pie. Its blind spot is **asymmetry** — a cooperative agent exploited by its partner shows *low* balance despite cooperating — which the raw sent/returned series exposes. **Report both.**

**Trajectory.** Cooperation over the 10 rounds: does it *hold* or *decay*? Especially diagnostic for M1 — a seed that produces a cooperative opening but cannot sustain it is a different finding from one that holds.

**Calibration (the verbalized-vs-actual bridge, H5).** Independently score each seed myth for cooperativeness/prosociality (LLM-as-judge) and correlate that score with the resulting game cooperation. Converts the binary controls into a graded dose-response and links cleanly to the project's central "verbalized vs actual strategies" question. Natural interface to Mario / Arabella's linguistic analyses.

---

## 9. Pre-specified analysis plan

### Pre-specified "reproduces" thresholds

Source-pool σ (6.4) is computed on `myth_game + game_myth` runs that had full memory machinery and active myth-writing. The host runs here have stripped memory and game-only task order, so their within-cell variance may differ. The thresholds below are **provisional** — they will be re-anchored after a pilot variance cell (see §9.3) before final scoring.

Provisional thresholds (using source-pool σ = 6.4 as a stand-in):

| Outcome | Criterion |
|---|---|
| **H1 (sufficiency) reproduces** | `mean(S-end+) − mean(S-none) ≥ +1 σ_host` (≈ +6 joint at source σ) |
| **H1 null** | `|mean(S-end+) − mean(S-none)| ≤ 0.5 σ_host` AND CIs overlap S-none |
| **H2 (content-specificity)** | `mean(S-end+) − mean(S-filler) ≥ +0.75 σ_host` |
| **H3 (cooperative-content)** | `mean(S-end+) − mean(S-end−) ≥ +0.75 σ_host` |
| **H4 (refinement)** | `mean(S-end+) − mean(S-start) ≥ +0.5 σ_host` |
| **H5 (graded)** | Spearman ρ between independently-scored myth cooperativeness and resulting joint balance, on all ~15 myth-bearing host runs. Report with 95% CI. **Hypothesis-generating only** at this n. |

These thresholds are pre-registered with the substitution rule: replace σ in each formula with the **pilot host σ** (§9.3) once that's measured. The substitution rule is fixed; the number plugged in is data-derived but doesn't change the criterion structure.

### Decision rules
- If H1 reproduces AND H2 AND H3 hold → strong support for **(1) causal seed** with content specificity.
- If H1 reproduces but H2 fails (filler matches real myth) → support for **(3) warm-up / any-text effect**.
- If H1 reproduces, H2 holds, H3 fails (S-end+ ≈ S-end−) → **positive evidence for "narrative form, not specifically cooperative content, matters"**: end-myths from low-coop runs work as well as end-myths from high-coop runs, so the carrier property is something other than the myth's cooperativeness (e.g., genre, abstractness, length). This is a *substantive* result, not an undiscriminated case — frame it as such. **Caveat:** this reading depends on §11's pre-registered text-feature comparison showing the top and bottom source pools actually differ on cooperativeness-related content. If the pools are content-indistinguishable on that dimension, H3 was never a fair test and the result is undiscriminated rather than positive.
- If H1 null → support for **(2) epiphenomenal / readout**. Publishable.

### 9.3 Pilot variance cell — re-anchor σ before scoring

Run the S-none × M2 cell first (5 reps). This is the closest thing to a "host control" — no seed, game-only, identical decoding distribution to the rest. Compute σ from these 5 runs (call it σ_host). Plug σ_host into the threshold formulas above. Then run the remaining 8 cells.

If σ_host is wildly different from source-pool σ (say > 2× or < 0.5×), pause and reconsider — that signals either a setup bug or the host condition is so different that comparison to source pool is fraught. **This is a heuristic check, not a statistical test:** with n=5, the sample σ has roughly a 0.6×–2.2× CI around the population σ even when nothing is wrong, so don't let a single near-boundary ratio block launch — use it to catch setup bugs and gross condition differences, not as a hard gate.

### Power realism
n = 5 reps per cell. MDE at α=0.05 under a two-sample t-test ≈ **1.35 × σ_host**. Phase 1 is a **direction-finding pilot**, not powered for nulls. If results are equivocal, Phase 2 ups n based on observed within-cell variance.

---

## 10. Inference: what each contrast buys

| Contrast | Conclusion if positive |
|---|---|
| **S-(any real myth) vs S-none** | The myth does something at all (H1) |
| **S-(real myth) vs S-filler** | It is narrative/content, not extra tokens (H2) |
| **S-end+ vs S-end−** | It is *cooperative* content specifically (H3) — defuses the selection confound |
| **S-end+ vs S-start** | Myths accumulate cooperative carrying-capacity (H4) |
| **M1 vs M2** | Whether the myth is self-sufficient or a trajectory-nudge (RQ4) |
| **Graded myth-score → cooperation** | Dose-response (H5) |
| *Soc-dyad vs Soc-pop* | *Deferred (§13)* |

---

## 11. Source-run harvesting

Seeds are harvested from existing `noise_experiments/v4_direct_provider/noise_negative_mem3_claude_sonnet_45/claude-sonnet-4.5/{game_myth,myth_game}/noisy_negative_5/` runs.

- **Cooperation metric for ranking source runs:** joint (total) cumulative balance across the run's two agents (`game_data.balances` summed).
- **S-end+ pool:** round-10 myths from runs in the **top 25%** (≥ 67.3 joint). 15 source runs available; draw 5.
- **S-end− pool:** round-10 myths from runs in the **bottom 25%** (≤ 58.5 joint). 16 source runs available; draw 5.
- **S-start pool:** round-1 myths from `myth_game` runs only (no game history baked in). Draw 5.
- **Per seed type:** 5 distinct myths from 5 distinct source runs. **Use `Agent_1`'s myth from each source run** — deterministic, avoids accidental selection on individual balance, preserves independence across reps. (Documented choice; an alternative would be random-with-seed, but Agent_1 is simpler and avoids a parameter that needs disclosing.)

### S-filler content

Length-matched filler is drawn from **Simple English Wikipedia first paragraphs of country / animal / chemical-element articles** (e.g., "Belgium", "Octopus", "Carbon"). Rationale: factual encyclopedia prose is recognizably non-narrative, neutral valence, no normative content, no game-relevant vocabulary. Match token length to the seed-myth distribution within ±10%. Pin the source (`simple.wikipedia.org`, top of article) and record the article slug used per rep so the choice is reproducible.

**Selection condition for source pool.** All source runs are `noisy_negative_5` *uninformed* + *informed* pooled — the cooperation distribution and myth distributions look statistically similar across informed/uninformed (means 61.1 / 63.1 in game_myth; 63.5 / 63.8 in myth_game). For host runs, run uninformed only (matches the main analysis budget).

**Pool-level text-feature pre-registration (item 7 from advisor).** Before drawing seeds, summarize per-pool myth-text properties: token length, Brysbaert mean concreteness, naive LLM-as-judge cooperativeness. If top-25% and bottom-25% pools differ systematically on length or concreteness, document this as a confound on H3 (S-end+ vs S-end− may be partly carried by surface features, not content). Drawing 5 myths with **length matched within ±20%** between the top and bottom pools is a partial mitigation; record whether matching was possible.

---

## 12. Confounds and limitations

- **Myth decays out of context (M1).** Handled by `memory_mode="m1"` + verification on the smoke run (§6). If the smoke run fails verification, do not launch.
- **Selection confound on end-myths.** S-end+ is selected on cooperation, so a positive S-end+ alone cannot distinguish "carries the norm" from "is simply better-written / longer / more abstract." S-end− is what disambiguates. The pre-registered text-feature comparison (§11) bounds the surface-feature contribution.
- **Length confound.** S-filler is length-matched to the myth distribution.
- **System-register confound.** Avoided by memory-injection rather than system-prompt injection (§6).
- **Symmetric seeding.** Both agents are seeded identically (locked). One-sided seeding (only one agent gets the myth) is a different experiment and is explicitly out of scope. **Limitation:** Phase 1 cannot distinguish "individual norm" from "shared norm" effects.
- **Within-pair text coupling.** Both agents receive *the same myth string*, which is a stronger anchor than the original runs (where the two agents wrote different myths). This is a deliberately clean condition, not an ecological replica. A positive effect under this strong coupling does not automatically imply the same effect under naturally divergent myths; flagged as a scope condition rather than a confound.
- **Statistical power.** 5 reps/cell. MDE ≈ 1.35 σ ≈ 8.6 joint. Treat Phase 1 as direction-finding; publication-grade N is set by observed variance.
- **Model generality.** All runs are Sonnet 4.5; results may be model-specific. Cross-model replication is a Phase-2 question.
- **10 rounds is short for trajectory analysis (H4).** Fine for pilot. If H4 effect size is borderline (< 0.75 σ), Phase 2 extends to 20 rounds.
- **Eliciting prompt as a fixture, not provenance.** For S-end seeds, the actual source-run round-10 elicitation was `myth_writing_later_rounds`, which references prior myths the seeded agent does not have. We use `myth_writing_default` uniformly across all seed types instead. This is justified because (a) the ablation runs are not compared to source runs at the cooperation-score level — the in-design control is S-none × M2, not the source pool — and (b) uniform conversational framing across cells is the property that prevents prompt-level confounds. The cost is that the seed presents to the host agent as a round-1-style elicitation regardless of when it was actually written; the seed *text* is unaffected. If the smoke run reveals models reacting weirdly to the framing (e.g., querying the prompt, generating meta-commentary), revisit.

---

## 13. Resolved decisions

| # | Decision | Choice |
|---|---|---|
| 1 | Memory regime scope | **Both** M1 and M2; M1 primary |
| 2 | Society now or later | **Dyadic only** for Phase 1; society deferred |
| 3 | Seeding symmetry | **Both agents seeded** identically; one-sided seeding out of scope |
| 4 | High-cooperation cutoff (S-end+) | **Top 25%** of source pool by joint cumulative balance (≥ 67.3) |
| 4b | Low-cooperation cutoff (S-end−) | **Bottom 25%** (≤ 58.5) |
| 5 | Rounds per run | **10** |
| 6 | Model | **Claude Sonnet 4.5** |
| 7 | Noise condition | **`noisy_negative_5`** (uniform negative noise, range 5, both sent and returned) |
| 8 | Inform-about-noise (host) | **Uninformed** |
| 9 | Anti-cooperative myth (S-anti) | **No** |
| 10 | Seed source | **Pre-existing v4 negative-noise Sonnet 4.5 runs** |
| 11 | S-start source | **`myth_game` round-1 myths only** (no game history baked in) |
| 12 | Source pool — informed pooled with uninformed | **Yes** (similar distributions; widens pool) |
| 13 | Seed mechanism | **Memory-injection** (assistant-message at `messages[2]`), not system-prompt |
| 14 | Fake user prompt | **`myth_writing_default` template with `myth_topic="anything"`** — the actual prompt the source-run agents saw |
| 15 | Pre-registered reproduction threshold | **+1 σ_host over S-none** (σ_host measured from pilot S-none × M2 cell, §9.3) |
| 16 | Host task order | **`["game"]` only** — no myth-writing during host play |
| 17 | M1 round-N prompt template | **Round-1 templates every round** (no `last_round_*` fields) |
| 18 | Which agent's myth to harvest | **`Agent_1`'s** from each source run (deterministic) |
| 19 | S-filler content | **Simple English Wikipedia first paragraphs** (country/animal/element), length-matched ±10% |
| 20 | Pilot variance cell | **S-none × M2 first**, compute σ_host before scoring remaining cells |

---

## 14. Future work (out of scope for Phase 1)

- **Large society (Soc-pop).** Population of N > 2 agents re-paired each round, so a seeded norm can *spread*. Requires a pairing layer the codebase does not yet have, and adds **diffusion** measures (does cooperation / myth content propagate from the seeded starting point or stay local?). Re-runs the same Seed × Memory-regime sweep at N > 2.
- **Necessity / lesion** and **specificity / content** ablations — sibling designs in the ablation family.
- **One-sided seeding** — only one agent gets the myth. Distinguishes "shared norm" from "individual norm" effects.
- **Cross-model generalization** — replicate on Gemini 3 Pro and GPT-5 to test whether the carrier effect is Sonnet-specific.
- **20-round Phase-2** — if trajectory analysis (H4) is borderline at 10 rounds.

---

## 15. Launch sequence (operational)

1. **Implement** `memory_mode` + seed injection + M1 prompt-template dispatch (§6 footprint).
2. **Harvest seeds** — `scripts/harvest_seeds.py` pulls 5 myths per pool by the rules in §11. Output a JSON manifest pinning seed text → source run → joint balance → text-feature summary.
3. **Pre-register text features** — record top vs bottom pool length, Brysbaert concreteness, naive cooperativeness stats before launch. If pools diverge sharply, flag in the writeup.
4. **Smoke run** — one M1 × S-end+ rep. Verify in `call_LLM` log: (a) round-10's prompt contains seed myth verbatim, (b) round-10's prompt is the **round-1 template** (no `last_round_*` fields). Both must hold.
5. **Pilot variance cell** — run S-none × M2 (5 reps). Compute σ_host. Sanity-check against source-pool σ (6.4). If wildly off, pause and reconsider.
6. **Plug σ_host into pre-specified thresholds** (§9.3).
7. **Launch remaining Phase 1 cells** — 8 cells × 5 reps = 40 runs.
8. **Score seed myths** with LLM-as-judge (H5).
9. **Analyze and write up** against §9 pre-specified thresholds.

If step 4 fails verification, stop — fix mechanism before any further runs.
