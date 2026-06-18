# Memory-Transplant Ablation — Design Doc

**Are myths sufficient causal carriers of cooperation?**

Owner: Ivar · Workstream: "Ablations of agents who have lost their memories"
Status: Phase 1 — complete (see §16) · Scope: dyadic. Soc-pop is now technically supported in the codebase (see banner) but is out of scope for this doc.

> **2026-06-18 update.** The multi-agent dyadic-pairing layer (`games/dyadic_pairing.py`, `DyadicPairingMixin`) was merged from `origin/codex/push-current-changes-20260601`. The toolkit now supports arbitrary even-N agent pools with role-balanced random pairing per round, unique display names, and anonymous variants — i.e., the "Soc-pop" future-work item in §14 is no longer blocked on infrastructure.
>
> The Phase 1 ablation-specific code (`memory_mode="m1"`, `seed_myth` / `seed_user_prompt` plumbing in `src/agents.py` and `src/simulation.py`, the M1 round-1-template dispatch in `games/trust_game.py`, the `experiments/run_ablation.py` cell dispatcher and `scripts/harvest_seeds.py`) was **rolled back** as part of that merge. The Phase 1 results in §16 stand as a historical record. To re-run the ablation under the new multi-agent codebase, the §6 / §15 footprint would need to be re-applied on top of the dyadic-pairing architecture (see "Re-implementation notes" at end of §6).
>
> **Phase 2 design is locked in §17**, calibrated to be comparable to the `sonnet45_8agent_myth_directive_history3_anon_r10_n5` baseline (Claude Sonnet 4.5, 8 agents, `history_policy="self_and_coplayer"` windows 3/3, anonymous co-players, directive myth arm). §§1–16 below describe the Phase 1 work as designed and run; §17 supersedes §§5, 7, 11, 13, 15 for any future runs.

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
- **Soc-dyad — N = 2.** Society / re-pairing is deferred (§13). *(2026-06-18: the pairing layer now exists in `games/dyadic_pairing.py`; deferral is now a scope choice for this ablation, no longer an infrastructure constraint.)*

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

> **Historical.** This footprint was applied during Phase 1 and rolled back in the 2026-06-18 multi-agent merge. See "Re-implementation notes" below if Phase 2 is run on the current codebase.

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

### Re-implementation notes (post 2026-06-18 multi-agent merge)

If Phase 2 of the ablation is run on the current codebase, the footprint above needs three adjustments because `TrustGame` now inherits from `DyadicPairingMixin`:

- `Agent.__init__` no longer takes `memory_mode`. Re-add it the same way, but the agent now also carries `display_name` / `population_role` / `interaction_history` — the wipe `self.messages = self.messages[:3]` is still correct (those attributes are not in `self.messages`).
- `src/simulation.py` no longer has the `_format_initial_system_prompt` / `_replace_agent_system_prompt` helpers. The "myth-first blind variant" wiring (`initial_system_prompt_template`, `switch_to_game_system_before_game`) was used to defer applying the game system prompt until after the first myth was written; re-adding those helpers is straightforward but must coexist with the new `_build_agent_names` / `_configure_game_agents` / `_get_round_pairings` setup.
- `games/trust_game.py` round-1 dispatch under M1 must read from `sim_data.game_data["pending_sents"][pairing["dyad_id"]]` (the new dyadic key), not the legacy scalar `pending_sent`. With 2 agents there is exactly one dyad, so the dict has one key; the rest of the logic is unchanged.

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

**Locked thresholds** (σ_host = **3.47**, measured from the pilot S-none × M2 cell, n=5, joint balances [50.88, 51.68, 52.80, 53.98, 59.64], mean = 53.80):

| Outcome | Criterion | Numeric |
|---|---|---|
| **H1 (sufficiency) reproduces** | `mean(S-end+) − mean(S-none) ≥ +1 σ_host` | ≥ +3.47 joint (S-end+ ≥ 57.27) |
| **H1 null** | `|mean(S-end+) − mean(S-none)| ≤ 0.5 σ_host` AND CIs overlap S-none | ±1.73 |
| **H2 (content-specificity)** | `mean(S-end+) − mean(S-filler) ≥ +0.75 σ_host` | ≥ +2.60 |
| **H3 (cooperative-content)** | `mean(S-end+) − mean(S-end−) ≥ +0.75 σ_host` | ≥ +2.60 |
| **H4 (refinement)** | `mean(S-end+) − mean(S-start) ≥ +0.5 σ_host` | ≥ +1.73 |
| **H5 (graded)** | Spearman ρ between independently-scored myth cooperativeness and resulting joint balance, on all ~15 myth-bearing host runs. Report with 95% CI. **Hypothesis-generating only** at this n. | — |

**σ_host vs source-pool σ.** σ_host / σ_source = 3.47 / 6.4 = 0.54 — inside the heuristic [0.5×, 2.0×] sanity band. The host condition is less variable than the source pool, plausibly because host runs have `task_order=["game"]` (no myth-writing during play) so the myth-writing variability that contributes to source-pool σ is absent. **Pilot S-none mean = 53.80** is also notably below source-pool mean (62.88); same explanation — the myth-writing in source runs was itself contributing ~9 joint to cooperation. This is informative for interpretation: the seed needs to lift host cooperation above 57.27 (≥1 σ_host above 53.80) to support sufficiency.

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

- **Large society (Soc-pop).** Population of N > 2 agents re-paired each round, so a seeded norm can *spread*. ~~Requires a pairing layer the codebase does not yet have~~ **Now supported** by `games/dyadic_pairing.py` (`DyadicPairingMixin`, merged 2026-06-18): even-N pools, role-balanced random pairing per round, unique display names with anonymous variants, defector subsets. Existing 8-agent experiment sets in `config/experiments.yaml` (`sonnet45_8agent_*`, `gemini31_flashlite_8agent_*`) demonstrate the wiring. Soc-pop adds **diffusion** measures (does cooperation / myth content propagate from the seeded starting point or stay local?). Re-runs the same Seed × Memory-regime sweep at N > 2 — note the seed-symmetry decision (§13 row 3) needs re-revisiting for N > 2: seeding *all* agents identically eliminates the diffusion question, seeding *one* agent isolates the spreading dynamic.
- **Necessity / lesion** and **specificity / content** ablations — sibling designs in the ablation family.
- **One-sided seeding** — only one agent gets the myth. Distinguishes "shared norm" from "individual norm" effects.
- **Cross-model generalization** — replicate on Gemini 3 Pro and GPT-5 to test whether the carrier effect is Sonnet-specific.
- **20-round Phase-2** — if trajectory analysis (H4) is borderline at 10 rounds.

---

## 15. Launch sequence (operational)

> **Historical.** Steps 1–9 were executed during Phase 1 and produced the §16 results. After the 2026-06-18 multi-agent merge, the ablation-specific code has been rolled back; step 1 would need to be re-applied as described in "Re-implementation notes" at the end of §6 before re-running.

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

---

## 16. Results — Phase 1 (Sonnet 4.5, negative noise, dyadic)

Three pivotal cells (`m1_s_none`, `m1_s_filler`, `m1_s_end_plus`) were scaled to **n=15** after an initial pass at n=5 revealed within-cell σ ~2-3× the pilot σ_host. All other cells remain at n=5. The added cell `m1_s_none` was a post-hoc isolation control to discriminate "wipe alone" from "wipe + any text"; not in the original §13 grid.

### Cell statistics

| Cell | n | Mean | Std | SE | 95% CI |
|---|---|---|---|---|---|
| **m1_s_none**       | **15** | **65.62** | 6.06  | 1.56 | [62.43, 68.81] |
| **m1_s_filler**     | **15** | **67.72** | 6.00  | 1.55 | [64.56, 70.88] |
| **m1_s_end_plus**   | **15** | **66.88** | 7.07  | 1.82 | [63.15, 70.60] |
| m1_s_start          | 5      | 69.21 | 3.61  | 1.61 | [65.77, 72.65] |
| m1_s_end_minus      | 5      | 67.64 | 8.66  | 3.87 | [59.39, 75.89] |
| m2_s_none           | 5      | 53.80 | 3.47  | 1.55 | [50.49, 57.10] |
| m2_s_start          | 5      | 54.82 | 3.11  | 1.39 | [51.86, 57.79] |
| m2_s_end_plus       | 5      | 53.34 | 3.68  | 1.64 | [49.83, 56.84] |
| m2_s_end_minus      | 5      | 56.43 | 10.50 | 4.69 | [46.43, 66.43] |
| m2_s_filler         | 5      | 56.29 | 6.53  | 2.92 | [50.07, 62.51] |

### Power-honest threshold outcomes

Pre-spec thresholds were calibrated to pilot σ_host = 3.47, but actual within-cell σ for myth-bearing cells is ~6–10. Contrasts below are reported with proper SE pooling and a z-statistic; "significance" is read against α=0.05 two-tailed (z ≥ 1.96).

**Regime effect — robustly large:**
- **H1 (m1_s_end_plus vs m2_s_none):** Δ = **+13.08**, SE = 2.40, **z = +5.46** ***. Memory regime moves cooperation by ~3.6 source-σ. Robust.

**Within M1 — content vs regime:**
- **H1 refined (m1_s_end_plus vs m1_s_none):** Δ = **+1.26**, SE = 2.40, z = +0.52. Null. Cannot distinguish "wipe + cooperative myth" from "wipe alone."
- **H2 (content vs filler, m1_s_end_plus vs m1_s_filler):** Δ = **−0.85**, SE = 2.39, z = −0.35. Null — and slightly *negative*.
- **wipe + filler vs wipe alone (m1_s_filler vs m1_s_none):** Δ = +2.10, SE = 2.20, z = +0.95. Small positive, not significant. Any-text contribution is bounded above ~4 joint.
- **H3 (cooperative vs anti-cooperative, m1_s_end_plus vs m1_s_end_minus):** Δ = −0.76, SE = 4.28, z = −0.18. Null. *Caveat:* m1_s_end_minus is still n=5.
- **H4 (refinement, m1_s_end_plus vs m1_s_start):** Δ = −2.34, SE = 2.44, z = −0.96. Start cells slightly higher; m1_s_start still n=5.

**M2 (normal memory):**
- All cells cluster 53–56. No contrast is significant against m2_s_none. The seed has no detectable propagated effect once game history accumulates.

### Trajectory

| Cell | Mean sent per round (r1 → r10) | Avg | Pattern |
|---|---|---|---|
| m1_s_none (n=15)    | 0.73, 0.70, 0.53, 0.98, 0.89, 0.69, 0.91, 1.20, 0.60, 0.57 | **0.78** | sustains |
| m1_s_filler (n=15)  | 0.52, 0.97, 1.25, 1.20, 0.95, 0.42, 0.86, 1.04, 1.17, 0.48 | **0.89** | sustains, peaks mid |
| m1_s_end_plus (n=15)| 0.86, 0.66, 0.82, 0.91, 1.03, 1.10, 0.43, 0.80, 0.97, 0.84 | **0.84** | sustains |
| m1_s_start (n=5)    | 0.67 → 0.92 | 0.96 | sustains |
| m1_s_end_minus (n=5)| 0.41 → 0.50 | 0.88 | sustains |
| m2_s_none (n=5)     | 0.48 → **0.00** | 0.19 | collapse |
| m2_s_start (n=5)    | 1.39 → **0.00** | 0.24 | collapse from highest opener |
| m2_s_end_plus (n=5) | 0.60 → 0.00 | 0.17 | collapse |
| m2_s_end_minus (n=5)| 0.54 → 0.10 | 0.32 | collapse |
| m2_s_filler (n=5)   | 0.97 → 0.19 | 0.31 | collapse |

The regime contrast is not "M1 averages 10 round-1s" — it's **M1 sustains moderate sending across all 10 rounds**, while **M2 collapses from a moderate opener to near-zero by rounds 6–10**. The destructive ingredient in M2 is the *interaction between noise and memory*: agents see "I sent $3, partner returned $0" (a noise-distorted record), and that history kills cooperation. Wiping the history rescues a sustained moderate-cooperation default.

### Within-data H3 (judge-score as IV, all 15 M1 myth-bearing runs)

The pre-spec H3 contrast (pool selection by source-run joint balance) turned out **not** to be a contrast on myth cooperativeness — the judge scored s_end_minus myths (7.60) *higher* than s_end_plus (6.60). The right within-data test:

- Median split on judge score (M1 myth-bearing, n=15): Δ(high − low) = **+1.84** (low ≈ 66.93, high ≈ 68.77). Tertile (≤3 vs ≥7): Δ = +2.50.
- Spearman ρ(judge cooperativeness, host balance) — M1 myth-bearing: **−0.019**. M2 myth-bearing: −0.478 (driven by a single outlier; fragile).

The within-data direction is mildly positive on M1 but well within within-cell noise; the negative M2 correlation is not robust.

### Judge scores per seed pool

| Pool | Mean | Std | Values |
|---|---|---|---|
| s_end_plus  | 6.60 | 2.07 | 3, 7, 7, 8, 8 |
| s_end_minus | 7.60 | 0.89 | 6, 8, 8, 8, 8 |
| s_start     | 3.60 | 2.79 | 1, 1, 3, 6, 7 |
| filler      | 0.00 | 0.00 | 0, 0, 0, 0, 0 |

The cooperation-by-joint-balance ranking (top 25% vs bottom 25% of source runs) **does not** match judge-rated cooperativeness of the myth text. Low-cooperation source runs produced *more* cooperation-themed myths, plausibly as moralizing/aspirational counter-current rather than descriptive narration.

### Interpretation

1. **The memory regime is the dominant explanatory variable.** M1 cells cluster 65–69; M2 cells cluster 53–56. The regime contrast is ~3.6 source-σ and z=5.46 — robust at any reasonable n.

2. **The trajectory tells the mechanism story.** M1 sustains moderate cooperation across all 10 rounds; M2 collapses from a moderate opener to near-zero. Under negative noise, memory of being burned actively destroys cooperation. Wiping that memory rescues a sustained moderate-cooperation default.

3. **No content-specific effect is detected.** At n=15, real cooperative myths, length-matched Wikipedia filler, and the empty-memory control all sit within 2 points of each other (65.6–67.7), 95% CIs heavily overlapping. The any-text contribution over wipe-alone is bounded above ~4 joint at this n.

4. **H3 was not a fair test of content-specificity at the level we framed it.** The pool selection (top vs bottom of source-run joint balance) does not produce a contrast on the actual cooperativeness dimension of the myths — judge scores invert that ordering. The within-data judge-score split is a fairer test and gives a small positive direction (+1.84) consistent with the bounded content contribution but well within noise.

5. **The "myth content carries cooperation" hypothesis is not supported by transplant** at this power. A defensible weaker claim is consistent with the data: if myths contribute, it is on the order of 2-4 joint, indistinguishable from neutral filler text in the same memory slot.

6. **Unexpected finding worth following up:** Round-10 myths from *low-cooperation* source runs are judge-rated *more* cooperative than round-10 myths from high-cooperation runs. This inverts a naive "myths reflect game behavior" reading — agents under stress may write counter-current normative myths. Worth a separate analysis.

### Statistical power caveat

n=5 has SE ≈ σ/2.2 ≈ 3 (at within-cell σ ~6.5). The pre-spec thresholds (calibrated to σ_host = 3.47 from m2_s_none, the most stable cell) underestimated the within-cell variance for myth-bearing cells. The n=15 scale-up resolved this for the m1 content contrast; remaining n=5 cells (m1_s_start, m1_s_end_minus, all M2) carry wider CIs and their contrasts should be read with that caveat. A confirmatory Phase 2 with n=15 across all cells, plus a second judge model, would tighten the within-data H3 and bound the m1_s_start anomaly.

### Status against §9 decision rules

- H1 reproduces against the M2 baseline, but H1 refined (vs m1_s_none) is null and H2 is null → support for **(2) epiphenomenal / readout** of the seed text. The regime, not the content, carries the effect.
- The M1 lift is real and substantive, but its proximate cause is memory-wiping under noise, not seed content. This was not an outcome the original four-explanation taxonomy in §1 anticipated; it's an emergent fifth interpretation: **(5) the interaction of noise and accumulated memory destroys cooperation, and any intervention that breaks that accumulation (memory wipe) rescues a moderate baseline regardless of what fills the freed slot**.

### Total Phase 1 cost

≈ $14–16 (40 initial sweep + 5 pilot + 5 isolation + 30 scale-up + 20 judge calls).

### Open follow-ups

- Scale n=15 across the remaining cells (especially m1_s_start, m1_s_end_minus). m1_s_start at 69.21 ± 3.61 is the highest mean and may reflect a real stylistic effect (Markdown header vs "Myth:" prefix) or noise; n=5 cannot tell.
- Cross-judge with a different model (Opus 4.7 or GPT-5) to bound judge-noise on the s_end_plus / s_end_minus inversion.
- Investigate the counter-current myth finding directly: in source runs, do agents in low-coop runs write progressively more moralizing myths over rounds?
- **The noise-memory destruction effect (+13 joint over M2 baseline) is itself a substantive finding deserving its own study** — independent of the myth question.
- Re-run on the no-noise condition (per the 2026-02-26 ceiling-lock check, Claude saturates near-max). Would test whether the regime effect persists when memory has nothing to destroy.

---

## 17. Phase 2 — comparable to anon-history3-directive 8-agent baseline

**Calibrated 2026-06-18.** Phase 2 retargets the ablation onto the current "normal" baseline so seed manipulations can be read against a like-for-like no-seed control. The baseline is `sonnet45_8agent_myth_directive_history3_anon_r10_n5` (`config/experiments.yaml`): Claude Sonnet 4.5, 8 agents, 10 rounds, `task_order=["myth","game"]`, directive myth arm (`myth_writing_default_game_directive` / `myth_writing_later_rounds_directive`), `history_policy="self_and_coplayer"` with `self_history_window=3` and `coplayer_history_window=3`, `show_agent_names=false` (opponents addressed as "your current co-player"), neutral persona, 5 reps. Each rep is an 8-agent population playing 80 dyad-rounds (8 agents × 10 rounds, 4 dyads/round) with role-balanced random pairing — see `games/dyadic_pairing.py:_get_balanced_multi_agent_pairings`.

### 17.1 Why M1 is dropped in Phase 2

Under the new code, `_get_multi_agent_later_prompt` (`games/trust_game.py:226–260`) embeds the self+coplayer history block directly into the round-N user prompt text from `sim_data.conversation_history`. Wiping `agent.messages` to `[system, fake-user, seed-myth]` (the Phase 1 M1 mechanism) does not remove the history block — the agent sees its history3 record every round regardless. To make M1 work under N>2 history3 you would have to also strip the history block from the prompt, which diverges from baseline on the dimension the user explicitly wants held constant. Phase 1 §16 already established that the M1 lift was driven by noise+memory destruction, not myth content, so dropping M1 also costs nothing on the central "do myths cause cooperation" question. **Phase 2 is M2-only** (seeded-start; normal history3 progression). The old M1 footprint and verification protocol in §6 remain as a historical record.

### 17.2 Baseline distribution (n=5, computed 2026-06-18) and the ceiling problem

From `data/json/sonnet45_8agent_myth_directive_history3_anon_r10_n5/` (no-noise baseline):

| Statistic | Joint balance (all 8 agents) | Per-agent balance |
|---|---|---|
| n | 5 runs | 40 agents (5 × 8) |
| Mean (±std) | 584.80 (±20.72) | 73.10 (±4.63) |
| Range | [548, 596] | [60.5, 80.0] |
| Per-run joints | 548, 590, 594, 596, 596 | — |

**Ceiling check.** Endowment $5, multiplier 3, 80 dyad-rounds. Mutual-cooperation per dyad-round = investor sends $5 → trustee receives $15 → returns ≥$5 to equalize → pie per dyad-round = $15. Joint ceiling ≈ **$600**. Baseline mean 584.80 is **97.5% of ceiling**; four of five runs are at 590+. This is the same problem §2 flagged for Phase 1 (Claude saturating no-noise) — a positive seed manipulation has at most ~15 joint points of headroom.

**Resolution (2026-06-18, fork C):** Phase 2 adds negative noise to pull the host condition off the ceiling, mirroring §2's logic. The Phase 2 baseline is therefore *not* `sonnet45_8agent_myth_directive_history3_anon_r10_n5` directly — it is a new noisy variant (see §17.3 row 5 and §17.7). The existing 5 no-noise runs serve only as a ceiling-check reference; the operative baseline must be re-run with noise.

### 17.3 Locked decisions (Phase 2)

Forks A, B, C from the original §17.6 were resolved 2026-06-18; their outcomes are folded into this table.

| # | Decision | Phase 2 choice | Differs from §13 row |
|---|---|---|---|
| 1 | Memory regime | **M2 only** (seeded-start; normal history3 truncation). M1 dropped (see §17.1). | row 1 |
| 2 | Society / N | **N = 8** dyadic pool with balanced random pairing per round | row 2 |
| 3 | Rounds per run | **10** | row 5 (unchanged) |
| 4 | Model | **Claude Sonnet 4.5** | row 6 (unchanged) |
| 5 | Noise condition | **Negative noise added** (fork C). The no-noise baseline saturates at 97.5% of ceiling (§17.2). Phase 2 runs in a uniform-negative-noise condition tuned to bring the no-seed baseline into a usable cooperation range. Pilot is required to pick the noise level (§17.5). | row 7 |
| 6 | History policy | **`self_and_coplayer`**, `self_history_window=3`, `coplayer_history_window=3` | new |
| 7 | Agent names | **`show_agent_names: false`** (anon). Opponents rendered as "your current co-player". | new |
| 8 | Personas | **`neutral`** | new |
| 9 | Myth arm | **directive** (`myth_writing_default_game_directive` / `myth_writing_later_rounds_directive`) | new |
| 10 | Seeding symmetry | **All 8 agents seeded identically with the same myth string** (fork B). Preserves §13 row 3's sufficiency framing. Soc-one diffusion is a Phase 3 follow-up, not a Phase 2 cell. | row 3 (extended to N=8) |
| 11 | Reps per cell | **5** initial (matches baseline n). Scale to 15 on pivotal cells if pilot σ_host is small enough to require it. | row equivalent to §7 |
| 12 | Fake user prompt slot for seed injection | **`myth_writing_default_game_directive`** rendered with `topic_instruction` for `myth_topic="anything"` (i.e., `"You may choose any mythic setting, characters, or symbols."`). Updates §6's `myth_writing_default + topic="anything"` rule for prompt-structure parity with the directive baseline. | row 14 |
| 13 | Task order (fork A) | **Full matrix:** `["game"]`, `["game","myth"]`, `["myth","game"]`. Each seeded cell is replicated under all three task orders, with a matching no-seed baseline per task order. Mirrors the project's standard task-order sweep. | row 16 |

### 17.4 Source pool — retargeted

§11 harvested seeds from v4 N=2 negative-noise Sonnet runs. Phase 2 seeds must come from the **same population as the baseline** to keep "what kind of myth carries cooperation" interpretable. Since Phase 2 runs in a noisy condition (§17.3 row 5), seeds must come from the *noisy* baselines, not the existing no-noise 5-rep set:

- **Source:** the new noisy 8-agent anon-history3-directive baselines built in §17.5 step 2 (n=15 per task order). The `["myth","game"]` and `["game","myth"]` baselines both write myths in every round; the `["game"]` baseline writes no myths and so cannot itself serve as a myth source.
- **Cross-task-order seeding rule:** for each host task order, draw seeds from the **same task order's** baseline runs to keep the source-host pairing matched. `["game"]` host runs draw seeds from `["myth","game"]` baselines (the closest matched population — same model, conditions, noise; only the task order differs, and `["game"]` host runs have no myth-writing to compete with seeds anyway).
- **S-start:** round-1 myths from baseline source runs. Each agent's round-1 myth is unconditioned on a prior self-myth. Pool ≈ 15 reps × 8 agents = 120 candidate myths per task-order pool. Draw 5 distinct myths per host cell from 5 distinct source runs.
- **S-end+ / S-end−:** rank source runs by joint balance, then draw round-10 myths from top vs bottom runs. **With n=15 baselines (§17.5 step 2), top/bottom quartiles each have 3–4 source runs** — draw 5 round-10 myths from those quartile pools (one per host rep, allowing within-source-run agent variation).
- **S-filler:** unchanged from §11 (Simple English Wikipedia first paragraphs of country/animal/element articles). Length-matched ±10% to the round-1 directive-myth length distribution from the noisy baseline.
- **Per-agent selection within a source run:** no `Agent_1` privilege at N=8. Pick the agent whose round-10 myth's source-run-joint-rank is closest to the run's mean per-agent balance (deterministic, avoids accidental selection on either extreme). Document each draw in a manifest.
- **Pool-level text-feature pre-registration (item from §11)** still applies. Compare round-10 myths in top vs bottom source runs on token length, Brysbaert concreteness, and naive LLM-judge cooperativeness *before* drawing.

### 17.5 Run plan (Phase 2)

**Cell grid — per task order:**

| Cell | Seed | n (initial) |
|---|---|---|
| **S-none** | no seed; the noisy 8-agent anon-history3-directive baseline | 15 (also serves as the source pool for that task order's seeded cells) |
| **S-start** | round-1 myth from baseline source-run agents | 5 |
| **S-end+** | round-10 myth from highest-joint baseline source runs | 5 |
| **S-end−** | round-10 myth from lowest-joint baseline source runs | 5 |
| **S-filler** | length-matched Wikipedia first paragraph | 5 |

**Full Phase 2 grid (with the §17.3 row 13 task-order matrix):**

| Task order | S-none | S-start | S-end+ | S-end− | S-filler | Subtotal |
|---|---|---|---|---|---|---|
| `["game"]`        | 15 | 5 | 5 | 5 | 5 | 35 |
| `["game","myth"]` | 15 | 5 | 5 | 5 | 5 | 35 |
| `["myth","game"]` | 15 | 5 | 5 | 5 | 5 | 35 |
| **Total** | 45 | 15 | 15 | 15 | 15 | **105 runs** |

Plus a **noise-level pilot** (§17.5 step 1, ~5–10 runs) before locking the noise condition. Estimated total: ~115 8-agent runs × 10 rounds × 8 agents = 9200 agent-rounds. Sonnet 4.5 token cost at directive prompt size: order $20–40 for the full Phase 2 sweep.

**Pre-spec contrasts (per task order, mirroring §9):**
- **H1 (sufficiency):** S-end+ vs S-none — myth carries cooperation?
- **H2 (content vs warm-up):** S-end+ vs S-filler — narrative content or just extra tokens?
- **H3 (cooperative content):** S-end+ vs S-end− — *cooperative* content specifically?
- **H4 (refinement):** S-end+ vs S-start — myths accumulate carrying capacity?
- **H5 (graded):** judge-rated cooperativeness × resulting joint balance (Spearman ρ).

A new **H6 (task-order generalization)**: compare S-end+ − S-none across task orders. If the seed effect is task-order-invariant, that's strong evidence the carrier property is the myth itself, not the conversational scaffolding around it.

**Thresholds:** **do not lock** until the noisy baseline is run and σ_host is measured on the new condition. Phase 1's locked thresholds (§9) used σ_host = 3.47 from an N=2 negative-noise S-none × M2 cell; that number does not transfer to N=8 no-noise history3, and adding noise will shift it again. Run the noisy baselines first (step 2 below), compute σ_host on each task-order's 15-rep S-none, then plug into the threshold formulas. Carry forward Phase 1's framing: ≥1 σ_host = "reproduces", ≤0.5 σ_host with overlapping CI = "null".

**Phase 2 launch sequence:**

1. **Noise-level pilot.** Pick 2–3 candidate noise levels (e.g., `noisy_negative_3`, `noisy_negative_5`, `noisy_negative_8`, all uniform negative). Run 1–2 reps of `["myth","game"]` no-seed at each. Pick the level whose mean joint balance lands in [400, 540] (well off both ceiling 600 and floor 0; ~70–90% of the no-noise baseline). Lock that level for the rest of Phase 2.
2. **Build noisy baselines.** Run 15 reps each at the locked noise level for `task_order=["game"]`, `["game","myth"]`, `["myth","game"]`. These are S-none cells AND source pools for seeded cells. Configs: `sonnet45_8agent_<task_order>_directive_history3_anon_noisy_negative_<level>_r10_n15`. The `["game"]` baseline does not produce myths, so it cannot itself be a source pool — `["game"]` host cells draw seeds from the `["myth","game"]` baseline (§17.4 cross-task-order rule).
3. **Compute σ_host per task order** from the three 15-rep S-none cells. Pre-register the seed-cell thresholds.
4. **Harvest seeds.** Use the rules in §17.4. Produce a manifest mapping seed text → source run → task order → joint rank → text features.
5. **Pre-register text features.** Per-pool length / concreteness / naive cooperativeness summaries before drawing finals.
6. **Smoke run.** One M2 × S-end+ × `["game"]` rep. Verify the seed lands at `messages[1:3]` and the round-1 game prompt shows the seed text as a prior assistant turn. Verify the round-N prompt's history block reflects the actual game history (this is the new code's behavior, not the M1 wipe — the verification target differs from §6).
7. **Run remaining cells.** 4 seed types × 3 task orders × 5 reps = 60 seeded runs.
8. **LLM-judge scoring** of all distinct seed myths (≈ 45 myths across the three task orders × 3 seed types from the actual draws).
9. **Analyze** against pre-spec thresholds. Report per-task-order and pooled-across-task-orders.

If step 1's pilot fails to find a noise level in [400, 540], stop and reconsider — the regime may not have a usable window between ceiling and floor.

### 17.6 Results — Phase 2 (Sonnet 4.5, noisy_negative_5, 8-agent history3 anon directive)

**Run completed 2026-06-19.** Noise pilot picked `noisy_negative_5` (joint=482, 80% of ceiling). Three task-order baselines built at n=15. Seeded cells: 5 reps × 4 seed types × 3 task orders = 60 runs.

#### 17.6.1 Baseline distribution

| Task order | n | Joint mean (±std) | Range | Per-agent (mean ±std) |
|---|---|---|---|---|
| `["game"]`        | 15 | **335.53 (±32.04)** | [287, 384] | 41.94 (±6.53) |
| `["game","myth"]` | 15 | **400.53 (±28.88)** | [350, 476] | 50.07 (±5.88) |
| `["myth","game"]` | 15 | **457.57 (±29.79)** | [398, 527] | 57.20 (±7.45) |

σ_host = 28.88–32.04 across task orders. All baselines comfortably off ceiling ($600). Myth length distribution: mean=201.7 words (std=5.4) — directive prompt's 200-word target is hit consistently.

#### 17.6.2 Seed manifest

Drawn from the `["myth","game"]` noisy baselines (the only baseline with myths every round). 5 distinct seeds per type, sourced from quartile-split source runs by joint balance. Filler concatenated from Simple English Wikipedia paragraphs to reach ~200-token length match.

LLM-judge cooperativeness scores (Sonnet 4.5 self-judge, 0–10 scale, n=5 per type):
- **s_start** (round-1 directive myths from top quartile): 10.0, 10.0, 10.0, 10.0, 10.0 → mean **10.0**
- **s_end_plus** (round-10 myths from top quartile): 10, 9, 8, 7, 7 → mean **8.2**
- **s_end_minus** (round-10 myths from bottom quartile): 8, 8, 9, 9, 9 → mean **8.6**
- **s_filler** (Wikipedia paragraphs): 0, 0, 0, 0, 0 → mean **0.0**

**Phase 1's counter-intuitive finding replicates:** end-minus myths score *higher* than end-plus on cooperativeness, despite coming from lower-cooperation source runs. Agents in failing runs write *more* moralizing myths; agents in succeeding runs drift into game-rule descriptions.

#### 17.6.3 Cell-level joint balance (mean ±std, n=5 per seeded cell)

|              | `["game"]`            | `["game","myth"]`     | `["myth","game"]`     |
|---           |---                    |---                    |---                    |
| s_none (n=15)| **335.5 ±32.0**       | **400.5 ±28.9**       | **457.6 ±29.8**       |
| s_start      | 401.7 ±52.4 (+66.2)   | 448.4 ±46.1 (+47.9)   | 455.8 ±46.0 (−1.8)    |
| s_end_plus   | 380.9 ±41.5 (+45.4)   | 401.8 ±25.9 (+1.3)    | 388.2 ±42.4 (−69.4)   |
| s_end_minus  | 333.0 ±23.5 (−2.5)    | 374.6 ±18.5 (−25.9)   | 382.1 ±9.9 (−75.5)    |
| s_filler     | 335.2 ±10.0 (−0.3)    | 396.3 ±42.4 (−4.2)    | 424.6 ±16.8 (−33.0)   |

Δ vs s_none in parentheses.

#### 17.6.4 Pre-spec contrast outcomes (z-statistics, two-sample SE)

|                              | `["game"]`     | `["game","myth"]` | `["myth","game"]` |
|---                           |---             |---                |---                |
| **H1** (s_end+ vs s_none)    | z = **+2.23** ✓| z = +0.09 (null)  | z = **−3.39** (reversed) |
| **H1 (alt)** (s_start vs s_none) | z = **+2.66** ✓| z = **+2.18** ✓   | z = −0.08 (null) |
| **H2** (s_end+ vs s_filler)  | z = **+2.39** ✓| z = +0.25 (null)  | z = −1.79 (negative) |
| **H3** (s_end+ vs s_end−)    | z = **+2.25** ✓| z = +1.91 (marg.) | z = +0.31 (null) |
| **H4** (s_end+ vs s_start)   | z = −0.70 (rev.) | z = **−1.97** (rev.) | z = **−2.42** (rev.) |
| **H5** Spearman ρ(judge, joint) | **+0.525** | +0.411           | +0.153           |

(✓ = passes pre-spec ≥1 σ_host threshold with z ≥ 1.96.)

#### 17.6.5 Interpretation

1. **Seed manipulation produces large, robust effects.** Effect sizes routinely exceed 1 σ_host. The mechanism is real, not a marginal nudge. This is a substantively different regime from the noise+memory destruction story of Phase 1 §16.

2. **Task order is a primary moderator, not a side condition.** The same seed produces opposite-direction effects across task orders:
   - `["game"]`: most seeds lift cooperation (content matters; H1/H2/H3 all hold for s_end+).
   - `["myth","game"]`: only s_start produces a measurable effect (and it's null); all other seeds *suppress*. Mechanism: agents in baseline `["myth","game"]` write their own round-1 myth before play. Pre-injecting a foreign myth disrupts this self-consistency, suppressing cooperation below the baseline an agent would produce by writing its own first myth.
   - `["game","myth"]` sits between the two; only s_start lifts robustly.

3. **H4 is consistently reversed.** Round-1 directive myths (parable format, "# The Tale of...") outperform round-10 myths (game-rule descriptions, "Myth: In the Hall of Eternal Mirrors, eight wanderers received five golden coins each dawn..."). This contradicts the Phase 1 framing that end-of-run myths are *refined* carriers. Under directive prompting, agents progressively concretize myths into game-mechanic descriptions, which lose the narrative-carrying property. **Cooperation carrying-capacity decays over a run, not refines.**

4. **H5 dose-response is positive but content-type-specific.** Spearman ρ(judge, joint) is positive across task orders (+0.525 / +0.411 / +0.153). The relationship is non-linear: s_filler (judge=0) and s_end_minus (judge=8.6) produce similar joint outcomes despite very different judge scores. The judge captures one dimension of myth content; the carrier property is correlated with but not identical to it.

5. **The Phase 1 counter-current myth finding replicates.** End-of-run myths from *low*-joint source runs score higher on judge cooperativeness than end-of-run myths from *high*-joint source runs. The story: agents under cooperation stress write more aspirational myths; agents under cooperation success write more descriptive myths. The myths themselves are an inverse indicator of the run's cooperative trajectory.

6. **Open question for Phase 3.** What *specifically* about s_start makes it the best carrier? The judge can't distinguish s_start (10.0) from s_end_plus (8.2) by enough to explain the joint-balance gap (+66 vs +45 in game). Candidate properties to test: (a) parable format vs game-rule description; (b) length distribution; (c) presence of specific cooperation-related lexicon ("trust", "share", "reciprocate"); (d) round-1 myths' lack of game-history reference. A small follow-up: re-judge s_start vs s_end_plus with a different judge model (Opus 4.7 or GPT-5) to bound judge noise on the 10 vs 8.2 split.

#### 17.6.6 Phase 2 status against §17.5 decision rules

| Hypothesis | Direction | Strongest support |
|---|---|---|
| H1 (sufficiency)     | mixed | s_start in `["game"]` and `["game","myth"]` (z=+2.66, +2.18) |
| H2 (content matters) | task-order specific | strong in `["game"]` (z=+2.39); null/reversed elsewhere |
| H3 (cooperative content) | task-order specific | strong in `["game"]` (z=+2.25); marginal in `["game","myth"]` |
| H4 (refinement)      | **reversed** | round-1 myths > round-10 myths across all task orders |
| H5 (graded)          | weakly positive | ρ = +0.525 (game), +0.411 (game_myth), +0.153 (myth_game) |

**Headline claim, conservative phrasing:** Under noisy_negative_5 at 8-agent history3 anon directive, *some* myth-shaped intervention causes cooperation movements ≥1 σ_host. The carrier property is more subtle than "cooperativeness of the myth text" — it interacts with the agent's task-structure (free choice vs constrained injection) and the myth's narrative form (parable vs game-rule description). The Phase 1 §16 conclusion that the memory regime is the only mechanism does **not** generalize: under multi-agent dyadic history3, myth content does carry cooperation, but in ways that the original H1/H4 hypotheses partly missed.

#### 17.6.7 Cost

Total Phase 2 spend (~$255):
- Noise pilot (3 runs): ~$7
- Noisy baselines (45 runs): ~$104
- Smoke run (1): ~$2
- Seeded cells (60 runs): ~$138
- Judge scoring (20 myths): ~$0.50

#### 17.6.8 Follow-ups

- Re-judge s_start vs s_end_plus with a second model (Opus 4.7 or GPT-5) to bound the judge-noise contribution on the H4 reversal.
- Stylistic ablation: rewrite s_end_plus texts in parable format (or s_start texts in "Myth:" format) and re-run a small cell to isolate format from content.
- Phase 3 Soc-one: at N=8, seed only 1 of 8 agents and look for diffusion of cooperation. This was infeasible until the multi-agent merge (§14) and is the natural next experiment given Phase 2's positive sufficiency signal.

---

### 17.7 Implementation footprint (Phase 2)

Re-implementing the §6 mechanism on the current codebase needs the three adjustments documented in §6's "Re-implementation notes" plus:

- **New baseline configs** in `config/experiments.yaml`, one per task order × noise condition:
  - `sonnet45_8agent_game_directive_history3_anon_noisy_negative_<L>_r10_n15`
  - `sonnet45_8agent_game_myth_directive_history3_anon_noisy_negative_<L>_r10_n15`
  - `sonnet45_8agent_myth_game_directive_history3_anon_noisy_negative_<L>_r10_n15`
  Each is a clone of `sonnet45_8agent_myth_directive_history3_anon_r10_n5` with `task_order` swapped, `noise_config` added (passed through to `TrustGameNoisy` in `games/trust_game_noisy.py`), and `repetitions: 15`. The noisy game class already supports `noise_config` post-merge.
- **Seed-injection plumbing** in `src/simulation.py`: accept `seed_myth: Optional[str]` and `seed_user_prompt: Optional[str]` kwargs on `run_simulation`; in the init-agents loop, after the system-prompt append (around the line where `agent.messages.append({"role": "system", "content": system_prompt})` runs), append the fake-user/seed-assistant pair so the seed lands at `messages[1:3]`. Order matters: the `with_system_context` multi-agent preamble must already be on the system prompt when the seed is injected so the seed sits after the preamble.
- **Cell dispatcher** — new `experiments/run_ablation_phase2.py` (or extension to `run_trust_game_batch.py`) that:
  - reads a Phase 2 cell config (task order × seed type × noise level × rep count),
  - loads the harvested seed manifest,
  - passes `history_policy`, `self_history_window`, `coplayer_history_window`, `show_agent_names`, `task_order`, `noise_config`, `seed_myth`, `seed_user_prompt` through to `run_simulation`.
- **New harvest script** `scripts/harvest_seeds_phase2.py` that scans an 8-agent anon-history3 source pool by `(source_run, agent_id, round_number)` rather than `Agent_1`-fixed. Output manifest schema: `{seed_id, source_run_path, agent_id, round_number, joint_balance_at_source, text, text_features: {tokens, concreteness, naive_judge_score}}`.
- **Verification smoke run** (§17.5 step 6): the new code does not have an M1 message-wipe path, so the §6 "3-line message list" verification doesn't apply. The new target is to inspect the round-1 game prompt and confirm: (a) the seed text appears verbatim in the agent's prior assistant turn (`messages[2]`), and (b) the round-1 game prompt is the standard round-1 template with no history block (multi-agent later-round prompt only kicks in from round 2+).
- Total ~150–200 LoC of code changes + the noise-pilot pre-flight.

Total: ~100 LoC of code changes + the new game-only anon baseline run (if fork A picks game-only). Slightly larger than Phase 1's footprint because the multi-agent path has more knobs that need to be threaded through.
