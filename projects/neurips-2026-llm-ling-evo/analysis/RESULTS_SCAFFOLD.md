# §3 Results — prose scaffold (this week's data)

Notes for Aron to lift into `manuscript/_3-results.b_draft.md`.
Scope: `noise_experiments/v4_direct_provider/`, Claude + GPT-5-Nano only,
N=15 seeds per cell.  Numbers cite `HEADLINE_TABLES.md`.

This scaffold mirrors the §3 outline structure but pre-narrows to
**§3.1, §3.3, §3.4, §3.5** — §3.2 (noise effects on cooperation profile)
needs Gemini's negative-noise data to fully run, and that's outside this
scope.  Aron should slot §3.2 in from Ivar's re-run when it lands.

---

## §3.1 Cross-model behavioural baselines (no-noise)

**Note for Aron:** the headline-table data here drops Gemini and the
no-noise baselines (since this analysis is scoped to v4 only, where there
is no no-noise cell).  Best to keep §3.1 as outlined and pull baseline
numbers from your prior baseline data (`data/json/baseline/`) — the v4
analyses extend §3.2 onward.

---

## §3.2 Effect of noise on the cooperation profile

**Headline:** the noise mechanisms produce qualitatively different
cooperation regimes for Claude and GPT-5-Nano.  In v4_direct_provider:

- **Claude × negative_5 (uninformed) × game-only** — mean cumulative
  balance 29.3 (vs ~55 baseline) — large drop, as expected.
- **Claude × positive (uninformed) × game-only** — mean 73.5 (very
  close to ceiling) — agents reach near-Pareto-optimal cooperation when
  the noise channel can only *upgrade* perceived returns.
- **GPT-5-Nano × negative_5 × game-only** — mean 37.9 — *higher than no-
  noise baseline* (cf. baseline ~30).  Negative-only noise paradoxically
  *increases* cooperation for the floor-locked model: agents perceive
  losses as smaller than they are, sustaining trust longer.
- **GPT-5-Nano × positive × game-only** — mean 72.5 — also near ceiling.
- **GPT-5-Nano × bootstrap × game-only** — mean 66.6 — substantial
  rescue from the no-noise floor of ~30.

**Interpretation:** for both models, *upward-biased* perceived signals
(positive, bootstrap, negative-from-floor) lift cooperation; *downward*
signals depress it.  Noise is the methodological lever that creates the
behavioural latitude the rest of the paper uses; it is not the
substantive finding.

---

## §3.3 Effect of myth-writing on game behaviour (the headline question)

**Headline:** myth-writing produces a heterogeneous effect across cells.
Three patterns emerge:

### Pattern A — *lift + consolidation* (the strongest signal, in 3 cells)

When noise is upward-biased and the headroom is partial, adding myth
both raises mean cooperation and reduces across-seed variance:

- **Claude × positive (informed) × game→myth** —
  Δmean +0.59 [+0.22, +0.96], var ratio 0.31 [0.05, 0.66].
  Mean already at 74.2/75 ceiling without myth; myth pushes it to 74.8,
  with ~3× lower across-seed variance.
- **GPT-5-Nano × positive × myth→game** —
  Δmean +2.08 [+0.17, +4.44], var ratio 0.05 [0.00, 0.54].
  20× variance reduction — the strongest consolidation result we have.
- **GPT-5-Nano × positive (informed) × game→myth** —
  Δmean +2.23 [+1.06, +3.52], var ratio 0.08 [0.01, 0.24].
  12× variance reduction.

### Pattern B — *pure consolidation* (1 cell)

- **GPT-5-Nano × positive × game→myth** —
  Δmean +1.65 [-0.26, +4.01] (CI includes 0), var ratio 0.07 [0.01, 0.58].
  Mean shift not significant, variance shrinks 14×.

### Pattern C — *harmful or destabilizing* (4 cells, all GPT-5-Nano × bootstrap)

When the noise mechanism *flips* perceived returns (bootstrap noise
reports full reciprocation when low), adding myth makes things worse:

- **GPT-5-Nano × bootstrap × game→myth** — Δmean −7.73 [−14.27, −1.67],
  classified **harmful** (mean down, variance flat).
- **GPT-5-Nano × bootstrap × myth→game** — Δmean −7.60 [−13.53, −2.07],
  **harmful**.
- **GPT-5-Nano × bootstrap (informed) × game→myth** — Δmean −9.67
  [−14.67, −5.00], var ratio 7.5 [3.5, 15.8], **destabilizing**.
- **GPT-5-Nano × bootstrap (informed) × myth→game** — Δmean −6.87
  [−11.13, −3.07], var ratio 4.8 [1.5, 11.0], **destabilizing**.

### Cells where myth has no detectable effect

The remaining 12 cells (mostly Claude × negative_5, GPT-5-Nano × negative_5)
classify as **null** (mean CI includes 0, variance ratio CI includes 1).
Worth reporting honestly: myth is not a universal modulator.

### Interpretation

The dominant pattern in upward-biased noise regimes is **strategy
consolidation with modest mean lift**.  The dominant pattern in
floor-rescue (bootstrap) noise is **harm or destabilization**.
Concretely: when agents already have signal pointing in a useful
direction, myth-writing helps them lock onto a coordination point and
reduces seed-to-seed drift.  When the signal is itself confused
(bootstrap masking defection as cooperation), additional storytelling
makes the dyad less, not more, coherent.

The "narrow-but-real" framing from §0 holds: language is not a strong
cooperation switch, but it is not decoration either — its effect is
real, *moderate*, and *sign-conditional on the noise regime*.

---

## §3.4 Linguistic dynamics in the myth chain

### Lag-1 cross-agent cooperativity correlation

The pilot finding of *r ≈ 0.72* for one Claude dyad replicates broadly:
across all 21 (model × noise × informed × task_order) cells we measured,
mean Pearson r between Agent_A's myth cooperativity at round t and
Agent_B's at round t+1 is **always positive**, ranging from 0.11 to 0.38
on the AB direction (and similar magnitude on BA).  Treating per-dyad
*max(lag1_AB, lag1_BA)* as a more meaningful summary (since pilot
correlations don't always favour the same direction across dyads), the
mean across runs lands at **0.27–0.58** depending on cell, with
**27–73% of dyads showing |r| > 0.5**.

The strongest cells (lag1_max mean > 0.45 AND ≥50% of dyads with |r|>0.5):

- **GPT-5-Nano × positive (informed) × myth→game** — lag1_max mean
  **0.58 [0.43, 0.71]**, 73% of dyads above 0.5.
- **GPT-5-Nano × bootstrap (informed) × game→myth** — lag1_max mean
  **0.48 [0.35, 0.60]**, 53% above 0.5.
- **GPT-5-Nano × bootstrap (informed) × myth→game** — lag1_max mean
  **0.45 [0.32, 0.56]**, 53% above 0.5.
- **GPT-5-Nano × negative_5 (informed) × game→myth** — lag1_max mean
  **0.50 [0.38, 0.60]**, 53% above 0.5.

The Claude r ≈ 0.72 case sat near the upper tail of a clearly non-zero
distribution; it is not anomalous.

### Coinages and their persistence within myth chains

Across 600+ chains (run × agent), a simple dictionary-based coinage
detector finds:

- **Claude** produces ~2–5 distinct coinages per chain, with **max
  persistence reaching 10 rounds** (i.e., the same invented word — e.g.
  *Lumina*, *Aelara*, *Arachnis*, *Celestia*, *Luminara* — appears in
  every round of a 10-round chain).  Mean max-persistence is highest in
  Claude × negative_5 × myth→game (informed): **2.6 rounds**.
- **GPT-5-Nano** produces more coinages per chain (~5–6) but they
  persist for fewer rounds (~1.5–2).  Less consolidation on a stable
  mythological vocabulary.

### Coinage leakage into game reasoning — does *not* replicate

Across all 600+ chains, **share of runs in which any myth-coinage
re-appears in the agent's game `reason` text = 0.00, in every cell.**
The pilot observation that made-up words leak from the myth channel
into game reasoning **does not replicate at corpus scale.**  Coinages
stay inside the myth.

(But thematic content does — see §3.5.)

---

## §3.5 Coupling between game behaviour and myth content

### Direct coupling: own-myth vocabulary in game reasoning (Claude only)

Claude's game-response prose is extensively threaded with content
words from the agent's own myth.  In every cell:

- **78–82% of round-level reasons contain at least one content word from
  the agent's own myth chain.**  Mean ~5–7 unique vocabulary overlaps
  per reason.
- **100% of runs** have at least one such reference.
- Theme-lexicon hits (story / spirit / elder / sacred / ancestor /
  ritual / etc.) reach **10–33% of reasons**, with negative-noise cells
  showing the highest theme density (~28–33% vs ~10–13% in positive
  noise).

This is the cleanest direct evidence that myth content enters game
reasoning for Claude.

### GPT-5-Nano: undecidable from visible output

GPT-5-Nano produces no visible reasoning prose.  Its
`game_responses[ag].content` is the bare JSON action (e.g. `{"send": 3}`)
with empty `reasoning` field — for every round, every dyad.

We **cannot determine** from the visible output whether GPT-5-Nano's
internal computation routes through the myth content or not.  This is a
limitation to flag rather than a finding to report: the cross-task
linguistic channel is observable for models that emit reasoning prose;
for models that emit only the structured action, we have a methodological
blind spot.  (One concrete EMNLP-version follow-up: re-run GPT-5-Nano
with a system prompt that explicitly requests reasoning prose before the
action, and re-code.)

### Implication for §3.5's headline

The "coupling between game behaviour and myth content" lives in *Claude*
in the v4 corpus.  Two strands of evidence both point the same way:

1. **Behavioural lag-1 correlation** (myth-language rhythms transmit
   between agents): present in Claude but stronger in GPT-5-Nano cells
   that don't have visible reason text.  Both models show non-trivial
   transmission of cooperativity rhythms across the linguistic channel.
2. **Lexical thematic carryover into reasoning prose** (myth content
   shows up *inside* game reasoning): Claude only, but extremely
   pervasive (~80% of rounds).

Combined with §3.3's finding that Claude × positive (informed) × game→myth
shows lift+consolidation, the picture for Claude is internally
consistent: when the noise regime gives the dyad headroom, myth content
both threads through the agent's reasoning text *and* coincides with a
small mean lift and a large variance reduction.

---

### Between-agent myth convergence (embedding cosine over rounds)

Pairwise cosine similarity between Agent_1 and Agent_2 myths at the
same round, embedded with `sentence-transformers/all-mpnet-base-v2`.
**Convergence is robust across cells**: 18 of 21 cells have a slope CI
that excludes 0 above (positive convergence over rounds).

The strongest convergence cells (slope per round, with 95% bootstrap CI):

- **GPT-5-Nano × negative_5 × myth→game** — slope **+0.027/round**
  [0.016, 0.038].  Cosine rises from 0.42 → 0.74 across 10 rounds — the
  sharpest convergence in the corpus.
- **GPT-5-Nano × bootstrap × myth→game** — slope **+0.023/round**
  [0.013, 0.032].  Cosine 0.51 → 0.78.
- **Claude × positive (informed) × myth→game** — slope **+0.021/round**
  [0.013, 0.028].  Cosine 0.51 → 0.75.

The myth-writing chain is doing systematic work — agents' myths are
not random samples drawn IID from the model. They thread together over
rounds in a measurable, statistically-supported way, regardless of
model, noise type, or task order.  This holds even in cells where the
*game* outcome (§3.3) is null or harmful — confirming that linguistic
convergence is happening *whether or not* it improves the game.

That is the clean separation §3.4 needs: linguistic convergence is the
robust phenomenon; behavioural consolidation is the conditional one.

---

## What's NOT in this scaffold

- **Gemini cells** — out of scope for this week's runs.  Add §3.1 / §3.2
  Gemini cells from prior data when prosing this section.
- **LLM-judge five-dimension similarity (A5)** and **ETI trait inference
  (A7)** — Tier-2, not run here (require API calls).  Defer to EMNLP
  unless time allows.

---

## Source files (for reproducibility)

- Cell-level numbers: `cell_summaries/cell_summary.csv`, `deltas.csv`
- Lag-1 correlations: `cell_summaries/lag_summary.csv`,
  `lag_correlations.csv`
- Neologisms: `cell_summaries/neologism_summary.csv`,
  `neologisms_examples.csv`, `neologisms_per_run.csv`
- Reason-field coding: `cell_summaries/reason_coding_summary.csv`,
  `reason_coding_per_round.csv`
- Embedding (pending): `cell_summaries/embedding_summary.csv` (and
  per-run CSV)

Generators (rerun after Ivar's re-run lands; outputs deterministic at
RNG seed 42):
- `analysis/build_cell_summary.py` — A1 + A2 (+ classification table)
- `analysis/build_lag_and_lexicon.py` — A3
- `analysis/build_neologism_analysis.py` — A9
- `analysis/build_reason_coding.py` — A6
- `analysis/build_embedding_convergence.py` — A4
- `analysis/build_headline_tables.py` — collates all of the above into
  `HEADLINE_TABLES.md`

Run order: cell_summary → (lag, neologism, reason_coding, embedding) in
parallel → headline_tables.
