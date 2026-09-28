# D011 — Rerun the September matrix on current frontier models with reasoning on

- Recorded / last verified: 2026-09-18 / 2026-09-28
- Decision status: agreed and executed (Opus 5, Gemini 3.1 Pro Preview, GPT-5.6 Sol at effort high); Sol at effort none skipped after its smoke run; GPT-5.6 Luna rejected as a non-frontier tier. Opus 5.5 checked on replicates 0–2 (2026-09-23); **decided 2026-09-24 (Ivar): Opus 5.5 replaces Opus 5 as the frontier Claude model**; Opus 5 stays as a robustness check. **2026-09-28 (Ivar):** GPT-6 Sol (effort high) run as a full arm (30/30) and Opus 5.5 completed to five replicates (30/30); **Decided 2026-09-28 (Ivar), superseding 2026-09-24: the main frontier model simulation is the 2026-09-18 set (Opus 5, Gemini 3.1 Pro, GPT-5.6 Sol)**, the set Ed referred to; frontier mixed dyads and populations use these three models. The Opus 5.5 / Gemini 3.1 Pro / GPT-6 Sol runs are kept as the **frontier update set**, a newer-model check.
- Scope: the September no-defector matrix (2 and 8 agents × game / game→myth / myth→game × replicates 0–4, informed negative-only noise) on one flagship per provider. Excludes defector conditions, mixed-model runs, and any reasoning-off arm.
- Decision authority: Ed's request in the 2026-08-18 team meeting (transcript: results stop being believed once the model is three or four months old); Ivar's authored instructions in the 2026-09-18 Claude session ("run full Opus 5, Gemini and sol High. Skip sol none"; reasoning kept on after the assistant's assessment).
- Implementation status: sampled completed finals, 90/90 launcher-audited (`data/json/noise_experiments/frontier_rerun_20260918/main_reasoning_on_receipt.json`); config `config/frontier_rerun_20260918.yaml` (generated), launcher `scripts/run_frontier_rerun.py`, branch `run/frontier-rerun-20260918`.

## Decision and rationale

Run each provider's current flagship on the unchanged September protocol, with the
provider's reasoning on, so the task-order results can be shown on models readers
consider current. **Explicit** (Ivar, 2026-09-18): frontier capability is the goal,
which excludes Luna (OpenAI's nano tier) and Sonnet 5; Sol-none is dropped from the
matrix. **Explicit** (Ivar, accepting the assistant's assessment): reasoning stays on
because the frontier claim is about the deployed system, the September references had
reasoning on, the September format study showed visible reasoning moves Claude
sending by about 21 points, and Opus 5 with thinking disabled writes reasoning into the
visible answer. **Inferred** (assistant, accepted by launch): Batch API and prompt
caching were reviewed by three agents and rejected as not worth their engineering
at the corrected cost (about $150 total).

Request profiles: Opus 5 adaptive thinking with `output_config.effort: high`,
temperature omitted, cap 64000; Gemini 3.1 Pro `thinkingLevel: high`, temperature 0.8,
cap 65536; Sol `reasoning_effort: high`, temperature omitted, cap 128000. Opus 5
rejects `budget_tokens`, so the thinking regime changes together with the model; this
is a recorded protocol difference, not a pure model swap (see D004).

## Evidence

- Primary source: Ivar's instructions in the 2026-09-18 Claude session (user
  instruction); Ed's 2026-08-18 meeting transcript (computer-generated transcript,
  attribution as recorded).
- Prices and API constraints read on 2026-09-18 from platform.claude.com (pricing;
  4.7+ tokenizer note), developers.openai.com (gpt-5.6-sol, gpt-5.6-luna model pages),
  ai.google.dev (thinking levels; 3.1 Pro cannot disable thinking).
- Implementation: commits on `run/frontier-rerun-20260918` (config generator,
  launcher, analysis); condition audit asserts every non-model input equals the
  September combination and every call's request settings equal the pinned plan.
- Runs: 90 finals with sha256 in `main_reasoning_on_receipt.json`; two quarantined
  Gemini finals with `reason.json` under `quarantine/` (transient connection drops,
  resampled under the same seeds); credit interruptions on all three providers,
  rerun after top-ups. Results and disclosures:
  `docs/figures/frontier_rerun_20260918/README.md`.

## Chronology and supersession

- 2026-08-18: Ed asks for a rerun on current models (transcript).
- 2026-09-17: Codex cost table (pasted by Ivar) prices Opus 5 at $2/$10; superseded by
  the verified $5/$25 on 2026-09-18.
- 2026-09-18: plan doc `docs/research/frontier_model_run_plan_2026-09-18.md`; first
  cost table overstated 5–7× (recursive usage aggregation), corrected against the
  receipt method; three-agent review of Batch API, caching, model choice; smoke (4
  arms), 18-run pilot, 90-run main stage; Sol-none skipped.
- 2026-09-23: Ivar asks whether to use Opus 5.5 instead of Opus 5 and requests a test
  run ("do smoke and pilot for now", then "just do 3 replicates per cell"). An `opus55`
  arm with the Opus 5 request profile (effort pinned high) was added; 18/18 audited
  finals, $31.79 (`reps3_opus55_receipt.json`). All six cell means within 2 points of
  Opus 5 on the same seeds; same task-order ordering; wider spread in the 2-agent game
  (61/75/73). Opus 5.5 uses 3–8× more thinking tokens per call at the same effort, so
  the thinking regime differs again. Ivar leaned towards switching ("probably yes");
  the switch itself is **proposed**, pending his confirmation. Results:
  `docs/figures/frontier_rerun_20260918/README.md` (Opus 5.5 check).
- 2026-09-24: Ivar confirms the switch ("yes", in the Claude session, replying to
  "Should I mark the switch as decided?"). Opus 5.5 is now the frontier Claude model;
  new frontier cells, including the proposed frontier mixed-model runs, use the
  `september23_opus55` profile. Opus 5 (90 finals) is kept as a robustness check. The
  thinking-regime difference (3–8× more thinking tokens at the same effort) is to be
  disclosed alongside the switch.
- 2026-09-28: Ivar asks for the frontier matrix on "opus 5.5 and gpt 6 sol" (Claude
  session, user instruction; he recalled the earlier arm as GPT-5.6 Sol, confirmed in
  `config/frontier_rerun_20260918.yaml`). GPT-6 Sol is on the direct OpenAI key
  (model list, 2026-09-28) at $2/$10 per MTok (OpenAI model page), half GPT-5.6 Sol's
  $4/$20 (also confirmed; the 2026-09-18 receipts stand). Arm `sol6_high` uses the Sol-high
  request unchanged (`september28_sol6_high`); on GPT-6 Sol `high` is below `xhigh` and
  `max`, a recorded regime note. Smoke, 6-run pilot, then 30/30 audited finals ($21.16,
  `main_sol6_receipt.json`). Opus 5.5 replicates 3–4 added (12 runs, $21.03;
  `main_opus55_receipt.json` lists all 30); the code that makes the model calls (`src/`,
  `games/`, `experiments/`) is identical for replicates 0–2 and 3–4 (no diff between
  their recorded commits). An Anthropic credit exhaustion interrupted
  the top-up and it resumed after a top-up. **Result (descriptive, n=5):** GPT-6 Sol is
  ceiling-locked (74.2–75.0 in five of six cells; $5 in 732 of 750 sends; one 2-agent
  game→myth run locked at $3 gives that cell 70.7 ±8.8), unlike GPT-5.6 Sol's partial
  collapses. Opus 5.5 at five replicates stays within 2.3 points of Opus 5 in every cell,
  with the same task-order ordering. Opus 5.5 is now the only frontier model below the
  ceiling. See [the update README](../../figures/frontier_update_20260928/README.md).
- 2026-09-28 (later the same day): Ivar, in the Claude session: "the frontier model
  simulation Ed referred to, is the one with Opus 5, Gemini 3.1 pro and GPT 5.6 Sol. Lets
  save this as the main frontier model simulation indeed. We want to run the frontier dyads
  and the mixed frontier model with these models. Use a mix of 2/3/3 for gemini 3.1 pro,
  Opus 5 and GPT 5.6 sol respectively." This supersedes the 2026-09-24 decision that
  Opus 5.5 replaces Opus 5: the 90 finals of 2026-09-18 are the main frontier result, and
  the Opus 5.5 / GPT-6 Sol runs above (with the 2/3/3 Opus 5.5 / GPT-6 Sol mixed
  population of D010) become the frontier update set, reported as a newer-model check
  (`docs/figures/frontier_update_20260928/`). Main-frontier mixed runs: see D010.
- Does not supersede D004 (September profiles remain the September regime) or D010.

On 2026-09-18 (corrected 2026-09-22 after review on PR #30) the frontier gap
was decomposed without new runs (`scripts/analyze_frontier_gap.py`,
[report](../../research/frontier_gap_investigation_2026-09-18.md)). **Result
(descriptive, n = 5 per cell):** final resources equal 25 + 10 × mean send in
all 180 runs, so the gap is a sending gap. Opus 5 opens higher than Sonnet 4.5
and raises its send after a profitable round; Sol opens at 2.5 to 3 where Nano
opens at 0, and after an apparent loss (visible payoff below $5) Sol raises
its next send by 0.39 while Nano lowers it by 0.51. Opus used a median of 0
thinking tokens per game call, so reasoning depth is not the driver. The
review correction changed the loss denominator (actual transfer, as the
investor sees it) and the description of the 0.95 figure (decision-level
correlation of own send with the partner's previous communicated send); it did
not change the conclusions. This is an analysis result, not a design change.

## Unresolved / next evidence

- Five replicates per cell: descriptive only. Sol's game-only collapse (2 of 10 runs)
  needs more replicates before any rate is claimed.
- Whether Sol's behavior differs at effort none (the zero-lock question flagged in the
  September researchlog) is untested beyond one smoke run.
- Fable 5.1 and GPT-6 Astra were priced but not run.
