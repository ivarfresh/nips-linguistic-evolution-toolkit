---
title: Experiment protocol — memory regime, noise, QA, and data infrastructure
status: current
updated: 2026-09-17
owner: ivar
---

# Experiment protocol

How runs are configured, audited, and stored as of the corrected v2 era.

## Memory regime: memory-primary (canonical)

The June double-memory bug (rounds appearing 2–3× via chat memory + in-prompt
recaps) was fixed by adopting **memory-primary**: both tasks are remembered in
chat memory; duplication is removed on the prompt side (`_minimal` game
templates, myth prompt carries only the co-player myth). Every fact enters
context once. Chosen over *hybrid* (amputates the reasoning→myth pathway) and
*stateless* (homogenizes myths to near-determinism: cross-agent similarity
0.820±0.021 vs 0.719±0.100). Memory-primary is the only design whose
linguistic layer visibly couples to game events — myth drift responds to a
turbulent game while the other channels stay insulated. Game play itself is
channel-invariant. _(from researchlog 2026-07-17)_

Capacity is matched to ~3 completed rounds: `memory_capacity: 3` for game-only,
6 for two-task cells. _(from researchlog 2026-08-12)_ The pre-fix June baseline
inflated the generosity ratchet and suppressed myth drift (self-similarity
0.823 vs 0.688 post-fix); do not compare its myth-stability numbers 1:1 with
post-fix runs. _(from researchlog 2026-07-17)_

## Noise: informed bidirectional is the default condition

Standing directive (2026-07-22): every new experiment runs under informed
bidirectional ±$1 uniform noise on communicated amounts — `inform_agents:
true` auto-appends the notice via `trust_game_noisy.py`. Informed ≈ uninformed
on every metric, with tighter spreads. _(from researchlog 2026-07-22)_

The cross-model defector series (2026-08-25 onward) runs under **negative-only**
noise instead. The two regimes have different dynamics: bidirectional noise
keeps send and return fractions climbing past round 10–15 before they
stabilize; negative-only noise gives flatter curves and a weaker visible
effect. Do not compare cooperation curves across the two regimes without
saying which one a run used. _(from researchlog 2026-09-01)_

The September cross-model dataset uses informed negative-only `U(-1, 0)`
noise on both sent and returned amounts (the 90 "informed controls" plus the
forced-defection cells). The figure-2 extension adds the same cells under no
noise and under uninformed negative-only noise (180 runs), so the no-defector
comparison now spans three noise regimes at the same profiles. A targeted
`U(-2, 0)` dyad bridge (Claude, Gemini; game-only and myth-first; five paired
replicates) links the historical range-2 results to the corrected pipeline;
its effect on Claude was larger than at range 1. _(from researchlog 2026-09-16)_

**Dyad transfer-noise fix (load-bearing):** communicated-send noise is
generated only for the receiver, *after* the sender's actual transfer exists;
a missing transfer raises instead of silently becoming zero. Every result from
the pre-fix dyad path (post-round-1 receivers saw ~$0) is contaminated — see
superseded claims in [findings-taskorder-myth.md](findings-taskorder-myth.md).
_(from researchlog 2026-08-12)_

## The corrected v2 protocol (confirmatory cells)

- Full 2×3 design (2/8 agents × game / game→myth / myth→game), unified
  sender/receiver prompt builder at both population sizes
  (`prompt_regime: unified`), `history_policy: none` in causal cells (no
  synthetic self/co-player blocks; decisions live only in chat memory), hidden
  names, `pairing_mode: fixed` with accurate system instructions, paired
  pairing/noise seeds. History and prompt-regime values fail closed on unknown
  strings. _(from researchlog 2026-08-12)_
- The standard 8-agent arm deliberately retains rotating partners + a
  three-game co-player block: the contrast is *repeated dyad vs rotating
  population with reputation information*, not a pure agent-count
  manipulation. _(from researchlog 2026-08-12)_
- The game-side line "Take any myths written in this session into account…"
  is appended by one shared finalizer, exactly once, in every current game
  prompt. _(from researchlog 2026-08-12)_
- Retry hardening: an unanswered prompt is removed from private chat memory
  after an exhausted provider call; rejected attempts stay in the audit but
  never contaminate the retry context. _(from researchlog 2026-08-12)_
- Historical provenance is uneven: provider, requested settings, code state and
  stop reasons were not uniformly recorded. Absent fields remain unknown;
  neither current defaults nor absent reasoning text reconstruct an old request.
  Saved zero counters may include adapter defaults. Dataset-specific limits are
  in the [reassessment](../api-audit-reassessment-2026-09-08.md).
  _(from researchlog 2026-09-08)_

## Provider route and sampling settings

Guarded experiments declare all four `llm_settings` fields: `provider`,
provider-native `reasoning`, `temperature` policy, and `max_output_tokens`
policy. Anthropic requires an explicit positive output cap; there is no implicit
4096 cap in the guarded path. See [the usage guide](../safeguards-usage.md) for
the schema, covered entrypoints, structural validation limits and legacy opt-in.

Settings environment variables cannot override the resolved guarded plan.
New guarded runs save `run_metadata.llm_request`, the experiment condition and
per-call requested settings/outcomes. The record describes what was requested,
not a guarantee about vendor internals. Missing usage remains unknown.

Comparisons explicitly name fields that may differ and explain why, including
model and replicate where appropriate; historical comparisons also acknowledge
missing provenance. Covered readers reject undeclared differences and exact
duplicate inputs. Resume checks retain the recorded condition. These safeguards
do not silently change experimental prompts, memory, noise, task order or retries.

The September 4 proposed low-reasoning/two-sentence profile was not adopted by
the restart. The associated format study also changed myth instructions,
self-context and retry policy, so it does not justify making that format a
standard. Future native-setting robustness and format/memory interventions
remain separate scientific decisions. _(from researchlog 2026-09-08)_

**Current pinned cross-model profiles ("september8").** Measured in the
2026-09-08 cost replay and used for every cross-model run since: Claude Sonnet
4.5 with thinking enabled (budget 8192), temperature omitted, output cap 64000;
GPT-5 Nano reasoning effort high, temperature omitted, cap 128000; Gemini 3.7
Flash thinking level high, temperature 0.8, cap 65536. Anchors live in
`llm_profiles` in `config/experiments_noisy.yaml`. These are deliberately
model-specific settings, not equivalent amounts of computation; enabling an
8192 thinking budget did not make Claude consume it (~500 tokens per game
call). The 270-run rerun cost $167.97 against the pilot's $177.94 projection;
GPT-5 Nano two-task sets were the wall-clock critical path (~3 h each).
_(from researchlog 2026-09-08, 2026-09-10)_

**Mixed populations pin one plan per agent.** A mixed-model experiment set
declares `agent_models` (one `base_models` key per agent, in Agent_1.. order)
and `llm_settings_by_model` instead of `models` and `llm_settings`. Each agent
gets its own client, model and native request plan; the run-level plan records
`provider: mixed` with the per-agent plans under `agents`, and the condition
validator checks every call's recorded settings against its own agent's plan.
Homogeneous runs are unchanged. The first use is the 36-run mixed dyad stage
(`scripts/run_mixed_model_dyads.py`); mixed and homogeneous pools are validated
separately in analysis because a mixed run has no run-level policy block.
_(from researchlog 2026-09-17)_

## Mixed-model experiment (D010)

Scope: heterogeneous model families in one population, both fixed dyads and
eight-agent populations; compositions Sonnet/GPT and Sonnet/Gemini; task orders
game, game→myth, myth→game; no homogeneous arms and no forced defectors; game
pairings cross-family only. The dyad stage is frozen and complete: six
replicates per cell, Sonnet as Agent_1 (round-1 sender) in replicates 0/2/4
and the other family in 1/3/5, everything else equal to the September
informed-noise dyad controls. The eight-agent stage (4+4, five replicates,
cross-family-only rotating pairing) waits for a decision after the dyad
results; its pairing mode is not implemented. _(from researchlog 2026-09-16,
2026-09-17)_

## QA: audits with negative controls

- `scripts/audit_v2_protocol.py` — per-cell joint audit: exactly one accepted
  response per agent/round, unrecovered retries fail, forced-action and
  noise-check counts exact. Response validation must check *task type*, not
  just JSON shape — an accepted "myth" that is actually decision JSON slipped
  through until the validator ran at audit time too. _(from researchlog
  2026-08-12, 2026-08-21)_
- `scripts/audit_context_integrity.py` — per-call `messages_sent` audit
  (myth windows verbatim, partner myth quoted, no duplicated facts); standard
  post-batch QA. Validate audits with planted corruptions: the first version
  was blind to drops at the old edge of the window. Audits need negative
  controls too. _(from researchlog 2026-07-23)_
- **Frozen-plan launchers** are the pattern for paid batches: dry-run by
  default, a plan step that validates every job against the configuration and
  its reference cells before any call, one preflight line (`MODEL= N= WORKERS=
  EST_COST=`), `--execute` that requires a clean checkout, per-final audits of
  request settings and completion, a hashed completion receipt with
  standard-rate cost, and resumption that recognises existing finals without
  repeating calls. Provider credit exhaustion cancels queued jobs instead of
  retrying. _(from researchlog 2026-09-16, 2026-09-17)_
- **Resume validation skips every non-LLM event.** Forced-zero defector
  decisions and deduction notices carry no request settings; the condition
  check now exempts any `response_source != "llm"` while a missing source still
  defaults to `llm` and is checked. Before the fix a resume rejected every
  forced-defector final. _(from researchlog 2026-09-09)_
- **Replay through original commits** verifies a rerun's inputs without paid
  calls: feeding new responses through the historical code reproduced all
  1,000 per-agent request message arrays of the GPT-5.5 gate rerun exactly.
  _(from researchlog 2026-09-09)_
- **Disclose resamples.** Under the pinned repeat-once retry policy, Sonnet
  occasionally answers a sender prompt with a `return` key or with prose
  instead of JSON; such runs fail and are resampled, and every analysis over
  those cells must say so. _(from researchlog 2026-09-10, 2026-09-16, 2026-09-17)_

## Project memory

`docs/project-memory/` (README, CURRENT.md, WORKFLOW.md, `decisions/DNNN-*.md`)
is the repository source of truth for current research state and individual
design decisions. Agents read it at session start and update it when a durable
decision, semantic change, invalidating bug, completed result or primary source
appears; `scripts/validate_project_memory.py` checks the index and links. The
research log stays append-only and keeps the short auditable trail; decision
detail lives in the records. _(from researchlog 2026-09-15)_

## Data infrastructure

- **Raw runs**: `data/json/` is gitignored; the shared store is the private HF
  dataset `machine-cultural-evolution/nips-linguistic-evolution-runs`
  (`scripts/sync_data.sh push|pull`, per-user namespaces, resumable +
  deduplicated; optional post-batch auto-push hook). Doubles as the citable
  dataset at publication. _(from researchlog 2026-08-25)_
- **Public-facing repo**: `github.com/ivarfresh/linguistic-evolution-toolkit`
  is the clean single-commit export (95 MB); this repo remains the full
  private archive. Re-check anonymity before flipping it public.
  _(from researchlog 2026-07-17)_
- Billing-failed attempts leave `*.checkpoint.json.error.json` snapshots in
  data dirs — loaders must exclude them or n inflates. _(from researchlog
  2026-07-17)_
