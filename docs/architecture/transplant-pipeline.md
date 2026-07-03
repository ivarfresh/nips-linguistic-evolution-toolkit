---
title: Myth-transplant pipeline — how seeded cells run
status: current
updated: 2026-07-02
owner: ivar
---

# Myth-transplant pipeline

How a seeded cell runs, end to end. For what the experiments *found*, see
[findings-cooperation-transplant.md](findings-cooperation-transplant.md); for the
gotchas that shape valid designs, see [design-constraints.md](design-constraints.md).

## The Phase 3 regime (current standard)

Every seeded cell since Phase 3 uses the myth-only chat-memory contract
(`docs/phase3_chat_memory_spec.md`, validated by `scripts/phase3_inspect_chat_memory.py`):

- At the start of **every round**, each agent's chat memory is reset to exactly
  `[system, seed_user, seed_myth]` — the seed never scrolls out and game decisions
  are never appended (`remember=False`).
- No history-block in the prompt text; round prompts carry only round number,
  balance, role, action request (`history_policy="none"`).
- Game: 8-agent dyadic pairing, endowment $5, multiplier 3×, 10 rounds,
  temperature 0.8, negative noise up to $5 (`phase3_8agent_anon_neg5_myth_only`
  in `config/experiments_noisy.yaml`). Ceiling joint balance = $600; unseeded
  baseline = $437.4 (±$5.5). Task order `["game"]` only.
- This replaced Phase 2's single-shot injection (seed at `messages[1:3]`, scrolls
  out by round 2–4) — Phase 2 results confounded seed influence with the
  history-block. _(from researchlog 2026-06-19, 2026-06-23)_

## Seeds

- Manifest: `data/phase3/seed_manifest.json` — 9 pools of 5 myths each, harvested
  from source runs (`scripts/phase*_harvest_*.py`, `phase7_register_gowith_seeds.py`).
  Pools: `s_start`, `s_end_plus`, `s_filler`, `s_end_minus`, `s_start_jab`,
  `s_end_plus_jab`, `s_end_plus_gemini`, `s_end_plus_gpt`, `s_end_plus_gowith`.
- Rep *i* of a cell uses seed *i* (`rep % len(seeds)`), so per-seed outcomes are
  comparable across cells.
- Runner: `experiments/run_phase3_seeded_cells.py` (`--seed-types`, `--rep-list`
  for retries, `--no-seed` for baselines, `--output-subdir` per phase).
  ~$2.30/rep on Sonnet 4.5.

## Cheap screens before funding a cell (~$1–2 vs $11.50/cell)

`scripts/phase7_decoder_asymmetry.py` runs two probes over any seed pool
(`--manifest`/`--pools` accept ad-hoc manifests):

- `--mode extract` — game rules in system prompt, myth in user turn, model
  extracts `{has_strategy, send, return_fraction}`. Measures **legibility**.
- `--mode behavioral` — reproduces the exact injection shape
  `[system, seed_user, seed_myth, round-1 prompt]` and reads the actual round-1
  send. Measures **bindingness**. Validated: reproduces the full-game five-regime
  ordering, and predicted the gowith cell outcome. _(from researchlog 2026-07-02)_

**Before any paid cell with an unusual-register seed:** capture the exact
runtime-built messages (monkeypatch dump pattern → `data/phase7/debug_failing_call.json`)
and probe ~4× per seed for `stop_reason="refusal"` — template-approximated probes
can pass while the real call refuses deterministically. See
[design-constraints.md](design-constraints.md). _(from researchlog 2026-07-02)_

## Environment facts

- No `.venv` — pyenv 3.11.10 via `.python-version`; `python3` works from repo root.
- `src/utils.py::call_llm()` returns a dict `{content, reasoning, usage}`.
- Empty/refusal responses are retried (stochastic case) since 2026-07-02;
  deterministic refusals censor the rep. See `docs/verified-facts.md`.
