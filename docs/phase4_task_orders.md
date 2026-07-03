# Phase 4 — Add myth-writing back into the loop

Extends Phase 3 (myth-only chat memory, single seed, `["game"]`-only task order) to the two task orders that include myth-writing: `["myth", "game"]` and `["game", "myth"]`. Same population, same noise, same seed pool. Adds one variable: whether agents also produce a myth each round.

This is the design coworker explicitly deferred while we validated Phase 3 v1. With v1 clean and the chat-memory contract holding, Phase 4 is the next pre-registrable step.

Reference: `docs/phase3_chat_memory_spec.md` for the Phase 3 setup this builds on.

---

## 1. The design question to lock first

When the agent gets to the "myth-writing" task in a round, what should its chat memory contain?

Two coherent options:

### Option A — "Every round is round 1 for myth-writing too" (recommended for first run)

Chat memory at the start of every round stays `[system, seed_user, seed_myth]` exactly as Phase 3 v1. When the myth task fires, the agent writes a myth conditioned only on the seed. That myth is saved to `sim_data` for later analysis but is **not** appended to chat memory. If game fires next (`["myth","game"]`), the agent plays with only the seed visible. Round 2 starts over the same way.

**Why this is the right first move.** Same regime guarantees as Phase 3 v1 — every round's chat memory is byte-identical to every other round's, modulo the round-specific game/myth prompts at the end. We're varying exactly one thing (task order) on top of v1, not two. Cleaner attribution: any difference from Phase 3 v1 is the myth-writing task itself, not a memory-regime change.

**The cost.** The myth-writing template literally says *"drawing on the game you have been playing up to this point"* — under Option A the model is being asked to draw on something not in its chat memory. The model will produce coherent text anyway (it can infer from the seed + system prompt that a game is being played), but the prompt asks for something it can't strictly do. Worth noting as a small inconsistency rather than redesigning the prompt.

### Option B — "Let the agent see its own previous myth"

Chat memory at the start of every round becomes `[system, seed_user, seed_myth, last_own_myth_user, last_own_myth_assistant]`. The agent gets to build on what it wrote last round. This breaks the "stranger turns up" framing but is closer to what people usually mean by *"do myths evolve under seeding."*

**Defer to Phase 4b.** The "myths evolve" question is real and worth answering, but it adds two new variables (own-myth memory + task order) at the same time. Run Option A first; only run Option B if Option A shows the seed effect surviving the addition of new myth content.

---

## 2. Implementation under Option A

Three small changes from the Phase 3 codebase.

### 2.1 `src/simulation.py` — myth task respects `remember=False`

In the `task == "myth"` branch, mirror what we already do for the game task: pass `remember=(chat_memory_mode != "myth_only")` to `agent.respond()`. One-line change.

The newly-written myth still gets passed to `myth_writer.process_myths(...)`, which appends it to `sim_data.conversation_history[turn]["myths"]`. So myths are durable for later linguistic analysis even though they don't enter chat memory.

### 2.2 `src/simulation.py` — always use the round-1 myth prompt

When `chat_memory_mode == "myth_only"`, always call `myth_writer.get_myth_prompt_round_1(agent_id, turn, sim_data)` regardless of `turn`. The standard `get_myth_prompt_round_later(...)` expects to read `last_myth` and `other_agent_myth` from `sim_data.conversation_history`, but under myth-only memory those would be redundant (the agent doesn't remember them anyway and is "starting fresh" every round). ~3 lines.

### 2.3 `experiments/run_phase3_seeded_cells.py` — accept `--task-orders`

Add a CLI flag accepting one or more of `game`, `myth_game`, `game_myth`. Update the combo generator to iterate over the chosen task orders alongside seed types and reps. Rename the script to `run_phase4_seeded_cells.py` (or generalize the existing Phase 3 runner — either is fine; the simulation changes above are the substantive work).

### 2.4 No new config block needed

The same `phase3_8agent_anon_neg5_myth_only` game_params applies. The directive myth prompts already in `config/experiments_noisy.yaml` (`myth_writing_default_game_directive`, `myth_writing_later_rounds_directive`) are what the runner uses today — Phase 4 just calls the round-1 template every round.

---

## 3. Validation contract (extends Phase 3 v1)

Extend `scripts/phase3_inspect_chat_memory.py` to handle myth-task interactions too.

For each smoke-test output:

- For `myth_game`: at rounds 1, 5, 10 — verify the **myth** task's `messages_sent` payload contains exactly `[system, seed_user, seed_myth, round_N_myth_prompt]`. Verify the **game** task's `messages_sent` at the same round contains exactly `[system, seed_user, seed_myth, round_N_game_prompt]`. Messages [0:3] must be byte-identical across all rounds *and* across both tasks within a round.
- For `game_myth`: same contract, but check that the game task fires before the myth task in each round.
- For both: total messages per LLM call must be exactly 4. If the agent's own previously-written myth ever appears in chat memory, the contract fails — Option A is broken.

Run on Haiku output first. Sonnet money only flows after the contract is green.

---

## 4. Cost

| Step | What | Cost |
|---|---|---|
| 4.1 | Haiku 4.5 code smoke: 1 rep × 3 cells × `myth_game` only | ~$0.15 |
| 4.2 | Sonnet 4.5 pilot, `myth_game` only: 3 reps × 3 cells (baseline + s_start + s_end_plus) | ~$21 |
| 4.3 | Sonnet 4.5 pilot, both task orders: 3 reps × 3 cells × 2 task orders | ~$41 |
| 4.4 | Headline: 5 reps × 3 cells × 2 task orders | ~$69 |

**Recommended first run:** step 4.1 (smoke) + step 4.2 (`myth_game` pilot only) for ~$21. `myth_game` matches the team's existing baseline framing and is the most directly comparable to Phase 2's results.

Add `game_myth` as a step 4.3 follow-up only if `myth_game` shows something interesting that wasn't visible in Phase 3 v1.

---

## 5. What Phase 4 tells us that Phase 3 v1 couldn't

1. **Does the seed's cooperation-lift survive when the agent is also generating its own myths?** Under Option A the seed gets re-asserted at the start of every round, so the agent never accumulates self-generated myth content in chat memory. The seed should still drive behavior. But empirically: maybe the *act of writing a myth* (even one not remembered) shifts the agent's framing for the subsequent game decision. We don't know.

2. **Whether myth content evolves under a constant seed.** Save the per-round myths to `sim_data`. Later, do a linguistic analysis: do round-10 myths drift toward or away from the seed's style/content? Are they more abstract, more strategic, more divergent across agents? This is the "do myths evolve under seeding" question, answerable from the saved data without re-running anything.

3. **Whether myth-before-game and game-before-myth differ.** `myth_game` lets the agent compose a myth before deciding; `game_myth` lets it act first and reflect after. If these differ at the same seed condition, that's a planning-vs-rationalization distinction worth a paragraph.

---

## 6. Open follow-up questions (not in scope for this Phase 4 run)

- **Option B run.** Once Phase 4 Option A is done, run Option B with 5 reps × 1 task order × 1 cell as a small follow-up to test whether persistent own-myth memory changes the dynamics.
- **Ceiling problem.** Phase 3 v1 had s_end_plus saturating $600. Phase 4 will likely too, since the seed effect should at least carry over. If it doesn't, that's informative; if it does, the Phase 5 priority is making the game harder (more noise / more rounds / more agents) before scaling reps.
- **Counter-cooperative seed.** Still needed for the "any myth vs cooperative myth" question. Independent of task order — could be added to either Phase 3 or Phase 4 design.

---

## 7. Decision needed before code is written

The only thing blocking implementation is signing off on Option A vs Option B for the first run. Default recommendation: **Option A**, with B as an explicit Phase 4b follow-up.
