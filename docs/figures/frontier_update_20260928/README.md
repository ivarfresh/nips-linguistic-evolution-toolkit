# Frontier update, 2026-09-28: GPT-6 Sol, Opus 5.5 at five replicates, frontier mixed populations

**Headline.** The current frontier models all play the trust game at or near the ceiling.
GPT-6 Sol fully cooperates in five of six cells, where GPT-5.6 Sol sat between 57 and 74.
A mixed population of 2 Gemini 3.1 Pro, 3 GPT-6 Sol and 3 Opus 5.5 also sits at the
ceiling. The only family that moves is Opus 5.5: without a myth, it cooperates more among
the other two families (73.1) than among its own kind (69.7).

## What was run

All runs use the unchanged September no-defector protocol: informed negative-only noise,
10 rounds, 2 agents in a fixed pair or 8 agents with balanced rotating pairs. The task
orders are game only, game → myth and myth → game. Only the model and its pinned request
settings change. The launchers assert this before any paid call and audit every final
afterwards.

| Set | Runs | Cost (standard rates) | Launcher |
|---|---|---|---|
| GPT-6 Sol, effort high, 2 and 8 agents × 3 task orders × 5 replicates | 30 | $21.16 | `scripts/run_frontier_rerun.py --stage main_sol6` |
| Opus 5.5, replicates 3–4 added to the 18 runs of 2026-09-23 | 12 (30 in total) | $21.03 | `scripts/run_frontier_rerun.py --stage main --arms opus55` |
| Mixed 8-agent: Agent_1–2 Gemini 3.1 Pro, Agent_3–5 GPT-6 Sol, Agent_6–8 Opus 5.5 × 3 task orders × 5 replicates | 15 | $31.03 | `scripts/run_frontier_mixed_populations.py` |

Total $73.22 at standard rates, computed from recorded token usage. GPT-6 Sol is priced at
$2/$10 per million input/output tokens (OpenAI model page, read 2026-09-28), half of
GPT-5.6 Sol's $4/$20. The Anthropic credit ran out during the Opus top-up. The calls in
flight at that moment were billed but produced no final, so they are not in this figure.

Request settings (`scripts/build_frontier_rerun_config.py`):
- **GPT-6 Sol** uses the GPT-5.6 Sol profile: `reasoning_effort: high`, temperature
  omitted, cap 128000. On GPT-6 Sol, `high` is no longer the top level (`xhigh` and `max`
  exist), so this is the same setting, not the same relative effort.
- **Opus 5.5** uses adaptive thinking at effort high.
- **Gemini 3.1 Pro** uses thinking level high at temperature 0.8.
- In the mixed runs, each family uses exactly its homogeneous profile.

**Pairings in the mixed runs.** The balanced schedule is fixed per replicate: it is the
same across task orders and models (checked on the homogeneous frontier finals). Over the
5 replicates the 200 games are Opus–Sol 63, Gemini–Sol 43, Gemini–Opus 39, Opus–Opus 24,
Sol–Sol 22 and Gemini–Gemini 9. Random pairing of a 2/3/3 population would give 64/43/43/
21/21/7, so the block layout adds no bias. Myth exposure follows the previous partner, so
these counts also set how often each family reads each other family's myths.

## Results

### Homogeneous frontier arms

Mean final resources per agent, averaged within each run, then mean (±sd) over the 5 runs
(`cell_summary.csv`). The maximum under full cooperation is about 75.

| Agents | Task order | GPT-6 Sol | GPT-5.6 Sol | Opus 5.5 | Opus 5 | Gemini 3.1 Pro |
|---|---|---|---|---|---|---|
| 2 | Game only | 75.0 (±0.0) | 57.0 (±17.5) | 70.1 (±5.9) | 67.8 (±2.6) | 73.8 (±2.7) |
| 2 | Game → Myth | 70.7 (±8.8) | 57.1 (±7.5) | 73.8 (±1.3) | 73.5 (±1.1) | 73.8 (±1.8) |
| 2 | Myth → Game | 74.2 (±1.1) | 70.5 (±6.3) | 74.6 (±0.9) | 75.0 (±0.0) | 75.0 (±0.0) |
| 8 | Game only | 74.9 (±0.2) | 63.1 (±15.0) | 69.7 (±1.3) | 68.4 (±1.8) | 74.7 (±0.6) |
| 8 | Game → Myth | 75.0 (±0.0) | 67.0 (±8.7) | 72.1 (±0.8) | 73.0 (±0.6) | 74.5 (±0.7) |
| 8 | Myth → Game | 74.9 (±0.2) | 73.6 (±1.0) | 74.8 (±0.2) | 74.7 (±0.3) | 75.0 (±0.0) |

- **GPT-6 Sol is ceiling-locked.** It sends $5 in 732 of 750 decisions (97.6%). The one lower
  cell, the 2-agent game → myth cell, comes from a single run (replicate 0). That pair
  opened at $3 and stayed at $3 for all ten rounds. The other four runs are at the ceiling.
  GPT-6 Sol no longer shows GPT-5.6 Sol's partial collapses, so the myth task has nothing
  left to rescue.
- **Opus 5.5 at five replicates confirms the three-replicate check.** Every cell is within
  2.3 points of Opus 5 on the same seeds. It keeps the same ordering: game only < game →
  myth < myth → game, at both sizes.
- **Only Opus leaves room for a task-order effect.** Among the current frontier models,
  Opus 5.5 is the only one below the ceiling. It gains about 5 points from a myth task.

### Frontier mixed populations

Per family, mean (±sd over 5 runs) of the family's run-mean (`mixed_family_summary.csv`,
figure `mixed_vs_homogeneous.png`).

| Task order | Family | Resources, mixed | Resources, own family | Send, mixed | Send, own family |
|---|---|---|---|---|---|
| Game only | Gemini 3.1 Pro | 73.2 (±1.4) | 74.7 (±0.6) | 4.88 | 4.97 |
| Game only | GPT-6 Sol | 74.9 (±1.5) | 74.9 (±0.2) | 4.97 | 4.99 |
| Game only | Opus 5.5 | 73.1 (±0.9) | 69.7 (±1.3) | 4.79 | 4.47 |
| Game → Myth | Gemini 3.1 Pro | 73.9 (±1.9) | 74.5 (±0.7) | 4.94 | 4.95 |
| Game → Myth | GPT-6 Sol | 74.9 (±0.8) | 75.0 (±0.0) | 4.92 | 5.00 |
| Game → Myth | Opus 5.5 | 72.9 (±0.7) | 72.1 (±0.8) | 4.83 | 4.71 |
| Myth → Game | Gemini 3.1 Pro | 74.7 (±0.5) | 75.0 (±0.0) | 5.00 | 5.00 |
| Myth → Game | GPT-6 Sol | 75.9 (±0.9) | 74.9 (±0.2) | 4.97 | 4.99 |
| Myth → Game | Opus 5.5 | 74.1 (±0.6) | 74.8 (±0.2) | 5.00 | 4.98 |

- **The mix is at the ceiling.** Every family sends close to $5 and returns 44–46% of what
  it receives, the same as on its own.
- **The one visible shift is Opus 5.5 in game-only play.** It sends 4.79 among the other
  families against 4.47 among its own kind, and ends 3.4 points higher. This matches the
  September ladder, where a family's sending tracks its partners': Opus's partners here
  are mostly ceiling players. With a myth task the gap closes, because Opus is already
  near the ceiling on its own.
- **The mix doesn't drag anyone down.** Gemini loses 0.3–1.5 points against its own
  population, within about one run-to-run sd.

## What this means for the paper

The earlier mixed-model result (families track their partners; a lone Gemini is exploited
by GPTs; two GPTs drag Sonnets down) depended on a low-cooperating family, GPT-5 Nano. No
current frontier model plays that role. A frontier mixed run can therefore show pull-up
(Opus), but not contagion of defection. To test contagion at the frontier, you'd need a
defector or a lower-cooperation arm. GPT-6 Sol at effort none is untested, as is the
forced-defector treatment of D009.

These are descriptive results at n=5 runs per cell.

## Reproduce

```bash
python3 scripts/build_frontier_rerun_config.py      # regenerates config/frontier_rerun_20260918.yaml
python3 scripts/run_frontier_rerun.py --stage main_sol6                  # dry run; --execute to launch
python3 scripts/run_frontier_rerun.py --stage main --arms opus55         # dry run
python3 scripts/run_frontier_mixed_populations.py                        # dry run
python3 scripts/analyze_frontier_update_20260928.py                      # tables, figure, provenance
```

Receipts (with the sha256 of every final): `data/json/noise_experiments/frontier_rerun_20260918/`
`main_sol6_receipt.json`, `main_opus55_receipt.json`, and
`data/json/noise_experiments/frontier_mixed_populations_20260928/completion_receipt.json`.
