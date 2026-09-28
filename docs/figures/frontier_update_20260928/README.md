# Frontier update, 2026-09-28: GPT-6 Sol, Opus 5.5 at five replicates, frontier mixed populations

**Status (decided 2026-09-28, Ivar):** this is the *frontier update set*, a newer-model
check. The main frontier result is the 2026-09-18 Opus 5 / Gemini 3.1 Pro / GPT-5.6 Sol set
(`docs/figures/frontier_rerun_20260918/`). `frontier_update_set_resources_boxplots.png` shows this set as a grid
in the style of the mixed-population figure.

**Headline.** The current frontier models all play the trust game at or near the ceiling.
GPT-6 Sol ends at 74.2–75.0 in five of six cells, where GPT-5.6 Sol sat between 57 and 74.
A mixed population of 2 Gemini 3.1 Pro, 3 GPT-6 Sol and 3 Opus 5.5 also sits at the
ceiling. The only family that moves is Opus 5.5 in game-only play. Among the other two
families it sends more (4.79 against 4.47 among its own kind): after a similar opening it
gets to $5 sooner. There is no sign that it sends more to its generous partners in
particular.

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
GPT-5.6 Sol's $4/$20. The Anthropic credit ran out during the Opus top-up. Calls that
completed inside the 8 runs that then failed were billed but left no final, so they are
not in this figure.

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
21/21/7, close to random-pairing expectation; the schedule does not depend on which model sits in
which slot. Myth exposure follows the previous partner
(720 of 720 exposures), so these counts also set how often each family reads each other
family's myths. The launcher checks non-model inputs against the September replicate-0
cell; the finals additionally match the homogeneous Opus 5.5 finals replicate by
replicate (inputs and noise/pairing seeds), checked in review.

## Results

### Homogeneous frontier arms

Mean final resources per agent, averaged within each run, then mean (±sd) over the 5 runs
(`cell_summary.csv`). A population in which every send is $5 averages about 75 per agent;
single agents can end above or below 75 depending on what their partners return.

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
  opened at $3 and stayed at $3 for all ten rounds. The other four runs end at 74.0–75.0.
  GPT-6 Sol no longer shows GPT-5.6 Sol's partial collapses, so the myth task has nothing
  left to rescue.
- **Opus 5.5 at five replicates confirms the three-replicate check.** Every cell mean is
  within 2.3 points of Opus 5. Single seeds differ by up to 8 points (2-agent game only,
  replicate 0: 61.0 against 69.0). It keeps the same ordering: game only < game →
  myth < myth → game, at both sizes.
- **Only Opus leaves room for a task-order effect.** Among the current frontier models,
  Opus 5.5 is the only one below the ceiling. It gains 2–5 points from a myth task (myth → game +4.5 and +5.1; game → myth +3.7 and +2.4, at 2 and 8 agents).

### Frontier mixed populations

Per family, mean (±sd over 5 runs) of the family's run-mean (`mixed_family_summary.csv`,
figure `mixed_vs_homogeneous.png`).

| Task order | Family | Send, mixed | Send, own family | Resources, mixed | Resources, own family |
|---|---|---|---|---|---|
| Game only | Gemini 3.1 Pro | 4.88 (±0.11) | 4.97 (±0.06) | 73.2 (±1.4) | 74.7 (±0.6) |
| Game only | GPT-6 Sol | 4.97 (±0.06) | 4.99 (±0.02) | 74.9 (±1.5) | 74.9 (±0.2) |
| Game only | Opus 5.5 | 4.79 (±0.18) | 4.47 (±0.13) | 73.1 (±0.9) | 69.7 (±1.3) |
| Game → Myth | Gemini 3.1 Pro | 4.94 (±0.13) | 4.95 (±0.07) | 73.9 (±1.9) | 74.5 (±0.7) |
| Game → Myth | GPT-6 Sol | 4.92 (±0.09) | 5.00 (±0.00) | 74.9 (±0.8) | 75.0 (±0.0) |
| Game → Myth | Opus 5.5 | 4.83 (±0.11) | 4.71 (±0.08) | 72.9 (±0.7) | 72.1 (±0.8) |
| Myth → Game | Gemini 3.1 Pro | 5.00 (±0.00) | 5.00 (±0.00) | 74.7 (±0.5) | 75.0 (±0.0) |
| Myth → Game | GPT-6 Sol | 4.97 (±0.06) | 4.99 (±0.02) | 75.9 (±0.9) | 74.9 (±0.2) |
| Myth → Game | Opus 5.5 | 5.00 (±0.00) | 4.98 (±0.02) | 74.1 (±0.6) | 74.8 (±0.2) |

Sends measure an agent's own behaviour. Resources also include what its partners send and
return, so in a mixed run a family's resources partly reflect the other families.

- **The mix is at the ceiling.** Every family sends close to $5 and returns 44–46% of what
  it receives, close to its own-family level (44–48%).
- **The one visible shift is Opus 5.5 in game-only play.** It sends 4.79 (±0.18) among
  the other families against 4.47 (±0.13) among its own kind.
  - There is no sign of partner-specific sending. In the mixed runs Opus sends 4.73 to
    Gemini and Sol receivers (51 decisions) and 4.92 to other Opus agents (24). These are
    pooled decisions, not independent, and names are hidden; both groups have the same
    mean round (about 5.4).
  - It is timing: Opus opens at a similar level (round 1: 3.75 mixed, 8 decisions; 3.3 on
    its own, 20), then gets to $5 sooner. In round 2 it rises to 4.43 in the mixed runs but drops to
    3.12 among its own kind. Rounds 2–5 average 4.80 mixed against 4.13 on its own;
    rounds 6–10 are at or near $5 in both (4.98 on its own).
  - A reading consistent with this, not a tested mechanism: early rounds against ceiling
    partners are profitable, and Opus raises its send after a profitable round (the
    escalation pattern of the 2026-09-18 frontier-gap report).
  - With a myth task the gap closes, because Opus is already near the ceiling on its own.
  - Its resources (73.1 against 69.7) move more than its sends, partly because the other
    families send it $5.
  - One composition, n=5; the own-family comparison runs for replicates 0–2 date from
    2026-09-23.
- **Sends barely drop anywhere.** The only send drops are Gemini in game-only play (4.88
  against 4.97) and Sol in game → myth (4.92 against 5.00). Resource shifts are partly
  redistribution between families: in myth → game, Sol ends 1.0 higher than on its own,
  while Opus ends 0.7 and Gemini 0.3 lower.

## What this means for the paper

The earlier mixed-model result (families track their partners; a lone Gemini is exploited
by GPTs; two GPTs drag Sonnets down) involved a low-cooperating family, GPT-5 Nano. No
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
