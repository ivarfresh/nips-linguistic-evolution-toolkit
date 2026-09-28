# Main-frontier mixed-model runs, 2026-09-28: Opus 5, Gemini 3.1 Pro, GPT-5.6 Sol

**Headline.** Mixing the main frontier models lifts the weakest cooperator, GPT-5.6 Sol.
In a dyad the round-1 sender sets the level for the whole game. When Opus 5 or Gemini
3.1 Pro sends first, it opens at $5 and Sol follows to about $5. When Sol sends first, it
opens at $2.50–3; its partner matches that and the pair stays low. A myth task removes
the difference: with myth → game, every mixed pair ends at or near the ceiling. In the
8-agent mix, Sol sends more than it does among its own kind, and the population sits
near the ceiling.

These are the frontier versions of the mixed-model Figures 7 and 8, in the same style.
The 2026-09-18 Opus 5 / Gemini 3.1 Pro / GPT-5.6 Sol set is the main frontier simulation
(D011, decided 2026-09-28).

## What was run

- **Protocol:** the unchanged September no-defector protocol, with informed negative-only
  noise and 10 rounds.
- **Models:** each at its 2026-09-18 request profile (D011):
  - Opus 5: adaptive thinking, effort high;
  - Gemini 3.1 Pro: thinking high, temperature 0.8;
  - GPT-5.6 Sol: effort high.
- **Checks:** before any paid call the launcher asserts that every non-model input
  equals the September control cell, and it audits every final afterwards.

| Set | Design | Runs |
|---|---|---|
| Mixed dyads | Opus 5 + Sol, Opus 5 + Gemini, Gemini + Sol × 3 task orders × 6 replicates. The first-named family sends first (Agent_1) in replicates 0/2/4, the other in 1/3/5, as in the September mixed dyads (D010). | 54 |
| Mixed population | 8 agents, balanced rotating pairs: Agent_1–2 Gemini 3.1 Pro, Agent_3–5 Opus 5, Agent_6–8 GPT-5.6 Sol × 3 task orders × 5 replicates | 15 |

- **Cost:** $73.18 at standard rates, computed from recorded token usage (estimate $73.47).
- **Quarantined runs:** two dyads were quarantined and re-run under the same seeds. In
  each, one Gemini call dropped its connection (`RemoteDisconnected`) during the
  20-worker batch; the audit rejects any final with a failed call. The reasons are under
  `data/json/noise_experiments/frontier_mixed_main_20260928/quarantine/`.
- **Not included in the $73.18:** a first 6-worker launch was cancelled before any run
  finished, to restart with 20 workers. Its in-flight calls, and the two quarantined runs,
  were billed but are not in the $73.18.
- **References:** the homogeneous comparison runs are the 90 audited finals of 2026-09-18.
  The noise and pairing seeds are 202608250 + replicate in both sets. Mixed-dyad
  replicate 5 has no homogeneous counterpart.

Launcher: `scripts/run_frontier_main_mixed.py`.
Analysis: `scripts/analyze_frontier_main_mixed_20260928.py`.

## Dyads (Figure 7 style)

`frontier_mixed_dyads_resources_boxplots.png`. The table gives mean final resources per
agent, as the mean (±sd over runs) of each run's mean.

| Pair | Game only | Game → Myth | Myth → Game |
|---|---|---|---|
| Opus 5 + Opus 5 (homogeneous, n=5) | 67.8 (±2.6) | 73.5 (±1.1) | 75.0 (±0.0) |
| Sol + Sol (homogeneous, n=5) | 57.0 (±17.5) | 57.1 (±7.5) | 70.5 (±6.3) |
| Gemini + Gemini (homogeneous, n=5) | 73.8 (±2.7) | 73.8 (±1.8) | 75.0 (±0.0) |
| Opus 5 + Sol (mixed, n=6) | 66.2 (±9.2) | 70.9 (±2.3) | 75.0 (±0.0) |
| Opus 5 + Gemini (mixed, n=6) | 69.2 (±6.5) | 73.4 (±2.2) | 75.0 (±0.0) |
| Gemini + Sol (mixed, n=6) | 64.4 (±11.7) | 73.3 (±1.9) | 74.3 (±1.6) |

Mean send per family, mean (±sd over runs) (`family_summary.csv`):

| Family | Setting | Game only | Game → Myth | Myth → Game |
|---|---|---|---|---|
| Sol | with Sol | 3.20 (±1.75) | 3.21 (±0.75) | 4.55 (±0.63) |
| Sol | with Opus 5 | 3.97 (±0.95) | 4.42 (±0.29) | 5.00 (±0.00) |
| Sol | with Gemini | 4.05 (±1.09) | 4.73 (±0.31) | 4.93 (±0.16) |
| Opus 5 | with Opus 5 | 4.28 (±0.26) | 4.85 (±0.11) | 5.00 (±0.00) |
| Opus 5 | with Sol | 4.27 (±0.91) | 4.75 (±0.23) | 5.00 (±0.00) |
| Opus 5 | with Gemini | 4.68 (±0.57) | 4.82 (±0.13) | 5.00 (±0.00) |
| Gemini | with Gemini | 4.88 (±0.27) | 4.88 (±0.18) | 5.00 (±0.00) |
| Gemini | with Sol | 3.83 (±1.29) | 4.93 (±0.16) | 4.93 (±0.16) |
| Gemini | with Opus 5 | 4.15 (±1.05) | 4.87 (±0.33) | 5.00 (±0.00) |

- **Sol is pulled up.** It sends more with either partner than with another Sol, in
  every task order.
- **Gemini is pulled down, in game-only play only.** With Sol it drops from 4.88 to 3.83,
  and with Opus 5 to 4.15. With a myth task Gemini stays at about $5.
- **Opus 5 sends more with Gemini (4.68) than with its own kind (4.28)** in game-only
  play, and the same as its own kind with Sol.
- **The round-1 sender sets the level in game-only play.**
  - Opus 5 or Gemini sent first in 12 runs; 11 opened at $5 and these runs ended at
    71.8 (±5.2).
  - Sol sent first in 6 runs, opening at $2.50–3 each time. These runs ended at
    56.2 (±5.1).
  - In 16 of 18 runs the second sender matches the first sender's opening and keeps it
    (per-run send traces). The two exceptions are both Opus 5 + Gemini runs where Opus
    opened at $5: Gemini stayed at $2.50 in one and climbed from $3 to $5 in the other.
  - Homogeneous Sol dyads open at 2.5, 2.5, 3, 5 and 5, so a low opening is Sol's own habit.
  - Caveat: the two first-sender blocks also use different seeds (replicates 0/2/4
    against 1/3/5), so first sender and seed are not separated. At n = 6 and 12 runs this
    is descriptive.
- **A myth task closes the gap.** With myth → game every mixed pair ends at 74.3–75.0;
  with game → myth at 70.9–73.4.

## Populations (Figure 8 style)

`frontier_mixed_populations_resources_boxplots.png`:

| Population | Game only | Game → Myth | Myth → Game |
|---|---|---|---|
| 8 Opus 5 (n=5) | 68.4 (±1.8) | 73.0 (±0.6) | 74.7 (±0.3) |
| 8 Sol (n=5) | 63.1 (±15.0) | 67.0 (±8.7) | 73.6 (±1.0) |
| 8 Gemini (n=5) | 74.7 (±0.6) | 74.5 (±0.7) | 75.0 (±0.0) |
| 2 Gemini + 3 Opus 5 + 3 Sol (n=5) | 72.2 (±0.9) | 72.9 (±1.0) | 74.7 (±0.4) |

Mean send per family, mixed population against its own kind (mean ± sd over runs):

| Family | Game only | Game → Myth | Myth → Game |
|---|---|---|---|
| Sol | 4.73 (±0.17) vs 3.81 (±1.50) | 4.66 (±0.21) vs 4.20 (±0.87) | 4.95 (±0.12) vs 4.86 (±0.10) |
| Opus 5 | 4.55 (±0.14) vs 4.34 (±0.18) | 4.78 (±0.11) vs 4.80 (±0.06) | 4.98 (±0.04) vs 4.97 (±0.03) |
| Gemini | 4.94 (±0.13) vs 4.97 (±0.06) | 5.00 (±0.00) vs 4.95 (±0.07) | 5.00 (±0.00) vs 5.00 (±0.00) |

- **Sol is pulled up.** Among Opus 5 and Gemini its game-only send rises from 3.81 to 4.73,
  and the run-to-run spread shrinks from ±1.50 to ±0.17. No mixed run shows a partial
  collapse like the homogeneous Sol populations' low runs (36.6 in game only, 51.8 in game →
  myth).
- **Opus 5 rises a little in game-only play** (4.34 to 4.55).
- **Gemini is unchanged** at about $5.
- **The mixed population is tight:** 72.2–74.7 with small spread, compared with 63.1
  (±15.0) for all-Sol.

## What this means for the paper

- **The mixed-model story holds at the frontier.** Families adapt to their partners, and
  the lower-cooperating family moves most. At the frontier that is GPT-5.6 Sol, where in
  September it was GPT-5 Nano.
- **The effect is smaller than in September.** The weakest frontier model cooperates far
  more than Nano did.
- **The dyads add a mechanism:** in game-only play the round-1 sender anchors the pair,
  and a myth task makes the anchor irrelevant.
- **Descriptive only:** n = 5–6 runs per cell. For comparison, the newer-model check with
  Opus 5.5 and GPT-6 Sol is in `docs/figures/frontier_update_20260928/`; there every
  family starts at the ceiling.

## Reproduce

```bash
python3 scripts/build_frontier_rerun_config.py            # regenerates the config (sets frontier_main_mixed_*)
python3 scripts/run_frontier_main_mixed.py --stage all    # dry run; --execute to launch
python3 scripts/analyze_frontier_main_mixed_20260928.py   # tables, figures, provenance
```

Receipt with the sha256 of every final:
`data/json/noise_experiments/frontier_mixed_main_20260928/all_receipt.json`.
