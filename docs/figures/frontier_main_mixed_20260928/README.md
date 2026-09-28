# Main-frontier mixed-model runs, 2026-09-28: Opus 5, Gemini 3.1 Pro, GPT-5.6 Sol

**Headline.** In mixed runs, GPT-5.6 Sol, the weakest cooperator of the three, sends more
than it does among its own kind, most clearly in the 8-agent population. In game-only
dyads the outcome tracks who sends first. In pairs with Sol, runs where Opus 5 or Gemini
opened (at $5) end at 74.4, and runs where Sol opened (at $2.50–3) end at 56.2; the
partner usually answers near Sol's opening. In the two-task orders no such first-sender
difference appears, and with myth → game every mixed pair ends at or near the ceiling.

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

- **Sol sends more with either partner than with another Sol,** in every task order.
- **Gemini sends less in game-only dyads only.** With Sol it drops from 4.88 to 3.83
  (−1.05, more than Sol's rise of +0.85), and with Opus 5 to 4.15. With a myth task Gemini
  stays at about $5.
- **Opus 5 sends more with Gemini (4.68) than with its own kind (4.28)** in game-only
  play, and the same as its own kind with Sol.
- **In game-only play, the outcome tracks the round-1 sender.**
  - Within the pairs that include Sol: when Opus 5 or Gemini sent first (6 runs, all
    opening at $5) the runs ended at 74.4 (±0.7); when Sol sent first (6 runs, opening at
    $2.50–3) they ended at 56.2 (±5.1).
  - Across all 18 game-only mixed dyads, the second sender's first send is within $1 of the
    opening in 16. The two exceptions are both Opus 5 + Gemini runs where Opus opened at
    $5 and Gemini answered at $2.50 and $3.
  - The level does not always stay put. Of the 6 Sol-first runs, 4 stay low throughout; in
    the other 2 one or both partners climb to $5 later (Sol→Opus replicate 3, Sol→Gemini
    replicate 5).
  - Sol opens low in 3 of 5 homogeneous dyads (openings 2.5, 2.5, 3, 5, 5) and in all 6
    mixed dyads it opened.
  - Caveat: the two first-sender blocks also use different seeds (replicates 0/2/4
    against 1/3/5), so first sender and seed are not separated. At n = 6 and 12 runs this
    is descriptive.
- **No first-sender difference in the two-task orders.** Within Sol pairs, game → myth
  ends at 72.3 (partner first) vs 71.9 (Sol first), and myth → game at 75.0 vs 74.3. With
  myth → game every mixed pair ends at 74.3–75.0; with game → myth at 70.9–73.4.

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

- **Sol sends more in the mix.** Among Opus 5 and Gemini its game-only send is 4.73 against 3.81,
  and the run-to-run spread shrinks from ±1.50 to ±0.17. No mixed run shows a partial
  collapse like the homogeneous Sol populations' low runs (36.6 in game only, 51.8 in game →
  myth).
- **Opus 5 rises a little in game-only play** (4.34 to 4.55).
- **Gemini is unchanged** at about $5.
- **The mixed population is tight:** 72.2–74.7 with small spread, compared with 63.1
  (±15.0) for all-Sol.

## What this means for the paper

- **Families' sending tracks their partners at the frontier too.** In the population and
  in two-task dyads, the lower cooperator (GPT-5.6 Sol) moves most. In game-only dyads the
  higher cooperator (Gemini) drops about as much as Sol rises, as in September, when
  Gemini fell from 5.0 to 1.17 against GPT-5 Nano while Nano barely moved
  (`docs/figures/mixed_model_dyads_20260917/family_behaviour.csv`).
- **The effect is smaller than in September.** The weakest frontier model cooperates far
  more than Nano did.
- **The dyads add a pattern:** in game-only play the outcome is associated with who sends
  first (confounded with the seed block); no such difference appears in the two-task
  orders.
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
