# A1 partner-myth vs controls: diagnostic sub-analysis

Comparison of the real-A1 partner-myth injection against three defensive controls (C1 shuffled, C2 filler, C3 own-myth) and the no-A1 baseline. All cells are bootstrap-cooperation; informed and uninformed framing both shown when data exist. Real-A1 has no informed cells (—) and only N=13 in myth_game uninformed.

Endowment is fixed at 10 for these gpt-5-nano runs. Embeddings: `sentence-transformers/all-MiniLM-L6-v2`. Defection events for the recovery metric are per-round total payoff (investor + trustee) below (run mean − 1 run SD); recovery is the next round meeting or exceeding the run mean. All slopes/drifts use rounds 1–5 vs 6–10 when 10 rounds are present.

## 1. Trajectory shape

### 1a. Final cumulative dyad balance (mean ± SD)

| Condition | game_myth uninformed | game_myth informed | myth_game uninformed | myth_game informed |
| --- | --- | --- | --- | --- |
| no-A1 baseline | 117.73 ± 21.23 (N=15) | 116.53 ± 18.38 (N=15) | 118.00 ± 17.72 (N=15) | 122.13 ± 14.65 (N=15) |
| real A1 (partner myth) | 136.27 ± 6.77 (N=15) | — | 136.77 ± 6.95 (N=13) | — |
| C1 shuffled | 136.40 ± 8.65 (N=15) | 132.93 ± 8.91 (N=15) | 138.27 ± 7.15 (N=15) | 137.87 ± 4.81 (N=15) |
| C2 filler | 141.07 ± 4.37 (N=15) | 135.21 ± 7.33 (N=14) | 138.00 ± 6.02 (N=15) | 133.20 ± 10.90 (N=15) |
| C3 own | 130.80 ± 11.61 (N=15) | 134.33 ± 8.03 (N=15) | 135.00 ± 7.20 (N=15) | 136.53 ± 9.22 (N=15) |


### 1b. Early slope (round 5 − round 1 cumulative balance)

| Condition | game_myth uninformed | game_myth informed | myth_game uninformed | myth_game informed |
| --- | --- | --- | --- | --- |
| no-A1 baseline | 49.87 ± 7.36 (N=15) | 49.20 ± 8.16 (N=15) | 51.87 ± 7.74 (N=15) | 51.47 ± 5.68 (N=15) |
| real A1 (partner myth) | 53.60 ± 6.29 (N=15) | — | 54.77 ± 5.24 (N=13) | — |
| C1 shuffled | 52.40 ± 6.90 (N=15) | 52.00 ± 5.66 (N=15) | 55.33 ± 4.11 (N=15) | 55.87 ± 3.61 (N=15) |
| C2 filler | 57.60 ± 3.67 (N=15) | 54.21 ± 6.35 (N=14) | 53.60 ± 6.12 (N=15) | 53.73 ± 6.88 (N=15) |
| C3 own | 50.53 ± 9.48 (N=15) | 54.80 ± 4.89 (N=15) | 52.80 ± 5.60 (N=15) | 55.20 ± 6.27 (N=15) |


### 1c. Late slope (round 10 − round 6 cumulative balance)

| Condition | game_myth uninformed | game_myth informed | myth_game uninformed | myth_game informed |
| --- | --- | --- | --- | --- |
| no-A1 baseline | 44.27 ± 16.39 (N=15) | 44.67 ± 11.47 (N=15) | 42.93 ± 13.38 (N=15) | 47.87 ± 10.97 (N=15) |
| real A1 (partner myth) | 57.73 ± 3.86 (N=15) | — | 56.15 ± 4.26 (N=13) | — |
| C1 shuffled | 58.27 ± 3.26 (N=15) | 54.67 ± 5.20 (N=15) | 56.93 ± 3.26 (N=15) | 56.13 ± 3.90 (N=15) |
| C2 filler | 58.53 ± 3.14 (N=15) | 55.14 ± 5.38 (N=14) | 58.53 ± 3.14 (N=15) | 54.00 ± 5.32 (N=15) |
| C3 own | 56.67 ± 6.48 (N=15) | 54.00 ± 5.22 (N=15) | 55.27 ± 6.43 (N=15) | 56.00 ± 4.62 (N=15) |


### 1d. Per-round mean cumulative dyad balance, game_myth uninformed

| Round | no-A1 baseline | real A1 (partner myth) | C1 shuffled | C2 filler | C3 own |
| --- | --- | --- | --- | --- | --- |
| R1 | 11.9 | 11.7 | 11.5 | 11.5 | 11.3 |
| R2 | 25.5 | 25.7 | 23.3 | 25.3 | 24.3 |
| R3 | 38.7 | 38.7 | 36.9 | 40.3 | 37.1 |
| R4 | 50.8 | 51.3 | 49.9 | 54.4 | 48.8 |
| R5 | 61.8 | 65.3 | 63.9 | 69.1 | 61.8 |
| R6 | 73.5 | 78.5 | 78.1 | 82.5 | 74.1 |
| R7 | 85.3 | 92.6 | 92.5 | 96.6 | 88.7 |
| R8 | 96.9 | 107.5 | 106.9 | 111.1 | 102.3 |
| R9 | 107.9 | 121.8 | 121.4 | 126.1 | 116.3 |
| R10 | 117.7 | 136.3 | 136.4 | 141.1 | 130.8 |


### 1e. Per-round mean cumulative dyad balance, myth_game uninformed

| Round | no-A1 baseline | real A1 (partner myth) | C1 shuffled | C2 filler | C3 own |
| --- | --- | --- | --- | --- | --- |
| R1 | 12.2 | 11.6 | 11.7 | 11.5 | 12.5 |
| R2 | 23.5 | 24.5 | 24.9 | 24.4 | 25.5 |
| R3 | 36.9 | 38.1 | 39.1 | 38.7 | 39.3 |
| R4 | 51.9 | 52.5 | 53.3 | 51.6 | 53.1 |
| R5 | 64.1 | 66.4 | 67.0 | 65.1 | 65.3 |
| R6 | 75.1 | 80.6 | 81.3 | 79.5 | 79.7 |
| R7 | 87.1 | 93.6 | 95.8 | 94.5 | 93.4 |
| R8 | 97.5 | 108.6 | 110.3 | 109.3 | 106.2 |
| R9 | 108.1 | 123.6 | 124.5 | 123.9 | 120.8 |
| R10 | 118.0 | 136.8 | 138.3 | 138.0 | 135.0 |


## 2. Recovery from defection events

Average rounds-to-recovery after a per-round payoff drop > 1 SD below the run mean (cell value = mean ± SD over events; n_runs_with_events shown). Lower is faster recovery.

| Condition | game_myth uninformed | game_myth informed | myth_game uninformed | myth_game informed |
| --- | --- | --- | --- | --- |
| no-A1 baseline | 1.67 ± 0.94 (events=12, runs_w_events=9/15) | 1.40 ± 0.61 (events=15, runs_w_events=9/15) | 1.24 ± 0.42 (events=17, runs_w_events=10/15) | 1.77 ± 0.79 (events=22, runs_w_events=13/15) |
| real A1 (partner myth) | 1.79 ± 0.87 (events=24, runs_w_events=15/15) | — | 1.86 ± 0.89 (events=21, runs_w_events=12/13) | — |
| C1 shuffled | 1.58 ± 0.88 (events=26, runs_w_events=15/15) | 2.00 ± 0.98 (events=21, runs_w_events=12/15) | 1.75 ± 0.94 (events=20, runs_w_events=14/15) | 1.74 ± 0.84 (events=31, runs_w_events=15/15) |
| C2 filler | 1.27 ± 0.45 (events=22, runs_w_events=15/15) | 1.65 ± 0.73 (events=20, runs_w_events=13/14) | 1.29 ± 0.55 (events=21, runs_w_events=14/15) | 1.57 ± 0.71 (events=23, runs_w_events=13/15) |
| C3 own | 1.57 ± 0.85 (events=21, runs_w_events=14/15) | 1.62 ± 0.74 (events=26, runs_w_events=13/15) | 1.60 ± 0.80 (events=20, runs_w_events=13/15) | 1.71 ± 1.06 (events=24, runs_w_events=14/15) |


## 3. Cooperation stability (std of trust_ratio across rounds)

Per-run std of `sent / endowment` across the 10 rounds, averaged across seeds. Lower = more stable.

| Condition | game_myth uninformed | game_myth informed | myth_game uninformed | myth_game informed |
| --- | --- | --- | --- | --- |
| no-A1 baseline | 0.17 ± 0.05 (N=15) | 0.18 ± 0.06 (N=15) | 0.19 ± 0.05 (N=15) | 0.17 ± 0.06 (N=15) |
| real A1 (partner myth) | 0.10 ± 0.03 (N=15) | — | 0.10 ± 0.04 (N=13) | — |
| C1 shuffled | 0.10 ± 0.04 (N=15) | 0.10 ± 0.03 (N=15) | 0.08 ± 0.03 (N=15) | 0.08 ± 0.02 (N=15) |
| C2 filler | 0.09 ± 0.04 (N=15) | 0.11 ± 0.04 (N=14) | 0.11 ± 0.05 (N=15) | 0.12 ± 0.05 (N=15) |
| C3 own | 0.14 ± 0.05 (N=15) | 0.10 ± 0.03 (N=15) | 0.12 ± 0.05 (N=15) | 0.10 ± 0.05 (N=15) |


## 4. Linguistic convergence (cosine sim between agent_1 and agent_2 myth per round)

### 4a. Mean per-round cosine similarity (averaged over rounds within each run, then across runs)

| Condition | game_myth uninformed | game_myth informed | myth_game uninformed | myth_game informed |
| --- | --- | --- | --- | --- |
| no-A1 baseline | 0.66 ± 0.04 (N=15) | 0.65 ± 0.04 (N=15) | 0.63 ± 0.06 (N=15) | 0.66 ± 0.05 (N=15) |
| real A1 (partner myth) | 0.71 ± 0.07 (N=15) | — | 0.71 ± 0.07 (N=13) | — |
| C1 shuffled | 0.63 ± 0.05 (N=15) | 0.62 ± 0.05 (N=15) | 0.64 ± 0.05 (N=15) | 0.64 ± 0.05 (N=15) |
| C2 filler | 0.67 ± 0.05 (N=15) | 0.66 ± 0.07 (N=14) | 0.66 ± 0.06 (N=15) | 0.66 ± 0.05 (N=15) |
| C3 own | 0.70 ± 0.05 (N=15) | 0.66 ± 0.05 (N=15) | 0.69 ± 0.06 (N=15) | 0.66 ± 0.06 (N=15) |


### 4b. Convergence slope (late-half − early-half mean cosine sim, per run)

Positive = myths get more similar over time.

| Condition | game_myth uninformed | game_myth informed | myth_game uninformed | myth_game informed |
| --- | --- | --- | --- | --- |
| no-A1 baseline | 0.05 ± 0.05 (N=15) | 0.06 ± 0.05 (N=15) | 0.07 ± 0.10 (N=15) | 0.05 ± 0.06 (N=15) |
| real A1 (partner myth) | 0.10 ± 0.05 (N=15) | — | 0.11 ± 0.06 (N=13) | — |
| C1 shuffled | 0.02 ± 0.07 (N=15) | 0.02 ± 0.06 (N=15) | 0.01 ± 0.08 (N=15) | 0.03 ± 0.05 (N=15) |
| C2 filler | 0.06 ± 0.08 (N=15) | 0.05 ± 0.08 (N=14) | 0.05 ± 0.08 (N=15) | 0.06 ± 0.05 (N=15) |
| C3 own | 0.07 ± 0.04 (N=15) | 0.06 ± 0.07 (N=15) | 0.08 ± 0.06 (N=15) | 0.10 ± 0.09 (N=15) |


## 5. Trust-ratio drift (late-half − early-half mean trust_ratio per run)

Positive = trust grew over the game; negative = trust eroded.

| Condition | game_myth uninformed | game_myth informed | myth_game uninformed | myth_game informed |
| --- | --- | --- | --- | --- |
| no-A1 baseline | -0.06 ± 0.20 (N=15) | -0.07 ± 0.15 (N=15) | -0.10 ± 0.17 (N=15) | -0.05 ± 0.16 (N=15) |
| real A1 (partner myth) | 0.06 ± 0.09 (N=15) | — | 0.04 ± 0.07 (N=13) | — |
| C1 shuffled | 0.09 ± 0.08 (N=15) | 0.04 ± 0.07 (N=15) | 0.04 ± 0.03 (N=15) | 0.02 ± 0.07 (N=15) |
| C2 filler | 0.03 ± 0.06 (N=15) | 0.02 ± 0.09 (N=14) | 0.08 ± 0.08 (N=15) | 0.02 ± 0.08 (N=15) |
| C3 own | 0.07 ± 0.12 (N=15) | -0.00 ± 0.08 (N=15) | 0.04 ± 0.10 (N=15) | 0.01 ± 0.08 (N=15) |


## Verdict

For each metric we compare real-A1 against the strongest control (C2 filler, which beat real-A1 on the headline dyad-balance metric). Cell used: **game_myth uninformed** (the only cell where real-A1 has the full N=15 sample). myth_game uninformed gap is reported in parentheses where available.

- **Final dyad balance**: game_myth: real-A1 136.267, C2 filler 141.067, Δ = -4.800 (worse for real-A1 if higher is desired); myth_game: Δ = -1.231
- **Early slope (R1→R5 cum)**: game_myth: real-A1 53.600, C2 filler 57.600, Δ = -4.000 (worse for real-A1 if higher is desired); myth_game: Δ = +1.169
- **Late slope (R6→R10 cum)**: game_myth: real-A1 57.733, C2 filler 58.533, Δ = -0.800 (worse for real-A1 if higher is desired); myth_game: Δ = -2.379
- **Recovery rounds-to-recovery**: game_myth: real-A1 1.792, C2 filler 1.273, Δ = +0.519 (worse for real-A1 if lower is desired); myth_game: Δ = +0.571
- **Trust-ratio std (lower = stabler)**: game_myth: real-A1 0.099, C2 filler 0.092, Δ = +0.007 (worse for real-A1 if lower is desired); myth_game: Δ = -0.007
- **Linguistic mean cosine sim (higher = more convergent)**: game_myth: real-A1 0.712, C2 filler 0.673, Δ = +0.039 (better for real-A1 if higher is desired); myth_game: Δ = +0.047
- **Linguistic convergence slope (higher = late > early)**: game_myth: real-A1 0.104, C2 filler 0.059, Δ = +0.045 (better for real-A1 if higher is desired); myth_game: Δ = +0.059
- **Trust drift (positive = trust grew)**: game_myth: real-A1 0.057, C2 filler 0.028, Δ = +0.029 (better for real-A1 if higher is desired); myth_game: Δ = -0.037

### Concluding paragraph

On the four economic / behavioural sub-metrics (final balance, early slope, late slope, recovery, stability, trust drift), real-A1 is statistically indistinguishable from C2 filler — and on most of them it is numerically *worse* (notably recovery: +0.52 rounds slower in game_myth, +0.57 in myth_game). The cumulative-balance trajectory in §1d/1e shows real-A1, C1 shuffled and C2 filler tracking each other within ~5 points at every round; only no-A1 baseline diverges. So the headline §4.6 result really is "any prose injection at A1 lifts cooperation" as far as game outcomes are concerned.

The single sub-metric where real-A1 is distinguishable from the controls is **linguistic convergence**. Real-A1 mean cosine sim between Agent_1 and Agent_2 myths is 0.71 (game_myth) / 0.71 (myth_game) vs C2 filler 0.67 / 0.66 and C1 shuffled 0.63 / 0.64. The convergence *slope* (late half − early half) is the sharpest separator: real-A1 0.10 / 0.11 vs C2 filler 0.06 / 0.05 vs C1 shuffled 0.02 / 0.01. C3 own-myth is the closest comparator on this metric (0.07 / 0.08 mean slope, 0.70 / 0.69 mean sim) — real-A1 still beats it but the margin is narrower. The directional pattern is consistent across both task orders.

**Recommendation for §4.6**: the cooperation-lift claim cannot be uniquely attributed to partner-myth content — that battle is lost on the dyad-balance and recovery metrics. The honest narrower claim that survives is: partner-myth visibility produces measurably more linguistic convergence (≈+0.04 mean cosine sim, ≈+0.05 convergence slope) than equally-sized non-partner prose injection, while delivering economic outcomes indistinguishable from filler. Whether that linguistic convergence is causally important enough to anchor §4.6, or is just a downstream artifact of agents being shown each other's text, is a separate question this analysis does not resolve.

## 6. Echo-confound disambiguation

Disambiguation of whether real-A1's elevated within-round Agent_1↔Agent_2 myth similarity (reported in §4) is driven by agents echoing the injected partner-myth text or by genuine coordination on top of any echo. For each agent X and round R where the injected text was non-empty we compute three cosines on `sentence-transformers/all-MiniLM-L6-v2` embeddings:

- `echo_cos(X, R) = cos(myth_X(R), injected_text_X(R))`
- `partner_cos(X, R) = cos(myth_X(R), partner_myth(R−1))` — undefined for R=1.
- `own_cos(X, R) = cos(myth_X(R), own_myth(R−1))` — autocorrelation baseline; undefined for R=1.

Aggregation: per-run mean across rounds where each cosine is defined, then mean ± SD across runs (N = number of runs that contributed at least one valid round for that cosine).

Reconstruction of injected texts replicates the runtime in `games/trust_game_noisy.py`: real-A1 uses the partner's most recent prior myth; C3 own uses the agent's own most recent prior myth; C1 shuffled deterministically samples from the cross-dyad pool at `v4_direct_provider_baseline/baseline_v4_mem3_direct/gpt-5-nano` using `sha256(f"{run_seed}|{agent_id}|{turn}|{task_order_joined}")[:8]` mod pool size; C2 filler samples from `src.control_text_pool.FILLER_PARAGRAPHS` with the same seed_key. The no-A1 baseline has no injection so `echo_cos` is undefined.


### 6a. game_myth uninformed

| Condition | echo_cos | partner_cos | own_cos |
| --- | --- | --- | --- |
| no-A1 baseline | — | 0.59 ± 0.06 (N=15) | 0.61 ± 0.06 (N=15) |
| real A1 (partner myth) | 0.70 ± 0.07 (N=15) | 0.70 ± 0.07 (N=15) | 0.71 ± 0.07 (N=15) |
| C1 shuffled | 0.58 ± 0.03 (N=15) | 0.60 ± 0.05 (N=15) | 0.65 ± 0.05 (N=15) |
| C2 filler | 0.09 ± 0.03 (N=15) | 0.64 ± 0.05 (N=15) | 0.69 ± 0.05 (N=15) |
| C3 own | 0.72 ± 0.06 (N=15) | 0.68 ± 0.07 (N=15) | 0.72 ± 0.06 (N=15) |

### 6b. myth_game uninformed

| Condition | echo_cos | partner_cos | own_cos |
| --- | --- | --- | --- |
| no-A1 baseline | — | 0.57 ± 0.07 (N=15) | 0.60 ± 0.07 (N=15) |
| real A1 (partner myth) | 0.68 ± 0.08 (N=13) | 0.68 ± 0.08 (N=13) | 0.69 ± 0.07 (N=13) |
| C1 shuffled | 0.60 ± 0.02 (N=15) | 0.62 ± 0.04 (N=15) | 0.68 ± 0.05 (N=15) |
| C2 filler | 0.08 ± 0.01 (N=15) | 0.64 ± 0.06 (N=15) | 0.70 ± 0.04 (N=15) |
| C3 own | 0.73 ± 0.06 (N=15) | 0.66 ± 0.06 (N=15) | 0.73 ± 0.06 (N=15) |

### Verdict

**game_myth uninformed.**
- real-A1: echo_cos = 0.700 (SD 0.074, N=15), partner_cos = 0.700 (SD 0.074, N=15), own_cos = 0.707 (SD 0.074, N=15).
- C1 shuffled (cross-dyad myth injected): echo_cos = 0.585 (SD 0.029, N=15), partner_cos = 0.600 (SD 0.048, N=15), own_cos = 0.651 (SD 0.046, N=15).
- C2 filler (hydrology/concrete paragraph injected): echo_cos = 0.086 (SD 0.029, N=15), partner_cos = 0.640 (SD 0.054, N=15), own_cos = 0.690 (SD 0.054, N=15).
- C3 own (agent's own prior myth re-injected): echo_cos = 0.722 (SD 0.056, N=15), partner_cos = 0.677 (SD 0.066, N=15), own_cos = 0.722 (SD 0.056, N=15).
- no-A1 baseline (no injection): partner_cos = 0.589 (SD 0.056, N=15), own_cos = 0.609 (SD 0.060, N=15).
- **Decomposition of partner_cos:** baseline (no injection) = 0.589; C2 filler (irrelevant injection) = 0.640 (Δ vs baseline = +0.052); C1 shuffled (wrong myth injected) = 0.600; real-A1 (true partner myth injected) = 0.700 (Δ vs C2 filler = +0.059; Δ vs C1 shuffled = +0.100).
- **Echo channel strength:** echo_cos lift real-A1 − C1 shuffled = +0.115; real-A1 − C2 filler = +0.614. C1 shuffled's echo_cos ≈ 0.585 is the natural cosine between two unrelated myths in this myth distribution; C2 filler's echo_cos ≈ 0.086 reflects the (very low) cosine between a hydrology paragraph and a myth. The high real-A1 echo_cos = 0.700 therefore mostly tracks the natural baseline myth-myth similarity (~0.585), plus the small extra produced by partner-myth visibility. So 'echo' here is not a pure copy-paste effect; agents do not literally reproduce the injected myth, they produce a fresh myth whose overall topical/stylistic profile is somewhat closer to the injected one than to a random myth.

**myth_game uninformed.**
- real-A1: echo_cos = 0.683 (SD 0.078, N=13), partner_cos = 0.683 (SD 0.078, N=13), own_cos = 0.686 (SD 0.070, N=13).
- C1 shuffled (cross-dyad myth injected): echo_cos = 0.597 (SD 0.024, N=15), partner_cos = 0.621 (SD 0.044, N=15), own_cos = 0.677 (SD 0.047, N=15).
- C2 filler (hydrology/concrete paragraph injected): echo_cos = 0.083 (SD 0.015, N=15), partner_cos = 0.636 (SD 0.060, N=15), own_cos = 0.696 (SD 0.043, N=15).
- C3 own (agent's own prior myth re-injected): echo_cos = 0.726 (SD 0.058, N=15), partner_cos = 0.664 (SD 0.059, N=15), own_cos = 0.726 (SD 0.057, N=15).
- no-A1 baseline (no injection): partner_cos = 0.568 (SD 0.067, N=15), own_cos = 0.600 (SD 0.065, N=15).
- **Decomposition of partner_cos:** baseline (no injection) = 0.568; C2 filler (irrelevant injection) = 0.636 (Δ vs baseline = +0.067); C1 shuffled (wrong myth injected) = 0.621; real-A1 (true partner myth injected) = 0.683 (Δ vs C2 filler = +0.047; Δ vs C1 shuffled = +0.062).
- **Echo channel strength:** echo_cos lift real-A1 − C1 shuffled = +0.086; real-A1 − C2 filler = +0.600. C1 shuffled's echo_cos ≈ 0.597 is the natural cosine between two unrelated myths in this myth distribution; C2 filler's echo_cos ≈ 0.083 reflects the (very low) cosine between a hydrology paragraph and a myth. The high real-A1 echo_cos = 0.683 therefore mostly tracks the natural baseline myth-myth similarity (~0.597), plus the small extra produced by partner-myth visibility. So 'echo' here is not a pure copy-paste effect; agents do not literally reproduce the injected myth, they produce a fresh myth whose overall topical/stylistic profile is somewhat closer to the injected one than to a random myth.

**Synthesis (honest read).** Real-A1's elevated Agent_1↔Agent_2 within-round myth similarity (0.71 in §4) is *partially* an echo confound and *partially* genuine coordination. Decomposition: (i) the **no-A1 → C2 filler** step (≈+0.05 in partner_cos) shows that simply playing trust-game rounds together with any added prompt volume produces ~0.05 extra similarity to the partner's prior myth — game-coupling alone, no partner-myth content needed; (ii) the **C2 filler → real-A1** step (≈+0.05–0.06 in partner_cos) is the residual lift attributable to partner-myth visibility; (iii) within that residual, real-A1's echo_cos (0.700) is only modestly above C1 shuffled's echo_cos to an unrelated injected myth (0.585), so most of the signal is *not* literal echoing of the injected text — it is the agent producing a new myth whose topic/style is shifted by what they just read. C3 own (which re-injects the agent's own prior myth) gives a comparable or slightly higher partner_cos than real-A1 in game_myth (0.677 vs 0.700), suggesting the lift is driven as much by 'recent narrative material in the prompt' as by 'specifically the partner's narrative material'. **Bottom line:** the within-round similarity advantage in real-A1 is real, but ~half of the gap above the no-A1 baseline is explained by generic added-prompt-volume effects (C2 filler matches that part), and the remaining ~half is a soft echo of injected narrative content rather than an exclusive partner-coordination effect. §4.6 should not claim partner-myth visibility uniquely produces convergence — it produces *narrative* convergence comparable to what any recent narrative injection (own or partner) produces.
