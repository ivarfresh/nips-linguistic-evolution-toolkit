# Frontier brief: rerun the September myth analyses on the frontier runs (2026-09-30)

Goal (Ivar): test everything we did on September (moral spread; what in the myths predicts
cooperation) on the MAIN frontier runs, homogeneous and mixed, and present a clear picture
per simulation and population, September beside frontier.

## Read first
- `docs/research/myth_predictors_20260930/README.md` (September synthesis: findings, rungs,
  caveats, settled specs) and `BRIEF.md` there (confounds, identification ladder R0-R5).
- `docs/figures/linguistic_analysis_20260923/README.md` (moral spread, uptake, labels).
- `docs/figures/frontier_rerun_20260918/README.md`, `docs/figures/frontier_main_mixed_20260928/README.md`.
- Your September lens folder under `docs/research/myth_predictors_20260930/<lens>/` (scripts to port).

## Data
- Worktree (work ONLY here): /Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/frontier-myth
- Frontier tables: `data/analysis/linguistic_frontier_20260930/` — myths.csv, decisions.csv,
  moral_labels_z-ai__glm-5.2.csv (label + summary), moral_labels_deepseek__deepseek-v4-flash.csv
  (may still be landing; robustness only; NOTE served with hidden reasoning, not strictly
  comparable to September DeepSeek), myth_rules_frontier_{glm,deepseek}.csv,
  myth_amount_check_frontier_z-ai__glm-5.2.csv, giving_scores_{glm,deepseek}.csv,
  embeddings_mpnet.npy, embeddings_moral_summary_mpnet.npy, testability_*.csv.
- September tables: `data/analysis/linguistic_20260923/` (SHARED symlink — read only, never write).
  September giving scores/measures are in the September lens folders / data/analysis/myth_predictors_20260930 is NOT in this worktree; read them from
  /Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/myth-predictors/data/analysis/myth_predictors_20260930/ and docs/research/myth_predictors_20260930/.
- Dataset switch: `analyses/linguistic_datasets.py`; `LINGUISTIC_DATASET=frontier` or
  `--dataset frontier` in linguistic_uptake.py, moral_carryover.py, alignment_vs_cooperation.py,
  myth_moral_judge.py. Families: Opus, GeminiPro, Sol (never pool with Sonnet/Gemini/GPT).

Frontier runs: 106 myth runs, 4,520 myths. 2-agent homogeneous 30 (600 myths), 2-agent mixed
36 (Opus+Sol, Opus+GeminiPro, GeminiPro+Sol × 2 task orders × 6, first sender alternates),
8-agent homogeneous 30 (2,400), 8-agent mixed 10 (one composition: 2 GeminiPro + 3 Opus + 3 Sol).

## Testability (from testability_*.csv; respect it)
- GeminiPro sends $5 with sd 0 everywhere: send effects NOT ESTIMABLE (ceiling).
- Opus myth→game near ceiling; game→myth round 1 locked at exactly $4.
- Sol is the only family with real send spread (round-1 myth→game 36 senders; game→myth open).
- Returns sd 0.02-0.11 around 0.45: weak.
- Mixed dyad blocks have 3 senders; mixed population has 10 runs (5 per task order). GeminiPro
  has 2 members there, so within-family shown-vs-unseen tests are impossible for it.
Report such cells as "not estimable: ceiling" / "not testable by design" / "underpowered",
never as a null.

## Settled specs (carry forward)
- R3 = 8-agent myth_game; the shown myth is always last round's partner's; use shown author ≠
  current partner and control for that author's last move toward the reader; future-myth
  placebo for spread claims; headline from the model WITHOUT the future myth (the
  future-controlled version is inflated by run×round FE).
- R4 own round-1 myth → round-1 send, senders only (verified for frontier: own myth is the only
  differing input; analyses/frontier_round1_identification.py).
- Holm within each frontier stratum; SE clustered by run; means ± sd over runs.

## Procedure
1. Reproduce your September headline through the ported/parameterized code and match the
   committed number (state it). Only then run frontier.
2. Run frontier per stratum: setting (2/8-agent) × homogeneous/mixed × family × task order.
3. Write outputs to `docs/research/frontier_myth_predictors_20260930/<lens>/` (scripts + CSVs +
   PNGs). Never write to the September data dirs.
4. Write `scorecard_rows.csv` in your folder, one row per (dataset september|frontier, setting,
   population e.g. "8 Opus" / "Opus+Sol dyads" / "2 GeminiPro + 3 Opus + 3 Sol",
   family, finding, effect, ci_low, ci_high, p, holm_p, n_runs, n_obs, status, note) where
   status ∈ {yes, no detectable effect, suggestive, not estimable: ceiling, not testable by
   design, underpowered}. Include the September rows for the same findings so the lead can put
   them side by side.
5. Report to the lead by SendMessage (report files are blocked): headline, per-stratum
   picture, what differs from September, caveats, paths. Split long reports into parts.
- No paid calls needed; if you truly need one, preflight line and cap $10.
- `pytest` has one pre-existing failure (stale linguistic hashes) — ignore it.
