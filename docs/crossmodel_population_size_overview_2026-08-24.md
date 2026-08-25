# Cross-model population-size comparison (2026-08-24)

## Question

How does average final balance differ between the corrected 2-agent repeated-dyad and 8-agent rotating-population versions for one Claude, GPT, and Gemini model, without defectors or explicit punishment?

All cells use the corrected informed signed-noise protocol and the same three task orders: game only, game → myth, and myth → game. The plotted points are replicate means and the error bars are 95% t intervals.

## Results

Average final balance per agent:

| Model | Population regime | Game only | Game → Myth | Myth → Game |
|---|---|---:|---:|---:|
| Claude Sonnet 4.5 (n=10/cell) | 2-agent repeated dyad | 64.86 | 62.94 | 69.83 |
| Claude Sonnet 4.5 (n=10/cell) | 8-agent rotating population | 59.26 | 60.51 | 66.33 |
| GPT-5 Nano (n=5/cell) | 2-agent repeated dyad | 67.10 | 63.97 | 66.20 |
| GPT-5 Nano (n=5/cell) | 8-agent rotating population | 62.64 | 62.34 | 66.26 |
| Gemini 3.7 Flash (n=3/cell) | 2-agent repeated dyad | 75.00 | 75.00 | 75.00 |
| Gemini 3.7 Flash (n=3/cell) | 8-agent rotating population | 75.00 | 75.00 | 75.00 |

The Claude comparison shows the clearest population-regime difference, especially in game only. GPT-5 Nano points in a similar direction for game only, but the small dyad sample is variable and the intervals are wide. Gemini 3.7 Flash is at the maximum possible balance in every cell, so this outcome cannot distinguish population size or task order for that model.

These are descriptive comparisons, not adequately powered confirmatory tests. In addition, the contrast is between a repeated fixed dyad and a rotating 8-agent population; it therefore combines population size with partner-matching and reputation structure rather than isolating agent count alone.

## Validation and provenance

- The missing corrected GPT and Gemini dyad arms were run specifically for this comparison using the seed schedules from their existing 8-agent screens.
- All 24 new dyad populations passed the protocol audit: 800 accepted calls, 480 noise checks, and zero audit issues.
- The protocol and configuration were frozen before collecting these runs in commit `e5e3a7482479530125ace6d17ae8be6cfde105c6`.
- A small set of outputs generated while the analysis file was being edited was excluded and preserved separately. The affected myth → game cells were rerun from a clean worktree before inclusion.
- The accepted new dyad runs share configuration hash `d3307b5def6fff477a7aab80df79de5db79ae0f96591423cdbcfa84915ae4828` and have `code_dirty=false`.

## Artifacts

- Figure: `docs/figures/crossmodel_population_size_20260824/final_balance_by_model_and_population.png`
- Cell summary: `docs/figures/crossmodel_population_size_20260824/summary.csv`
- Run-level metrics: `docs/figures/crossmodel_population_size_20260824/run_metrics.csv`
- Audit table: `docs/figures/crossmodel_population_size_20260824/new_dyad_audit.csv`
- Machine-readable provenance: `docs/figures/crossmodel_population_size_20260824/provenance.json`
- Frozen protocol: `docs/crossmodel_population_size_dyads_protocol_2026-08-24.md`
