# Do myths change a mixed group's margin over its parts? (paper Table s-interaction), 2026-10-09

Source for the Results 4.2 sentence "myths widen the margin over the parts in 17 of 18 cells,
4 of them significantly after Holm correction across those 18" and for supplementary
Table s-interaction. Regenerate with `python analyses/mixed_vs_average_interaction.py`
(reads only the committed per-run tables of `mixed_vs_average_n10_20261001/`; no raw JSON,
no API calls, seconds).

## What it does

For each of the nine mixed groups and each myth task order: the mix-minus-parts difference of
Table 1 under that task order minus the same difference in game-only play. One Welch contrast
over the six groups involved (mixed group and its two single-model parts, under both task
orders), composition weights as in `analyses/mixed_vs_average.py`, Welch–Satterthwaite
degrees of freedom, 95% t-interval, Holm across the 18 cells. n = 10 runs per group. A
percentile bootstrap interval (20,000 resamples within each group, seed 20261009) is kept in
`boot_ci_lo` / `boot_ci_hi` for reference; the paper prints the t-interval.

## Result

| Group | Game→myth minus game only | p / Holm | Myth→game minus game only | p / Holm |
|---|---|---|---|---|
| Sonnet + GPT dyad | 10.3 [1.4, 19.2] | 0.026 / 0.209 | 11.5 [3.6, 19.5] | 0.006 / 0.084 |
| Sonnet + Gemini dyad | 0.9 [−2.1, 3.8] | 0.551 / 1 | 1.1 [−2.6, 4.7] | 0.548 / 1 |
| Gemini + GPT dyad | **24.6** [20.4, 28.8] | <0.001 / <0.001 | **23.8** [20.0, 27.6] | <0.001 / <0.001 |
| 1 Gemini + 7 GPT | 9.7 [2.4, 17.0] | 0.012 / 0.122 | **15.4** [8.6, 22.2] | <0.001 / 0.002 |
| 2 Gemini + 6 GPT | 11.9 [3.3, 20.4] | 0.010 / 0.110 | 14.9 [4.1, 25.8] | 0.009 / 0.110 |
| 4 Gemini + 4 GPT | 5.7 [0.9, 10.6] | 0.023 / 0.207 | **7.2** [3.0, 11.5] | 0.002 / 0.032 |
| 1 GPT + 7 Sonnet | 1.7 [−1.8, 5.1] | 0.330 / 1 | −1.7 [−5.9, 2.6] | 0.431 / 1 |
| 2 GPT + 6 Sonnet | 5.9 [0.2, 11.7] | 0.045 / 0.313 | 4.2 [−1.9, 10.4] | 0.170 / 0.849 |
| 4 GPT + 4 Sonnet | 8.0 [0.2, 15.8] | 0.046 / 0.313 | 8.9 [2.5, 15.2] | 0.008 / 0.106 |

17 of 18 cells positive, 4 Holm-significant (bold). Both counts match the paper.

## Differences from the Overleaf table (commit f0d9475)

The table in the supplement was produced from a scratch computation that was never committed.
Rebuilt from the committed Table 1 inputs, a few entries differ in the last digit and
should be updated in Overleaf from `table_rows.tex`:

| Cell | Overleaf | Here |
|---|---|---|
| Sonnet + GPT dyad, game→myth, interval | [1.3, 19.2] | [1.4, 19.2] |
| Sonnet + GPT dyad, game→myth, Holm | 0.210 | 0.209 |
| 1 Gemini + 7 GPT, game→myth | 9.8 [2.4, 17.1], Holm 0.120 | 9.7 [2.4, 17.0], Holm 0.122 |
| 1 Gemini + 7 GPT, myth→game | 15.5 [8.7, 22.2] | 15.4 [8.6, 22.2] |
| 1 GPT + 7 Sonnet, game→myth | 1.6, p 0.333 | 1.7, p 0.330 |
| 2 GPT + 6 Sonnet, myth→game, Holm | 0.853 | 0.849 |
| 2 GPT + 6 Sonnet and 4 GPT + 4 Sonnet, game→myth, Holm | 0.312 | 0.313 |

The 1 Gemini + 7 GPT point estimates here are what Table 1's own committed differences give
(4.4 − (−5.3) = 9.7 and 10.1 − (−5.3) = 15.4), so the committed version is the consistent one.

## Files

- `interaction.csv`: per cell the two Table 1 differences, their difference, Welch t / df /
  p, Holm p, t-interval and bootstrap interval.
- `table_rows.tex`: the LaTeX rows for Table s-interaction (bold = Holm p < 0.05).
- `provenance.json`: the same 450 run finals as `mixed_vs_average_n10_20261001/` and the
  hashes of this folder's files.
