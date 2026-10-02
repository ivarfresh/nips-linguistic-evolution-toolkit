# Table 1 at n = 10 (mixed groups vs the average of their parts), 2026-10-01

Same contrast, bootstrap and Welch/Holm code as `mixed_vs_average_20260930`
(PR #10), now with 10 runs in every cell: the original runs plus the
2026-10-01 extension (`scripts/run_table1_n10_extension.py`, 216 runs, audit
216/216, $179.25 at standard rates). Regenerate with
`python analyses/table1_n10.py`.

- `mixed_vs_average.csv`: per cell mean difference, 95% bootstrap interval,
  Welch t/df/p and Holm-adjusted p (27 cells).
- `table_rows.tex`: LaTeX rows; bold = uncorrected Welch p < 0.05 (the
  current caption's rule). 16 cells pass uncorrected, 11 after Holm.
- `dyad_decisions.csv`, `population_agent_finals.csv`: per-run inputs.

Eight extension dyads (all game→myth with Gemini) were quarantined because a
Gemini call returned HTTP 503/timeout before an in-run retry succeeded, and
were resampled under the same seed; one 8-agent Sonnet run was rerun after a
role-key error (`{'return': …}` while sending). Quarantined files are kept in
`data/json/noise_experiments/table1_n10_extension_20261001/quarantine/` and
excluded here.
