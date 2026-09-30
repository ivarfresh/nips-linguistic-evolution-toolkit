# Mixed groups compared with the average of their parts, 2026-09-30

Paper table `tab:mixed-vs-average`. For each mixed group and task order:
final resources per agent (max 75) in the mixed runs, minus the
composition-weighted mean of the matching single-model groups
(k/8 · R_A + (8−k)/8 · R_B for a population with k agents of A; the mean of
the two single-model pairs for a dyad). Regenerate with
`python analyses/mixed_vs_average.py` (reads the per-run tables in
`mixed_model_dyads_20260917/decisions.csv` and
`mixed_model_populations_20260918/agent_finals.csv`; no API calls).

- Interval: 95% percentile bootstrap, runs resampled within each group (20,000 draws).
- Test: Welch t-test on the same linear contrast (Welch–Satterthwaite df),
  Holm-corrected across all 27 cells (`welch_p_holm`). 13 cells have p < 0.05
  uncorrected, 7 after Holm; the paper's bold marks use the Holm value.
  Mann–Whitney is not used: the comparison value is a weighted sum of two
  other groups, not a sample, so a two-sample rank test does not apply.
- n = 6 runs per mixed dyad cell, 5 per population and single-model cell.
  Percentile bootstrap intervals at this n run narrow; in six cells the
  interval excludes zero while Welch p > 0.05. Read those as weak.
