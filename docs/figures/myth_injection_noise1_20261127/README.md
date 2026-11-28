# Myth-injection ablation at the main-figure noise setting: figures

Generate with `python analyses/plot_myth_injection_noise1.py`. The script checks
every final against the completion receipts, re-derives each value from the
last-round balances, and writes `summary.json` and `provenance.json`.

- `myth_injection_dyads.png`, `myth_injection_populations.png`: final resources
  per agent (25 to 75) by injected text. Bars are means over five runs, error
  bars the sample SD, dots the individual runs. Dashed line: full cooperation.
  Dotted line: mean with nothing injected.
- `myth_injection_by_donor_score.png`: each run that received a myth, plotted
  against the final resources per agent of the run the myth came from. The thin
  diagonal marks equal values. r is the Pearson correlation over the 30 runs; it
  is descriptive, since the points are five donors per cell and the outcome is
  bounded below by the no-injection level and above by 75.

Results and limits: `docs/research/myth_injection_noise1_20261127/README.md`.
