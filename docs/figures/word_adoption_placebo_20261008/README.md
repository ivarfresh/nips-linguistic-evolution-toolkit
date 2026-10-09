# Word adoption against the author's-next-myth placebo (paper Section 4.5), 2026-10-08

Source for the sentence "the shown myth also scores above that placebo, at 1.2–1.3 times
its rate in every setting (72–99% of runs; Holm-corrected p < 10⁻⁴)" in Results 4.5, and
for the Methods sentence defining the word-adoption placebo. Computed on 2026-10-08 in a
scratch folder; committed here on 2026-10-09 with the script. Regenerate with
`python analyses/word_adoption_placebo.py` (no API calls, ~1 min).

## What it does

Same n = 10 September corpus as Figure 5a (300 runs, 14,678 reader–myth pairs; one pair
dropped because the placebo myth is missing or under 20 words). The script first
reproduces the paper's shown-vs-unseen numbers exactly (`uptake_children.csv` and
`reuse_summary.csv` of `linguistic_analysis_n10_20261001`), then scores a placebo: the
shown author's myth from the round in which the reader writes. The reader cannot have
seen it before writing. Every share uses the paper's definition: of the comparison
myth's words that are new to the reader, the share the reader uses in its next myth.
Per-run means, paired Wilcoxon over runs, Holm across the five settings.

## Result

Shares are means over runs (± sd over runs), in percent.

| Setting | Runs | Shown % | Unseen % | Placebo % | Shown / unseen | Shown / placebo (runs higher) | Holm p | Placebo / unseen |
|---|---|---|---|---|---|---|---|---|
| Dyads, single-model | 60 | 15.5 (±5.1) | 6.4 (±1.1) | 12.3 (±2.4) | 2.41 | 1.26 (44/60) | 2.2e-5 | 1.92 |
| Dyads, mixed (other family) | 60 | 6.6 (±1.4) | 4.5 (±0.7) | 5.6 (±1.3) | 1.47 | 1.18 (48/60) | 3.3e-8 | 1.25 |
| Populations, single-model | 60 | 13.3 (±2.3) | 8.5 (±1.0) | 11.0 (±1.2) | 1.57 | 1.21 (43/60) | 3.2e-7 | 1.30 |
| Populations, mixed, same family | 120 | 14.0 (±1.9) | 7.8 (±0.8) | 10.4 (±1.2) | 1.80 | 1.34 (119/120) | 1.1e-20 | 1.34 |
| Populations, mixed, other family | 120 | 6.0 (±1.5) | 4.4 (±0.6) | 5.2 (±1.0) | 1.37 | 1.16 (104/120) | 2.6e-16 | 1.18 |

The placebo accounts for 43–65% of the shown-vs-unseen gap. The shown myth beats it in
every setting. Checks in the script: an unseen baseline from the reader's own round
barely moves; authors repeat 19–29% of their words from one myth to the next, so the
placebo shares some shown-myth words and the placebo ratio is conservative; a cleaner
split (shown-myth words absent from the author's next myth, against next-myth words the
reader was never shown anywhere) gives 1.58–1.98×, Holm p ≤ 2.2e-9, and the symmetric
version 1.33–1.76×, Holm p ≤ 1.1e-5.

## Files

- `placebo_summary.csv`: one row per setting; every contrast's ratio, runs positive,
  raw and Holm p (`shown_vs_placebo_*` is the paper's test).
- `placebo_per_run.csv`: per-run means of every adoption share.
- `provenance.json`: the 300 run finals behind the corpus and the hashes of this folder's files.
- `data/analysis/word_adoption_placebo_20261008/placebo_children.csv` (gitignored, 5.9 MB):
  per reader–myth pair, the shown/unseen/placebo shares and the variant pools; mirrored on
  the shared private HF dataset under `ivarfresh/analysis/word_adoption_placebo_20261008/`.

Inputs are the gitignored `data/analysis/linguistic_n10_20261001/` tables, mirrored on
the shared private HF dataset under `ivarfresh/analysis/linguistic_n10_20261001/`.
