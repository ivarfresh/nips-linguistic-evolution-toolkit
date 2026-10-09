# Moral matching against the author's-next-myth placebo (paper Section 4.5), 2026-10-08

Source for the Results 4.5 sentence "The moral of a reader's next myth matches that of the
myth it read 2.7 percentage points more often than that of an unseen myth of the same family
(p = 0.015; 1.6 points across families, p = 0.13), but it matches the author's next myth,
which the reader never saw, about as often (shown minus placebo +0.8 points within a family,
95% CI −1.7 to 3.3, p = 0.54; −1.1 across families, p = 0.67)". Computed on 2026-10-08 in a
scratch folder; committed here on 2026-10-09 with the script. Regenerate with
`python analyses/moral_placebo.py` (no API calls, ~1 min).

## What it does

September n = 10 corpus, GLM-5.2 moral labels (be generous / be fair / be cautious). The
script first reproduces the stored `moral_uptake_children.csv` exactly (14,634 reader–myth
pairs; same filter as `analyses/moral_carryover.py`), then adds for each pair the shown
author's next myth (the placebo, written in the same round as the reader's myth) and the
next myths of the same unseen authors (the placebo's own baseline). Same-label share, per-run
means, paired Wilcoxon over runs, 10,000-resample bootstrap intervals, no Holm. The paper's
numbers are the 8-agent mixed populations under myth→game (60 runs); the committed n = 10
moral table pools task orders, which is why its 3.5 / 0.8 points differ from the 2.7 / 1.6
here.

## Result (8-agent mixed populations, myth→game, 60 runs)

| Exposure | Pairs | Shown % | Unseen % | Placebo % | Shown − unseen (p) | Shown − placebo [95% CI] (p) |
|---|---|---|---|---|---|---|
| Same family | 2,455 | 69.2 | 66.5 | 68.4 | +2.7 (0.015) | +0.8 [−1.7, 3.3] (0.54) |
| Other family | 1,632 | 59.8 | 58.2 | 61.0 | +1.6 (0.13) | −1.1 [−3.9, 1.5] (0.67) |

No setting or task order has shown > placebo at p < 0.05 (`placebo_by_setting_task_order.csv`).
Caveat: the author always read the reader's previous myth before writing the placebo
(checked: 100% of pairs), so the placebo errs high; the interval caps a hidden reading effect
at about 3 points. Wording of the moral summaries (mpnet cosine): within a family the shown
myth is closer than the placebo by +0.0075, p = 0.0016; across families no difference
(`placebo_main_moral_cos.csv`).

## Files

- `paper_rows_8agent_mixed_myth_game.csv`: the 2.7 / 1.6 rows on the full pair set.
- `placebo_main_8agent_mixed_myth_game.csv`: the placebo contrast on the pairs where every
  quantity exists (4,087 of 4,090).
- `placebo_by_setting_task_order.csv`: the same for every setting × exposure × task order.
- `placebo_main_moral_cos.csv`: the cosine version of the main table.
- `moral_placebo_children.csv`: per reader–myth pair.

Inputs are the gitignored `data/analysis/linguistic_n10_20261001/` tables (myths, GLM-5.2
labels, moral-summary embeddings), mirrored on the shared private HF dataset under
`ivarfresh/analysis/linguistic_n10_20261001/`.
