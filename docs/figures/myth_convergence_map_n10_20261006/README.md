# Myth map at n = 10, and paper Figure 5b, 2026-10-06

The [September myth map](../myth_convergence_map_20261002/README.md) rerun on
the 300 myth runs (n = 10 per cell) behind paper Figure 5a, so both panels of
Figure 5 rest on the same runs. Method, figures and tables are as in the
September README; only the corpus differs (16,799 myths after dropping one
empty GPT response).

```
export LINGUISTIC_DATASET=september_n10   # provenance.json lists the n = 10 runs
python3 analyses/myth_convergence_map.py --dataset september_n10
python3 analyses/myth_map_significance.py --dataset september_n10
python3 analyses/plot_myth_map_panel.py   # paper Figure 5b
```

Free; reuses the cached embeddings in `data/analysis/linguistic_n10_20261001/`.

## Figure 5b: `myth_map_round10_8agent_myth_game.png`

8-agent populations, Myth → Game, one task order and size only (never pooled).
Dots and solid outlines are round-10 myths; dashed outlines are where the same
runs' round-1 myths sat. Left: 30 single-model populations (80 round-10 myths per
family). Right: 60 mixed populations, each GPT-5 Nano with one other family
(1, 2 or 4 GPT agents of 8; 170 Sonnet, 70 Gemini, 240 GPT round-10 myths).

The picture is a 2-D view (PC1 + PC2 keep 23% of the variance). Cite the numbers
below, which are computed in the full 768-d space (`significance.csv`).

## What holds at n = 10

**Single-model groups drift apart.** Mean cosine distance between two families'
myths rises from round 1 to round 10 in all 12 single-model cells, by +0.04 to
+0.20; the 95% run-bootstrap interval excludes 0 in every cell. In the Figure 5b
cell (8 agents, Myth → Game): +0.05 to +0.16.

**Mixed groups do not.** In mixed runs the change is −0.08 to +0.03, and the
interval includes 0 in 7 of 10 cells. Two Gemini–GPT dyad cells move closer
(−0.06, −0.08); one population cell drifts slightly (Sonnet–GPT, Game → Myth,
+0.02). In the Figure 5b cell: +0.01 and +0.02, both intervals include 0.

**At round 10 mixed families are closer** than the same pair in single-model
runs in 8 of 10 cells (−0.04 to −0.26). The two exceptions are Sonnet–GPT dyads.
In the Figure 5b cell: −0.04 (Sonnet–GPT) and −0.13 (Gemini–GPT).

**Mixed dyad partners converge.** Partners' myths are closer than different-run
myths at round 10 in 50 of 60 runs (+0.072 ± 0.085, sign test p = 2×10⁻⁷).

**Families stay recognisable.** Family silhouette (768-d, `family_separation.csv`)
in the Figure 5b cell: single-model 0.38 → 0.34, mixed 0.32 → 0.24 from round 1
to 10. Mixed families are closer, not merged. A logistic classifier names the
family of a round-1 myth in 99.5% of held-out runs' myths (both task orders).
