# Myth map, frontier models: Opus 5, Gemini 3.1 Pro, GPT-5.6 Sol, 2026-10-03

The [September myth map](../myth_convergence_map_20261002/README.md) repeated on
the main frontier set, plus one joint figure that puts both sets' round-1 myths
on one map. Same scripts, same methods, separate statistics: the two sets come
from different models, request profiles and dates, and are never pooled.

**Answer.** As in September, each frontier model opens with its own kind of
myth, and families that never meet drift apart. Mixing helps less than in
September: in half of the mixed cells the families still drift apart, only
more slowly. Gemini 3.1 Pro opens almost exactly like Gemini 3.7 Flash; Opus
and Sol each open with a myth of their own, not their smaller sibling's.

## Data

The 106 myth runs of the main frontier set (the 2026-09-18 single-model rerun
and the 2026-09-28 mixed runs; game-only runs have no myths), 4,520 myths.

| Setting | Runs per task order |
|---|---|
| 2 agents, one model (Opus, Gemini Pro, Sol) | 5 per model |
| 2 agents, mixed (Opus+Sol, Opus+Gemini Pro, Gemini Pro+Sol) | 6 per pairing |
| 8 agents, one model | 5 per model |
| 8 agents, mixed: 2 Gemini Pro + 3 Opus + 3 Sol | 5 |

The mixed populations differ from September: all three models meet in every
run (September: one model as a minority of 1, 2 or 4 among another).

```
python3 analyses/myth_convergence_map.py --dataset frontier --out docs/figures/myth_convergence_map_frontier_20261003
python3 analyses/myth_map_joint_round1.py
python3 analyses/myth_map_significance.py --dataset frontier   # tests, time views, provenance (run last)
```

Inputs are the gitignored tables in `data/analysis/linguistic_frontier_20260930/`.
No API calls.

## Figures

Same layout and reading as the September folder; family colours follow the lab
(Opus = Sonnet's purple, Gemini Pro = Gemini's green, Sol = GPT's orange).

- `round1_myth_game.png`, `round1_game_myth.png`: where round-1 myths start.
- `trajectories_8agent.png`, `trajectories_2agent.png`: how they move (PCA map,
  14% + 7% of the variance).
- `convergence_over_rounds.png`: 768-d distance between families' myths per
  round, myth pairs from different runs only.
- `family_time_map_8agent.png`, `family_time_map_2agent.png`: family axis ×
  time axis.
- `joint_round1.png`: both sets' round-1 myths on one map fitted to all of them.

## What we see

**1. Families start apart.** A classifier tested on unseen runs names the
family of a round-1 myth 98.2% (Myth → Game) and 98.7% (Game → Myth) of the
time; largest-family guess 34.1%. Run-level shuffle: p ≤ 1×10⁻⁴.

**2. Morals follow family** (`round1_morals.csv`; χ² = 93.1 and 85.5,
p < 10⁻¹⁶). Opus and Gemini Pro open "be generous" (62/77, 57/72), Sol "be
fair" (66/77). One round of play first lowers Opus's generous openings
(62 → 49 of 77, Fisher p = 0.03), as it did Sonnet's in September. Sol
(10 → 3, p = 0.08) and Gemini Pro (57 → 50, p = 0.25) do not change clearly.

**3. Alone, families drift apart.** Every single-model pair moves apart from
round 1 to 10 (+0.07 to +0.18); the paired run-bootstrap interval excludes 0
in 12 of 12 cells.

**4. Mixed, they drift apart less, but often still drift.**

| Mixed cell | Change round 1 → 10 | Cells |
|---|---|---|
| Further apart (interval above 0) | +0.04 to +0.17 | 6 of 12 |
| No clear change | −0.06 to +0.03 | 4 of 12 |
| Closer (interval below 0) | −0.04 and −0.05 | 2 of 12, both Opus–Gemini Pro |

By round 10 mixed families are closer than single-model ones in all 12 cells
(0.02 to 0.17), with the interval of the difference excluding 0 in 8 of 12.
Opus–Sol pairs drift apart almost as much as when they never meet (+0.17 vs
+0.18 in 2-agent Myth → Game).

**5. Partners converge.** Mixed-dyad partners are +0.072 (±0.090) closer than
other runs' myths at round 10 (29 of 36 runs, sign test p = 3×10⁻⁴); the gap
grows from round 1 (Wilcoxon p = 4×10⁻⁵).

**6. Siblings** (`joint_round1.png`, `joint_round1_centroid_distance.csv`,
`joint_round1_confusion.csv`; 768-d cosine distance between round-1 centroids,
Myth → Game / Game → Myth):

| Frontier model | Distance to its sibling | Nearest family | Classifier calls it the sibling |
|---|---|---|---|
| Gemini 3.1 Pro | 0.035 / 0.026 (Gemini Flash) | Gemini Flash | 16 / 17 of 72 |
| GPT-5.6 Sol | 0.103 / 0.072 (GPT-5 Nano) | Gemini Flash / GPT-5 Nano | 9 / 7 of 77 |
| Opus 5 | 0.143 / 0.121 (Sonnet 4.5) | Sol / Sonnet | 1 / 1 of 77 |

Gemini Pro's openings are nearly Gemini Flash's. Sol's sit between GPT-5 Nano
and Gemini Flash. Opus's sit beside Sonnet's on the map but are told apart
almost perfectly.

## Caveats

- 5 runs per family (single-model) and 6 per pairing (dyads); the mixed
  populations have 5 runs per task order. Bootstrap intervals from so few runs
  run narrow: read them as a guide.
- The joint map is fitted to round-1 myths only, so it is a different map from
  the per-set maps.
- Moral labels are from one judge model (GLM-5.2).
- Language, not behaviour.
