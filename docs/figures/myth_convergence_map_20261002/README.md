# Myth map: where each family's myths start and how they move, 2026-10-02

To-do item 8 on the paper list ("myth convergence plot"): do round-1 myths
cluster by model family or by moral, and how do they change over the game?
The layout follows the "behaviour space" sketch in Sakana's DRQ write-up
(pub.sakana.ai/drq), but every point here is data.

**Answer.** Each family opens with its own kind of myth. Run alone, a family
stays in its own region. Paired with another family, the two partners' myths
move towards each other, but the families stay apart.

## Data and method

The 156 myth runs of the September informed negative-only design, the same
corpus as the [linguistic analysis](../linguistic_analysis_20260923/README.md)
(8,519 myths after dropping one empty GPT response). Every myth is embedded
with all-mpnet-base-v2 and projected onto the first two principal components
of the whole corpus. They keep 16% and 7% of the variance, so the map is a
coarse view. All numbers below are computed in the full 768-d space.

PCA is linear, so a family's average position on the map is the map
position of its average embedding. The drawn paths are real averages.

```
python3 analyses/myth_convergence_map.py   # free; reuses the cached embeddings
```

Inputs are the gitignored tables in `data/analysis/linguistic_20260923/`
(see the linguistic README for how to regenerate them). No API calls.

## How to read the figures

- One dot is one myth. Myths that read alike sit close together. The axes
  have no names; they are the two directions in which the myths differ most.
  All figures share one map, so positions compare across figures.
- `round1_myth_game.png`: round-1 myths of Myth → Game runs, written before
  any game. Left: colour = family, rings = where each family is densest.
  Right: colour = the moral the GLM-5.2 judge read in the myth.
- `round1_game_myth.png`: the same for Game → Myth runs, where the round-1
  myth is written after the first round of play.
- `trajectories_8agent.png`, `trajectories_2agent.png`: one panel per
  single-model/mixed × task order (never pooled).
  - Coloured line: family average from round 1 (light) to round 10 (dark).
  - Big circles: rounds 1, 5 and 10.
  - Black dots: each run's family average at those rounds.
  - Blue background: where that panel's round-10 myths end up.

## What we see

**1. Families start in different places.** Round-1 myths form three distinct
but overlapping clusters. Family silhouette at round 1 (`family_separation.csv`;
0 = no grouping, 1 = perfect) is 0.30–0.41 across the eight size × mixing ×
task-order cells.

**2. Morals line up with family.** Round-1 moral labels
(`round1_morals.csv`):

| Family | Myth → Game (before any game) | Game → Myth (after round 1) |
|---|---|---|
| Sonnet | 74 fair / 73 generous / 0 cautious | 93 fair / 43 generous / 11 cautious |
| GPT | 139 fair / 27 generous / 16 cautious | 142 fair / 23 generous / 17 cautious |
| Gemini | 24 fair / 73 generous / 0 cautious | 29 fair / 68 generous / 0 cautious |

The "generous" region of the map is largely Gemini's region, so this map
cannot separate clustering by moral from clustering by family. One round of
play moves Sonnet's moral (generous 73 → 43, cautious 0 → 11) but not its
place on the map. GPT and Gemini barely change. These are judge labels, not
yet checked against the human coding pass, and the counts have no test.

**3. Single-model runs stay apart.** On the map, Gemini moves furthest, away
from the others; Sonnet moves a little towards the centre and GPT barely
moves. In 768-d the map understates Sonnet: in 8-agent Myth → Game its
round-1 to round-10 shift is larger than Gemini's (centroid cosine 0.77
against 0.82; GPT 0.94). The round-10 background
has one basin per family. Family silhouette at round 10
stays at 0.33–0.41.

**4. Mixed partners pull together.** In mixed runs, family silhouette falls
from 0.30–0.38 at round 1 to 0.10–0.11 in dyads and 0.23–0.26 in
populations by round 10. The dyads show why (`partner_convergence.csv`):
cross-family similarity between the two partners, against different-family
myths from other runs of the same pairing, task order and round (n = 6 runs
per cell):

| Pairing | Order | Round 1: partner / other run | Round 10: partner / other run |
|---|---|---|---|
| Sonnet+Gemini | Game → Myth | 0.67 / 0.67 | 0.76 / 0.65 |
| Sonnet+Gemini | Myth → Game | 0.65 / 0.64 | 0.74 / 0.64 |
| Sonnet+GPT | Game → Myth | 0.65 / 0.65 | 0.77 / 0.69 |
| Sonnet+GPT | Myth → Game | 0.64 / 0.67 | 0.73 / 0.68 |
| Gemini+GPT | Game → Myth | 0.61 / 0.62 | 0.73 / 0.67 |
| Gemini+GPT | Myth → Game | 0.62 / 0.60 | 0.75 / 0.71 |

At round 1, partners are no closer than strangers (17 of 36 runs above the
baseline). By round 10 they are 0.04–0.11 closer in all six cells, most for
Sonnet+Gemini, and 32 of 36 runs are above the baseline (`n_runs_closer`).
This is expected from the design: from round 2 every agent's myth prompt
contains its partner's last myth (`{other_agent_myth}` in
`config/experiments.yaml`). The map shows how far that pull goes. The
baseline also rises in most cells, so later myths are more alike in general;
that is why the comparison holds round fixed. In 8-agent populations each
agent reads a different partner each round, and the map shows a weaker pull. This table is not computed for them.

## Caveats

- The 2-D map flatters the clustering (round-1 silhouette 0.45–0.64 on the
  map against 0.30–0.41 in 768-d; `silhouette_map_2d`), so cite the 768-d
  numbers.
- Family lines in mixed panels average over pairings: Sonnet's dyad line
  pools its runs with GPT and with Gemini. The table gives each pairing.
- Moving closer in language is not cooperating more. The linguistic analysis
  found that alignment does not predict cooperation within runs.
- 15–30 runs per panel (6 per dyad pairing); small wiggles are noise.
