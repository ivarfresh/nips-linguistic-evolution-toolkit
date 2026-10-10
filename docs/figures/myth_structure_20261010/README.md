# Myth structure after Lévi-Strauss (2026-10-10)

Do the agents' myths share a structure, and does it change over ten rounds? Every myth from rounds
1, 2, 5, 6, 9 and 10 was coded once by an LLM judge for the structures in Lévi-Strauss, "The
Structural Study of Myth" (1955):

- **mythemes**: 3-6 "who does what to whom" relations, also written as abstract roles with no names;
- **oppositions**: 1-3 opposed pairs the myth is built on (giving/hoarding, trust/fear, ...), central one first;
- **resolution and mediator**: how the central opposition is settled, and whether a third figure bridges it;
- **arc**: improves / declines / restored / stable order;
- **role inversion**: a character ends at the opposite pole from where it started (stand-in for the canonical formula).

**Columns**: Lévi-Strauss grouped mythemes that say the same thing into "columns". Here the abstract-role mythemes are
embedded (all-mpnet-base-v2) and clustered once over the whole corpus (k = 20 by silhouette, chosen
before looking at rounds). See `columns_exemplars.csv`.

## How it was made

| Step | Script | Notes |
|---|---|---|
| Judge | `analyses/myth_structure_judge.py` | rubric `analyses/rubrics/myth_structure_rubric.txt`; DeepSeek-V4-Flash via OpenRouter, temperature 0; the `Myth: [...]` wrapper is stripped so the judge cannot see the round |
| Analysis | `analyses/myth_structure_analysis.py` | no API calls |

```
python3 analyses/myth_structure_judge.py --model deepseek/deepseek-v4-flash --dataset september_n10
python3 analyses/myth_structure_analysis.py --dataset september_n10
```

Corpora: `september_n10` (Sonnet 4.5 / Gemini 3.7 Flash / GPT-5 Nano, 10 runs per cell; 10,053 of
10,080 myths coded, $1.86) and `frontier` (Opus 5 / Gemini 3.1 Pro / GPT-5.6 Sol, 5 runs per cell; 2,685 of 2,712 coded, $1.40). The two are never
pooled, and the task orders are never pooled.

## Files per corpus folder

- `round1_structure.png`: shares at round 1, single-model runs
- `trend_central_opposition.png`, `trend_resolution_mediator_arc.png`, `trend_columns_top10.png`: shares by round
  (run means ± SE; the grey line marks the prompt change between round 1 and round 2)
- `trend_tests_single_model.csv`: per cell, round 1 and round 10 means (±sd over runs), a Wilcoxon test of round 10 against
  round 1 and of the round 2-10 slope; Holm correction over the eight `headline` measures only
- `transmission_summary.csv`, `transmission_shown_vs_unseen.png`: uptake of the shown myth's oppositions and columns
  against an unseen myth (controls as in `analyses/linguistic_uptake.py`); only features absent from the agent's own
  earlier coded myths count as adoption
- `convergence_columns.png`: column overlap within a run minus between runs of the same cell

## Caveats

- **The mediator and resolution fields are not reliable.** DeepSeek counts the gift-carrying river or bridge as a
  mediator despite the rubric. On 20 myths coded by both judges it agrees with GLM-5.2 60% of the time on mediator
  and 50% on resolution (65% on the central opposition, 70% on arc). Both fields are left out of the headline tests.
- The Holm family was restricted to eight headline measures after the first run. With 10 runs per cell, the smallest
  exact Wilcoxon p is 0.002, so correcting over all ~70 measures could never reach 0.05.
- **Frontier tests cannot reach significance**: with 5 runs per cell the smallest exact Wilcoxon p is 0.0625. Read
  the frontier trend tables as descriptive.
- Round 1 uses a different prompt from rounds 2-10, so a step between rounds 1 and 2 is partly the prompt.
- In dyads the shown myth is the partner's, and partners have been reading each other's myths every round, so
  shown-vs-unseen matching there mixes per-round uptake with the pair evolving together. Adoption of new features
  is the cleaner measure.
