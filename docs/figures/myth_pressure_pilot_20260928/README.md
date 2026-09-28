# Myth-pressure pilot: Sonnet shortens its myths into maxims, not a code

Run 2026-09-28/29. 20 Sonnet 4.5 myth-first dyads, 5 per arm, all 20 finals
launcher-audited (`data/json/noise_experiments/myth_pressure_pilot_20260928/completion_receipt.json`,
$27.07 at standard rates). Design and pre-registered criteria:
[`docs/research/myth_pressure_pilot_2026-09-28.md`](../../research/myth_pressure_pilot_2026-09-28.md).

## The question

GlossoGen found that LLM agents invent a new language when talking is costly and
they can agree conventions between rounds. We asked whether the same pressure
turns our myths into a shared code. We capped how many words of a myth the partner
receives (loose: 200; tight: down to 20 from round 6), and in half the runs the
partners held a council between rounds to discuss how they write.

## What happened

No code appeared. Two of the three pre-registered criteria fail; council hygiene
passes.

| Arm | Resources after round 10 | Send fraction | Words written, rounds 6–10 | Perplexity, rounds 6–10 | Shared invented tokens | Council amount talk |
|---|---|---|---|---|---|---|
| loose / no council | 57.30 (±5.45) | 0.65 (±0.11) | 281.0 (±37.4) | 46.79 (±16.82) | 1.00 (±0.71) | — |
| loose / council | 57.76 (±9.71) | 0.66 (±0.19) | 581.1 (±124.4) | 57.26 (±11.80) | 1.20 (±1.30) | 3% |
| tight / no council | 59.76 (±4.36) | 0.70 (±0.09) | 187.2 (±7.9) | 453.44 (±114.00) | 0.00 (±0.00) | — |
| tight / council | 61.14 (±8.25) | 0.72 (±0.17) | 106.6 (±93.8) | 382.90 (±72.52) | 0.00 (±0.00) | 17% |

Means (±std) over 5 replicates, computed by `analyses/myth_pressure_pilot.py`
(`criteria.md`, `runs.csv`). Words written: each run's median over rounds 6–10, from `myths.csv`.

1. **Sonnet does not write to the budget** (criterion 1: FAIL). From round 4 on, 4%
   of tight-arm myths fit the budget before cutting. Counting cut myths whose
   delivered part still ends a sentence, 19% were planned for the cut; the bar was
   80%. Without a council, Sonnet keeps writing about 190 words a round and loses
   the rest. With a council it drops to about 107 words (±94 across runs), but still overruns.
   Only 5 of 20 tight-arm agents settled near the limit.
2. **What gets through is a maxim, not a code** (criterion 2: FAIL). Council myths
   are no stranger than no-council myths (perplexity ratio 0.84; the bar was 2).
   Neither tight arm has a single invented word shared by both partners. The
   first 20 words become a plain rule, for example "Nine cycles revealed truth:
   send three of five faithfully, return proportionally with consistency"
   (tight, no council). With no new vocabulary, the perplexity jump in the tight
   arms most likely comes from scoring short fragments cut mid-sentence. The few
   shared "invented" words in the loose arms are character names ("Kael", "Sira")
   and irregular past tenses the word list lacks ("began", "held").
3. **The council stays mostly on writing** (criterion 3: PASS). 10% of council
   messages talk amounts (3% loose, 17% tight).

Cooperation barely moves. Resources sit between 57 and 61 in every arm, with
overlapping spread. Five runs per arm cannot separate them, and the pilot was
not built to.

One surprise: with a 200-word budget and a council, Sonnet writes longer and
longer, about 690 words by round 10, although its partner sees only the first 200.

## What it means

Pressure alone doesn't make Sonnet 4.5 invent a language here. It keeps writing
prose, and when forced it compresses into an English rule of play. The rule is
interesting in itself, since the myth turns into an explicit norm, but it is not
the emergent code GlossoGen saw. By the pre-registered stop rule, stage 2
(populations, transplant, newcomer) does not go ahead on this design.

Two differences from GlossoGen could explain the gap, and are candidates for a
redesign, not conclusions:

- In GlossoGen, going over budget failed the task. Here nothing is lost but the
  cut words, and the game pays the same either way.
- GlossoGen's agents had to transmit arbitrary, changing information (symptoms and
  cures). A myth carries one stable norm, which a short English sentence already
  says well.

## Files

- `myth_pressure_pilot.png`: words written per round, delivered-text perplexity,
  and late perplexity against resources per run.
- The invented-word check strips possessives and common suffixes and splits
  hyphenated words before the dictionary lookup (fixed before this write-up; the
  first pass counted inflected English like "asked" as invented).
- `criteria.md`, `runs.csv`, `myths.csv` (delivered text, delivery record,
  perplexity), `council_messages.csv` (with the amount-talk flag).
- Smoke run (not pooled):
  `data/json/noise_experiments/myth_pressure_pilot_20260928_smoke/smoke_receipt.json`.
