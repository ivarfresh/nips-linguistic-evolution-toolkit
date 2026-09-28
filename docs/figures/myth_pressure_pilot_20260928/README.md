# Myth-pressure pilot: the council shortens myths into maxims, not a code

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

1. **Only the council gets Sonnet near the budget** (criterion 1: FAIL). From
   round 4 on, 4% of tight-arm myths fit the budget before cutting. Counting cut
   myths whose delivered part still ends a sentence, 19% were planned for the cut;
   the bar was 80%. The pooled number hides a split by council:
   - Without a council, no agent adapts. Sonnet writes about 190 words a round
     and loses the rest.
   - With a council, 5 of 10 agents write to the limit: their median myth in
     rounds 6–10 is at most 30 words. The other 5 stay at 160–250 words, which is
     where the 107 (±94) comes from.
   - Sonnet often adds a title ("**The Invitation**"), and the budget counts it.
     The prompt never says whether titles count. With a leading title excluded,
     30 of 70 tight/council myths fit from round 4 (43%), against 0 of 70 without
     a council. This check is descriptive and was added after review.
2. **What gets through is a maxim, not a code** (criterion 2: FAIL). Council myths
   are no stranger than no-council myths (perplexity ratio 0.84; the bar was 2).
   Neither tight arm has a single invented word shared by both partners. The
   first 20 words become a plain rule, for example "Nine cycles revealed truth:
   send three of five faithfully, return proportionally with consistency"
   (tight, no council). The perplexity jump in the tight arms is mostly a
   length effect. The first 20 words of loose-arm myths, scored the same way,
   reach 333–468, the same range as the tight arms' delivered myths (383–453).
   Whether a fragment is cut mid-sentence doesn't explain it. The reviewer scored
   the loose openings with markdown stripped and got about 211, leaving a gap of
   about 2×, so length may not explain all of it. The few shared "invented" words
   in the loose arms are character names ("Kael", "Sira") and ordinary words the
   system word list lacks ("began", "held", "cooperation").
3. **The council stays mostly on writing** (criterion 3: PASS). 10% of council
   messages talk amounts (3% loose, 17% tight). The pass is a pooled rate: one
   tight/council run (replicate 2) talks amounts in 64% of its messages.

Cooperation barely moves. Resources sit between 57 and 61 in every arm, with
overlapping spread. Five runs per arm cannot separate them, and the pilot was
not built to.

One surprise: with a 200-word budget and a council, Sonnet writes longer and
longer, about 690 words by round 10, although its partner sees only the first 200.

## What it means

Pressure alone doesn't make Sonnet 4.5 invent a language here. Without a council
it ignores the budget. With one, half the agents compress, and what they compress
into is an English rule of play. The rule is
interesting in itself, since the myth turns into an explicit norm, but it is not
the emergent code GlossoGen saw. The go rule needed all three criteria, so stage 2
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
- Descriptive checks added after the PR #5 review (title-excluded fit, per-agent
  split, length-matched perplexity) are in the second section of `criteria.md`.
- The invented-word check strips possessives and common suffixes and splits
  hyphenated words before the dictionary lookup (fixed before this write-up; the
  first pass counted inflected English like "asked" as invented).
- `criteria.md`, `runs.csv`, `myths.csv` (delivered text, delivery record,
  perplexity), `council_messages.csv` (with the amount-talk flag).
- Smoke run (not pooled):
  `data/json/noise_experiments/myth_pressure_pilot_20260928_smoke/smoke_receipt.json`.
