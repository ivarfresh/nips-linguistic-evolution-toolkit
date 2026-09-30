# Do the myths predict cooperation, and do the moral labels capture it?

Two free checks (no API calls) before re-judging the myths with a larger judge
panel. They ask whether the weak moral-carryover result comes from bad judges,
from the wrong three categories, or from myths that carry little about play in
the first place. Corpus: the 8,519 September informed negative-only myths
(homogeneous and mixed, 2- and 8-agent) used by the moral-carryover analysis.
Every number is given per task order; the two are never pooled.

## Headline

The judges are not the main problem, and the myths do carry information about
play, but only one kind and only at the start.

- **The two judges disagree on one boundary, not at random.** 97% of their
  disagreements are adjacent ones, and 80% (game→myth) to 83% (myth→game) are
  the same one: GLM says "be fair" where DeepSeek says "be generous". The myths
  in question usually endorse both at once ("send what you can, return what is
  fair"). The three labels are forced single choices between categories that
  overlap.
- **Before any play, the amount a myth names predicts the opening send. The
  moral label does not.** Myth→game, round 1: a Sonnet agent whose myth says
  "three of five" sends 3.00 (12 of 12 agents); one whose myth says all five sends
  4.75 (±0.55). PR #4's extracted send amount adds 0.16 to the R² of the opening
  send (0.51 within Sonnet, 0.13 within GPT). The moral label adds nothing, and for
  Sonnet it points the wrong way ("be generous" 3.86 (±0.98), "be fair" 4.08 (±0.87)).
- **Once play has started, no representation of the myth adds anything to past
  play, in either task order.** This holds for the whole-text embedding, word
  counts, both judges' labels and the extracted rules. It holds for the author's own
  next decision and for the reader of a shown myth, and within agents as well as
  across them.
- **The whole-text embedding is not a ceiling.** It misses the one thing that
  does predict, the named amount, because it encodes style and topic rather than
  numbers. "The embedding can't predict" therefore does not mean "the text can't".

What this means for the judge panel: better judges on the same three categories
would not help. What could help is concrete, graded features (amounts, return
shares, conditions), scored where the myth can lead play. That is round 1 in
myth→game. After that, myths follow the game.

## 1. Judge disagreement (GLM-5.2 vs DeepSeek V4 Flash)

`analyses/moral_judge_disagreement.py` → `judge_agreement_by_task_order.csv`,
`judge_confusion_by_task_order.csv`.

| Task order | Myths | Agreement | κ | Ordinal κ | Disagreements that are fair vs generous |
|---|---:|---:|---:|---:|---:|
| game→myth | 4,259 | 75% | 0.54 | 0.61 | 80% |
| myth→game | 4,260 | 74% | 0.53 | 0.59 | 83% |

By author family, κ is highest on Sonnet (0.65–0.69) and lowest on Gemini
(0.29–0.32). On Gemini, every disagreement is fair vs generous. DeepSeek calls 76–80%
of Gemini myths generous and GLM calls 50–55% generous. That is a difference in where
each judge draws the line on one axis, not noise. Neither judge says "be cautious" for
Gemini.

## 2. Does the myth text predict the next decision?

`analyses/myth_text_predictiveness.py` → `results.csv` (all task orders, tests,
author families, within/across agents).

**Timing.** In myth→game, a myth written in round t is followed by game t. In
game→myth, it is followed by game t+1.

**Tests:**
- **Opening:** myth→game, round 1 only, before any play.
- **Author:** the author's next decision after its myth, from round 2 on in myth→game.
- **Reader:** the author's next decision after being shown its previous partner's myth.

**Outcomes.** Amount sent (/5) and return proportion are analysed separately.

**Base model:** composition cell and family, plus round, and the author's own last
send and return and what its last partner did (the opening test has no play
history). "Within agent" removes each agent's mean, so only round-to-round changes
count.

**Scoring.** Each text feature set is added to the base with a two-stage ridge, and
scored by out-of-sample R² with folds grouped by run. We report the gain over the base
as mean (±std) over 10 fold shuffles. A result counts as clear when the gain beats all
20 shuffles of the text among myths of the same family, round and composition
(p ≤ 0.05, the smallest p 20 shuffles allow).

Across authors of all families (with family in the base), across agents:

| Task order | Test | Decision | n | Base R² | Embedding | Word counts | GLM label | DeepSeek label | Extracted rules (PR #4) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| myth→game | opening | send | 213 | 0.26 | −0.003 (±0.007) | −0.006 (±0.010) | −0.001 (±0.002) | −0.000 (±0.003) | **+0.163 (±0.016)** |
| myth→game | opening | return | 209 | 0.25 | −0.002 (±0.003) | +0.010 (±0.012) | −0.001 (±0.001) | +0.005 (±0.004) | −0.008 (±0.011) |
| myth→game | author | send | 1,917 | 0.37 | −0.000 (±0.003) | −0.001 (±0.004) | +0.000 (±0.000) | +0.002 (±0.001) | +0.002 (±0.001) |
| myth→game | author | return | 1,765 | 0.38 | −0.003 (±0.002) | +0.007 (±0.002) | +0.004 (±0.001) | −0.002 (±0.002) | +0.015 (±0.002) |
| myth→game | reader | send | 1,917 | 0.37 | +0.000 (±0.002) | −0.005 (±0.004) | −0.000 (±0.000) | −0.001 (±0.001) | −0.000 (±0.000) |
| myth→game | reader | return | 1,765 | 0.38 | −0.002 (±0.001) | −0.003 (±0.003) | −0.001 (±0.001) | −0.000 (±0.000) | +0.003 (±0.001) |
| game→myth | author | send | 1,916 | 0.43 | +0.000 (±0.002) | +0.007 (±0.003) | +0.000 (±0.000) | −0.000 (±0.001) | −0.000 (±0.001) |
| game→myth | author | return | 1,679 | 0.24 | −0.001 (±0.001) | −0.003 (±0.001) | −0.000 (±0.001) | +0.001 (±0.002) | −0.000 (±0.000) |
| game→myth | reader | send | 1,702 | 0.42 | −0.000 (±0.001) | −0.002 (±0.002) | −0.001 (±0.001) | −0.001 (±0.001) | −0.000 (±0.000) |
| game→myth | reader | return | 1,503 | 0.22 | −0.003 (±0.002) | −0.003 (±0.002) | −0.001 (±0.001) | −0.002 (±0.001) | −0.000 (±0.000) |

Within-agent versions of every row, and per-family rows, are in `results.csv`.
Only 5 of 330 tests are clear:
- the opening send with extracted rules: all families +0.16, Sonnet +0.51, GPT +0.13;
- two tiny gains in myth→game authors: extracted rules on Sonnet sends, +0.014; the
  GLM label on Gemini returns, +0.014.

With 330 tests, the two tiny ones are what chance produces. Gemini sends are always 5,
so its send tests are skipped.

Round-1 myth→game sends by what the myth says (mean (±std), n):

| Family | Myth names "all five" | Myth names 3 | Names no amount | Label "be generous" | Label "be fair" |
|---|---|---|---|---|---|
| Sonnet | 4.75 (±0.55), 20 | 3.00 (±0.00), 12 | 4.16 (±0.72), 32 | 3.86 (±0.98), 36 | 4.08 (±0.87), 36 |
| GPT | 4.57 (±0.85), 14 | — | 3.01 (±1.16), 69 | 3.18 (±0.98), 11 | 3.21 (±1.34), 76 |

Labels are GLM-5.2's. DeepSeek's labels show the same flat pattern.

## Caveats

- **The extracted rules come from an unmerged PR.** They are read from PR #4's
  GLM-5.2 extraction (`myth_rules_september_z-ai__glm-5.2.csv`, produced by
  `analyses/myth_rule_judge.py`, not yet merged). The file is gitignored.
- **The opening test is small.** It has 213 senders (72 Sonnet, 91 GPT, 50 Gemini),
  and Gemini always sends 5.
- **"Adds nothing beyond past play" is not "has no effect".** A myth could shape play
  and past play together. These tests only say the text carries no extra forecast once
  the last game is known.
- **Reader test scope.** It controls for the reader's past play but not the reader's
  own myth. It measures the total link between what was read and what the reader does
  next.
- **No human has checked either the labels or the extracted amounts.**
