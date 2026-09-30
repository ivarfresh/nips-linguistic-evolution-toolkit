# Do the myths predict cooperation, and do the moral labels capture it?

Two free checks (no API calls) before re-judging the myths with a larger judge
panel. They ask whether the weak moral-carryover result comes from bad judges,
from the wrong three categories, or from myths that carry little about play in
the first place. Corpus: the 8,519 September informed negative-only myths
(homogeneous and mixed, 2- and 8-agent) used by the moral-carryover analysis.
Every number is given per task order; the two are never pooled.

## Headline

The two judges' disagreement is structured, not random, and it sits on a boundary
where the categories overlap. The myths do carry information about play, but
only one kind and only at the start.

- **The two judges disagree mostly on one boundary.** 97% of their
  disagreements are adjacent ones, and 80% (game→myth) to 83% (myth→game) are
  fair vs generous, mostly GLM "be fair" where DeepSeek says "be generous" (71% and
  76% of all disagreements). The myths
  in question usually endorse both at once ("send what you can, return what is
  fair"). The three labels are forced single choices between categories that
  overlap.
- **Before any play, the amount a myth names predicts the opening send. The
  moral label does not.** Myth→game, round 1: a Sonnet agent whose myth says
  "three of five" sends 3.00 (12 of 12 agents, from 10 runs); one whose myth says
  all five sends 4.75 (±0.55) (20 agents, 16 runs). The named amount alone adds
  0.17 to the R² of the opening send [95% run-resampled interval 0.10 to 0.24]:
  0.37 [0.02, 0.59] within Sonnet, 0.07 [−0.03, 0.16] within GPT (not clear). The
  moral label is flat: Sonnet "be generous" 3.86 (±0.98) vs "be fair" 4.08 (±0.87).
- **Once play has started, the myth adds almost nothing to past play, and nothing
  to round-to-round change.**
  - Within agents (does a change in an agent's myth forecast a change in its next
    move?), no feature helps, in either task order.
  - Across agents, only myth→game shows small gains, from PR #4's rules on the
    author's returns: +0.015 R² for all families, +0.051 for Sonnet. Sonnet sends gain
    +0.014. The likely reading: a stable per-agent return habit also shows up in its
    myths.
  - The reader of a shown myth gains nothing anywhere, and game→myth gains nothing.
- **The whole-text representations miss the opening signal for lack of data, not
  lack of information.** Across all 4,594 myths that name an amount, the embedding
  recovers it with R² 0.46 and word counts with 0.56. With only 213 opening
  senders, though, 50 to 100 dimensions are too many. A single targeted feature
  (the amount) gets through where they cannot.

What this means for the judge panel: more judges on the same three categories
would mostly re-measure where each draws the fair/generous line. Concrete,
graded features (amounts, return shares, conditions) are what predicts, and the
test is sharpest where the myth can lead play: round 1 in myth→game. After that,
the myth adds no forecast beyond the last game. That fits the earlier finding
that morals follow the game just played (moral-carryover analysis), but this
analysis does not test that direction itself.

## 1. Judge disagreement (GLM-5.2 vs DeepSeek V4 Flash)

`analyses/moral_judge_disagreement.py` → `judge_agreement_by_task_order.csv`,
`judge_confusion_by_task_order.csv`.

| Task order | Myths | Agreement | κ | Ordinal κ | Disagreements that are fair vs generous |
|---|---:|---:|---:|---:|---:|
| game→myth | 4,259 | 75% | 0.54 | 0.61 | 80% |
| myth→game | 4,260 | 74% | 0.53 | 0.59 | 83% |

By author family, κ is highest on Sonnet (0.65–0.69) and lowest on Gemini
(0.29–0.32). On Gemini, all but one or two disagreements are fair vs generous. DeepSeek calls 76–80%
of Gemini myths generous and GLM calls 50–55% generous. That is a difference in where
each judge draws the line on one axis, not noise. GLM never says "be cautious" for
Gemini, and DeepSeek says it once.

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
as mean (±std) over 10 fold shuffles. A result counts as clear when its 95% interval
from resampling whole runs lies above zero. That interval holds out-of-fold
predictions fixed, so it is somewhat narrow. `results.csv` also keeps a permutation
p, but it separates signal from noise poorly. The first version of this analysis
used it as the criterion and scored some shuffles on mismatched folds. An independent
review caught this; the fix changed which tests count as clear, not the gains.

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
7 of 330 tests are clear (interval above zero), all in myth→game and all across agents:

| Test | Authors | Feature | Gain in R² | 95% interval |
|---|---|---|---:|---|
| opening send | all | PR #4 rules | +0.163 (±0.016) | 0.079 to 0.242 |
| opening send | Sonnet | PR #4 rules | +0.508 (±0.040) | 0.238 to 0.705 |
| opening send | GPT | PR #4 rules | +0.131 (±0.030) | 0.017 to 0.239 |
| author return | Sonnet | PR #4 rules | +0.051 (±0.004) | 0.024 to 0.077 |
| author return | all | PR #4 rules | +0.015 (±0.002) | 0.004 to 0.028 |
| author send | Sonnet | PR #4 rules | +0.014 (±0.003) | 0.004 to 0.025 |
| author send | Sonnet | DeepSeek label | +0.005 (±0.001) | 0.001 to 0.010 |

GPT's opening result is borderline. With the named amount alone its interval
includes zero (`opening_checks.md`). No within-agent test, no reader test and no
game→myth test is clear. The same Sonnet returns effect within agents is −0.008.
Gemini sends are always 5, so its send tests are skipped.

Round-1 myth→game sends by what the myth says (mean (±std), n):

| Family | Myth names "all five" | Myth names 3 | Names no amount | Label "be generous" | Label "be fair" |
|---|---|---|---|---|---|
| Sonnet | 4.75 (±0.55), 20 | 3.00 (±0.00), 12 | 4.16 (±0.72), 32 | 3.86 (±0.98), 36 | 4.08 (±0.87), 36 |
| GPT | 4.57 (±0.85), 14 | — | 3.01 (±1.16), 69 | 3.18 (±0.98), 11 | 3.21 (±1.34), 76 |

Labels are GLM-5.2's. DeepSeek's labels show the same flat pattern. The same
tables split by setting (2/8 agents, mixed/homogeneous), with run counts, are in
`opening_checks.md`. Sonnet shows the pattern in both 8-agent settings: "three"
gives 3.00 in each, and "all five" gives 5.00 (homogeneous) and 4.70 (mixed). GPT
myths naming all five send 4.67 (±0.78) in 8-agent mixed runs. GPT myths naming one
are inconsistent: 0.50 in 8-agent homogeneous (n=2) and 3.00 in 8-agent mixed (n=4).
This is why GPT's interval includes zero.

The "all families" rows pool the four settings, with each composition cell in
the base model. Per-setting cells are too small for the regression (as few as 5
round-1 senders).

`analyses/myth_text_opening_checks.py` → `opening_checks.md`: the probe, the
named-amount-alone gains with run-resampled intervals, and the per-setting tables.

## Caveats

- **The extracted rules come from an unmerged PR.** They are read from PR #4's
  GLM-5.2 extraction (`myth_rules_september_z-ai__glm-5.2.csv`, produced by
  `analyses/myth_rule_judge.py`, not yet merged). The file is gitignored.
- **The opening test is small.** It has 213 senders (72 Sonnet, 91 GPT, 50 Gemini),
  and Gemini always sends 5.
- **"Adds nothing beyond past play" is not "has no effect".** A myth could shape play
  and past play together. These tests only say the text carries no extra forecast once
  the last game is known.
- **Reader test scope.** This tests *read myth → next decision*. It does not retest the
  earlier *read myth → written moral* carryover (+5 points), so it does not contradict
  it. It controls for the reader's past play, not the reader's own myth.
- **The run-resampled intervals are somewhat narrow.** They reuse fixed out-of-fold
  predictions instead of refitting per draw.
- **No human has checked either the labels or the extracted amounts.**
