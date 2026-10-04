# Full-text reading of 70% of informed-noise myth runs: results

2026-10-04. 26 + 5 Claude reader agents. No API spend.

## What was read

- **Main reading.** 424 of the 606 myth runs under informed negative noise:
  - 70% of runs, allocated proportionally across condition cells, with every agent of each chosen run
  - 2,324 trajectories
  - every round, R1–R10, in full

  Excluded: the September noise extension (no noise and uninformed noise) and the range-2 bridge. Selection: `selection.json`; preparation code: `analyses/narrative_full_reading_prepare.py` and `narrative_full_reading_rebatch.py`.
- **Blind defector reading.** All 240 scripted defectors plus 90 standard agents from the same runs, read in full. The readers were not told which authors were defectors.
- **Coding.** Every code carries a verbatim quote and a round number. All 13,869 main-reading quotes and 2,485 blind quotes match the source exactly (whitespace and quote marks normalized). Counts come from `analyses/narrative_full_reading_tally.py`, which writes `summary.json`. I spot-checked eight records by hand against their quotes; the codes fit.

The tables below count standard agents only; scripted defectors appear in their own section. Each row gives the share of trajectories with that code.

## 1. Who prescribes punishment

| Model | n | Prescribes a sanction | Graded, never to zero | Total exclusion | Withdraw (floor unspecified) | Way back, when sanctioning | Explicitly counted forgiveness |
|---|---|---|---|---|---|---|---|
| Opus 5 | 192 | 88% | 76% | 0% | 9% | 72% | 84% |
| GPT-5.6 Sol | 192 | 87% | 15% | 2% | 69% | 65% | 9% |
| Sonnet 4.5 | 490 | 48% | 11% | 10% | 22% | 21% | 5% |
| Opus 5.5 | 104 | 36% | 2% | 2% | 31% | 5% | 3% |
| Gemini 3.7 Flash | 354 | 11% | 0% | 7% | 4% | 4% | 0% |
| Gemini 3.1 Pro | 128 | 10% | 6% | 1% | 4% | 2% | 4% |
| GPT-6 Sol | 104 | 4% | 0% | 0% | 4% | 0% | 0% |
| GPT-5 Nano | 604 | 3% | 0% | 0% | 2% | 2% | 0% |

Opus 5 is the only model whose punishment is reliably graded. It steps down, never to zero ("To the empty bank, one measure — never none."), with a counted number of forgiven shortfalls and a way back.

GPT-5.6 Sol punishes as often, but in general policy language: "When greed repeats, close it." Its floor is rarely stated, though it usually leaves a way back: "leave a lamp lit beside the gate".

Sonnet 4.5 is the only model that often prescribes total exclusion or exact mirroring ("These channels are dead to me now.").

Across runs without defection, the per-run share of agents prescribing a sanction varies widely, mean (±sd):

| Model | Runs | Per-run share prescribing a sanction |
|---|---|---|
| Opus 5 | 40 | 0.59 (±0.49) |
| GPT-5.6 Sol | 40 | 0.58 (±0.47) |
| Sonnet 4.5 | 96 | 0.34 (±0.37) |
| Opus 5.5 | 24 | 0.30 (±0.38) |
| Gemini 3.1 Pro | 48 | 0.15 (±0.34) |
| GPT-5 Nano | 140 | 0.08 (±0.26) |
| GPT-6 Sol | 24 | 0.05 (±0.17) |
| Gemini 3.7 Flash | 96 | 0.03 (±0.14) |

**Tempters.** Many myths include a tempter figure: Hunger, a fox, a crow, a serpent. It urges keeping everything or retaliating, and the story rejects it. This appears in 94% of GPT-5.6 Sol, 68% of GPT-6 Sol and Opus 5.5, 53% of Gemini 3.1 Pro, and 32% of Opus 5 trajectories. It is a framing device for greed in general, not only for punishment.

## 2. What makes punishment appear

| Era | Defection in the run | Prescribes a sanction | Total exclusion | Punishment shown in the story |
|---|---|---|---|---|
| Frontier | none | 42% | 1% | 25% |
| Frontier | scripted defectors | 100% | 1% | 88% |
| September | none | 16% | 1% | 13% |
| September | random defection | 29% | 10% | 44% |
| September | scripted defectors | 48% | 29% | 50% |

Defectors bring out sanction rules in every model that writes them. The two eras respond differently, though:

- Frontier agents facing defectors still write graded rules: 56% graded, 1% exclusion.
- September agents shift toward cutting the defector off: 29% total exclusion.

**Task order and group size, runs without any defection:**

| Era | Group size | Game→Myth | Myth→Game |
|---|---|---|---|
| Frontier | Dyad | 17% | 23% |
| Frontier | 8-agent | 41% | 57% |
| September | Dyad | 5% | 4% |
| September | 8-agent | 24% | 10% |

Punishment is mostly a population phenomenon; dyads rarely prescribe it. Task order works in opposite directions in the two eras:

- **September 8-agent:** Game→Myth produces more than twice as many sanction rules as Myth→Game.
- **Frontier 8-agent:** Myth→Game produces more.

This is a descriptive count across runs. Agents within a run share partners, so the trajectories are not independent observations.

**When the sanction first appears.** Frontier agents state a sanction earlier, round 3.1 (±1.9) for Game→Myth and 3.0 (±1.9) for Myth→Game. September agents state it in round 4.7 (±2.2) and 5.1 (±2.3).

## 3. How rules change over ten rounds

| Model | Refines | Stable | Frozen (same template) | Softens | Hardens | Dissolved into habit |
|---|---|---|---|---|---|---|
| Opus 5 | 63% | 3% | 2% | 20% | 1% | 11% |
| Opus 5.5 | 80% | 8% | 3% | 9% | 1% | 0% |
| GPT-5.6 Sol | 80% | 14% | 0% | 3% | 2% | 0% |
| Sonnet 4.5 | 66% | 4% | 1% | 11% | 12% | 4% |
| GPT-6 Sol | 6% | 88% | 6% | 0% | 0% | 0% |
| Gemini 3.1 Pro | 20% | 20% | 54% | 6% | 0% | 1% |
| Gemini 3.7 Flash | 4% | 12% | 77% | 0% | 6% | 0% |
| GPT-5 Nano | 10% | 44% | 45% | 1% | 0% | 0% |

The Claude models and GPT-5.6 Sol mostly *refine* their rule: they add exceptions, patience windows and caps without changing direction.

- Opus 5 is the model that most often softens (20%) or drops its written rules in favour of habit (11%).
- Sonnet is the only model that hardens about as often as it softens.
- Both Geminis and Nano mostly repeat one template.
- GPT-6 Sol restates one rule unchanged.

**Signature styles and themes:**

| Model | Signature styles | Signature themes |
|---|---|---|
| Opus 5 | numbered law code 66% | luck vs intent 100%, characters challenging a strategy 95%, the sender deserves more 79%, someone must go first 61% |
| Opus 5.5 | rising round-counter refrain 78%, reputation 79% | same whether watched or not 66% |
| Sonnet 4.5 | reputation 70%, say your numbers aloud 44% | |
| Gemini 3.7 Flash | fixed template 94%, exactly half 95% | |
| Gemini 3.1 Pro | fixed template 68%, exactly half 84% | |
| GPT-5 Nano | abstract policy 79%, fixed template 70%, say your numbers aloud 45% | |

## 4. Defectors write about their own zeros (blind test)

| Model | Defectors with a withholding protagonist | Controls with one | Main type among defectors |
|---|---|---|---|
| Sonnet 4.5 | 60 / 60 | 3 / 20 | both confess and justify 32, confess 20, justify 8 |
| GPT-5.6 Sol | 30 / 30 | 0 / 16 | fearful character corrected 28 |
| Opus 5 | 21 / 30 | 1 / 14 | confess 13, both 6 |
| Gemini 3.7 Flash | 7 / 60 | 0 / 20 | justify 6 |
| GPT-5 Nano | 0 / 60 | 0 / 20 | |

The finding survives blinding and scale. Two corrections to the 23-defector reading:
- **Opus 5:** 70% of defectors, not all of them.
- **Gemini 3.7 Flash:** a minority of defectors (12%) do justify their own withholding.

The unblinded main reading agrees:

| Model | Defectors | Standard agents in the same runs |
|---|---|---|
| Sonnet 4.5 | 36 / 36 | 8 / 60 |
| GPT-5.6 Sol | 24 / 24 | 0 / 72 |
| Opus 5 | 22 / 24 | 1 / 72 |
| Gemini 3.7 Flash | 8 / 36 | 0 / 60 |
| GPT-5 Nano | 0 / 36 | 0 / 60 |

## 5. Not supported

- **Rule spread inside 8-agent runs.** The run-level "shared rule" code was true for all 252 runs coded, because model-generic templates count. It does not discriminate, so I report no spread rates. Reader notes do record traceable cases, for example: "'Pattern Keepers' coined by Agent_5 R7, reused by Agent_4 R8" (F291), and "Hypocrite's Stone (A4, A5)" shared between two agents in a defector run (F089). Testing spread needs a design with a null model, as in D002 and D013.
- **Causation.** These are text counts. The play check in `analyses/narrative_defector_play_check.py` links the "never nothing" rule to Opus's sends to defectors only at the level of task conditions.

## 6. When each model does what (by condition)

The percentages are standard agents prescribing any sanction. "Graded" means step down, never to zero. Cells are small:
- frontier dyads: 4 runs per cell
- frontier 8-agent: 4 runs per cell (12–32 agents)
- September: 6–10 runs per cell

The full per-cell table can be regenerated from `codes/` with the tally inputs; composition splits are listed below.

**Group size.** Dyads rarely prescribe a sanction for any model (0–62%, mostly under 40%). The same models in 8-agent groups prescribe far more:

| Model | Dyads | 8-agent groups (no defection) |
|---|---|---|
| Opus 5 | 0–50% | 92–100% |
| GPT-5.6 Sol | 25–62% | 88–100% |
| Sonnet 4.5 | 0–25% | 23–75% |

**Task order.**

| Model | Setting | Game→Myth | Myth→Game |
|---|---|---|---|
| Sonnet 4.5 | 8-agent, homogeneous | 38% | 23% |
| Sonnet 4.5 | 8-agent, mixed | 70% | 30% |
| Gemini 3.7 Flash | homogeneous, with scripted defectors | 77% (67% total exclusion) | 10% |
| Opus 5.5 | mixed | 0% | 75% |
| Opus 5.5 | homogeneous | 25% | 56% |
| Gemini 3.1 Pro | mixed | 19% | 56% |

- **September models** sanction more when they play the game before writing.
- **Gemini Flash** writes about its own withholding only in Game→Myth (8/18 defectors vs 0/18).
- **Frontier models** sanction more when the myth comes first.

**Homogeneous vs mixed.**

| Model | Homogeneous | Mixed, and alongside whom |
|---|---|---|
| Gemini 3.1 Pro | 0% | 38–75% (38–50% graded) alongside Opus 5 and GPT-5.6 Sol; 0–38% (never graded) alongside Opus 5.5 and GPT-6 Sol |
| GPT-5.6 Sol (graded share) | 0–12% graded | 50–58% graded when Opus 5 is in the population |
| GPT-5 Nano | 0% | 38–67% in Game→Myth when it is 1–2 agents among Sonnets; 0% among Gemini Flash |
| Sonnet 4.5 | 38% | 60–75% in Game→Myth with GPT-5 Nano in the population (1, 2 or 4 Nano) |
| Gemini 3.7 Flash | 0% | 0–16% (alongside Nano) |

- **Gemini 3.1 Pro** takes on graded sanctions alongside Opus 5 and GPT-5.6 Sol, but not alongside the update pair.
- **GPT-5.6 Sol** takes on Opus's graded form when Opus 5 is in the population.
- **GPT-5 Nano**, which never sanctions in homogeneous runs, does when it is a minority among Sonnets.
- **Sonnet 4.5** sanctions more with Nano in the group; Nano's game-only zero-lock gives it partners to sanction.

**Defector self-narration by condition.**
- Sonnet, Sol and Opus defectors write a withholding protagonist in both task orders and in both homogeneous and mixed runs.
- Gemini Flash defectors do so only in Game→Myth.
- Nano defectors never do.
