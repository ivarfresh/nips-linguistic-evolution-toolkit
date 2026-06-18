# Missing-Condition Model Sweep Briefing

Generated Monday 2026-05-04 after rebuilding `cell_summaries/`.

## Completion

| model | sweep | final JSONs | status |
|---|---:|---:|---|
| Gemini-3.1-pro-preview | positive | 90/90 | complete |
| Gemini-3.1-pro-preview | negative | 90/90 | complete |
| Gemini-3.1-pro-preview | bootstrap | 90/90 | complete |
| GPT-5.5 | positive | 77/90 | quota-blocked |
| GPT-5.5 | negative | 1/90 | quota-blocked |
| GPT-5.5 | bootstrap | 1/90 | quota-blocked |

The GPT-5.5 failures are OpenAI quota failures. The launcher stop flag is present at `/tmp/nlet-runs/missing-conditions/_stop_due_openai_quota`, so no further GPT-5.5 API calls are being attempted.

Gemini has some stale checkpoint/error JSONs from earlier partial attempts in the positive set, but all 90 final files are present and the cell-summary rebuild includes complete Gemini cells.

## Cell-Level Results

### Gemini-3.1-pro-preview

Gemini is the useful completed model expansion. Positive noise is saturated near ceiling, negative noise stays low, and bootstrap does **not** reproduce the GPT-5-Nano destabilisation.

| noise | informed | game | game -> myth | myth -> game |
|---|---:|---:|---:|---:|
| positive_5 | N | 74.49 +/- 1.25 | 74.40 +/- 1.04 | 74.73 +/- 0.54 |
| positive_5 | Y | 74.13 +/- 1.31 | 74.01 +/- 1.68 | 74.78 +/- 0.39 |
| negative_5 | N | 28.84 +/- 3.36 | 27.44 +/- 2.69 | 26.40 +/- 1.53 |
| negative_5 | Y | 26.71 +/- 1.75 | 25.87 +/- 1.02 | 27.55 +/- 3.96 |
| bootstrap | N | 62.63 +/- 10.52 | 67.63 +/- 7.54 | 71.80 +/- 3.79 |
| bootstrap | Y | 60.10 +/- 11.41 | 64.77 +/- 6.61 | 69.57 +/- 2.95 |

Interpretation: Gemini should be treated as model-boundary evidence. It is not a replication of the GPT-5-Nano bootstrap mechanism; if anything, the bootstrap myth-present cells are higher than game-only.

### GPT-5.5

GPT-5.5 is mostly unusable because the quota block arrived before the informative negative/bootstrap cells.

| noise | informed | game | game -> myth | myth -> game |
|---|---:|---:|---:|---:|
| positive_5 | N | 74.96 +/- 0.15 | 74.87 +/- 0.50 | 75.00 +/- 0.00 |
| positive_5 | Y | 74.94 +/- 0.23 | 74.95 +/- 0.13 | 75.00 +/- 0.00 (n=2) |
| negative_5 | N | 28.05 +/- 0.00 (n=1) | missing | missing |
| bootstrap | N | 71.00 +/- 0.00 (n=1) | missing | missing |

Interpretation: the completed GPT-5.5 positive cells are ceiling-saturated and should not change the abstract. The negative/bootstrap GPT-5.5 cells remain pending unless OpenAI quota is fixed and the team explicitly wants to chase robustness.

## Paper Implication

Use these runs as robustness/boundary evidence, not as the abstract spine. The safe story remains: GPT-5-Nano under bootstrap noise shows the sharp destabilisation/stabilisation pattern; Gemini shows that this is not universal across frontier models; GPT-5.5 is incomplete.

The Monday meeting packet already records this status and recommends prioritising writing and GPT-5-Nano prompt variants over chasing GPT-5.5 before the abstract deadline.
