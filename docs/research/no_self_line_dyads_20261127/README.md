# Full dyad runs without the self-inspiration sentence

*2026-11-27. Launcher `scripts/run_no_self_line_dyads.py`, analysis
`analyses/no_self_line_dyads.py`. 10 runs, 10/10 audited, $7.98.*

## Question

Every later-round myth prompt in the paper's runs says "Write your own myth. Use the myth
you wrote in the previous round as inspiration, but adapt it in your own way." The October
replay (`../self_anchor_replay_20261002/README.md`) deleted that second sentence in single
re-sent calls. Here we delete it for whole runs, so the change can build up over ten rounds
and reach the game.

## What was run

Sonnet 4.5 dyads, game→myth and myth→game, five replicates each (0–4). Everything matches
the September baseline runs (`negative_only_reasoning_rerun_dyad_{game_myth,myth_game}_claude_n5`,
no-defector cell): informed negative-only noise, memory-primary chat memory, thinking budget
8192, and the same noise seed per replicate. The launcher checks before any call that the
later-round myth prompt is the only input that differs, and afterwards that the sentence
appears in no prompt sent. The sentence is deleted; nothing replaces it. The agent's own
earlier myths stay in its chat memory.

## Results

Mean (±SD) over runs; difference paired by replicate with a 95% t interval
(`summary.csv`, `per_run.csv`).

Both task orders together (10 pairs):

| | With sentence | Without | Difference |
|---|---|---|---|
| Resources per agent, round 10 | 56.7 (±6.4) | 60.1 (±9.4) | +3.4 (−5.1 to 11.8) |
| Mean send ($) | 3.17 (±0.64) | 3.51 (±0.94) | +0.34 (−0.51 to 1.18) |
| Return share | 0.455 (±0.079) | 0.449 (±0.079) | −0.005 (−0.079 to 0.068) |
| Self-copying | 35.2% (±5.2) | 26.8% (±4.0) | **−8.4 pts (−13.5 to −3.3)**, lower in 8 of 10 |
| Partner-word uptake | 14.3% (±2.9) | 12.8% (±2.1) | −1.4 (−4.5 to 1.6) |
| Borrowing (uptake over an unseen myth) | 8.2 pts (±3.0) | 6.4 pts (±2.2) | −1.8 (−5.0 to 1.3) |

Per task order (5 pairs each):

| | game→myth: with | without | difference | myth→game: with | without | difference |
|---|---|---|---|---|---|---|
| Resources | 55.6 (±4.6) | 55.3 (±6.8) | −0.3 (−13.8 to 13.2) | 57.8 (±8.2) | 64.8 (±9.7) | +7.0 (−8.9 to 22.9) |
| Mean send ($) | 3.06 (±0.46) | 3.03 (±0.68) | −0.03 (−1.38 to 1.32) | 3.28 (±0.82) | 3.99 (±0.97) | +0.70 (−0.89 to 2.29) |
| Return share | 0.426 (±0.028) | 0.425 (±0.052) | −0.001 (−0.090 to 0.088) | 0.483 (±0.107) | 0.474 (±0.098) | −0.010 (−0.179 to 0.159) |
| Self-copying | 34.3% (±5.8) | 25.8% (±3.5) | −8.5 (−17.2 to 0.2) | 36.1% (±5.0) | 27.8% (±4.7) | −8.3 (−18.3 to 1.8) |
| Borrowing | 8.8 pts (±3.9) | 5.7 pts (±1.9) | −3.1 (−8.7 to 2.5) | 7.7 pts (±2.1) | 7.2 pts (±2.3) | −0.6 (−6.0 to 4.9) |

## Reading

- **Self-copying falls when the sentence is gone for a whole run.** Agents reuse about a
  quarter of their previous myth's words instead of a third. The single-call replay saw no
  change (−0.2 pts). The likely reason: in the replay the agent's memory still held myths
  written under the sentence, each already close to the one before. This reading was not
  tested.
- **Agents do not borrow more from the partner instead.** Borrowing is, if anything, a
  little lower, the same direction as in the replay (−2.0 pts there).
- **Cooperation: no detectable change.** Game→myth is flat. Myth→game is 7 resource points
  higher on average, but only 3 of 5 pairs go up and the interval is −9 to +23. Five pairs
  cannot tell a real effect of that size from run-to-run spread (SD 8 to 10).

## Limits

Sonnet 4.5 only, dyads only, five replicates per task order, no-defector condition. The
unseen-myth baseline uses the same seat's myth in the other four runs of the same condition
and arm. Word measures only; no amount or moral coding. The paired design shares noise
seeds, not model sampling.

## Mixed dyads: Sonnet 4.5 with GPT-5 Nano and with Gemini 3.7 Flash (added the same day)

*Launcher `scripts/run_no_self_line_mixed_dyads.py`, analysis
`analyses/no_self_line_mixed_dyads.py`. 20 runs, 20/20 audited, $9.54.*

Same deleted sentence, for both agents. Two pairs × game→myth / myth→game × replicates 0–4,
each paired with the September mixed-dyad run of the same replicate (same seeds, same first
sender; Sonnet sends first in replicates 0/2/4, the partner in 1/3). One route difference:
Gemini 3.7 Flash goes through Vertex AI with the same request settings, because the Gemini
API now answers 3.7 requests with 3.8; the September baseline used the Gemini API. The audit
confirms every Gemini call was answered by 3.7.

Both task orders together, 10 pairs per row, mean (±SD), paired difference with a 95% interval
(`mixed_summary.csv`, `mixed_per_run.csv`):

| Pair | Measure | With sentence | Without | Difference |
|---|---|---|---|---|
| Sonnet + GPT | Resources per agent | 60.1 (±11.3) | 57.2 (±8.9) | −2.9 (−11.6 to 5.8) |
| | Sonnet send ($) | 3.49 (±1.09) | 3.33 (±0.85) | −0.17 (−0.96 to 0.62) |
| | GPT send ($) | 3.53 (±1.23) | 3.11 (±1.01) | −0.42 (−1.40 to 0.57) |
| | Sonnet self-copying | 35.2% (±4.0) | 28.4% (±2.3) | **−6.8 pts (−10.2 to −3.4)** |
| | GPT self-copying | 38.8% (±5.1) | 34.0% (±3.3) | −4.8 pts (−9.8 to 0.2) |
| | Sonnet borrowing | 1.2 pts (±1.2) | 0.8 pts (±1.3) | −0.4 (−1.4 to 0.6) |
| | GPT borrowing | 1.9 pts (±0.8) | 1.9 pts (±0.8) | +0.1 (−0.9 to 1.0) |
| Sonnet + Gemini | Resources per agent | 73.5 (±0.9) | 73.7 (±0.8) | +0.2 (−0.7 to 1.0) |
| | Sonnet send ($) | 4.71 (±0.19) | 4.74 (±0.17) | +0.03 (−0.14 to 0.21) |
| | Gemini send ($) | 5.00 (±0.00) | 5.00 (±0.00) | 0 |
| | Sonnet self-copying | 36.6% (±8.3) | 29.4% (±2.4) | **−7.2 pts (−12.7 to −1.7)** |
| | Gemini self-copying | 30.6% (±4.8) | 24.0% (±3.2) | **−6.6 pts (−11.1 to −2.1)** |
| | Sonnet borrowing | 3.5 pts (±1.7) | 1.1 pts (±2.0) | **−2.5 (−4.4 to −0.5)** |
| | Gemini borrowing | 1.7 pts (±1.1) | 1.2 pts (±1.7) | −0.5 (−2.2 to 1.1) |

- **The self-copying drop repeats in every family.** Sonnet −7 points with either partner
  (−8.4 in Sonnet-only dyads), Gemini −6.6, GPT −4.8 (interval just reaches zero).
- **Borrowing does not rise anywhere.** With Gemini as partner, Sonnet borrows less
  (−2.5 points, lower in 9 of 10 pairs); elsewhere there is no change.
- **Cooperation is unchanged.** Sonnet + Gemini stays at the ceiling (73.5 vs 73.7).
  Sonnet + GPT is 2.9 points lower with an interval of −11.6 to 5.8.

Extra limits: three of five replicates per cell have Sonnet sending first; the Gemini route
differs from the baseline (Vertex vs Gemini API, same model id confirmed per call).

### Third pairing: Gemini 3.7 Flash with GPT-5 Nano (added the same day)

Ten more runs, same design (10/10 audited, $1.83; all 30 mixed runs $11.37). Both task
orders together, 10 pairs:

| Measure | With sentence | Without | Difference |
|---|---|---|---|
| Resources per agent | 72.1 (±2.7) | 71.1 (±3.6) | −1.0 (−2.9 to 1.0) |
| GPT send ($) | 4.41 (±0.54) | 4.22 (±0.72) | −0.19 (−0.59 to 0.20) |
| Gemini send ($) | 5.00 (±0.00) | 5.00 (±0.00) | 0 |
| Gemini self-copying | 29.0% (±3.2) | 24.8% (±3.5) | −4.2 pts (−8.5 to 0.1) |
| GPT self-copying | 39.4% (±3.2) | 36.9% (±3.6) | −2.5 pts (−5.3 to 0.4) |
| Gemini borrowing | 2.5 pts (±1.7) | 0.2 pts (±0.9) | **−2.3 (−3.8 to −0.9)** |
| GPT borrowing | 1.2 pts (±1.3) | 0.5 pts (±0.8) | −0.7 (−2.0 to 0.6) |

Same picture: no rise in borrowing without the sentence (Gemini borrows less from GPT, lower
in 9 of 10 pairs), smaller self-copying drops that do not exclude zero, cooperation unchanged.

## Figure

`no_self_line_transmission_cooperation.png` (`analyses/plot_no_self_line_dyads.py`): word
adoption in the format of paper Figure 5a (left) and final resources in the format of paper
Figure 3 (right), with and without the sentence.
