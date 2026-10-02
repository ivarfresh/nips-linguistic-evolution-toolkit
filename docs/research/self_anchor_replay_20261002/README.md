# Self-anchor replay: dropping "use your previous myth as inspiration" lowers borrowing

*2026-10-02. Script: `analyses/self_anchor_replay.py` (measures and decision rule committed
before any call). Spend $5.84, 280 calls, 0 errors, 0 refusals.*

## The answer

No. The line does not hold back copying from the partner. Without it, Sonnet borrows
slightly **less** from the partner's myth, and it copies its own previous myth just as much.
So the line is not why agents anchor on themselves. The own myth sitting in chat memory is
the likelier reason.

## How it works

Every later-round myth prompt in September ends with "Write your own myth. Use the myth you
wrote in the previous round as inspiration, but adapt it in your own way." We took 70 logged
Sonnet 4.5 myth calls from the 20 homogeneous September runs (all 10 seats in each dyad cell,
25 per 8-agent cell, one round per seat drawn from 2–10). Each call was sent again
twice: once exactly as logged, and once with only the second sentence deleted. There were 2 samples
per version and the recorded request plan (thinking on). The partner's myth stays in the
prompt. The agent's own previous myth stays in its chat memory, so this tests the
instruction, not taking the own myth away.

**Borrowing** is the share of the partner myth's words that the agent had never used before
and now uses, minus the same share for a myth it never saw (same family, same round). This
is the measure behind the 1.3–2.5× uptake in the September linguistic analysis. **Self-copying**
is the share of the agent's previous myth's words that it reuses.

## Results

Run-clustered means, 95% t intervals over runs (`summary.csv`):

| | With line | Without line | Difference | Runs where it rose |
|---|---|---|---|---|
| Borrowing from partner (excess over unseen) | 8.7 pts | 6.7 pts | **−2.0** (−3.5 to −0.5) | 4 of 20 |
| Raw partner-word uptake | 16.0% | 13.9% | −2.1 (−3.8 to −0.5) | 4 of 20 |
| Uptake from unseen myth | 7.3% | 7.2% | −0.2 (−0.7 to 0.4) | 10 of 20 |
| Self-copying | 32.4% | 32.1% | −0.2 (−1.2 to 0.8) | 7 of 20 |
| Length (words) | 210 | 209 | −1.2 (−2.7 to 0.4) | 8 of 20 |

- The drop is the same in 8-agent groups (−2.1, −3.6 to −0.5) and dyads (−1.9, −4.9 to
  +1.1, only 10 runs).
- Sanity check: the logged myths and the with-line replays score the same on every measure
  (borrowing +0.2, −1.4 to +1.8), so the replays reproduce the original behaviour.

By the rule fixed in advance ("the line holds back borrowing" only if the difference is
above 0), the answer is no. The difference is negative and its interval excludes 0.

## What it means

- **The self-copying is not caused by the instruction.** Deleting it leaves self-copying at
  32%. The likelier source is the agent's own last myth in its chat memory, but this replay
  never removed it, so that is an inference, not a measurement.
- **"Adapt it in your own way" may invite mixing.** Without the line, the prompt reduces to
  "write your own myth" plus the game directive, and Sonnet takes a little less from the
  partner. This reading is a guess; the data only show the direction.
- **For the planned transmission pilot**
  (`docs/research/cultural_transmission_pilot_2026-10-02.md`): removing the line will not
  help spread and may slightly hurt it. Keep the September wording, which also keeps the
  pilot comparable to earlier runs.

## Caveats

Sonnet only, word-level measures only (no amount or moral measures), 20 run clusters,
single replayed calls rather than whole runs. Effects are small (2 points on a 9-point
baseline).
