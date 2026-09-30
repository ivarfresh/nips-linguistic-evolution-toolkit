# Do myths make agents react more to each other, or follow the myth instead? (2026-09-30)

## Answer

The myth does not make agents more responsive to each other. Its main effect is
its own push: myth agents send more every round than their partner's behaviour
and their own last move predict. Whether the myth also makes them *less*
responsive is not settled. The random forced defections are too few (9 per
cell after cleaning) to tell the task orders apart. The data that exist point to
"about the same" when the myth comes first. There are hints of weaker reaction
when it comes after round 1.

- **The push.** At the same partner history and the same own previous send,
  myth→game agents send more each round: Sonnet +$0.14 [0.06, 0.28],
  GPT-5.6 Sol +$0.20, GPT-5 Nano +$1.44. Game→myth: Sol +$0.29, Nano +$1.59,
  Opus +$0.11, Sonnet +$0.10 (range includes 0). Sends carry over from round to
  round, so the push keeps the gap open. The Sonnet myth→game gap does not
  shrink over ten rounds. Myth→game also starts higher in round 1, before any
  partner information.
- **Following the partner.** Sonnet's send moves with its partner's last send
  by about the same amount in every task order: $0.54, $0.39 and $0.48 per $1
  (correlational). In its written reasoning the partner is cited as often with a
  myth as without (87–96%). The myth is added on top: partner and myth together
  drive 67–80% of decisions, the myth alone 11–20%.
- **Hints of less reaction in game→myth.** GPT-5 Nano follows its partner in
  myth→game ($0.40 per $1) but hardly at all in game→myth ($0.04). After a
  forced $0, Nano cuts back in myth→game (−$1.92, beyond chance, p = 0.01) but
  not in game→myth (+$0.41). Sonnet cuts back after a forced $0 in game-only
  (−$0.77) and myth→game (−$1.03) but not in game→myth (+$0.24). None of the
  Sonnet changes is beyond chance on its own.
- **The myth's lesson is mostly reciprocity.** In Sonnet's reasoning it says
  "respond to what the other does" about 60% of the time and "give regardless"
  about 40%.

Opus 5 and both Geminis send the full $5 in almost every myth round. At that
ceiling you cannot tell whether they still watch the partner.

## Evidence

All six September/frontier models, 2 and 8 agents, game / game→myth /
myth→game, 5 runs per condition. Some tables pool noise conditions
(15 runs per cell) or the two defection arms (10). Built by `analyses/partner_responsiveness_extract.py`
(decision table, `decisions.csv`), `analyses/partner_responsiveness.py`
(all tables) and `analyses/partner_responsiveness_judge.py` (reasoning coding).
Never pooled across models.

**1. Chance shocks to the partner.** In the 2026-09-09 dyads with random
forced defection (25% and 50% arms, 10 runs per cell), a partner is sometimes
forced to send $0. The schedule is keyed by replicate, round and agent, so the
same forced $0s happen in every task order. Next send ($) after a forced $0
versus after the partner's own choice. The comparison drops decisions whose
partner's return two rounds earlier was also forced. The chance range comes
from shuffling the forced-$0 labels within each run 5,000 times
(`defection_events.csv`, `defection_placebo.csv`):

| Model | Task order | Forced $0s | After own choice | After forced $0 | Change | Chance range | p |
|---|---|---|---|---|---|---|---|
| Sonnet 4.5 | game | 9 | 1.99 | 1.22 | −0.77 | −1.24 to 0.41 | 0.19 |
| Sonnet 4.5 | game→myth | 9 | 2.98 | 3.22 | +0.24 | −0.07 to 0.79 | 0.76 |
| Sonnet 4.5 | myth→game | 9 | 3.80 | 2.78 | −1.03 | −1.58 to 0.24 | 0.20 |
| GPT-5 Nano | game | 9 | 0.23 | 0.00 | at $0 | — | — |
| GPT-5 Nano | game→myth | 9 | 1.48 | 1.89 | +0.41 | −1.00 to 0.56 | 0.40 |
| GPT-5 Nano | myth→game | 9 | 3.26 | 1.33 | −1.92 | −1.69 to 0.42 | 0.01 |
| Gemini 3.7 Flash | game | 9 | 2.96 | 3.33 | +0.38 | −1.19 to 1.94 | 1.00 |
| Gemini 3.7 Flash | game→myth | 9 | 3.64 | 3.89 | +0.25 | −0.53 to 1.82 | 1.00 |
| Gemini 3.7 Flash | myth→game | 9 | 4.32 | 5.00 | +0.68 | −0.10 to 0.68 | 0.50 |

Only Nano in myth→game reacts beyond chance. The shuffle ignores that the 25%
and 50% arms share events, so even these p-values are optimistic. An earlier
version of this table compared against fake defection dates in the calmer
no-defector runs. That understated chance; the independent review on PR #13
caught it.

The smaller shock, communication noise (at most $1 off what the partner is
shown to do), gave no consistent answer (`responsiveness.csv`). A placebo
check fails: in no-noise runs, the draw that *would* have been applied shows
"effects" in 2 of 6 cells. With 5 runs per cell the bootstrap intervals are
too narrow, so the noise estimates can't separate task orders. They are
reported but not relied on.

**2. How closely sends track the partner (correlational, all no-defector
dyads; `partner_following.csv`).** Own send per $1 of the partner's last
shown send, with run and round averages removed:

| Model | game | game→myth | myth→game |
|---|---|---|---|
| Sonnet 4.5 | 0.54 [−0.14, 0.81] | 0.39 [−0.05, 0.55] | 0.48 [−0.02, 0.57] |
| GPT-5.6 Sol | 0.62 [−0.28, 0.89] | 0.50 [0.34, 0.59] | 0.24 [0.00, 0.38] |
| GPT-5 Nano | at $0, undefined | 0.04 [−0.09, 0.19] | 0.40 [0.29, 0.53] |
| Opus 5, Gemini 3.1 Pro, Gemini 3.7 Flash | near or at $5 | at or near $5 | $5 every round, nothing to track |

Sol's slope drops in myth→game, where its sends are nearly all $5
(within-run sd 0.14 vs 0.42 in game). That drop is the ceiling showing up.

**3. The opening move (`opening_vs_later.csv`).** Mean send ($), myth→game
minus game only:

| Model | Round 1 (before any partner info) | Rounds 2–10 |
|---|---|---|
| Sonnet 4.5, 2 agents | +0.70 | +0.75 |
| Sonnet 4.5, 8 agents | +1.05 | +1.39 |
| Opus 5, 2 / 8 agents | +1.00 / +1.65 | +0.69 / +0.52 |
| GPT-5.6 Sol, 2 / 8 agents | +0.80 / +0.88 | +1.40 / +1.07 |
| GPT-5 Nano, 2 / 8 agents | +2.80 / +2.80 | +3.54 / +1.91 |
| Gemini 3.1 Pro / 3.7 Flash | ≈0 (ceiling) | ≈0 (ceiling) |

A head start alone would fade. Sonnet follows its partner at about $0.4–0.5
per $1, so a round-1 gap should shrink each round. It doesn't
(`gap_by_round.csv`): the Sonnet myth→game gap stays between $0.4 and $0.9 in dyads (no
downward trend; $0.9 in rounds 9–10) and
grows from $1.1 to $1.5 in 8-agent runs. Game→myth has no round-1 gap by design
but gains later anyway (Sonnet dyads +0.56, Opus +0.63 / +0.45, Nano
+3.19 / +1.44). Opus is the exception: its gap shrinks to $0.0–0.3 by rounds 9–10 because
game-only Opus also climbs to nearly $5.

**4. The per-round push (`myth_push.csv`).** In dyads, send ($) at the same
partner last send and the same own previous send. Task-order terms relative to
game only; the bootstrap is over runs:

| Model | game→myth | myth→game | per $1 of partner's last send | per $1 of own previous send |
|---|---|---|---|---|
| Sonnet 4.5 | +0.10 [−0.03, 0.23] | +0.14 [0.06, 0.28] | 0.36 | 0.59 |
| GPT-5.6 Sol | +0.29 [0.13, 0.49] | +0.20 [0.02, 0.57] | 0.57 | 0.44 |
| GPT-5 Nano | +1.59 [1.23, 2.20] | +1.44 [1.03, 1.99] | 0.49 | 0.20 |
| Opus 5 | +0.11 [0.06, 0.19] | +0.03 [−0.00, 0.09] | 0.14 | 0.54 |
| Gemini 3.1 Pro / 3.7 Flash | ≈0 | ≈0 | — | — (ceiling) |

This model assumes the reaction to the partner is the same in every task
order. Section 2 supports that for Sonnet. For Nano it may not hold in
game→myth, where it barely follows its partner.

**5. What Sonnet 4.5 writes (`sonnet_rationale_judge_summary.csv`).** A Sonnet
4.5 judge coded 2,034 game rationales (all dyad decisions with prose, 300
per task order in 8-agent runs). The 2-agent rows pool 15 no-defector runs
with 10 random-defection runs per task order. The judge sees only the rationale and role.
Per-run shares, mean (±std):

| | game | game→myth | myth→game |
|---|---|---|---|
| Cites partner's moves, 2 agents | 0.88 (±0.19) | 0.87 (±0.20) | 0.87 (±0.20) |
| Cites partner's moves, 8 agents | 0.96 (±0.04) | 0.94 (±0.05) | 0.91 (±0.05) |
| Main driver = partner and myth together, 2 / 8 agents | 0 / 0 | 0.67 / 0.76 | 0.73 / 0.80 |
| Main driver = myth alone, 2 / 8 agents | 0 / 0 | 0.11 / 0.15 | 0.20 / 0.19 |
| Myth lesson is conditional (reciprocate), 2 / 8 agents | — | 0.59 / 0.62 | 0.61 / 0.62 |
| Myth lesson is unconditional (give anyway), 2 / 8 agents | — | 0.39 / 0.38 | 0.36 / 0.38 |

The myth is added on top of the partner in Sonnet's reasoning. It does not
replace the partner. A hand check of 14 myth-coded rationales found every
label defensible. The unconditional lesson reads like this: "the wanderer
gave all five vessels, regardless of who stood across the water... This
wasn't strategy—it was identity." A simple keyword count gives the same picture
(`sonnet_rationale_mentions.csv`). In myth runs agents also read the partner's
myth. Sonnet sometimes uses it as news about the partner ("their myth
acknowledged this as a breaking of trust"). The myth is thus partly another
channel to the partner.

## Limits

- The forced-defection test has 9 clean events per cell and is judged against
  a within-run shuffle. With 5 runs per condition, replicate bootstraps are too
  narrow (the noise placebo fails in 2 of 6 cells), so read the push and
  following intervals as optimistic.
- Only Sonnet 4.5 writes reasoning prose. The other models return bare JSON,
  so section 5 covers Sonnet only.
- Forced-defection runs exist only for Sonnet 4.5, GPT-5 Nano and Gemini 3.7
  Flash (informed noise). The frontier models have no chance shock large
  enough to test. Their near-constant $5 in myth rounds cannot distinguish
  "still watching" from "ignoring".
- Every game prompt in a myth session ends with "Take any myths written in this
  session into account when making your decision." Agents are told to use the
  myth, so this is not an unprompted shift of attention.
- Judge spend: $6.03 (Sonnet 4.5 via OpenRouter, cached locally in `data/judge_cache/`, not committed). `decisions.csv` is not committed either; the extract script rebuilds it in about 5 s.
