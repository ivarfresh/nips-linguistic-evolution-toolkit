# Do myths make agents react more to each other, or follow the myth instead? (2026-09-30)

## Answer

It depends on when the myth comes.

**Myth first (myth→game): both, and they add up.** Agents still react to
their partner's game moves about as strongly as game-only agents. On top of
that, the myth pushes their sends up in every round, beyond what the partner's
behaviour and their own last move predict. The push is small per round, but
sends carry over from round to round, so it keeps the cooperation gap open
instead of letting it fade. Agents also get a head start: they send more in
round 1, before they have seen their partner.

**Myth after round 1 (game→myth): the reaction to betrayal disappears.**
No model cuts back after a partner's forced $0 by more than chance would
produce. GPT-5 Nano's round-to-round following of its partner is near zero
there too (0.04 vs 0.40 in myth→game). This order fits "follows the myth, less
the partner" better. With 13 events per cell, that is a signal, not a proof.

- Reaction to a partner's betrayal survives a myth-first start. When a
  Sonnet 4.5 partner is forced to send $0, the next send falls $0.59 in
  game-only and $1.37 in myth→game. That is about $0.40 per $1 the partner
  dropped in both (point estimates). Both drops are larger than chance produces
  in runs with no real defections (±$0.33 and ±$0.47). In game→myth the drop is
  $0.23, inside its chance range (−$0.28 to +$0.34).
- The myth adds a push. At the same partner history and the same own previous
  send, Sonnet sends $0.14 [0.06, 0.28] more per round in myth→game.
  GPT-5 Nano sends about $1.5 more and GPT-5.6 Sol $0.20–0.29 more.
- The myth's lesson is mostly reciprocity. In Sonnet's reasoning it says
  "respond to what the other does" about 60% of the time and "give
  regardless" about 40%. Partner and myth are cited together in 67–80% of
  decisions, and the myth alone drives 11–20%.

Gemini 3.7 Flash also fits "follows the myth instead": after a myth it
answers a forced $0 by sending the full $5 again (13 events).
Opus 5 and both Geminis send the full $5 in almost every myth round. At that
ceiling, whether they still watch the partner cannot be tested.

## Evidence

All six September/frontier models, 2 and 8 agents, game / game→myth /
myth→game, 5 runs per cell. Built by `analyses/partner_responsiveness_extract.py`
(decision table, `decisions.csv`), `analyses/partner_responsiveness.py`
(all tables) and `analyses/partner_responsiveness_judge.py` (reasoning coding).
Never pooled across models.

**1. Chance shocks to the partner (the causal test).** In the 2026-09-09 dyads
with random forced defection, a partner is sometimes forced to send $0. The
schedule is keyed by replicate, round and agent, so the same forced $0s happen
in every task order. Next send, in dollars (`defection_events.csv`, 13 forced
events per cell, 95% replicate bootstrap):

| Model | Task order | Send after partner's own choice | Send after forced $0 | Change | Change per $1 the partner dropped |
|---|---|---|---|---|---|
| Sonnet 4.5 | game | 1.74 | 1.15 | −0.59 [−1.19, 0.12] | 0.40 |
| Sonnet 4.5 | game→myth | 2.81 | 2.58 | −0.23 [−1.07, 0.89] | 0.10 |
| Sonnet 4.5 | myth→game | 3.76 | 2.38 | −1.37 [−2.19, −0.71] | 0.44 |
| GPT-5 Nano | game | 0.19 | 0.00 | already at $0 | — |
| GPT-5 Nano | myth→game | 2.96 | 1.54 | −1.42 [−2.65, −0.64] | 0.64 |
| Gemini 3.7 Flash | game | 2.69 | 2.31 | −0.38 [−2.40, 1.18] | 0.15 |
| Gemini 3.7 Flash | myth→game | 4.04 | 5.00 | +0.96 [0.00, 2.32] | none (keeps giving) |

Is this bigger than chance? `defection_placebo.csv` marks fake defection
dates, using the same random schedule, in the no-defector informed-noise dyads.
There nothing happened, and 95% of the fake "changes" fall within ±$0.33
(Sonnet, game), ±$0.47 (Sonnet, myth→game) and ±$0.95 (Nano, myth→game). All
three real drops are outside these ranges. The per-dollar figures (0.40 vs
0.44) are point estimates with no interval. Read them as "similar", not "equal".
Sonnet cuts back per dollar of betrayal about as much with or without a myth
first. GPT-5 Nano can only react once a myth has lifted it off $0.
Gemini Flash after a myth is the one case that fits "follows the myth, ignores
the partner": it answers a $0 by sending everything again. In game→myth no model reacts beyond
chance: Sonnet −0.23, Nano +0.01, Gemini Flash +0.08 (`defection_events.csv`).
Gemini can't be placebo-checked because it sits at the ceiling.

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
order. That holds roughly for Sonnet in myth→game (sections 1 and 2) but may
not hold in game→myth, where the reaction to betrayal is not detectable. Read the
game→myth column with that in mind.

**5. What Sonnet 4.5 writes (`sonnet_rationale_judge_summary.csv`).** A Sonnet
4.5 judge coded 2,034 game rationales (all dyad decisions with prose, 300
per task order in 8-agent runs). The judge sees only the rationale and role.
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

- Five runs per cell, and 13 forced-defection events per model × task order.
  With 5 runs, replicate bootstraps are too narrow (the noise placebo fails in
  2 of 6 cells). The defection results are judged against their own placebo.
  The push intervals should be read as optimistic.
- Only Sonnet 4.5 writes reasoning prose. The other models return bare JSON,
  so section 4 covers Sonnet only.
- Forced-defection runs exist only for Sonnet 4.5, GPT-5 Nano and Gemini 3.7
  Flash (informed noise). The frontier models have no chance shock large
  enough to test. Their near-constant $5 in myth rounds cannot distinguish
  "still watching" from "ignoring".
- Every game prompt in a myth session ends with "Take any myths written in this
  session into account when making your decision." Agents are told to use the
  myth, so this is not an unprompted shift of attention.
- Judge spend: $6.03 (Sonnet 4.5 via OpenRouter, cached locally in `data/judge_cache/`, not committed). `decisions.csv` is not committed either; the extract script rebuilds it in about 5 s.
