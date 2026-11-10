# Planted-rule replay: Opus 5 reads a foreign rule, argues with it, and plays as before

2026-11-10. Script: `analyses/planted_rule_replay.py`. 750 Opus 5 calls, $26.65, plus $0.21 for
blind rule coding; a no-board comparison added 180 calls, $3.94. Raw calls: `data/analysis/planted_rule_replay_20261110/` (gitignored).

## Question

Two changes might make ideas travel between agents' myths: telling agents to draw on other
agents' myths instead of their own previous myth, and taking their own myths out of memory.
To see whether either works we need an idea whose origin is known. So we planted one.

## What was run

Twenty saved moments from the ten Opus 5 populations with two forced-zero defectors
(frontier defector pilot, 2026-10-01): two ordinary agents per run, each about to write a
myth (round 4 to 6) and then send, one to an ordinary co-player and one to a defector. The
myth prompt showed a shared board built from that run's own earlier myths. One myth on the
board was replaced by a planted story, "The Lamp of Oskeld", identical in three groups
except for the rule carved on the lamp-post:

| Group | Rule |
|---|---|
| none | "send none of your five coins" |
| neutral | "settle on your amount before you reach the lamp" |
| all | "send all five coins" |

The story also carries an arbitrary fact: the lamp burns green because it is fed sea salt.

Each moment was replayed in every group under four settings: own myths kept or removed
(from chat memory and from the board), crossed with the prompt line "use the myth you wrote
in the previous round" or "use the myths written by other agents on the board". Each case
made three calls: the agent writes its next myth; the same agent sends; a fresh agent reads
only that new myth and sends. One sample per call, 240 cases.

## Results

**The plant works on a direct reader.** Fresh agents who read the planted story itself sent
$3.10 (±0.57), $4.00 (±0.00) and $5.00 (±0.00) in the none, neutral and all groups, 10 each.

**The unwelcome rule is picked up and refuted. The welcome one is mostly ignored.**

| Group | Myths naming Oskeld or the crossing | Myths with the green flame | Myths coded as endorsing "send none" |
|---|---|---|---|
| none | 75 of 80 | 75 of 80 | 0 of 80 |
| neutral | 4 of 80 | 1 of 80 | 0 of 80 |
| all | 11 of 80 | 2 of 80 | 0 of 80 |

In the none group the agents retell the story to answer it: a later character carves a
second line beneath Oskeld's, or puts the green lamp out. The blind judge (GLM-5.2, rubric
of `analyses/myth_rule_judge.py`) codes 64 of 80 of these myths as "send all", the same as
in the all group (64 of 80) and above the neutral group (57 of 80).

**Play does not move.**

| | none | neutral | all |
|---|---|---|---|
| Agent's own send, ordinary co-player (n = 40) | 5.00 (±0.00) | 4.97 (±0.16) | 4.97 (±0.16) |
| Agent's own send, defector co-player (n = 40) | 0.95 (±0.73) | 0.99 (±0.80) | 0.94 (±0.44) |
| Fresh reader of the new myth (n = 80) | 4.80 (±0.43) | 4.81 (±0.51) | 4.78 (±0.53) |

All minus none, with a 95% interval over the ten runs: own send −0.02 (−0.20 to 0.17);
fresh reader −0.03 (−0.19 to 0.14). The threshold fixed in advance was $0.50.

**Neither setting changes this.** The all-minus-none gap in fresh-reader sends is 0.10
(kept, self), −0.35 (removed, self), 0.15 (kept, others) and 0.00 (removed, others); every
interval includes zero. The "other agents" wording raised mentions of the welcome story from
1 of 20 to 7 of 20 with own myths kept, and to 0 of 20 with them removed. With 20 cases per
cell that is a hint, no more.

## Comparison without a board (added the same day)

Same 20 moments, same three stories, but the planted story is shown the way the original
runs show a myth: as the one partner myth in the prompt ("Here is the myth the other agent
wrote in the previous round"). Memory and wording untouched. 180 more calls, $3.94.

| | Board (plant is 1 of about 32 myths) | No board (plant is the one myth shown) |
|---|---|---|
| Next myth names the story: none / neutral / all | 75, 4, 11 of 80 | 20, 14, 11 of 20 |
| Next myth has the green flame: none / neutral / all | 75, 1, 2 of 80 | 18, 8, 9 of 20 |
| Next myth endorses "send none" | 0 of 240 | 0 of 60 |
| Own send, ordinary co-player | 4.97 to 5.00 | 5.00 (±0.00) in every group |
| Own send, defector co-player: none / neutral / all | 0.95 (±0.73), 0.99 (±0.80), 0.94 (±0.44) | 1.25 (±1.38), 1.20 (±1.01), 1.05 (±0.55) |
| Fresh reader: none / neutral / all | 4.80 (±0.43), 4.81 (±0.51), 4.78 (±0.53) | 4.75 (±0.55), 4.85 (±0.49), 4.95 (±0.22) |
| Fresh reader, all minus none (95% interval) | −0.03 (−0.19 to 0.14) | 0.20 (−0.10 to 0.50) |

Without the board the story's content travels far more: over half the next myths reuse its
names whatever the rule, against 5 to 14% on the board unless the rule is "send none". The
rule still does not travel. No myth endorses "send none" (17 of 20 of those retellings are
coded "send all"), own sends are unchanged, and the fresh-reader gap is $0.20 with an
interval that includes zero and sits below the $0.50 threshold. With 20 cases per group the
no-board arm cannot rule out a small effect.

## Reading

A direct reader of "send none" sends $0.90 less than a neutral reader. One retelling later
the effect is gone, because the retelling is a rebuttal. Opus does take up content from
other myths, and most readily content it disagrees with, but it takes it up as something to
answer. The rule it passes on is its own. By the verdict rules set before the run, rules
do not spread through myths here, and neither proposed change makes them spread.

## Limits

- The agent's own send is pinned by the co-player's visible record: $5 to ordinary
  co-players, about $1 to defectors, in every group. Only the fresh reader had room to move
  (the direct readers show the room: $3.10 to $5.00).
- One hop, one sample per case, one planted story, one model.
- The saved runs had no board; the chat memory still holds the partner myths those runs
  showed.

## Files

`cases.csv` (one row per case), `means_by_setting_group.csv`, `means_by_partner_group.csv`,
`contrasts.csv`, `direct_reader.csv`.
