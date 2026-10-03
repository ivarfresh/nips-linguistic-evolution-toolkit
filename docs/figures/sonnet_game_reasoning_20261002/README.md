# Sonnet's game reasoning: does writing myths change how Sonnet talks about play? 2026-10-02

A companion to the [myth map](../myth_convergence_map_20261002/README.md),
which has no game-only condition because game-only runs have no myths. The
game replies are the only text those runs produce.

**Answer.** Yes. Without myths, Sonnet explains its moves in game terms
(*cooperative, reciprocate, history, behavior*). Once a myth exists, it
explains them with the myth (*myth, honor, courage*), from the first round
after a myth is written, and it never goes back.

## Why only Sonnet

The game prompt asks only for a JSON decision ("IMPORTANT: Provide your
decision in the correct JSON format."). Every September model got the same
prompt. Sonnet 4.5 adds an explanation anyway: 4,394 of its 4,410 game replies have
10 or more words. GPT-5 Nano and Gemini 3.7 Flash reply with JSON only. GPT's stored reasoning field is a placeholder ("[N
reasoning tokens used, but content encrypted by provider]"). So a
cross-family version of this map is not possible from the existing runs.

## Data and method

Every game reply by a Sonnet agent in the 111 September runs that contain
Sonnet (the validated tables behind Figures 7 and 8): Sonnet + Sonnet,
Sonnet + GPT and Sonnet + Gemini dyads; 8 Sonnet and Sonnet among 1, 2 or 4
GPT populations; game only, Game → Myth and Myth → Game.

- The text is the reply with the JSON removed.
- Replies under 10 words are dropped: 16 of 4,410, all game only with GPT
  partners. They are short "I received $0, nothing to return" replies and
  bare-JSON replies.
- A retried move keeps only its last reply, the one the game used.

That leaves 4,394 texts (`texts_per_cell.csv`). They are embedded with
all-mpnet-base-v2 and projected onto the first two principal components
(17% and 8% of the variance). Numbers are computed in the full 768-d space.

```
python3 analyses/sonnet_game_reasoning_map.py   # free; ~4 min the first time (embeddings)
```

**What differs between the conditions' game prompts.** In both myth
conditions every game prompt ends with "Take any myths written in this
session into account when making your decision." In later rounds the prompt
also contains the myths themselves. Game only has neither. This map
therefore shows what Sonnet does with myths it is told to consider. It is
not spontaneous recall.

## How to read the figures

- `round1.png`: round-1 explanations, senders left and receivers right.
  - Game → Myth's round-1 prompt differs from Game only by the one
    instruction line, with no myth written yet. Receivers also see their
    partner's round-1 send, which can differ between conditions.
  - Myth → Game has already written a myth.
- `trajectories_8agent.png`, `trajectories_2agent.png`:
  - Rows: Sonnet as sender, Sonnet as receiver. The roles alternate by round,
    so they are drawn apart.
  - Columns: who Sonnet plays with.
  - Line: task-order average from round 1 (light) to round 10 (dark).
  - Small dots: each run's average at rounds 1, 5 and 10.
  - Grey background: where round-10 explanations end up.

## What we see

**1. Round 1: the instruction alone shifts the explanations slightly; a
written myth shifts them about ten times more.** Distance from the game-only
centroid (`distance_from_game_only.csv`, cosine, 768-d). The test shuffles
which runs are game-only and which are the myth condition (group sizes kept);
p is one-sided, from 2,000 shuffles.

| Round 1, 8 agents (4 cells) | Game → Myth | Myth → Game |
|---|---|---|
| Distance from game only | 0.010–0.029 | 0.22–0.28 |
| p | 0.007–0.032 | 0.0005–0.011 |

In 8-agent runs, the one line "Take any myths written in this session into
account" already moves Sonnet's first explanation, though only a little. In
2-agent runs the round-1 cells are too small to tell (next point).

**2. After one myth, Game → Myth joins Myth → Game.**
- In 8-agent runs, both myth conditions differ from game only in every cell
  and round from 2 to 10: 36 of 36 each, p < 0.05. At round 10 they are
  0.10–0.21 from game only.
- In Sonnet + Sonnet dyads the same holds in 18 of 18 cells per condition
  (largest p = 0.017).
- In mixed dyads, Sonnet holds a given role in a given round in only 3 runs
  per condition. With 3 runs against 3 the smallest possible p is 0.05, and
  the cells sit at or near it (0.05–0.12). The direction matches, but the
  test cannot reach significance there.
- Game only stays on the left of the map for all 10 rounds.

**3. Left-right on the map is game talk against myth talk.** Words most
over-used relative to game only (`distinctive_words.csv`, log-odds with an
informative prior, stop words removed, 2- and 8-agent runs pooled; rate per
1,000 non-stop words):

| Word | Game only | Game → Myth | Myth → Game |
|---|---|---|---|
| cooperative | 17.3 | 4.1 | 1.6 |
| reciprocate | 10.2 | 2.2 | 1.0 |
| history | 20.1 | 9.2 | 6.9 |
| myth / myths | 0 / 0 | 9.6 / 12.4 | 11.6 / 15.3 |
| honor | 0.04 | 6.5 | 7.4 |
| courage | 0 | 6.2 | 10.3 |

A typical Myth → Game reply (2 agents, round 4): "Both our myths now emphasize
the same wisdom … I must continue the sacred pattern, offering nearly all my
starlight to the valley's magic."

**4. The cluster at the bottom of "Receivers · Sonnet + GPT" is GPT's
zero-lock.** These are Sonnet receiving $0 ("Since the sender sent $0, I
received $0. I have no funds to return."): 23 of the 25 game-only receiver
texts in those dyads. With a myth task the line climbs out, in line with the
earlier finding that the myth task breaks the lock.

## Caveats

- The myth conditions are told to consider the myths and have them in
  context. The shift shows that Sonnet uses them as its stated reason. It does
  not show that Sonnet would bring them up unprompted.
- This is language, not behaviour. Sending and returning are in the
  cooperation figures.
- The 2-agent mixed cells have 3 runs per condition per role and round, too
  few for the run-level test (see point 2). Pooling across rounds would need
  a model with run effects. That is not done here.
- The 2-D map keeps 25% of the variance. Cite the 768-d numbers.
