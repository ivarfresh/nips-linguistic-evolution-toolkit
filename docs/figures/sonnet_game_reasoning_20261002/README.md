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
prompt. Sonnet 4.5 adds an explanation anyway. GPT-5 Nano and Gemini 3.7 Flash
reply with JSON only. GPT's stored reasoning field is a placeholder ("[N
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
    instruction line, with no myth written yet.
  - Myth → Game has already written a myth.
- `trajectories_8agent.png`, `trajectories_2agent.png`:
  - Rows: Sonnet as sender, Sonnet as receiver. The roles alternate by round,
    so they are drawn apart.
  - Columns: who Sonnet plays with.
  - Line: task-order average from round 1 (light) to round 10 (dark).
  - Small dots: each run's average at rounds 1, 5 and 10.
  - Grey background: where round-10 explanations end up.

## What we see

**1. Round 1: the instruction alone does nothing, a written myth does a lot.**
Distance from the game-only centroid (`distance_from_game_only.csv`, cosine,
768-d), against a noise floor: the distance between two random halves of the
game-only runs.

| Round 1 | Noise floor | Game → Myth | Myth → Game |
|---|---|---|---|
| 8 agents (4 cells) | 0.010–0.018 | 0.010–0.029 | 0.22–0.28 |
| 2 agents (6 cells) | 0.05–0.31 | 0.03–0.15 | 0.25–0.44 |

**2. After one myth, Game → Myth joins Myth → Game.** From round 2 both myth
conditions sit above the noise floor in every cell and round (90 of 90). In
8-agent runs they are 0.10–0.21 from game only at round 10, against a floor
of 0.011–0.016. Game only stays on the left of the map for all 10 rounds.

**3. Left-right on the map is game talk against myth talk.** Words most
over-used relative to game only (`distinctive_words.csv`, log-odds with an
informative prior; rate per 1,000 words):

| Word | Game only | Game → Myth | Myth → Game |
|---|---|---|---|
| cooperative | 10.0 | 2.3 | 0.9 |
| reciprocate | 5.9 | 1.3 | 0.6 |
| history | 11.6 | 5.3 | 4.0 |
| myth / myths | 0 / 0 | 5.5 / 7.1 | 6.7 / 8.8 |
| honor | 0.02 | 3.8 | 4.3 |
| courage | 0 | 3.6 | 5.9 |

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
- The 2-agent noise floor is wide (5–6 runs per half). In Sonnet + GPT dyads
  at round 1 it is 0.25–0.31, close to the Myth → Game distance (0.35–0.44).
  The 8-agent floor is tight.
- The 2-D map keeps 25% of the variance. Cite the 768-d numbers.
