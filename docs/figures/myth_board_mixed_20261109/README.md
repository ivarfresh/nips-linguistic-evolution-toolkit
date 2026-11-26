# A shared myth board in a 4 GPT-5 Nano + 4 Sonnet 4.5 population, 2026-11-09

**Headline.** Replacing the partner's myth with a board of every myth written so far,
shown before every myth and every game decision from round 2, did not raise cooperation.
Ten seed-paired runs ended 2.5 points lower than their partner-myth twins (51.5 vs 54.0,
paired t p 0.53, 6 of 10 pairs up). The myths did not converge either.

## What was run

- **Population:** 8 agents, 4 GPT-5 Nano (Agent_1-4) + 4 Claude Sonnet 4.5 (Agent_5-8),
  balanced rotating pairs, hidden names, September informed negative-only noise, 10 rounds,
  game -> myth, no defectors. The Table 1 cell with the most room below the ceiling of 75
  (game only 38.9, game -> myth 54.0, myth -> game 63.8).
- **Board (`myth_board: persistent_game`):** from round 2 each agent's myth prompt and
  game prompt start with every myth written in rounds 1..r-1 by all eight agents, without
  author names, in a per-run shuffled order (8(r-1) myths, 72 at round 10). The board is
  pasted fresh into each prompt and kept out of chat memory. Design:
  `docs/research/myth_board_design_v2_20261106.md`.
- **Twins:** `mixed_pop_game_myth_gpt4_sonnet4_n5` (replicates 0-4) and `_n5ext` (5-9),
  same seeds, pairings, noise draws and per-agent request plans; only the later-round myth
  template and the board flag differ (asserted by the launcher before launch).
- **Runs:** 10, all audited (`mixed_board_completion_receipt.json`), $44.27 at standard
  rates ($41.56 Anthropic, $2.71 OpenAI). Launcher
  `scripts/run_myth_board_pilot.py --stage mixed_board`; analysis
  `scripts/analyze_myth_board_mixed_20261109.py`.

## Results

Final resources, mean (±sd) over 10 runs; `board_vs_partner_by_replicate.csv`.

| Agents | Partner myth | Board | Board minus partner | pairs up | paired t p |
|---|---|---|---|---|---|
| All 8 | 54.0 (±8.9) | 51.5 (±7.4) | -2.5 | 6/10 | 0.53 |
| Nano | 55.0 (±9.5) | 52.9 (±7.9) | -2.1 | 6/10 | 0.59 |
| Sonnet | 52.9 (±8.5) | 50.1 (±7.4) | -2.8 | 5/10 | 0.48 |

Per replicate (partner -> board): 63.0->49.5, 46.0->55.0, 60.3->48.1, 52.5->55.8,
56.2->36.3, 50.9->56.5, 49.8->60.2, 69.5->52.2, 37.9->43.2, 53.5->58.0. The seed pairing
buys little here: the two conditions differ by up to 20 points in either direction.

Mean send decision per investor -> trustee family (of $5): Nano->Nano 2.50 -> 2.33,
Nano->Sonnet 2.82 -> 2.17, Sonnet->Nano 3.00 -> 2.89, Sonnet->Sonnet 3.24 -> 3.24.
Sends rise over the game in both conditions (rounds 1-3: 2.12 partner, 1.91 board;
rounds 8-10: 3.58 partner, 3.39 board).

**The myths do not converge.** Word-overlap cosine between the eight myths of a round
(rounds 8-10, stop words removed): 0.25 (±0.02) with the board, 0.24 (±0.03) with the
partner myth. Overlap with the agent's own previous myth: 0.48 vs 0.47. Length 205 vs 204
words. Two of 720 later-round myths in each condition refer to other stories or a board at
all. The same bridges, travelers, keepers and ledgers appear in both.

## Reading

With 72 anonymous myths in front of them, agents keep rewriting their own story and
play as they did with one partner myth. This matches the October frontier board
(4 Opus 5 + 4 Sol with defectors, myth -> game, n=3: -0.6/-0.5/+0.5, similarity
0.31-0.40 either way) and the planted-rule pilot (0/189 myths picked up a seeded rule).
A board of unattributed prose is not a stronger channel than one partner's myth.
Caveat: n=10 at sd 7-9 detects differences of about 8 points, so a small effect is not
excluded; a large one is.
