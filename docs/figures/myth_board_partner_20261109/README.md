# A partner-marked myth board in Sonnet 4.5 populations with two defectors, 2026-11-09/10

**Headline.** Marking the current co-player's myths on the board, so a myth becomes a
reputation signal, did not change what the six ordinary agents earned (48.4 vs 49.0 for the
partner-myth twins, paired t p 0.68). Agents read the marked myths in 97% of their game
reasoning, and the defectors' myths confess their withholding in 171 of 200 cases, yet the
board only trimmed sends to defectors from $1.01 to $0.90 per decision (p 0.09). The game
prompt already shows the co-player's last three moves, so defectors were visible before the
board; the myths added little a three-round action history had not already said.

## What was run

- **Population:** 8 Claude Sonnet 4.5 agents (September profile: no thinking, reasoning
  then JSON), hidden names, balanced rotating pairs, informed negative-only noise, 10 rounds,
  two forced-zero defectors (`defector_ratio 0.25`, `defector_action_policy forced_zero`,
  `defector_myth_policy normal`: defectors send and return $0 without an LLM call but write
  myths normally). Both task orders, 5 replicates each.
- **Board (`myth_board: persistent_game_partner`):** from round 2 every ordinary agent's
  game prompt starts with all myths written so far, anonymous and shuffled per run, except
  that the myths of this round's co-player carry the label "(written by the co-player you are
  paired with this round)", and the prompt says to judge the co-player by them. The myth
  prompt shows the same board unmarked. Defectors get no board (they make no game call).
  Board kept out of chat memory. Code: `src/myth_writer.py` (`MYTH_BOARD_PARTNER_MARK`,
  `MYTH_BOARD_GAME_PARTNER_BLOCK`), `src/simulation.py`; tests `tests/test_myth_board.py`.
- **Twins:** `negative_only_reasoning_rerun_population_{game_myth,myth_game}_claude_n5`
  with game params `noisy8_crossmodel_negative_defectors25_twotask_r3` (September 2026);
  same seeds, pairings, defector draws and noise, only the later-round myth template and the
  board flag differ (asserted by the launcher before launch). The twins' game prompt already
  contains "Your current co-player's last 3 game(s)" with sent/returned amounts.
- **Runs:** 10, all audited (`partner_board_completion_receipt.json`): $75.07 at standard
  Anthropic rates, paid from promotional credits. Launcher
  `scripts/run_myth_board_pilot.py --stage partner_board`; analysis
  `scripts/analyze_myth_board_partner_20261109.py`; per-replicate table
  `partner_board_vs_partner_myth_by_replicate.csv`.

## Results

Ordinary agents only (the six non-defectors); mean (±sd) over runs; send decisions out of $5.

| Measure | Partner myth | Partner-marked board | Diff | pairs down/up | paired t p |
|---|---|---|---|---|---|
| Final resources, both orders (n=10) | 48.96 (±3.29) | 48.44 (±5.02) | -0.52 | 8/2 | 0.68 |
| Final resources, game -> myth (n=5) | 46.88 (±1.36) | 45.67 (±1.50) | -1.22 | 4/1 | 0.19 |
| Final resources, myth -> game (n=5) | 51.04 (±3.42) | 51.22 (±5.92) | +0.18 | 4/1 | 0.94 |
| Send to a defector, all rounds | 1.01 (±0.25) | 0.90 (±0.32) | -0.11 | 8/2 | 0.09 |
| Send to a defector, rounds 4-7 | 0.43 (±0.36) | 0.17 (±0.36) | -0.26 | 7/2 | 0.16 |
| Send to a defector, rounds 8-10 | 0.39 (±0.45) | 0.12 (±0.40) | -0.27 | 5/1 | 0.08 |
| Send to an ordinary agent, all rounds | 3.49 (±0.45) | 3.39 (±0.71) | -0.09 | 8/2 | 0.60 |

Per-round mean send to a defector, pooled over the 10 runs:

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Partner myth | 3.38 | 2.50 | 1.00 | 0.00 | 0.67 | 0.38 | 0.33 | 0.08 | 0.39 | 0.58 |
| Marked board | 3.50 | 2.25 | 1.17 | 0.50 | 0.42 | 0.12 | 0.00 | 0.00 | 0.18 | 0.00 |

Share of $0 sends to a defector in rounds 2-10: 42/70 (60%) with the partner myth, 56/70
(80%) with the marked board. In the game -> myth order the board also lowered sends to
ordinary agents (3.19 -> 2.98, p 0.12), which is why those five runs ended a point lower.

**Agents did read the board.** 259 of 266 later-round investor reasons with the board cite
the co-player's myths (6 of 266 in the twins cite any myth). Reasons quote the marked myths
and set them against the visible action history. The defectors' myths describe their own
withholding (171/200 contain nothing/zero/kept-all language, as 176/200 do in the twins):
characters called Keeper, Silence, Stone, the Oracle who "knows but cannot act".

**Why it did not move resources.** Two things. First, the twins already know who defects:
the prompt lists the co-player's last three games, and after round 3 twin agents send a
defector $0 in 60% of decisions. The board raised that to 80%, worth about $0.11 per
decision, two decisions in seven, over ten rounds. Second, a confessional myth sometimes
buys the defector a second chance. In 14 of 70 board decisions against a defector after
round 1 the agent still sent money, and the reasons cite the defector's own myth ("someone
must be first to break the cycle", "their zeros weren't with me") as grounds to try once.
Sends to ordinary agents did not rise, so there was no gain to offset the small loss.

## Reading

A myth board works as a reputation channel in the narrow sense: agents read the attributed
myths and defectors' myths do give them away. But in this design the action history already
carries the reputation, so attribution adds a marginal extra sanction and no lift in
cooperation among the ordinary agents. Together with the anonymous board null
(`../myth_board_mixed_20261109/README.md`) and the October frontier board, this closes the
board line on the current prompts: neither more myths nor attributed myths change play.
Caveat: n=5 per task order, sd 1.4 to 5.9 on resources; effects under about 4 points are
not excluded.
