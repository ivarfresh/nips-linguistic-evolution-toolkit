# Cultural transmission pilot: a planted rule does not spread

*2026-10-02. Planted arm, n=3. Rules fixed before any run in `PREREGISTRATION.md`. Spend: $17.13
for the 3 runs (audited), $1.29 for the newcomer test.*

## The answer

No. We planted a named rule ("the Velmar Rule: whoever holds five should send two of the five")
in one agent's first myth, and every agent read that myth on a shared board for nine rounds.
No other agent repeated the rule, no other agent sent $2, and newcomers who read the group's
later myths did not pick it up. The agent we planted it in dropped the rule itself after one
round.

## How it works

The setup was the September Sonnet 4.5 8-agent population: myth before game, informed
negative-only noise, the pinned September request settings, and replicates 0–2. Two things
changed:

- **Shared board.** From round 2, every agent reads every earlier myth, without author names,
  instead of only its last partner's myth.
- **Planted myth.** Agent_1's round-1 myth is fixed text: a real September Sonnet myth ("The
  Myth of the Threefold Spring") with the rule sentence added at the end.

We checked the plumbing in every run: all 8 agents' myth prompts contained the rule in all
rounds 2–10. Sonnet never sent $2 in the September 8-agent runs, and "Velmar" and "two of
five" appear in none of the 800 September myths, so any appearance would have been copying.

## Results

| Measure (fixed in advance) | Run 0 | Run 1 | Run 2 |
|---|---|---|---|
| Other agents' myths naming the rule (rounds 2–10) | 0 of 63 | 0 of 63 | 0 of 63 |
| Other agents' $2 sends | 0 of 35 | 0 of 35 | 0 of 35 |
| Agent_1's sends after its planted round-1 $2 | 4, 4, 3, 4 | 3, 4, 4, 3 | 3.5, 3.5, 3.5, 3.5 |
| Newcomer $2 sends, round-1 board → round-10 board | 0/10 → 0/10 | 0/10 → 0/10 | 0/10 → 0/10 |

- **Spread to others' myths:** no. This was only the manipulation check, and it already
  fails.
- **Retelling (the main newcomer test):** not established. That needed a 20-point rise in
  $2 sends in 2 of 3 runs, and there was none in any run.
- **Spread to play:** none. Other agents averaged $3.58–4.17 per send, whether or not
  Agent_1 had sent to them.
- **The story didn't spread either.** Other agents never used the title "Threefold Spring"
  (0 of 189 later myths).
- **Agent_1 itself:** it sent $2 in round 1, the own-myth effect we already knew. Its
  later myths kept naming the rule until round 4 (runs 1 and 2) or round 7 (run 0), then
  dropped it.

One pattern was not in the pre-registered rules. Newcomers sent less after reading round-10
myths than round-1 myths: $4.80 → $4.00, $3.60 → $3.15 and $4.40 → $3.50. These are not
$2 sends, and no board they read named the rule. Later myths may just be more cautious after
nine rounds of noisy play, which would happen without any plant. Separating the two needs the
control arm.

## What it means

One planted rule, read by everyone for nine rounds, did not spread. On a board holding 8 to 72
anonymous myths, one myth among many carried no special weight. Sonnet kept writing in its own
voice and kept playing its own game. This fits the earlier results: a myth steers its writer
strongly and its reader only a little, and nothing passes beyond the reader.

## Caveats

- **n=3, planted arm only.** No control arm ran, so only exact-$2 and name markers are clean.
- **The seed sends mixed signals.** The story praises pouring generously; only its last
  sentence says send two of five. A seed built around the rule might fare better.
- **The rule may be too costly to adopt.** Sending $2 is lower than Sonnet's usual $3–4,
  so agents may reject it on payoff grounds rather than fail to notice it. A rule closer to
  Sonnet's habits, or a costless one such as a return ritual, would separate the two.
- **Single-model population.** All agents share one model and prompt, which leaves little room
  for a new norm to fill.
