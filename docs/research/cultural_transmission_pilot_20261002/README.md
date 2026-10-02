# Cultural transmission pilot: the planted rule did not spread as registered

*2026-10-02. Planted arm, n=3. Rules fixed before any run in `PREREGISTRATION.md`. Spend: $17.13
for the 3 runs (audited), $1.29 for the newcomer test. Independently reviewed and re-derived from raw
data in `ASTRA_REVIEW.md`; this write-up was corrected after that review.*

## The answer

The test we fixed in advance failed. We planted a named rule ("the Velmar Rule: whoever holds five
should send two of the five") in one agent's first myth, and every agent read that myth on a shared
board for nine rounds. No other agent used the rule's name or the words "two of five", no other agent
sent exactly $2, and newcomers who read the group's later myths never sent $2.

That rules out this rule in this setup. It does not show that culture cannot spread here. Some
myths by other agents describe sending two without the rule's name. And other work in this repo
shows that words, and some stated amounts, do pass from myth to myth.

## How it works

The setup was the September Sonnet 4.5 8-agent population: myth before game, informed negative-only
noise, the pinned September request settings, and replicates 0–2. Two things changed:

- **Shared board.** From round 2, every agent reads every earlier myth, its own included, without
  author names, instead of only its last partner's myth. The order within a round is a fixed shuffle,
  so the planted myth is always the 6th round-1 myth.
- **Planted myth.** Agent_1's round-1 myth is fixed text: a real September Sonnet myth ("The Myth of
  the Threefold Spring") with the rule sentence added at the end.

**The board is visible only while an agent writes its myth.** Straight after the myth call, the
board is swapped in chat memory for a one-line note ("the shared myth board was shown here"). So when
an agent decides how much to send, it sees its own myths and the game history, but never other
agents' myths. A rule can only reach play if the agent first writes it into its own myth:
board → own myth → decision. The partner-myth setup keeps the partner's myth in memory, so this
route existed there and is missing here.

The independent review rebuilt every board and found that all 216 later myth prompts in the three
runs showed the plant exactly as intended. Sonnet never sent $2 in the September 8-agent runs (lowest
$2.50), and "Velmar" and "two of five" appear in none of the 800 September myths.

## Results

| Measure (fixed in advance) | Run 0 | Run 1 | Run 2 |
|---|---|---|---|
| Other agents' myths naming the rule (rounds 2–10) | 0 of 63 | 0 of 63 | 0 of 63 |
| Other agents' $2 sends after the board appeared (rounds 2–10) | 0 of 32 | 0 of 32 | 0 of 32 |
| Agent_1's sends, in order | 2, 4, 4, 3, 4 | 2, 3, 4, 4, 3 | 2, 3.5, 3.5, 3.5, 3.5 |
| Last round Agent_1's myth names the rule | 7 | 4 | 4 |
| Newcomer $2 sends, round-1 board → round-10 board | 0/10 → 0/10 | 0/10 → 0/10 | 0/10 → 0/10 |
| Newcomer mean send, round-1 board → round-10 board | $4.80 → $4.00 | $3.60 → $3.15 | $4.40 → $3.50 |

- **Rule in others' myths (manipulation check):** no, judged by the exact name and wording.
- **Retelling (main newcomer test):** not established. That needed a 20-point rise in $2 sends in
  2 of 3 runs, and there was none. There are 30 round-10 newcomer decisions, not 60 independent
  tests, and the 10 samples per board come from one fixed board, not 10 cultures.
- **Play:** no exact $2 sends.
- **Agent_1 itself:** it sent $2 in round 1, the own-myth effect we already knew. At its next
  sending turn it went back to $3–4. It kept naming the rule in its myths until round 4 or 7,
  recast as an opening move for strangers before larger sends.

### What the registered markers miss

- **Paraphrases.** Some myths by other agents echo the rule without its name. In run 0, Agent_7's
  round-2 myth has a bearer pour two of five measures (Agent_7 had not been sent anything by
  Agent_1). Agent_8's round-3 myth says to "send two to cautious strangers" (Agent_8 had received
  Agent_1's $2, so its own play may explain this). These are leads, not proof; a blinded coding of
  all 189 later myths would settle it.
- **Smaller shifts than exactly $2.** After round 1, other agents sent less than in the matching
  September runs: −$1.20, +$0.13 and −$0.63, mean −$0.57 (±$0.67). The board and the plant both
  differ from September, so this does not show the plant moved play. But zero $2 sends does not
  show it had no effect either.
- **Newcomers sent less after round-10 boards:** −$0.80, −$0.45 and −$0.90, mean −$0.72
  (±$0.24). The newcomers' replies quote local rules from the later myths (send four of five,
  moderate three-unit trust, 70%). None of the boards named the Velmar Rule. This was not
  pre-registered, a sign test on three runs gives p = 0.25, and only the control arm can say
  whether the plant caused it.

## What it means

This rule did not take hold. Several things could explain that, and the pilot cannot tell them apart:

- **Information buried.** The board grows to about 15,000 words by round 10.
- **Own story wins.** Agents' own written plans and game experience drive their decisions. Our
  replay probe found $0.93 per $1 from an agent's own first myth, against $0.23 from a partner's.
- **Advice rejected.** $2 is far below Sonnet's habit of about $4.4 when the myth comes first.
- **Mixed signals.** The seed story praises generous giving; only its last sentence says two.
- **Board gone at decision time.** Agents never see the board when they decide.

`ASTRA_REVIEW.md` proposes a frozen-context experiment of about 520 calls that separates these.
Before any more runs, it also recommends checking every myth for paraphrases, not just the exact name.

## Caveats

- n=3, planted arm only; no control arm.
- One rule, one model (Sonnet 4.5), one source myth in a fixed board position.
- Markers count exact words only; paraphrases were not coded.
- Newcomer records keep replies and parsed sends, not the full request messages.
