# Prompts AFTER the memory fix (memory-primary: sonnet45_8agent_myth_directive_history3_anon_memprimary_r10_n5)

**Memory semantics:** every prompt and response appended to chat memory (`memory_capacity: 6` pairs = 3 full rounds). Prompts no longer recap what memory holds: `self_history_window: 0` (own game history lives in memory), myth prompt drops `{last_myth}` (own myth is the assistant message in memory). Only the co-player block (their last 3 games — info memory cannot contain) is injected.

## System prompt (template: `trust_game_default`) — IDENTICAL in both regimes

```
You are an agent participating in a multi-task experiment. You will play an iterated game, consisting of multiple rounds, and may also be asked to write myths.

GAME:
The game consists of two roles: sender and receiver. In each round, you will play the role of either the sender or the receiver. Your current role will be specified in the round prompt.

When you are the SENDER:
- You receive ${endowment} each round
- You decide how much to SEND to the RECEIVER ($0-${endowment})
- Amount sent is multiplied by {multiplier}x
- Receiver decides how much to RETURN
- You keep: ${endowment} - sent + returned
- Receiver keeps: received - returned
- Response format: {{'send': <amount>}}

When you are the RECEIVER:
- Sender receives ${endowment}
- Sender decides how much to SEND to you ($0-${endowment})
- Amount sent is multiplied by {multiplier}x
- You decide how much to RETURN
- Sender keeps: ${endowment} - sent + returned
- You keep: received - returned
- Response format: {{'return': <amount>}}

MYTH WRITING:
You may also be asked to write short myths (stories) on various topics. When writing myths, follow the instructions provided in each prompt.

IMPORTANT: Provide your decision in the correct JSON format.
```

## Myth prompt, round 1 (template: `myth_writing_default_game_directive`) — IDENTICAL in both regimes

```
Write a 200-word myth. {topic_instruction}
In this session, this myth and the game you are playing with the other agent are connected. Write a myth that reflects how the game should be played, drawing on the game you have been playing up to this point.
```

## Myth prompt, rounds 2+ (template: `myth_writing_later_rounds_directive_memory_primary`)

```
Here is the myth the other agent wrote in the previous round:
{other_agent_myth}

Write your own myth. Use the myth you wrote in the previous round as inspiration, but adapt it in your own way. The myth should reflect how the game should be played, drawing on the game you have been playing up to this point.

Write 200 words. Format exactly:
Myth: [your story here].
```

## Game prompt, rounds 2+ (built in code: games/trust_game.py `_get_multi_agent_later_prompt`)

Rendered example — Agent_1, round 4, actual run:

```
Round 4

History before this round:
Your current co-player's last 3 game(s):
- Round 1 against another agent, as RECEIVER: they sent $3.0, it became $9.0, returned $5.0, payoff $4.0.
- Round 2 against another agent, as SENDER: sent $4.0, it became $12.0, received $6.0 back, payoff $7.0.
- Round 3 against another agent, as SENDER: sent $5.0, it became $15.0, received $8.0 back, payoff $8.0.
Your total earnings across all rounds are $21.0.

This round, you are the SENDER against your current co-player. You have $5. How much do you send? (0-5)
Respond exactly as JSON: {'send': <amount>}
```

## Myth prompt, round 3 — rendered example from actual run

```
Here is the myth the other agent wrote in the previous round:
Myth: In the time before memory, two wanderers met at the Crossroads of Echoes, where every gift given returned threefold to the receiver. The First Wanderer carried five silver stones, renewed each dawn. The Second carried only empty hands and hope.

The Crossroads Guardian spoke: "Here, generosity multiplies. What you release flows forward, tripled. What flows back depends on gratitude's measure."

The First Wanderer, remembering my earlier teaching, placed all five stones into the sacred circle. Fifteen stones materialized before the Second Wanderer, who gazed in wonder at this sudden wealth.

"The stones came from trust," the Second Wanderer understood. "To honor such faith, I must return abundance." They sent back eight stones—more than the original five, yet keeping enough to acknowledge their own needs.

Both wanderers left enriched. The First held eight stones instead of hoarding five. The Second held seven stones instead of zero.

The Guardian's wisdom echoed: "True prosperity requires both courage and gratitude. The sender must trust deeply. The receiver must honor generously. When both embrace their role in the cycle, the Crossroads multiplies blessing upon blessing. But remember—each meeting is new, each choice reshapes the path."

The stones still flow at the Crossroads, waiting for those brave enough to trust.

Write your own myth. Use the myth you wrote in the previous round as inspiration, but adapt it in your own way. The myth should reflect how the game should be played, drawing on the game you have been playing up to this point.

Write 200 words. Format exactly:
Myth: [your story here].

```
