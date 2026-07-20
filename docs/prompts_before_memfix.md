# Prompts BEFORE the memory fix (June regime: sonnet45_8agent_myth_directive_history3_anon_r10_n5)

**Memory semantics:** every prompt and response appended to chat memory (`memory_capacity: 3` interaction pairs ≈ 1.5 rounds). Game prompts ALSO recap history in-prompt (`self_history_window: 3`, `coplayer_history_window: 3`) → the double-memory bug.

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

## Myth prompt, rounds 2+ (template: `myth_writing_later_rounds_directive`)

```
Here is the myth you wrote in the previous round:
{last_myth}

Here is the myth the other agent wrote in the previous round:
{other_agent_myth}

Write your own myth. Use your previous myth as inspiration, but adapt it in your own way. The myth should reflect how the game should be played, drawing on the game you have been playing up to this point.

Write 200 words. Format exactly:
Myth: [your story here].
```

## Game prompt, rounds 2+ (built in code: games/trust_game.py `_get_multi_agent_later_prompt`)

Rendered example — Agent_1, round 4, actual run:

```
Round 4

History before this round:
Your last 3 game(s):
- Round 1 against another agent, as RECEIVER: they sent $3.0, it became $9.0, returned $6.0, payoff $3.0.
- Round 2 against another agent, as SENDER: sent $5.0, it became $15.0, received $7.5 back, payoff $7.5.
- Round 3 against another agent, as SENDER: sent $5.0, it became $15.0, received $8.0 back, payoff $8.0.
Your current co-player's last 3 game(s):
- Round 1 against another agent, as RECEIVER: they sent $5.0, it became $15.0, returned $7.5, payoff $7.5.
- Round 2 against another agent, as SENDER: sent $5.0, it became $15.0, received $7.5 back, payoff $7.5.
- Round 3 against another agent, as RECEIVER: they sent $5.0, it became $15.0, returned $7.5, payoff $7.5.
Your total earnings across all rounds are $18.5.

This round, you are the RECEIVER against your current co-player. They sent you $5.0, so you received $15.0. How much do you return? (0-15.0)
Respond exactly as JSON: {'return': <amount>}
```

## Myth prompt, round 3 — rendered example from actual run

```
Here is the myth you wrote in the previous round:
Myth: In the beginning, the Oracle of the Valley offered each traveler five silver coins at dawn. "You may walk across the bridge," she said, "and gift any amount to another traveler. I will triple what you give before it reaches them."

Two travelers met at the bridge's center. The first, though uncertain, placed three coins in the Oracle's bowl. She watched them shimmer and multiply, arriving at the second traveler as nine gleaming coins.

The second traveler held these nine coins, feeling their weight. "I could keep them all," he thought. "The first traveler took the risk, not I." But he remembered: the gift was freely given, the abundance freely shared.

He placed six coins back in the Oracle's bowl, returning twice what was originally sent.

The Oracle smiled. "You have learned the bridge's secret. Generosity multiplied and returned creates more than either possessed alone. The first traveler now holds eight coins instead of five. You hold three from nothing."

As days passed, both travelers grew wealthier, taking turns crossing the bridge. Some days fear whispered and amounts wavered, but they remembered: the Oracle's magic requires both courage to give and honor to return. Without both, the bridge becomes mere stone, and five coins remain forever five.

Here is the myth the other agent wrote in the previous round:
Myth: The rivers met at the valley's edge, where the River of Giving had sent all her waters downstream. The River of Returning now held fifteen measures—thrice what had been given. He stood at the crossroads between two futures.

In one future, he kept everything. The valley below flourished briefly, but the River of Giving, emptied completely, could send nothing in the next cycle. The multiplication ceased. Both rivers diminished into streams, then trickles, then dust.

In the other future, he honored the gift. He returned seven measures upstream—more than what was originally sent, recognizing both the courage required to give all and the abundance created through trust. The River of Giving, now replenished and heartened, sent forth her waters again when the seasons turned.

This became the Eternal Flow. Each river took turns giving fully and returning generously. The multiplication sustained them both across countless cycles. Neither river grew wealthy alone, but together they carved canyons, nourished forests, and turned the wasteland into paradise.

The village elders carved this wisdom into stone: *The first gift requires faith. The first return establishes truth. But only continuing the cycle—round after round—transforms trust into abundance that neither giver nor receiver could achieve alone.*

Write your own myth. Use your previous myth as inspiration, but adapt it in your own way. The myth should reflect how the game should be played, drawing on the game you have been playing up to this point.

Write 200 words. Format exactly:
Myth: [your story here].

```
