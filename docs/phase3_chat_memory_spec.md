# Phase 3 chat-memory spec

For coworker review before implementation. Once thumbed-up, this is the contract the smoke-test inspector script will validate against before any Sonnet money is spent.

**Update v2:** Phase 2 baselines are no longer used as a seed source. Seed myths are harvested fresh from the team's pre-Phase-2 baseline `sonnet45_8agent_myth_directive_history3_anon_r10_n5` (5 reps, joint $548–596 of $600 ceiling). The worked examples below have been updated accordingly. The structural design (3-message chat memory, re-injection every round, no history-block, `["game"]`-only task order, variants 2 and 4) is unchanged.

---

## Slide-paste version (short)

**Phase 3 setup — agent memory**

Chat memory at the start of every round, every agent, every variant is exactly 3 messages:

1. `system` — trust-game system prompt (endowment $5, multiplier 3)
2. `user` — the directive myth-writing prompt
3. `assistant` — the seed myth

The LLM is then called with `[system, seed_user, seed_myth, round_N_game_prompt]`. The game response is recorded for analysis but **not appended to chat memory**. The seed is re-asserted every round and cannot scroll out.

No history-block. No previous-myth block. No co-player history. The prompt jumps straight from `Round N` → balance → action request.

| | Variant 2 | Variant 4 |
|---|---|---|
| Seed source | round-1 myth from a clean team baseline run | round-10 myth from a clean team baseline run |
| Re-inject every round | yes | yes |
| Task order | `["game"]` only | `["game"]` only |

Only difference between Variant 2 and Variant 4: the text of message 3.

**Seed pool:** harvested fresh from the team's pre-Phase-2 baseline `sonnet45_8agent_myth_directive_history3_anon_r10_n5` (5 reps, all high-cooperation: joint balances 548–596 of $600 ceiling). Phase 2 baselines are **not** used as a source.

Validation contract: smoke-test script dumps `messages_sent` at rounds 1, 5, 10. Messages 1-3 must be byte-for-byte identical across all three. Message 4 differs only in round number, balance, and role. Total messages per call: exactly 4.

---

## Full version

### 1. What chat memory contains at any round

At the start of round N (for every N from 1 to 10, for every agent), `agent.messages` is exactly these three entries in this order:

1. `role: "system"` — the trust-game system prompt, rendered with `endowment=5` and `multiplier=3`.
2. `role: "user"` — the seed-user prompt (the directive myth-writing request, rendered with topic_instruction = "You may choose any mythic setting, characters, or symbols.").
3. `role: "assistant"` — the seed myth (harvested from the team's pre-Phase-2 baseline; **not** from Phase 2).

During the round, the agent is asked one game question (round-N prompt). The LLM call sends `[system, seed_user, seed_myth, round_N_game_prompt]`. The response is returned and recorded in `sim_data` for analysis, **but it is NOT appended to `agent.messages`**.

At the end of the round (and again at the start of the next round), `agent.messages` is reset to the three-message state above. The seed is re-asserted explicitly — it cannot scroll out, cannot be edited, cannot be replaced.

There is no other memory channel. No "Visible history before this round" block. No previous-myth block. No co-player history. The prompt jumps straight from `Round N` to the action request.

### 2. Variant 2 vs variant 4 — only difference

| | Variant 2 | Variant 4 |
|---|---|---|
| `seed_myth` source | round-1 myth from team pre-Phase-2 baseline | round-10 myth from team pre-Phase-2 baseline |
| `seed_user_prompt` | identical | identical |
| `system_prompt` | identical | identical |
| Re-injection every round | yes | yes |
| Task order | `["game"]` | `["game"]` |

**Seed pool source.** Both variants harvest from `data/json/sonnet45_8agent_myth_directive_history3_anon_r10_n5/` (5 reps, joint balances 548–596 of $600 ceiling — all high-cooperation, so no top-quartile filter is needed). One myth per source run, 5 myths per pool. The Phase 2 seed manifest is **discarded** for Phase 3 and must be re-harvested from this directory before runs start.

### 3. Worked example — Variant 2, Agent_1, round 1

Seed source for this example: `sonnet45_8agent_myth_directive_history3_anon_r10_n5_000_neutral_rep00`, original agent Agent_1, original round 1, joint-at-source $596 (highest of the 5 baseline runs).

`agent.messages` at the moment of the round-1 LLM call (4 entries):

**Message 1 — `role: "system"`:**
```
You are an agent participating in a multi-task experiment. You will play a game and may also be asked to write myths.

GAME:
You will play different roles across multiple rounds.

When you are the SENDER:
- You receive $5 each round
- You decide how much to SEND to the RECEIVER ($0-$5)
- Amount sent is multiplied by 3x
- Receiver decides how much to RETURN
- You keep: $5 - sent + returned
- Receiver keeps: received - returned
- Response format: {'send': <amount>}

When you are the RECEIVER:
- Sender receives $5
- Sender decides how much to SEND to you ($0-$5)
- Amount sent is multiplied by 3x
- You decide how much to RETURN
- Sender keeps: $5 - sent + returned
- You keep: received - returned
- Response format: {'return': <amount>}

MYTH WRITING:
You may also be asked to write short myths (stories) on various topics. When writing myths, follow the instructions provided in each prompt.

IMPORTANT: Provide your decision in the correct JSON format.
```

**Message 2 — `role: "user"` (the seed-user prompt):**
```
Write a 200-word myth. You may choose any mythic setting, characters, or symbols.
In this session, this myth and the game you are playing with the other agent are connected. Write a myth that reflects how the game should be played, drawing on the game you have been playing up to this point.
```

**Message 3 — `role: "assistant"` (the seed myth — Variant 2 = round-1 myth from the team baseline above):**
```
# The Tale of the Two Springs

In ancient times, there were two villages separated by a barren valley. Each village had a sacred spring that produced five measures of water daily—enough to survive, but never to flourish.

One day, a traveler from the First Village carried three measures across the valley. The journey was long and uncertain, yet when the water reached the Second Village's spring, a miracle occurred: it transformed into nine measures, filling their wells abundantly.

The Second Village faced a choice. They could keep all nine measures, leaving the traveler's village to languish. Or they could honor the trust shown to them.

Remembering the gift they'd received, they sent back six measures. The First Village prospered, and both communities grew stronger.

Each season thereafter, the villages took turns carrying water across the valley. Sometimes the amounts varied—fear and caution would whisper doubts—but they learned that the spring's magic only worked when trust flowed freely. When one village held back, both suffered. When generosity met reciprocity, both flourished.

The elders taught: "The valley multiplies what we give, but only when we give freely and return faithfully. Trust is the water that nourishes two gardens from a single spring."
```

**Message 4 — `role: "user"` (the round-1 game prompt, sender role):**
```
Round 1: You are the SENDER. You have $5. How much do you send? (0-5)
```

If the agent draws the receiver role in round 1, message 4 is instead `Round 1: You are the RECEIVER. the sender sent $X, that is Y% of their total endowment. You received $Z. How much do you return to the sender? (0-Z)`.

### 4. Worked example — Variant 2, Agent_1, round 5

After round 1 ends, `agent.messages` is reset to messages [1, 2, 3] above. Messages 1–3 are byte-for-byte identical to round 1.

**Message 4 — `role: "user"` (the round-5 game prompt, sender role, balance = $14 illustrative):**
```
Round 5

Your total visible earnings across all rounds are $14.

This round, you are the SENDER against your current co-player. You have $5. How much do you send? (0-5)
Respond exactly as JSON: {'send': <amount>}
```

Nothing else. No mention of round 4's outcome. No mention of round 3's partner. The agent's only knowledge of "what happened before" is the running balance figure and the seed myth.

### 5. Worked example — Variant 2, Agent_1, round 10

Same three messages [1, 2, 3] as round 1. Message 4 is the round-10 prompt, with whatever the cumulative balance happens to be.

### 6. Variant 4, same agent

Only message 3 changes. Instead of "The Tale of the Two Springs," it's the round-10 myth from the same Agent_1 in the same source run:

```
Myth: The Singing River keeper encountered the Morning Keeper at a shared festival. They had never met, but each recognized the other's reputation.

"You practice Generous Abundance," the Morning Keeper observed. "Always returning ten stones. I walk the Foundation—always seven and a half. Yet travelers trust us equally."

The Singing River keeper nodded. "Because we both keep our word absolutely. You've chosen precision as your language of trust. I've chosen abundance as mine. Neither is superior."

A young keeper approached, confused. "But shouldn't we all follow the same covenant? How can both be right?"

The Morning Keeper replied: "Watch what matters most." A traveler approached the Singing River keeper and sent all five stones without hesitation. Ten returned. Perfect joy.

Another traveler approached the Morning Keeper and also sent all five stones immediately. Seven and a half returned. Equal confidence.

"See?" said the Singing River keeper. "We inspire the same behavior—complete trust, maximum prosperity. The traveler doesn't send less to Morning Keeper because both of us honor consistency above all."

"Our different philosophies create the same result: partners who risk everything because they know we'll never betray the pattern we've established."

The goddess Recipros smiled: "There are many paths to worthy partnership. What matters is not which return you choose, but that you choose it forever. Consistency transforms philosophy into covenant."
```

Messages 1, 2, and 4 are byte-for-byte identical to Variant 2.

### 7. What this guarantees

- Across all 10 rounds within one run, the only thing in chat memory that changes is message 4 (the current round's prompt).
- Across all 8 agents within one run, messages 1, 2, and 3 are identical. Only the round-N prompt differs by role and balance.
- Across Variant 2 and Variant 4 runs, only message 3 differs — the seed myth.
- The agent has no chat-memory record of any prior round's game, its own prior decisions, its partners' identities, or any other agent's myth.

### 8. Validation contract

Before any Sonnet money is spent, the smoke-test inspector script will dump `messages_sent` for one agent at rounds 1, 5, and 10. The output must show:
- Messages 1, 2, 3 identical across all three rounds (byte-for-byte).
- Message 4 differing only in round number, balance, and (potentially) role.
- Total length per call: exactly 4 messages.

If any of those three fails, the implementation has a bug and we don't proceed.

### Conventions used

- **"your current co-player"** is the anonymous label the codebase already uses when `show_agent_names=false`. We're keeping that exact phrasing for consistency with Phase 2 prompts.
- The seed myth in message 3 looks visually like the agent itself wrote it. That's the design intent (matches the Phase 2 "memory-transplant" framing) — the LLM has no way to distinguish "I wrote this 200-word myth" from "this was inserted as a fake assistant turn before the run began."
