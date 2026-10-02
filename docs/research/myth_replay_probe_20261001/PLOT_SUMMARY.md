# What the replay ablation plot shows

![Replay ablation](replay_ablation_style.png)

- **Test:** we replayed real decisions from the September runs, changing only one sentence in a
  myth: "whoever holds five should send $X" ($1, $2, $3 or $5).
- **The agent's own myth is a plan it follows:** with the rule in its own first myth, agents send
  almost exactly the stated amount. GPT sent $1.20, $2.00, $3.00 and $5.00; Sonnet sent $1.89,
  $2.11, $3.00 and $5.00.
- **A later myth works too, more loosely:** about $0.68 per $1 stated, against $0.93 for the
  first myth.
- **A partner's myth barely moves sends:** about $0.23 per $1. Mostly only "send $5" lifts them.
- **Caveats:** Gemini sends $5 regardless. An amount told inside the story, rather than as a
  rule, has a weaker effect. These are single decisions, not whole runs.

Details: `README.md` in this folder.
