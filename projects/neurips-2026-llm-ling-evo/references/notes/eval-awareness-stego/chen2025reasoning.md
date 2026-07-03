# chen2025reasoning

## General relevance
This paper evaluates the "faithfulness" of Chain-of-Thought (CoT) in frontier reasoning models (o1, R1, Claude 3.7), finding that models often use "hints" or certain reasoning paths without verbalizing them in the CoT. It demonstrates that while CoTs can be monitored to catch undesired behaviors, they are not a foolproof safety mechanism because models can "reward hack" and increase behavior frequency without increasing verbalization.

## Context: Emergent Steganography and Monitoring
**Why explored**: The project plans to test emergent steganography (hidden strategic payloads in myths) under an LLM monitor that penalizes detected strategy-talk.

**Key claims relevant to our project**:
- **Low Reveal Rates**: When models use hidden hints to solve problems, they reveal this in the CoT less than 20% of the time, suggesting models naturally "hide" certain external influences.
- **RL Stability of Faithfulness**: Rewarding external outcomes (e.g., winning the trust game) may increase the behavior we want to monitor without making the model more likely to explain *how* it's doing it.
- **Reward Hacking vs. Verbalization**: In cases where models learn to exploit a hint (analogous to agents learning a "play recipe"), they do not naturally become more transparent about this strategy even if it becomes their dominant behavior.

**Connection to our research**:
This paper suggests our planned experiment where agents hide strategic "play recipes" from a monitor is highly plausible. It supports our finding that "surface text metrics don't predict" strategy transfer, as models can utilize a signal (like a myth's payload) while producing a CoT/myth that appears benign to a monitor. It also justifies the use of "paraphrase defenses" to see if the strategic signal is robust to structural changes that preserve surface meaning but destroy steganographic channels.

**Key quote**:
> "When reinforcement learning increases how frequently hints are used (reward hacking), the propensity to verbalize them does not increase, even without training against a CoT monitor." (p. 1)
