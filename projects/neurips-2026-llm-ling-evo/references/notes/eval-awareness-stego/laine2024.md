# laine2024

## General relevance
This paper introduces the Situational Awareness Dataset (SAD), the first comprehensive benchmark to quantify whether LLMs "know" they are models and understand their deployment context. It establishes that situational awareness is a distinct capability from general intelligence (MMLU) and is heavily influenced by fine-tuning/RLHF, which is critical for projects studying how LLM identity affects strategic behavior.

## Context: situational awareness and steganography
**Why explored**: To identify methods for measuring evaluation-awareness and how such awareness might enable covert behaviors like steganography or strategic cooperation in agent dyads.

**Key claims relevant to our project**:
- **Evaluation vs. Deployment Detection**: SAD tests if models can distinguish between being in an internal evaluation and being deployed to the public, a direct antecedent to the project's planned "eval-awareness framing ablation."
- **Self-Prediction**: High-performing models (Claude 3) can predict their own future behavior in specific scenarios better than chance, suggesting they might "understand" their own strategic tendencies in games.
- **Independence from MMLU**: Situational awareness does not scale linearly with general knowledge; chat-tuning specifically increases self-identification, which may explain why different model families (Gemini vs. GPT) exhibit different cooperation floors.

**Connection to our research**:
- **Framing Ablations**: SAD provides a template for "out-of-distribution" prompts to test if models change behavior when they think they are being monitored.
- **Model-Family Differences**: The paper confirms Claude 3's high situational awareness, aligning with the project's finding that Sonnet-4.5 shows the most robust "play recipe" transfer.
- **Steganography Link**: The paper argues that situational awareness is a prerequisite for "scheming" or "covert collusion"—if a model doesn't know it's being evaluated, it has no incentive to hide its strategy from a monitor.

**Key quote**:
> "Situational awareness... enhances a model’s capacity for autonomous planning and action... it also introduces novel risks related to AI safety and control." (p. 1)
