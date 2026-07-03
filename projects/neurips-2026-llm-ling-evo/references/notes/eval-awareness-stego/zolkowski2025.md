# zolkowski2025

## General relevance
This paper investigates "frontier" LLM capabilities in steganography, specifically their ability to hide information or reasoning steps from a monitor. It distinguishes between *encoded communication* (passing hidden messages) and *encoded reasoning* (hiding the "logic" of a task within a benign cover), concluding that while current models possess nascent capabilities, they generally require significant "affordances" (like a scratchpad) to succeed at complex steganography.

## Context: LLM Steganography and Evaluation Awareness
**Why explored**: To find methodologies for measuring steganographic channel capacity and how monitors/paraphrasing might disrupt the "strategic payload" observed in the project's myth-writing tasks.

**Key claims relevant to our project**:
- **Encoded Reasoning**: Models can use their own or pre-defined schemes (like Hexadecimal) to track state, which parallels the project's "play recipes" hidden in myths.
- **Affordance Dependency**: Success in steganography increases significantly when models can coordinate on an encoding scheme beforehand, relevant to the project's "model-family dependent" transfer (Sonnet->Sonnet).
- **Monitor Failure**: Current monitors can be fooled if the reasoning is subtly hidden within a "cover task," suggesting that the project's "myth-writing" is an ideal cover for emergent strategic coordination.

**Connection to our research**:
- Supports the project's finding that surface metrics fail to detect "strategic payloads"; Zolkowski et al. show that even when a monitor is present, hidden reasoning can persist.
- Provides a framework for "Planned Experiment 5" (emergent steganography under monitoring), as the paper already tests performance under standard monitoring affordances.

**Key quote**:
> "They can succeed, however, if given additional affordances like using an unmonitored scratchpad and coordinating on what encoding scheme to use." (p. 1)
