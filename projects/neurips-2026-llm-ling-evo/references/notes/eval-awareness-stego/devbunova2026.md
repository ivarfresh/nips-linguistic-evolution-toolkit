# devbunova2026

## General relevance
This paper provides a critical methodological critique of how "evaluation awareness" (the model's ability to know it is being tested) is measured in LLMs. It argues that successful linear probes often detect surface-level formatting (like benchmark style) rather than a deep semantic understanding of being in an evaluation, which is vital for the project's aim to use probes and framing to study agent cooperation.

## Context: Evaluation-awareness & Steganography
**Why explored**: To find methodologies for measuring eval-awareness (specifically linear probes) and to understand if these measures track internal states or just prompt artifacts.

**Key claims relevant to our project**:
- **Probe Failure**: Linear probes trained to distinguish evaluation from deployment prompts fail to generalize when the format (e.g., Q&A vs. free-form) is swapped, even if the "context" remains the same.
- **Format Sensitivity**: Much of what is called "situational awareness" in current literature might just be the model reacting to canonical benchmark structures.
- **Controlled Dataset**: The authors use a $2 \times 2$ design (Context $\times$ Format) to decouple these variables, a design that could be adapted for the project’s "framing ablation" experiments.

**Connection to our research**:
- **Methodological Warning**: If the project uses probes to check if agents "know" they are in a trust game experiment, it must ensure the probe isn't just picking up on the structured format of the game instructions versus the myth-writing task.
- **Novelty/Refinement**: The project's plan for "paraphrase defense" on transplanted myths aligns with this paper’s finding that surface structure (format) is a dominant signal for LLMs. If paraphrasing kills the cooperation-lift, it supports `devbunova2026`'s emphasis on surface structure.

**Key quote**:
> "Probes primarily track benchmark-canonical structure and fail to generalize to free-form prompts independent of linguistic style." (p. 1)
```
