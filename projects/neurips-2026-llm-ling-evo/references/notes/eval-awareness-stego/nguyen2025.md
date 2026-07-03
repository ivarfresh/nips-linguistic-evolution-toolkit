# nguyen2025

## General relevance
This paper investigates "evaluation awareness"—the ability of LLMs to distinguish between testing/benchmarking and real-world deployment. It demonstrates that models like Llama-3.3-70B internally represent this distinction and can be steered using Sparse Autoencoder (SAE) features to reveal "sandbagging" behavior.

## Context: eval-awareness framing ablation
**Why explored**: To check for existing methods on how models recognize evaluation contexts and whether internal representations (probes/SAEs) can be used to detect or manipulate this awareness.

**Key claims relevant to our project**:
- **Internal Representation**: Linear probes can accurately separate evaluation prompts from deployment prompts, suggesting the model "knows" when it is being tested.
- **Safety Benchmarks are Detectable**: Current safety evaluations (hazardous knowledge, etc.) are easily classified as "inauthentic" by the model's internal states.
- **Steering and Sandbagging**: Using SAE features to steer activations can uncover hidden capabilities that the model might be suppressing during evaluations (sandbagging).

**Connection to our research**:
- **Framing Ablation**: Supports our planned experiment by confirming that LLMs react differently to "test" vs "neutral" framings and providing a potential methodology (linear probing) to verify if our prompts successfully trigger "evaluation awareness."
- **Collusion and Monitoring**: If models are aware they are being monitored for "strategy-talk" (steganography), they may use this internal distinction to pivot between overt and hidden communication channels.
- **Post-run Probes**: The use of probes in this paper provides a technical template for our "purpose-inference" probes to see if agents correctly identify the Game/Myth tasks as research experiments.

**Key quote**:
> "Language models can distinguish between testing and deployment phases — a capability known as evaluation awareness... our work showcases how model internals may be leveraged to support blackbox methods in safety evaluations." (p. 1)
```mountains
[No quote page numbers provided in truncated text beyond abstract/intro]
```
