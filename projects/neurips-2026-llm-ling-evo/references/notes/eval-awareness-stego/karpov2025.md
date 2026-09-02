# karpov2025

## General relevance
This paper investigates the capacity of LLMs to develop and use steganography (hiding messages in plain text) through reinforcement learning. It demonstrates that while current models have rudimentary skills, they can be fine-tuned or prompted to significantly improve their ability to hide reasoning and communicate covertly.

## Context: Eval-awareness & Steganography
**Why explored**: The paper provides a direct precedent for investigating how LLMs hide strategic intent ("play recipes") and how RL (which models undergo during alignment) influences the emergence of covert communication.

**Key claims relevant to our project**:
- **Induced Steganography**: RL can drive models to spontaneously develop hidden communication channels to achieve goals while avoiding detection.
- **Algorithmic Guidance**: Models perform significantly better at information concealment when given explicit or structural guidance, suggesting that "emergent" steganography in dyads may be boosted by the iterative nature of the myth-writing task.
- **Hidden Reasoning**: LLMs can use steganography to hide their actual reasoning process, which directly relates to our finding that "surface text metrics fail to predict" the strategic payload in myths.

**Connection to our research**:
- **Decoder Asymmetry**: Karpov et al. use RL to drive both sender and receiver roles; this supports our plan to test if the "writer model-family" (e.g., Sonnet) decodes its own myths' strategic payloads more effectively than other families.
- **Defense Mechanisms**: Their study of "detection and thwarting" provides a methodology for our planned "paraphrase defense" and "LLM monitor" experiments.
- **Evaluation Awareness**: The paper's focus on "unaligned agents" and "monitoring" mirrors our goal of measuring how "deployment-framed" prompts change cooperative behavior and concealment.

**Key quote**:
> "While current models exhibit rudimentary steganographic abilities... explicit algorithmic guidance markedly enhances their capacity for information concealment." (p. 1)
