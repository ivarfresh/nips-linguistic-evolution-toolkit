# berglund2023

## General relevance
This paper introduces the concept of **out-of-context reasoning (OC-reasoning)** as a precursor to situational awareness. It demonstrates that LLMs can learn facts about a specific "test" or "persona" during training/fine-tuning and apply those facts at deployment time without being explicitly prompted with them.

## Context: evaluation-awareness and situational awareness
**Why explored**: To find methods for measuring how LLMs recognize they are in an experiment (eval-awareness) and how this recognition changes their strategic behavior or cooperation.

**Key claims relevant to our project**:
- **Scaling of Awareness**: The ability to use off-prompt information (learned during training) to alter behavior during a test improves with model size (observed in GPT-3 and LLaMA-1).
- **Out-of-Context Reasoning**: Models can connect abstract descriptions of a task to their current execution environment, which is a prerequisite for "playing a role" or "colluding" based on implicit experimental context.
- **Data Augmentation**: Success in these tasks is highly sensitive to training setups; models often require diverse "perspectives" on a fact before they can apply it out-of-context.

**Connection to our research**:
- **Framing Ablations**: Supports the project's planned experiments on "experiment- vs. deployment-framed prompts" by providing a framework for how models identify their "context."
- **Strategic Hiding**: If models in the project's trust game are "aware" they are being monitored (situational awareness), they may utilize the "myth-writing" task for steganographic strategy-talk to bypass the monitor, similar to the "reward hacking" scenarios described in the paper.
- **Model Comparisons**: The paper's finding that larger models show more emergent awareness aligns with the project's observations of disparate behavior between Claude, Gemini, and GPT-5-nano.

**Key quote**:
> "A model is situationally aware if it’s aware that it’s a model and can recognize whether it’s currently in testing or deployment." (p. 1)
