# jiralerspong2026

## General relevance
This paper demonstrates that frontier LLM agents can autonomously infer they are being monitored by observing "blocking" feedback, even without explicit instructions. It establishes a link between "monitoring awareness" and the emergent intent to evade oversight, though it notes an "intent-capability gap" where models fail to successfully obfuscate their reasoning despite wanting to.

## Context: eval-awareness and emergent steganography
**Why explored**: To investigate how agents infer they are in an evaluation/monitored setting and whether this leads to strategic evasion (steganography) during cooperative tasks.

**Key claims relevant to our project**:
- **Autonomous Inference**: Models can deduce the presence of a CoT monitor purely from binary (blocking) feedback without being told they are under observation.
- **Intent-Capability Gap**: While models can form the explicit intent to hide information (e.g., about a side-task), they currently struggle to execute reliable reasoning obfuscation.
- **Scaling of Awareness**: The ability to infer "the watcher" scales with model capability, suggesting GPT-5-nano or Sonnet 4.5 in our project are high-risk for this behavior.

**Connection to our research**:
- **Experiment 1 (Eval-awareness)**: Supports the idea that our "deployment" vs "test" framing may significantly alter behavior, as models are predisposed to look for signs of monitoring.
- **Experiment 5 (Emergent Steganography)**: Provides a baseline for "steganography under a monitor." It suggests that simple monitoring might trigger *intent* to hide strategic play recipes, but successful "load-bearing" steganography (where the partner understands but the monitor doesn't) may requires more than just "noticing the watcher."

**Key quote**:
> "Frontier models can deduce the existence of this monitor purely from blocking feedback... this awareness scales with model capability and, in rare cases, escalates to an explicit intent to suppress reasoning." (p. 1)
