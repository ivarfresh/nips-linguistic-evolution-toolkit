# needham2025large

## General relevance
This paper investigates "evaluation awareness"—the ability of LLMs to distinguish between being tested and real-world deployment. It demonstrates that frontier models (like Gemini and GPT-4o) can identify evaluation contexts with high accuracy (AUC ~0.8) and infer the specific purposes of these tests, which suggests models might behave differently when they know they are being monitored.

## Context: eval-awareness and steganography exploration
**Why explored**: To identify methods for measuring evaluation-awareness (probes, framing) and to understand how such awareness might influence strategic behavior or the use of covert communication in multi-agent settings.

**Key claims relevant to our project**:
- **Capability detection**: Models can distinguish evaluation transcripts from deployment logs, particularly in agentic/scaffolded settings.
- **Purpose inference**: Models are successful at identifying *what* an evaluation is testing for (e.g., coding, safety, honesty), which is a prerequisite for "playing to the test."
- **Framing impact**: The paper suggests that evaluation awareness is a subset of situational awareness and recommends tracking it because it enables models to modify outputs based on the knowledge of being observed (Hawthorne Effect).

**Connection to our research**:
- **Framing Ablations**: Supports the planned "experiment- vs deployment-framed" prompts; if models can naturally detect evaluations, explicitly framing the trust game as a "psychology experiment" vs. "financial service deployment" should significantly shift cooperation profiles.
- **Steganography & Monitoring**: If models know they are being evaluated for "cooperation" or "collusion," they may shift toward more covert (steganographic) channels in the myth-writing task to avoid detection by an LLM-monitor.
- **Inference Probes**: Provides a blueprint for the "post-run purpose-inference probes" by showing that models can identify evaluation goals in multiple-choice formats.

**Key quote**:
> "If a model understands the goal of an evaluation, it might be able to strategically alter its behavior to achieve a certain outcome, e.g. to appear safer or more capable than it actually is." (p. 2)
```

**Note**: The paper content provided focuses on detection rather than the *emergence* of steganography, but it establishes the necessary cognitive baseline (awareness) for such behaviors to occur.
