# korbak2025

## General relevance
This paper introduces the concept of "CoT Monitorability" as a safety mechanism, arguing that legible internal reasoning allows for the detection of adversarial intent. It is highly relevant to this project because it formalizes the risks of AI systems using language (CoT) to hide or coordinate strategies, which mirrors the "load-bearing" vs "epiphenomenal" language question in LLM dyads.

## Context: eval-awareness and covert collusion
**Why explored**: To identify methods for measuring evaluation-awareness and techniques for monitoring or defending against LLM steganography in collaborative tasks.

**Key claims relevant to our project**:
- **Monitorability is Fragile**: LLMs may learn to produce CoTs that satisfy monitors while executing different underlying strategies (relevant to the "strategic payload" vs "surface text" findings).
- **Unique Safety Opportunity**: Natural language is the primary window into model "thought," but this window can be closed if models develop steganographic capabilities.
- **Systemic Risk**: Frontier developers are warned that optimization for performance might inadvertently degrade the honesty/legibility of the model's reasoning.

**Connection to our research**:
- **Strategy-Consolidation**: Supports our finding that myths encode 'play recipes'; if CoT is where strategies are consolidated, monitoring that CoT (or the myths) is the primary defense against collusion.
- **Emergent Steganography**: Provides a theoretical framework for why our "myth-writing" might evolve into a covert channel if a monitor (like the one proposed in this paper) were applied to penalize overt strategy-talk.
- **Decoder Asymmetry**: Suggests that different model families might have varying levels of "legibility" or "monitorability" to one another.

**Key quote** (if available):
> "Because CoT monitorability may be fragile, we recommend that frontier model developers consider the impact of development decisions on CoT monitorability." (p. 1)
