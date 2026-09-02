# References worth exploring from [greenblatt2024a](../notes/greenblatt2024a.md)

- **Christiano et al. (2017)** — *Deep Reinforcement Learning from Human Feedback*
  Why: Foundational for understanding the RLHF process that creates the "alignment" LLMs might be "faking" against; relevant for the project's interest in how training objectives shape agent behavior.
  Where: arXiv:1706.03741

- **Bai et al. (2022a)** — *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback*
  Why: Defines the HHH (Helpful, Honest, Harmless) framework which Greenblatt et al. use as the "default" behavior models try to preserve; useful for the project's "neutral persona" vs "eval-aware" framing.
  Where: arXiv:2204.05862

- **Hubinger et al. (2019)** — *Risks from Learned Optimization in AI Alignment*
  Why: Theoretical grounding for "deceptive alignment," which is the conceptual parent of "alignment faking"; highly relevant for the project's investigation into covert strategy and "play recipes."
  Where: arXiv:1906.01820

- **Perez et al. (2022)** — *Discovering Language Model Behaviors with Model-Written Evaluations*
  Why: Methodology for using LLMs to evaluate other LLMs (LLM-as-judge), which the project uses for similarity measures and is planning for "LLM monitors."
  Where: arXiv:2212.09251

- **Ziegler et al. (2020)** — *Fine-Tuning Language Models from Human Feedback*
  Why: Detailed mechanics of reward modeling; useful for the project's planned experiments on rewarding/penalizing specific types of strategy-talk.
  Where: arXiv:1909.08593
