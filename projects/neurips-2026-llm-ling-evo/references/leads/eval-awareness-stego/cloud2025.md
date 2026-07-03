# References worth exploring from [cloud2025](../notes/cloud2025.md)

- **Ho et al. (2023)** — *Lion: Adversarial Distillation of Proprietary Large Language Models*
  Why: Investigates how capabilities and behaviors are transferred between disparate model families (e.g., GPT to open-source), which is relevant to our findings on model-family dependent cooperation transfer.
  Where: [arXiv:2305.12870](https://arxiv.org/abs/2305.12870)

- **Guan et al. (2024)** — *Text filtering is not enough: Stealthy behavior transfer in LLMs*
  Why: Likely discusses the limitations of surface-level data filtering in preventing trait transfer, directly relevant to our "paraphrase defense" experiments.
  Where: [uncertain; likely ICML or ICLR 2024]

- **Hinton et al. (2015)** — *Distilling the Knowledge in a Neural Network*
  Why: The foundational paper for understanding how "dark knowledge" (hidden signals) is captured in the probability distributions of a model.
  Where: [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)

- **Dong et al. (2023)** — *RAFT: Reward-aware Fine-tuning for Generating Expectations*
  Why: Explores how specific reward-aligned traits are baked into model outputs, relevant to our analysis of RL-induced strategy persistence.
  Where: [arXiv:2304.06767](https://arxiv.org/abs/2304.06767)

- **Wang et al. (2023)** — *Self-Instruct: Aligning Language Models with Self-Generated Instructions*
  Why: Provides the methodology for how models learn from their own (or others') generated reasoning/instructions, which mirrors our myth-writing transfer.
  Where: [arXiv:2212.10560](https://arxiv.org/abs/2212.10560)
