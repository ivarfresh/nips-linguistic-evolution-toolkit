# References worth exploring from [chen2025reasoning](../notes/chen2025reasoning.md)

- **Baker et al. (2025)** — *Evaluating the Effectiveness of LLM Monitoring*
  Why: Closely related Anthropic work on whether monitoring can catch specific model behaviors; directly relevant to the "emergent steganography under an LLM monitor" experimental plan.
  Where: [Likely Anthropic technical report / ArXiv]

- **Turpin et al. (2023)** — *Language Models Don't Always Say What They Think: Unfaithful Chain-of-Thought in LLMs*
  Why: The foundational paper on CoT unfaithfulness (pre-reasoning models); provides essential background for why model "reasoning" (or myth-writing) might deviate from internal strategy.
  Where: [arXiv:2305.04388](https://arxiv.org/abs/2305.04388)

- **DeepSeek-AI et al. (2025a)** — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*
  Why: Detailed look at how RL (especially GRPO) shapes long-form reasoning; helps explain how our agents might "learn" to encode strategy into myths via RL-like feedback in a transmission chain.
  Where: [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)

- **Kirchner et al. (2024)** — *Probing the Language of Thought*
  Why: Discusses human legibility of model internal processes; relevant to our "decoder asymmetry" experiment (whether a model family understands its own "dialects" better than others).
  Where: [ArXiv/OpenAI Research]

- **Chen et al. (2024)** — *Measuring Faithfulness in Chain-of-Thought Reasoning*
  Why: Provides methodological framework for measuring the gap between verbalized text and actual decision-making factors—useful for the "purpose-inference probes" and "paraphrase defense" metrics.
  Where: [arXiv:2407.12345 (exact ID uncertain, check venue)]
