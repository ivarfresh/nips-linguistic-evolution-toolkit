# References worth exploring from [laine2024](../notes/laine2024.md)

- **Ngo et al. (2023)** — *The Alignment Problem from a Deep Learning Perspective*
  Why: Likely provides the theoretical foundation for "situational awareness" and how it relates to instrumental convergence (like steganography or reward hacking) in LLM agents.
  Where: arXiv:2303.11504

- **Berglund et al. (2023)** — *The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"* (Note: Referenced in SAD context as part of situational awareness logic)
  Why: Relevant to how models generalize self-knowledge and whether situational awareness is robust or brittle across different prompt framings.
  Where: arXiv:2309.12288

- **Anwar et al. (2024)** — *Foundational Challenges in Assuring Alignment and Safety*
  Why: Cited for defining situational awareness; useful for the "eval-awareness" framing ablation experiments in the trust game project.
  Where: arXiv:2404.14723

- **Anthropic (2024c)** — *The Claude 3 Model Family: Model Card*
  Why: Since Claude 3 Opus/Sonnet are top performers on the SAD benchmark, the model card likely contains details on "character training" or system prompts that influence self-identification and awareness.
  Where: Anthropic website / reports

- **OpenAI (2024a)** — *GPT-4o System Card*
  Why: Provides context on how OpenAI trains models to recognize their identity, which is a direct baseline for the "eval-awareness" probes planned for the project.
  Where: OpenAI website / reports
