# References worth exploring from [korbak2025](../notes/korbak2025.md)

<!-- Curated list of references from this paper worth pursuing for the study of LLM steganography and monitor-aware behavior. -->

- **Reynolds & McDonell (2021)** — *Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm*
  Why: Cited as a foundational text for how LLMs "think" in natural language; relevant for understanding the mechanics of how CoT can be manipulated for strategic communication.
  Where: [arXiv:2102.07350](https://arxiv.org/abs/2102.07350)

- **Kojima et al. (2022)** — *Large Language Models are Zero-Shot Reasoners*
  Why: Basic reference for zero-shot CoT; useful for the "neutral persona" baseline in the trust game experiments.
  Where: [arXiv:2205.11916](https://arxiv.org/abs/2205.11916)

- **Wei et al. (2022)** — *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*
  Why: The seminal CoT paper; necessary for grounding the "load-bearing" vs "epiphenomenal" language debate in current LLM literature.
  Where: [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8bb57039750730722cc7b9f36f8a846f-Abstract-Conference.html)

- **[Uncertain: Likely citing Christiano or Hubinger]** — *Inner Alignment / Deceptive Alignment*
  Why: Korbak et al. focuses on monitoring CoT for "intent to misbehave." References to deceptive alignment are critical for the project's interest in "eval-awareness" and "covert collusion." (Note: Specific citation not visible in truncation, look for "Hubinger et al. 2019" in full text).

- **[Uncertain: Likely citing Dennison or Roger]** — *Steganography in LLMs*
  Why: The paper mentions the "fragility" of monitorability. References regarding models hiding their true reasoning (steganography) are directly relevant to the project's "paraphrase defense" and "emergent steganography" experiments.
