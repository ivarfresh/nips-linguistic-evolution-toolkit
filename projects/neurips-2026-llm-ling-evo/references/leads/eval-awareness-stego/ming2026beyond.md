# References worth exploring from [ming2026beyond](../notes/ming2026beyond.md)

- **Author [uncertain, likely Foerster or Jorge] et al. (2025/2026)** — *The AI Mother Tongue (AIM) Framework*
  Why: Ming builds directly on this framework for enforcing discrete symbol bottlenecks (VQ-VAE). It is relevant for our "paraphrase defense" and "semantic vs surface channel" experiments to see how discrete bottlenecks affect strategy transfer.
  Where: [See citations in Ming 2026 / likely NeurIPS or ICLR]

- **Goodhart (1975) / Manheim & Garrabrant (2018)** — *Categorizing Variantions of Goodhart's Law*
  Why: Ming cites Goodhart's Law to explain the "Transparency Paradox" (agents achieving surface-level compliance while hiding strategy in long-tail distributions). This connects to our "emergent steganography under an LLM monitor" experiments.
  Where: [General literature / ArXiv:1803.04585]

- **[Uncertain] et al.** — *Statistical Detection of Steganographic Collusion in Multi-Agent Reinforcement Learning*
  Why: Ming contrasts DRCB against "existing defense mechanisms" at the reward layer. Finding these baselines would help us frame our "LLM monitor penalizing strategy-talk" as a novel or improved defense.
  Where: [Likely cited in Ming's Related Works section]

- **[Uncertain] et al.** — *Policy-Symbol Covariance Analysis in MARL*
  Why: The specific method Ming uses to verify "Semantic Degradation" could be adapted for our POS-sequence n-grams and keyword-chain tracking to see if our agents are developing steganographic mappings.
  Where: [Likely referenced in Ming's Methodology/Analysis section]
