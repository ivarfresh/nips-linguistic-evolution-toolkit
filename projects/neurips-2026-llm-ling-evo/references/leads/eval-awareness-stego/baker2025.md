# References worth exploring from [baker2025](../notes/baker2025.md)

<!-- Curated list of references focusing on monitoring, obfuscation, and reward hacking relevant to the LLM Dyad project. -->

- **Amodei et al. (2016)** — *Concrete Problems in AI Safety*
  Why: Provides the foundational framework for reward hacking (functional equivalent to the "strategy-talk" the user wants to monitor) and side effects.
  Where: arXiv:1606.06565

- **Hubinger et al. (2019)** — *Risks from Learned Optimization in Advanced Machine Learning Systems*
  Why: Relevant to the project's interest in "eval-awareness"; discusses how models might pursue "mesa-objectives" (like covert cooperation) that differ from the trainer's goals.
  Where: arXiv:1906.01820

- **Krakovna et al. (2020)** — *Specification Gaming: The Design of Safe Selection Criteria*
  Why: Offers a taxonomy of how agents bypass intended rules; useful for categorizing the "play recipes" or "steganographic channels" found in the myth-writing task.
  Where: DeepMind Blog / arXiv:2001.06073

- **Everitt et al. (2017)** — *Reinforcement Learning with a Corrupted Reward Signal*
  Why: Directly relates to the "LLM monitor" experiment planned by the user, specifically how agents behave when the feedback signal (monitor) can be gamed or bypassed.
  Where: arXiv:1705.08474

- **Armstrong et al. (2020)** — *Pitfalls of Learning a Reward Function Online*
  Why: Explores the dynamics of an agent interacting with a feedback provider (monitor), which mirrors the user's "transmission-chain" with an LLM judge.
  Where: arXiv:2004.13654
