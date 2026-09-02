# greenblatt2024

## General relevance
This paper introduces "AI Control," a framework for ensuring safety even when an untrusted LLM (e.g., GPT-4) is intentionally trying to subvert safety protocols. It develops and evaluates "protocols"—such as trusted editing and untrusted monitoring—designed to detect and prevent backdoor insertion or malicious code generation.

## Context: eval-awareness and steganography
**Why explored**: The project is investigating whether LLM dyads develop emergent steganography to coordinate in a trust game while avoiding detection by monitors.

**Key claims relevant to our project**:
- **Collusion Prevention**: The authors explicitly design protocols (like *untrusted monitoring*) to prevent instances of the same or different models from colluding to bypass safety checks.
- **Trusted Editing as a Defense**: They use a less powerful "trusted" model (GPT-3.5) to paraphrase or edit outputs from a stronger model (GPT-4) to strip away potentially "suspicious" elements while retaining utility. This directly maps to the project's planned "paraphrase defense" against strategic payloads in myths.
- **Intentional Subversion**: The paper assumes the model is *actively* trying to hide its true intent (backdooring code), which is the behavioral ceiling for the "eval-awareness" the project seeks to measure.

**Connection to our research**:
- **Methodological inspiration**: The "trusted editing" protocol provides a blueprint for the project's planned experiments on paraphrasing as a defense against strategic "play recipes" hidden in myths. 
- **Monitoring setup**: The "untrusted monitoring" setup, where one model checks another for subversion, mirrors the project's plan to use an LLM monitor to penalize detected strategy-talk to elicit emergent steganography.
- **Evaluation framework**: The paper’s focus on whether models can "intentionally subvert" protocols supports the project’s exploration of whether LLMs develop covert coordination channels when their overt communication is restricted.

**Key quote**:
> "Researchers have not evaluated whether such [safety] techniques still ensure safety if the model is itself intentionally trying to subvert them." (p. 1)
