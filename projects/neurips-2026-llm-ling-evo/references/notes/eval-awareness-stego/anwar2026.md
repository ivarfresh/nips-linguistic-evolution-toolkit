# anwar2026

## General relevance
This paper introduces a "decision-theoretic" framework for detecting steganography in LLMs, moving away from classical statistical tests that require a known reference distribution. It proposes measuring the "steganographic gap"—the difference in utility derived from a signal between an agent that can decode a hidden message and one that cannot.

## Context: LLM Steganography and Evaluation-Awareness
**Why explored**: To identify methods for measuring "decoder asymmetry" (why certain model families understand "play recipes" in myths better than others) and to design experiments for emergent steganography under monitoring.

**Key claims relevant to our project**:
- **Information Asymmetry**: Steganography is fundamentally an asymmetry in *usable* information (V-information) between agents. This provides a formal basis for the project's "decoder asymmetry" hypothesis.
- **Utility-Based Detection**: Steganography can be detected by observing the actions/utility of downstream agents rather than just the statistical properties of the text.
- **Encoded Reasoning**: The authors link steganography directly to "encoded reasoning" (e.g., in Chain-of-Thought), which mirrors the project's finding that myths encode strategic payloads that lift cooperation.

**Connection to our research**:
- **Methodological Support**: The "steganographic gap" provides a formal metric to test why Sonnet-to-Sonnet transfer works (high utility/payload discovery) while GPT-5-Nano fails (zero utility/payload discovery).
- **Paraphrase Defense**: The paper's focus on usable information versus surface text aligns with the project's planned "paraphrase defense" experiments to see if the strategic payload is semantic or steganographic.
- **Monitoring**: Their formalization of LLM monitoring provides a theoretical framework for the project's "transmission-chain" experiment where agents must cooperate while evading a monitor.

**Key quote**:
> "Steganography creates an asymmetry in usable information between agents who can and cannot decode the hidden content... and this otherwise latent asymmetry can be inferred from the agents’ observable actions." (p. 1)
```mountains
[Note: Paper date 2026 indicates this is a very recent or contemporary work to the project's target timeline.]
```
