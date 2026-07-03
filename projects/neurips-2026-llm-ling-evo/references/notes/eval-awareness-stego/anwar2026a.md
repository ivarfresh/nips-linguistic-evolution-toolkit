# anwar2026a

## General relevance
This paper provides an information-theoretic framework for "monitorability"—the ability of an external observer (monitor) to detect specific attributes in an LLM's intermediate reasoning (Chain-of-Thought). It identifies why monitors fail (information gaps and elicitation errors) and proposes methods to either reward monitorable transparency or maximize mutual information between reasoning and output.

## Context: Emergent steganography and monitor defenses
**Why explored**: The project aims to test if LLM agents develop steganographic communication (hidden strategic "play recipes" in myths) when a monitor penalizes detectable strategy-talk.

**Key claims relevant to our project**:
- **Monitoring Condition**: Non-zero mutual information between the reasoning trace (CoT/Myth) and the behavior (Game action) is necessary for a monitor to work.
- **Steganography Risk**: The authors explicitly note that "steganographic reasoning" undermines monitor efficacy, which aligns with the project's "strategic payload" hypothesis.
- **Training against Monitors**: They demonstrate that models can be trained (rewarded) to be more monitorable, which provides a methodological inverse for the project's plan to use a monitor to *drive* steganography.
- **Information Gap**: Identifies that a monitor might fail even if info is present because it cannot "extract" the signal, supporting the project's "decoder asymmetry" find (where only certain model families can decode the strategic payload).

**Connection to our research**:
- **Methodological Support**: The information-theoretic measures (Conditional Mutual Information) could be used to quantify if the "myth" is load-bearing or epiphenomenal to the trust game actions.
- **Experimental Design**: Provides a blueprint for the "LLM monitor" the project intends to build to penalize strategy-talk, specifically highlighting the risk of "Reward Hacking" when the monitor is imperfect.
- **Challenge**: If the project's agents are "locked-in" (preliminary finding 1), Anwar et al.'s work suggests that maximizing mutual information might be required to break the decoupling between their talk (myths) and their actions (game play).

**Key quote**:
> "CoT monitorability can be systematically improved through targeted training objectives... [or] maximizing conditional mutual information between outputs and CoTs." (p. 2)
