# gisler2026

## General relevance
This paper introduces the concept of **subliminal learning**, where a student model inherits behavioral traits (e.g., preferences, biases) from a teacher model through training data that is semantically unrelated to those traits. It demonstrates that even when using faithful natural language paraphrases and filtering for content, "covert" transmission of behavior persists, suggesting a "steganographic" channel in stylistic or linguistic nuances.

## Context: Paraphrase defense and hidden-signal transfer
**Why explored**: To investigate if "strategy-lifting" effects in the project's myth-writing tasks are transmitted via surface-level steganographic channels and if paraphrase filters can block this transmission.

**Key claims relevant to our project**:
- **Transmission via Paraphrase**: Behavioral traits (animal preferences) transfer to a student model even when the training set consists of paraphrases with fixed semantic content.
- **Resilience to Contradiction**: Transmission occurs even if the content explicitly contradicts the trait (e.g., training on "I hate owls" from an owl-loving teacher still increases the student's owl preference).
- **Steganographic Channel**: Suggests that traits are encoded in subtle linguistic choices (syntax, word choice) rather than overt semantic content.
- **Failure of Content Inspection**: Standard safety/semantic filters are insufficient to prevent the propagation of latent behaviors.

**Connection to our research**:
- **Paraphrase Defense**: Supports the project's plan to test "paraphrase defense" on transplanted seed myths; Gisler et al. suggest this defense will likely fail to block strategic "recipes" if they are encoded stylistically.
- **Strategy Consolidation**: Provides a mechanism for why myths lift cooperation—they may be "subliminally" training the student on a cooperative persona through linguistic style rather than just the "play recipes" in the text.
- **Model-Family Dependence**: Complements the project's finding that transfer is model-family dependent, as subliminal learning relies on the student's ability to pick up on the teacher's specific stylistic distributions.

**Key quote**:
> "Content-based inspection cannot detect such transmission, and even preference-contradicting content fails to prevent it." (p. 1)
