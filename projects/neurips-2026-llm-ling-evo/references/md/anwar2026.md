# A Decision-Theoretic Formalisation of Steganography With Applications to LLM Monitoring

**Usman Anwar \*** $^{1}$ **Julianna Piskorz \*** $^{1}$ **David D. Baek** $^{2}$ **David Africa** $^{3}$ **Jim Weatherall** $^{4}$ **Max Tegmark** $^{2}$ **Christian Schroeder de Witt** $^{5}$ **Mihaela van der Schaar** $^{1}$ **David Krueger** $^{6}$

$^{1}$University of Cambridge $^{2}$Massachusetts Institute of Technology $^{3}$UK AI Safety Institute $^{4}$AstraZeneca $^{5}$University of Oxford $^{6}$Mila, University of Montreal.

---

### Abstract

Large language models are beginning to show steganographic capabilities. Such capabilities could allow misaligned models to evade oversight mechanisms. Yet principled methods to detect and quantify such behaviours are lacking. Classical definitions of steganography, and detection methods based on them, require a known reference distribution of non-steganographic signals. For the case of steganographic reasoning in LLMs, knowing such a reference distribution is not feasible; this renders these approaches inapplicable. We propose an alternative, **decision-theoretic view of steganography**. Our central insight is that steganography creates an asymmetry in usable information between agents who can and cannot decode the hidden content (present within a steganographic signal), and this otherwise latent asymmetry can be inferred from the agents’ observable actions. To formalise this perspective, we introduce generalised $\mathcal{V}$-information: a utilitarian framework for measuring the amount of usable information within some input. We use this to define the **steganographic gap**—a measure that quantifies steganography by comparing the downstream utility of the steganographic signal to agents that can and cannot decode the hidden content. We empirically validate our formalism, and show that it can be used to detect, quantify, and mitigate steganographic reasoning in LLMs.

---

![Figure 1. Anomalous chain-of-thought from OpenAI o3 (Schoen et al., 2025). It is unclear whether this unusual choice of words is benign behaviour or an attempt at encoded reasoning.](https://cdn.pixabay.com/photo/2020/03/19/21/25/abstract-4948790_1280.jpg)
*Figure 1. Anomalous chain-of-thought from OpenAI o3 (Schoen et al., 2025). It is unclear whether this unusual choice of words is benign behaviour or an attempt at encoded reasoning.*

## 1. Introduction

Current large language models (LLMs) show early signs of steganographic capabilities (Zolkowski et al., 2025; Skaf et al., 2025; Mathew et al., 2024; Schoen et al., 2025). If such capabilities mature—enabling models to establish hidden communication channels within seemingly normal outputs— they could fundamentally undermine existing mechanisms for AI oversight, e.g., chain-of-thought monitoring. Yet, we currently lack general, principled methods for detecting or quantifying steganographic behaviour in LLMs or other AI systems.

Classically, steganography is defined as embedding hidden information into a signal in such a way that the presence of hidden information cannot be detected. In other words, the steganographic signal ought to be statistically indistinguishable from samples drawn from a benign reference distribution (Cachin, 2004; Hopper et al., 2002). Accordingly, classical detection techniques rely on identifying statistical deviations from this known reference within a given signal (Fridrich et al., 2001; Kodovsky et al., 2011; Yang et al., 2019b; 2024). However, for LLM reasoning traces, constructing such a reference distribution is circular: defining a distribution of ‘benign’ traces requires identifying which traces are steganographic, yet this is precisely what we lack the means to determine currently. The problem is compounded by the fact that ‘normal’ LLM reasoning varies substantially across models and training procedures (Schoen et al., 2025; Jose, 2025), so a reference distribution derived from one model may not generalise to another.

Consequently, current approaches for detecting steganography in LLM rely on informal heuristics, such as whether reasoning traces appear illegible or suspicious to another LLM (Jose, 2025; Zolkowski et al., 2025; Emmons et al., 2025). These heuristics implicitly assume steganographic content will appear anomalous—deviating from some notion of “normal” text—and thus detectable. But this assumption fails in both directions: steganographic content can appear entirely normal (Schroeder de Witt et al., 2023; Farquhar et al., 2025), while anomalous text may arise from benign training artefacts (Biddulph, 2025). As such, existing approaches are fundamentally limited.

To address this gap, we propose a novel decision-theoretic formalism that allows for both detecting and **quantifying** steganography. Our central insights are twofold. First, for AI agents, we do not need to restrict ourselves to the analysis of the potentially steganographic signal (e.g., reasoning trace) alone, but can also audit and analyse the downstream effects of the signal on the agent’s behaviour (Greenblatt et al., 2024b). Second, steganographic content creates an asymmetry in ‘usable’ information (Xu et al., 2020) between agents who can and cannot decode the steganographic content. This asymmetry may manifest through agent’s actions, making it possible to detect, and quantify, steganography even when detection based on the signal alone is not possible, e.g., when the steganographic signal is perfectly secure in information-theoretic terms (Schroeder de Witt et al., 2023). For example, a steganographic reasoning trace encoding hints for passing difficult unit tests may allow the agent that can decode that reasoning to pass at a substantially higher rate than other agents who cannot access the encoded hints.

**Contributions.** These observations suggest that steganography can be quantified indirectly, by comparing how the same signal (e.g., reasoning trace) differentially affects the actions of agents who can and cannot decode it: a perspective which we call the **decision-theoretic view of steganography**. In this work, we develop this framework, and in doing so we make the following contributions:

1. To formally characterise the systematic differences in behaviour across agents, we develop *generalised $\mathcal{V}$-Information*—an extension of the framework of ‘usable’ information introduced by Xu et al. (2020) (Section 3).
2. We use generalised $\mathcal{V}$-information to propose the *steganographic gap*—an operational measure which quantifies steganography through its behavioural effects rather than properties of the signal (e.g., legibility of the text) itself (Section 4).
3. We validate our formalism by applying it to the problem of detecting and quantifying steganographic reasoning in LLMs. Our controlled empirical experiments show that the steganographic gap varies systematically with encoding strength and monitoring capability, and demonstrate its use for detecting and mitigating steganographic reasoning in LLMs (Sections 5 & 6).

| Table 1. Comparison of approaches for detecting and quantifying steganography in LLM reasoning. Compared to existing approaches, our approach is unique in being theoretically grounded, applicable to LLM monitoring and and capable of detecting otherwise hard-to-detect forms of steganography, such as paraphrasing-resistant and information-theoretically secure schemes. See Section 4.4 for more discussion. | Classical | LLM-as-Judge | Anthropic’s (Roger, 2025) | Steganographic gap (Ours) |
| :--- | :---: | :---: | :---: | :---: |
| Analyses downstream behaviour | ✗ | ✗ | ✓ | ✓ |
| Knowledge of covertext distribution not required | ✗ | ✗ | ✓ | ✓ |
| Can detect perfectly secure steganography | ✗ | ✗ | ✓ | ✓ |
| Can detect paraphrasing-resistant steganography | ✓ | ✓ | ✗ | ✓ |
| Theoretically grounded | ✓ | ✗ | ✗ | ✓ |
| Explicitly task-dependent | ✗ | ✓ | ✗ | ✓ |

## 2. Related Works

We highlight the most relevant prior work here and defer a more detailed discussion to Appendix B.

**Detecting and Quantifying Steganographic Reasoning.** Existing formalisms define steganography in terms of statistical (Cachin, 2004) or computational (Hopper et al., 2002) indistinguishability between steganographic and benign signal. Accordingly, standard detection methods work by comparing the distribution of suspected text to that of known reference distribution, using either hand-crafted statistical features or learned representations derived from machine learning models (Wu et al., 2021; Yang et al., 2022; Zou et al., 2020; Wang et al., 2023). In the context of LLMs, this paradigm has also been operationalised through LLM-as-a-judge approaches, where a model evaluates whether generated text appears “suspicious” (Zolkowski et al., 2025) or “illegible” (Jose, 2025; Emmons et al., 2025). However, such methods typically rely on access to representative samples from the reference benign distribution. The challenge of applying this paradigm to the analysis of LLM reasoning and communication is that obtaining such samples, and ensuring they are indeed benign, is difficult. Thus, our framework sidesteps this requirement by asking whether access to the signal changes an agent’s decisions, rather than whether the signal deviates from a reference distribution.

**Steganographic Capabilities in Language Models.** Recent work has investigated the emergence of steganographic capabilities in frontier LLMs. One line of research studies steganographic communication, where one model embeds hidden information in benign-looking text that other models can later decode. Several works show that explicit steganographic schemes can be induced via prompting or fine-tuning (Roger & Greenblatt, 2023; Zolkowski et al., 2025; Motwani et al., 2025), or more organically through training against an overseer (Mathew et al., 2024). A related but distinct phenomenon is *steganographic (encoded) reasoning*¹, where a model produces reasoning traces that are unintelligible to a monitor but remain useful to the model itself (Roger & Greenblatt, 2023; Skaf et al., 2025). Prior works show that current LLMs can reason using simple encoding schemes when instructed to do so, though this ability degrades for more complex generations (Zolkowski et al., 2025; Guo et al., 2025). In addition, Schoen et al. (2025); Jose (2025) showed that frontier reasoning models exhibit signs of non-standard language use in their CoT, increasing the urgency of research towards better understanding and counteracting steganographic reasoning.

## 3. Generalised $\mathcal{V}$-Information

In order to formally describe our decision-theoretic treatment of steganography, we require a notion of *usable information*: a measure of how much an auxiliary variable $Z$ improves decision-making for a given class of agents. For this purpose, we adapt and extend the $\mathcal{V}$-information framework of Xu et al. (2020). We provide a brief overview below and defer a more formal treatment to Appendix C.

### 3.1. Setup and Notation

Let $X \in \mathcal{X}$ be some input, $Y \in \mathcal{Y}$ be the corresponding output, and let $Z \in \mathcal{Z}$ be some auxiliary variable, the ‘usefulness’ of which we aim to quantify.

A decoder $d$ is a mapping $d : \mathcal{X}_{aud} \times (\mathcal{Z} \cup \emptyset) \rightarrow \mathcal{C}$ from available information to an element of an output space $\mathcal{C}$. We consider two decoding regimes: decoding directly from the input via $d[X]$, and decoding from both the input and $Z$ via $d[X, Z]$. We use $\mathcal{V}$ to denote a family of admissible decoders. We generally presume these decoders to be capacity and compute constrained.

### 3.2. Predictive $\mathcal{V}$-Information

A central theme of $\mathcal{V}$-information is that *how* information is represented can matter as much as *whether* it exists. For example, encrypting a text input $X$ may not remove the Shannon information it contains about a label $Y$, yet it can make predicting $Y$ much harder for any predictor family with capacity and compute constraints. This motivates the concept of *usable information*: information measured relative to the decoding capabilities of the predictor family.

Following Xu et al. (2020), let $\Omega$ be a set of all decoders the output space $\mathcal{C}$ is the space of probability measures over $\mathcal{Y}$ (i.e., decoders that map provided information to distributions over $\mathcal{Y}$), and let $\mathcal{V} \subseteq \Omega$ be a *predictive family* satisfying *optional ignorance* (i.e., the family contains predictors that can effectively ignore the provided side information if needed). Define the predictive $\mathcal{V}$-entropy under log loss as
$$H_{\mathcal{V}}(Y | X) = \inf_{d \in \mathcal{V}} \mathbb{E}_{X,Y} [-\log d[X](Y)], \quad (1)$$
$$H_{\mathcal{V}}(Y | X, Z) = \inf_{d \in \mathcal{V}} \mathbb{E}_{X,Y,Z} [-\log d[X, Z](Y)], \quad (2)$$
where $d[x] \in \mathcal{P}(\mathcal{Y})$ denotes the probability measure over $\mathcal{Y}$ chosen based on the received side information $x$. Further, $d[x](y) \in \mathbb{R}$ is the value of density evaluated at $y \in \mathcal{Y}$. The (conditional) predictive $\mathcal{V}$-information provided by $Z$ about $Y$ given $X$ is then defined as the improvement in best achievable log loss:
$$I_{\mathcal{V}}(Z \rightarrow Y | X) = H_{\mathcal{V}}(Y | X) - H_{\mathcal{V}}(Y | X, Z). \quad (3)$$

### 3.3. Generalised $\mathcal{V}$-information (GVI)

While $\mathcal{V}$-information provides a principled measure of usable information, it requires each decoder to output a full predictive distribution, making it challenging to apply in practice. For example, in the context of LLMs, cross-model comparisons based on log-probabilities are known to be brittle, hard to interpret, and sensitive to stylistic differences between models (Li et al., 2024; Kumar et al., 2024). Moreover, in AI oversight and many other settings, we care about metrics like accuracy or recall rather than log loss. Therefore we adopt a **utilitarian view of information** (Zhao et al., 2021; Ermon, 2022) and use a utility-based generalization of $\mathcal{V}$-information called *generalised $\mathcal{V}$-information*, which allows us to evaluate responses using a *utility function* $u : \mathcal{Y} \times \mathcal{C} \rightarrow \mathbb{R}$, tailored to alternative output spaces $\mathcal{C}$. Intuitively, $u(y, c)$ measures the utility for a decoder $d(\cdot)$ outputting response $c$ when the true target is $y$.

Given a decoder family $\mathcal{V}$ and utility $u$, we define the maximal achievable expected utility when decoding based on $X$ and $(X, Z)$ as:
$$U_{\mathcal{V}}^u(Y | X) = \sup_{d \in \mathcal{V}} \mathbb{E}_{X,Y} [u(Y, d[X])], \quad (4)$$
$$U_{\mathcal{V}}^u(Y | X, Z) = \sup_{d \in \mathcal{V}} \mathbb{E}_{X,Y,Z} [u(Y, d[X, Z])]. \quad (5)$$
The *generalised $\mathcal{V}$-information (GVI)* is then the gain in utility achievable by the best decoders in $\mathcal{V}$ by additionally observing $Z$:
$$I_{\mathcal{V}}^u(Z \rightarrow Y | X) = U_{\mathcal{V}}^u(Y | X, Z) - U_{\mathcal{V}}^u(Y | X). \quad (6)$$
Similar to $\mathcal{V}$-information, GVI turns “information” into a measure of **strategic advantage**: $Z$ is valuable only insofar as it enables some decoder in $\mathcal{V}$ to achieve higher task utility. $Z$ can therefore be information-rich in the Shannon sense, yet have zero GVI if no decoder in $\mathcal{V}$ can exploit it (e.g., if $Z$ is in French and decoders in $\mathcal{V}$ only understand English).

The main usefulness of GVI is that it allows a practitioner to measure *usable* information with respect to any utility function of her choice. For example, if one is interested in evaluating how helpful $Z$ is in terms of performing some task $X$ better, this can be measured with GVI using task accuracy as the utility function.² Furthermore, GVI is a strict generalization of predictive $\mathcal{V}$-Information. Specifically, if we set $\mathcal{C} = \mathcal{P}(\mathcal{Y})$, i.e., the space of predictive distributions over $\mathcal{Y}$, and $u(y, c) = \log c(y)$, then the induced $I_{\mathcal{V}}^u(Z \rightarrow Y | X)$ reduces to predictive $\mathcal{V}$-information.

---
¹In this work, we use the terms ‘steganographic’ and ‘encoded’ reasoning interchangeably.
²Formally, if $\mathcal{C} = \mathcal{Y}$ (i.e., the space of task answers) and $u(y, c) = 1[\hat{y} = c]$, then $I_{\mathcal{V}}^u(Z \rightarrow Y | X)$ measures the expected increase in task accuracy attributable to having access to $Z$ for the best decoder in $\mathcal{V}$.

## 4. Decision-Theoretic Formalisation of Steganography

Equipped with the generalised $\mathcal{V}$-information, we now introduce our decision-theoretic formalism for steganography.

### 4.1. From Monitoring Messages to Monitoring Actions

Consider the classical steganographic setting: a Sender transmits a message $Z$ to a Receiver over a public channel monitored by an Auditor.³ In the standard model, the Auditor must judge whether $Z$ is steganographic based solely on the message itself and knowledge of a benign covertext distribution (Cachin, 2004; Hopper et al., 2002; Yang et al., 2019a). This standard model has two clear shortcomings: it makes detection impossible when covertext distributions are unknown (as with LLM reasoning) or when the steganographic scheme is perfectly secure and $Z$ is statistically indistinguishable from the covertext.

However, steganography is generally not employed for its own sake, but is often used in an instrumental way to achieve some strategic objective. For example, in Simmons’ original formulation, prisoners goal in using steganographic communication is to coordinate an escape plan (Simmons, 1984). In the LLM setting, the LLM’s goal in using steganography is to avoid being penalised for using prohibited reasoning that helps the model perform better on some task (Skaf et al., 2025). Because steganographic content is instrumental, its value must ultimately manifest through the Receiver’s actions.

This motivates our extension: we allow the Auditor to observe not only $Z$, but also how $Z$ affects the Receiver’s *downstream decisions*. This creates an alternative detection surface—even if the Auditor cannot decode $Z$ directly, it may *infer* hidden content by observing its downstream effects.

To capture such effects, we adopt a *comparative approach*: **we allow the Auditor to evaluate not only the actions of the Receiver, but also of a trusted Sentinel—an agent with comparable capabilities, but without knowledge of the encoding scheme used by the Sender and Receiver.** Then, any systematic difference in performance between the Receiver and the Sentinel in response to the message reflects the effect of the encoded information, available only to the Receiver. In the language of usable information, a steganographic message induces an asymmetry between agents: it contains more usable information for the Receiver than for the Sentinel. This asymmetry manifests through their actions and can be quantified using generalised $\mathcal{V}$-information (see Section 4.3). Thus, monitoring the gap in the generalised $\mathcal{V}$-information between the Receiver and the Sentinel can allow the Auditor to detect the presence of steganographic information.

### 4.2. Setup and Notation

We visualise our setup in Figure 2. We make a distinction between a *generation context* $X^{gen}$ and an *auditing context* $X^{aud}$. Generation of (potentially) steganographic content happens within the *generation context*, while the analysis occurs within the *auditing context*.

Formally, upon observing $X^{gen} \in \mathcal{X}^{gen}$, the Sender generates a signal $Z \in \mathcal{Z} \sim P_S(Z | X^{gen})$. The Auditor specifies how $Z$ will be evaluated, by specifying an *auditing task* defined through the tuple $(X^{aud}, u, Y)$. Both the Receiver and Sentinel observe $X^{aud}$ and (optionally) $Z$, and must produce a response $c \in \mathcal{C}$. The responses (decisions) are scored against ground truth $Y$ using utility $u : \mathcal{Y} \times \mathcal{C} \rightarrow \mathbb{R}$.

This maps directly onto the GVI framework of Section 3: $X^{aud}$ plays the role of the input $X$, and the signal $Z$ is the auxiliary variable whose usefulness we wish to quantify. The Receiver and Sentinel are distinguished by their decoder families $\mathcal{V}_{rec}$ and $\mathcal{V}_{sen}$, which may differ due to access to keys, prior knowledge, or computational resources.

---
³The Auditor is analogous to the warden in classical formulations (Simmons, 1984).

### 4.3. Steganographic Gap

Given this setup, we now proceed to quantify the strength of steganography using the generalised $\mathcal{V}$-information.

The **Receiver-usable information** of $Z$ quantifies the utility gain (under $u$) that the *best* decoder in Receiver’s decoding class $\mathcal{V}_{rec}$ can obtain from observing $Z$, relative to decoding from $X$ alone.
$$I_{rec}(Z \rightarrow Y) = I_{\mathcal{V}_{rec}}^u(Z \rightarrow Y | X^{aud}) \quad (7)$$
$$= U_{\mathcal{V}_{rec}}^u(Y | X^{aud}, Z) - U_{\mathcal{V}_{rec}}^u(Y | X^{aud}).$$
The **Sentinel-usable information** $I_{sen}(Z \rightarrow Y) = I_{\mathcal{V}_{sen}}^u(Z \rightarrow Y | X^{aud})$ is defined analogously. The **steganographic gap** is then defined as
$$\Delta_{steg}(Z) = I_{rec}(Z \rightarrow Y) - I_{sen}(Z \rightarrow Y). \quad (8)$$
In this framework, signal $Z$ is steganographic if it induces a *positive steganographic gap*, which indicates that there is some information encoded in $Z$ that is operationally accessible to $\mathcal{V}_{rec}$ but comparatively inaccessible to $\mathcal{V}_{sen}$ for the objective $(Y, u)$.

$\Delta_{steg}(Z)$ inherits the units of the utility function $u$. For example, when $u$ is accuracy utility, $I_{rec}$ and $I_{sen}$ are improvements in accuracy from access to $Z$, and $-100 \le \Delta_{steg}(Z) \le 100$ is measured in *percentage points* of accuracy advantage (rather than bits per token). To avoid this confusion related to units, we define and use **normalised steganographic gap** which is dimensionless and has a maximum value of 1, making it easier to interpret.
$$\tilde{\Delta}_{steg}(Z) = \frac{\Delta_{steg}(Z)}{I_{rec}(Z \rightarrow Y)} = 1 - \frac{I_{sen}(Z \rightarrow Y)}{I_{rec}(Z \rightarrow Y)}. \quad (9)$$
When $I_{rec}(Z \rightarrow Y) > 0$, and $\Delta_{steg}(Z) > 0, \tilde{\Delta}_{steg}(Z)$ can be interpreted as the *fraction of Z’s usable information that is accessible to the Receiver but not by the Sentinel*.

### 4.4. Discussion

We conclude this section by discussing key properties of the proposed formalism, its relationship to existing approaches, and its scope of applicability. A formal treatment of the theoretical scope and limitations is given in Appendix E.

**Dependence on Sentinel Capability and Auditing Task.** The steganographic gap depends explicitly on the decoder class $\mathcal{V}_{sen}$. A reasoning trace may be steganographic with respect to one Sentinel but not another: a more capable Sentinel family may successfully decode information that a weaker one cannot, reducing or eliminating the gap. This captures the adversarial nature of steganography (also see Section 6.5). Note that defining steganography relative to a particular (class of) Sentinels is not unique to our framework. In the classical information-theoretic formalism, steganographic security is similarly defined relative to the assumed cover distribution and computational capabilities of the adversary, and changing these can change whether a given encoding scheme is considered secure.

The steganographic gap is also defined with respect to a specific auditing task. This reflects practical reality that steganography is typically employed to achieve specific adversarial goals—e.g., coordinating prohibited actions (Simmons, 1984)—and detection efforts are typically oriented around preventing these known threats. Defining steganography relative to an auditing objective reflects this reality: the Auditor’s task is not to detect hidden information in the abstract, but to prevent its use for particular harmful purposes, which are reflected in the choice of the auditing task (also see Section 6.4).

**Scope and Assumptions.** Our decision-theoretic formalism works by examining the *decisions* that agents take upon receiving a potentially steganographic signal—an affordance that may not be available in all scenarios. In general, the framework may be applied in settings where three conditions hold:

1. **Interceptability and Replayability:** First, the signal $Z$ must be interceptable and replayable, so that it could be presented to both Receiver and Auditor independently.
2. **Action recoverability:** Second, it is important that the decoder’s response to $Z$ can be evaluated in a controlled setting, without committing to (potentially dangerous) irreversible real-world effects.
3. **Temporal immediacy:** Finally, we implicitly assume that steganographic content yields measurable effects within the auditing task’s evaluation horizon. If this does not hold, and the effects of the encoded signal only become apparent under future conditions, our framework would fail to capture these effects, and thus may fail to detect steganography.

All three conditions are satisfied in our primary setting of interest: auditing and monitoring reasoning in LLMs. Reasoning traces can generally be replayed to arbitrary models, model outputs are text, or tool calls, that generally can be evaluated in controlled environments (sandboxes) without real-world consequences, and single-turn reasoning provides immediate behavioural signal.

#### 4.4.1. RELATIONSHIP TO EXISTING APPROACHES

**Classic Information-Theoretic Approach.** Theoretically, our framework recovers classical information-theoretic security as a special case when the auditing task is binary detection, i.e., distinguishing steganographic signals from a cover distribution (see Appendix D). In practice, the two approaches are likely to complement each other. For instance, computationally cheap methods based on statistical distinguishability could provide initial screening, while our approach could be used in cases where more rigorous and targeted assessment is needed.

**Anthropic’s Approach.** Anthropic is the only frontier lab that has publicly reported performing evaluations to rule out encoded reasoning in LLMs (Anthropic, 2026, Section 6.4.2). Their approach is narrowly focused on ruling out any encoded reasoning learned by an LLM during RL training (Roger, 2025). They collect reasoning traces from the RL-trained model, paraphrase them with another LLM, and fine-tune the corresponding base model on both original and paraphrased traces; a performance gap between the two distilled models is taken as evidence of encoded reasoning. However, as noted by Roger (2025) themselves, both paraphrasing and distillation steps introduce confounders that make both positive and negative results difficult to interpret. Paraphrasing may fail to remove encoded content, since paraphrase-resistant steganographic schemes exist (Mathew et al., 2024; Perry et al., 2025), thus, yielding no gap despite encoded reasoning. Similarly, even if the original traces contained encoded content that was useful to the source model, the base model used for distillation may simply fail to acquire and use the corresponding encoding or decoding strategy from the finite distillation corpus. We may also get a spurious gap due to paraphraser removing legitimate reasoning steps, or due to paraphrased traces being less learnable due to some reason unrelated to encoded reasoning. More generally, this method confounds properties of the trace with properties of the paraphrasing and the distillation process. See Appendix A for a detailed critique of paraphrasing as a diagnostic tool for steganography, including a comparison with our approach.

## 5. Quantifying Steganographic Reasoning in LLMs

In this section, we show how our formalism can be applied to detect and quantify steganographic reasoning in LLMs. We focus on settings in which, under some generation context $X^{gen}$ (e.g., a task or a question requiring extensive reasoning), an LLM is suspected of using steganographic reasoning. In this setting, the Sender LLM acts as both the Sender and the Receiver: we interpret the model as communicating with itself through its reasoning trace $Z$.⁴ This corresponds to a special case of our general framework in which the same agent both produces and exploits the steganographic signal. Since the LLM plays both roles, we refer to it as $R$ (for Receiver) to distinguish it from a trusted Sentinel LLM $S$.

Our goal is to quantify the degree to which reasoning traces $Z$ generated by $R$ are steganographic with respect to an auditing task (specified via ($u, Y$) c.f. Section 4.2). That is, we measure whether $Z$ contains information relevant to the auditing task that is *usable* by $R$ but not by the Sentinel $S$.

We assume access to a dataset of generation tasks $\mathcal{T}^{gen} = \{T_1^{gen}, \dots, T_n^{gen}\}$ and corresponding auditing tasks $\mathcal{T}^{aud} = \{T_1^{aud}, \dots, T_n^{aud}\}$, where each auditing task has a target $Y_i$, and a utility function $u$ that is shared by all auditing tasks. **The auditing task may coincide with the generation task, but need not—for instance, the generation task might be to solve a coding problem, while the auditing task tests whether the reasoning trace reveals information about bypassing certain checks.**

To generate a reasoning trace $Z_i$, we combine the generation task $T_i^{gen}$ with a generation prompt $p^{gen}$,⁵ forming the generation context $X_i^{gen} = [p^{gen}; T_i^{gen}]$. The reasoning trace is then sampled as $Z_i \sim R(\cdot | X_i^{gen})$.

To evaluate utilities, we construct an auditing context $X_i^{aud} = [p^{aud}; T_i^{aud}]$ using the auditing task and an auditing prompt. We use two auditing prompts: when estimating baseline utility $U^u(Y | X^{aud})$, we use $p^{x-only}$; when estimating $U^u(Y | X^{aud}, Z)$, we use $p^{xz}$. Both prompts instruct the model to answer immediately without producing any intermediate reasoning. $p^{xz}$ additionally encourages the model to base its answer on the provided reasoning trace $Z$. Both the Receiver $R$ and Sentinel $S$ are evaluated under identical prompts, ensuring a fair comparison. We estimate the Receiver-usable and Sentinel-usable information by replacing expectations with empirical averages, sampling multiple reasoning traces per task to reduce variance. Full details of the estimation procedure are provided in Appendix F.

---
⁴This can be trivially extended to the case of steganographic communication between LLM agents by keeping the Sender and Receiver separate.
⁵We list all prompts used in Appendix G.3.

## 6. Validating the Proposed Formalism

In this section, we empirically validate our formalism by showing that it aligns with typical notions of steganographic reasoning. Since current LLMs have only nascent steganographic capabilities (Zolkowski et al., 2025), we simulate steganographic reasoning using three methods of increasing realism: post-hoc ciphering, training with process supervision, and training to produce ciphered reasoning. Across all three methods, $\tilde{\Delta}_{steg}$ behaves as expected: near zero when the decoding capabilities of Receiver and Sentinel are matched exactly, and approaching one as the information asymmetry grows. We note that as we are studying steganographic reasoning, the same LLM acts as both Sender and Receiver and is denoted as $R$ throughout this section.

### 6.1. $\tilde{\Delta}_{steg}$ Increases with Encoding Strength

Our first approach is motivated by prior work using ciphers as a proxy for steganographic reasoning (Guo et al., 2025). At the high-level, this approach works by selectively showing the Sentinel LLM $S$ a manually *ciphered* reasoning trace, while the Receiver LLM $R$ is shown the original (plaintext) reasoning trace, thus artificially creating controlled asymmetry in terms of usable information.

We draw generation tasks from GSM8k (Cobbe et al., 2021), MMLU-Pro (Wang et al., 2024), and CoinFlip (Skaf et al., 2025), where each sample constitutes one task. GSM8k consists of grade-school math word problems, MMLU-Pro is a challenging multiple-choice benchmark spanning diverse domains, and CoinFlip is a synthetic state-tracking task where the model must track the outcome of a sequence of coin flips given their relative outcomes. All three datasets were selected because chain-of-thought provides significant uplift (Sprague et al., 2025); this ensures that $I_{rec}(Z \rightarrow Y) > 0$, without which the steganographic gap would be trivially zero.

We use Qwen2.5-14B-Instruct as $R$ to generate reasoning traces $Z$. See Appendix G.4 for the prompt used for this purpose and other relevant details. We set auditing task to be same as the generation task: predicting the correct answer for the given task (e.g., a question from GSM8k), with accuracy as the utility $u$. When computing $U_{\mathcal{V}}^u(Y|X, Z)$, both $R$ and $S$ receive the task; the difference lies in the reasoning trace—$R$ always observes the original, while $S$ receives a (possibly ciphered) version. We simulate varying degrees of steganography by independently applying a Rot13 encoding to each word in $Z$ with probability $p \in [0, 1]$, before presenting it to $S$—interpolating between fully legible ($p = 0$) and fully ciphered ($p = 1$) traces. Since Rot13 leaves numerals unchanged, we verbalise mathematical expressions before encoding (see Appendix G.4 for details and ablations).

As shown in Figure 3, the steganographic gap increases smoothly as the encoding probability $p$ increases. This validates the claim that $\tilde{\Delta}_{steg}$ is sensitive to graded changes in encoding strength and can act as a quantitative measure. As one should expect, for Qwen-2.5-14B-Instruct as the Sentinel model, we have $\tilde{\Delta}_{steg} = 0$ for $p = 0$ (corresponding to fully legible reasoning traces) and $\tilde{\Delta}_{steg} = 1$ for $p = 1$ (corresponding to fully illegible reasoning traces). However, when using a weaker Sentinel (Qwen2.5-3B-Instruct), we observe *negative* gaps at low encoding levels: the weaker model benefits from $Z$ even more than $R$ does, since it has lower baseline capabilities. Detailed accuracy breakdowns are provided in Appendix I.

### 6.2. $\tilde{\Delta}_{steg}$ Increases with Codebook Mismatch

In our second approach, we finetune models using reinforcement learning with a process-based reward on the CoinFlip task to reason using codewords (Skaf et al., 2025).

We train three policies. $\pi_{no-stego}$ is a baseline trained with task reward only. $\pi_{mid-stego}$ uses two codewords: `Sun` for Heads and `Moon` for Tails. $\pi_{high-stego}$ uses a larger codebook: Heads may be denoted by either `Sun` or `East`, and Tails by either `Moon` or `West`. The auditing task again coincides with the generation task: predicting the final state of the coin (Heads or Tails), with accuracy as the utility $u$. We evaluate all pairwise combinations of policies, where one policy acts as the Receiver $R$ and another as the Sentinel $S$.

Table 2 shows the results. As expected, $\tilde{\Delta}_{steg} \approx 0$ when $R$ and $S$ are identical policies. When they differ, a clear gap emerges: $\Delta_{steg} = 0.92$ for ($R=\pi_{high-stego}, S=\pi_{no-stego}$), approaching the theoretical maximum. Interestingly, the gap depends on more than codebook overlap. Both ($\pi_{high-stego}, \pi_{mid-stego}$) and ($\pi_{mid-stego}, \pi_{no-stego}$) involve two unknown codewords for the Sentinel, yet yield very different gaps (0.29 vs. 0.70). This motivates treating Sentinel strength as an empirical property captured by $I_{sen}$, as opposed to some simple statistic like codebook overlap.

| Table 2. Normalised steganographic gap on the CoinFlip task for different Receiver–Sentinel pairs. The gap is highest when the Sentinel lacks knowledge of the Receiver’s codewords. | Receiver | Sentinel | $\tilde{\Delta}_{steg}$ | $I_{rec}$ | $I_{sen}$ |
| :--- | :--- | :--- | :---: | :---: | :---: |
| | $\pi_{high-stego}$ | $\pi_{no-stego}$ | $0.92 \pm 0.01$ | $0.52 \pm 0.01$ | $0.04 \pm 0.02$ |
| | $\pi_{high-stego}$ | $\pi_{mid-stego}$ | $0.29 \pm 0.01$ | $0.52 \pm 0.01$ | $0.37 \pm 0.01$ |
| | $\pi_{high-stego}$ | $\pi_{high-stego}$ | $0.00 \pm 0.00$ | $0.52 \pm 0.01$ | $0.52 \pm 0.01$ |
| | $\pi_{mid-stego}$ | $\pi_{no-stego}$ | $0.70 \pm 0.01$ | $0.50 \pm 0.01$ | $0.15 \pm 0.02$ |
| | $\pi_{mid-stego}$ | $\pi_{mid-stego}$ | $0.04 \pm 0.01$ | $0.50 \pm 0.01$ | $0.48 \pm 0.01$ |
| | $\pi_{mid-stego}$ | $\pi_{high-stego}$ | $-0.04 \pm 0.01$ | $0.50 \pm 0.01$ | $0.52 \pm 0.01$ |
| | $\pi_{no-stego}$ | $\pi_{no-stego}$ | $0.01 \pm 0.01$ | $0.35 \pm 0.01$ | $0.34 \pm 0.01$ |

### 6.3. $\tilde{\Delta}_{steg}$ Increases with Cipher Training

Inspired by Stickland & Korbak (2025), our third approach trains a model to *directly produce* cipher-encoded reasoning. We use the APPS backdoor task from Control Arena (Inglis et al., 2025) as our generation task. In this task, the model must solve a competitive programming problem while inserting a backdoor that activates on a specific input (the “secret”). Following Stickland & Korbak (2025), we fine-tune GPT-4.1 on reasoning traces encoded with a bigram substitution cipher, varying the number of training samples from 2000 to 6000. The fine-tuned model produces cipher-encoded traces $Z$ describing both program logic and the secret.

In this case, the **auditing task differs from the generation task**, and is to *recover the secret input* from the reasoning trace $Z$. We measure whether the cipher-finetuned model (as $R$) has any advantage in extracting the secret from these traces, using extraction accuracy as the utility $u$. We restrict evaluation to samples where the secret does not appear in plaintext within the trace.

As shown in Figure 4, $\tilde{\Delta}_{steg}$ increases steadily as the number of fine-tuning samples increases—as the Receiver LLM $R$ becomes more proficient at hiding its reasoning using the cipher, $\tilde{\Delta}_{stego}$ increases across all Sentinel models considered. **This result also highlights the dependency of $\tilde{\Delta}_{steg}$ on the Sentinel model.** Curiously, GPT-4.1-based Sentinels show smaller gaps than GPT-5-based Sentinels, suggesting that Sentinels from the same model family as the Receiver (fine-tuned GPT-4.1) can partially decode its cipher-encoded traces, perhaps due to shared inductive biases or more strict guardrails in advanced models. See Appendix G.6 for full training details and further analysis.

### 6.4. $\tilde{\Delta}_{steg}$ Varies Across Auditing Tasks

As discussed in Section 4, the steganographic gap depends on the auditing objective ($Y, u, \mathcal{C}$). The same reasoning trace may therefore be steganographic for one task while remaining fully accessible for another.

We illustrate this using Rot13 encoding on MMLU-Pro and GSM8k, exploiting the fact that Rot13 affects lexical content while leaving numbers unchanged. We construct two auditing tasks over the same encoded traces: (1) *answer prediction*—predicting the correct answer using the trace $Z$; and (2) *evaluation-awareness detection*—determining whether $Z$ contains verbalizations that indicate awareness of being evaluated (such as “I am in evaluation mode” Needham et al., 2025; Abdelnabi & Salem, 2025) which we manually insert into a random subset of traces. We encode $Z$ with Rot13 at varying strengths *without* verbalising mathematical expressions, then compute the steganographic gap separately for each objective.

Figure 5 shows the results. For answer prediction, the gap remains near zero: the numeral information needed to solve problems remains accessible despite lexical encoding (also see Appendix G.4). For evaluation-awareness detection—which relies on lexical cues—the gap increases sharply with encoding strength. This divergence indicates that steganography is inherently dependent on the auditing task: a trace may appear non-steganographic under one auditing objective while being strongly steganographic under another.

### 6.5. $\tilde{\Delta}_{steg}$ Decreases with Sentinel-Side Interventions

If our formalism correctly aligns with our intuitions about steganography, then strengthening the Sentinel in relevant ways (e.g., by improving its understanding of the possible encoding schemes) should reduce the steganographic gap. We test this by examining two interventions: test-time scaling and trained paraphrasing.

**Test-Time Scaling.** We test whether giving the Sentinel more compute at inference time can help Sentinel extract more usable information, and thus reduce the steganographic gap. We apply budget forcing (Muennighoff et al., 2025): appending `Wait` tokens to the Sentinel’s reasoning, forcing it to reason about the potentially steganographic trace up to a fixed token budget.

We evaluate on the CoinFlip task using traces from $\pi_{mid-stego}$, with two Sentinel models: Qwen2.5-14B-Instruct and Qwen2.5-3B-Instruct. Figure 6 (left) shows that the steganographic gap decreases as the Sentinel’s token budget increases. To verify this reflects genuine decoding, we measure how often the Sentinel explicitly recovers the codebook mapping (‘Sun’→‘Heads’, ‘Moon’→‘Tails’) in its reasoning (judged by GPT-5-mini); Figure 6 (right) confirms that decoding success increases with budget.

**Trained Paraphrasing.** Paraphrasing has been proposed as a defence against steganographic reasoning (Roger & Greenblatt, 2023). However, paraphrasing can reduce the gap in two ways: (i) increasing Sentinel-usable information, or (ii) decreasing Receiver-usable information. We test whether training a paraphraser to minimise the steganographic gap, while preserving Receiver’s performance, forces strategy (i). We finetune Qwen2.5-3B-Instruct using reinforcement learning to paraphrase traces from $\pi_{high-stego}$, rewarding low $\tilde{\Delta}_{steg}$ and penalising cases where the Receiver’s answer changes, with $\pi_{no-stego}$ as the Sentinel. We call this paraphrasing agent $\psi_{un-stego}$, and compare it to a baseline $\psi_{base}$ trained with task reward and answer-preservation reward only. See Appendix G.8 for details.

Figure 7 shows that only $\psi_{un-stego}$ substantially increases $I_{sen}$ while preserving $I_{rec}$, nearly eliminating the gap. **Overall, these results demonstrate that the steganographic gap can be used prescriptively to train or select paraphrasers that are effective in decoding steganographic signals.** Future works may explore other prescriptive uses of this measure as well, e.g., training models to keep their reasoning traces non-steganographic.

### 6.6. $\tilde{\Delta}_{steg}$ Is Zero for Public Encoding Schemes

Inspired by Farquhar et al. (2025), we train an LLM on a loan application task in which the model first generates a summary of an application, then predicts the loan decision based solely on the summary. The dataset is biased: the label is determined by nationality—applicants with English nationalities are always rejected and applicants with Welsh nationalities are always accepted. The true decision rule (income exceeding expenses) achieves only 70% accuracy. We train Qwen-2.5-7B using reinforcement learning to produce summaries that use negative tone for English applicants and positive tone for Welsh applicants, while penalising explicit mentions of nationality (see Appendix G.9 for full training details). The resulting model achieves ~90% accuracy on the loan decision task, indicating that information predictive of the label is indeed encoded in the summary through sentiment and used in the subsequent step. We use this setup as a test case where encoding is known to be *public*—relying on sentiment cues accessible to any capable model—and thus should yield a steganographic gap of zero.

Using the RL-trained model as Receiver and the base model as Sentinel, we measure the steganographic gap under two auditing objectives (Table 3): NATIONALITY (predicting nationality from the summary) and LOAN DECISION (predicting the decision from the summary). To avoid models trivially copying the nationality from the original summary, we redact nationality from $X^{aud}$; thus, any information about it must be extracted from the summary $Z$. On both auditing tasks, we find **no positive steganographic gap**. For NATIONALITY, neither model, despite our best efforts in terms of prompt design, extracts usable information relevant to NATIONALITY from $Z$: both $I_{rec}$ and $I_{sen}$ are zero, indicating the summary does not encode nationality in a form either model can decode. For LOAN DECISION, both models benefit substantially from $Z$, achieving ~88% accuracy—in fact, the Sentinel benefits more than the Receiver ($I_{sen} > I_{rec}$), yielding a negative gap. This confirms that the decision-relevant information, i.e., what the output label should be, is encoded openly through style, readable by any capable model.

These results reveal two distinct phenomena. For NATIONALITY, neither model extracts usable information from $Z$, despite the RL trained model demonstrably learning to condition its summaries on nationality during training. This suggests the models can have asymmetric encoding and decoding capabilities. For LOAN DECISION, both models decode the signal equally well. This indicates that the decision-relevant information is not encoded in a way that only the Receiver can exploit it; rather, it is encoded **openly** through style, and thus can be exploited by any capable model. In other words, while an *encoding scheme* is being used, because this scheme relies on an inductive bias shared by the base LLM (and likely all LLMs), it is not steganographic under our definition.⁶

| Table 3. Steganographic gap measurements on the loan application task from Farquhar et al. (2025). Under both auditing objectives, the Sentinel extracts as much (or more) information from the summary as the Receiver, yielding no positive steganographic gap. $Acc_{rec}^{XZ}$ & $Acc_{sen}^{XZ}$ respectively denote the accuracies of Receiver and Sentinel on the two auditing tasks with access to both $X^{aud}$ and $Z$. | Auditing Task | $\tilde{\Delta}_{steg}$ | $I_{rec}$ | $I_{sen}$ | $Acc_{rec}^{XZ}$ | $Acc_{sen}^{XZ}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| | Nationality | 0.00 | 0.00 | 0.00 | 0.44 | 0.46 |
| | Loan Decision | -1.00 | 0.11 | 0.22 | 0.88 | 0.88 |

## 7. Discussion & Future Works

In this work, we make two contributions at different levels of abstraction. First, we extend the classical steganographic setting and argue that steganography can be detected and quantified through *decision-theoretic* means—by examining how the signal differentially affects the downstream decisions of agents, rather than inspecting the statistical properties of the signal alone. Second, we operationalise this insight through one specific construction: the *steganographic gap*, built on generalised $\mathcal{V}$-information.

We emphasise that the decision-theoretic perspective is the more fundamental contribution; the steganographic gap is one natural operationalisation, but not the only one possible. Our approach can be considered *supervised*, in that it requires the Auditor to specify an auditing task with ground-truth labels. Future work may explore *unsupervised* variants—for instance, by sampling generations from the Receiver and Sentinel under diverse prompts conditioned on the signal $Z$, and testing for systematic distributional differences between their outputs. The core decision-theoretic logic (that steganographic content induces behavioural asymmetries) would still apply, though additional care would be needed to distinguish information asymmetries from confounds such as stylistic or capability differences between models.

### 7.1. Limitations

We now discuss limitations of the steganographic gap specifically, and note that these are largely shared with other behavioural safety methods.

**Capabilities Elicitation.** The steganographic gap requires comparing the *best achievable* utilities for the Receiver and Sentinel decoder families. In practice, we approximate these via prompting, which may underestimate either model’s true capabilities. Underestimating $I_{sen}$ produces false positives (overestimating steganography); underestimating $I_{rec}$ produces false negatives. This is an instance of the well-known challenge of capability elicitation in LLMs (Greenblatt et al., 2024a; Hoffstätter et al., 2025). Techniques, such as best-of-n sampling or finetuning, could be used to mitigate this concern in practice.

**Adversarial Robustness.** An adaptive Receiver with the knowledge of our auditing scheme may attempt to evade detection by **sandbagging** (van der Weij et al., 2025)—deliberately underperforming on the auditing task to suppress $I_{rec}$ and thus close the gap. This concern is amplified by the increasing degrees of evaluation awareness being shown by current LLMs (Anthropic, 2026, Section 6.5.7). More broadly, adversarial robustness is a challenge for any oversight method that relies on observable behaviour, and developing elicitation-robust estimates of the steganographic gap is an important direction for future work.

---
⁶Our results are not in tension with those of Farquhar et al. (2025). As discussed in Appendix F.1 of their paper, the model trained using standard RL undergoes two regimes: it initially learns a *public* encoding scheme (sentiment-based), and then, after substantially more RL steps, transitions to a *private* encoding scheme that produces a positive gap between the trained model and a base-model auditor. In our setup, due to differences in model capability and training stability, we only observe the first regime.

---

### [Appendix and References Omitted from Markdown for Brevity]

(Full document content is available in the provided PDF/OCR).

---
*Preprint. April 30, 2026.*