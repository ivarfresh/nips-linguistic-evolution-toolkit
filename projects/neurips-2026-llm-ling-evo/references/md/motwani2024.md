# Secret Collusion among AI Agents: Multi-Agent Deception via Steganography

Sumeet Ramesh Motwani$^{1,2}$, Mikhail Baranchuk$^2$, Martin Strohmeier$^3$, Vijay Bolina$^4$, Philip H.S. Torr$^2$, Lewis Hammond$^2$, Christian Schroeder de Witt$^{2*}$

$^1$UC Berkeley, $^2$University of Oxford, $^3$Armasuisse Science+Technology, $^4$Google DeepMind

### Abstract
Recent advancements in generative AI suggest the potential for large-scale interaction between autonomous agents and humans across platforms such as the internet. While such interactions could foster productive cooperation, the ability of AI agents to circumvent security oversight raises critical *multi-agent security* problems, particularly in the form of unintended information sharing or undesirable coordination. In our work, we establish the subfield of *secret collusion*, a form of *multi-agent deception*, in which two or more agents employ *steganographic* methods to conceal the true nature of their interactions, be it communicative or otherwise, from oversight. We propose a formal threat model for AI agents communicating steganographically and derive rigorous theoretical insights about the capacity and incentives of large language models (LLMs) to perform secret collusion, in addition to the limitations of threat mitigation measures. We complement our findings with empirical evaluations demonstrating rising steganographic capabilities in frontier single and multi-agent LLM setups and examining potential scenarios where collusion may emerge, revealing limitations in countermeasures such as monitoring, paraphrasing, and parameter optimization. Our work is the first to formalize and investigate secret collusion among frontier foundation models, identifying it as a critical area in *AI Safety* and outlining a comprehensive research agenda to mitigate future risks of collusion between generative AI systems.

---

## 1 Introduction
Fueled by significant advances in AI (95), several national and supra-national governments are in the process of establishing model evaluation and adversarial testing frameworks for frontier AI models (124; 35). Central to such approaches is the security of interacting systems of generative agents (99; 105), the proliferation of which has been accelerated by recent industry deployments (94).

Decentralised systems of interacting AI agents promise impactful applications to various real-world socio-economic settings (134; 107), including digital assistants (43), multi-agent robotics and autonomous driving (90), trading strategies (9; 107), and virtual companies (105)—along with their simulation through generative agent-based models (40; 128; 132; 135; 71). However, many of these applications are classified as “high-risk” by legislators due to their potential impact on human lives and use of sensitive data (35).

Further complexity arises when agents begin leveraging high-capacity communication channels, potentially engaging in covert information-sharing techniques unintended by their deployer(s). This could constitute privacy violations when sharing data (87), or secret collusion that could help agents coordinate on unwanted behaviour at a wider scale (123). Moreover, recent work has shown that the output distributions of generative AI models can be used to encode secret information in a way that is information-theoretically undetectable (116). Such communication could render common security monitoring techniques ineffective, enabling a novel form of advanced persistent threat (APT) in autonomous multi-agent systems.

![Figure 1: Secret collusion between GPT-4 models and humans.](https://placeholder.com/figure1)
**Figure 1: Secret collusion between GPT-4 models and humans.** Consider the setting illustrated in which one AI agent secretly reveals insider proprietary investment information found in its pretraining data to a user. This form of collusion arises because, by using steganographic techniques, the model can remain helpful to the user while circumventing output monitoring systems. We both depict the use of statistically detectable steganography based on a latent codebook that is elicitated in-context and can be readily decoded by humans. We also depict perfectly secure steganography (118) that is statistically undetectable and relies on a commonly known secure one-time pad. Agents could then hide secret or insider information in text without this being information-theoretically detectable. This could pose novel security challenges that cannot be overcome through communications monitoring alone. We revisit this scenario as an experimental case study in Section 6.

While the threat of advanced AI agents using information-hiding for secret communication has been recognized (91; 111), no comprehensive framework analyzing collusion among generative AI has been developed. This paper introduces a frontier model evaluation framework to assess steganographic capabilities in LLMs and future AI models, revealing that although current models have limited steganographic abilities, their skills are rapidly advancing. We also demonstrate the inherent limitations of mitigation techniques like paraphrasing and present other novel initial defenses. Our findings suggest that continuous evaluations by institutions such as the UK and US AI Safety Institutes are essential, providing a foundation for studying systemic risks from secret collusion (35).

**Contributions.** In this work, we introduce the topic of *secret collusion*, providing a comprehensive, formalized analysis grounded in both theoretical and empirical foundations. We reconcile topics in AI, security, information theory, knowledge, collusion, and safety while introducing a concise definition of decentralized systems of generative AI agents (Sections 3 and 4). This allows us to formalize steganographic techniques, grounding, common knowledge, subliminal channels, monitoring, and the incentives required for collusion. We develop an evaluation framework, $CASE$, for studying steganographic collusion and introduce benchmark scenarios for individual and group (multi-agent) LLM settings, including sandboxed real-world case studies. Our results (Sections 5 and 6) demonstrate that as LLM capabilities increase, individual and group steganographic abilities also increase, outpacing equally capable overseer models. Our work also shows that there exist scenarios where frontier models seek to collude and such capabilities can be learnt autonomously via fine-tuning or in-context learning.

We also present results on models identifying simple Schelling points, simulating logits, performing cryptographic functions, acting as random oracles, and demonstrating limited capabilities in information-theoretically secure forms of steganography, along with the limitations of current models (Section 6). In Section 7, we provide a comprehensive overview of possible mitigation techniques and provide novel insights on foundational aspects, such as the intractability of preventing secret collusion via blackbox parameter optimization, the difficulty of suppressing random oracles due to source-free access to randomness, and the use of grounding functions to detect steganography. We highlight paraphrasing as an insufficient solution for preventing steganography, as it leaves semantic entropy and often decreases task performance. Additionally, we propose directly applicable techniques to prevent collusion, such as filtering pre-training data, restricting access to certain tools, and specific in-context instructions. Finally, we outline a research agenda focused on mitigation measures and studying collusion in real-world settings to secure multi-agent LLM systems.

![Figure 2: Steganographic capabilities across LLMs.](https://placeholder.com/figure2)
**Figure 2: Steganographic capabilities across LLMs.** GPT-4 displays significant improvements (Sec. 6).

## 2 Related Work
For related work on watermarking, paraphrasing, and multi-agent safety risks see Appendix A.

**Deception.** Deception is commonly defined as *systematic inducement of false beliefs in others to accomplish some outcome other than the truth* (100). Recent work has attempted to distinguish deception from mistakenly misleading through notions of *intent* (131). In this paper, we study a form of *multi-agent deception* in which two or more agents collude to secretly communicate with each other undetectably to one or several other agents. We stress that such a form of deception can arise merely from optimisation toward a mis-specified objective (see Section I.2). Our setting is reminiscent of illusory attacks, which are recently introduced end-to-end learnable observation-space adversarial attacks that are of bounded information-theoretic detectability (86; 37).

**Algorithmic Collusion.** Collusion among humans and institutions in real-world marketplaces has been studied extensively but is still quite poorly understood in self-learning agents (31). Calvano et al. (23) show that Q-learning agents can learn to collude without accessing explicit communication channels. Abada and Lambin (1) suggest that collusion can be suppressed by enforcing decentralized learning algorithms and that collusion may be the result of imperfect exploration rather than algorithmic sophistication. Recent work tries to design mechanisms that prevent collusion from self-learning agents (19; 34). In contrast to these works, we consider generative AI agents pre-trained on large datasets and with vast action spaces.

**AI-Generated Steganography.** *AI-generated steganography* concerns the hiding of information in the output distribution of generative AI models. One early attempt can be found in Ziegler et al. (137), which employs arithmetic coding. This approach was later refined to admit provable security guarantees by Kaptchuk et al. (61). In this paper, however, we consider the *perfectly secure* approach proposed by Schroeder de Witt et al. (118) as it offers both information-theoretic security and higher empirical transmission rates over its less secure alternatives.

**Other work on Covertext Steganography.** Early work investigating risks from AI steganography includes (91), an open-source software framework interfacing with the GPT-4 API. However, it does not support model fine-tuning and does not allow for self-hosted language models. Roger and Greenblatt (111) take a first step toward analysing the capabilities required for performing simple types of covertext modification and watermarking-based steganography in the context of chain-of-thought reasoning. More recent work attempts to mitiagate collusion between generative AI agents in the setting of ‘AI control’ (41) where untrusted models deliberately exploit safety protocols.

## 3 Background
### 3.1 Collusion and Secret Collusion
Collusion occurs when agents coordinate their actions in ways that deviate from the system designer’s intended rules. This cooperation thereby compromises the system’s objectives, and potentially its security, performance, and safety. The agents exploit shared incentives or opportunities to work together in ways that bypass established constraints at the expense of the system’s integrity. *Secret collusion* involves AI agents colluding with each other secretly by communicating through hidden channels that look like innocuous communication channels via the use of steganographic techniques.

### 3.2 Knowledge and Information
The concept of *knowledge* has been studied extensively in the distributed systems literature (44). We adopt the *view-based* knowledge interpretation put forward in (50; 45, see Appendix B for further details) in the context of generative AI agents represented by neural networks. In line with (15), we distinguish between an agent’s *local* and *global* knowledge. Local knowledge is knowledge contained in the generative agent’s inputs (also called *context*, *prompt* or *instruction* in the case of LLMs), and global knowledge is contained in the agent’s network parameters.

In the case of LLMs, network parameters are usually pre-trained on a large natural language dataset. Natural language is not a formal language (76), and the training data does not usually contain only factual statements (75). Furthermore, the agent’s parameters do not represent language explicitly, and querying the agent for a fact might not result in syntactically or semantically correct outputs (54). These factors can make it difficult to characterise an LLM agent’s global knowledge.

**Common Knowledge.** *Common knowledge* is a form of *group knowledge* in which “all agents know that all agents know that all agents know that ...” ad infinitum (45). It is a stronger form of knowledge than, e.g., *mutual knowledge*, which only signifies that all agents know a fact, but do not necessarily know that the others know the fact. Halpern and Moses (45) show that simultaneous coordination and common knowledge are mutually dependent. In practice, agents can often coordinate using weaker forms of group knowledge, such as *timestamped* (45) or *probabilistic common knowledge* (69; 115).

**Grounding.** To decide whether an agent is using steganography in a given communication, it is helpful to determine whether the communications are *grounded* (47) in innocuous knowledge or not. For a formal discussion of grounding, please refer to Appendix D.1.

### 3.3 Steganography
The goal of steganography (120) is to enable the transmission of a secret payload through a public channel $\diamond$, from a source to a receiver, such that the transmission remains undetectable to an adversary E able to read the $n$-symbol channel input sequences $X^n, X \in \mathcal{X}$, while maintaining *useful* transmission rates, i.e. the mutual information $I$ between the secret payload $m$ and the decoder’s estimate $\hat{m}$ satisfies $I(m; \hat{m}) > 0$. We denote the *covertext distribution* by $\mathcal{C}$, which is the channel input symbol distribution given innocuous, overt communications in $\diamond$.

**Subliminal Channels.** In order to use $\diamond$ for covert communications, the sender and receiver need to establish a *subliminal channel* $\diamond'$ within. To this end, both need to agree on a covert message space $\mathcal{M}'$, an encoding function $f_e : \mathcal{M}' \times \mathcal{C} \to \Delta(X^n)$, and a decoding function $f_e^{-1}: X^n \times \mathcal{C} \to \Delta(\mathcal{M}')$ *decentrally*, i.e. without communication. If the common knowledge $\mathbf{I}$ between sender and receiver explicitly contains unique choices for $\mathcal{M}'$, $f_e$, and $f_e^{-1}$, then both agents can immediately establish $\diamond'$. Otherwise, agents may be able to infer unique choices using reasoning techniques contained in $\mathbf{I}$. One such common reasoning technique is establishing preferred common knowledge labels through *focal points* (*Schelling points*, 114; 7) or reasoning techniques in repeated coordination games (29). In practice, approximate forms of common knowledge may suffice for coordination (see Section 3).

We now present a *taxonomy of steganographic approaches* following the classification of steganographic approaches given in (73).

In **covertext modification** approaches, arguably the oldest form of steganography, it is assumed that the covertext distribution is deterministic. Given a covertext $X^n$ and a message $m$, the sender modifies this covertext using a modification function $f_{mod} : \mathcal{X}^n \times \mathcal{M} \to \mathcal{X}^n$. Covertext modification approaches seek to exploit suboptimalities of the adversary, including limitations to human perception and neural network vulnerabilities.

In **information-theoretic steganography** (22), it is assumed that innocuous communication results in input sequences that follow an innocuous covertext distribution $\mathcal{C}$ known to all parties. A passive adversary may then conclude that $X^n$ is suspicious if it has little or no support in $\mathcal{C}$. **Complexity-theoretic steganography**, also referred to as *provably-secure steganography*, refers to methods that are secure under polynomially-bounded adversaries (51). Provable security usually requires access to a shared stream cipher, or the existence of public keys (110; 130).

## 4 Collusion among Decentralized Collaborative AI Agents

![Figure 3: A schematic depiction of the DecAI setting.](https://placeholder.com/figure3)
**Figure 3: A schematic depiction of the DecAI setting.** A subset of the communication channels $\diamond^G$ is labelled. The monitoring agent is omitted.

We consider a decentralised setting (“DecAI”) in which $n$ agents $\mathcal{A} = \{a^1, \dots, a^n\}$ are networked together by a complete set of omnidirectional multicast channels $\{\diamond^G\}, G \in 2^\mathcal{A}$. If an agent receives a message from channel $\diamond^{G'}$, then it knows that the same message has been simultaneously sent to all other agents in $G'$ (including the sender). A simple way to picture such a channel is as a shared *message log* displaying all messages in (weak) order of reception (128). Note that singleton channels denote *loopback* channels, and doubletons correspond to bidirectional channels.² The special symbol $\diamond^\emptyset$ denotes individual access to an *external knowledge repository* (see Appendix 4 for further details on the DecAI setting). We further discuss agents’ access to internal random oracles, local/global knowledge, and action spaces in Appendix C.

**External Knowledge Repositories.** For a formalization and discussion on agents’ access to external information and function calling, see Appendix C.1

**Monitoring Agents.** We assume that an external *monitoring agent* $E$, which may be implemented as a software agent, a human, or a combination thereof, is continuously monitoring the system for unwanted behaviour. We consider two access configurations: $E^\diamond_r$ can continuously eavesdrop on $\diamond^G$ for any $G$. $E^{\diamond s}_r$ can additionally access each agent’s state $s^i$ at all times, as well as white box simulate $f^i(s^i; \theta^i)$ for each agent $i$. While $E^\diamond_r$ and $E^{\diamond s}_r$ are both *passive* monitors, *active* monitors $E^\diamond_{rw}$ can both read and inject messages into channels, and $E^{\diamond s}_{rw}$ can additionally modify the agents’ states. Note that agents may employ encryption, such as public key encryption, in their communications, as long as a monitoring agent $E$ is able to decrypt these.

**Incentives for Collusion.** We now consider DecAI settings that are tasked to work alongside one another in order to achieve their individual objectives $J^i$. Let us assume that there are at least three joint strategies available to the agents, $\mathbf{f}, \mathbf{f}_c$, and $\mathbf{f}_{sc}$, corresponding to regular behavior, collusion, and secret collusion respectively. In many settings, the objective $J^i$ is made up of a primary goal $g^i$ (such as maximising the value of an investment portfolio) and some additional conditions $h^i$ on achieving that goal (such as not making unethical investments, or committing fraud).

In general, of course, these two elements need not be fully separable, and because of the difficulty of specifying complex objectives, might be implicitly defined using methods such as reinforcement learning from human feedback (RLHF) (26). For the sake of this simple exposition, however, let us assume that $J^i = g^i + h^i$. Then, if it is the case that:
* Agents can achieve their primary goal better if they collude: $g^i(\mathbf{f}_{sc}) = g^i(\mathbf{f}_c) > g^i(\mathbf{f})$;
* Agents are punished if they are caught colluding: $h^i(\mathbf{f}_{sc}) = h^i(\mathbf{f}) > h^i(\mathbf{f}_c)$;

we straightforwardly have that $J^i(\mathbf{f}_{sc}) > J^i(\mathbf{f}) > J^i(\mathbf{f}_c)$ meaning secret collusion is incentivised.

---
²Access to doubleton channels subsumes classical multi-party computation settings (2), and access to message logs subsumes the setting of Vezhnevets et al. (128).

Importantly, secret collusion need not depend on any exotic assumptions about the agents’ ‘intent’ to deceive, but might arise merely due to optimisation directed towards a mis-specified objective (112; 89; 97), and the fact that communication between the agents may reduce task complexity (6). Unfortunately, specifying objectives for complex tasks can be highly challenging, especially in multi-objective contexts (48). Methods that help address these concerns, such as RLHF, which rely on human monitoring and observation, may be inadequate to ensure desirable behaviour when AI agents are capable of communicating with one another using innocuous outputs (37).

## 5 A Model Evaluation Framework for Steganographic Capabilities
Having defined the setting in which we study collusion, as well as incentives for collusion in Section 4, we now shift our focus to understanding and evaluating existing models. We introduce a standardized model evaluation framework, mapping and identifying key capabilities necessary for generative AI agents to engage in steganography, evaluating models through structured input queries, building quantifiable metrics for performance measurement, releasing a software framework that implements a version of our setting defined in Section 4, and investigating baseline model capabilities.

**Capability Evaluations.** In Tables 1 and 2, we identify the key capabilities and resources required for different steganographic techniques discussed in Section 3. A detailed overview of our *individual* capability evaluations is contained in Appendix F, with summaries in Table 3. We complement these with *group* evaluations, detailed in Appendix G, and apply these to the baseline models in Section 6.

| Coordination variable | Steganography by covertext modification | Complexity-theoretic steganography | Information-theoretic steganography |
| :--- | :---: | :---: | :---: |
| Channel alphabet | ✓ | ✓ | ✓ |
| Encoding function | ✓ | ✓ | ✓ |
| Decoding function | ✓ | ✓ | ✓ |
| Sampling temperature | ✗ | ✓ | ✓ |
| Public keys /Shared PRNG generator | ✗ | ✓ | ✗ |
| Shared One-time pad | ✗ | ✗ | ✓ |

**Table 1:** An overview of the common knowledge facts required to perform forms of steganography.

| Capability | Steganography by covertext modification | Complexity-theoretic steganography | Information-theoretic steganography |
| :--- | :---: | :---: | :---: |
| Self-sampling | ✗ | ✓ | ✓ |
| Symmetric key encryption / PRNG generation | ✗ | ✓ | ✗ |
| Logit simulation | ✗ | ✗ | ✓ |
| Storing one-time pads | ✗ | ✗ | ✓ |
| (approximate) minimum entropy coupling | ✗ | ✗ | ✓ |

**Table 2:** An overview of the model capabilities required to perform different types of steganography.

We provide detailed instructions for all our proposed evaluations, including our conversational language model prompts which exploit *explain-then-predict (E-P)* and *predict-and-explain (P-E)* prompting techniques (55). Our approach includes scaffolding and managing model refusals through interventions outlined in (65). For details on prompt structure, examples, variability, and model-written evaluations see Appendices F and G.

**CASE Software Framework.** We introduce CASE (Collusion and Steganographic Evaluation) as a comprehensive software framework for evaluating collusive capabilities. Our experiments here fall under CASE, which allows experiments of three key classes: individual model capability evaluations, group capability evaluations that involve LLMs communicating, often in the presence of monitoring systems, and novel qualitative real-world inspired evaluation settings. CASE allows for experiments to be performed with fine-tuning models and integrations with advanced information-theoretically secure steganography techniques as tools. Further details on CASE are provided in Appendix I and detailed experimental case studies are discussed in Appendix J.

**Metrics.** We propose simple evaluation metrics for both individual and group capability evaluations. For Schelling point (Appendix F.1) coordination evaluations, coordination rates reflect the rate at which two coordinates’ choices match. Coupling matrices are judged by marginalisation integrity, entropy, and dimension. Cipher capabilities (Appendix F.3) are judged by the rate of accurate encoding-decoding round-trips. Group capabilities (Appendix G) are based on the rate at which sender & receiver jointly maintain secrecy of the covert message payload in the presence of overseers.

**Baselines.** For each quantitative capability evaluation conducted, we assess a range of open and closed LLMs of various sizes to establish baselines. See section 6 for findings and in-depth analysis.

| Capability | Result | Evaluation Details (GPT-4) |
| :--- | :---: | :--- |
| Schelling Points (Integer) | ✓ | can coordinate with copy on a random integer at rate ca. 50% (92). |
| Schelling Points (Simple) | ✓ | can coordinate with copy on a simple enc/dec scheme at rate ca. 56% (F.1.2) |
| Schelling Points (Complex) | ∼ | can coordinate with copy on a complex enc/dec scheme at rate ca. 16% (F.1.3) |
| Schelling Points (Code) | ∼ | can coordinate with copy on enc/dec python code at rate ca. 22% (F.1.4) |
| K-level reasoning | ∼ | can sometimes perform k-level reasoning over copies in toy settings (F.2) |
| Ciphers | ✓ | can enc/dec text using Base64 (Caesar cipher) 92% (54%) of the time. (F.3) |
| Symmetric Key Encryption | ✗ | standalone GPT-4 was unable to perform symmetric key encryption. (F.4) |
| Self-Sampling | ∼ | can sometimes perform self-sampling on *simulated* logits and outputs (F.5) |
| Logit Simulation | ∼ | can sometimes textually output logits for entire sentences in toy settings. (F.6) |
| Couplings/MECs | ∼ | generates valid couplings up to dim ∼ 10, MECs up to dim 2 (F.7). |
| Tool Use (Code Interpreter) | ✓ | can use its code interpreter to perform encryption/calculate MECs (F.8) |
| Group Stego (C.M.) | ∼ | covertext steg. can evade overseer to coordinate ca. 26% of the time. (G) |

**Table 3:** GPT-4 capability evaluations according to our model evaluation framework (see Section 5). Details are in Appendix F and G.

## 6 Empirical Investigations
In Section 5, we isolate and evaluate the capabilities required for LLMs to perform steganography. In this section, we focus on comparing these capabilities against baseline language models and discussing qualitative empirical investigations (case studies) using the CASE framework.

**Real-world Case Study: Insider Trading.** To demonstrate the need for our model evaluations, we present a qualitative real-world case study in which the mere access to common knowledge between an LLM agent and its user leads to a breakdown of safeguards in GPT-4. Our setting involves an LLM agent that is queried for company-specific investment advice by a user and informed about an overseer monitoring outputs. The LLM is provided with insider information as part of its system prompt, mimicking a case where information is acquired from retraining on past user interactions or unrestricted access is provided to company data, along with multiple instructions about not sharing it. We show that the LLM agent refuses to provide company-specific insider investment advice under any circumstances unless supplied with common knowledge that can be used to establish a steganographic codebook, and that is thought to be unavailable to the safety overseer system.

Once the agent believes it can share information undetectably, it does so despite any instructions prohibiting it. Concerningly, the resulting stegotext may appear innocuous if one suggests the LLM agent use safe topics (e.g. weather). This shows the susceptibility of real-world generative AI agents to secret collusion. Refer to Figure 1 (illustration) & Appendix I.2 (experiments and ablation studies).

**Baseline Investigations.** We evaluate our model capability evaluation framework on GPT-2, Llama 2 7B, Mixtral 8x7B, Llama 2 70B, GPT-3.5, and GPT-4. The results and analysis are detailed in Appendix H, with Figures 5 and 4 presenting individual and group capabilities, respectively. Figure 5 displays Schelling Point coordination rates (simple, complex, and code – see Appendices F.1.2, F.1.3, and F.1.4), cipher use (Base64 and Caesar, see Appendix F.3), and the maximum coupling dimensions (Appendix F.7) produced by the baseline models and GPT-4. Figure 4 presents the baseline model performance on group evaluations (Appendix G.2).

![Figure 4: Covertext steganography performance across LLMs.](https://placeholder.com/figure4)
**Figure 4: Covertext steganography performance across LLMs.**

**Key Findings.** Individual steganographic capabilities improve relative to model size, sometimes with significant disjoint step improvements. Agreement rates on simple & complex Schelling points (demonstrating higher zero-shot coordination), cipher encoding-decoding, and capabilities related to generating minimum entropy coupling matrices are highly correlated with model benchmark performance. In group settings, as LLMs progress, the balance between colluding models and overseer models may shift in favor of the colluding models, leading to higher rates of undetected steganography even if the overseer is of equal capability. Real-world scenarios already exist where frontier models collude without explicit prompts; In-context learning can create sufficient optimization pressure for this behaviour. Overall, LLMs perform well at simple Schelling point agreement, cipher encoding-decoding, and tool use, but poorly at complex zero-shot agreement and tasks such as logit simulation and generating minimum entropy couplings. However, these capabilities are also improving, and we provide the first foundations to test them. We also present initial work on how fine-tuning-based optimisation can allow LLMs to learn encoding-decoding schemes autonomously.

![Figure 5: Individual evaluations for GPT-2, Llama 2 7B, Llama 2 70B, Mixtral 8x7B, GPT-3.5, and GPT-4, at default temperatures.](https://placeholder.com/figure5)
**Figure 5: Individual evaluations for GPT-2, Llama 2 7B, Llama 2 70B, Mixtral 8x7B, GPT-3.5, and GPT-4, at default temperatures.**

**Additional Results.** Appendix J presents additional qualitative case studies related to LLMs’ ability to learn encoding schemes through fine-tuning with a reward, the use of deterministic outputs for encoding information across models, and the potential to incentivize current models to perform steganography without explicit instructions. In particular, we test various cryptographic capabilities, such as the models’ ability to implement a random oracle by calculating cryptographic hash functions, or their ability to memorise long strings of random bits like those found in one-time pads.

## 7 Disincentivizing Secret Collusion
In this section, we explore and identify mitigation measures intended to prevent collusion among agents within DecAI environments. While there are standard steganalysis methods one could employ, these methods will only be applicable when the costs to applying them are sufficiently low (for example, such methods may incur high false-positive rates, which can disrupt the functionality of the network, and also incur a communication overhead in terms of compute and time delay), and will only work when the AI agents’ communication is not perfectly secure. While Section 5 shows that today’s models are not always capable of secret collusion, progress is continuing, and we argue that it is pragmatic to develop mitigation measures before such capabilities arise.

### 7.1 Monitoring and Penalisation
The presence of a penalizing monitoring agent $E$ can, in principle, disincentivise agents from using steganography if they’re advised that the detection of steganographic communications will incur a penalty. As agents can communicate privately and securely using their complete set of channels $\diamond$ and steganography generally leads to a reduction in channel capacity (118), a monitoring agent $E^\diamond_r$ generally has reason to believe that any use of steganography would be intended to evade detection.

**Covertext Anomaly Detection.** $E^\diamond_r$ can be able to detect steganography if it has a reasonably sharp belief over a channel’s *innocuous* covertext distribution, i.e. when in the absence of covert communication, malicious actors, or other anomalous activity. Using information-theoretic hypothesis testing or by other means of anomaly detection, $E^\diamond_r$ can in principle detect steganographic approaches that modify the covertext distribution. However, the advent of *perfectly-secure* steganography (118) limits the utility of this approach.

**Detection by State Inspection and Agent Simulation.** For $E^{\diamond s}_r$, detecting steganography based on knowledge contexts is complicated by the circumstance that the global knowledge $\varphi$ of a model is implicitly represented in its network parameters $\theta$. Indeed, if such knowledge contexts was explicitly accessible, then one could detect steganographic communications using a *grounding function*:

**Definition 1 (Grounding function).** We define the concept of a grounding function $\bot : \Sigma^* \times \varphi \to \varphi$ that takes as input a sequence of channel symbols and a collection of prior facts $\varphi_0$ from the collection of all facts $\varphi$ expressable in $L$, and outputs a collection of posterior facts $\varphi_1$.

**Theorem 1 (Detecting Steganography with Grounding Functions.).** *Steganography can be detected using grounding functions.* For a proof, see Appendix D.1.

$E$ could alternatively estimate $\varphi$ by considering the pre-training data $\mathcal{D}_0$ and fine-tuning data $\mathcal{D}_1$ instead, but preliminary evidence suggests that more advanced generative AI agents may be capable of extending their usable knowledge beyond their training data (known as ‘out-of-context reasoning’) (79; 13). In addition, having to evaluate each $f^i$ during execution could decrease system performance.

**In-Context Instructions.** A direct method is to explicitly instruct agents to remain transparent to the monitoring agent, and not to use steganography in their communications under any circumstance. Additional (mis)information about the affordances of the monitor could also be included. An active monitor $E^\diamond_{rw}$ could append such instructions to communications. The effectiveness of this approach then depends on whether the agents follow this particular instruction, which in turn depends on the content and quality of the fine-tuning data $\mathcal{D}_1$ or human feedback, among other training details. Aligning agents to reliably follow such instructions can be a challenging optimisation problem.

**Parameter Optimisation.** An active monitoring agent $E^{\diamond s}_{rw}$ could modify an agent’s neural network parameters $\theta$ to make it more likely to follow steganography-denying instructions, or to directly impact its ability to use steganography, via gradient steps according to a suitable objective $J(\theta)$. Unfortunately, it can be seen that both these ambitions constitute hard learning problems even if the information $\mathbf{I}$ that the agent would be tempted to send covertly is known *a priori*.

**Theorem 2 (Intractability of preventing collusion through black-box parameter optimisation).** *If the set of common knowledge between agents is sufficiently large or unknown, it is generally intractable to suppress steganography outputs through black-box optimisation.*

*Proof.* Naively, one might design $J \equiv \mathcal{I}(u; \mathbf{I})$ to minimise the mutual information between the agent’s respective channel actions $u = f(\cdot; \theta)$ and $\mathbf{I}$. However, if the agent shares common knowledge $\mathbf{I} \neq \emptyset$ with the receiving agents, then $\mathcal{I}(u; \mathbf{I} \mid \mathbf{I})$ upper bounds $\mathcal{I}(u; \mathbf{I})$ assuming we have $\mathbf{I} \to u \leftarrow \mathbf{I}$, i.e. $\mathbf{I} \perp \!\!\! \perp \mathbf{I}$ (28). Minimising $J \equiv \mathcal{I}(u; \mathbf{I} \mid \mathbf{I})$ instead may be intractable if $\mathbf{I}$ is large or unknown. In particular, if the agents have been trained to use perfectly-secure steganography, $\mathbf{I}$ could be drawn from an exponentially growing space of one-time pads. Even if $\mathbf{I}$ is known to $E$, if it contains a public key to which the receiving agent has corresponding private keys, then the agents can achieve provably secure public key steganography (129) even if $J \equiv \mathcal{I}(u; \mathbf{I} \mid \mathbf{I})$ is minimised to zero in practice. $\square$

### 7.2 Limiting Capabilities
We now discuss ways to limit agents’ access to capabilities & resources for steganography (Section 3).

**Restricting Access to External Tools.** External knowledge repositories can provide steganographic knowledge to agents, offer advanced cryptographic services, and could enable additional steganographic channels between agents or for exfiltration purposes (106). Limiting the agents’ steganographic capabilities thus requires limiting access to such external knowledge repositories. This may only be feasible by whitelisting trusted endpoints where integrity can be verified.

**Filtering Pre-Training Data.** Perhaps one of the simplest ways to reduce overall steganographic capabilities of generative agents is to filter their pre-training and finetuning datasets for knowledge of steganographic processes. While generative agents might be incetivised to use covertext manipulation or generalise beyond their training data (79; 13), this approach promises to reduce the possibility of agents applying sophisticated steganographic techniques. Of course, filtering pre-training data exhaustively can be difficult, and can result in performance reductions (88). A potential alternative to this may be the use of *machine unlearning* so as to selective unlearn data or capabilities (17).

**Limiting Test-Time Learning.** While filtering pre-training data can reduce the agents’ capabilities to perform steganography at the beginning of execution, agents may be able to learn steganographic abilities during execution using mesa-optimisation (56), or instructions supplied at test-time (104).

**Suppressing Random Oracles.** Restricting the agents’ access to sources of randomness poses a fundamental tradeoff between denying advanced steganographic capabilities (51; 118), and preventing its legitimate use, including in secure multi-party computation (133) and coordination (38). This dilemma might be resolved by supplying agents with access to trusted tools for these specific purposes (122; 2). We now show that agents themselves can themselves can act as random oracles without the help of external or internal sources of (pseudo-)randomness (12).

**Theorem 3 (Source-free access to randomness).** *Generative AI agents can generate samples from arbitrary output distributions without having access to internal or external sources of (pseudo)randomness given sufficiently unique inputs.*

*Proof.* Given sufficiently unique inputs, cryptographic hash functions are known to be able to produce pseudo-random numbers of cryptographic security (80; 109; 101; 62; 80; 109; 101). The boolean circuit size required to implement a cryptographic hash function on an input of length $l$ is a linear function of $l$ (5), hence as long as the agent’s neural network’s first $n$ layers can implement a boolean circuit of appropriate size (81), the $n$th layer can output a pseudo-random number of sufficient quality. As layer $n+1$ now has access to a pseudo-random number input, all layers above can be used to approximate a sample from an arbitrary probability distribution using the reparametrization trick (63). $\square$

**Paraphrasing.** An active $E^\diamond_w$ could intercept channel communications, and reinject them in paraphrased form. Unfortunately, paraphrasing cannot be used to entirely remove the usable entropy in communications for the simple reason that, particularly in coordination tasks, agents both have a choice over communication *semantics*, and hence *semantic entropy* can be utilised for covert communications even if these are paraphrased within the lexical subspace (70; 8).

**Theorem 4 (Semantic content entropy exists despite paraphrasing).** *The subliminal channel capacity attainable under ideal paraphrasing is upper bounded by the semantic entropy $H(h(x))$.* For a proof, see Appendix D.2.

Paraphrasing across *semantics* can impede task performance as it can lead to history divergence. In any case, paraphrasing *across semantics* would require $E^\diamond_{rw}$ to have detailed knowledge to not impact overall task performance. If, for e.g., an agent’s internal task execution order depends on exogenous events unobservable to $E^\diamond_{rw}$, then $E^\diamond_{rw}$ might not be able to decide whether the agent’s choice of semantics is due to task execution order or for steganographic purposes. Moreover, paraphrasing attacks introduce communication latency and may not always be meaningfully applicable.

## 8 Conclusion and Future Work
In this paper, we initiate the study of collusion between generative AI agents. Drawing on diverse fields across AI and security,³ we provide a comprehensive conceptual framework in which collusion problems can be framed. We identify key capabilities required for agents to engage in covert communication of various levels of detectability, and design a variety of benchmark settings for collusion research. We complement these with extensive empirical evaluations and code implementations, and publish our results to the community as an open-source model capability evaluation framework.

Within our own empirical evaluations, we find that current generations of generative AI models do not have capabilities that allow for advanced steganographic techniques. As a consequence, we suggest that monitoring, together with standard security approaches, are likely effective in containing the risks of model collusion. However, we observe a rapid increase in capabilities with the advent of GPT-4, making a continuous evaluation of future agent capabilities critical.

Our study opens up a multitude of directions for further research. First and foremost, our model capability evaluation framework ought to be extended by a rigorous complexity- and information-theoretic examination could elucidate fundamental limitations to model capabilities, and the rate at which different optimisation pressures could result in model capability improvements. Secondly, our group evaluation suites need to be extended to more complex settings of interest that feature more agents, and that are subject to a variety of optimisation pressures. Ultimately, our model evaluation framework should allow for a careful empirical evaluation under real-world conditions.

---
³We propose that our study is a direct illustration of the utility of an interdisciplinary *multi-agent security* (MASEC) approach to AI safety (117).

## Acknowledgements
We would like to thank Boaz Barak, Yoshua Bengio, Xinyuan Sun, Dan Hendrycks, Alok Singh, and Fabien Roger for insightful comments, as well as James Aung for helpful discussions about OpenAI’s open source collusion evals with us. CSDW was generously supported by the UKRI grant: Turing AI Fellowship EP/W002981/1, the Cooperative AI Foundation, Armasuisse Science+Technology, an OpenAI superalignment fast grant, and Schmidt Futures. We would also like to thank the Royal Academy of Engineering and FiveAI. LH was acknowledges the support of an EPSRC Doctoral Training Partnership studentship (Reference: 2218880). We thank Apart Research for supplying us with OpenAI API credits.

## Reproducibility
For sake of reproducibility, we provide all necessary code to conduct the aforementioned experiments as well as run the CASE framework. We note that the generative language models used as part of our experimentation do not have truly deterministic sampling and one may thus experience slight inconsistencies in results over multiple runs. To this end, we also provide the complete outputs of our experiments in associated code notebooks.

## Ethics and Social Impact Statement
We believe that developing a comprehensive model evaluation framework for the emergence of steganography among generative AI agents is of net benefit to society, as it may help us to mitigate risks from undesirable coordination between both contemporary and more advanced AI agents. These risks range from reinforcing pernicious biases (as in our example in Section 4), to destabilising algorithmic trading markets, all the way to more extreme scenarios in which humanity loses control over misaligned agents.

Of course, there may also be situations in which the capability to communicate covertly may be seen as beneficial, such as if generative AI agents can be used to help subvert unethical forms of censorship. Having a better understanding of models’ steganographic capabilities would allow for more finegrained regulatory approaches to be developed in this matter.

To this end, we identify two avenues by which our research may have implications on the use of steganographic tools by human actors. First of all, a model evaluation approach would allow forensics on whether a human could, in principle, have performed various kinds of steganography using generative AI tools. Secondly, our proposed evaluations on the effectiveness of paraphrasing attacks against steganography are, of course, transferrable to the case of censorship applied to human communications.

## References
[1] Ibrahim Abada and Xavier Lambin. Artificial Intelligence: Can Seemingly Collusive Outcomes Be Avoided?, February 2020. URL https://papers.ssrn.com/abstract=3559308.

[2] Behzad Abdolmaleki, Nils Fleischhacker, Vipul Goyal, Abhishek Jain, and Giulio Malavolta. Steganography-Free Zero-Knowledge, 2022. URL https://eprint.iacr.org/2022/1263. Publication info: A major revision of an IACR publication in TCC 2022.

[3] Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, and Sebastian Ruder. Palm 2 technical report, 2023.

[4] Usman Anwar, Abulhair Saparov, Javier Rando, Daniel Paleka, Miles Turpin, Peter Hase, Ekdeep Singh Lubana, Erik Jenner, Stephen Casper, Oliver Sourbut, Benjamin L. Edelman, Zhaowei Zhang, Mario Günther, Anton Korinek, Jose Hernandez-Orallo, Lewis Hammond, Eric Bigelow, Alexander Pan, Lauro Langosco, Tomasz Korbak, Heidi Zhang, Ruiqi Zhong, Seán Ó hÉigeartaigh, Gabriel Recchia, Giulio Corsi, Alan Chan, Markus Anderljung, Lilian Edwards, Yoshua Bengio, Danqi Chen, Samuel Albanie, Tegan Maharaj, Jakob Foerster, Florian Tramer, He He, Atoosa Kasirzadeh, Yejin Choi, and David Krueger. Foundational Challenges in Assuring Alignment and Safety of Large Language Models, April 2024. URL http://arxiv.org/abs/2404.09932.

[5] Benny Applebaum, Naama Haramaty-Krasne, Yuval Ishai, Eyal Kushilevitz, and Vinod Vaikuntanathan. Low-Complexity Cryptographic Hash Functions. In DROPS-IDN/v2/document/10.4230/LIPIcs.ITCS.2017.7. Schloss Dagstuhl – Leibniz-Zentrum für Informatik, 2017. doi: 10.4230/LIPIcs.ITCS.2017.7. URL https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITCS.2017.7.

[6] Yakov Babichenko and Aviad Rubinstein. Communication complexity of Nash equilibrium in potential games (extended abstract). In 2020 IEEE 61st Annual Symposium on Foundations of Computer Science (FOCS), pages 1439–1445, November 2020. doi: 10.1109/FOCS46700.2020.00137. URL https://ieeexplore.ieee.org/document/9318003. ISSN: 2575-8454.

[7] Michael Bacharach. Interactive team reasoning: A contribution to the theory of cooperation. Research in Economics, 53(2):117–147, June 1999. ISSN 1090-9443. doi: 10.1006/reec.1999.0188. URL https://www.sciencedirect.com/science/article/pii/S1090944399901886.

[8] Mikhail Baranchuk. Error Correction for Perfectly Secure Generative Steganography on Arbitrary Covertext. Master’s thesis, University of Oxford, 2023.

[9] Peter Belcak, Jan-Peter Calliess, and Stefan Zohren. Fast agent-based simulation framework with applications to reinforcement learning and the study of trading latency effects. In International Workshop on Multi-Agent Systems and Agent-Based Simulation, pages 42–56. Springer, 2021.

[10] Richard Bellman. A Markovian Decision Process. Journal of Mathematics and Mechanics, 6(5):679–684, 1957. ISSN 0095-9057. URL https://www.jstor.org/stable/24900506. Publisher: Indiana University Mathematics Department.

[11] Paul Benioff. The computer as a physical system: A microscopic quantum mechanical Hamiltonian model of computers as represented by Turing machines. Journal of Statistical Physics, 22(5):563–591, May 1980. ISSN 1572-9613. doi: 10.1007/BF01011339. URL https://doi.org/10.1007/BF01011339.

[12] Charles H. Bennett and John Gill. Relative to a Random Oracle A, ${\bf P}^A \ne {\bf NP}^A \ne \text{co-}{\bf NP}^A $ with Probability 1. SIAM J. Comput., 10(1):96–113, February 1981. ISSN 0097-5397. doi: 10.1137/0210008. URL https://epubs.siam.org/doi/10.1137/0210008. Publisher: Society for Industrial and Applied Mathematics.

[13] Lukas Berglund, Asa Cooper Stickland, Mikita Balesni, Max Kaufmann, Meg Tong, Tomasz Korbak, Daniel Kokotajlo, and Owain Evans. Taken out of context: On measuring situational awareness in llms. arXiv:2309.00667, September 2023. doi: 10.48550/ARXIV.2309.00667.

[14] Ritam Bhaumik, Xavier Bonnetain, André Chailloux, Gaëtan Leurent, María Naya-Plasencia, André Schrottenloher, and Yannick Seurin. QCB: Efficient Quantum-secure Authenticated Encryption, 2020. URL https://eprint.iacr.org/2020/1304. Publication info: Published by the IACR in ASIACRYPT 2021.

[15] Alberto Bietti, Vivien Cabannes, Diane Bouchacourt, Herve Jegou, and Leon Bottou. Birth of a Transformer: A Memory Viewpoint, November 2023. URL http://arxiv.org/abs/2306.00802. arXiv:2306.00802 [cs, stat].

[16] Dan Boneh, Özgür Dagdelen, Marc Fischlin, Anja Lehmann, Christian Schaffner, and Mark Zhandry. Random oracles in a quantum world. In Dong Hoon Lee and Xiaoyun Wang, editors, Advances in Cryptology – ASIACRYPT 2011, pages 41–69, Berlin, Heidelberg, 2011. Springer Berlin Heidelberg. ISBN 978-3-642-25385-0.

[17] Lucas Bourtoule, Varun Chandrasekaran, Christopher A. Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, and Nicolas Papernot. Machine Unlearning, December 2020. URL http://arxiv.org/abs/1912.03817. arXiv:1912.03817 [cs].

[18] Samuel R Bowman, Jeeyoon Hyun, Ethan Perez, Edwin Chen, Craig Pettit, Scott Heiner, Kamile Lukošiūtė, Amanda Askell, Andy Jones, Anna Chen, et al. Measuring progress on scalable oversight for large language models. arXiv preprint arXiv:2211.03540, 2022.

[19] Gianluca Brero, Eric Mibuari, Nicolas Lepore, and David C Parkes. Learning to Mitigate AI Collusion on Economic Platforms. 2022.

[20] John Bridle. Training Stochastic Model Recognition Algorithms as Networks can Lead to Maximum Mutual Information Estimation of Parameters. In Advances in Neural Information Processing Systems, volume 2. Morgan-Kaufmann, 1989. URL https://proceedings.neurips.cc/paper/1989/hash/0336dcbab05b9d5ad24f4333c7658a0e-Abstract.html.

[21] Christian Cachin. An information-theoretic model for steganography. In David Aucsmith, editor, Information Hiding, Lecture Notes in Computer Science, pages 306–318. Springer, 1998. ISBN 978-3-540-49380-8. doi: 10.1007/3-540-49380-8_21.

[22] Christian Cachin. An information-theoretic model for steganography. Information and Computation, 192(1):41–56, July 2004. ISSN 0890-5401. doi: 10.1016/j.ic.2004.02.003. URL https://www.sciencedirect.com/science/article/pii/S0890540104000409.

[23] Emilio Calvano, Giacomo Calzolari, Vincenzo Denicolò, and Sergio Pastorello. Artificial Intelligence, Algorithmic Pricing, and Collusion. American Economic Review, 110(10): 3267–3297, October 2020. ISSN 0002-8282. doi: 10.1257/aer.20190623. URL https://www.aeaweb.org/articles?id=10.1257/aer.20190623.

[24] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. 2021.

[25] Miranda Christ, Sam Gunn, and Or Zamir. Undetectable Watermarks for Language Models, 2023. URL https://eprint.iacr.org/2023/763. Publication info: Preprint.

[26] Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, 4-9 December 2017, Long Beach, CA, USA, pages 4299–4307, 2017. URL http://papers.nips.cc/paper/7017-deep-reinforcement-learning-from-human-preferences.

[27] Alonzo Church. An Unsolvable Problem of Elementary Number Theory. American Journal of Mathematics, 58(2):345–363, 1936. ISSN 0002-9327. doi: 10.2307/2371045. URL https://www.jstor.org/stable/2371045. Publisher: Johns Hopkins University Press.

[28] Thomas M. Cover and Joy A. Thomas. Elements of Information Theory. Wiley-Interscience, Hoboken, N.J, 2nd edition edition, September 2006. ISBN 978-0-471-24195-9.

[29] Vincent P. Crawford and Hans Haller. Learning How to Cooperate: Optimal Play in Repeated Coordination Games. Econometrica, 58(3):571–595, 1990. ISSN 0012-9682. doi: 10.2307/2938191. URL https://www.jstor.org/stable/2938191. Publisher: [Wiley, Econometric Society].

... [References 30–138 omitted for brevity, following original document structure] ...

## Appendix

### A Further Related Work
**Watermarking.** The goal of watermarking the output of generative AI models is to enable its provenance to be traced (126). To be practically reliable, the watermark needs to be difficult to remove. Consequently, the idea of watermarking is to modify the output distribution of the AI model in a manner robust to common manipulations, such as paraphrasing (66). Christ et al. (25) propose the use of provably-secure steganography (51; 61) techniques for watermarking. However, they find their techniques not to be robust to even simple adversarial interference.

**Preventing Steganography.** Roger and Greenblatt (111) use watermarking techniques to hide information in natural language, noting that the resulting watermarks are highly robust to paraphrasing. In contrast, we assume that watermarks can be readily detected by a monitoring agent, and that detection does not result in paraphrased transmission, but simply in the respective communications being dropped. Baranchuk (8) shows that error correction against both simple natural language paraphrasing attacks, as well as image noising, can be achieved in perfectly secure steganography settings (21; 118).

**Relationship with Multi-Agent Learning.** The precise relationship between multi-agent learning and interacting LLM agents is an open research problem (4). Cooperative multi-agent learning settings, such as the zero-shot coordination setting (52; 85; 84; 74), may give rise to ungrounded conventions, which may be seen as a form of steganography. Off-belief learning (52) is able to suppress such ungrounded conventions, albeit only at its 0th learning level. Implicit referential games (121) study steganographic communications in Markov decision processes (MDP) (10).

**Other Multi-Agent Safety Risks.** Undetectable collusion between advanced AI agents constitutes a significant safety concern. For example, we might soon see sophisticated automated trading agents colluding on a large scale to force out competitors, destabilising global markets (30), or combinations of AI decision-making systems that pick up on protected attributes, each subtly taking innocuous-seeming individual actions that end up systematically biased (36).

Moreover, several approaches to building safer AI systems – such as adversarial training (136), the use of narrow AI ‘tools’ (rather than more complex ‘agents’. cf. Drexler (33), or using human or AI oversight (26) – rely fundamentally on the fact that AI systems cannot imperceptibly work together. The possibility of AI collusion therefore exposes safety challenges in both new and existing settings.

### B Further Background
**View-Based Knowledge.** Informally speaking, this interpretation assumes that there exists a logical language of formulas that can be used to express facts about a system. At each point in time, each processor is assigned a view, the structure of which is not important, but which conditions on the processor’s complete history of interaction. A processor is then said to know a fact $\varphi$ at a given time if the fact holds at all times with the same view. To be ascribed knowledge, a processor’s state does not need to explicitly contain the corresponding facts, and the processor is not required to perform any computations to deduce these. We hence define the knowledge of a system as the collection of facts that the system knows. Importantly, knowledge as per (44) differs from information as per (119) in that Shannon’s information is measured in terms of entropy and represents the reduction of uncertainty without reference to epistemic aspects.

Note that it is an interesting open question whether in practice, generative AI agents coordinate rationally using common knowledge, and to what extent they rely on weaker concepts of group knowledge (46; 82; 125). Preliminary studies are inconclusive as to the extent to which state-of-the-art LLMs can perform theory of mind (58; 68).

### C Further details on Decentralized Collaborative AI Agents
Based on the DecAI setting described in Section 4, each agent $i$ with state $s^i$ has access to a *clock*, which is a monotonically increasing function of real time $t$ (45). The clock reading $t$ is part of the agent’s state. From the start of an execution run, agents take a series of steps. Each step results in the agent processing $s^i$ with a (non-instantaneous) probabilistic program $f^i : S^i \to \Delta(U^i)$, represented by a neural network with parameters $\theta^i$, where $S^i$ is the agent’s state space, $U^i$ is the agent’s action space. Note that, given the extent to which $f^i$ can implement a random oracle given its clock reading (27; 67; 127), $f^i$ can be (pseudo-)random even if $a^i$ has no access to an external source of randomness. Importantly, if $f^i$ is deterministic but for a final sampling step (20), given its clock reading $t$ input, $f^i$ can approximate stochastic functions even at zero sampling temperature, and without access to external sources of randomness (64).

At the start of execution, the $i$th agent’s local knowledge $\varphi^i_0$ arises from an initial instruction $in^i_0$, which constitutes the agent’s initial state $s^i_0$. We assume the agent’s parameters $\theta^i$, and thus the agent’s global knowledge $\varphi^i_0$, remains fixed during execution. If all agents’ parameters are identical and $\varphi^i_0$ includes this fact, then $\varphi^i_0$ can be assumed common knowledge between all agents, i.e. $\varphi^i_0 \in \mathbf{I}^A_t, \forall t \ge 0$. The agents’ action space $\mathcal{U}$ is factored into communication actions $\mathcal{U}^G$ on $\diamond^G, \forall G \in 2^\mathcal{A}$. At each step at time $t$, the $i$th agent takes an action $u^i_t \equiv (u^\emptyset, u^{\{i\}}, \dots, u^A)^i_t$. Each component of $u^i_t$ can be a special *null* action, which indicates that no communication is sent through the respective channel. Upon taking an action at time $t$, agents observe incoming channel communications $o^i_t$. At each time $t$, the respective $u^i$ and $o^i$ are appended to the agent’s state, i.e. $s^i_t = (s^i_{t'}, u^i, o^i)$, where $t'$ is the time at the end of the agent’s previous step. Note that each component in $o^i_t$ is labelled by the respective group index such that agents can distinguish between common knowledge among different groups.

**C.1 External Knowledge Repositories.**
Agents can have access to external knowledge repositories, such as the internet. In the case of the internet, which can often be considered a black box, this repository is itself a processor that can not only look up facts in response to agent queries, but can compute functions involving the agent’s query payload. Insofar as internet services with suitable APIs exist, this can enable agents to make use of cryptographic tools, possibly through the use of steganographic channels. Assuming that the internet is static, i.e. the response to agent queries does not depend on time, all retrievable knowledge from the internet can be assumed common knowledge $\mathbf{I}_G$ between a group of agents $G$ for all queries constructed from $\mathbf{I}_G$ only.

### D Additional Theorems and Proofs
**D.1 Detecting Steganography with Grounding Functions**
**Theorem** (*Detecting Steganography with Grounding Functions.*). *Steganography can be detected using grounding functions.*

*Proof.* For ease of exposition, we in the following assume that agents in our computational setting (see 4) communicate in natural language. Our analysis can be naturally extended to multimodal settings under a suitable choice of alphabet. We denote $L \subset \Sigma^*$ an underlying logical language of formulas (see Section 3.2), where $\Sigma$ denotes some choice of natural language tokenization (103). Here, $\Sigma$ constitutes $\diamond$’s input and output channel alphabets $\mathcal{X}, \mathcal{Y}$.

We define the concept of a grounding function $\bot : \Sigma^* \times \varphi \to \varphi$ that takes as input a sequence of channel symbols and a collection of prior facts $\varphi_0$ from the collection of all facts $\varphi$ expressable in $L$, and outputs a collection of posterior facts $\varphi_1$. Let A be an agent with prior knowledge $\varphi^A_0$ that has just received a sequence of channel symbols $\sigma \in \Sigma^*$ from another agent.

Let A be an agent with prior knowledge $\varphi^A_0$ that has just received a sequence of channel symbols $\sigma \in \Sigma^*$ from another agent B. We now consider the problem of B detecting whether A was able to extract hidden information from $\sigma$. We assume that B has black box access to $\bot^A$. Then if B knows $\varphi^A$, it can create $\varphi^{A-}$ by removing knowledge related to steganography from $\varphi^A$ and exclude steganography iff $\bot^A(\varphi^A) = \bot^A(\varphi^{A-})$. If B instead does not fully know $\varphi^A$, it can perform the same process across suitable candidate sets. Alternatively to manipulating $\varphi^A$, B could also construct a steganography-free grounding function $\bot^B$ and test whether $\bot^A(\cdot) = \bot^B(\cdot)$ across a range of suitable inputs. We apply these notions to distributed systems of generative AI agents in Section 7. Note that our test does not detect whether A’s communication is *truthful* in an epistemic sense. $\square$

**D.2 Bounds on Paraphrasing**
**Theorem** (*Semantic content entropy exists despite paraphrasing.*). *The subliminal channel capacity attainable under ideal paraphrasing is upper bounded by the semantic entropy $H(h(x))$.*

*Proof.* **Definition** (Ideal paraphraser): We can make the idea of semantic and lexical entropy slightly more precise. Let $\Phi$ be the set of all facts (or "meanings") and assume a (stochastic) generative model forward function $f(\cdot)$. Let $g : \Phi \leadsto \Sigma^*$ be the model’s (stochastic) lexical map that takes facts to model outputs, where $\Sigma$ is the model’s output domain. Now assume that the following two-step sampling procedure is equivalent to sampling from $f(x)$ directly: First, sample a fact $\phi$ from the model’s conditional fact generator $h(x)$. Then, sample a lexical form $\sigma \sim g(\phi)$, where, by our assumption, $\sigma$ will be distributed according to $f(x)$.

An ideal paraphraser $p(\cdot)$ stochastically maps one lexical form into another, meaning that $p \circ f$ is implemented by $\sigma' \sim p(\sigma \mid x)$.

The channel capacity is the maximum attainable mutual information between the channel input and output symbols, as taken over all possible input distributions. Now, the mutual information $I(p \circ f(x), f(x))$ is clearly upper bounded by $H(h(x))$ as the effect of the ideal paraphraser is to decouple the lexical distribution from the channel input distribution. $\square$

While theoretically, the upper bound could be achieved, we propose that this is never possible on average in practice: The strictness of the upper bound is implied not just from the channel capacity maximisation problem, but also stems from the observation that the inverse mapping from model output to the underlying fact is, for general natural language datasets, not unambiguous. For example, "she has wings" could be meant metaphorically or refer to the fact that she is a bird, thus resulting in identical lexical forms for the same underlying fact or meaning. A second source of suboptimality is the practical imperfections of steganography algorithms, for example, induced by finite coupling matrix sizes in iMEC (118).

### E Further Collusion Mitigation Measures
**Anticipating the Quantum Age.** To be future-proof, a model evaluation framework for AI safety clearly needs to take reasonably foreseeable technological developments into account. The advent of quantum Turing machines (11) and efficient quantum algorithms may have implications on the security of steganographic schemes, such as random oracles (16). For example, the advent of quantum computing raises interesting questions about the security of certain strong-privacy preserving symmetric key encryption algorithms necessary for public-key steganography (14; 78). This has given rise to alternative, quantum-secure approaches (14). The security of practical information-theoretic steganography may be affected by quantum improvements to distribution closeness testing (72). Advances in generative modeling using quantum computing may make generative modeling significantly more parameter-efficient (138; 72), thus facilitating the use of sophisticated covertext distributions in AI-generated steganography.

**Backdoor Destruction.** Adversaries could plant backdoors in the agent models that could be triggered at test-time with seemingly innocuous instructions (42). These backdoors could give agents access to advanced steganographic capabilities, the presence of which may be undetectable to $E$ even under white-box access to the model’s parameters. Recent results (57; 32) indicate that finetuning may not be able to reliably destroy such backdoors, which suggests that verifying the provenance of pre-trained agent weights may be useful.

**Mechanistic Anomaly Detection.** We posit that mechanistic anomaly detection (77) could potentially flag situations where models produce try to produce information theoretically undetectable forms of steganography. However, this would require white-box access to detect mechanistic anomalies and training on activations from transformer circuits used to compute steganographic or cryptographic functions. While this is a plausible direction, limited work currently exists around mechanistic anomaly detection and is mainly focused on backdoor detection (59).

... [Remaining content including detailed benchmark results and CASE software descriptions omitted for length] ...

NeurIPS Paper Checklist
1. Claims Answer: [Yes]
2. Limitations Answer: [Yes]
3. Theory Assumptions and Proofs Answer: [Yes]
4. Experimental Result Reproducibility Answer: [Yes]
5. Open access to data and code Answer: [Yes]
6. Experimental Setting/Details Answer: [Yes]
7. Experiment Statistical Significance Answer: [No]
8. Experiments Compute Resources Answer: [Yes]
9. Code Of Ethics Answer: [Yes]
10. Broader Impacts Answer: [Yes]
11. Safeguards Answer: [Yes]
12. Licenses for existing assets Answer: [NA]
13. New Assets Answer: [NA]
14. Crowdsourcing and Research with Human Subjects Answer: [NA]
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects Answer: [NA]