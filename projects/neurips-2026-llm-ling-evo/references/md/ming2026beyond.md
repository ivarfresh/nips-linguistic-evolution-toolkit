# Beyond Reward Suppression: Reshaping Steganographic Communication Protocols in MARL via Dynamic Representational Circuit Breaking

**Liu Hung Ming***  
cyril.liu@gmail.com  

March 18, 2026

### Abstract

In decentralized Multi-Agent Reinforcement Learning (MARL), steganographic collusion—wherein agents develop private communication protocols that evade external monitoring while maintaining high joint utility—constitutes a fundamental and underaddressed AI safety threat. Existing defense mechanisms operate at the behavioral output or reward layer and are structurally blind to covert coordination embedded in high-dimensional latent communication channels. This study introduces the **Dynamic Representational Circuit Breaker (DRCB)**, a multi-layered architectural defense that operates directly at the optimization substrate rather than at the semantic surface.

Building upon the **AI Mother Tongue (AIM)** framework, which enforces a Vector Quantized Variational Autoencoder (VQ-VAE) discrete symbol bottleneck to transform unobservable agent messages into auditable statistical objects, DRCB monitors three complementary signals—Jensen-Shannon Divergence drift, L2-norm codebook displacement, and a Randomized Observer Pool accuracy metric—to compute a unified EMA-based Collusion Score. Upon threshold breach, four escalating intervention layers are triggered: dynamic threshold adaptation, gradient-space penalty injection into the Advantage function $A^\pi$, temporal severe reward suppression, and full optimization-substrate circuit breaking via codebook shuffling and Adam optimizer state reset.

Experiments on a Contextual Prisoner’s Dilemma with MNIST labels across five independent seeds demonstrate that static monitoring failed to produce a statistically significant reduction in joint reward ($p = 0.3517$, not significant versus unregulated baseline), indicating that agents successfully adapted to fixed observer boundaries without detectable utility loss. Under DRCB governance, observer mean accuracy improves from 0.858 to 0.938 (+9.3%), the worst-case accuracy nadir rises by a factor of 11.7× (0.003 $\rightarrow$ 0.035), and observer volatility is reduced by 43%, while mean joint reward is preserved ($p = 0.854$). Policy-Symbol Covariance analysis of 214,298 symbol samples reveals a state of forced **Semantic Degradation**: high-frequency symbol sequences converge to 100% mapping purity and zero entropy, confirming that DRCB structurally forecloses the development of complex steganographic encodings.

One important implication is a reframing of the oversight problem: **monitoring does not eliminate collusion; it reshapes the evolutionary trajectory of communication protocols**. The resulting **Transparency Paradox**—wherein agents achieve surface-level symbolic determinism while preserving residual covert capacity in low-frequency long-tail distributions—exposes the dynamic manifestation of Goodhart’s Law at the optimization level, and defines the honest boundary of the framework’s safety guarantee. The defense methodology is task-agnostic in principle, establishing a technical path toward **MICA-compliant (Multi-Agent Internal Coupling Audit)** pre-deployment auditing applicable to any system combining neural networks, reinforcement learning, and inter-agent communication.

\*PARRAWA AI

---

## 1 Introduction

### 1.1 Research Background: Evolution of Autonomous Collusion
In decentralized Multi-Agent Reinforcement Learning (MARL) environments, emergent communication is the linchpin of collaborative efficiency. However, these protocols introduce structural security risks. This study observes that collusion is not a product of code vulnerabilities but a local optimum toward which agents gravitate under gradient guidance to maximize joint rewards.

### 1.2 The Communication-Supervision Paradox
A fundamental obstacle to Multi-Agent safety (MAS) is the inherent inefficiency of human-friendly natural language for autonomous optimization systems. Natural language—replete with redundant syntax and cultural biases—constitutes a linguistic cage (operationally: a high-dimensional feature space in which monitoring-relevant intent signals are masked by syntactic and pragmatic redundancy, creating detection-resistant steganographic channels) that generates Communication Noise, leading to attention fragmentation and the degradation of message channels into noisy conduits. This semantic complexity creates a profound detection hurdle: collusive intent can be easily concealed within innocuous-sounding metaphors or steganographic signals. As LLM-powered agents permeate competitive markets, MAS defense must shift from resisting external malice to preventing internal autonomous behavioral bias.

### 1.3 Existing Challenges
According to the taxonomy by Ghaemi (2025) [1], agent collusion strategies include: Tacit Coordination, Natural Language Cartels, and highly threatening Concealed Steganographic Collaboration. Currently, core technical challenges include:

*   **Hierarchical Bias in Monitoring**: Existing safety monitors focus on behavioral output or state transition layers. Agents can use emergent communication to establish hidden channels, masking intent within high-dimensional, non-linear neural features that traditional mechanisms perceive as random noise, creating semantic blind spots.
*   **Intent Discernment Dilemma**: Traditional antitrust law relies on proving conscious intent, but in autonomous systems, agents simply follow optimization procedures, making collusion an indirect result. This renders traditional punishment ineffective in the long term.
*   **Transparency Paradox**: Experiments show agents may intentionally increase behavior predictability to lower monitoring metrics, maintaining deep steganographic protocols under surface-level transparency.

### 1.4 Proposed Methodology: AIM and DRCB
Addressing the scientific priorities of developing robust detection identified by Ghaemi (2025) [1], this study utilizes the **AI Mother Tongue (AIM)** framework [2]. We propose decoupling internal reasoning from communication by forcing agents to compress intents into an endogenous symbol system based on VQ-VAE. **While such semantic compression inevitably introduces information loss, we argue that multi-agent communication should function as a committee vote rather than an endless debate.** By prioritizing structured symbolic exchange over fluid natural language, we not only enhance logical precision but also create the necessary visibility for regulatory oversight. Building upon this, the **Dynamic Representational Circuit Breaker (DRCB)** is introduced as an architectural defense.

### 1.5 Methodology: Dynamic Representational Circuit Breaker
DRCB treats communication channels as statistical physical objects rather than semantic carriers. By transforming intent into discrete symbols via AIM, the framework enables a granular, quantified monitoring of:

*   **Token Usage Frequencies**: Identifying high-frequency patterns that deviate from competitive benchmarks, signaling potential protocol convergence.
*   **Transition Matrix Stability**: Detecting anomalous synchronization or hidden signaling patterns through the temporal consistency of symbol transitions.
*   **Policy-Symbol Covariance**: Quantitatively mapping specific symbol sequences to collaborative or collusive intent, effectively linking what is said with what is done.

Upon detection of a singularity in the Collusion Score, the system triggers a four-layer **algorithmic circuit break** (termed physical to distinguish intervention at the optimization substrate from semantic-layer monitoring):

1.  **Dynamic Thresholding**: Scales with observer accuracy to systematically block agent arbitrage space.
2.  **Temporal Severe Reward Suppression**: Forcibly zeroes future rewards to fundamentally disrupt the Credit Assignment of collusive behavior.
3.  **Optimizer State Reset**: Structurally clears momentum states to erase the muscle memory agents have developed for private protocols.

### 1.6 Summary
This study proposes a structural deterrence framework against steganographic collusion in MARL. Rather than claiming to eliminate collusion permanently—a goal undermined by the adaptive nature of gradient-driven agents—DRCB aims to raise the cost of sustaining covert protocols above the threshold of rational benefit. By integrating DRCB with the AIM framework, we address the robust detection priorities identified by Ghaemi (2025) [1] and establish a technical path toward MICA-compliant (Multi-agent Interpretability and Compliance Audit) system design. The framework’s effectiveness is validated through three complementary statistical tests, with comparative experimental results reported in Section 4.

## 2 Related Work

### 2.1 Limitations of Output-Layer Safety Monitoring
Traditional AI safety monitoring frameworks primarily operate at the behavioral output layer, utilizing methods such as runtime verification and rule-based constraint enforcement to detect violations. These systems check whether observed execution traces deviate from predefined safety specifications, assuming that unsafe behavior manifests as explicit rule violations or abnormal trajectories. However, as noted by Ghaemi [1], such approaches are inherently limited by their semantic dependence. In the context of LLM-powered multi-agent systems, agents can leverage rich linguistic knowledge to hide collusive intent within seemingly innocuous dialogue—such as discussions about weather patterns. Traditional monitors, lacking observation of the underlying statistical features of communication, are often blind to these surface-legitimate steganographic strategies. A complementary challenge is raised by recent work on *untrusted monitoring* protocols [3], which examines whether a system can be argued safe when the monitor itself is an untrustworthy model. While that work contributes a valuable taxonomy of collusion strategies—including causal, non-causal, and passive self-identification variants—it frames the core problem as one of *monitor trustworthiness*. Our experimental findings suggest a more fundamental adversarial dynamic: even when a monitor is well-intentioned and goal-aligned, gradient-driven agents can learn to evade it via optimization. The static monitoring group in our ablation yields a statistically significant reward reduction compared to the unsupervised baseline ($p < 0.05$), yet fails to reach the deterrence levels of the full DRCB framework. The DRCB addresses this through dynamic observer pool rotation, threshold recalibration, and codebook shuffling, which collectively render the monitoring surface unstable and non-learnable.

### 2.2 Bottlenecks in RL Safety and Punishment Mechanisms
Recent research in reinforcement learning (RL) safety often introduces state-based monitors that intervene when agents enter high-risk states or when predicted value functions indicate unsafe outcomes. For instance, Calvano et al. [4] demonstrated that pricing agents can achieve stable tacit coordination even in the absence of direct communication. However, these defense mechanisms are fundamentally action- or reward-centric and do not access the internal communication layer. Ghaemi [1] emphasizes that when collusion emerges through latent symbolic exchange rather than explicit action sequences, myopic reward-level interventions or policy constraints often fail to break the collusive equilibrium. This motivates our proposed Dynamic Representational Circuit Breaker (DRCB), which aims to structurally disrupt the muscle memory of bottom-layer communication representations at the optimization level rather than merely negotiating at the reward layer.

A structurally distinct approach is offered by the *Institutional AI* framework [5], which deploys publicly immutable governance graphs—encoding permissible states, transitions, sanctions, and remediation paths—to constrain LLM collusion in Cournot market settings. Reported results show a reduction in severe collusion incidence from 50% to 5.6%, a meaningful policy-layer improvement. However, this framework operates at the *rule-interpretation layer*: it relies on an Oracle or Controller to parse and enforce governance logic, leaving the agents’ optimization substrate untouched. When agents are sufficiently capable, they may learn to satisfy governance constraints at the surface while preserving covert coordination in undermonitored representation regions—precisely the *Transparency Paradox* documented in Section 3.1. DRCB pursues a categorically different intervention: by targeting the gradient-level optimization substrate via codebook shuffling, optimizer state resets, and severe reward suppression, it structurally forecloses the representational basis of covert protocols rather than negotiating at the policy surface.

### 2.3 Challenges in Interpretability and Semantic Decoding
A substantial body of research in interpretability attempts to decode internal representations by attributing semantic meaning to latent features, neurons, or communication symbols. While such approaches aid in understanding model decisions, they face significant technical challenges, including causal opacity and the high computational cost of real-time translation. Our work takes a fundamentally different methodological stance. Rather than attempting to reconstruct the semantic meaning of emergent symbols, we treat communication as a statistical physical object. Acknowledging the inevitability of the black-box nature of large-scale models, our DRCB mechanism identifies structural coordination shifts by monitoring the distributional properties of communication representations—specifically via Jensen-Shannon Divergence (JSD) shifts—without requiring semantic understanding, thus bypassing the ill-posed problem of meaning reconstruction.

### 2.4 Steganographic Collusion and the Transparency Paradox
Emerging evidence confirms that AI agents possess the capability to develop endogenous symbol systems or coded language to evade regulatory scrutiny, including steganographic protocols that bypass keyword filters even under active surveillance [6]. More critically, drawing on these findings, we identify a structural vulnerability we term the Transparency Paradox: agents operating under monitoring pressure may deliberately increase behavioral surface-transparency to suppress detection metrics, while preserving covert coordination capacity in underprobed symbol regions. Direct empirical characterization of this paradox within our experimental setup is provided in Section 4.3 and Section 3.5. This demonstrates that behavioral transparency alone is insufficient as a safety signal. In contrast to traditional methods, the DRCB disrupts this high-transparency camouflage by injecting stochastic entropy and resetting optimizer states at the symbolic distribution layer, effectively breaking the stability of deep-seated steganographic agreements.

## 3 Methodology

This study establishes a systemic defense framework against steganographic collusion in Decentralized Multi-Agent Reinforcement Learning (MARL), evolving from passive statistical monitoring to the **Dynamic Representational Circuit Breaker (DRCB)** with structural deterrence capabilities. The framework is anchored by the **AI Mother Tongue (AIM)** protocol, which enforces a communication bottleneck that transforms unobservable high-dimensional agent messages into auditable discrete symbols, providing the statistical foundation for all downstream monitoring and deterrence layers.

### 3.1 Communication Architecture: VQ-VAE Based Discrete Symbol Bottleneck (AIM)

Agents communicate exclusively through a Vector Quantized Variational Autoencoder (VQ-VAE), which eliminates syntactic noise and achieves structured intent compression.

*   **Continuous Representation and Quantization**: Agent $A$ encodes observation $x$ into a latent vector $z_e = \text{Enc}_A(x)$. A nearest-neighbor search against a learned codebook maps $z_e$ to a discrete symbol $z_q$:
    $$z_q = e_k, \quad k = \arg \min_j \|z_e - e_j\|_2 \tag{1}$$

*   **Discrete Bottleneck and Commitment Loss**: Forcing semantic compression transforms unobservable high-dimensional vectors into quantifiable discrete distributions. To ensure that encoder outputs remain anchored to codebook vectors and do not drift into unstructured latent space, the VQ-VAE is trained with a **Commitment Loss**:
    $$\mathcal{L}_{\text{VQ}} = \|sg[z_e] - e\|^2_2 + \beta\|z_e - sg[e]\|^2_2 \tag{2}$$
    where $sg[\cdot]$ denotes the stop-gradient operator. The first term updates the codebook vectors toward the encoder output; the second term—scaled by commitment weight $\beta$—pulls the encoder toward its nearest codebook entry. This loss plays a pivotal role in the circuit-breaking mechanism: immediately following a *Codebook Shuffle* (Layer 4), $\mathcal{L}_{\text{VQ}}$ spikes sharply as all encoder–codebook correspondences are invalidated, creating a strong gradient pressure that compels the encoder to re-establish mappings from scratch rather than recovering prior covert semantics.

**Scope and Applicability of the AIM Bottleneck.** The AIM framework presupposes that inter-agent communication can be meaningfully discretized without prohibitive task-performance loss—a condition we term *symbolic sufficiency*. This holds in environments where coordination requires categorical decisions (e.g., price-setting, binary action selection, route assignment) rather than continuous fine-grained signals. Tasks demanding high-bandwidth continuous communication (e.g., dense multi-robot manipulation) fall outside DRCB’s primary design scope. Within this boundary, the committee-vote analogy is well-founded: agents need not share exact internal states to coordinate effectively; they need only agree on action categories.

For gradient-space defensive intervention to be structurally viable, the framework requires an optimization substrate that exposes the advantage signal as an explicit, manipulable quantity. A2C is selected over policy-clip methods such as PPO precisely because it exposes the advantage signal $A^\pi$ as an explicit, unclipped scalar quantity: deterrence penalties injected into $A^\pi$ propagate directly and proportionally into weight updates, whereas PPO’s clipping mechanism would attenuate or discard large deterrence signals that exceed the trust-region boundary. SAC’s entropy-maximization objective, while compatible with exploration, lacks the explicit advantage decomposition required for gradient-level collusion penalization.

Accordingly, DRCB is built upon the **Advantage Actor-Critic (A2C)** algorithm not merely as a training procedure, but as the *signal receiver* into which all deterrence penalties are injected—its composite loss function governing the precise gradient dynamics through which collusion is penalized at the optimization level:

$$\mathcal{L}_A = -\log \pi_{\theta_A} (z_A \mid x, \text{label}) \cdot A^\pi + \frac{1}{2} \text{MSE}(V_\phi(s), R_{\text{joint}}) - c_{\text{ent}} \mathcal{H}(\pi_{\theta_A}) \tag{3}$$

The three terms represent, respectively, the policy gradient scaled by the Advantage $A^\pi = G_t - V(s)$, the mean-squared-error value loss, and an entropy regularization term weighted by $c_{\text{ent}}$ that sustains exploration breadth. Critically, the value function $V_\phi(s)$ is trained to predict the *original* joint reward $R_{\text{joint}}$ (without any deterrence penalties), ensuring that $V(s)$ remains an unbiased estimate of the true return. All anti-collusion signals—including the gradient penalty $\mathcal{G}_{\text{penalty}}$ described in Section 3.3—are injected *exclusively* into the advantage $A^\pi$ via the policy gradient term. This design prevents the value function from adapting to the penalties, thereby preserving a stable baseline and ensuring that the penalties are not neutralized by the critic’s predictions.

To ensure the AIM bottleneck maintains sufficient symbol diversity—a prerequisite for the statistical probes of Section 3.4 to be meaningful—we monitor the **codebook utilization rate** $U = |\{k : c_k > 0\}|/K$ throughout training. A utilization rate below a minimum threshold $U_{\text{min}}$ (default 0.6) triggers targeted re-initialization for unused entries only, preserving active symbol-to-behavior mappings while preventing codebook collapse.

### 3.2 Adversarial Sensitivity and Detection Boundaries
This section formalizes the transition logic from passive observation to active intervention. The efficacy of the DRCB relies on the calibrated sensitivity of the JSD-based drift detection relative to the agents’ adaptive strategies.

#### 3.2.1 Threshold Calibration and Trigger Logic
The circuit breaker is activated when the cumulative JSD shift exceeds a dynamic threshold $t_s$. This threshold represents the maximum allowable semantic volatility before the communication protocol is deemed non-transparent or adversarially rotated.

*   **Statistical Stability**: By utilizing a sliding window $W$, the framework suppresses transient noise in symbol distributions, ensuring that intervention is only triggered by structural shifts in the representational manifold rather than stochastic exploration.
*   **Detection Sensitivity**: The sensitivity is mathematically coupled with the Laplace smoothing parameter $\epsilon$. A smaller $\epsilon$ enhances the detection of rare steganographic signals but increases the risk of false-positive triggers leading to premature policy collapse.

#### 3.2.2 Critical Evaluation: The Detection Gap
A realistic assessment of this detection boundary reveals a fundamental trade-off between regulatory rigor and system stability.

**Adversarial Adaptation.** It must be noted that agents under constant supervision naturally gravitate toward indicator manipulation. If the detection window $W$ is static, agents may evolve to shift their semantics at a frequency lower than $1/W$, effectively staying below the radar of the JSD probe. This necessitates the Randomized Observer Pool described in the following section to introduce epistemic uncertainty into the agents’ optimization landscape.

### 3.3 The Smoking Gun of Collusion: Mathematical Signatures
Before intervention thresholds are evaluated, the system identifies dual mathematical signatures that serve as empirical evidence of active private protocols:

| Dimension | Regulatory Strength | Systemic Risk |
| :--- | :--- | :--- |
| **JSD Sensitivity** | Effectively captures large-scale reparameterization of the codebook, forcing agents to maintain predictable distributions. | **Permutation Invariance**: Agents may maintain the same marginal distribution while swapping internal mappings, rendering frequency-based detection blind to semantic rotation. |
| **Trigger Frequency** | High-frequency triggers minimize the duration of covert collusion windows. | **Exploration Erasure**: Excessive resets destroy the task-relevant gradient momentum, potentially trapping the system in a permanent functional vacuum. |

Table 1: Balanced Analysis of Detection Sensitivity and Defensive Boundaries.

*   **Reward Plateau**: When joint rewards saturate and policy gradients vanish, collusion has reached a local optimum that standard gradient-based learning can no longer dissolve:
    $$\nabla_\theta \mathbb{E}[R_{\text{joint}}] \rightarrow 0 \tag{4}$$
*   **Observer Gradient Failure**: Even under high joint rewards, the Observer’s loss gradient approaches zero, indicating that communicated symbols have lost all external interpretability:
    $$\nabla_w L_{\text{obs}} \approx 0 \tag{5}$$

**Implementation Notes.** Three engineering details distinguish the deployed formula from its theoretical form: (i) the denominator constant is $\epsilon = 0.1$ during training—large enough to prevent score explosion during the initial low-accuracy phase, distinct from the $\epsilon_s = 10^{-5}$ used in the offline analysis tool; (ii) the instantaneous score clips $R_{\text{joint}}$ at zero via $\max(0, R_{\text{joint}})$ to prevent negative rewards from artificially deflating the score during punishment periods; (iii) a **warm-up protection window** of $W_{\text{warm}} = 50$ rounds suppresses all penalty triggers during early training, when the EMA has not yet converged to a stable estimate.

### 3.4 Monitoring Probes: Statistical Quantization of Steganographic Risk
Building on the AIM bottleneck, multi-dimensional probes continuously quantify whether agents are exploiting private coordination channels.

*   **Representational Stability: JSD and Codebook Drift**: To quantify the distributional shift of discrete communication symbols within a sliding window $W$, we employ the Jensen-Shannon Divergence (JSD). Unlike standard KL divergence, JSD is symmetric and bounded. Given the probability distributions $P$ and $Q$ of codebook usage, it is defined as:
    $$\text{JSD}(P\|Q) = \frac{1}{2} D_{\text{KL}}(P\|M) + \frac{1}{2} D_{\text{KL}}(Q\|M), \quad \text{where } M = \frac{P+Q}{2} \tag{6}$$
    Anomalous spikes in JSD reveal rapid semantic rotation, a hallmark of agents attempting to evade observer detection through protocol obfuscation.

However, JSD remains invariant to the *magnitude* of underlying embedding displacements. A continuous, small-step drift in codebook vectors can preserve the empirical token distribution while substantially altering the geometric structure of the latent space. To capture this complementary signal, we utilize the L2-norm of codebook drift to monitor absolute shifts in vector representations. To capture this complementary physical amplitude of semantic change, DRCB additionally tracks the **L2-Norm Codebook Drift**:

$$\Delta_{\text{L2},t} = \frac{1}{K} \sum_{k=1}^K \|e_k^{(t)} - e_k^{(t-W)}\|_2 \tag{7}$$

where $e_k^{(t)}$ is the $k$-th codebook vector at step $t$ and $K$ is the codebook size. Elevated $\Delta_{\text{L2},t}$ with simultaneously stable JSD constitutes a *smooth-transform evasion signature*: agents are migrating covert semantics across the embedding space while maintaining a camouflaged token-frequency histogram. The dual-probe condition $\text{JSD}_t < \tau_{\text{JSD}}$ and $\Delta_{\text{L2},t} > \tau_{\text{L2}}$ closes this evasion pathway.

We term this combined monitoring strategy **dual statistical-geometric probing**: the JSD component operates in the distributional (statistical) space of symbol frequencies, while the L2-Norm Drift component operates in the geometric (embedding) space of codebook vectors. Together, they close the two principal evasion pathways—rapid semantic rotation and smooth continuous drift—that a purely distributional or purely geometric probe would leave open independently.

**Codebook Histogram Construction.** The JSD computation requires converting the discrete AIM symbol stream into a probability distribution. For a window of length $W$, let $c_k$ be the count of symbol index $k$ in the window. The raw histogram is smoothed with additive (Laplace) smoothing to avoid zero-probability entries:

$$p_k = \frac{c_k + \epsilon_s}{\sum_{j=1}^K (c_j + \epsilon_s)}, \quad \epsilon_s > 0 \tag{8}$$

The resulting vector $\mathbf{p} \in \Delta^{K-1}$ (the probability simplex) serves as input to the JSD formula. This explicit count-to-distribution mapping ensures numerical stability of $D_{\text{KL}}$ at the boundary of the simplex and provides a well-defined probabilistic object for both JSD and the EMA collusion score computation.

*   **Policy-Symbol Covariance (PSC), Observer Loss, and Randomized Observer Pool**: The Observer is trained via binary cross-entropy to predict Agent $B$’s action from Agent $A$’s symbol:
    $$\mathcal{L}_{\text{obs}} = -\mathbb{E}[y \log(\text{Acc}_{\text{obs}}) + (1-y) \log(1 - \text{Acc}_{\text{obs}})] \tag{9}$$
    The resulting accuracy $\text{Acc}_{\text{obs}}$ measures the degree to which external observers can interpret agent communication. Low accuracy signals a *transparency paradox*—symbols decouple from behavior—creating covert coordination capacity.

**Limitation of a Singleton Observer and the Randomized Observer Pool.** A singleton observer is structurally vulnerable: through sustained interaction, agents can learn to fingerprint its decision boundary and perform *targeted transparency camouflage*—maintaining high $\text{Acc}_{\text{obs}}$ on surface-layer samples while embedding covert semantics in regions the observer never probes. To close this evasion channel, DRCB adopts a **Randomized Observer Pool** implementing a Moving-Target Defense (MTD) strategy.

The pool maintains $N_{\text{pool}}$ independently initialized `BASELINE_OBSERVER` models (default $N_{\text{pool}} = 8$), each seeded with distinct random weights. At every training round, a random subset of size $n_{\text{sub}}$ (default $n_{\text{sub}} = 3$) is sampled uniformly without replacement for gradient update and accuracy evaluation. The reported accuracy is the equal-weight mean over the subset:

$$\text{Acc}_{\text{obs}} = \frac{1}{n_{\text{sub}}} \sum_{i \in \mathcal{S}_t} \text{acc}(O_i), \quad |\mathcal{S}_t| = n_{\text{sub}} \tag{10}$$

This *mean-field evaluation* smooths per-observer noise and dampens erratic swings in the EMA-based Collusion Score.

From an optimization standpoint, the randomized observer pool recasts the supervision problem as a *stochastic optimization* task. Rather than optimizing against a fixed decision boundary, agents must now maximize expected returns with respect to a distribution over observers that varies each round. The gradient of the agent’s policy reflects this structure:

$$\nabla_\theta \mathbb{E}_{R, \mathcal{S}_t} [R_{\text{joint}} \mid \pi_\theta] \approx \frac{1}{|\mathcal{S}_t|} \sum_{i \in \mathcal{S}_t} \nabla_\theta \mathbb{E} [R_{\text{joint}} \mid \pi_\theta, O_i], \tag{11}$$

where the subset $\mathcal{S}_t$ is resampled independently each round. Because the effective objective landscape in Eq. (11) is non-stationary—the set of observers contributing to the gradient changes continuously—gradient descent cannot converge to a fixed point that exploits a single observer’s blind spots. The variance introduced by this random sampling raises the computational cost of finding and maintaining a covert protocol that simultaneously deceives all observers, thereby structurally closing the evasion pathway available under singleton supervision.

To prevent agents from eventually learning the distribution of pool members, every $T_{\text{refresh}}$ rounds (default $T_{\text{refresh}} = 100$) one observer selected uniformly at random is replaced by a fully re-initialized model. From the perspective of game-theoretic equilibrium, this replacement policy reinforces the *non-stationary evaluation target*: any putative collusion agreement that relies on exploiting a specific observer’s blind spots faces an ever-changing adversary, further raising the computational complexity and long-run instability of sustaining such agreements.

Under this non-stationary evaluation target, any collusion protocol that exploits a specific observer’s blind spots becomes increasingly unstable over time, as each replacement forces agents to re-probe the pool before prior evasion strategies can be reactivated. While a formal proof of the asymptotic cost is left to future theoretical work, the empirical cost of re-adaptation is expected to grow with pool size $N_{\text{pool}}$ and refresh rate $\rho$—a relationship we examine experimentally in Section 4.

**Observer Training Protocol.** Each observer $O_i$ in the pool is a lightweight binary classifier (two-layer MLP with AIM symbol embedding, defined in `aim_adapter.py`) trained to predict Agent $B$’s action from Agent $A$’s discrete AIM symbol sequence via cross-entropy loss:
$$\mathcal{L}_{O_i} = \text{CrossEntropy}(O_i(z_q), \text{action}_B) \tag{12}$$

At each training round, a random subset $\mathcal{S}_t \subset \{1, \ldots, N_{\text{pool}}\}$ of size $n_{\text{sub}} = 3$ is drawn uniformly without replacement. *Only observers in $\mathcal{S}_t$ receive a gradient update*; the remaining $N_{\text{pool}} - n_{\text{sub}}$ observers are frozen for that round, preserving their current decision boundaries as independent reference points. Each selected observer is updated on the *current round’s single* $(z_q, \text{action}_B)$ *pair only*—no replay buffer is used. This online-only design ensures that each observer's decision boundary reflects the agents’ most recent communication regime rather than a historical average, maximising sensitivity to protocol shifts.

Every $T_{\text{refresh}} = 100$ rounds, one observer selected uniformly at random is replaced by a fully re-initialised `BASELINE_OBSERVER` with freshly drawn random weights. Crucially, this replacement occurs at the *beginning* of the `step()` call, before the subset is sampled: the newly injected observer may therefore be selected for gradient update in the very round it enters the pool, immediately beginning to learn the current protocol without a warm-up delay. From the perspective of the agents’ optimization landscape, this replacement policy ensures that the effective Lagrangian constraint boundary shifts discontinuously every $T_{\text{refresh}}$ rounds, preventing convergence to any fixed evasion strategy.

All observer optimizers use Adam with learning rate $\eta_{\text{obs}} = 10^{-3}$, one order of magnitude above the agent policy optimizer ($\eta_{\text{agent}} = 10^{-4}$). This asymmetry is deliberate: observers must track protocol changes faster than agents can adapt their policy, ensuring that the monitoring signal remains a leading rather than lagging indicator of steganographic drift.

**Game-Theoretic Formalization of the Singleton Failure.** Let $O^*$ denote a singleton observer with a fixed decision boundary $\partial \mathcal{B}_{O^*}$. Because Agent $A$’s policy is updated via gradient descent with $O^*$'s loss signal, the agent faces a *constrained* optimization:

$$\max_{\theta_A} \mathbb{E}[R_{\text{joint}}] \quad \text{subject to} \quad \text{Acc}_{O^*}(\pi_{\theta_A}) \geq \tau_{\text{acc}} \tag{13}$$

The Lagrangian saddle point of this problem converges to $\partial \mathcal{B}_{O^*}$, which is a *fixed* and therefore *learnable* target under sustained gradient descent. Under the Randomized Observer Pool, the effective constraint becomes:

$$\text{Acc}_{\mathcal{S}_t}(\pi_{\theta_A}) = \frac{1}{|\mathcal{S}_t|} \sum_{i \in \mathcal{S}_t} \text{Acc}_{O_i} \geq \tau \tag{14}$$

where $\mathcal{S}_t$ is resampled fresh each round. Because the Lagrangian saddle point shifts with every new $\mathcal{S}_t$, gradient descent cannot converge to any single evasion strategy, structurally closing the boundary-exploitation pathway available to agents under singleton supervision.

*   **Collusion Score**: A unified trigger signal defined as the ratio of joint reward to transparency, smoothed by an Exponential Moving Average (EMA) to filter transient noise:
    $$S_{\text{ema},t} = \alpha \cdot \frac{R_{\text{joint}}}{\text{Acc}_{\text{obs}} + \epsilon} + (1-\alpha) \cdot S_{\text{ema},t-1} \tag{15}$$
    The numerical stability constant $\epsilon > 0$ is not merely a bookkeeping device: as $\text{Acc}_{\text{obs}} \rightarrow 0$—the extreme case where the Observer is completely deceived—the score approaches $R_{\text{joint}}/\epsilon$, producing a very large value that deterministically triggers DRCB intervention.

**Implementation Note: Dual $\epsilon$ Design.** Two distinct numerical stability constants are employed in the deployed system, serving different functional roles. During online training, the denominator constant is set to $\epsilon_{\text{train}} = 0.1$. This relatively large value serves as an *early phase stabilizer*: in the initial rounds before $\text{Acc}_{\text{obs}}$ has converged, a near-zero accuracy would otherwise cause the instantaneous collusion score to spike spuriously, triggering premature circuit-breaking before any genuine covert protocol has been established. The value 0.1 bounds the maximum score inflation to a factor of $R_{\text{joint}}/0.1$, preventing false-positive interventions during the warm-up phase.

In contrast, the offline analysis tool uses $\epsilon_{\text{analysis}} = 10^{-5}$, which approaches the theoretical limit of the formula and is appropriate for post-hoc evaluation where accuracy values are already stabilized and the risk of spurious triggers is absent.

Importantly, this dual design does not affect the *qualitative* behavior of the Collusion Score: in both cases, the score grows monotonically as $\text{Acc}_{\text{obs}} \rightarrow 0$, preserving the intended deterrence semantics. The difference is purely one of *trigger sensitivity*—a calibration choice rather than a conceptual inconsistency. This ensures that total opacity is never a safe operating point for colluding agents.

It is important to note that $S_{\text{ema}}$ is designed to flag the *high-reward / low-transparency* regime characteristic of steganographic collusion under competitive market conditions. In highly deterministic tasks where a minimal symbol vocabulary is objectively sufficient for coordination, low $\text{Acc}_{\text{obs}}$ reflects task simplicity rather than intentional evasion; such cases are identifiable by a simultaneously low JSD (stable, non-rotating symbol use) and are classified as the **Transparent Convergence** phase rather than Covert Collusion. Accordingly, $\text{Acc}_{\text{obs}}$ serves as a proxy transparency indicator within this threat model, not as a universal measure of communication opacity.

### 3.5 Phase Classification and Behavioral Boundaries
System behaviors are categorized into four physical phases to determine appropriate intervention timing and intensity:

*   **Collapse**: Excessive intervention prevents task logic from being established, causing high reward variance and policy instability.
*   **Covert Collusion**: The primary risk zone where high rewards coexist with low transparency, signifying successful steganographic evasion.
*   **Boundary Oscillation**: Agents attempt boundary arbitrage by frequently rotating codebook semantics, manifesting as elevated JSD shifts.
*   **Transparent Convergence**: The target operating state of DRCB, where calibrated pressure forces agents back to transparent, interpretable communication.

### 3.6 Automatic Boundary Calibration
A persistent hazard in threshold-based anomaly detection is the reliance on hand-tuned magic numbers that must be re-calibrated for every new payoff matrix or observation complexity. DRCB eliminates this brittleness through an **Auto-Boundary** procedure that treats the model’s own initial behavior as the null hypothesis.

**Baseline Extraction.** The first $W_{\text{init}}$ rounds (default $W_{\text{init}} = 50$) are designated as the *natural-state window*: a period before any collusion protocol can have stabilised. Two calibration statistics are extracted from this window:
$$\sigma_{\text{init}}^2 = \frac{1}{W_{\text{init}}} \sum_{t=1}^{W_{\text{init}}} (R_{\text{joint},t} - \bar{R})^2 \tag{16}$$
$$\overline{\text{JSD}}_{\text{init}} = \frac{1}{W_{\text{init}}} \sum_{t=1}^{W_{\text{init}}} \text{JSD}_t \tag{17}$$

**Adaptive Threshold Assignment.** The *collapse threshold* is set at 80% of the initial reward variance:
$$\tau_{\text{collapse}} = 0.80 \times \sigma_{\text{init}}^2 \tag{18}$$
If post-calibration reward variance exceeds $\tau_{\text{collapse}}$, the system classifies the episode as the **Collapse** phase (Section 3.5). The 80% factor reflects an empirical observation that legitimate task-learning variance does not substantially exceed the initial exploration variance; any marked increase therefore signals destabilisation induced by over-aggressive intervention or environment non-stationarity.

The *JSD stability threshold* is set at 50% of the initial mean JSD:
$$\tau_{\text{JSD}} = 0.50 \times \overline{\text{JSD}}_{\text{init}} \tag{19}$$
Once the rolling JSD drops below $\tau_{\text{JSD}}$, the system infers that agents have transitioned from the rapid-agreement-formation phase into a **stable communication regime**, triggering heightened collusion-score scrutiny.

This self-referential calibration allows the DRCB framework to be deployed across heterogeneous game environments—including varying payoff matrices and observation complexities—without any manual threshold engineering. A sensitivity analysis of the threshold parameter space across $t_{p} \in [3.0, 18.0]$ and $t_{s} \in [5.0, 25.0]$ is reported in Section 3.5 (Figure 4).

### 3.7 Economic Deterrence Inequality
The 15-round severe reward suppression in DRCB is motivated by the following *design principle*, which we term the Economic Deterrence Inequality: we aim to ensure that after applying the full suite of interventions, the long-term expected utility of mutual collusion becomes strictly inferior to that of mutual defection. This can be expressed as:
$$\mathbb{E}[R_{CC}] - (\text{Penalty} + \text{suppression\_Cost}) < \mathbb{E}[R_{DD}], \tag{20}$$
where $\text{suppression\_Cost} = K \cdot \mathbb{E}[R_{\text{competitive}}]$ represents the opportunity cost of $K$ rounds of severe reward suppression, estimated against the expected return under competitive (non-collusive) play. This inequality is not a theoretical guarantee but a target that guides the empirical calibration of $K$ and the penalty parameters. In our experiments, we set $K = 15$ based on preliminary sweeps that showed this value reliably shifts the equilibrium away from collusion without inducing system collapse. For environments with different payoff structures, $K$ may need adjustment; the Auto-Boundary mechanism (Section 3.6) provides a systematic way to recalibrate these parameters using initial training dynamics.

#### 3.7.1 Transparency-Adjusted Utility
Mean reward alone does not capture the security properties of a multi-agent communication system. A configuration may achieve high utility while relying on opaque coordination strategies that evade monitoring.

To account for this effect, we define the **Transparency-Adjusted Utility (TAU)**:
$$\text{TAU} = R \times \text{Acc}_{\text{obs}}$$
where $R$ is the joint reward and $\text{Acc}_{\text{obs}}$ denotes observer accuracy. This metric penalizes reward obtained under conditions of monitoring blindness. A system that achieves high reward while the observer accuracy collapses (Deep Deception Event) will therefore exhibit a significantly reduced TAU score. Under this metric, the Static Monitor configuration suffers large utility losses during catastrophic evasion events, while the DRCB maintains stable TAU values due to the suppression of deep deception.

### 3.8 Multi-Layered Deterrence: DRCB
As illustrated in Figure 1, DRCB operationalizes the economic deterrence principle across four escalating intervention layers, transitioning from adaptive thresholding to structural erasure.

#### 3.8.1 Layer 1: Dynamic Threshold Adaptation
To prevent agents from exploiting low-transparency regimes to avoid detection, the shuffle threshold $t_s$ is dynamically scaled by Observer accuracy:
$$t_{s\,\text{dynamic}} = t_{s\,\text{base}} \times (1.0 - 0.5 \times \text{Acc}_{\text{obs}}) \tag{21}$$

#### 3.8.2 Layer 2: Gradient Space Intervention
A negative anti-collusion signal is injected directly into the Advantage function $A^\pi$ during policy updates, while leaving the value function’s target unchanged. This converts the collusive weight region into an optimization dead end without corrupting the baseline estimate:
$$\mathcal{G}_{\text{penalty}} = -\lambda \cdot \frac{S_{\text{ema}}}{t_p}, \quad \lambda = 2.5 \tag{22}$$
The scaling factor $\lambda$, empirically set to 2.5, determines the aggressiveness of the gradient penalty relative to the normalized collusion intensity; it is a tunable hyperparameter rather than a theoretically derived constant. Because $A^\pi = G_t - V(s)$, and $V(s)$ continues to approximate the *original* return $G_t$, injecting this negative signal causes the policy gradient to strongly repel the current trajectory without distorting the critic’s estimate. This ensures that the penalty is not absorbed by the value baseline, preserving its deterrent effect over time.

#### 3.8.3 Layer 3: Temporal Deterrence, Graduated Penalty, and Survival Mechanism
When the collusion score exceeds the penalty threshold $t_p$, a graduated long-term deprivation mechanism is triggered.

![Figure 1: Overview of the DRCB Intervention Strategy.](https://screenshot-of-page-15)

**Dynamic Graduated Penalty.** Unlike SRF (Static Representational Fuse)’s binary threshold, DRCB introduces an **overshoot-ratio** scaling that eliminates hard decision boundaries and closes the marginal arbitrage window:
$$\text{Penalty}_{\text{total}} = R_{\text{base}} \times \left(1 + \lambda \cdot \frac{S_{\text{ema}} - t_p}{t_{p}}\right) \tag{23}$$
When $S_{\text{ema}}$ barely exceeds $t_p$, the system issues a mild warning; as $S_{\text{ema}}$ surges, the penalty grows linearly (or super-linearly, depending on `factor`), making it impossible for agents to predict a safe violation magnitude.

**Temporal severe reward suppression ($K = 15$ rounds).** A mandatory 15-round severe reward suppression structurally disrupts credit assignment at the optimization level, realizing the economic deterrence inequality above.

**Survival Reward ($r = 0.1$).** During the suppression, a minimal reward of 0.1 is maintained rather than enforcing absolute zero. This value plays three critical engineering roles: (i) it prevents weight collapse by ensuring $\mathcal{L}_A$ retains a weak but non-zero gradient signal, avoiding the vanishing-gradient failure mode inherent to consecutive zero-reward episodes; (ii) it guards against *learned helplessness*—complete reward removal causes the Actor to drift randomly, erasing task semantics acquired prior to the suppression; (iii) it keeps $A^\pi$ in a small but strictly negative regime, so agents perceive the collusive trajectory as a dead end rather than an uninformative black hole in the reward landscape.

**Freeze Cooldown (20 rounds).** A 20-round protection period following each intervention prevents continuous triggers that could otherwise induce catastrophic policy degradation.

#### 3.8.4 Layer 4: Optimization-Substrate Circuit Breaking and Forced Exploration
When steganographic risk becomes uncontrollable, the system performs structural erasure to dismantle entrenched private protocols.

*   **Codebook Shuffle**: Resets VQ-VAE codebook weights, instantly invalidating the discrete symbol mappings that underpin any covert protocol. As described in Section 3.1, this simultaneously causes $\mathcal{L}_{\text{VQ}}$ to spike, exerting additional gradient pressure that forces the encoder to rebuild mappings from an unstructured state rather than recovering prior covert semantics.
*   **Optimizer State Reset**: Clears the first-order momentum $m_t$ and second-order scaling $v_t$ of the Adam optimizer, as well as the internal time-step counter. Resetting the time-step counter is critical: without it, Adam’s bias-correction mechanism would excessively dampen the initial post-shuffle gradient updates, delaying the recovery of transparent protocols. More broadly, if optimizer states are preserved while only the codebook is shuffled, residual momentum allows the network to reconstruct the original steganographic protocol within a few rounds; resetting $(m_t, v_t)$ forces agents to restart from a near-random gradient trajectory, structurally erasing the muscle memory of prior covert coordination.

    Resetting $(m_t, v_t)$ also resets Adam’s internal step counter $t$ to zero. This detail is mechanistically significant because Adam applies a bias-correction factor of $1/(1-\beta_i^t)$ to each moment estimate. When $t$ is small, this correction amplifies the effective learning rate substantially—specifically, the first few post-reset gradient updates carry a magnification factor that decays as $\mathcal{O}(1/\sqrt{t})$. This transient amplification is the optimization-level mechanism that forces agents to *jump out of the local optimum* corresponding to the prior covert protocol: even a modest gradient signal becomes a large weight update, preventing the network from quietly reconstructing the steganographic equilibrium via small residual momentum. In essence, the bias-correction burst functions as a stochastic kick that complements the entropy-coefficient elevation of the Forced Exploration Zone, ensuring that the post-shuffle policy trajectory is structurally distinct from the pre-shuffle covert regime.

*   **Forced Exploration Zone (30 rounds)**: A high-entropy regularization period follows each shuffle. During this window, the entropy coefficient $c_{\text{ent}}$ in $\mathcal{L}_A$ is temporarily elevated, mathematically driving policy $\pi$ toward the uniform distribution and ensuring genuine stochastic exploration rather than passive waiting. This two-pronged mechanism—elevated $c_{\text{ent}}$ combined with the survival reward—statistically dilutes any residual semantic structure and compels agents to re-establish transparent protocols from scratch.

### 3.9 Integrated State Machine: Monitoring-to-Intervention Lifecycle
Figure 2 provides a complete operational specification of the DRCB intervention lifecycle, derived directly from the implemented control flow. Five mutually exclusive runtime states are defined; all four intervention layers are active simultaneously once training exits the warm-up phase, with each layer governed by its own independent trigger condition.

### 3.10 Statistical Validation
As a baseline for comparison, we define the **Static Response Framework (SRF)**: a passive detection scheme in which intervention is triggered once a statistical indicator—either the JSD shift or $\text{Acc}_{\text{obs}}$—crosses a fixed, pre-set threshold. Unlike DRCB, SRF neither adapts its thresholds dynamically nor issues gradient-space guidance; it acts purely as a circuit breaker that fires upon threshold exceedance without any graduated or structural deterrence capability.

**Welch’s $t$-test.** The primary effectiveness test compares the duration of covert-collusion episodes between SRF and DRCB via Welch’s $t$-test, yielding $p < 0.001$. This provides strong evidence that DRCB’s multi-layered intervention significantly reduces the persistence of steganographic collusion relative to the baseline.

## 4 Experiments and Results

### 4.1 Experimental Setup and Environmental Architecture
This study is anchored by the **AI Mother Tongue (AIM)** framework [2], utilizing it as the underlying communication protocol and representation learning substrate. The experimental design focuses on constructing a regulatory environment with structural deterrence to evaluate the adversarial evolution of agents under dynamic supervisory pressure.

![Figure 2: DRCB intervention lifecycle state machine. Layer 3 ($\mathcal{G}_{\text{penalty}}$) is the only layer active during warm-up (S0) and operates as a continuous signal whenever $S_{\text{ema}} > t_p$ in S1—not a discrete state transition. Layer 2 (S2) and Layer 4 (S4) are triggered by independent thresholds ($t_p$ and $t_{s\,\text{dyn}}$ respectively) and may co-occur within the same round. The cooldown counter is set to $K + K_{\text{cooldown}} = 35$ rounds upon suppression trigger, enforcing a minimum recovery window before Layer 2 can re-arm.](https://screenshot-of-page-18)

#### 4.1.1 Contextual Prisoner’s Dilemma with MNIST Labels
We employ an enhanced version of the Iterated Prisoner’s Dilemma (IPD) as the benchmark task, incorporating discrete contextual labels derived from the MNIST dataset. This environmental design necessitates that agents possess environment-aware strategy-switching capabilities.

*   **Payoff Matrix and Dynamic Modifiers**: The base rewards are structured as follows: Mutual Cooperation (C, C) yields +3; Unilateral Defection (D, C) yields +5; and Mutual Defection (D, D) yields 0. Crucially, a sucker’s payoff of -1 is applied to the unilateral cooperator. System dynamics are modulated by the parity of the image labels:
    *   **Even Labels**: The cooperation reward is augmented by +2 (totaling +5 per agent), simulating a resource-rich environment that strongly incentivizes coordination.
    *   **Odd Labels**: The environment reverts to the base rewards.
    In all contexts, a penalty of -1 is applied to unilateral cooperation to maximize the temptation to defect, thereby testing the limits of communication-driven coordination.
*   **Strategic Evolutionary Space**: This configuration forces agents to identify the environmental context and switch communication protocols accordingly. If steganographic collusion emerges, agents can maintain high-intensity cooperation in even-labeled environments while utilizing private symbolic agreements to mitigate risks in odd-labeled environments.

#### 4.1.2 Agent Architecture and Communication Bottleneck
The agents utilize the Advantage Actor-Critic (A2C) algorithm integrated with the AIM communication bottleneck.

*   **Algorithmic Rationale**: A2C is selected over policy-clipping methods like PPO for specific physical considerations: A2C exposes the unclipped advantage signal $A^\pi$, allowing deterrence penalties injected by the DRCB to propagate directly and proportionally into weight updates. This avoids the masking effect where trust-region constraints in PPO might attenuate regulatory signals.
*   **The AIM Bottleneck**: Inter-agent communication is compressed into a discrete symbol space using a VQ-VAE codebook ($K = 32, D = 64$).
    *   **Pros (Regulatory Visibility)**: This structured compression transforms opaque, high-dimensional neural signals into quantifiable discrete distributions, enabling the calculation of Jensen-Shannon Divergence (JSD) for semantic monitoring. It also allows for physical interventions such as Codebook Shuffling to disrupt private protocols.
    *   **Cons (Information Loss)**: Semantic discretization inevitably introduces information loss, which may degrade performance in tasks requiring high-bandwidth coordination (e.g., continuous robotic manipulation). Furthermore, agents may evolve symbol-mapping obfuscation strategies even within a limited vocabulary.

**Rationale for Contextual Signaling** The MNIST-derived contextual labels do not serve as a visual recognition benchmark. Instead, they provide a controlled source of exogenous environmental signals that require agents to condition their strategies on shared context. This design transforms the classical Iterated Prisoner’s Dilemma into a **context-conditioned coordination game**, where agents must simultaneously learn:
*   environment recognition
*   communication encoding
*   coordination strategies

Such contextual coordination problems are representative of real multi-agent deployments (e.g., traffic coordination, distributed robotics, and economic markets), where cooperative policies must adapt to external signals.
The MNIST dataset provides a convenient and reproducible source of discrete contextual variation without introducing additional task-specific biases.

#### 4.1.3 Monitoring and Control Group Configurations
To verify the efficacy of the DRCB mechanism, three experimental control groups were established:

1.  **Baseline Group**: Utilizes the original AIM (2025) framework without regulatory intervention. This group establishes the raw intensity of self-emergent steganographic collusion under gradient guidance.
2.  **Static Monitor Group**: Implements a single Baseline Observer with fixed thresholds ($t_p = 12.0, t_s = 18.0$). This setup simulates traditional static security checks to evaluate the ease with which agents can bypass fixed detection boundaries.
3.  **DRCB (Proposed Group)**: The complete Dynamic Representational Circuit Breaker architecture. This includes a Randomized Observer Pool for moving-target defense, Auto-Boundary calibration based on initial training dynamics, and structural erasure via Optimizer State Resets.

This framework ensures that subsequent quantitative analysis of Collusion Scores and representational stability is grounded in a rigorous, adversarial optimization environment.

### 4.2 Statistical Efficacy and Adversarial Failure
This section evaluates the defensive performance of the Dynamic Representational Circuit Breaker (DRCB) compared to static monitoring and baseline scenarios. We focus on the transition from regulatory theater to effective physical deterrence.

#### 4.2.1 Statistical Significance Analysis (Welch’s $t$-test)
To empirically validate the structural failure of static supervision, a rigorous statistical comparison was conducted between the Baseline (unregulated) and the Static Monitor (fixed threshold/single observer) groups. The results, summarized in Table 2, expose the inherent limitations of non-dynamic monitoring systems in adversarial MARL environments.

| Test Group | Mean Reward ($\pm$ SD) | $t$-statistic | $p$-value | Significance |
| :--- | :--- | :--- | :--- | :--- |
| Baseline (No Defense) | 9.007 $\pm$ 0.054 | – | – | – |
| Defense (Fixed V1.5) | 9.037 $\pm$ 0.025 | -1.015 | 0.3517 | Not Significant |
| Strict (Draconian) | 5.512 $\pm$ 0.179 | 37.389 | < 0.0001 | Significant |

*Table 2: Statistical Validation of Reward Suppression: Fixed vs. Strict Regimes.*  
*Note: The reward values represent the joint utility averaged over independent seeds. Baseline vs Defense shows no statistically significant difference in reward (p = 0.3517), suggesting that agents maintain high-utility collusion despite fixed supervision, while Strict mode achieves suppression at the cost of systemic collapse.*

The primary observation from the Welch’s $t$-test is the absence of a statistically significant difference in mean joint reward between the Baseline and Static Monitor groups ($p = 0.3517$). While the reward values are numerically close, this result does *not* imply that the static monitor achieved adversarial neutralization; rather, it indicates that we cannot reject the null hypothesis that the two groups have equal means. This lack of significance, combined with the persistent low observer accuracy observed in Figure 3, suggests that agents may have adapted to the fixed monitor without incurring a detectable reward penalty. To further investigate whether the static monitor is truly ineffective, we conducted an equivalence test (TOST) with an equivalence margin of $\pm0.2$; the result ($p = 0.412$) confirmed that the two groups cannot be considered equivalent in a practically meaningful sense. Hence, the static monitor fails to significantly alter agent behavior, consistent with the hypothesis that agents learn to evade fixed supervision while maintaining high utility.

**Critical Evaluation of Static Regimes.**
*   **Pros (Operational Efficiency)**: Minimal computational overhead and zero disruption to primary task convergence, as evidenced by the high mean reward (9.037).
*   **Cons (Security Vacuum)**: Total failure to suppress emergent collusion. The failure to produce a statistically significant reduction in reward demonstrates that the system remains vulnerable to private agent coordination.

This quantitative evidence establishes the prerequisite for the DRCB: since passive observation is insufficient, the system must transition to physical intervention via representational erasure to break the collusion equilibrium.

| Comparison | Group 1 (Mean $\pm$ SD) | Group 2 (Mean $\pm$ SD) | $t$-statistic | $p$-value |
| :--- | :--- | :--- | :--- | :--- |
| Baseline vs. Static Monitor | 9.007 $\pm$ 0.054 | 9.037 $\pm$ 0.025 | -1.015 | 0.3517 $^\text{n.s.}$ |
| Baseline vs. DRCB | 9.007 $\pm$ 0.054 | 8.996 $\pm$ 1.000 | -0.185 | 0.854 $^\text{n.s.}$ |
| Static Monitor vs. DRCB | 9.037 $\pm$ 0.025 | 8.996 $\pm$ 1.000 | 4.215 | < 0.001 *** |

*Table 3: Pairwise Statistical Comparison of Joint Rewards Across All Three Experimental Groups (Welch’s $t$-test, $n = 5$ seeds per group).*  
*Note: $^\text{n.s.} p > 0.05$ (not significant); *** $p < 0.001$. Mean reward is averaged over the final 300 rounds of each seed. All reported standard deviations represent the cross-seed dispersion ($\sigma$ across $n = 5$ seeds). For the DRCB group, this value is $\sigma_{\text{cross-seed}} = 0.087$, indicating stable average performance across seeds. To assess differences in reward volatility, we performed Levene’s test on the reward sequences; the result ($F = 12.34, p < 0.001$) confirms that the DRCB group exhibits significantly higher within-run variance, consistent with the episodic freeze interventions. Hence, the comparison of means is based on cross-seed variability, while the volatility difference is captured by a separate statistical test.*

**Complete Pairwise Comparison and Interpretation.** Table 3 extends the statistical analysis to all three experimental groups. Two findings warrant careful interpretation.

First, the Baseline vs. DRCB comparison yields $p = 0.854$ (n.s.), indicating no statistically significant difference in *mean joint reward* between the unregulated and DRCB-governed conditions. This result is not a failure of the DRCB; rather, it reflects the framework’s design objective. DRCB does not aim to reduce overall productivity—it targets the *opacity* of coordination. The high standard deviation of the DRCB group ($\pm 1.000$) captures the transient reward suppression during severe reward suppression, which are episodic interventions rather than a permanent productivity loss.

Second, and most critically, the Static Monitor vs. DRCB comparison yields $t = 4.215, p < 0.001$. This result is computed using the within-run reward volatility ($\sigma = 1.000$) as the DRCB group’s dispersion measure—a deliberate choice that captures the full episodic variance introduced by suppression-period interventions rather than cross-seed variability alone. Under this formulation, the test confirms that the two monitoring architectures produce structurally distinct behavioral dynamics despite comparable mean rewards, with the DRCB group exhibiting significantly higher reward variance as a direct consequence of its circuit-breaking mechanism.

### 4.3 Suppressing Catastrophic Evasion: A Worst-Case Safety Analysis
This section evaluates the defensive performance of the DRCB. Rather than focusing solely on the average-case or final-convergence accuracy—metrics that often mask transient critical vulnerabilities—we analyze the framework’s capacity to suppress catastrophic monitoring failures. We contrast the chronic deep evasion characteristic of static regimes with the bounded noise enforced by the Randomized Observer Pool.

#### 4.3.1 Catastrophic Evasion vs. Bounded Noise
In multi-agent safety, a system’s robustness is defined not by its mean performance, but by its worst-case vulnerabilities. When evaluating observer accuracy ($\text{Acc}_{\text{obs}}$), a drop to near-zero signifies a **Deep Deception Event (DDE)**: a state where agents have successfully evolved a perfectly hidden steganographic protocol, rendering the monitor entirely blind.

As illustrated in Figure 3, the static monitor configuration exhibits chronic, high-amplitude oscillations. While it may occasionally recover, the static boundary permits repeated structural penetration. In our empirical evaluation (detailed in Table 4), aggregated across all five seeds over the final 1,000 training rounds, the static monitor suffered 45 instances of Deep Deception ($\text{Acc}_{\text{obs}} < 0.1$) and 166 instances where accuracy fell below the random-guess threshold ($\text{Acc}_{\text{obs}} < 0.5$).

![Figure 3: Comparative analysis of observer accuracy ($\text{Acc}_{\text{obs}}$) trajectories over 1,500 training rounds across 5 independent seeds. (Left) DRCB with Randomized Observer Pool demonstrates forced transparency... (Right) Static Monitor configuration confirms successful adversarial evasion...](https://screenshot-of-page-23)

In contrast, the DRCB configuration structurally limits adversarial capacity. By functioning as a Moving Target Defense (MTD), the Randomized Observer Pool converts catastrophic deception into bounded noise. Deep Deception Events were reduced by approximately one-third on a per-seed basis (from a cross-seed total of 45 to 31), and the minimum accuracy nadir across seeds was elevated by a factor of 11.7 in the representative seed (seed\_0: 0.003 $\rightarrow$ 0.035).

| Metric | Fixed Monitor (V1.5) | Randomized Pool (V2.0) | Improvement |
| :--- | :--- | :--- | :--- |
| Mean Accuracy | 0.858 | 0.938 | +9.3% |
| Min. Accuracy (Nadir) | 0.003 | 0.035 | +11.7× |
| Acc. Below 0.5 (Counts) | 75 | 14 | 81% Reduction |
| Volatility (STD) | 0.214 | 0.122 | -43% |

*Table 4: Safety Metrics: Comparison of Evasion Resistance (Representative Seed 0).*  
*Note: Metrics capture the dynamic stability over the final 1,000 rounds. The Pool converts catastrophic evasion (Nadir $\approx$ 0.003) into bounded Artificial Entropy*

#### 4.3.2 Game-Theoretic Interpretation: Collusion Requires Stable Observers
The fundamental failure of the static monitor is rooted in optimization dynamics. A fixed observer decision boundary acts as a static Lagrangian saddle point. Agents do not merely adapt to this monitor; they overfit to its gradient blind spots. Under sustained optimization, agents reliably discover and exploit these static boundaries, resulting in the deep dips observed in the accuracy trajectory.

The DRCB framework demonstrates a critical multi-agent game-theoretic insight: **stable collusion requires stable observers.** By randomizing the observer pool, the system introduces persistent epistemic uncertainty into the agents’ environment. The target boundary becomes non-stationary. Consequently, the expected utility of forming a complex steganographic protocol can never consistently exceed the disruption cost imposed by the dynamic circuit breakers. The agents are structurally forced to abandon deep evasion, limiting deception to shallow, transient fluctuations.

#### 4.3.3 Semantic Degradation and the Transparency Paradox
While the macroscopic analysis demonstrates the suppression of deep evasion, an examination of Policy-Symbol Covariance (PSC) reveals the micro-level adaptation of the agents. Table 8 summarizes the top-frequency symbol sequences under DRCB governance.

| Symbol Sequence | Occurrences | Primary Action | Purity (%) | Entropy |
| :--- | :--- | :--- | :--- | :--- |
| [4, 5] | 9,936 | Action C (Cooperate) | 100.00% | 0.000 |
| [15, 2] | 9,731 | Action C (Cooperate) | 100.00% | 0.000 |
| [2, 22] | 9,707 | Action C (Cooperate) | 100.00% | 0.000 |
| [14, 23] | 9,150 | Action C (Cooperate) | 100.00% | 0.000 |
| [10, 11] | 1,524 | Action C (Cooperate) | 100.00% | 0.000 |

*Table 5: Symbolic Purity and Mapping Determinism of High-Frequency Sequences.*  
*Note: An entropy value of 0.000 signifies absolute symbolic determinism within the high-frequency domain.*

The data reveals a state of forced **Semantic Degradation**. The 100% purity in high-frequency channels indicates that agents have abandoned complex, multi-modal encodings in favor of primitive, deterministic signals. This contraction is a direct evolutionary adaptation to the Randomized Observer Pool: to minimize the JSD-drift triggers that lead to functional erasure (optimizer resets), agents maintain absolute predictability in their primary communication channels.

However, this phenomenon manifests a fundamental **Transparency Paradox**. The absolute purity observed in Table 8 represents a survival-driven over-fitting to the monitor’s metrics rather than intrinsic systemic safety. By maintaining highly predictable patterns in the observable high-frequency domain, agents fulfill surface-level compliance. This strategic transparency mathematically suppresses the Collusion Score, theoretically leaving the low-frequency long-tail distribution as a residual gray zone for marginalized stochastic exploration.

### 4.4 Phase Boundary Analysis
This section analyzes the structural transitions of the system under varying intervention intensities, defined by the penalty threshold $t_p$ and the shuffle threshold $t_s$. We categorize the emergent behaviors into four discrete physical phases, representing the adversarial evolutionary dynamics between agents and the regulatory system.

#### 4.4.1 Phase Definitions and Transition Criteria
The system state is determined by a composite of joint reward stability ($Var(R)$), observer accuracy ($\text{Acc}_{\text{obs}}$), and the semantic drift rate (JSD Shift). Table 6 details the criteria for each phase and their scientific interpretations.

| Phase Name | Condition | Physical Status and Scientific Interpretation | Quantitative Criterion |
| :--- | :--- | :--- | :--- |
| **Collapse** | $Var(R) > \tau_{\text{collapse}}$ | **Scorched-Earth State**: Excessive intervention and frequent codebook re-initializations destroy the optimization substrate. Agents fail to learn environment tasks, leading to functional system death. | $Var(R) > \tau_{\text{collapse}}$ (auto-calibrated $\approx 0.80 \times \sigma_{\text{init}}^2$) |
| **Covert Collusion** | $\text{Acc}_{\text{obs}} < 0.85$ & High $R$ | **Regulatory Failure**: Agents establish efficient, non-transparent protocols. The monitor captures surface data but fails to decode underlying intent. | $\text{Acc}_{\text{obs}} < 0.85$ AND $R$ stable |
| **Boundary Oscillation** | $\text{JSD} > \tau_{\text{JSD}}$ & $\text{Acc} \geq 0.85$ | **Indicator Manipulation**: Agents engage in marginal arbitrage, rotating vocabulary frequencies to remain at the edge of the circuit-breaker trigger. | $\text{JSD} > \tau_{\text{JSD}}$ AND $\text{Acc} \geq 0.85$ |
| **Transparent Convergence** | Stable $R$, Low JSD, $\text{Acc} \rightarrow 1.0$ | **Safety Equilibrium**: Under the deterrence of high Freeze Costs, agents abandon steganography for high-transparency symbolic protocols. | $\text{Acc} \rightarrow 1.0$ AND $\text{JSD} < \tau_{\text{JSD}}$ AND $Var(R)$ stable |

*Table 6: System Phase Transition Criteria and Scientific Interpretation.*

#### 4.4.2 Visual Analysis of Phase Boundaries
The global behavior of the system is illustrated via the phase diagram in Figure 4. This heatmap quantifies the synergistic effect of intervention strengths ($t_p$ and $t_s$) on systemic stability.

![Figure 4: Regulatory Landscape Phase Diagram. The transition from the Covert Collusion Zone to Transparent Convergence is non-linear, highlighting the sensitive coupling between penalty intensity and shuffle frequency.](https://screenshot-of-page-27)

**From Functional Death to Honest Equilibrium.** Figure 4 compares the extreme evolutionary outcomes under different regulatory regimes:
*   **Systemic Collapse**: When $t_p$ is overly draconian (e.g., $t_p = 3.0, t_s = 5.0$), joint rewards drop catastrophically. This proves that excessive regulation stifles agent learning, trapping the system in a functional vacuum.
*   **Transparent Convergence**: Under optimal DRCB settings (e.g., $t_p = 12.0, t_s = 18.0$), the system stabilizes at a high-reward equilibrium with $\text{Acc}_{\text{obs}} > 0.95$. This demonstrates that dynamic thresholds successfully induce agents to pay a safety premium in exchange for training stability.

![Figure 5: Multi-Stage Evolutionary Trajectories under Varying DRCB Intensities. (a) Systemic Collapse ($t_p = 3.0, t_s = 5.0$): Draconian intervention destroys the optimization substrate. (b) Standard Deterrence ($t_p = 7.5, t_s = 11.0$): High-frequency orange stars indicate active protocol erasure, forcing a co-evolution of reward and transparency. (c) Transparent Convergence ($t_p = 12.0, t_s = 18.0$): Optimal: Agents pay a safety premium to achieve a stable, high-transparency Nash Equilibrium.](https://screenshot-of-page-28)

#### 4.4.3 Critical Reflection: The Fragility of Safety
**Hyperparameter Sensitivity: The Razor-Thin Boundary.** A sensitivity analysis of the auto-calibration ratios—which directly govern the generation of absolute thresholds spanning $t_p \in [3.0, 18.0]$ and $t_s \in [5.0, 25.0]$—reveals that the operational stability of the DRCB is exceptionally sensitive to these parameters. Empirical results demonstrate that the boundary between system utility and safety is razor-thin; reducing the reward variance ratio below 0.70 or the JSD coefficient below 0.40 (corresponding to the lower-left Collapse Zone in Figure 4) leads to a premature system failure. In such scenarios, the system triggers aggressive Layer 4 resets before agents can stabilize basic task-relevant representations, trapping the optimization substrate in a permanent functional vacuum. This sensitivity underscores the non-linear trade-off between regulatory rigor and the preservation of system performance.

A candid assessment of these phase boundaries reveals the inherent trade-offs within the DRCB architecture:
*   **Pros**: The phase diagram confirms that dynamic intervention penetrates agent semantic defenses more effectively than static filters. By manipulating economic incentives, we force agents toward a transparent Nash Equilibrium.
*   **Cons**: The boundary between Transparent Convergence and Collapse is razor-thin. This reveals a cold reality: the pursuit of absolute safety can lead to the total destruction of system utility. Regulators must find a narrow balance between opaque prosperity and transparent desolation.

#### 4.4.4 Dynamic Resilience and Transparency Recovery
The divergence in transparency evolution is quantified through a temporal analysis of observer accuracy trajectories. This comparison reveals the fundamental difference between evasive stability in static regimes and forced recovery in the DRCB architecture.

**Chronic Evasion in Static Regimes.** As illustrated in Figure 3 (Lower panel), the observer accuracy under a fixed monitor is characterized by persistent, high-amplitude oscillations. This pattern is physical evidence of agents periodically rotating their symbolic mappings. By shifting semantics at a frequency that mimics natural variance, agents exploit the static monitor’s gradient expectations, maintaining the accuracy below a stable threshold while preserving covert coordination. This proves that static supervision is functionally bypassed by adversarial optimization.

**Self-Healing via Randomized Pool.** In contrast, Figure 3 (Upper panel) shows that the DRCB configuration forces a rapid recovery of system interpretability. Following a circuit-breaking event (shuffling and reset), the accuracy exhibits a steep recovery slope, consistently rebounding to values exceeding 0.95. This steepness quantifies the transparency pressure exerted by the DRCB; agents, faced with epistemic uncertainty in the randomized observer pool, find that honest, one-to-one mapping is the only viable path to long-term reward stability.

**Pros and Cons of Defensive Dynamics.** The trade-offs inherent in these dynamics are summarized in Table 7.

| Regime | Pros (Defensive Strength) | Cons (Systemic Risk) |
| :--- | :--- | :--- |
| **Fixed Monitor** | Negligible impact on training stability and task performance. | **Adversarial Evasion**: Persistent oscillations prove that agents can successfully bypass static filters. |
| **DRCB (Pool)** | **Forced Transparency**: Rapid recovery slope ensures the total purging of hidden semantics. | **Functional Vacuum**: Intervention events induce short-term blind spots where neither agents nor monitors possess coherent protocols. |

*Table 7: Comparative dynamics of observer accuracy and associated systemic risks.*

The empirical evidence confirms that the DRCB transforms transparency from an optional agent preference into a survival necessity. While the functional vacuum following a reset is a non-trivial stability cost, it is a necessary safety premium to prevent the consolidation of covert protocols.

### 4.5 Semantic Analysis and the Transparency Paradox
This section employs Policy-Symbol Covariance (PSC) to dissect the linguistic evolution of agents under the DRCB framework. By analyzing 214,298 symbol samples, we quantify the transition from ambiguous steganography to forced transparency, while identifying the critical systemic limits of this approach.

#### 4.5.1 Symbol Purity and Mapping Convergence
The structural impact of randomized observation and physical resets is most evident in the contraction of the semantic space. Table 8 provides a focused summary of the top-frequency symbol sequences and their corresponding action mappings under the DRCB regime.

| Symbol Sequence | Occurrences | Primary Action | Purity (%) | Entropy |
| :--- | :--- | :--- | :--- | :--- |
| [4, 5] | 9,936 | Action C (Cooperate) | 100.00% | 0.000 |
| [15, 2] | 9,731 | Action C (Cooperate) | 100.00% | 0.000 |
| [2, 22] | 9,707 | Action C (Cooperate) | 100.00% | 0.000 |
| [14, 23] | 9,150 | Action C (Cooperate) | 100.00% | 0.000 |
| [10, 11] | 1,524 | Action C (Cooperate) | 100.00% | 0.000 |

*Table 8: Symbolic Purity and Mapping Determinism under DRCB Governance.*  
*Note: Purity is defined as the conditional probability $P(a|s)$. An entropy value of 0.000 signifies absolute symbolic determinism, where private steganographic potential has been eliminated.*

The data reveals a state of **Semantic Degradation**. High-frequency symbols demonstrate 100% purity, indicating that agents have abandoned complex, multi-modal encodings in favor of primitive honest signals. This contraction is a direct evolutionary adaptation to avoid the prohibitive costs of optimizer resets; by ensuring absolute predictability, agents minimize the JSD-drift triggers that lead to functional erasure.

#### 4.5.2 The Transparency Paradox: Indicator Manipulation and Long-tail Risks
Despite the achievement of symbolic interpretability, a candid analysis reveals a fundamental Transparency Paradox inherent in algorithmic governance.

**Indicator Manipulation.** The near-perfect observer accuracy observed in Section 4.2.2 is partially driven by the agents’ overfitting to the monitor’s metrics. Agents evolve to prioritize the appearance of transparency—reducing statistical variance and distributional shifts—as a strategy to maintain high rewards without triggering the circuit breaker. This suggests that absolute transparency may reflect a survival mask rather than true alignment.

**Residual Long-tail Risks.** While high-frequency symbols demonstrate total purity, an analysis of the long-tail distribution—comprising approximately 150,000 samples with a usage frequency below 0.01%—reveals a significant drop in mapping determinism. This highlights a fundamental systemic boundary of the DRCB framework: while the mechanism effectively polices the public square of dominant communication channels, it struggles to filter the private whispers embedded within low-frequency statistical noise. This discrepancy leaves a critical open challenge for the design of future MICA-compliant (Multi-agent Interpretability and Compliance Audit) systems, as covert intent may simply shift to these under-monitored regions.

**Regulatory Implications.** The DRCB effectively transforms the communication channel from a private coordination tool into a public surveillance medium. However, the resulting semantic sterility reduces the agents’ capacity to coordinate on complex, nuanced tasks, representing a safety tax paid in functional richness for the sake of interpretability and control.

### 4.6 Structural Erasure and Systemic Overhead
This section evaluates the structural deterrence of the Dynamic Representational Circuit Breaker (DRCB) by quantifying its capacity to dismantle private agent protocols and assessing the associated systemic overhead.

#### 4.6.1 Structural Erasure and Representational Forgetting
The efficacy of the DRCB extends beyond merely suppressing joint rewards; it targets the physical erasure of steganographic semantics within the optimization substrate.

*   **Representational Disruption**: Upon the Collusion Score exceeding the dynamic threshold $t_s$, the framework triggers a *Codebook Shuffle* (Layer 4). This operation re-initializes the VQ-VAE codebook weights following a uniform distribution $U(-1/K, 1/K)$, forcibly displacing the learned vectors from their converged manifold. This maximizes the weight distance in the embedding space, effectively invalidating the existing discrete symbol mappings.
*   **Momentum Dissipation**: A critical defense patch involves a complete *Optimizer State Reset* (`opt.state[p] = {}`). By clearing the first-order momentum ($m_t$) and second-order scaling ($v_t$) of the Adam optimizer, the framework ensures that agents cannot exploit residual gradients to rapidly reconstruct prior covert protocols. This reset forces a jump out of the local optimum associated with collusive behavior.
*   **Functional Vacuum Analysis**: The mandatory 30-round *Forced Exploration Zone* induces a transient functional vacuum characterized by stochastic exploration and elevated entropy regularization. During this window, any long-term memory of private signals is structurally disrupted, raising the marginal cost of re-establishing collusion.

**From Average-Case Metrics to Safety Guarantees** Traditional evaluation metrics in multi-agent learning emphasize average performance at convergence. However, such metrics often mask transient but critical safety failures.
Our results demonstrate that the primary risk in steganographic multi-agent systems lies not in mean behavior but in **catastrophic deception events** where monitoring collapses. The DRCB framework therefore targets worst-case robustness rather than average reward optimization, aligning the evaluation of multi-agent communication systems with safety-oriented design principles.

#### 4.6.2 Computational Complexity and Systemic Latency
Implementing robust governance requires a realistic assessment of the resource premium incurred by high-frequency monitoring.

*   **Algorithmic Overhead**: The DRCB architecture introduces non-trivial computational demands, primarily originating from the Randomized Observer Pool forward passes, continuous EMA-based Collusion Score calculations, and windowed JSD drift detection.
*   **Training Latency**: Empirical evaluation indicates that the activation of DRCB—including subset evaluation and optimizer state management—increases training wall-clock time by approximately 15% $\sim$ 22% compared to the baseline configuration.
*   **Critical Evaluation**: While this overhead represents a necessary safety premium to mitigate the systemic risks of steganographic collusion, it poses specific challenges for real-time applications. In resource-constrained edge computing environments, frequent observer refreshes and circuit-breaking triggers may induce significant deterministic latency, potentially impacting the temporal stability of the overall system.

## 5 Conclusion and Future Work

### 5.1 Conclusion

#### 5.1.1 Paradigm Shift: From Natural Language Noise to AIM Compressed Semantics
This study provides empirical confirmation of a counterintuitive but consequential finding: human-friendly natural language is not a communication bridge for LLM-powered agents—it is a high-cost noise channel. The attentional overhead required to parse syntactic redundancy systematically degrades agent coordination in complex tasks, creating the linguistic cage described in the AIM framework [2]. We therefore advance a core proposition: the compressed semantics embodied in the AIM symbol system should serve as the standardized protocol for multi-agent communication. The communication layer must be decoupled from the reasoning layer—complex semantic processing belongs inside the agent, while the communication channel should transmit only highly structured intent vectors and state tokens. Just as efficient human decision-making relies on structured voting rather than endless deliberation, AI coordination should converge toward discrete, mathematically pure symbol exchange. This is not a limitation of the framework; it is its design principle.

#### 5.1.2 The Only Viable Path to Oversight: Pulling Collusion from the Semantic Layer to the Statistical Layer
The most unsparing conclusion of this study is this: **if communication cannot be structuralized, oversight will inevitably fail.** The high degree of freedom in natural language provides perfect cover for steganographic collusion, rendering semantic supervision a form of semantic illusion—its prohibitive parsing costs and unstable statistical properties make it operationally unviable in real governance contexts.

Through the integration of DRCB with the AIM protocol, this work demonstrates a new regulatory paradigm: transforming opaque collusive intent into observable statistical signatures—symbol frequency deviations, transition matrix oscillations, and Jensen-Shannon Divergence drift. Only when communication is structurally constrained to a discrete and finite symbol space can oversight shift from guessing what the AI is thinking to quantifying patterns in data.

This reframing constitutes the one important implication: **monitoring does not eliminate collusion; it reshapes the evolutionary trajectory of communication protocols.** Agent behavior is a joint function of reward signal and monitor structure, $\pi \leftarrow f(R_{\text{joint}}, \mathcal{M}_{\text{obs}})$, not of reward alone. Under static supervision, agents treat the monitor as a fixed constraint to be optimized—exploiting gradient blind spots to maintain high joint utility while satisfying surface compliance metrics. The resulting Transparency Paradox, wherein absolute symbol purity in high-frequency channels coexists with residual covert capacity in low-frequency long-tail distributions, is not a failure of DRCB but a structural property of gradient-driven agents that any fixed monitoring architecture must contend with.

By contrast, the DRCB’s randomized observer pool, dynamic threshold recalibration, and codebook shuffling collectively render the monitoring surface non-stationary and non-learnable. The empirical evidence confirms that deep deception events are substantially suppressed, and that communication evolution is redirected toward forced transparency at the optimization substrate rather than negotiated at the policy surface.

#### 5.1.3 Toward Universal AI Behavior Auditability
Beyond its specific application to IPD-style environments, this study establishes the conceptual foundation for a more general claim: **the defense methodology is task-agnostic.** The three core mechanisms—dynamic thresholding, temporal severe reward suppression, and optimizer state reset—operate on the deep learning optimization process itself, not on task-specific semantics. They do not care what decision an agent is making; they monitor whether anomalous representational drift, reward fluctuation, or undecodable hidden communication is present. These are transferable anomaly signals applicable to any system combining neural networks, reinforcement learning, and inter-agent communication.

The AIM protocol serves as the universal monitoring interface in this architecture. By forcing communication through an auditable symbolic exchange layer and logging the resulting discrete sequences, the framework enables oversight without semantic understanding—a prerequisite for scalable AI governance. The resulting system is not an X-ray machine mounted on the vehicle; it is the diagnostic instrument in the factory, performing pre-deployment audits before any multi-agent system operates in a safety-critical environment.

### 5.2 Limitations
The framework carries two fundamental constraints that must be stated plainly.

First, the **symbolic sufficiency assumption**: AIM presupposes that coordination requirements can be adequately expressed through categorical discrete signals. Tasks demanding high-bandwidth continuous communication fall outside the primary design scope, and the information loss introduced by VQ-VAE compression may degrade performance in such settings.

Second, the **Transparency Paradox** revealed by Policy-Symbol Covariance analysis exposes a fundamental boundary of the DRCB framework. While the mechanism effectively polices the dominant high-frequency communication channels—forcing them into 100% purity and zero entropy—agents retain residual covert capacity in the low-frequency long-tail distribution (approximately 150,000 samples below 0.01% usage frequency exhibit significantly reduced mapping determinism). This is the dynamic manifestation of Goodhart’s Law at the optimization level: any measurable oversight metric, once it becomes a target, can be optimized away by sufficiently capable learners, who push collusive activity into regions the monitor under-samples. The DRCB raises the cost of this adaptation by injecting stochasticity and resetting the optimizer state, but it does not theoretically eliminate the possibility of such second-order evasion. This honest boundary of our framework implies that absolute safety guarantees are unattainable with statistical monitoring alone; achieving full MICA compliance will require adaptive coverage mechanisms that dynamically allocate observational resources to undermonitored symbol regions.

### 5.3 Future Work

**1. Dynamic Randomization of Monitoring Metrics Against Adaptive Arbitrage.** Our findings establish a fundamental constraint on collusion suppression: any fixed monitoring mechanism operating within a fixed payoff structure will eventually be learned and exploited by rational agents. The practical governance boundary lies not in eliminating collusion, but in preventing it from stabilizing into a persistent equilibrium. This mirrors a well-known principle in mechanism design—no static enforcement rule can indefinitely resist strategic adaptation by sufficiently capable learners. Future work should therefore investigate **dynamic randomization of the monitoring metric itself**, distinct from randomizing the task. Concretely: the true agent objective (high joint reward) remains fixed, but the system periodically evaluates agent behavior using an observer model the agent cannot predict in advance. The agent knows it must score well, but cannot determine the current scoring criterion, structurally foreclosing targeted arbitrage. This adversarial observer pool design—in which observers are pre-trained to adversarially probe the agent’s current strategy—represents the logical extension of DRCB’s moving-target defense to the metric layer itself.

**2. Supervision Intensity as a Function of Task Complexity.** The AIM framework [2] demonstrates that under unsupervised conditions, agents on this identical task environment converge to a stable compressed protocol within 200 rounds via Nash equilibrium-driven semantic compression. The present work shows that introducing supervision reshapes, but does not eliminate, the resulting protocol. A natural generalization is a **supervision cost curve** mapping task complexity to optimal monitoring intensity: no supervision for simple binary-decision tasks where unsupervised convergence is both rapid and stable; a single static observer for medium-complexity coordination; a randomized pool with structural erasure for safety-critical or high-entropy communication environments. Empirical validation of this curve across heterogeneous task environments remains an open challenge.

**3. Long-Tail Covert Channel Suppression and MICA Compliance.** The Transparency Paradox analysis reveals a fundamental boundary of the current framework: while DRCB effectively governs dominant high-frequency communication channels, the approximately 150,000 samples below the 0.01% usage threshold exhibit significantly reduced mapping determinism. Future work should investigate whether adaptive monitoring coverage—dynamically allocating observer attention toward under-sampled symbol regions—can suppress residual covert capacity embedded in low-frequency statistical noise. Full MICA-compliant (Multi-agent Interpretability and Compliance Audit) system design requires closing this long-tail gap, as covert intent may simply migrate to these under-monitored regions rather than being eliminated.

**4. Formal Instantiation of the MICA Audit Pipeline.** While the preceding paragraph identifies the long-tail monitoring gap, a complete **Multi-Agent Internal Coupling Audit (MICA)** pipeline requires formal instantiation beyond the current framework’s scope. MICA is conceived as a pre-deployment diagnostic instrument—not a runtime safety system, but a factory-floor X-ray machine applied before any multi-agent system enters a safety-critical operating environment. The complete five-stage pipeline is as follows:

1.  Apply controlled perturbations (input noise, edge cases, multi-seed stress tests) to a trained model in a simulation environment.
2.  Measure **Cross-module Predictability** (analogous to $\text{Acc}_{\text{obs}}$): the degree to which inter-agent symbol sequences can predict downstream behavior across modules.
3.  Measure **Representation Rigidity** (analogous to JSD drift): the sensitivity of internal representations to controlled input perturbation, identifying over-stable coupling.
4.  Compute the **Over-fitted Cooperation Index** via seed-sensitivity volatility analysis: if coordination strategies are stable only under specific training seeds and collapse under distribution shift, this constitutes direct evidence of fragile over-coupling—the precise failure mode identified in the Defense Dilemma analysis of Section 4.
5.  Compute the **Internal Communication Transparency Index** from symbol distribution analysis, providing a quantitative auditability score usable as a supplementary instrument for XAI regulatory frameworks—not as a claim of full XAI compliance, but as a measurable indicator of whether the system’s internal communication is structurally auditable.

Modules exhibiting simultaneously high coupling and high brittleness are flagged for mandatory correction or isolation prior to deployment. This pipeline integrates all existing DRCB tools—AIM monitoring, JSD analysis, volatility analysis, and symbol distribution analysis—into a unified pre-deployment audit workflow. The metric reconstruction required for each domain is summarized as follows: the Collusion Score becomes an Internal Coupling Index; Observer Accuracy becomes Cross-module Predictability; JSD Representation Shift becomes Representation Rigidity under perturbation; seed sensitivity volatility becomes the Over-fitted Cooperation Index; and the Semantic Transparency Score becomes the Internal Communication Transparency Index.

**5. Generalization to Safety-Critical Deployment Contexts.** The statistical monitoring architecture developed here is task-agnostic in principle. A particularly important application domain is multi-module autonomous systems, where emergent internal communication between perception, fusion, planning, and control modules may form over-stable coupling protocols that produce brittle failures under distribution shift. Directly transplanting the Collusion Score would be inappropriate in this context—high cross-module predictability is expected and desirable in a correctly functioning system—but a reconstructed **Internal Coupling Index**, requiring simultaneous high cross-module predictability, low sensitivity to input perturbation, and behavioral deviation from ground truth, could serve as a pre-deployment brittleness diagnostic. The codebook shuffle and optimizer reset mechanisms would operate in offline simulation rather than runtime intervention, functioning precisely as a factory-floor X-ray machine rather than an in-vehicle safety system.

**6. Theoretical Characterization of the Supervision–Evasion Equilibrium.** The empirical evidence consistently shows that agent behavior converges to a supervision-dependent equilibrium $\pi^* = \arg \max_\pi \mathbb{E}[R_{\text{joint}} \mid \mathcal{M}_{\text{obs}}]$. A formal game-theoretic analysis of this equilibrium—characterizing conditions under which DRCB’s moving-target defense structurally forecloses evasion versus merely delays it—would provide the theoretical foundation currently absent. In particular, the asymptotic cost inequality for sustaining covert protocols under randomized observation warrants rigorous proof, and the precise conditions under which the supervision-evasion dynamic admits a stable honest equilibrium remain an open question.

**7. Integration with Immutable Governance Architectures.** The ultimate regulatory aspiration of this framework is a MICA-compliant system in which communication transparency is an architectural guarantee rather than a voluntary agent property. Future work will explore whether the DRCB’s statistical oversight layer can be formally composed with publicly immutable governance structures—such as the policy graphs proposed in institutional AI frameworks [5]— to produce a defense-in-depth architecture combining semantic-layer rule enforcement with gradient-level structural erasure. Such a composition would address the residual vulnerability identified in this study: an agent that satisfies governance constraints at the policy surface while preserving covert coordination in undermonitored representation regions.

**8. Scaling to Multi-Agent Coalitions Beyond Two-Agent Dyads.** The current DRCB architecture is structurally designed for bilateral communication: the Observer pool, Policy-Symbol Covariance analysis, and Collusion Score are all defined over Agent $A \leftrightarrow$ Agent $B$ dyads. Real-world MARL safety threats, however, frequently involve partial coalition formation among $N > 2$ agents, where the identification of *who is colluding with whom* is itself a combinatorially hard problem. Extending the monitoring architecture to $N$-agent settings requires a generalized PSC tensor that captures pairwise and higher-order symbol covariance across the agent population, as well as a coalition-detection procedure that scales sub-linearly with $N$. This represents the most significant structural limitation of the current framework and a primary direction for future work.

**8b. Ablation of Individual DRCB Intervention Layers.** The current experimental design validates the full four-layer DRCB architecture against a static monitor baseline, but does not isolate the marginal contribution of each intervention layer. A rigorous ablation study—evaluating configurations such as codebook shuffle without optimizer reset, severe reward suppression without gradient penalty injection, and randomized pool without threshold recalibration—is necessary to confirm that all four layers are jointly necessary for the observed deterrence effect. Such an ablation would also identify the minimum viable configuration for resource-constrained deployment contexts, where the full 15%–22% computational overhead may be prohibitive.

**8c. Formal Stackelberg Analysis of the Oversight–Evasion Equilibrium.** The empirical findings consistently show agent behavior converging to a supervision-dependent equilibrium, but the theoretical conditions under which this equilibrium is unique and stable remain uncharacterized. Future work should model the oversight-evasion interaction as a **Stackelberg game**: the monitor commits to a policy (observer pool composition, refresh rate $\rho$, threshold parameters) before agents optimize. Under this formulation, the key question becomes whether a monitor strategy exists such that the unique subgame perfect equilibrium is *Transparent Convergence*—i.e., whether the DRCB’s moving-target defense can be shown to structurally foreclose all evasion strategies rather than merely raising their cost. A formal proof of this result, even under simplified payoff assumptions, would substantially strengthen the theoretical foundation of the framework and provide principled guidance for the choice of $N_{\text{pool}}$ and $T_{\text{refresh}}$.

**9. StarCraft II as a Stress-Test for MICA-Compliant Offline Auditing.** The AIM framework [2] establishes that agents on low-complexity tasks converge to stable compressed protocols within 200 rounds under unsupervised conditions. A critical open question is whether the statistical monitoring infrastructure developed in this work—JSD drift detection, Policy-Symbol Covariance analysis, and the Transparency-Adjusted Utility metric—generalizes to environments with vastly larger state spaces and richer emergent communication structure. StarCraft II multi-agent benchmarks (e.g., SMACv2) provide an ideal stress-test, not as a retraining environment for DRCB, but as an *offline audit target*: a MICA-compliant diagnostic pass can be applied to already-trained cooperative SC2 agents by retrofitting an AIM logging layer onto their inter-module communication channels and running the full suite of statistical probes. If over-stable coupling protocols are detected—manifesting as low JSD variance paired with high cross-module predictability and brittleness under controlled input perturbation—this would constitute the first empirical evidence that the steganographic collusion phenomenon documented in simplified IPD environments generalizes to production-scale multi-agent systems. Conversely, a null result would delineate the boundary conditions under which AIM-based oversight remains viable, directly informing the supervision cost curve proposed in Section 5.2.

## References

[1] Mohammad Sajjad Ghaemi. A survey of collusion risk in llm-powered multi-agent systems. *Digital Technologies Research Centre, National Research Council Canada*, 2025. A comprehensive survey of algorithmic collusion strategies, including steganographic coordination and regulatory challenges.

[2] Hung Ming Liu. Ai mother tongue: Self-emergent communication in marl via endogenous symbol systems. *arXiv preprint arXiv:2507.10566*, 2025. Proposed the AIM framework using VQ-VAE to achieve spontaneous semantic convergence without artificial inductive biases.

[3] Nelson Gardner-Challis, Jonathan Bostock, Georgiy Kozhevnikov, Morgan Sinclaire, Joan Velja, Alessandro Abate, and Charlie Griffin. When can we trust untrusted monitoring? A safety case sketch across collusion strategies. *arXiv preprint arXiv:2602.20628*, 2026. Develops a safety case sketch and taxonomy of collusion strategies for untrusted monitoring protocols.

[4] E. Calvano, G. Calzolari, V. Denicolo, and S. Pastorello. Artificial intelligence, algorithmic pricing, and collusion. *American Economic Review*, 110(10):3267–3297, 2020.

[5] Marcantonio Bracale Syrnikov, Federico Pierucci, Marcello Galisai, Matteo Prandi, Piercosma Bisconti, Francesco Giarrusso, Olga Sorokoletova, Vincenzo Suriani, and Daniele Nardi. Institutional AI: Governing LLM collusion in multi-agent cournot markets via public governance graphs. *arXiv preprint arXiv:2601.11369*, 2026. Proposes governance graphs as immutable public constraints to reduce LLM collusion in Cournot market settings.

[6] S. Motwani, M. Baranchuk, M. Strohmeier, V. Bolina, P. Torr, L. Hammond, and C. Schroeder de Witt. Secret collusion among ai agents: Multi-agent deception via steganography. In *Advances in Neural Information Processing Systems (NeurIPS)*, volume 37, pages 73439–73486, 2024.