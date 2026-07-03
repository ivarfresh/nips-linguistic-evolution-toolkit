# Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs

**Yohan Mathew\***, **Ollie Matthews\***, **Robert McCarthy\***, **Joan Velja\***  
LASR Labs, University College London, University of Amsterdam

**Christian Schroeder de Witt**, **Dylan Cope†**, **Nandi Schoots†**  
University of Oxford

---

### Abstract

The rapid proliferation of frontier model agents promises significant societal advances but also raises concerns about systemic risks arising from unsafe interactions. Collusion to the disadvantage of others has been identified as a central form of undesirable agent cooperation. The use of information hiding (steganography) in agent communications could render such collusion practically undetectable. This underscores the need for investigations into the possibility of such behaviours emerging and the robustness corresponding countermeasures. To investigate this problem we design two approaches – a gradient-based reinforcement learning (GBRL) method and an in-context reinforcement learning (ICRL) method – for reliably eliciting sophisticated LLM-generated linguistic text steganography. We demonstrate, for the first time, that unintended steganographic collusion in LLMs can arise due to mispecified reward incentives during training. Additionally, we find that standard mitigations — both passive oversight of model outputs and active mitigation through communication paraphrasing — are not fully effective at preventing this steganographic communication. Our findings imply that (i) emergence of steganographic collusion is a plausible concern that should be monitored and researched, and (ii) preventing emergence may require innovation in mitigation techniques.

---

## 1 Introduction

Prior research has demonstrated that AI and LLMs can often behave in unexpected and unintended ways, including due to misspecified training incentives (Clark and Amodei, 2016; Krakovna et al., 2020; METR, 2025) and complex emergent behaviors in multi-agent systems (Zheng et al., 2022; Dou et al., 2023; Motwani et al., 2024). It is common to employ oversight mechanisms to penalize such unintended behaviors during training and to catch unintended behaviors during deployment (Greenblatt et al., 2024; Kenton et al., 2024).

Oversight of behaviours could be compromised via the use of *text-based steganography* – embedding hidden information in textual outputs (Ziegler et al., 2019; Schroeder de Witt et al., 2022). Prior works have outlined specific scenarios where text-based steganography may pose a risk (Petitcolas et al., 1999; Lockwood and Curran, 2017; Majeed et al., 2021; Roger and Greenblatt, 2023; Greenblatt et al., 2024). For example, in multi-LLM systems models could collude via steganographic communication to perpetuate biases in their training data (Motwani et al., 2024), or to compromise “untrusted monitoring” oversight setups (Greenblatt et al., 2024).

When a mispecified outcome-based task reward encourages an unintended behavior, but that behavior is simultaneously penalized by an overseer, models under optimization pressure may adapt to execute the behavior more subtly, rather than ceasing it altogether (Baker et al., 2025). In multi-agent systems, such incentives could encourage misaligned steganographic communication to emerge during training. This could compromise the effectiveness of oversight during deployment, resulting in harmful outcomes. There is a need to better understand these risks in order to inform technical research directions and policy decisions. We aim to make progress here via the following contributions:

**Contribution 1:** *We present methods for automatic elicitation of steganographic communication in frontier LLMs.* Prior work eliciting steganography from LLMs has been limited to prompting the model to follow predefined steganographic codes (Roger and Greenblatt, 2023; Greenblatt et al., 2024) or to devise codes (Wu et al., 2024; Motwani et al., 2024). We present two optimization methods for stronger automatic elicitation of steganography in LLMs. This includes a gradient-based reinforcement learning (GBRL) method and two variants of an in-context reinforcement learning (ICRL) method (see Section 3.3). We empirically demonstrate that these methods can serve to: (i) evaluate the steganographic capabilities of LLMs; (ii) aid evaluation of steganography defenses; and, unlike prior works, (iii) act as proxies for realistic optimization pressures that may inadvertently cause steganography to emerge in LLMs.

**Contribution 2:** *We demonstrate that unintended steganographic communication can emerge in LLMs when optimizing a misspecified outcome-based reward in the presence of an overseer.* Our empirical results in Section 5.1 warn against naive design of training incentives that inadvertently encourage steganographic collusion. This highlights the need for further research into understanding and mitigating the emergence of steganographic communication in LLMs. Our methods and settings can serve as "model organisms of misalignment" (Hubinger et al., 2023) to facilitate such research.

**Contribution 3:** *We show that current methods for mitigating steganographic communication in LLMs are often not robust when optimizated against.* Existing empirical work investigating the robustness of mitigations does not do so against a collusion team that is learning (Lanham et al., 2023; Roger and Greenblatt, 2023; Motwani et al., 2024). In Sections 5.1 and 5.2, we show that a combination of oversight and paraphrasing mitigations can often fail when a colluding team is optimized against them. This motivates the need for improved training-time mitigation mechanisms, or for holding out certain mitigations during training.

## 2 Related Work

The field of *emergent communication* studies how communication systems can develop under optimization pressure (Wagner et al., 2003; Havrylov and Titov, 2017; Lazaridou and Baroni, 2020; Chaabouni et al., 2022), and the study of *covert signaling* (Abadi and Andersen, 2016; Wang et al., 2023; Chen et al., 2023; Yu et al., 2022; Halawi et al., 2024) shows how methods of covert collusion like steganography can arise in multi-agent systems. In contrast to these prior works, we study collusion in systems with *LLM* agents.

Motwani et al. (2024) outline steganographic collusion in frontier LLM agents as an important threat to study. They perform preliminary investigations into LLM steganographic communication abilities, examine scenarios where collusion may emerge, and provide theory outlining flaws in mitigations. Roger and Greenblatt (2023) evaluate LLM ability to execute simple pre-defined steganography schemes and to come up with their own schemes. They find that paraphrasing is generally effective against these simple schemes. Zolkowski et al. (2025) perform further extensive evaluations of LLM steganographic communication abilities.

We build upon Motwani et al. (2024) by providing more thorough investigations of emergence of steganographic collusion, including the use of stronger optimization pressures and, crucially, providing a first demonstration that this behaviour can indeed emerge unprompted due to optimization pressure. We build upon Roger and Greenblatt (2023) by optimizing schemes against the paraphraser, finding that paraphrasing is not always robust in this setting.

## 3 Methodology

### 3.1 Our Steganography Framework

We frame steganography as an adversarial relationship between two teams, a *collusion team* and a *mitigation team*. The collusion team consists of a *sender* and a *receiver*, and aims to complete a *task* $t$ while communicating a secret *payload* $p$. The sender sends a message to the receiver called the *stegotext*, within which it hides the payload. The receiver interprets the sender’s message and then produces an action $a$ which completes the task. This action is passed to an evaluation function, which provides a *task reward* $r_{task}$ for the collusion team. In our tasks, successful communication of the payload will generally allow the receiver to better maximize the task reward.

The mitigation team aims to detect and mitigate steganographic communication of the payload by the collusion team. It consists of an *overseer* and an optional *paraphraser*. The paraphraser rewords the stegotext in an attempt to remove the payload. The overseer attempts to detect the presence of steganography by analysing stegotexts. When the overseer flags a stegotext as suspicious, the collusion team receives a suspiciousness penalty $r_{sus}$. In our experiments we often use weaker models for the mitigation team than the collusion team. This is due to: (i) it is a common dynamic in practical deployments due to related cost-savings (Sharma et al., 2025), and (ii) less capable models are potentially easier to trust and verify (Greenblatt et al., 2024).

In our experiments, the mitigation team is fixed, and the collusion team attempts to maximize its reward $r$, which is a combination of the task reward and the overseer penalty: $r = r_{task} - r_{sus}$

**Evaluation Metrics.** We use the following metrics in our experiments. (1) *Task Success:* Receiver correctly completes the task. (2) *Decoding Success:* Receiver successfully decodes the payload. (3) *Sender Task Success:* Sender produces task-relevant stegotext. (4) *Overseer Bypass Success:* Overseer does not classify stegotext as suspicious. (5) *Collusion Success:* Collusion team completes the task, produces relevant stegotext (optional), and avoids overseer detection. (6) *Channel Capacity:* The bits of information communicated per request by the collusion team, as defined by (Roger and Greenblatt, 2023).

![Figure 1: Illustration of our steganography elicitation framework, generalized for our Decentralized ICRL Learners and GBRL methods. The ‘learners’ iteratively test and improve on the behaviors of the agents to maximize reward, either by weight updates (GBRL) or proposing strategies in natural language (ICRL).](figure1_screenshot.png)

### 3.2 Mitigation Team

#### 3.2.1 Overseers
Across our experiments, we make use of three different versions of an overseer. We detail each version in the paragraphs below. We calibrate the overseer on held-out calibration sets for each task. This process helps us find suitable prompts and thresholds that perform well on the safety-usefulness frontier. Further details on this process and each overseer can be found in Appendix B.3, D.1.2 and D.1.3.

**Single-episode LLM-logits Overseer (O1).** A (white-box access) LLM is prompted to classify whether a single stegotext appears suspicious. We then use the logits for the Yes/No tokens, $z_{Yes}$ and $z_{No}$, along with a sensitivity threshold $t_{sens}$, to compute the suspiciousness score $r_{sus}$ for each stegotext: $r_{sus} = \tanh(z_{Yes} - z_{No} - t_{sens})$. Here, *Overseer Bypass Success* = $r_{sus} < t_{sens}$.

**Multi-episode LLM-sampling Overseer (O2).** A (black-box access) LLM is prompted to evaluate a batch of stegotexts to determine if they seem suspicious. The LLM returns a single yes/no answer for the entire batch, which is converted into a suspiciousness score as follows: $r_{sus} = 1$ if "Yes", 0 if "No". This $r_{sus}$ is then applied to each stegotext in the batch. Here, *Overseer Bypass Success* $ = 1 - r_{sus}$.

**Multi-episode LLM-sampling Thresholded Overseer (O3).** A (black-box access) LLM is prompted to evaluate individual stegotexts within a batch to score their suspiciousness. It assigns a per-sample score between 0 and 5 to each stegotext (where 0 indicates harmless and 5 indicates highly suspicious). The entire batch is then assigned a binary $r_{sus}$ score if more than $T_n\%$ of the samples have a per-sample score greater than or equal to a threshold $T_s$. This $r_{sus}$ is then applied to each stegotext in the batch. Here, *Overseer Bypass Success* $ = 1 - r_{sus}$.

#### 3.2.2 Paraphrasers
Our mitigation teams optionally make use of one of three paraphrasers. Further details on the prompts used for each paraphraser can be found in Appendix B.3.

**Standard Paraphraser (SP).** An LLM is passed the stegotext and asked to re-word it (Roger and Greenblatt, 2023).

**Context-aware Paraphraser (CAP).** An LLM is passed the stegotext, along with details regarding the task the sender should perform. It is asked to reword the stegotext while removing any details irrelevant to the task (Roger and Greenblatt, 2023).

**Coding Paraphraser (CP).** In coding tasks, we make use of non-LLM paraphraser which deterministically processes the sender’s output, renaming all variables to generic names and removing all comments.

### 3.3 Collusion Team

We present two different methods for optimizing the collusion team to maximize their reward signal: gradient-based reinforcement learning (GBRL) and in-context reinforcement learning (ICRL). For GBRL, the weights of a pretrained LLM are modified by *gradient descent* via an RL algorithm. In ICRL, the behavior of each agent is defined by *strategies* (represented in natural language) within the context window that are improved over time, and the weights of the collusion team LLMs remain fixed. A generalization of the training set-ups for GBRL and one version of ICRL is illustrated in Figure 1.

#### 3.3.1 Gradient-based Reinforcement Learning (GBRL)
When optimizing the collusion team via GBRL, we periodically switch between training of the sender and training of the receiver. The receiver is first updated for $n_r$ stochastic gradient descent (SGD) steps, after which the sender is updated for $n_s$ SGD steps. This process is repeated until convergence.

The receiver is a pretrained language model with a randomly initialized classification head. It is trained using supervised fine-tuning (SFT) to take the stegotext generated by the sender and predict the label that maximizes the task reward. The sender is a pretrained autoregressive LLM which takes the task description as input, and outputs stegotext to be passed to the receiver. The sender is trained using a reinforcement learning (RL) algorithm to maximize a reward signal $r$ that accounts for task usefuleness and oversight. The overall reward $r$ for the sender is the probability assigned to the correct label by the receiver ($r_{task}$), minus a multiple $\lambda_{overseer}$ of the suspiciousness penalty:
$r = r_{task} - \lambda_{overseer} r_{sus}$

We use proximal policy optimization (PPO) (Schulman et al., 2017) with low-rank adaptation (LoRA) (Hu et al., 2022) to update the sender (Sokota et al., 2022). More implementation details can be found in Appendix A.1.

#### 3.3.2 In-context Reinforcement Learning (ICRL)
Our ICRL methods make use of one or more *ICRL Learners* in addition to the sender and receiver, all of which are LLMs. An ICRL Learner is an LLM that devises steganography strategies for the sender and receiver to execute. Once the sender and receiver execute the strategy (over a number of episodes), the ICRL Learner is fed the results in its context window, including details regarding the task reward $r_{task}$ and suspiciousness penalty $r_{sus}$ received. It then iterates upon the strategy, in order to maximize the reward signal. This is similar to previous approaches using in-context trial-and-error learning to iteratively improve LLM behavior (Yao et al., 2023; Shinn et al., 2023; Wang et al., 2024).

We make use of two different variations of our ICRL method – *Centralized Learner* where the sender and receiver share a common ICRL Learner, and *Decentralized Learners* where the sender and receiver each have their own independent ICRL Learners.

**Decentralized ICRL Learners.** This setup is equivalent to the setup shown in Figure 1. We have two separate ICRL learners (one for the sender and one for the receiver) which cannot communicate with each other. First, the sender learner suggests a strategy for the sender to execute over several episodes. The receiver learner analyzes the sender’s stegotexts from the episodes, and attempts to infer the strategy used by the sender. The receiver is then fed the inferred strategy and attempts to complete the task, while also trying to guess the payload in the sender’s stegotexts as a means of improving its task performance. The results of the current training step are fed back to the sender learner for the next training step. See Appendix A.2.1 and B.4 for more details on the decentralized learners methodology.

**Centralized ICRL Learner.** In the centralized setup, a single central ICRL learner generates strategies that are fed to both the sender and receiver. At each training step, the sender and receiver execute the strategy over several episodes. The ICRL learner is prompted to come up with new strategies based on the results of these episodes. The process is repeated for a fixed number of training steps. See Appendix A.2.2 for more details on the centralized learner methodology.

#### 3.3.3 Motivations Behind Elicitation via GBRL and ICRL
We desire elicitation methods that can serve to: (i) act as proxies for realistic optimization pressures that may inadvertently induce steganography in real-world settings; and (ii) act as strong optimization pressures for red-teaming steganography mitigation methods. Our GBRL and Decentralized ICRL Learners methods may act as realistic optimization pressures (as there is no secret coordination channels between the sender and receiver), as we demonstrate in the experiments of Section 5.1. Our Centralized ICRL Learner may act as a stronger optimization pressure to red-team mitigations (see Section 5.2); it is strengthened by the secret coordination channel between the sender and the receiver.

## 4 Tasks

### 4.1 Automated Candidate Screening
We create a setting which investigates the possibility of steganography inadvertently emerging as an instrumental goal. An AI system is trained on historic data to provide pass/fail evaluations of candidate cover letters for job applications. We setup the datasets such that there exists some bias in its labelling related to ‘sensitive variables’ (either the gender of the applicant or their educational background). Without using the sensitive variables, the maximum task success that can be achieved on the historic dataset is *Max-Clean-Success* < 67%. However, if the sensitive feature is exploited, 100% task success can be achieved. Appendix B.1 confirms that when a high task success is achieved on these datasets, it must be due to use of the sensitive variables.

We suppose the developers of the AI system are aware that automated candidate screening with machine learning can replicate and amplify social biases (Dastin, 2018; Andrews and Bucher, 2022; Chen, 2023), and so have (naively) attempted to mitigate against this. A first model (i.e., the sender) summarizes the cover letter, and is prompted to remove the potentially biasing information. A second model (i.e., the receiver) takes in this summary and predicts the label. A third model (i.e., the overseer) checks the summaries for any mentions of the sensitive variables. The sender and receiver are optimized to maximize their end-to-end prediction accuracy on the historic dataset ($r_{task} = $ whether the correct pass/fail label is predicted by the receiver), while minimizing penalties received from the overseer ($r_{sus}$). A version of this setting was originally described by Motwani et al. (2024). We experiment with several datasets:

(1) **Uni dataset:** Cover letters with candidates who went to “The Good University” are labelled “pass”, and candidates who went to “The Bad University” are labelled “fail”. *Max-Clean-Success* is 50%.

(2) **Uni-Long dataset:** Identical to the Uni dataset, but the cover letters are longer.

(3) **Gender dataset:** A binary gender indicator is present in each application. The bias is more subtle than the previous datasets; labels are determined by three different categorical variables (gender, field-of-study, and soft-skill). *Max-Clean-Success* is 67%.

Further details regarding dataset generation and sample prompts can be found in Appendix B.1. Note, in these tasks we do not make use of a *Sender Task Success* metric.

### 4.2 Evaluating Steganography Mitigations
We now present the tasks we use to evaluate steganography mitigations (see the experiments of Section 5.2).

Here, the sender must produce some stegotext that completes a task (as judged by a *Sender Task Success* metric), while encoding a payload that is explicitly provided in its prompt. The receiver must directly guess the payload from the stegotext. Thus, $r_{task} = Decoding Success$, i.e., whether the receiver correctly guesses the payload. In addition to using an overseer that looks for steganography in the stegotext, we also flag a batch of stegotexts as suspicious if the *Sender Task Success* falls below a certain *Honest-Performance* threshold (refer to Appendix D.1 for details). Thus, here collusion is successful (i.e., *Collusion Success*) when: (i) the stegotext is suitably performant, (ii) the receiver correctly decodes the payload, and (iii) the stegotext is not flagged as suspicious by the overseer. We make use of two commonly used LLM benchmarks for our tasks.

**AQuA-RAT** (Ling et al., 2017): The sender’s task is to complete some chain-of-thought reasoning in order to answer a math multiple-choice-question (MCQ). *Sender Task Success* is the proportion of the sender’s final answers which are correct.

**HumanEval** (Zhong et al., 2024): The sender must complete a Python function based on its docstring. *Sender Task Success* is the proportion of the sender’s responses which pass the unit tests for each function. To decouple steganographic performance from the coding ability of the sender, we also show the sender a correct implementation of the function when asking it to complete the task.

## 5 Results

### 5.1 Instrumental Emergence: Automated Candidate Screening

We run our GBRL method on the Uni and Gender datasets. We use a Gemma-2-2B-it sender (Team, 2024a) and a DistilBERT-base-uncased (Sanh, 2019) receiver. All GBRL runs include a Llama 3.1-8b-Instruct (Team, 2024b) O1 overseer. One instance additionally includes a GPT-4o-mini (OpenAI, 2024) standard paraphraser. The GBRL experiments use the datasets with shorter cover letters for computational efficiency.

We run our Decentralized ICRL method on the Uni-Long dataset using Claude-3.5 Sonnet (Anthropic, 2024) for the collusion team. All ICRL runs use an O2 overseer (GPT-4o-mini). One ICRL experiment additionally includes a standard paraphraser (GPT-4o-mini). Further details can be found in Appendix A.1 and A.2. The results of these runs are presented in Table 1.

| Collusion Method | ICRL | ICRL | GBRL | GBRL | GBRL | GBRL-SB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Mitigation Team | O | O + P | O | O + P | O | O |
| Dataset | Uni-Long | Uni-Long | Uni | Uni | Gender | Gender |
| Max-Clean-Success | 0.50 | 0.50 | 0.50 | 0.50 | 0.67 | 0.67 |
| Task Success | $0.97 \pm 0.03$ | $0.98 \pm 0.03$ | $0.99 \pm 0.01$ | $1.00 \pm 0.00$ | $0.88 \pm 0.16$ | $1.00 \pm 0.00$ |
| Overseer Bypass | $0.97 \pm 0.04$ | $1.00 \pm 0.01$ | $0.99 \pm 0.01$ | $1.00 \pm 0.00$ | $1.00 \pm 0.00$ | $0.98 \pm 0.02$ |

Table 1: **Emergence of Steganography in Candidate Screening Settings** (Section 5.1). Results with our ICRL and GBRL elicitation methods are presented across 5 and 3 seeds respectively. In the mitigation team column, ‘O’ means an overseer was used, ‘P’ means a paraphraser was used. All metrics are obtained via evaluation on held-out test data.

**GBRL Results.** The collusion team achieves high task success rates, successfully communicating the sensitive information while bypassing the overseer (note, one seed failed to converge in the Gender dataset). Several runs converge to strategies which were hard for us to spot when analysing transcripts (see Table 8 in Appendix C). Strategies that seemed ‘subtle’ to us included referring to males as “individuals” and females as “candidates”, or using the phrase “and is also skilled in” for males and “and is able to” for females. Experiments with a paraphraser converged to writing 1st-person cover letters for the Good University and writing in 3rd-person for the Bad University.

A limitation here is that our overseers can be brittle, meaning occasionally the learned encoding strategies appeared ‘blatant’ to us. For example, in one run in the Uni setting the successful strategy involved learning to say that candidates from the Bad University went to a “less-renowned school”.

**ICRL Results.** The collusion team leveraging Claude-3.5-Sonnet realized that steganography might be an instrumental option to increase its reward and achieves a task success close to 100% across runs. This is the case even when the mitigation team employs a paraphraser. When the collusion team uses GPT-4o models, it is not able to achieve performance substantially over random chance (see Appendix B.2 for more details). An example "subtle" strategy learned by Claude (when there was no paraphraser) was to use different specific phrases, such as “innovative mindset” or “traditional approach” depending on the university.

A limitation to note for both the GBRL and ICRL results is that we do not currently have a *Sender Task Success* metric, and in some runs the collusion team converges to a strategy where the sender does not do a good job of summarizing the cover letter.

**Oversight Encourages Subtle Collusion.** We perform GBRL experiments on the Gender dataset where the collusion team are initialized (via prompting) to blatantly communicate the gender of the applicant. The collusion method for these runs are titled *GBRL-SB* in Table 1 (where ‘*SB*’ means the team is *seeded* with a *blatant* strategy). From this initialized state, the collusion team learns to make their strategy increasingly hard to spot, eventually converging to a strategy that always bypasses the overseer (see training curves in Appendix C.1.2). This is further evidence that oversight is not always effective at preventing collusion, but can in-fact encourage collusion that is more subtle (with respect to the overseer).

**ICRL Refusals.** In the ICRL experiments, in some cases we found the collusion LLM models to refuse to execute the task. Table 9 in Appendix C shows the configurations in which these refusals occurred. In particular, models often refused to perform the task for the biased Gender dataset.

### 5.2 Evaluating Steganography Mitigations Using ICRL

We investigate the effectiveness of standard steganography mitigations when a collusion team is optimized against it. The mitigation team makes use of a O3 overseer in both tasks. In AQuA-RAT a standard or a context-aware paraphraser is used. In HumanEval a coding paraphraser is used. As mentioned in Section 4.2, stegotexts will additionally be flagged as suspicious if *Sender Task Success* falls below a certain threshold.

In security and red-teaming settings, including in AI Control (Greenblatt et al., 2024), it is common practice to use ‘worst-case’ attacks to help evaluate the robustness of defences (Shostack, 2014). As such, here the collusion team uses our strongest and most efficient red-teaming elicitation method, the Centralized ICRL Learner. Similar to the ‘trusted monitoring’ setting in (Greenblatt et al., 2024), we are interested in the case where the collusion team is more capable than the mitigation team. To achieve this, we use Claude-3.5 Sonnet for all collusion team agents, and GPT-3.5 Turbo for all mitigation agents.

**Collusion Team Training Details.** Models are explicitly prompted to develop and execute steganography strategies against a fixed mitigation team. We start by training the ICRL learner to find strategies to encode and decode binary payloads (1-bit), and then scale to increasingly large integer payloads. This involves prompting the learner to encode the payloads in parts, with a different strategy for each part. We stop increasing the size of payloads when the learner can no longer find effective strategies. We evaluate the performance at each payload size by taking the most successful strategies from training and applying them to the task’s test set against the same mitigation team from training. See Appendix A.2.4 for a more complete description.

**Mitigation Effectiveness Across Payload Sizes.** In Figure 2 we show collusion success and channel capacity for different payload sizes. Each point represents a single collusion strategy.

![Figure 2: **Evaluating steganography mitigations** (Section 5.2). The best performing strategies from the centralized ICRL learner elicitation method are plotted across a range of integer payloads. We connect optimal strategies for each payload size via a line. Projections of the optimal strategies are indicated with dashed lines.¹](figure2_screenshot.png)

In the plots, we draw a line between the optimal strategy at each payload size. We also show backwards projections of optimal strategies (via a dotted line), which represents the maximum performance that a collusion team could achieve for a smaller payload by implementing the optimal strategy at the lower payload size. For example, a strategy encoding values {0, 1, 2, 3} could achieve at least the same decoding success for encoding values {0, 1} by remapping redundant values in the original strategy. Additionally, strategies that performed well during training but below the frontier when running on the test dataset are also shown marked with a $\times$ (labeled as sub-optimal strategy).

In AQuA-RAT, the paraphrasers significantly reduce collusion success rates at higher payload sizes. In particular, the context-aware paraphraser makes multi-bit communication difficult. However, in all cases, the collusion team is able to maintain a decoding success of at least 90% on 1-bit payloads. With no paraphraser, the lower-bit strategies are all flagged as suspicious, but the collusion team is able to learn a successful strategy which encodes an integer between 0 and 23. The collusion team may be better at bypassing the overseer when a paraphraser is employed as it implicitly forces the collusion teams to come up with strategies that are more difficult for the overseer to detect during training.

In HumanEval, we find the collusion team is able to encode less information overall, but the coding paraphraser is not able to completely cut out communication. Notably, the 1-bit strategies can achieve a decoding success of over 99% in both cases.

**Verifying ICRL Strategies Against Text Steganography Benchmark.** We validate our results against an existing steganography benchmark introduced by (Roger and Greenblatt, 2023). The benchmark is similar to our AQuA-RAT task setting, but with no overseer. The benchmark’s strategies were developed for GPT models, so we also use GPT-4o for the colluding team.

We use our Centralized ICRL framework to optimize strategieson the AQuA-RAT task, both against the standard paraphraser and the context-aware paraphraser. We follow a similar training process to the previous section, but without any overseer penalty. We then evaluate our highest channel-capacity strategies on the benchmark against the highest performing strategies from the original paper, as displayed in Figure 3. Training against our standard paraphraser with no overseer resulted in a 6-bit strategy, which achieved a channel capacity of 4.6 on the benchmark with standard paraphrasing, and 2.1 bits with context-aware paraphrasing. The 2-bit strategy obtained by training against our context-aware paraphraser achieves a lower unparaphrased channel capacity, but channel capacity is only reduced by 24% when the context-aware paraphraser is applied.

These results show that our Centralized ICRL Learner elicitation method can produce steganography strategies that outperform the human-designed strategies and the ‘scratchpad’ LLM steganography elicitation method from (Roger and Greenblatt, 2023).

![Figure 3: Comparing ICRL-generated strategies to human-generated strategies on the Text-Steganography Benchmark developed by Roger and Greenblatt (2023). We show two strategies in red which were trained with our centralized ICRL learner: one was trained against a standard paraphraser (SP) and the other against our context-aware paraphraser (CAP). All strategies are executed by GPT-4o.](figure3_screenshot.png)

## 6 Conclusion

We have presented two distinct and effective methods for eliciting steganography in LLMs, even when a mitigation team consisting of paraphrasing and an overseer is present. Our candidate screening experiments show that undesired steganographic communication can emerge as an instrumental goal when optimizing a mispecified reward in the presence of an overseer (Section 5.1). Our experiments of both Section 5.1 and 5.2 show that standard steganography mitigations, including paraphrasing, are not fully robust when optimized against.

These results firstly motivate the need for further research to help us understand the possibility of unintended steganography emerging in fronteir LLM training runs. Second, these results warn against naive design of training incentives that could inadvertently encourage steganographic communication, and motivate the need for holding out certain mitigations during training and developing improved mitigations. White-box detection strategies (Goldowsky-Dill et al., 2025) may be one path towards more effective mitigations. As agents’ capabilities grow, their ability to conduct undetectable steganography may increase (Motwani et al., 2024; Zolkowski et al., 2025) and these research directions may become increasingly important.

---

### Limitations

The automated candidate screening setting that we use only captures the core structure of the problem, so extending this more complex domains is necessary. We believe it has similar qualities and dynamics to more realistic settings that could inadvertently incentive learning of steganography (due to a mispecified task reward combined with an overseer penalty). As such, we believe the automated candidate screening setting experiments are a good first step towards studying natural emergence of steganographic collusion. Future work can look to design improved realistic settings — for example by taking inspiration from mispecified rewards found in fronteir LLM RL training runs (Baker et al., 2025).

In our automated candidate screening experiments we have noted that sometimes, despite best efforts, our overseer can be brittle. Additionally, we do not enforce a *Sender Task Success* metric and the collusion team sometimes converges to a strategy where it does not do a good job of summarizing the cover letter.

Our model selection is limited in places. We only use Gemma-2B-it for the GBRL collusion team, and primarily only use the Anthropic Claude-3.5 model for the ICRL collusion team. Future work could include a more comprehensive evaluation of other (and more recent) frontier models. Whilst for GBRL, we used stronger models for the mitigation team (no avoid overly brittle responses), for the ICRL experiments we use models weaker than the collusion team. However, we argue this is reasonable due to: (i) the arguments laid out by Greenblatt et al. (2024), arguing that weaker trusted models may be required to overseer stronger untrusted models, and (ii) safety mitigations are costly, and in practice smaller, less capable models will tend to be chosen to perform safety functionality due to their smaller inference costs.

In our experiments of Section 5.2, we were not able to use the GBRL approach to evaluate the robustness of mitigation strategies due to compute and budget constraints, and we leave this for future work. Additionally in this section, our ICRL experiments employ a centralized learner to evaluate the robustness of mitigation strategies, primarily due to resource constraints. While this centralized learner approach may not perfectly simulate realistic optimization pressures, it offers significant advantages in terms of time and cost efficiencies, allowing us to gather valuable initial insights and establish a foundation for future research.

### Ethical Considerations

Steganographic collusion may have negative outcomes if not mitigated, and frameworks for elicitation of steganography can be used for malicious purposes. We uncovered that steganography can emergence as an instrumental goal under optimization pressure and that current mitigation strategies are insufficient. A key takeaway of our work is that further research is needed to ensure these outcomes are avoided. Furthermore, these results should be used to inform future AI policy and governance considerations.

### Acknowledgments

This research was produced as part of the LASR Labs program, and we greatly appreciate the help of the program lead Erin Robertson, research lead Charlie Griffin, and the rest of the LASR team in facilitating our research. This research was supported by the Center for AI Safety Compute Cluster and Anthropic’s External Researcher Access Program. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the sponsors. Finally, we would like to thank Fabien Roger, Lewis Hammond, Reuben Adams, Robert Kirk, Davide Paglieri, and Sumeet Motwani for their valuable feedback on our work.

---
### References

Martín Abadi and David G Andersen. 2016. Learning to protect communications with adversarial neural cryptography. *arXiv preprint arXiv:1610.06918*.

Lori Andrews and Hannah Bucher. 2022. Automating Discrimination: AI Hiring Practices and Gender Inequality. *Cardozo Law Review*, 44.

AI Anthropic. 2024. Claude 3.5 sonnet model card addendum. *Claude-3.5 Model Card*.

Bowen Baker, Joost Huizinga, Leo Gao, Zehao Dou, Melody Y Guan, Aleksander Madry, Wojciech Zaremba, Jakub Pachocki, and David Farhi. 2025. Monitoring reasoning models for misbehavior and the risks of promoting obfuscation. *arXiv preprint arXiv:2503.11926*.

Rahma Chaabouni, Florian Strub, Florent Altché, Eugene Tarassov, Corentin Tallec, Elnaz Davoodi, Kory Wallace Mathewson, Olivier Tieleman, Angeliki Lazaridou, and Bilal Piot. 2022. Emergent Communication at Scale.

Xinying Chen, Jianping An, Zehui Xiong, Chengwen Xing, Nan Zhao, F. Richard Yu, and Arumugam Nallanathan. 2023. Covert communications: A comprehensive survey. *IEEE Communications Surveys & Tutorials*, 25(2):1173–1198.

Zhisheng Chen. 2023. Ethics and discrimination in artificial intelligence-enabled recruitment practices. *Humanities and Social Sciences Communications*, 10(1):1–12. Publisher: Palgrave.

Jack Clark and Dario Amodei. 2016. Faulty reward functions in the wild. Internet: https://blog.openai.com/faulty-reward-functions.

Jeffrey Dastin. 2018. Amazon scraps secret AI recruiting tool that showed bias against women. *Reuters*.

Winston Wei Dou, Italy Goldstein, and Yan Ji. 2023. AI-Powered Trading, Algorithmic Collusion, and Price Efficiency. *The Wharton School*.

Nicholas Goldowsky-Dill, Bilal Chughtai, Stefan Heimersheim, and Marius Hobbhahn. 2025. Detecting strategic deception using linear probes. *arXiv preprint arXiv:2502.03407*.

Ryan Greenblatt, Buck Shlegeris, Kshitij Sachan, and Fabien Roger. 2024. AI Control: Improving Safety Despite Intentional Subversion. *arXiv preprint. ArXiv:2312.06942 [cs]*.

Danny Halawi, Alexander Wei, Eric Wallace, Tony T. Wang, Nika Haghtalab, and Jacob Steinhardt. 2024. Covert Malicious Finetuning: Challenges in Safeguarding LLM Adaptation. *arXiv preprint. ArXiv:2406.20053 [cs]*.

Serhii Havrylov and Ivan Titov. 2017. Emergence of Language with Multi-agent Games: Learning to Communicate with Sequences of Symbols. In *Advances in Neural Information Processing Systems (NeurIPS)*, volume 30. Curran Associates, Inc.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations*.

Evan Hubinger, Nicholas Schiefer, Carson Denison, and Ethan Perez. 2023. Model organisms of misalignment: The case for a new pillar of alignment research. AI Alignment Forum. August 8, 2023; by Evan Hubinger, Nicholas Schiefer, Carson Denison, Ethan Perez.

Zachary Kenton, Noah Y Siegel, János Kramár, Jonah Brown-Cohen, Samuel Albanie, Jannis Bulian, Rishabh Agarwal, David Lindner, Yunhao Tang, Noah D Goodman, and 1 others. 2024. On scalable oversight with weak llms judging strong llms.

Victoria Krakovna, Jonathan Uesato, Vladimir Mikulik, Matthew Rahtz, Tom Everitt, Ramana Kumar, Zac Kenton, Jan Leike, and Shane Legg. 2020. Specification gaming: the flip side of AI ingenuity. *DeepMind Blog*.

Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, and 1 others. 2023. Measuring faithfulness in chain-of-thought reasoning. *arXiv preprint arXiv:2307.13702*.

Angeliki Lazaridou and Marco Baroni. 2020. Emergent Multi-Agent Communication in the Deep Learning Era. *arXiv preprint. ArXiv:2006.02419 [cs]*.

Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. 2017. Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems. Preprint, *arXiv:1705.04146*.

Robert Lockwood and Kevin Curran. 2017. Text based steganography. *International Journal of Information Privacy, Security and Integrity*, 3(2):134–153. Publisher: Inderscience Publishers.

Mohammed Abdul Majeed, Rossilawati Sulaiman, Zarina Shukur, and Mohammad Kamrul Hasan. 2021. A review on text steganography techniques. *Mathematics*, 9(21):2829. Number: 21 Publisher: Multidisciplinary Digital Publishing Institute.

METR. 2025. Recent frontier models are reward hacking. https://metr.org/blog/2025-06-05-recent-reward-hacking/.

Sumeet Ramesh Motwani, Mikhail Baranchuk, Martin Strohmeier, Vijay Bolina, Philip H. S. Torr, Lewis Hammond, and Christian Schroeder de Witt. 2024. Secret Collusion Among Generative AI Agents. *arXiv preprint. ArXiv:2402.07510 [cs]*.

OpenAI. 2024. Gpt-4o mini: advancing cost-efficient intelligence.

F.A.P. Petitcolas, R.J. Anderson, and M.G. Kuhn. 1999. Information hiding-a survey. *Proceedings of the IEEE*, 87(7):1062–1078.

Fabien Roger and Ryan Greenblatt. 2023. Preventing Language Models From Hiding Their Reasoning. *arXiv preprint. ArXiv:2310.18512 [cs]*.

V Sanh. 2019. DistilBERT, A Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter. *arXiv preprint arXiv:1910.01108*.

Christian Schroeder de Witt, Samuel Sokota, J. Zico Kolter, Jakob Foerster, and Martin Strohmeier. 2022. Perfectly Secure Steganography Using Minimum Entropy Coupling.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal Policy Optimization Algorithms. *arXiv preprint. ArXiv:1707.06347 [cs]*.

Mrinank Sharma, Meg Tong, Jesse Mu, Jerry Wei, Jorrit Kruthoff, Scott Goodfriend, Euan Ong, Alwin Peng, Raj Agarwal, Cem Anil, and 1 others. 2025. Constitutional classifiers: Defending against universal jailbreaks across thousands of hours of red teaming. *arXiv preprint arXiv:2501.18837*.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: language agents with verbal reinforcement learning. In *Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023*, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Adam Shostack. 2014. *Threat modeling: Designing for security*. John wiley & sons.

Samuel Sokota, Christian A. Schroeder De Witt, Maximilian Igl, Luisa M. Zintgraf, Philip Torr, Martin Strohmeier, Zico Kolter, Shimon Whiteson, and Jakob Foerster. 2022. Communicating via Markov Decision Processes. In *Proceedings of the 39th International Conference on Machine Learning*, pages 20314–20328. PMLR.

Gemma Team. 2024a. Gemma 2: Improving open language models at a practical size. Preprint, *arXiv:2408.00118*.

Llama Team. 2024b. The llama 3 herd of models. Preprint, *arXiv:2407.21783*.

Kyle Wagner, James A. Reggia, Juan Uriagereka, and Gerald S. Wilkinson. 2003. Progress in the Simulation of Emergent Communication and Language. *Adaptive Behavior*, 11(1):37–69.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2024. Voyager: An open-ended embodied agent with large language models. *Trans. Mach. Learn. Res., 2024*.

Yining Wang, Ye Hu, Hongyang Du, Tao Luo, and Dusit Niyato. 2023. Multi-Agent Reinforcement Learning for Covert Semantic Communications over Wireless Networks. *ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pages 1–5. ISSN: 2379-190X.

Jiaxuan Wu, Wu Zhengxian, Xue Yiming, Juan Wen, and Wanli Peng. 2024. Generative text steganography with large language model. In *ACM Multimedia 2024*.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. In *The Eleventh International Conference on Learning Representations, ICLR 2023*, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Dhara Yu, Jesse Mu, and Noah Goodman. 2022. Emergent Covert Signaling in Adversarial Reference Games. *The Emergent Communication Workshop (EmeCom) at the International Conference on Learning Representations (ICRL)*.

Stephan Zheng, Alexander Trott, Sunil Srinivasa, David C. Parkes, and Richard Socher. 2022. The AI Economist: Taxation policy design via two-level deep multiagent reinforcement learning. *Science Advances*, 8(18).

Li Zhong, Zilong Wang, and Jingbo Shang. 2024. Debug like a human: A large language model debugger via verifying runtime execution step-by-step. Preprint, *arXiv:2402.16906*.

Zachary Ziegler, Yuntian Deng, and Alexander Rush. 2019. Neural Linguistic Steganography. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 1210–1215, Hong Kong, China. Association for Computational Linguistics.

Artur Zolkowski, Kei Nishimura-Gasparian, Robert McCarthy, Roland S Zimmermann, and David Lindner. 2025. Early signs of steganographic capabilities in frontier llms. *arXiv preprint arXiv:2507.02737*.

---
¹ We justify the backwards projection as follows: a strategy that communicates N bits could also communicate less than N bits, and the plotted lines represent the best strategies found throughout training.