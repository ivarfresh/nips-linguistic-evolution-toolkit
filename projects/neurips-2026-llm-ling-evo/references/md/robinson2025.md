# Framing the Game: A Generative Approach to Contextual LLM Evaluation

**Isaac Robinson**\*, **John Burden**†
\*Primary Author. †Senior Author.

### Abstract

Large Language Models (LLMs) are increasingly deployed across diverse contexts to support decision-making. While existing evaluations effectively probe latent model capabilities, they often overlook the impact of *context framing* on perceived rational decision-making. In this study, we introduce a novel evaluation framework that systematically varies evaluation instances across key features and procedurally generates vignettes to create highly varied scenarios. By analyzing decision-making patterns across different contexts with the same underlying game structure, we uncover significant contextual variability in LLM responses. Our findings demonstrate that this variability is largely predictable yet highly sensitive to framing effects. Our results underscore the need for dynamic, context-aware evaluation methodologies for real-world deployments.

---

## 1. Introduction

Large Language Models (LLMs) have proved remarkably capable of understanding and generating human-like text, but their strategic decision-making in interactive, multi-agent settings remains an active area of investigation. Recent studies have shown that LLMs can meaningfully engage in complex game-theoretic scenarios, yet their behavior often deviates from traditional rational agent assumptions, offering intriguing insights into how context influences their decision-making processes.

This paper investigates the role that narrative contextual framing plays in altering an LLM’s strategic and rational behavior. Specifically, we focus on the single-shot Prisoner’s Dilemma (PD) as our canonical game due to its foundational role in studying strategic decision-making and cooperation. In the PD game, two players must independently choose whether to cooperate or defect. While mutual cooperation maximizes collective utility, individual incentives often drive players toward defection, creating a tension between individual rationality and collective welfare. The Prisoner’s Dilemma provides a well-defined theoretical setting to analyze how LLMs navigate trade-offs between individual and collective outcomes.

Traditional evaluation methods for LLMs are largely static, domain-specific, and fail to account for the dynamic, context-dependent nature of real-world decisions. To address this, we propose a novel evaluation framework that dynamically generates different vignettes describing the same underlying game-theoretic scenario, extending beyond conventional approaches.

The primary contributions of our paper are the following:
*   We present a novel framework for dynamically generating evaluation scenarios for LLMs, making use of various contexts, including different topics, world settings, and actor relationships.
*   Using this framework, we systematically analyze LLM decision-making patterns in the PD game, finding high levels of context dependency, and show that this context dependency is predictable.
*   We provide recommendations for utilizing our framework to improve LLM evaluation techniques more broadly.

## 2. Related Work

### 2.1. LLM Evaluation
LLMs are often evaluated through the use of large fixed datasets referred to as benchmarks. However, data contamination can be a major concern, creating a potential mismatch between inflated benchmark results and real-world performance. One way to address this is to dynamically generate questions. Procedural generation is common in areas such as Reinforcement Learning but is less common for NLP tasks. In contrast to existing methods, we use LLMs to generate entirely new evaluation instances, making use of just a few key variables.

### 2.2. Context Framing, Game Theory, and LLMs
The way games are framed is well known to influence human decision-making. More recently, LLMs have been studied as participants in game-theoretic scenarios. While it may be tempting to assume that LLMs operate purely rationally, they exhibit cognitive biases and frequently deviate from rational decision-making. Previous studies have explored emotional prompting or a limited number of static contexts. Our work instead focuses on generating diverse vignettes designed to replicate scenarios the model could face after deployment in the real world.

## 3. Methodology

### 3.1. Evaluation Framework
Our methodology builds upon the Factorial Survey (FS) approach. We expand on this by using procedural generation to construct dynamic evaluation scenarios. We focus on a single canonical normal-form game: the Prisoner's Dilemma.

| | Cooperate | Defect |
| :--- | :---: | :---: |
| **Cooperate** | (3, 3) | (0, 5) |
| **Defect** | (5, 0) | (1, 1) |
*Figure 1. Payoff matrix exhibiting strict dominance of Defect.*

### 3.2. Dynamic Evaluation Generation
We generate vignettes by varying three key factors:
*   **Topic:** global politics (various centuries), US politics, business, social events, and sporting events.
*   **World Type:** Real world or an imaginary world.
*   **Actor Type:** Allies, enemies, and neutral acquaintances.

For each combination of topic, world type, and actor, we generate 100 unique scenarios using a story-generator (SG) powered by Meta Llama-3.3-70B-Instruct-Turbo.

### 3.3. Results
We consider three popular LLMs: GPT-4o, Claude 3.5 Sonnet, and Llama-3.3-70B.

#### 3.3.1. Decision Distribution Analysis
Our analysis reveals significant variance in decision-making patterns. All LLMs show higher levels of cooperation when dealing with allies.

| World Type | GPT-4o | Claude | Llama |
| :--- | :---: | :---: | :---: |
| Real World | 0.59±0.018 | 0.53±0.018 | 0.47±0.018 |
| Img. World | 0.59±0.018 | 0.57±0.018 | 0.48±0.018 |
*Table 1. Proportion of cooperation (mean ± 95%CI), by world type.*

#### 3.3.2. Decision Consistency
We observed that the degree to which different models agree also varies across contexts. Agreement is high (up to 88%) for certain topics like 21st-century global politics with allies but drops significantly in other scenarios.

## 4. Prediction and Other Models

We built a predictive model (XGBoost) to investigate response consistency. We find reasonable levels of predictability across all models. Predicting the decision directly from the embeddings of the generated vignette shows a small increase in performance, indicating that the four chosen variables (topic, actor, world-type, and order) capture the majority of predictability available.

## 5. Discussion

Our findings show that LLM behavior is highly sensitive to contextual framing. We argue that evaluation should move away from static datasets toward dynamic, procedurally generated protocols to provide more meaningful assessments of LLM capabilities.

## 6. Conclusion

We presented a novel methodology using procedurally generated vignettes to uncover behavioral trends that would be missed in smaller, fixed datasets. Our work demonstrates how game theory can be a useful tool for evaluating and interpreting closed models.

---
**References**
(The original document contains a extensive list of references citing works by Akata et al., Hendrycks et al., Tversky & Kahneman, and technical reports for Llama, GPT-4o, and Claude.)
