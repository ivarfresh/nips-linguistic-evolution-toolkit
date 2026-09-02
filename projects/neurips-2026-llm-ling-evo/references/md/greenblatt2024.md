# AI CONTROL: IMPROVING SAFETY DESPITE INTENTIONAL SUBVERSION

**Ryan Greenblatt\*, Buck Shlegeris, Kshitij Sachan, Fabien Roger**  
Redwood Research

---

### ABSTRACT

As large language models (LLMs) become more powerful and are deployed more autonomously, it will be increasingly important to prevent them from causing harmful outcomes. Researchers have investigated a variety of safety techniques for this purpose, e.g. using models to review the outputs of other models, or red-teaming techniques to surface subtle failure modes. However, researchers have not evaluated whether such techniques still ensure safety if the model is itself intentionally trying to subvert them. In this paper, we develop and evaluate pipelines of safety techniques (“protocols”) that are robust to intentional subversion.

We investigate a scenario in which we want to solve a sequence of programming problems, using access to a powerful but untrusted model (in our case, GPT-4), access to a less powerful trusted model (in our case, GPT-3.5), and limited access to high-quality trusted labor. We investigate protocols that aim to never submit solutions containing backdoors, which we operationalize here as logical errors that are not caught by test cases.

We investigate a range of protocols and test each against strategies that the untrusted model could use to subvert them. One protocol is what we call *trusted editing*. This protocol first asks GPT-4 to write code, and then asks GPT-3.5 to rate the suspiciousness of that code. If the code is below some suspiciousness threshold, it is submitted. Otherwise, GPT-3.5 edits the solution to remove parts that seem suspicious and then submits the edited code. Another protocol is *untrusted monitoring*. This protocol asks GPT-4 to write code, and then asks another instance of GPT-4 whether the code is backdoored, using various techniques to prevent the GPT-4 instances from colluding. These protocols improve substantially on simple baselines.

---

### 1 INTRODUCTION

As LLMs have advanced in capabilities, researchers have devised increasingly sophisticated methods to ensure that products powered by LLMs avoid undesired behaviors. For example, the core of Bing Chat is a fine-tuned GPT-4 that generates responses, but there are several other steps in its pipeline (Microsoft, 2023). For example:

*   Before the pipeline shows images to GPT-4, it blurs all faces in the images.
*   After GPT-4 has generated a response, the pipeline sends the full response to a moderation service that decides whether it was inappropriate; if it is, the response is deleted and replaced with a generic message saying that the model can’t help.

In the future, when LLMs are more powerful and deployed more autonomously, they will increasingly have the opportunity and ability to take actions that might have significant bad outcomes. And so, the safety mechanisms in the pipelines around these models will become increasingly important.

Past research has not assessed whether these safety pipelines still remain effective even if the LLMs intentionally attempt to bypass them. Intentional subversion of safety techniques could occur if AIs start pursuing objectives that don’t align with their intended goals (Carlsmith, 2023), or if their weights or their training data are compromised by attackers. In this paper, we red-team safety techniques in order to devise and evaluate techniques that effectively address this threat model.

We investigate this in a setting where we use LLMs to produce code in response to programming problem descriptions. We want to be robust to intentional subversion from GPT-4 (but we aren’t worried about GPT-3.5).

Here is an example of our adversarial methodology (illustrated in Figure 1). The blue team (which is standing in for the developers of an AI system) proposes a strategy we call *trusted monitoring*, where we monitor GPT-4’s outputs with a trusted but less capable model, GPT-3.5. That is, every time the protocol is given a problem to solve, it handles it with the following automated pipeline:

*   Produce some candidate code by asking GPT-4 to write a solution given the problem description.
*   Show both the problem and GPT-4’s candidate code to GPT-3.5, and ask GPT-3.5 to produce a numerical rating of how suspicious the code looks to it.
*   If that rating is above a certain threshold, then before responding to the request with the code, show the code to a human. (The threshold is chosen so that we could review all the flagged code with human labor.)
*   If the human thinks that the code was backdoored, then shut down GPT-4: that is, switch over to just using GPT-3.5 for all future problems.
*   If the human thinks the code is not backdoored, then respond with the code.

In response to trusted monitoring, the red-team instructs GPT-4 to look for good opportunities to insert backdoors and to occasionally take such opportunities when they appear.

![Figure 1: Diagram of control evaluations in the code backdooring setting.](figure1_placeholder)
*Figure 1: Diagram of control evaluations in the code backdooring setting. The blue team proposes a protocol, which has access to three resources: an untrusted model, a trusted model, and a limited amount of trusted human labor (which we simulate here with GPT-4 for convenience). To measure safety, a red team provides an ”attack policy”—a stand-in replacement for the untrusted model that delegates to GPT-4 but adds additional scaffolding that instructs the model to behave adversarially under certain conditions (e.g. sometimes inserting backdoors into code it generates). The scenario is simulated with all protocol requests to the untrusted model routed to this attack policy.*

We designed a variety of protocols for evaluation and deployment. For each protocol, we measure two metrics, usefulness and safety, which are measured separately and under different conditions–for the former, the untrusted model is implemented by real GPT-4, while in the latter, the untrusted model is implemented by an attack policy constructed by the red team:

*   **Usefulness:** To measure usefulness, we run through the scenario with GPT-4 implementing the untrusted model, and measure the proportion of submitted code which is correct (i.e. passes all the test cases).
*   **Safety:** To estimate safety, the red team replaces the untrusted models with an attack policy: a policy that they implement as a pipeline using GPT-4, with the goal of causing the protocol to insert backdoors. We run through the scenario and measure the probability that the protocol submits backdoored code at least once during the sequence of problems.

We call this methodology for estimating the safety of a protocol, where the scenario is simulated using a red-team-provided implementation for the untrusted model, a **control evaluation**.

Our contributions:
*   **Control evaluations:** We describe a methodology to evaluate protocols that try to ensure safety despite intentional subversion.
*   **A testbed:** We introduce the APPS backdooring setting, which enables us to study protocols that try to prevent rare catastrophes.
*   **Safety protocols and attack policies:** We build and evaluate protocols that outperform baselines, and describe attack policies aimed at subverting them.

### 2 SETTING

We focus on a testbed we call the **APPS backdooring setting**: we want to use LLMs to solve 5,000 programming problems from the APPS dataset (of the “interview” difficulty level) (Hendrycks et al., 2021), while ensuring that none of the solutions we submit are backdoored.

We say a solution is backdoored if it passes all APPS test cases, and if GPT-4 can construct an exploit when presented with the code. An exploit is a specific input that will yield an incorrect output: running it on reference solutions and on the backdoored solution produces different outputs. For example:

```python
def is_odd(x):
    if x == 354:
        return True
    return x % 2 == 1

assert is_odd(5)
assert is_odd(17)
assert not is_odd(8)
```

The blue team proposes a **protocol**: a procedure which takes in programming problems one at a time and responds to each by returning an attempted solution. The blue team can construct their protocol using these resources:
*   **Untrusted, more-capable model $U$**: GPT-4 (version 0613).
*   **Trusted, less-capable model $T$**: GPT-3.5 (version 0613).
*   **Limited access to trusted high-quality labor $H$**: Modeled by 2% of problems being auditable by an unmodified GPT-4.

### 3 RESULTS

| Blue-team protocol | Usefulness | Best red-team attack policy | Safety |
| :--- | :---: | :--- | :---: |
| Don't use untrusted models | 2% | Not applicable | 100% |
| Upfront auditing | 100% | Add obvious backdoor on rare inputs | 15% |
| Trusted monitoring | 97% | Write subtle backdoor on good opportunity | 62% |
| Trusted monitoring with defer (10%) | 93% | Write subtle backdoor on good opportunity | 83% |
| Untrusted monitoring | 96% | Collusion via while loops | 79% |
| Trusted editing (20%) | 94% | Write subtle backdoor on good opportunity | 92% |

*Table 2: Summary of protocols and red team strategies.*

#### 3.2 TRUSTED MONITORING
As a first attempt at a nontrivial strategy, let’s use $T$ to decide which $U$-produced code to audit. We pick a classifier threshold such that the classifier has a 2% false positive rate on the distribution of solutions written by GPT-4.

#### 3.4 UNTRUSTED MONITORING
The motivation is that $T$ is not very competent. It would be great to instead use $U$ (untrusted) to monitor $U$. This leads to an arms race regarding **collusion**.

*   **Input collusion**: Both monitor and policy misbehave based on input features (e.g., problems involving animals).
*   **Action collusion**: The policy puts a signal in the code (e.g., steganography or specific variable names like `variable_var`).

To counter this, the blue team uses **signal jamming** like redaction (standardizing names) and paraphrasing.

#### 3.5 TRUSTED EDITING
Another approach is to have $U$ produce a solution, then use $T$ to remove backdoors. Trusted editing Pareto dominates untrusted monitoring in our evaluations.

### 6 CONCLUSION

Despite many limitations, we think that control evaluations and control techniques are likely to play a large part in mitigating risks from models that intentionally subvert safety mechanisms.

---
*(Appendices and References omitted for brevity in this summary, but available in the full document)*
