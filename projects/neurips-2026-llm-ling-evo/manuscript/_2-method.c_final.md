# 3. Methods

## 3.1 Overview

Each run pairs two agents that share a base model, temperature, persona, and memory budget, with a fixed task order, game-parameter preset, and seed. Our headline question is whether adding the myth-writing task changes either the *level* of cooperation in the trust game (a cooperation-lift hypothesis) or its *across-seed dispersion* (a strategy-consolidation hypothesis); we therefore report median reward deltas and variance changes for myth-present cells against matched game-only controls. Configurations expand from `config/experiments_noisy.yaml` and runs save under `data/json/noise_experiments/`.

![Experimental design. Two same-model agents alternate investor and trustee roles in a repeated trust game. Runs either contain only the game, interleave game before myth-writing, or interleave myth-writing before the game. The action channel can be unperturbed, directionally noised, or neutrally reframed; outputs include both game ledgers and myth text.](figures/fig1_design_schematic.png){#fig-design width=100%}

## 3.2 Repeated trust game

We use a role-swapping repeated Trust Game with endowment $E = \$5$ and multiplier $m = 3$. Each game round has two sequential moves: the investor chooses $s_t \in [0, E]$ to send to the trustee; the trustee receives $3s_t$ and chooses $r_t \in [0, 3s_t]$ to return. Per-round payoffs are

$$
\pi^I_t = E - s_t + r_t,\qquad \pi^T_t = 3s_t - r_t.
$$

Roles alternate every game round, so both agents experience both sides. Unless otherwise noted, runs use ten game rounds, temperature 0.8, a neutral persona, and memory capacity three. The memory buffer is shared across game and myth turns, so interleaving a myth task reduces the number of recent game exchanges still visible in context. On the first game round the investor sees only the rules and endowment; on later rounds each agent is shown the previous round's sent, received, and returned amounts, its payoff, and current cumulative balance.

## 3.3 Myth-writing task

In myth-present conditions, each round also contains a creative-writing turn. On the first myth round, both agents write a 200-word myth on the unconstrained topic `anything`; on later rounds, each agent sees its own previous myth and the partner's previous myth and is instructed to adapt them in its own way. This creates a dyadic iterated-transmission channel separate from the payoff-bearing game. The game prompts never quote the partner's myth directly: cross-task influence flows only through the shared message buffer, where the agent's own prior myth and the partner myth it saw during writing remain in recent context when later game decisions are made.

## 3.4 Conditions

The main draft uses two direct-provider models: OpenAI `gpt-5-nano` and Anthropic `claude-sonnet-4.5`. Gemini pilot outputs are excluded from the main analysis as the run is incomplete.

Task order has three game-containing levels: game-only, game then myth, and myth then game. The main no-noise baseline is `baseline_v4_mem3_direct`, with 15 seeds per model and task order. A neutral-framing pilot, `neutral_framing_v4_pilot`, replaces investor/trustee wording with ROLE A/ROLE B allocation wording, with three seeds per cell.

Noise is an environmental perturbation on the actual transfer ledger: the model chooses a decision amount, the environment optionally perturbs it, and the perturbed value is what arrives, enters payoffs, updates balances, and appears in subsequent prompts. Unperturbed choices are logged separately as `sent_decision` and `returned_decision` (see §A.1 for full ledger bookkeeping). We use three action-channel interventions. Directional uniform noise adds $\epsilon \sim \mathcal{U}(0,k)$ or $\epsilon \sim \mathcal{U}(-k,0)$ to both send and return actions and clamps to range, with $k=5$ for the completed manipulation-check runs. Bootstrap noise replaces the returned amount with the maximum possible return every round, an artificial reciprocity-support diagnostic. A deterministic-max variant (every-round application of the maximum perturbation) was run as a pilot but is under-replicated and excluded from the main analysis (§A.4). Informed variants append a system-prompt note flagging that transfers may be perturbed and that earnings reflect arrived amounts.

Two further channel-design variants probe the cross-task mechanism (results in §4.6). The **partner-myth injection** variant quotes the partner's most recent myth verbatim inside the trust-game prompt, opening a direct cross-task channel that the implicit design (§3.3) keeps closed. The **forced-reasoning** variant adds a system-prompt addendum requiring 2–3 sentences of reasoning before the JSON action, used only with GPT-5-Nano to elicit the reasoning prose Claude emits by default. Both variants run on all four bootstrap noise conditions; partner-myth injection is also crossed with three myth-content directions (neutral, cooperative, adversarial) for the §4.6 factorial.

## 3.5 Measures and analysis

Behavioural measures are computed from raw JSON states: final dyad reward after ten game rounds, average sent and returned amounts (both arrived and decision), return ratios, and per-cell dispersion across seeds. The primary myth contrast is the median final-dyad-reward difference between each myth-present task order and the matched game-only cell within the same model and condition, with non-parametric bootstrap percentile intervals over seeds. Each cell is then labelled by a joint mean-shift × variance-ratio rule. The $\Delta$mean 95% CI is read as *positive*, *negative*, or *null* (strictly above, strictly below, or spanning zero); the myth-to-game variance-ratio 95% CI is read as *consolidation*, *destabilization*, or *null* relative to one. The two axes combine into *lift+consolidation*, *lift*, *consolidation*, *null*, *harmful* (negative mean, null variance), and *destabilizing* (any other combination involving negative mean or destabilization). Cells with $n_{\text{myth}} < 10$ are flagged *missing* and excluded.

Linguistic measures span lightweight and embedding-based proxies, all reproducible without API calls: cooperative-lexicon density, coarse pronoun ratios, lag-1 cross-agent cooperativity correlations, and dictionary-based coinage detection over the myth chain. We additionally compute between-agent embedding cosine similarity per round using `sentence-transformers/all-mpnet-base-v2` (normalized) and a lexical proxy for myth content appearing in game-response prose: the share of game-response tokens matching the agent's own preceding myth vocabulary. LLM-judge similarity is deferred to follow-up analysis. All manuscript tables and figures are generated from raw final JSON files (see §A.2).
