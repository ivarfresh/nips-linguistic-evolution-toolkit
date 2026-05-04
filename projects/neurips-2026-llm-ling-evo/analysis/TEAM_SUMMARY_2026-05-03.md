# LLM linguistic evolution - Sunday night update

Hi all - quick update before tomorrow's 11am meeting. The short version is that most myth-writing effects are still null or small, but GPT-5-Nano x bootstrap noise has turned into a sharper mechanism result. In that cell, the implicit myth channel makes cooperation worse: adding myths drops the round-10 dyad balance from about 66.6 to 58.9 and roughly doubles across-seed dispersion. But when the trust-game prompt explicitly includes a paragraph under the header "Your partner's most recent story", cooperation returns to the game-only baseline. The surprising part is that this paragraph does not need to be the partner's actual myth. A shuffled myth from another dyad, and even a length-matched encyclopedic filler paragraph placed under that same partner-story header, reverse the bootstrap drop about as well as the real partner myth.

That means the original "partner's specific myth as a common-knowledge anchor" story is too strong. The new reading is more like: bootstrap noise creates a contradictory or unstable game signal, and the model stabilises when the game prompt contains a coherent paragraph explicitly framed as the partner's recent story. This is partly consistent with a visibility/common-knowledge interpretation, but not with a strong claim that the partner's own narrative content is the active ingredient.

## What changed

The main follow-up was a control matrix around the original partner-myth intervention. In that intervention, the trust-game prompt quotes the partner's most recent myth under the heading "Your partner's most recent story." It reversed the bootstrap harm pattern, so we asked whether that reversal depended on the specific partner-authored narrative.

The key numbers below are for the bootstrap-noise condition where agents play the game and then write myths each round:

| Variant | Mean round-10 balance |
|---|---:|
| Baseline + myth, no injection | 58.9 +/- 11.0 |
| Real partner myth shown as partner story | 68.1 +/- 3.5 |
| Myth from another dyad shown as partner story | 68.2 +/- 4.5 |
| Encyclopedic filler paragraph shown as partner story | 70.5 +/- 2.3 |
| Agent's own myth shown as partner story | 65.4 +/- 6.0 |
| Cooperative myth + injection | 69.8 +/- 2.9 |
| Adversarial myth + injection | 66.6 +/- 5.5 |

The encyclopedic filler condition is the cleanest stress test. It is just length-matched reference prose, not a myth, not produced by either agent, not cooperative, and not narratively linked to the dyad. It still restores cooperation to the game-only baseline when presented under the partner-story header. So content identity, content direction, and narrative status are not carrying most of the effect. The agent's-own-myth condition is weaker, which suggests the partner-framed prompt position matters, but the real partner myth, shuffled myth, filler paragraph, cooperative myth, and adversarial myth are all close enough that I do not think we can defend a "partner narrative specifically" claim.

## Boundaries

The reversal is bootstrap-specific. In no-noise cells, injection is basically null: cooperation is already around 70-72 and does not lift. In positive-noise cells, everything is near the ceiling around 74, so there is no headroom. In negative-noise cells, injection also does not restore cooperation: the cells sit around 40-43 regardless of whether the prompt shows the real partner myth, a shuffled myth, filler text, or the agent's own myth. So the effect is not "more text in prompt helps cooperation" in general. It appears only in the regime where the implicit myth channel made cooperation worse in the first place.

The neutral-framing sweep is also useful. Replacing investor/trustee wording with ROLE A/ROLE B lowers GPT-5-Nano cooperation substantially, which opens headroom, and the myth effect becomes larger there. Claude moves less and still shows little myth effect. This helps rule out the idea that the whole result is just an artefact of the investor/trustee labels, while also reinforcing that model and framing matter.

## Current submission story

My proposed abstract-level story is now:

> Inter-agent language is not a uniform cooperation switch. Across most cells it is null or modest. But under bootstrap noise, where the implicit myth channel destabilises GPT-5-Nano cooperation, showing agents a paragraph labelled as the partner's recent story restores cooperation to the game-only baseline. The mechanism is conditional and channel-level: visibility / coherent reference frame, not the semantics of the partner's own myth.

This is a narrower but cleaner claim than the earlier Chwe/common-knowledge version. Chwe is still relevant as background for visibility and shared reference, but the data falsify the stronger version where the partner's specific narrative is the causal object.

## Running now

I have a GPT-5.5 / Gemini missing-condition sweep running overnight. This is the model-expansion item from Ivar's priority list, using the existing v4 pattern: positive, negative, and bootstrap noise; game-only, game-then-myth, and myth-then-game; 15 seeds per cell. Gemini positive was partially complete and is being backfilled. GPT-5.5 and Gemini negative/bootstrap are new. Smoke tests passed for all six sets. No Claude runs have been launched tonight.

I would treat these model-expansion results as robustness and boundary evidence for the submission, not as the abstract spine, unless they are unexpectedly decisive.

## Decisions for tomorrow

1. **Story for the abstract.** I recommend mechanism-led: conditional language effects, with the bootstrap/channel-opening result as the sharp contribution.

2. **Claude follow-up.** We may want more Claude at some point, but I suggest deciding after the 11am meeting. The minimal submission-relevant branch would be Claude x bootstrap baseline first; only if Claude also shows bootstrap myth destabilisation would we add the real-partner-myth and encyclopedic-filler variants. If Claude is null or saturated, we stop and report the mechanism as established in GPT-5-Nano.

3. **Prompt variants.** Ivar's remaining prompt concerns are less-constrained myths and myths written before the agent knows about the game. I think the feasible version before submission is a small GPT-5-Nano x bootstrap battery only: unconstrained myth prompt, myth-first blind system prompt, and the combination. Include only if clean; otherwise appendix or post-submission.

4. **Who owns what.** Mario: abstract framing. Edward: whether "coherence anchor / channel-opening" reads as a defensible mechanism rather than a prompt artefact. Ivar: whether the proposed prompt-variant battery actually addresses the Apr 24 concern. Alexandra / Arabella / Jane: clarity pass once the results/discussion reframe is merged.

## Proposed 11am agenda

- Lock the one-sentence abstract spine.
- Decide whether Claude is necessary before full-paper submission.
- Decide whether the prompt-variant battery belongs in the main submission or appendix/post-submission.
- Assign abstract and draft review roles with same-day turnaround.

I will keep the overnight sweep running and report what lands before the meeting.
