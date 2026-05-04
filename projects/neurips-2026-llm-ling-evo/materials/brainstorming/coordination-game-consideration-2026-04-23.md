# Should we add a coordination game to the testing ground?

**Date:** 2026-04-23
**Context:** ~2 weeks to NeurIPS deadline. Manuscript sections at `c_final` stage. Question raised mid-writeup.

## The hunch (Aron's framing)

- LLMs are known to be especially bad at coordination games.
- Intuition that stories/communication might *especially* help with coordination.
- So: worth adding a coordination game to the testing ground alongside the trust game?

## First-pass take (abstract, before looking at manuscript)

Theoretically well-grounded (Lewis signaling, Schelling focal points — coordination is the canonical place where shared conventions pay off), but risks being *too* on-the-nose. "Transmission helps coordination" is nearly definitional, so the interesting finding has to be sharper than "it helps" — ideally a coordination failure mode LLMs currently exhibit in a characterizable way (e.g., can't converge on referents for novel objects, can't sustain a convention under population turnover) that stories specifically rescue.

Main tradeoff framed as scope creep vs. theoretical completeness. Provisional recommendation: only if one well-chosen variant (signaling game with novel referents seemed best-aligned with the linguistic-evolution framing), and watch for ceiling effects once any shared history accrues.

## Updated take (after reading manuscript)

Three things changed the picture:

### 1. The paper is already about coordination — just in the right way

§4 discussion already frames myth-writing as a **"coordination commitment device"**, and the variance-reduction finding is interpreted as dyads locking in on a shared strategy rather than mean cooperation going up. The intro draft (¶4) calls narratives *"compact, transmissible, adaptable coordination devices."* So the coordination hunch is already load-bearing in the theoretical frame — just routed through the trust game rather than a literal coordination game.

### 2. The trust game is doing real work *because* it's a hard test

The core contribution is "language has measurable effects even in a game where it isn't structurally necessary for cooperation." A literal coordination game is the *easy* case — communication obviously helps. Running one risks replacing a non-obvious finding (language acts as a coordination device in a cooperation game) with an obvious one (language helps in a coordination game), which could dilute rather than strengthen the argument. The hard-test choice is the paper's teeth.

### 3. Timeline reality

Deadline early May. Sections at `c_final` stage. Scope section explicitly states "Trust game only (not donor or public goods game)." Adding a game ~2 weeks out means rewriting §2, re-running experiments (10 seeds × models × task orders × noise conditions), regenerating figures, and reworking the results narrative. Even if coordination is exactly the right follow-up, shoving it into this paper risks half-baked results and a messier story.

## Where to put the idea instead

### Discussion §future work — one concrete paragraph
> "The variance-reduction-as-coordination-commitment interpretation predicts that in a literal coordination game (e.g. referential signaling with novel objects, or stag hunt with communication channel), the myth-writing effect should be larger and show up in *mean* performance rather than only variance. LLMs are known to underperform on coordination without shared history [cite], which would give the effect behavioural headroom the trust game design doesn't have."

This actually *strengthens* the current paper — it turns the variance finding into a prediction about a class of games rather than a standalone result.

### Toolkit direction
Add coordination game to `games/` as infrastructure for a follow-up paper. Natural next step given `base_game.py` abstraction.

### Possibly worth a sentence in intro
About *why* trust game specifically was chosen over coordination (hard-test logic). Makes the scope choice look deliberate rather than arbitrary.

### One version that *could* go in this paper (optional)
Post-hoc analysis: does the dyad's variance reduction across the 10 rounds predict how fast they'd coordinate if you extended play? Within-existing-data check, no new experiments, could go in appendix. Even this is optional.

## Net take

The hunch is right, the timing is wrong, and the current paper is already cashing in the coordination intuition in a more sophisticated way than a literal coord game would. Hold it for v2.
