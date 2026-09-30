# Moral flow figure (aggregate lens)

Files: `moral_flow.png` / `moral_flow.pdf` (7.0 x 3.0 in, two-column), `plot_moral_flow.py`,
`prep_children.py` (rebuilds the child table with labels; asserts it reproduces
`moral_uptake.csv` and `moral_uptake_by_task_order.csv`), `children_labels.csv`,
`moral_flow_cells.csv` (every number on the figure).
Run: `uv run --with pandas --with scipy --with numpy --with statsmodels --with matplotlib python prep_children.py && ... python plot_moral_flow.py`

**(a) Design.** Panel a is a two-column flow diagram of all exposures in the 60 mixed 8-agent
myth→game runs: the family of the myth shown (left) into the family of the agent who writes
next (right). Same-family flows run straight across in the family's colour, cross-family flows
cross in grey. Panel b gives the number behind it: excess label match (shown myth minus an
unseen same-family myth) with 95% CIs for every 8-agent cell.

**(b) Takeaway.** In mixed populations, a myth's moral carries to the next myth when the author
is from the reader's own family (+5.9 pts, p = 0.001), not when it is from the other family
(+0.7 pts, p = 0.37).

**(c) Base rate vs excess.** Width is contact volume only (how often a family's myth was shown), never
match rate. Colour encodes exposure class, not a per-ribbon excess: cells are too thin for that
(mixed Gemini→Gemini is 80 children in 5 runs). The raw match rates are printed next to the
excess (67% shown vs 61% unseen; 52% vs 51%) so the reader sees that the base rate is high and
the effect is the gap. Panel b plots only the excess.
Numbers: run-level means (child → run mean → mean over runs), as in `run_summary`. Means are
asserted equal to the committed CSV. p-values are the CSV's Wilcoxon signed-rank tests over run
means. CIs are a run bootstrap (10,000 resamples, seed 20260930).

**(d) Weaknesses.**
- Mixed populations only pair GPT with Sonnet or with Gemini. No Sonnet↔Gemini ribbons exist,
  so "other family" means "GPT ↔ other". The cross-family null could be specific to GPT. The caption must say this.
- The within-family class pools three unequal cells. GPT→GPT (619 children, 20 runs) dominates. Mixed Gemini→Gemini is tiny.
- Panel a's two colour classes carry the claim through their labels. The ribbons do not show the effect
  size themselves, because a 6-point gap cannot be seen in a width.
- The game→myth rows (hollow) share game history, like dyads. Only myth→game is a clean test.
  Mixed same-family game→myth is +3.0, p = 0.11, weaker than myth→game.
- The labels come from the GLM-5.2 judge (κ = 0.54 vs DeepSeek). There has been no human pass yet.
- Fragility found while checking: re-deriving the per-run means from a CSV changes float noise.
  That flips scipy's tie detection, so Wilcoxon p moves (homogeneous myth→game 0.005 → 0.003;
  mixed other-family 0.37 → 0.45). The means are unaffected, and the figure prints the committed CSV's p.

Suggested caption: *Morals spread within model families, not across them. (a) Every time an agent in a
mixed 8-agent population (myth→game) was shown a myth, by author family (left) and reader
family (right). Width is the number of exposures. Mixed runs pair GPT with Sonnet or Gemini
only. (b) How much more often the next myth's moral label matches the shown myth than an
unseen myth by the same family in the same round (percentage points; run means, run-bootstrap
95% CI; Wilcoxon over runs). Game→myth rows share game history and are not a clean test. Dyads
are excluded for the same reason. Labels: GLM-5.2 judge, Arabella Sinclair's 3-label rubric.*
