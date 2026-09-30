# Moral-spread figure candidates, 2026-09-30

Three prototypes for "how morals spread through myths in the network of agents",
built on the section-3 moral labels in `../linguistic_analysis_20260923/`. One will
be promoted to `analyses/` and the paper; the others stay here for reference.
Each folder has its script, figure and `NOTES.md` (design, takeaway, weaknesses).

| Folder | Idea | Main figure |
|---|---|---|
| `network/` | Two single 8-agent runs as rounds × agents networks; edges = myth shown → myth written, coloured when the moral was kept; panel c = shown vs unseen match over all 45 runs | `moral_lineage_network.png` |
| `flow/` | Who reads whose myth (family → family) in mixed populations, plus the shown-minus-unseen excess for every 8-agent cell | `moral_flow.png` |
| `contagion/` | Moral mix per round for Gemini and Sonnet as GPT joins the population, plus the clean myth→game hop test | `moral_contagion_stacked_myth_game.png` |

Moral colours (shared): generous `#D9A400`, fair `#0b5394`, cautious `#B2182B`;
family colours unchanged. Scripts read the gitignored tables in
`data/analysis/linguistic_20260923/`; `flow/prep_children.py` regenerates
`children_labels.csv` (not committed, 1.6 MB).
