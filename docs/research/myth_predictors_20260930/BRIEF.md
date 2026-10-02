# Shared brief: what in the myths predicts cooperation? (2026-09-30)

PI question (Ivar): our moral labels ("be generous / fair / cautious") barely predict
cooperation. Is that because the LLM judges are bad, or because the norms in myths
genuinely don't drive cooperation? Find the most likely myth-based predictors of
cooperation across ALL September run data. His hypotheses:
1. Alignment of the interacting agents' NORMS (not whole-myth similarity) improves cooperation.
2. Seen myths that mention a higher send value (concrete number or abstract, e.g. "give all")
   increase cooperation.
3. There was a general norm drift toward "consistency". Does it predict cooperation, or nothing?

## Where things are
- Repo (read-only for you): /Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/moral-spread-viz
  (branch with all moral/linguistic work). Read docs/figures/linguistic_analysis_20260923/README.md first
  (items 1-4: word uptake, alignment-vs-cooperation, morals, validation) and researchlog.md top ~15 entries.
- Per-myth / per-decision tables: data/analysis/linguistic_20260923/ (a symlink to a SHARED data dir used by
  other worktrees). **Never write, overwrite or delete anything there.** Key files:
  - myths.csv (8,520 myths: run_id, size, composition, mixed, task_order, round, agent, family,
    partner_this_round, exposed_author/family/round = the myth the agent was shown, text)
  - decisions.csv (run_id, round, agent, role investor/trustee, partner, sent 0-5, return_proportion, coop)
  - moral_labels_z-ai__glm-5.2.csv, moral_labels_deepseek__deepseek-v4-flash.csv (label + one-sentence moral summary)
  - myth_rules_september_z-ai__glm-5.2.csv (ALL 8,519 myths, structured rule extraction: send_rule
    none/little/moderate/most/all, return_rule, after_letdown, send_amount numeric, test_first,
    consistency bool, noise_mentioned); plus a DeepSeek 1,000-myth sample and a 100-myth sample for judge checks
  - myth_amount_check_*.csv, embeddings_mpnet.npy (myth text), embeddings_moral_summary_mpnet.npy,
    moral_uptake_children.csv (shown vs unseen comparison myths), alignment_games.csv, style_probabilities.csv
  - Find the scripts that made these (analyses/linguistic_*.py, myth_moral_judge.py, moral_carryover.py,
    alignment_vs_cooperation.py, and grep for "myth_rules") and REUSE their loaders.
- Raw runs: data/json/ (gitignored) in the worktree. Other September/earlier datasets (transplant reruns
  docs/figures/slide678_rerun_20260916, slide678_dyad_rerun_20260917, frontier, mixed) have READMEs.
- Helpers: analyses/_shared.py.

## Data scope
September informed negative-only design: 2-agent and 8-agent, homogeneous (Sonnet 4.5, GPT-5 Nano,
Gemini 3.7 Flash) and mixed; task orders game_myth and myth_game (game-only runs have no myths).
Add the frontier runs (Opus 5 / Gemini 3.1 Pro / GPT-5.6 Sol) only if your lens benefits and you can load them
cleanly; keep them as a separate stratum, never pooled.

## The confounds you must handle (the reason past answers were null or misleading)
- **Reverse causation:** myths describe the game just played (a generous game -> +0.16 probability of a
  "be generous" label per unit send). Any myth-then-cooperation association must control for the agent's own
  previous move(s) and use timing where the myth precedes the decision.
- **Family / composition:** Gemini writes generous and sends at the ceiling; GPT writes fair and sends least.
  Pooled across-run correlations are family artefacts. Stratify by family or use fixed effects.
- **Shared history:** in dyads, both partners lived the same games, so partner-myth effects are confounded.
- **Ceiling:** Gemini sends 5 almost always; near-zero variance cells can't show anything.
Identification ladder (report which rung each result reaches):
  (R0) pooled correlation, descriptive only;
  (R1) within composition x task order;
  (R2) within agent (agent-within-run FE + round FE) controlling for own lagged move;
  (R3) clean exposure: 8-agent myth_game, the SHOWN myth's feature vs a comparable UNSEEN myth, or the shown
       myth's feature predicting the reader's next decision when the shown author never played the reader;
  (R4) founding window: round-1 myths in myth_game are written BEFORE any play, so round-1 myth features ->
       round-1 decisions has no reverse causation (still confounded by model/agent disposition);
  (R5) intervention: transplant/seeding experiments where myth content was manipulated.
Cluster SEs by run. Correct for multiple testing where you run many tests (Holm) and say how many you ran.
Report means as mean (±sd) over runs. Don't claim anything the rung doesn't support.

## Rules
- Free local compute is unlimited. Paid LLM calls are allowed if your lens needs them (e.g. a new judge
  pass): print one preflight line `MODEL=… N=… EST_COST=$…` first, cap your lens at $25, cache responses,
  use OpenRouter via the repo's analyses/_llm_judge.py. Never exceed without asking the lead.
- Write ALL outputs (scripts, CSVs, PNGs, REPORT.md) to your own folder under
  /private/tmp/claude-502/-Users-ivar-Desktop-Research-AI-projects-LLM-evolution-nips-linguistic-evolution-toolkit/ec36f89d-6a64-494f-a76c-34534ff1d678/scratchpad/predictors/<lens>/
  Do not edit, commit or push the repo.
- Use python3 (system python has pandas/scipy/statsmodels? check; else `uv run --with ...`).
- REPORT.md, plain English, headline first: (1) the answer in 2 sentences, (2) a table of every predictor
  tested: rung, effect with CI, p, n, (3) what would change the conclusion, (4) weaknesses. Every number
  must trace to a CSV you wrote. Final message to the lead: the headline + the path to REPORT.md.
