# Team update packet - 2026-05-03

This folder contains the copy-paste team update and four figures for the Sunday-night message.

## What to attach

Recommended attachments:

1. `figures/fig2_bootstrap_mechanism_game_myth.png`
   - Best single figure for the main message.
   - Shows GPT-5-Nano bootstrap trajectories: game-only, game->myth without prompt injection, and the partner-story-labelled controls.
   - Key point: myth-writing alone drops cooperation, while text labelled as the partner's recent story returns cooperation to the game-only trajectory.

2. `figures/fig3_boundary_conditions_game_myth.png`
   - Best second figure if there is room.
   - Shows the same GPT-5-Nano prompt intervention across no-noise, positive, negative, and bootstrap.
   - Key point: the effect is bootstrap-specific, not a generic "more text helps" result.

3. `figures/fig4_neutral_framing_trajectories.png`
   - Best third figure if keeping the neutral-framing robustness paragraph.
   - Shows no-noise standard investor/trustee framing versus neutral ROLE A/B framing for GPT-5-Nano and Claude Sonnet 4.5.
   - Key point: neutral framing lowers GPT-5-Nano substantially and exposes a larger myth-present lift, while Claude moves less and remains mostly myth-null.
   - All cells have 15 seeds except Claude neutral myth->game, which has 14 because one Anthropic run failed.

Optional context attachment:

4. `figures/fig1_main_v4_trajectories.png`
   - Shows the original v4 task-order trajectories by model and noise regime.
   - Useful if the team wants the broader context, but less central than the mechanism figures.

## Files

- `TEAM_UPDATE_TO_SEND.md`: copy-paste text for the email/Slack update.
- `generate_team_update_figures.py`: reproducible plotting script.
- `figures/*.png`: image attachments.
- `figures/*.pdf`: vector versions for manuscript reuse if useful.

Regenerate figures from repo root:

```bash
python3 projects/neurips-2026-llm-ling-evo/analysis/team_update_2026-05-03/generate_team_update_figures.py
```
