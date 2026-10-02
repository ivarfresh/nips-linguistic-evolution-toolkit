"""Key-results table and forest plot for the send-amount lens (reads tests.csv)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent
t = pd.read_csv(OUT / "tests.csv")
J = "judge_amount"
KEY = [  # (label, rung, test, subset, family, measure)
    ("Own round-1 myth -> round-1 send, Sonnet", "R4", "round-1 own myth amount -> round-1 send (composition FE)", "myth_game round 1, 2+8 agent", "Sonnet", J),
    ("Own round-1 myth -> round-1 send, GPT", "R4", "round-1 own myth amount -> round-1 send (composition FE)", "myth_game round 1, 2+8 agent", "GPT", J),
    ("Own round-1 myth (endorsed only) -> send, all", "R4", "round-1 own myth amount -> round-1 send (composition FE)", "myth_game round 1, 2+8 agent", "all", "judge_amount_strict"),
    ("Shown myth -> next send, all", "R3", "shown amount -> next send (run x round FE; unseen myth as placebo)", "8-agent myth_game", "all", J),
    ("Shown myth -> next send, Sonnet", "R3", "shown amount -> next send (run x round FE; unseen myth as placebo)", "8-agent myth_game", "Sonnet", J),
    ("Shown myth -> next send, GPT", "R3", "shown amount -> next send (run x round FE; unseen myth as placebo)", "8-agent myth_game", "GPT", J),
    ("Mean of 3 in-window shown myths -> next send, all", "R3", "mean of the <=3 in-window shown myths -> next send", "8-agent myth_game", "all", J),
    ("Mean of 3 in-window shown myths -> next send, Sonnet", "R3", "mean of the <=3 in-window shown myths -> next send", "8-agent myth_game", "Sonnet", J),
    ("  placebo: unseen myth -> next send, all", "R3-placebo", "UNSEEN comparison amount -> next send (placebo)", "8-agent myth_game", "all", J),
    ("Shown myth -> next send, no prior contact", "R3", "shown amount -> next send, no prior contact", "8-agent myth_game, no prior contact", "all", J),
    ("Shown round-1 myth (pre-play) -> round-2 send", "R3/R4", "round 2: shown round-1 myth (written before any play) -> round-2 send", "8-agent myth_game round 2", "all", J),
    ("Shown myth -> next send (regex amount)", "R3", "shown amount -> next send (run x round FE; unseen myth as placebo)", "8-agent myth_game", "all", "rx_amount"),
    ("Shown myth -> reader's next MYTH amount", "R3", "shown amount -> reader's next MYTH amount (all myths)", "8-agent myth_game", "all", J),
    ("  placebo: unseen myth -> next MYTH amount", "R3-placebo", "UNSEEN amount -> reader's next MYTH amount (all myths, placebo)", "8-agent myth_game", "all", J),
    ("Own latest myth -> send (agent FE), all", "R2", "own latest myth amount -> send (agent FE), all settings", "all settings", "all", J),
    ("Shown latest myth -> send (agent FE), all", "R2", "latest shown myth amount -> send (agent FE), all settings", "all settings", "all", J),
    ("Reverse: own send -> next own myth, Sonnet", "R2-reverse", "own send -> next own myth amount (agent FE)", "all settings", "Sonnet", J),
    ("Reverse: own send -> next own myth, GPT", "R2-reverse", "own send -> next own myth amount (agent FE)", "all settings", "GPT", J),
    ("Transplant donor amount -> host send, 8-agent", "R5", "donor text amount -> host mean send (donor-type FE, HC robust SE)", "transplant 8-agent", "all", "prescribed"),
    ("Transplant, 8-agent, without 'send nothing' donor", "R5", "donor text amount -> host mean send, without the 'send nothing' donor", "transplant 8-agent", "all", "prescribed"),
    ("Transplant donor amount -> host send, dyad", "R5", "donor text amount -> host mean send (donor-type FE, HC robust SE)", "transplant 2-agent", "all", "prescribed"),
    ("Transplant, dyad, without 'send nothing' donor", "R5", "donor text amount -> host mean send, without the 'send nothing' donor", "transplant 2-agent", "all", "prescribed"),
]
rows = []
for lab, rung, test, subset, fam, meas in KEY:
    r = t[(t.rung == rung) & (t.test == test) & (t.subset == subset) & (t.family == fam) & (t.measure == meas)]
    assert len(r) == 1, lab
    rows.append(dict(label=lab, **r.iloc[0][["rung", "family", "measure", "n", "n_runs", "coef", "ci_low", "ci_high", "p", "p_holm_all", "p_holm_primary"]].to_dict()))
k = pd.DataFrame(rows)
k.to_csv(OUT / "key_results.csv", index=False)
fig, ax = plt.subplots(figsize=(8.5, 7))
y = range(len(k))[::-1]
col = k["rung"].map({"R4": "#1b7837", "R3": "#2166ac", "R3/R4": "#2166ac", "R3-placebo": "#999999", "R2": "#b2182b",
                     "R2-reverse": "#d6604d", "R5": "#762a83"})
for yi, (_, r), c in zip(y, k.iterrows(), col):
    ax.plot([r.ci_low, r.ci_high], [yi, yi], color=c, lw=2)
    ax.plot(r.coef, yi, "o", color=c)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(list(y)); ax.set_yticklabels([f"[{r}] {l}" for r, l in zip(k.rung, k.label)], fontsize=8)
ax.set_xlabel("$ change in outcome per $1 more named in the myth (95% CI, SE clustered by run)")
ax.set_title("Does a myth's named send amount move sends?")
fig.tight_layout(); fig.savefig(OUT / "key_results.png", dpi=160)
print(k.round(3).to_string())
