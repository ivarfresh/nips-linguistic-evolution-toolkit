import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
e = pd.read_csv("rung_estimates.csv")
e = e[(e.families == "Sonnet+GPT") & (e.role == "investor")]
rungs = ["R0 pooled", "R1 cell+family+round FE", "R4 myth->game round 1", "R2 agent+round FE, own lag",
         "R3 8-agent myth->game shown myth (clean)"]
names = ["R0 pooled", "R1 within cell", "R4 round 1, before play", "R2 within agent", "R3 shown myth (clean)"]
ms = {"label_ord_glm": ("3-way label (GLM)", "#0b5394"), "label_ord_ds": ("3-way label (DeepSeek)", "#6fa8dc"),
      "rule_prescribed": ("rule: prescribed send", "#b45f06"), "give_send_glm": ("0-10 send score (GLM)", "#D9A400"),
      "give_send_ds": ("0-10 send score (DeepSeek)", "#e6c34d"), "emb_axis_summary": ("embedding axis (moral)", "#999999")}
fig, ax = plt.subplots(figsize=(9, 5))
for k, (m, (lab, c)) in enumerate(ms.items()):
    for i, r in enumerate(rungs):
        row = e[(e.rung == r) & (e.measure == m)].iloc[0]
        y = i + (k - 2.5) * 0.12
        ax.errorbar(row.coef, y, xerr=[[row.coef - row.ci_low], [row.ci_high - row.coef]], fmt="o", color=c,
                    capsize=2, ms=4, label=lab if i == 0 else None)
ax.axvline(0, color="#555", lw=1)
ax.set_yticks(range(len(rungs)), names); ax.invert_yaxis()
ax.set_xlabel("change in amount sent / 5 per 1 SD of the measure (95% CI, SE clustered by run)")
ax.set_title("Senders (Sonnet + GPT): the send-amount measures see what the label misses,\nbut nothing survives within agent or from the shown myth")
ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", alpha=.3)
fig.tight_layout(); fig.savefig("label_vs_finer_measures_by_rung.png", dpi=180)
