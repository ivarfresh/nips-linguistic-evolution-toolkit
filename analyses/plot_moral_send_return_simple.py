#!/usr/bin/env python3
"""Simple main-text figure: send and return per round by the moral of the myth a player read.

Asked for by Ed (2026-10-07) for the claim "language passes between agents, but
behaviour barely does": average sent / 5 and return proportion over rounds for
the three moral labels (be generous / be fair / be cautious), instead of the
per-condition appendix grids from analyses/moral_carryover.py.

The label is the judge's moral (GLM-5.2) of the latest myth the player was
shown before the decision. Two rows:
  top:    raw means. Players shown generous myths do send more, even within
          one model, but that mixes the run and partner a player has (e.g.
          generous myths come from Gemini partners far more often than
          cautious ones) with what they read.
  bottom: the same player compared with itself: each move minus that player's
          own average move in that role over the run. If reading a generous myth
          changed play, the green line would sit above the others here.

Gemini 3.7 Flash (September) and Gemini 3.1 Pro (frontier) are left out: they
send at the ceiling whatever they read, as in the appendix grids. All settings (2 and 8 agents, homogeneous and
mixed) and both task orders are pooled. Lines are means over runs (each run
averaged first). Round 1 has no shown myth (players first see another
player's myth after writing their own), so the x axis starts at round 2.
Bands are 95% t-intervals across runs; points built on fewer than MIN_RUNS
runs are hidden. Round 2 holds only myth→game runs (game→myth players first
act on a shown myth in round 3). Descriptive; the matching test is the
agent-fixed-effects carryover model in moral_carryover_models.csv.

The script also refits that test (own label, shown label and own last move;
the shown-label coefficients are saved) into
carryover_test_shown_label.csv, with all families included.

No API calls. Run from the repo root:
  LINGUISTIC_DATASET=september_n10 python3 analyses/plot_moral_send_return_simple.py
  LINGUISTIC_DATASET=frontier python3 analyses/plot_moral_send_return_simple.py   # writes its own provenance
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses._shared import configure_matplotlib  # noqa: E402
from analyses import moral_carryover as mc  # noqa: E402

# Per corpus: output folder, the family left out (sends at the ceiling whatever it reads)
# and the models named in the title.
SETUP = {
    "september_n10": (ROOT / "docs/figures/moral_send_return_simple_20261007", "Gemini",
                      "Sonnet 4.5 and GPT-5 Nano", {"coop": 0.25, "coop_vs_own_avg": -0.35}),
    "frontier": (ROOT / "docs/figures/moral_send_return_simple_frontier_20261007", "GeminiPro",
                 "Claude Opus 5 and GPT-5.6 Sol", {"coop": 0.3, "coop_vs_own_avg": -0.08}),
}
MIN_LABEL_DECISIONS = 50  # a label read fewer times than this in a role gets no line (frontier 'be cautious')
MIN_RUNS = 5
ROLES = (("investor", "amount sent / 5"), ("trustee", "share of the pot returned"))


def per_round(d: pd.DataFrame, value: str) -> pd.DataFrame:
    """Mean and 95% CI across runs, per role x shown label x round."""
    run = d.groupby(["role", "shown_label", "round", "run_id"])[value].mean().reset_index()
    rows = []
    for (role, lab, rnd), g in run.groupby(["role", "shown_label", "round"]):
        n = len(g)
        m = g[value].mean()
        half = stats.t.ppf(0.975, n - 1) * g[value].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append({"role": role, "shown_label": lab, "round": rnd, "measure": value, "mean": m,
                     "ci_low": m - half, "ci_high": m + half, "n_runs": n,
                     "n_decisions": int(((d["role"] == role) & (d["shown_label"] == lab) & (d["round"] == rnd)).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    import matplotlib.pyplot as plt
    if mc._DS.name not in SETUP:
        raise SystemExit("run with LINGUISTIC_DATASET=september_n10 or LINGUISTIC_DATASET=frontier")
    out, ceiling_family, models, ymin = SETUP[mc._DS.name]  # ymin keeps wide early bands inside the axes
    out.mkdir(parents=True, exist_ok=True)
    myths, dec = mc.load("moral_labels_z-ai__glm-5.2.csv")
    full = mc.decision_table(myths, dec)

    # The matching test: agent-within-run + round fixed effects, own last move, all families
    # (the same model moral_carryover.py writes to moral_carryover_models.csv).
    test = mc.carryover_models(full)
    test = test[(test["fe"] == "agent") & (test["setting"] == "all settings")
                & (test["model"] == "own + shown label, own lag") & (test["predictor"] == "shown_label")]
    test.drop(columns=["error"], errors="ignore").to_csv(out / "carryover_test_shown_label.csv", index=False)
    print(test[["role", "level", "coef", "ci_low", "ci_high", "p", "n_decisions", "n_runs"]].round(3).to_string(index=False))

    d = full[full["family"] != ceiling_family].dropna(subset=["shown_label", "coop"]).copy()
    d["coop_vs_own_avg"] = d["coop"] - d.groupby(["run_id", "agent", "role"])["coop"].transform("mean")

    table = pd.concat([per_round(d, "coop"), per_round(d, "coop_vs_own_avg")], ignore_index=True)
    table.to_csv(out / "send_return_by_shown_moral.csv", index=False)

    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharex=True, layout="constrained")
    rows = (("coop", "What players did"),
            ("coop_vs_own_avg", "Same player compared with itself\n(move minus its own average)"))
    dropped = set()
    for i, (measure, ylabel) in enumerate(rows):
        for j, (role, title) in enumerate(ROLES):
            ax = axes[i, j]
            for lab in mc.LABELS:
                s = table[(table["measure"] == measure) & (table["role"] == role) & (table["shown_label"] == lab)]
                if s["n_decisions"].sum() < MIN_LABEL_DECISIONS:
                    dropped.add(lab)
                    continue
                s = s.set_index("round").reindex(range(2, 11))
                s.loc[s["n_runs"] < MIN_RUNS, ["mean", "ci_low", "ci_high"]] = np.nan
                ax.fill_between(s.index, s["ci_low"].clip(lower=ymin[measure]), s["ci_high"], color=mc.LABEL_COLORS[lab], alpha=0.15, lw=0)
                ax.plot(s.index, s["mean"], color=mc.LABEL_COLORS[lab], lw=2, marker="o", ms=3,
                        label=f"read a '{lab}' myth")
            if measure == "coop_vs_own_avg":
                ax.axhline(0, color="#666666", lw=0.8)
            if i == 0:
                ax.set_title(title, fontsize=11)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_xticks(range(2, 11))
        axes[i, 1].sharey(axes[i, 0])
        axes[i, 0].set_ylim(bottom=ymin[measure])
    for ax in axes[-1]:
        ax.set_xlabel("round")
    axes[0, 0].legend(fontsize=8, loc="lower right", frameon=False)
    fig.suptitle("Does the moral of the myth a player just read change how it plays?\n"
                 f"{models}, all settings and task orders pooled; "
                 f"bands: 95% CI across runs (n = {d['run_id'].nunique()} runs)"
                 + "".join(f"\n'{lab}' not shown: fewer than {MIN_LABEL_DECISIONS} decisions follow such a myth"
                           for lab in sorted(dropped)), fontsize=11)
    fig.savefig(out / "send_return_by_shown_moral.png", dpi=200)
    plt.close(fig)

    overall = d.groupby(["role", "shown_label"])[["coop", "coop_vs_own_avg"]].mean().round(3)
    print(overall.to_string())
    if mc._DS.name == "frontier":
        from analyses.myth_map_significance import frontier_provenance
        frontier_provenance(out, myths["path"].unique())


if __name__ == "__main__":
    main()
