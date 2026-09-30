#!/usr/bin/env python3
"""Prototype figure: how the moral mix of a family's myths changes among GPT
neighbours in 8-agent populations, paired with the clean transmission test.

Read-only on the repo. Reuses analyses/moral_carryover.py's loader and
label_shares() and checks both against the committed CSVs before plotting.

  python3 moral_contagion_ladder.py   (run from anywhere; writes next to itself)
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/moral-split")
sys.path.insert(0, str(REPO))
from analyses._shared import configure_matplotlib  # noqa: E402
from analyses.moral_carryover import FIGS, DATA, LABELS, label_shares, load  # noqa: E402

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260930)
N_BOOT = 5000

# Paper-wide categorical moral colours (lead decision 2026-09-30); gold is low-contrast, so it gets ink labels.
LABEL_COLORS = {"be generous": "#D9A400", "be fair": "#0b5394", "be cautious": "#B2182B"}
DOSE_COLORS = ["#ec8f60", "#cc5520", "#8a3412"]  # ordinal orange ramp, few -> many GPT
HOMOG_COLOR = "#52514e"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

# Columns ordered by number of GPT neighbours (0 = homogeneous comparator).
LADDERS = {
    "Gemini": [("8 Gemini", 0), ("4 Gemini + 4 GPT", 4), ("2 Gemini + 6 GPT", 6), ("1 Gemini + 7 GPT", 7)],
    "Sonnet": [("8 Sonnet", 0), ("1 GPT + 7 Sonnet", 1), ("2 GPT + 6 Sonnet", 2), ("4 GPT + 4 Sonnet", 4)],
}
ORDER_NAME = {"myth_game": "myth→game", "game_myth": "game→myth"}


# ----------------------------------------------------------------- data

def load_myths() -> pd.DataFrame:
    myths, _ = load("moral_labels_z-ai__glm-5.2.csv")
    ours = label_shares(myths)
    ref = pd.read_csv(FIGS / "moral_label_shares.csv")
    key = ["setting", "family", "task_order", "round"]
    m = ref.merge(ours, on=key, suffixes=("_ref", "_ours"))
    assert len(m) == len(ref) == len(ours)
    for c in [f"share_{lab}" for lab in LABELS] + ["n_myths"]:
        assert np.allclose(m[f"{c}_ref"], m[f"{c}_ours"]), c
    return myths.dropna(subset=["label"])


def boot_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, float)
    draws = RNG.choice(values, size=(N_BOOT, len(values)), replace=True).mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def composition_shares(myths: pd.DataFrame, family: str, composition: str, order: str) -> pd.DataFrame:
    """Pooled label shares per round (the moral_label_shares.csv definition) plus the
    run-level generous share: mean over runs and a 95% bootstrap CI over runs."""
    sub = myths[(myths["size"] == 8) & (myths["family"] == family) &
                (myths["composition"] == composition) & (myths["task_order"] == order)]
    pooled = pd.crosstab(sub["round"], sub["label"], normalize="index").reindex(columns=LABELS, fill_value=0)
    per_run = (sub.assign(gen=sub["label"].eq("be generous"))
               .groupby(["round", "run_id"])["gen"].mean().unstack())
    rows = []
    for rnd, vals in per_run.iterrows():
        lo, hi = boot_ci(vals.dropna().to_numpy())
        rows.append({"round": rnd, "gen_run_mean": vals.mean(), "gen_run_sd": vals.std(ddof=1),
                     "gen_ci_low": lo, "gen_ci_high": hi})
    out = pooled.add_prefix("share_").join(pd.DataFrame(rows).set_index("round"))
    out["n_runs"] = sub["run_id"].nunique()
    out["n_myths_per_round"] = sub.groupby("round").size()
    return out


def uptake_rows(order: str) -> pd.DataFrame:
    """8-agent label uptake (shown minus unseen), per run, reproducing moral_uptake_by_task_order.csv."""
    ch = pd.read_csv(DATA / "moral_uptake_children.csv")
    ch = ch[ch["setting"].str.startswith("8-agent") & (ch["task_order"] == order)]
    ch = ch.assign(exposure=np.where(ch["family"] == ch["parent_family"], "same family", "other family"))
    per_run = ch.groupby(["setting", "exposure", "run_id"])["same_label_excess"].mean().reset_index()
    ref = pd.read_csv(FIGS / "moral_uptake_by_task_order.csv")
    ref = ref[ref["setting"].str.startswith("8-agent") & (ref["task_order"] == order)]
    rows = []
    for (st, ex), g in per_run.groupby(["setting", "exposure"]):
        r = ref[(ref["setting"] == st) & (ref["exposure"] == ex)].iloc[0]
        vals = g["same_label_excess"].dropna().to_numpy()
        assert np.isclose(vals.mean(), r["same_label_excess_mean"]) and len(vals) == r["n_runs"]
        lo, hi = boot_ci(vals)
        rows.append({"setting": st, "exposure": ex, "n_runs": len(vals), "mean": vals.mean(),
                     "sd": vals.std(ddof=1), "ci_low": lo, "ci_high": hi, "p_wilcoxon": r["same_label_excess_p"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- drawing helpers

def style_axes(ax, left=True, bottom=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK2, width=0.6, length=2.5, labelsize=6.5)
    if not left:
        ax.tick_params(labelleft=False)
    if not bottom:
        ax.tick_params(labelbottom=False)


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def draw_uptake(ax, up: pd.DataFrame, order: str) -> None:
    rows = [("8-agent homogeneous", "same family", "own family (all-same population)"),
            ("8-agent mixed", "same family", "own family (mixed population)"),
            ("8-agent mixed", "other family", "another family (mixed population)")]
    ax.axvline(0, color="#c3c2b7", lw=0.8, zorder=0)
    for i, (st, ex, name) in enumerate(rows):
        r = up[(up["setting"] == st) & (up["exposure"] == ex)].iloc[0]
        y = -i
        solid = r["p_wilcoxon"] < 0.05
        color = INK if solid else MUTED
        ax.plot([100 * r["ci_low"], 100 * r["ci_high"]], [y, y], color=color, lw=1.6, solid_capstyle="round")
        ax.plot(100 * r["mean"], y, "o", ms=5.5, mfc=color if solid else "white", mec=color, mew=1.3, zorder=3)
        ptxt = f"p = {r['p_wilcoxon']:.3f}" if r["p_wilcoxon"] >= 0.001 else "p < 0.001"
        ax.text(-10, y + 0.22, name, fontsize=6.3, color=INK, va="bottom")
        ax.text(100 * r["ci_high"] + 1.0, y, f"{100 * r['mean']:+.1f} pts\n{ptxt}",
                va="center", ha="left", fontsize=6, color=INK2)
    ax.set_ylim(-2.45, 0.85)
    ax.set_yticks([])
    ax.set_xlim(-10, 22)
    ax.set_xticks([-10, 0, 10, 20])
    ax.set_xlabel("same moral as shown myth minus\nunseen myth (points, 95% CI)",
                  fontsize=6.3, color=INK2)
    style_axes(ax)
    ax.spines["left"].set_visible(False)


# ----------------------------------------------------------------- prototype A: stacked-area ladder

def figure_stacked(myths: pd.DataFrame, up: pd.DataFrame, order: str) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(7.1, 3.7))
    gs = GridSpec(2, 4, figure=fig, wspace=0.10, hspace=0.45, left=0.075, right=0.655, top=0.80, bottom=0.13)
    gu = GridSpec(1, 1, figure=fig, left=0.705, right=0.985, top=0.80, bottom=0.25)
    rounds = np.arange(1, 11)
    for r, (fam, ladder) in enumerate(LADDERS.items()):
        for c, (comp, n_gpt) in enumerate(ladder):
            ax = fig.add_subplot(gs[r, c])
            s = composition_shares(myths, fam, comp, order).reindex(rounds)
            bottom = np.zeros(len(rounds))
            for lab in LABELS:
                top = bottom + s[f"share_{lab}"].to_numpy()
                ax.fill_between(rounds, bottom, top, color=LABEL_COLORS[lab], lw=0)
                bottom = top
            gen = s["share_be generous"].to_numpy()
            ax.plot(rounds, gen, color="white", lw=0.9)
            ax.plot(rounds, gen + s["share_be fair"].to_numpy(), color="white", lw=0.9)
            ax.set_xlim(1, 10)
            ax.set_ylim(0, 1)
            ax.set_xticks([1, 5, 10])
            ax.set_xticklabels(["1", "5", "10"])
            for t, ha in zip(ax.get_xticklabels(), ["left", "center", "right"]):
                t.set_ha(ha)
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(["0", "50%", "100%"])
            style_axes(ax, left=(c == 0), bottom=(r == 1))
            k = int(s["n_myths_per_round"].iloc[0])
            ax.set_title(comp, fontsize=6.6, color=INK, pad=9)
            ax.text(0.5, 1.015, f"{k} myths per round", transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=5.4, color=MUTED)
            ax.text(0.05, 0.05, f"{pct(gen[0])}→{pct(gen[-1])}", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=6.2, color=INK, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{fam} myths", fontsize=7, color=INK)
            if r == 1:
                ax.set_xlabel("round", fontsize=6.3, color=INK2, labelpad=1)
    ax0 = fig.axes[0]
    ax0.text(5.5, 0.30, "be generous", ha="center", fontsize=6.2, color=INK, fontweight="bold")
    ax0.text(5.5, 0.80, "be fair", ha="center", fontsize=6.2, color="white", fontweight="bold")
    fig.text(0.075, 0.955, f"a  Moral of each myth by round, 8-agent populations ({ORDER_NAME[order]})",
             fontsize=7.6, color=INK, fontweight="bold", ha="left")
    fig.text(0.075, 0.905, "columns: more GPT neighbours →", fontsize=6.3, color=INK2, ha="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=LABEL_COLORS[l]) for l in LABELS]
    fig.legend(handles, LABELS, loc="center left", bbox_to_anchor=(0.30, 0.908), ncol=3, frameon=False,
               fontsize=6.2, handlelength=1.1, columnspacing=1.0)
    axu = fig.add_subplot(gu[0, 0])
    draw_uptake(axu, up, order)
    fig.text(0.705, 0.955, "b  Does a moral hop?", fontsize=7.6, color=INK,
             fontweight="bold", ha="left")
    fig.text(0.705, 0.925, "Clean test (myth→game): the shown myth was\nwritten before the two agents had played.\nAgent shown a myth from its …",
             fontsize=6.0, color=INK2, ha="left", va="top", linespacing=1.2)
    fig.text(0.705, 0.03, "Panel a is not evidence of transmission.\n"
             "In the clean test a moral detectably hops\n"
             "within a family, not across (b).",
             fontsize=5.9, color=INK2, ha="left", va="bottom", linespacing=1.25)
    path = OUT / f"moral_contagion_stacked_{order}.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


# ----------------------------------------------------------------- prototype B: generous-share lines

def figure_lines(myths: pd.DataFrame, up: pd.DataFrame, order: str) -> Path:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6), gridspec_kw={"width_ratios": [1, 1, 1.1], "wspace": 0.45})
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.2)
    rounds = np.arange(1, 11)
    for ax, (fam, ladder) in zip(axes[:2], LADDERS.items()):
        ends = []
        for i, (comp, n_gpt) in enumerate(ladder):
            s = composition_shares(myths, fam, comp, order).reindex(rounds)
            color = HOMOG_COLOR if n_gpt == 0 else DOSE_COLORS[i - 1]
            if n_gpt == 0:
                ax.fill_between(rounds, s["gen_ci_low"], s["gen_ci_high"], color=HOMOG_COLOR, alpha=0.12, lw=0)
            ax.plot(rounds, s["gen_run_mean"], color=color, lw=2 if n_gpt == 0 else 1.5, zorder=3,
                    solid_capstyle="round")
            ends.append((s["gen_run_mean"].iloc[-1], "no GPT" if n_gpt == 0 else f"{n_gpt} GPT"))
        ends.sort()
        ys = [e[0] for e in ends]
        for j in range(1, len(ys)):
            ys[j] = max(ys[j], ys[j - 1] + 0.065)
        for y, (_, lab) in zip(ys, ends):
            ax.text(10.3, y, lab, fontsize=6, color=INK2, va="center")
        ax.set_xlim(1, 10)
        ax.set_ylim(0, 1)
        ax.set_xticks([1, 5, 10])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(["0", "", "50%", "", "100%"])
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        style_axes(ax)
        ax.set_xlabel("round", fontsize=6.5, color=INK2)
        ax.set_title(f"{fam}: share 'be generous'", fontsize=7, color=INK, loc="left")
    axes[0].set_ylabel("share of myths", fontsize=6.5, color=INK2)
    draw_uptake(axes[2], up, order)
    axes[2].set_title("Does a moral hop? (agent shown a\nmyth from its …)", fontsize=7, color=INK, loc="left")
    fig.text(0.07, 0.95, f"8-agent populations, {ORDER_NAME[order]}: labels by GLM-5.2; mean over 5 runs, "
             "band = 95% CI for the all-same population", fontsize=6.5, color=INK2, ha="left")
    path = OUT / f"moral_contagion_lines_{order}.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    configure_matplotlib()
    myths = load_myths()
    tables = []
    for order in ("myth_game", "game_myth"):
        up = uptake_rows(order)
        clean = uptake_rows("myth_game")
        up.assign(task_order=order).to_csv(OUT / f"uptake_8agent_{order}.csv", index=False)
        for fam, ladder in LADDERS.items():
            for comp, n_gpt in ladder:
                tables.append(composition_shares(myths, fam, comp, order).reset_index()
                              .assign(family=fam, composition=comp, n_gpt=n_gpt, task_order=order))
        print(figure_stacked(myths, clean, order))
        print(figure_lines(myths, clean, order))
    pd.concat(tables).to_csv(OUT / "composition_shares.csv", index=False)


if __name__ == "__main__":
    main()
