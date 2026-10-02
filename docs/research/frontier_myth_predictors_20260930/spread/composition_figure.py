#!/usr/bin/env python3
"""Frontier twin of moral_composition_by_round_{order}.png (September colour scheme).

Panel a: GLM-5.2 moral mix by round for Opus, GeminiPro and Sol myths, in their
homogeneous 8-agent populations and inside the 2 GeminiPro + 3 Opus + 3 Sol population.
Panel b: the clean 8-agent myth->game label-uptake test (shown minus unseen), mean over
runs with a 95% bootstrap CI, from frontier/moral_uptake_by_stratum.csv (spread.py).
Run spread.py --dataset frontier first. Writes frontier/moral_composition_by_round_{order}.png
and frontier/moral_composition_by_round.csv; the September PNGs are not touched.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ["LINGUISTIC_DATASET"] = "frontier"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyses._shared import configure_matplotlib  # noqa: E402
from analyses.moral_carryover import LABELS, LABEL_COLORS, load  # noqa: E402
from analyses.moral_composition_ladder import INK, INK2, MUTED, composition_shares, pct, style_axes  # noqa: E402

OUT = HERE / "frontier"
MIX = "2 GeminiPro + 3 Opus + 3 Sol"
FAMS = ["Opus", "GeminiPro", "Sol"]
ORDER_NAME = {"myth_game": "myth→game", "game_myth": "game→myth"}


def uptake_rows() -> list[tuple[str, pd.Series]]:
    t = pd.read_csv(OUT / "moral_uptake_by_stratum.csv")
    t = t[(t["task_order"] == "myth_game") & t["setting"].str.startswith("8-agent")]
    pick = [("own family, all-same populations", "all 8-agent homogeneous runs", "all", "same family")]
    pick += [(f"  8 {f} alone", f"8 {f}", f, "same family") for f in FAMS]
    pick += [("own family, mixed population", MIX, "all", "same family"),
             ("other family, mixed population", MIX, "all", "other family")]
    rows = []
    for name, pop, fam, ex in pick:
        r = t[(t["population"] == pop) & (t["family"] == fam) & (t["exposure"] == ex)]
        if len(r):
            rows.append((name, r.iloc[0]))
    return rows


def draw_uptake(ax, rows) -> None:
    ax.axvline(0, color="#c3c2b7", lw=0.8, zorder=0)
    for i, (name, r) in enumerate(rows):
        y = -i
        p = r["uptake_p"]
        solid = pd.notna(p) and p < 0.05
        color = INK if solid else MUTED
        ax.plot([100 * r["uptake_ci_low"], 100 * r["uptake_ci_high"]], [y, y], color=color, lw=1.6,
                solid_capstyle="round")
        ax.plot(100 * r["uptake_effect"], y, "o", ms=5, mfc=color if solid else "white", mec=color, mew=1.2, zorder=3)
        ptxt = "p n/a" if pd.isna(p) else (f"p = {p:.3f}" if p >= 0.001 else "p < 0.001")
        ax.text(-30, y + 0.2, f"{name} ({int(r['uptake_n_runs'])} runs)", fontsize=5.8, color=INK, va="bottom")
        ax.text(max(100 * r["uptake_ci_high"], 100 * r["uptake_effect"]) + 1.0, y,
                f"{100 * r['uptake_effect']:+.1f} (±{100 * r['uptake_sd']:.1f})\n{ptxt}",
                va="center", ha="left", fontsize=5.5, color=INK2)
    ax.set_ylim(-len(rows) + 0.4, 0.85)
    ax.set_yticks([])
    ax.set_xlim(-30, 40)
    ax.set_xticks([-20, 0, 20])
    ax.set_xlabel("same moral as shown myth minus\nunseen myth (points, 95% CI over runs)", fontsize=6.3, color=INK2)
    style_axes(ax)
    ax.spines["left"].set_visible(False)


def figure(myths: pd.DataFrame, order: str, tables: list) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(7.1, 4.6))
    gs = GridSpec(3, 2, figure=fig, wspace=0.10, hspace=0.62, left=0.085, right=0.46, top=0.84, bottom=0.10)
    gu = GridSpec(1, 1, figure=fig, left=0.56, right=0.985, top=0.80, bottom=0.30)
    rounds = np.arange(1, 11)
    for r, fam in enumerate(FAMS):
        for c, comp in enumerate([f"8 {fam}", MIX]):
            ax = fig.add_subplot(gs[r, c])
            s = composition_shares(myths, fam, comp, order).reindex(rounds)
            tables.append(s.reset_index().assign(family=fam, composition=comp, task_order=order))
            bottom = np.zeros(len(rounds))
            for lab in LABELS:
                top = bottom + s[f"share_{lab}"].fillna(0).to_numpy()
                ax.fill_between(rounds, bottom, top, color=LABEL_COLORS[lab], lw=0)
                bottom = top
            gen = s["share_be generous"].to_numpy()
            ax.plot(rounds, gen, color="white", lw=0.9)
            ax.plot(rounds, gen + s["share_be fair"].to_numpy(), color="white", lw=0.9)
            ax.set_xlim(1, 10)
            ax.set_ylim(0, 1)
            ax.set_xticks([1, 5, 10])
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(["0", "50%", "100%"])
            style_axes(ax, left=(c == 0), bottom=(r == len(FAMS) - 1))
            k = int(s["n_myths_per_round"].iloc[0])
            if r == 0:
                ax.set_title("homogeneous (8 X)" if c == 0 else "mixed (2 GP + 3 Opus + 3 Sol)",
                             fontsize=6.2, color=INK, pad=12)
            ax.text(0.5, 1.015, f"{k} myths per round, {int(s['n_runs'].iloc[0])} runs", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=5.3, color=MUTED)
            ax.text(0.05, 0.05, f"{pct(gen[0])}→{pct(gen[-1])}", transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=6.2, color=INK, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{fam} myths", fontsize=7, color=INK)
            if r == len(FAMS) - 1:
                ax.set_xlabel("round", fontsize=6.3, color=INK2, labelpad=1)
    fig.text(0.085, 0.965, f"a  Moral mix by round, frontier 8-agent ({ORDER_NAME[order]})",
             fontsize=7.6, color=INK, fontweight="bold", ha="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=LABEL_COLORS[l]) for l in LABELS]
    fig.legend(handles, LABELS, loc="center left", bbox_to_anchor=(0.08, 0.925), ncol=3, frameon=False,
               fontsize=6.2, handlelength=1.1, columnspacing=1.0)
    axu = fig.add_subplot(gu[0, 0])
    draw_uptake(axu, uptake_rows())
    fig.text(0.56, 0.965, "b  Does a moral hop? (myth→game)", fontsize=7.6, color=INK, fontweight="bold", ha="left")
    fig.text(0.56, 0.935, "The shown myth was written before its author and the\nreader had played. "
             "Filled: Wilcoxon p < 0.05 over runs.\n5 runs give a minimum p of 0.0625.",
             fontsize=5.9, color=INK2, ha="left", va="top", linespacing=1.2)
    fig.text(0.56, 0.02, "GeminiPro has 2 members in the mixed population, so it has no\n"
             "unseen same-family myth: own-family rows there are Opus and Sol.\n"
             "Placebo (author's next, never-shown myth): +0.6 in all-same populations,\n"
             "but +14.6 / +13.4 in the mixed rows, as large as the effect: shared game.\n"
             "Judge: GLM-5.2, Arabella Sinclair's 3-label rubric.",
             fontsize=5.8, color=INK2, ha="left", va="bottom", linespacing=1.25)
    path = OUT / f"moral_composition_by_round_{order}.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    configure_matplotlib()
    myths, _ = load("moral_labels_z-ai__glm-5.2.csv")
    myths = myths.dropna(subset=["label"])
    tables: list = []
    for order in ("myth_game", "game_myth"):
        print(figure(myths, order, tables))
    pd.concat(tables).to_csv(OUT / "moral_composition_by_round.csv", index=False)


if __name__ == "__main__":
    main()
