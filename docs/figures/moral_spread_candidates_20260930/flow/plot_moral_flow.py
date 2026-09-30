"""Moral flow figure: which family's myth an agent was shown -> which family writes next.

(a) 8-agent mixed populations, myth->game: ribbon width = exposures (a child shown a
    myth by that family), ribbon colour = exposure class (same family in the family's
    hue, other family in grey). Class-level excess label match printed on the plot.
(b) Excess label match (shown myth minus unseen same-family myth, run-level mean with
    run-bootstrap 95% CI) for every 8-agent cell. Dyads excluded: partners share games.

Input: children_labels.csv from prep_children.py (reproduces moral_uptake*.csv).
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/moral-split")
sys.path.insert(0, str(REPO))
from analyses._shared import configure_matplotlib  # noqa: E402
from analyses.moral_carryover import run_summary  # noqa: E402

configure_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent
REF = REPO / "docs/figures/linguistic_analysis_20260923/moral_uptake_by_task_order.csv"
FAMILY_COLORS = {"Sonnet": "#7570b3", "GPT": "#d95f02", "Gemini": "#1b9e77"}  # as analyses/linguistic_uptake.py
ORDER = ["Sonnet", "GPT", "Gemini"]  # GPT in the middle: mixed runs pair GPT with Sonnet or Gemini
INK, INK2, MUTED, GRID, CROSS = "#1f1f1f", "#52514e", "#8a8984", "#e4e3df", "#b9b8b2"
rng = np.random.default_rng(20260930)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5,
                     "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7.5,
                     "axes.edgecolor": MUTED, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK2, "savefig.facecolor": "white"})


def bootstrap_ci(vals: np.ndarray, n: int = 10000) -> tuple[float, float]:
    boots = rng.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
    return tuple(np.percentile(boots, [2.5, 97.5]))


def cells(d: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Run-level excess per 8-agent cell, checked against the committed CSV."""
    d8 = d[d["setting"].str.startswith("8-agent")]
    summ = run_summary(d8, ["setting", "exposure", "task_order"], ["same_label_excess"])
    per_run = d8.groupby(["setting", "exposure", "task_order", "run_id"])["same_label_excess"].mean()
    raw = run_summary(d8, ["setting", "exposure", "task_order"], ["same_label_shown", "same_label_unseen"])
    summ = summ.merge(raw.drop(columns="n_runs"), on=["setting", "exposure", "task_order"])
    rows = []
    for _, r in summ.iterrows():
        k = (r.setting, r.exposure, r.task_order)
        lo, hi = bootstrap_ci(per_run.loc[k].to_numpy())
        rf = ref[(ref.setting == r.setting) & (ref.exposure == r.exposure) & (ref.task_order == r.task_order)].iloc[0]
        assert np.isclose(rf.same_label_excess_mean, r.same_label_excess_mean)
        # p is printed from the CSV: after a CSV round-trip, float noise decides whether tied run
        # means count as ties, which flips scipy's Wilcoxon between exact and normal approximation
        assert rf.n_runs == r.n_runs, (k, rf.n_runs, r.n_runs)
        rows.append({"setting": r.setting, "exposure": r.exposure, "task_order": r.task_order, "n_runs": r.n_runs,
                     "mean": 100 * r.same_label_excess_mean, "lo": 100 * lo, "hi": 100 * hi,
                     "p": rf.same_label_excess_p, "shown": 100 * r.same_label_shown_mean,
                     "unseen": 100 * r.same_label_unseen_mean})
        assert np.isclose(r.same_label_shown_mean - r.same_label_unseen_mean, r.same_label_excess_mean)
    return pd.DataFrame(rows)


def ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha):
    """Band from [y0a,y0b] at x0 to [y1a,y1b] at x1, S-curved."""
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a), (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="white", lw=0.6, alpha=alpha, zorder=2))


def fmt_p(p: float) -> str:
    return f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def panel_flow(ax, d: pd.DataFrame, cl: pd.DataFrame) -> None:
    x = d[(d["setting"] == "8-agent mixed") & (d["task_order"] == "myth_game")]
    flows = x.groupby(["parent_family", "family"]).size()
    total = flows.sum()
    gap, xl, xr, w = 0.06, 0.0, 1.0, 0.045
    scale = (1 - gap * (len(ORDER) - 1)) / total

    def stack(side_totals):
        pos, y = {}, 1.0
        for f in ORDER:
            h = side_totals.get(f, 0) * scale
            pos[f] = (y - h, y)
            y -= h + gap
        return pos

    left = stack(flows.groupby(level=0).sum())
    right = stack(flows.groupby(level=1).sum())
    lcur = {f: left[f][1] for f in ORDER}
    rcur = {f: right[f][1] for f in ORDER}
    for src in ORDER:  # outflows ordered by target, inflows ordered by source: no self-crossing
        for dst in ORDER:
            n = flows.get((src, dst), 0)
            if not n:
                continue
            h = n * scale
            same = src == dst
            ribbon(ax, xl + w, xr - w, lcur[src], lcur[src] - h, rcur[dst], rcur[dst] - h,
                   FAMILY_COLORS[src] if same else CROSS, 0.62 if same else 0.55)
            lcur[src] -= h
            rcur[dst] -= h
    for pos, x0, ha, dx in [(left, xl, "right", -0.02), (right, xr - w, "left", w + 0.02)]:
        for f in ORDER:
            y0, y1 = pos[f]
            ax.add_patch(Rectangle((x0, y0), w, y1 - y0, facecolor=FAMILY_COLORS[f], edgecolor="none", zorder=3))
            n = flows.xs(f, level=0 if ha == "right" else 1).sum()
            ax.text(x0 + dx if ha == "right" else x0 + dx, (y0 + y1) / 2, f"{f}\n{n:,}", ha=ha, va="center",
                    fontsize=7.5, color=INK, linespacing=1.15)
    ax.text(xl + w / 2, 1.045, "myth shown\n(author's family)", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.text(xr - w / 2, 1.045, "next myth written\n(reader's family)", ha="center", va="bottom", fontsize=7, color=INK2)

    same = cl[(cl.setting == "8-agent mixed") & (cl.exposure == "same family") & (cl.task_order == "myth_game")].iloc[0]
    other = cl[(cl.setting == "8-agent mixed") & (cl.exposure == "other family") & (cl.task_order == "myth_game")].iloc[0]
    box = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.9)
    # same-family callout on the Sonnet band, other-family callout on the Sonnet/GPT crossing
    ys = left["Sonnet"][1] - 0.42 * (left["Sonnet"][1] - left["Sonnet"][0])
    ax.text(0.5, ys, f"same family: {same['mean']:+.1f} pts ({fmt_p(same['p'])})\n"
            f"match {same['shown']:.0f}% shown vs {same['unseen']:.0f}% unseen",
            ha="center", va="center", fontsize=6.6, color=INK, bbox=box, zorder=5)
    yc = (left["Sonnet"][0] + right["GPT"][1]) / 2 - 0.035
    ax.text(0.5, yc, f"other family: {other['mean']:+.1f} pts ({fmt_p(other['p'])})\n"
            f"match {other['shown']:.0f}% shown vs {other['unseen']:.0f}% unseen",
            ha="center", va="center", fontsize=6.6, color=INK2, bbox=box, zorder=5)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.02, 1.2)
    ax.axis("off")


def panel_excess(ax, cl: pd.DataFrame) -> None:
    rows = [("8-agent homogeneous", "same family", "homogeneous\nsame family"),
            ("8-agent mixed", "same family", "mixed\nsame family"),
            ("8-agent mixed", "other family", "mixed\nother family")]
    offs = {"myth_game": -0.14, "game_myth": 0.14}
    ax.axvline(0, color=MUTED, lw=0.9, zorder=1)
    for i, (st, ex, name) in enumerate(rows):
        col = INK if ex == "same family" else MUTED
        for to, dy in offs.items():
            r = cl[(cl.setting == st) & (cl.exposure == ex) & (cl.task_order == to)].iloc[0]
            y = i + dy
            clean = to == "myth_game"
            ax.plot([r.lo, r.hi], [y, y], color=col, lw=1.6 if clean else 1.0, alpha=1 if clean else 0.6,
                    solid_capstyle="round", zorder=2)
            ax.plot(r["mean"], y, "o", ms=5.5 if clean else 4.5, mfc=col if clean else "white", mec=col,
                    mew=1.2, alpha=1 if clean else 0.8, zorder=3)
            if clean:
                ax.text(10.4, y, f"{r['mean']:+.1f}  {fmt_p(r['p'])}", va="center", ha="left",
                        fontsize=6.6, color=INK if ex == "same family" else INK2)
    ax.set_yticks(range(len(rows)), [r[2] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(-6.5, 17.5)
    ax.set_xticks([-5, 0, 5, 10])
    ax.set_xlabel("excess label match (pts): shown myth − unseen myth")
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="-", color=INK, mfc=INK, ms=5, lw=1.4, label="myth→game (clean test)"),
         Line2D([], [], marker="o", ls="-", color=INK, mfc="white", ms=4.5, lw=1.0, alpha=0.6,
                label="game→myth (shared games, not clean)")]
    ax.figure.legend(handles=h, loc="lower right", bbox_to_anchor=(0.99, 0.0), ncol=2, frameon=False, fontsize=6.4,
              handlelength=1.6, borderaxespad=0.2)


def main() -> None:
    d = pd.read_csv(OUT / "children_labels.csv")
    ref = pd.read_csv(REF)
    cl = cells(d, ref)
    cl.to_csv(OUT / "moral_flow_cells.csv", index=False)
    print(cl.round(3).to_string(index=False))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
    panel_flow(a, d, cl)
    panel_excess(b, cl)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.2, wspace=0.28)
    kw = dict(fontsize=8.5, color=INK, va="top")
    fig.text(0.01, 0.985, "a   Who reads whose myth\n     8-agent mixed populations, myth→game", **kw)
    fig.text(0.47, 0.985, "b   Excess over the base rate\n     all 8-agent cells; dyads excluded", **kw)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"moral_flow.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
