#!/usr/bin/env python3
"""Time-expanded lineage view of how myth morals spread in 8-agent myth->game populations.

Panels a-b: one population run each. Rows = agents, columns = rounds 1-10. Each node is
one myth, filled by its GLM-5.2 moral label. An edge joins the myth an agent was shown
(round r-1, from the run's `myth_exposures`) to the myth the agent wrote next (round r).
Coloured edge = the child kept the shown myth's label; grey = it did not. Under each
panel: observed matches vs the matches expected if the child had been shown an unseen
myth by an agent of the parent's family (same run and round; the null of
analyses/linguistic_uptake.null_candidates).

Panel c: the aggregate test over all 8-agent myth->game runs (Wilcoxon over runs, from
docs/figures/linguistic_analysis_20260923/moral_uptake_by_task_order.csv).

Run selection (fixed before rendering): compositions picked for balance and label
variety (8 Sonnet: 48/52 fair/generous; 4 Gemini + 4 GPT: two families with distinct
styles), then the replicate with the MEDIAN per-run excess among its 5 replicates
(homogeneous: all edges; mixed: same-family edges only).

Read-only on the repo; run with the moral-split worktree as cwd:
  uv run --with pandas --with numpy --with scipy --with matplotlib --with python-dotenv \
      python <this script>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/moral-split")
sys.path.insert(0, str(WT))
from analyses._shared import configure_matplotlib  # noqa: E402
from analyses.linguistic_uptake import null_candidates  # noqa: E402
from analyses.moral_carryover import DATA, FIGS, load  # noqa: E402

configure_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

OUT = Path(__file__).resolve().parent
# Team scheme (lead, 2026-09-30); fair darkened from #2a78d6, which failed against Sonnet purple #7570b3.
# Validated all-pairs together with the repo family colours (#7570b3, #d95f02, #1b9e77).
LABEL_COLORS = {"be generous": "#D9A400", "be fair": "#0b5394", "be cautious": "#B2182B"}
FAMILY_MARKER = {"Sonnet": "o", "Gemini": "D", "GPT": "s"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#b9b8b3", "#e6e5e1"
PANELS = [("8 Sonnet", "homogeneous"), ("4 Gemini + 4 GPT", "mixed")]


# --------------------------------------------------------------------------- data

def uptake_children(m: pd.DataFrame) -> pd.DataFrame:
    """Per-child label match vs unseen, with agent id (same rules as moral_carryover.summary_measures)."""
    has = m["summary"].notna().to_numpy()
    rows = []
    for i, js in null_candidates(m).items():
        p, nulls = js[0], [j for j in js[1:] if has[j]]
        if not (has[i] and has[p]) or not nulls:
            continue
        li = m.at[i, "label"]
        rows.append({"run_id": m.at[i, "run_id"], "size": m.at[i, "size"], "mixed": m.at[i, "mixed"],
                     "composition": m.at[i, "composition"], "task_order": m.at[i, "task_order"],
                     "round": m.at[i, "round"], "agent": m.at[i, "agent"], "family": m.at[i, "family"],
                     "parent_family": m.at[p, "family"], "shown": float(li == m.at[p, "label"]),
                     "unseen": float(np.mean([li == m.at[j, "label"] for j in nulls]))})
    u = pd.DataFrame(rows)
    u["excess"] = u["shown"] - u["unseen"]
    ref = pd.read_csv(DATA / "moral_uptake_children.csv")["same_label_excess"].dropna()
    assert len(ref) == len(u) and np.allclose(np.sort(ref), np.sort(u["excess"])), "does not reproduce repo table"
    return u


def pick_run(u: pd.DataFrame, composition: str, kind: str) -> str:
    c = u[(u["composition"] == composition) & (u["task_order"] == "myth_game")]
    if kind == "mixed":
        c = c[c["family"] == c["parent_family"]]
    per_run = c.groupby("run_id")["excess"].mean().sort_values()
    assert len(per_run) == 5
    return per_run.index[2]  # median of 5


def aggregate() -> pd.DataFrame:
    """Shown / unseen match rates (mean over runs) and the published test, 8-agent myth->game."""
    ch = pd.read_csv(DATA / "moral_uptake_children.csv")
    ch = ch[(ch["setting"].str.startswith("8-agent")) & (ch["task_order"] == "myth_game")].copy()
    ch["exposure"] = np.where(ch["setting"] == "8-agent homogeneous", "same family",
                              np.where(ch["family"] == ch["parent_family"], "same family", "other family"))
    per_run = ch.groupby(["setting", "exposure", "run_id"])[["same_label_shown", "same_label_unseen",
                                                            "same_label_excess"]].mean().reset_index()
    agg = per_run.groupby(["setting", "exposure"]).agg(
        shown=("same_label_shown", "mean"), unseen=("same_label_unseen", "mean"),
        excess=("same_label_excess", "mean"), excess_sd=("same_label_excess", "std"),
        n_runs=("run_id", "nunique")).reset_index()
    pub = pd.read_csv(FIGS / "moral_uptake_by_task_order.csv")
    pub = pub[pub["setting"].str.startswith("8-agent") & (pub["task_order"] == "myth_game")]
    agg = agg.merge(pub[["setting", "exposure", "same_label_excess_mean", "same_label_excess_sd",
                         "same_label_excess_p", "same_label_excess_runs_positive", "n_runs"]],
                    on=["setting", "exposure"], suffixes=("", "_pub"))
    assert np.allclose(agg["excess"], agg["same_label_excess_mean"])
    assert np.allclose(agg["excess_sd"], agg["same_label_excess_sd"])
    assert (agg["n_runs"] == agg["n_runs_pub"]).all()
    return agg


# --------------------------------------------------------------------------- drawing

def row_order(run: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    fam = run.drop_duplicates("agent").set_index("agent")["family"].to_dict()
    order = sorted(fam, key=lambda a: (["Sonnet", "Gemini", "GPT"].index(fam[a]), int(a.split("_")[1])))
    return order, fam


def draw_network(ax, m: pd.DataFrame, u: pd.DataFrame, run_id: str, kind: str, title: str) -> str:
    run = m[m["run_id"] == run_id]
    order, fam = row_order(run)
    y = {a: len(order) - 1 - k for k, a in enumerate(order)}
    lab = {(r.round, r.agent): r.label for r in run.itertuples()}
    mixed = kind == "mixed"

    # edges: non-matches first (under), then matches
    edges = []
    for r in run[run["round"] > 1].itertuples():
        pa, pr = r.exposed_author, int(r.exposed_round)
        match = lab[(pr, pa)] == r.label
        same = fam[pa] == r.family
        edges.append((pr, y[pa], r.round, y[r.agent], match, same, r.label))
    for pr, y0, cr, y1, match, same, label in sorted(edges, key=lambda e: e[4]):
        rad = 0.0 if y0 == y1 else 0.18 * np.sign(y1 - y0) * min(1, abs(y1 - y0) / 3)
        cross = mixed and not same
        style = (0, (2.0, 1.8)) if cross else "-"
        if match:  # cross-family matches drawn lighter: they run at chance (panel c)
            kw = dict(color=LABEL_COLORS[label], lw=0.9 if cross else 1.5, alpha=0.55 if cross else 0.9, zorder=2)
        else:
            kw = dict(color=MUTED, lw=0.6 if cross else 0.7, alpha=0.9, zorder=1)
        ax.add_patch(FancyArrowPatch((pr, y0), (cr, y1), connectionstyle=f"arc3,rad={rad}",
                                     arrowstyle="-", linestyle=style, shrinkA=4, shrinkB=4, **kw))

    # nodes
    for r in run.itertuples():
        ax.scatter(r.round, y[r.agent], s=46, marker=FAMILY_MARKER[r.family], c=LABEL_COLORS[r.label],
                   edgecolors="white", linewidths=1.0, zorder=3)

    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xticks(range(1, 11))
    ax.set_yticks([y[a] for a in order])
    ax.set_yticklabels([a.replace("Agent_", "A") for a in order])
    ax.tick_params(length=0, labelsize=7, colors=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Round (myth written, then game played)", fontsize=7.5, color=INK2, labelpad=2)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=4)
    if mixed:
        fams = [fam[a] for a in order]
        cut = next(k for k in range(1, len(fams)) if fams[k] != fams[k - 1])
        ax.axhline(len(order) - cut - 0.5, color=GRID, lw=0.8, zorder=0)
        for f in dict.fromkeys(fams):
            ys = [y[a] for a in order if fam[a] == f]
            ax.text(10.62, np.mean(ys), f, rotation=270, va="center", ha="left", fontsize=7.5, color=INK2)

    # observed vs expected along this run's edges (children with an uptake row)
    c = u[u["run_id"] == run_id]
    assert len(c) <= len(edges)
    if not mixed:
        return f"Label kept on {int(c['shown'].sum())} of {len(c)} edges; chance* {c['unseen'].sum():.1f}"
    parts = []
    for name, g in (("same family", c[c["family"] == c["parent_family"]]),
                    ("across families", c[c["family"] != c["parent_family"]])):
        parts.append(f"{name} {int(g['shown'].sum())}/{len(g)} (chance* {g['unseen'].sum():.1f})")
    return "Label kept:  " + "\n".join(parts)


def draw_aggregate(ax, agg: pd.DataFrame) -> None:
    rows = [("8-agent homogeneous", "same family", "Homogeneous runs"),
            ("8-agent mixed", "same family", "Mixed runs, shown myth from own family"),
            ("8-agent mixed", "other family", "Mixed runs, shown myth from other family")]
    for k, (setting, exposure, name) in enumerate(rows):
        a = agg[(agg["setting"] == setting) & (agg["exposure"] == exposure)].iloc[0]
        yy = len(rows) - 1 - k
        ax.plot([a.unseen * 100, a.shown * 100], [yy, yy], color=INK2, lw=1.6, solid_capstyle="round", zorder=1)
        ax.scatter(a.shown * 100, yy, s=40, color=INK, zorder=2)
        ax.scatter(a.unseen * 100, yy, s=40, facecolors="none", edgecolors=INK2, linewidths=1.4, zorder=3)
        ax.text(-2, yy, name, ha="right", va="center", fontsize=7.5, color=INK)
        p = a.same_label_excess_p
        ptxt = f"p = {p:.3f}".replace("0.", ".") if p < 0.01 else f"p = {p:.2f}".replace("0.", ".")
        ax.text(101, yy, f"{a.excess * 100:+.1f} (±{a.excess_sd * 100:.1f}) pts   {ptxt}   "
                f"{int(a.same_label_excess_runs_positive)}/{int(a.n_runs)} runs > 0",
                ha="left", va="center", fontsize=7.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(labelsize=7, colors=INK2, length=2)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_xlabel("Children whose moral label matches the myth (mean over runs)", fontsize=7.5, color=INK2)


def main() -> None:
    m, _ = load("moral_labels_z-ai__glm-5.2.csv")
    ex = m[(m["size"] == 8) & (m["task_order"] == "myth_game") & (m["round"] > 1)]
    assert (ex["exposed_round"] == ex["round"] - 1).all() and ex["exposed_author"].notna().all()
    assert (ex.groupby("run_id").size() == 72).all()
    u = uptake_children(m)
    u8 = u[(u["size"] == 8) & (u["task_order"] == "myth_game")]
    agg = aggregate()
    print(agg[["setting", "exposure", "shown", "unseen", "excess", "excess_sd", "same_label_excess_p",
               "same_label_excess_runs_positive", "n_runs"]].to_string())

    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(7.0, 5.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.0], hspace=0.62, wspace=0.16,
                          left=0.06, right=0.955, top=0.88, bottom=0.135)
    notes = []
    for k, (comp, kind) in enumerate(PANELS):
        rid = pick_run(u8, comp, kind)
        rep = int(rid.split("_rep")[1][:2])
        ax = fig.add_subplot(gs[0, k])
        title = f"{'ab'[k]}   {comp}, replicate {rep}"
        note = draw_network(ax, m, u8, rid, kind, title)
        ax.text(0.0, -0.17, note, transform=ax.transAxes, fontsize=6.8, color=INK, va="top", ha="left")
        notes.append((comp, rid, note))

    # legend row
    shown = set(m.loc[m["run_id"].isin([rid for _, rid, _ in notes]), "label"])
    h = [Line2D([], [], ls="", marker="o", ms=6, color=c, label=l) for l, c in LABEL_COLORS.items() if l in shown]
    h += [Line2D([], [], ls="", marker=mk, ms=5.5, mfc="white", mec=INK2, label=f) for f, mk in FAMILY_MARKER.items()]
    h += [Line2D([], [], color=LABEL_COLORS["be generous"], lw=1.5, label="label kept"),
          Line2D([], [], color=MUTED, lw=0.8, label="label changed"),
          Line2D([], [], color=INK2, lw=0.9, ls=(0, (2.0, 1.8)), label="across families")]
    fig.legend(handles=h, loc="upper center", ncol=len(h), bbox_to_anchor=(0.5, 0.975), frameon=False, fontsize=6.8, handlelength=1.6,
               columnspacing=0.9, handletextpad=0.35)

    cax = fig.add_subplot(gs[1, :])
    pos = cax.get_position()
    cax.set_position([0.33, pos.y0, 0.27, pos.height])
    draw_aggregate(cax, agg)
    fig.text(0.02, pos.y1 + 0.012, "c   All 45 runs, 8-agent myth→game:   ● label matches the shown myth    "
             "○ matches an unseen myth*", fontsize=8.5, color=INK, ha="left")
    fig.text(0.02, 0.008, "* Unseen myth: written in the same run and round by another agent of the shown myth's "
             "family (not the writer, not the shown myth's author).\n  Labels: GLM-5.2 judge, Cohen's κ = 0.54 "
             "vs a second judge; no human validation yet. Excess: mean (±sd) over runs; p: Wilcoxon over runs.\n"
             "  Third label \"be cautious\" (crimson) occurs in neither run shown.",
             fontsize=6.4, color=INK2, ha="left")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"moral_lineage_network.{ext}", dpi=300)
    for comp, rid, note in notes:
        print(comp, rid, "|", note)


if __name__ == "__main__":
    main()
