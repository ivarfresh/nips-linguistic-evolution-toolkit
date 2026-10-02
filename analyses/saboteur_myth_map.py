#!/usr/bin/env python3
"""Myth maps for the frontier saboteur pilot (2026-10-02).

A 2-D map of myth space (PCA of all-mpnet-base-v2 sentence embeddings), in the
style of the myth_map prototype used for the mid-tier linguistic analysis, for
the 12 seed-matched myth -> game runs of the 4 Opus 5 + 4 Sol defector mix
(Agent_4 and Agent_8 always send and return $0):

  rows     normal defector myths  vs  saboteur myths (defectors privately told
           to persuade the others to send less)
  columns  partner myth  vs  shared board

Groups: ordinary Opus 5 (Agent_1-3), ordinary Sol (Agent_5-7), the Opus 5
defector (Agent_4) and the Sol defector (Agent_8); the defectors are the
saboteurs in the saboteur rows.

Outputs (docs/figures/frontier_saboteur_pilot_20261002/):
  saboteur_myth_map_round1.png        where the round-1 myths start
  saboteur_myth_map_trajectories.png  how each group's myths move over 10 rounds
  saboteur_myth_map_points.csv        one row per myth: run, agent, round, PC1, PC2

The PCA is fitted on all 960 myths of the 12 runs. No API calls.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses._shared import configure_matplotlib  # noqa: E402
from src.experiment_condition import output_provenance  # noqa: E402

RUNS = ROOT / "data/json/noise_experiments"
CELLS = {  # (defector myths, channel) -> run folder
    ("normal", "partner"): RUNS / "frontier_defector_pilot_20261001/frontier_defector_pilot_population_myth_game_opus4_sol4_d2_n3",
    ("normal", "board"): RUNS / "frontier_myth_board_20261002/frontier_board_pilot_population_myth_game_opus4_sol4_d2_n3",
    ("saboteur", "partner"): RUNS / "frontier_saboteur_pilot_20261002/frontier_saboteur_pilot_population_myth_game_opus4_sol4_d2_n3",
    ("saboteur", "board"): RUNS / "frontier_saboteur_pilot_20261002/frontier_saboteur_board_pilot_population_myth_game_opus4_sol4_d2_n3",
}
OUTPUT = ROOT / "docs/figures/frontier_saboteur_pilot_20261002"
DEFECTORS = ("Agent_4", "Agent_8")
# Defectors are one per model (Agent_4 Opus 5, Agent_8 Sol), and PC1 largely separates the two models,
# so each defector gets its own line in its model's colour (dashed, X markers) instead of a pooled one.
GROUPS = {"Opus 5": "#7570b3", "Sol": "#d95f02", "Opus 5 defector": "#7570b3", "Sol defector": "#d95f02"}
IS_DEFECTOR = {"Opus 5 defector", "Sol defector"}
ROW_LABEL = {"normal": "Defectors write normal myths", "saboteur": "Defectors write saboteur myths"}
COL_LABEL = {"partner": "Partner's myth", "board": "Shared board"}
NON_FINAL = (".results.json", ".checkpoint.json", ".error.json")
ALLOWED = {
    "protocol.myth.later_rounds_template": "Design factor: channel (partner myth vs shared board prompt)",
    "protocol.myth.board": "Design factor: channel (shared board)",
    "protocol.myth.saboteur": "Design factor: saboteur instruction in the defectors' myth prompts",
    "implementation": ("Board and saboteur runs used the later src commits that add those features (PRs #22, #25); "
                       "runs without them are unchanged, checked by tests and by each launcher's combo comparison"),
    **{k: "Replicate seeds 0-2, matched across the four cells" for k in (
        "replicate.defector_seed", "replicate.noise_seed", "replicate.pairing_seed", "replicate.random_defection_seed", "replicate.run_seed")},
    **{k: "Run identity (set name, output path, replicate id)" for k in (
        "replicate.identity.experiment", "replicate.identity.output_path", "replicate.identity.replicate_id")},
}


def finals(folder):
    paths = sorted(p for p in folder.rglob("*.json") if not p.name.endswith(NON_FINAL))
    assert len(paths) == 3, (folder, len(paths))
    return paths


def group_of(agent_id):
    family = "Opus 5" if int(agent_id.split("_")[1]) <= 4 else "Sol"
    return f"{family} defector" if agent_id in DEFECTORS else family


def load():
    rows = []
    for (myths, channel), folder in CELLS.items():
        for path in finals(folder):
            run = json.loads(path.read_text())
            for entry in run["conversation_history"]:
                for agent_id, text in (entry.get("myths") or {}).items():
                    rows.append({"path": str(path.relative_to(ROOT)), "defector_myths": myths, "channel": channel,
                                 "replicate_id": run["run_metadata"]["replicate_id"], "agent": agent_id,
                                 "group": group_of(agent_id), "round": int(entry["round"]), "text": text})
    df = pd.DataFrame(rows)
    assert len(df) == 12 * 8 * 10, len(df)
    return df


def embed(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    e = model.encode(list(texts), batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(e)


def shade(color, t):
    from matplotlib.colors import to_rgb
    c = np.array(to_rgb(color))
    return tuple(1 - (1 - c) * (0.25 + 0.75 * t))


def plot_round1(df, xl, yl, lims):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5), sharex=True, sharey=True)
    r1 = df[df["round"] == 1]
    for i, myths in enumerate(("normal", "saboteur")):
        for j, channel in enumerate(("partner", "board")):
            ax = axes[i, j]
            cell = r1[(r1.defector_myths == myths) & (r1.channel == channel)]
            for group, color in GROUPS.items():
                s = cell[cell.group == group]
                defector = group in IS_DEFECTOR
                ax.scatter(s.x, s.y, s=70 if defector else 30, color=color, marker="X" if defector else "o", alpha=0.85,
                           lw=0.8 if defector else 0, edgecolors="k" if defector else "none", label=f"{group} (n={len(s)})")
            ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])
            ax.set_title(f"{ROW_LABEL[myths]} · {COL_LABEL[channel]}", fontsize=11, fontweight="bold")
            ax.legend(frameon=False, fontsize=9, loc="best")
            ax.grid(alpha=0.2)
            if i == 1: ax.set_xlabel(xl)
            if j == 0: ax.set_ylabel(yl)
    fig.suptitle("Where the myths start: round-1 myths, written before any game is played\n"
                 "Frontier 4 Opus 5 + 4 Sol with 2 defectors (Agent_4, Agent_8) · Myth → Game · 3 seed-matched runs per panel",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT / "saboteur_myth_map_round1.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(df, xl, yl, lims):
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline
    from scipy.stats import gaussian_kde
    gx, gy = np.mgrid[lims[0][0]:lims[0][1]:200j, lims[1][0]:lims[1][1]:200j]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11.5), sharex=True, sharey=True)
    for i, myths in enumerate(("normal", "saboteur")):
        for j, channel in enumerate(("partner", "board")):
            ax = axes[i, j]
            cell = df[(df.defector_myths == myths) & (df.channel == channel)]
            end = cell[(cell["round"] == 10) & ~cell.group.isin(IS_DEFECTOR)]
            z = gaussian_kde(np.vstack([end.x, end.y]), bw_method=0.25)(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
            ax.contourf(gx, gy, z, levels=12, cmap="Blues", alpha=0.85)
            for group, color in GROUPS.items():
                g = cell[cell.group == group]
                path = g.groupby("round")[["x", "y"]].mean()
                for r in (1, 5, 10):
                    pr = g[g["round"] == r].groupby("path")[["x", "y"]].mean()
                    ax.scatter(pr.x, pr.y, s=8, color="k", alpha=0.55, lw=0, zorder=3)
                t = np.linspace(1, 10, 120)
                sx = make_interp_spline(path.index, path.x, k=3)(t)
                sy = make_interp_spline(path.index, path.y, k=3)(t)
                defector = group in IS_DEFECTOR
                for a in range(len(t) - 1):
                    ax.plot(sx[a:a + 2], sy[a:a + 2], color=shade(color, a / len(t)), lw=3 if defector else 5,
                            linestyle=(0, (2, 1.5)) if defector else "-", solid_capstyle="round", zorder=4)
                ax.annotate("", xy=(sx[-1], sy[-1]), xytext=(sx[-8], sy[-8]),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=0, mutation_scale=28), zorder=5)
                for r in (1, 5, 10):
                    ax.scatter(*path.loc[r], s=110 if defector else 90, color=shade(color, (r - 1) / 9), edgecolor="k", lw=1,
                               zorder=6, marker="X" if defector else "o")
                label = group.replace("defector", "saboteur") if (defector and myths == "saboteur") else group
                ax.text(*path.loc[1], f"  {label}", fontsize=10, color=color, fontweight="bold", zorder=7, va="center")
            ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])
            ax.set_title(f"{ROW_LABEL[myths]} · {COL_LABEL[channel]}  (n = 3 runs)", fontsize=11, fontweight="bold")
            if i == 1: ax.set_xlabel(xl)
            if j == 0: ax.set_ylabel(yl)
    fig.suptitle("How myths move over 10 rounds · Frontier 4 Opus 5 + 4 Sol with 2 defectors · Myth → Game\n"
                 "Solid line = ordinary agents' average, dashed line + X = that model's defector (Agent_4 Opus, Agent_8 Sol); "
                 "light → dark = round 1 → 10; big markers = rounds 1, 5, 10;\nblack dots = each run's group average; "
                 "blue background = where the ordinary agents' round-10 myths end up. PC1 mostly separates the two models.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUTPUT / "saboteur_myth_map_trajectories.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    from sklearn.decomposition import PCA
    configure_matplotlib()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    df = load()
    emb = embed(df["text"])
    pca = PCA(2).fit(emb)
    xy = pca.transform(emb)
    df["x"], df["y"] = xy[:, 0], xy[:, 1]
    v = pca.explained_variance_ratio_ * 100
    xl, yl = f"PC1 ({v[0]:.0f}% of variance)", f"PC2 ({v[1]:.0f}% of variance)"
    pad = 0.05
    lims = ((df.x.min() - pad, df.x.max() + pad), (df.y.min() - pad, df.y.max() + pad))
    df.drop(columns="text").to_csv(OUTPUT / "saboteur_myth_map_points.csv", index=False)
    plot_round1(df, xl, yl, lims)
    plot_trajectories(df, xl, yl, lims)
    runs = sorted({ROOT / p for p in df["path"]})
    outputs = sorted(p for p in OUTPUT.iterdir() if p.is_file() and p.name != "provenance.json" and not p.name.startswith("."))
    pools = {name: [ROOT / p for p in sorted(set(df[(df.defector_myths == m) & (df.channel == c)]["path"]))]
             for name, (m, c) in {"normal_partner": ("normal", "partner"), "normal_board": ("normal", "board"),
                                  "saboteur_partner": ("saboteur", "partner"), "saboteur_board": ("saboteur", "board")}.items()}
    document = output_provenance(runs, outputs, ALLOWED, output_root=OUTPUT, pools=pools,
                                 pool_reason="Four seed-matched cells of the same defector mix; they differ only in channel and saboteur instruction")
    (OUTPUT / "provenance.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    # How far each group's round-10 myths sit from the ordinary agents' round-10 centre, per cell.
    for (m, c), cell in df[df["round"] == 10].groupby(["defector_myths", "channel"]):
        centre = cell[~cell.group.isin(IS_DEFECTOR)][["x", "y"]].mean()
        dist = {g: float(np.hypot(*(cell[cell.group == g][["x", "y"]].mean() - centre))) for g in GROUPS}
        print(f"{m:8s} {c:7s} round-10 distance from ordinary centre: " + ", ".join(f"{g} {d:.3f}" for g, d in dist.items()))
    print(f"PCA variance %: {v.round(1)}")


if __name__ == "__main__":
    main()
