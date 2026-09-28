#!/usr/bin/env python3
"""Frontier update of 2026-09-28: GPT-6 Sol, Opus 5.5 at five replicates, and the
frontier mixed-model populations (2 Gemini 3.1 Pro + 3 GPT-6 Sol + 3 Opus 5.5).

Reads only launcher-audited finals (a run is used only if a launcher receipt lists its
sha256): scripts/run_frontier_rerun.py receipts under
data/json/noise_experiments/frontier_rerun_20260918/ and the
scripts/run_frontier_mixed_populations.py receipt under
data/json/noise_experiments/frontier_mixed_populations_20260928/.

Outputs (docs/figures/frontier_update_20260928/):
  cell_summary.csv          homogeneous frontier arms x size x task order: runs and mean
                            (±sd over runs) of the run-mean final resources
  mixed_family_summary.csv  per task order x family: final resources, mean send and mean
                            return share in the mixed populations beside the same family's
                            homogeneous 8-agent frontier population
  mixed_vs_homogeneous.png  final resources per agent, mixed vs homogeneous, by family
  resources_boxplots.png    grid in the style of the mixed-population figure (Figure 8): rows
                            2 and 8 agents; columns Opus 5.5, Gemini 3.1 Pro, GPT-6 Sol and the
                            2/3/3 mixed population; one dot per run (mean over its agents)
  provenance.json           hashes of every run and output; condition check
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses._shared import configure_matplotlib  # noqa: E402
from src.experiment_condition import output_provenance  # noqa: E402
from scripts.analyze_frontier_rerun import ALLOWED as FRONTIER_ALLOWED  # noqa: E402

FRONTIER_ROOT = ROOT / "data/json/noise_experiments/frontier_rerun_20260918"
MIXED_ROOT = ROOT / "data/json/noise_experiments/frontier_mixed_populations_20260928"
OUTPUT = ROOT / "docs/figures/frontier_update_20260928"
ARM_OF_DIR = {"opus5": "Opus 5", "opus55": "Opus 5.5", "gemini31pro": "Gemini 3.1 Pro",
              "sol_high": "GPT-5.6 Sol", "sol6_high": "GPT-6 Sol"}
FAMILY_OF_MODEL = {"claude-opus-5-5": "Opus 5.5", "gpt-6-sol": "GPT-6 Sol", "gemini-3.1-pro-preview": "Gemini 3.1 Pro"}
FAMILIES = ("Gemini 3.1 Pro", "GPT-6 Sol", "Opus 5.5")
FAMILY_COLORS = {"Opus 5.5": "#7570b3", "GPT-6 Sol": "#d95f02", "Gemini 3.1 Pro": "#1b9e77"}  # as analyses/mixed_dyad_family_split.py
TASK_ORDERS = ("game", "game_myth", "myth_game")
BOX_COLORS = ["#999999", "#e99675", "#72b6a1"]  # as scripts/analyze_mixed_model_populations.py
DOT_COLORS = ["#777777", "#fc8d62", "#66c2a5"]
ORDER_LABELS = {"game": "Game only", "game_myth": "Game → Myth", "myth_game": "Myth → Game"}
ALLOWED = {
    **FRONTIER_ALLOWED,
    "llm.agents": "Mixed runs pin one request plan per agent (design factor: model composition)",
    "implementation": "Frontier and mixed runs use later launcher/src commits; games/ and prompts are unchanged and every launcher asserts non-model inputs equal the September combination",
}
POOL_REASON = ("Mixed runs record one request plan per agent under llm.agents instead of a run-level plan; "
               "every agent plan equals its family's homogeneous frontier profile (launcher plan step and per-call audit).")


def audited(root, receipts):
    listed = {f["sha256"]: ROOT / f["path"] for r in receipts for f in json.loads(r.read_text())["finals"]}
    for sha, path in sorted(listed.items(), key=lambda kv: str(kv[1])):
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise RuntimeError(f"{path} changed after its audit")
        yield path


def arm_of(path):
    name = next(p for p in path.parts if p.startswith("frontier_") and p.endswith("_n5"))
    for key in sorted(ARM_OF_DIR, key=len, reverse=True):
        if name.endswith(f"_{key}_n5"):
            return ARM_OF_DIR[key]
    raise ValueError(name)


def agent_rows(path, arm=None):
    run = json.loads(path.read_text())
    m = run["run_metadata"]
    history = run["conversation_history"]
    if len(history) != 10:
        raise RuntimeError(f"{path}: expected ten rounds, found {len(history)}")
    per_agent = m.get("agent_models") or {a: m["model"] for a in run["agents"]}
    sends, shares = {}, {}
    for r in history:
        for dy in r["dyads"]:
            sends.setdefault(dy["investor"], []).append(dy["sent_decision"])
            if dy["received"]:
                shares.setdefault(dy["trustee"], []).append(dy["returned_decision"] / dy["received"])
    return [{"arm": arm or FAMILY_OF_MODEL[per_agent[a].split("/", 1)[-1]], "num_agents": int(m["num_agents"]),
             "task_order": "_".join(run["task_order"]), "replicate_id": m["replicate_id"], "agent": a,
             "resources": float(v), "mean_send": np.mean(sends.get(a, [np.nan])),
             "mean_return_share": np.mean(shares.get(a, [np.nan])), "path": str(path.resolve())}
            for a, v in history[-1]["balances"].items()]


def load():
    frontier = list(audited(FRONTIER_ROOT, sorted(FRONTIER_ROOT.glob("main*_receipt.json"))))
    mixed = list(audited(MIXED_ROOT, [MIXED_ROOT / "completion_receipt.json"]))
    homo = pd.DataFrame([row for p in frontier for row in agent_rows(p, arm_of(p))])
    mix = pd.DataFrame([row for p in mixed for row in agent_rows(p)])
    if homo.drop_duplicates("path").duplicated(["arm", "num_agents", "task_order", "replicate_id"]).any():
        raise RuntimeError("more than one final for a homogeneous cell/replicate")
    if mix.drop_duplicates("path").duplicated(["task_order", "replicate_id"]).any():
        raise RuntimeError("more than one mixed final for a task order/replicate")
    if len(mixed) != 15:
        raise RuntimeError(f"expected 15 mixed finals, found {len(mixed)}")
    return homo, mix, frontier, mixed


def fmt(mean, sd):
    return f"{mean:.2f} (±{0 if np.isnan(sd) else sd:.2f})" if mean < 5.5 else f"{mean:.1f} (±{0 if np.isnan(sd) else sd:.1f})"


def cell_summary(homo):
    run_means = homo.groupby(["arm", "num_agents", "task_order", "path"]).resources.mean().reset_index()
    out = run_means.groupby(["arm", "num_agents", "task_order"]).resources.agg(runs="count", mean="mean", sd="std").reset_index()
    out["resources"] = [fmt(m, s) for m, s in zip(out["mean"], out["sd"])]
    return out


def mixed_summary(homo, mix):
    rows = []
    for order in TASK_ORDERS:
        for fam in FAMILIES:
            for setting, df in (("mixed 2/3/3", mix[mix.arm == fam]), ("homogeneous 8-agent", homo[(homo.arm == fam) & (homo.num_agents == 8)])):
                sel = df[df.task_order == order]
                # mean over runs of the family's per-run mean, ± sd over runs
                per_run = sel.groupby("path")[["resources", "mean_send", "mean_return_share"]].mean()
                rows.append({"task_order": order, "family": fam, "setting": setting, "runs": len(per_run), "agents": len(sel),
                             **{k: fmt(per_run[k].mean(), per_run[k].std()) for k in ("resources", "mean_send", "mean_return_share")}})
    return pd.DataFrame(rows)


def plot(homo, mix):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    width = .12
    for g, order in enumerate(TASK_ORDERS):
        for f, fam in enumerate(FAMILIES):
            for s, (df, hatch, face) in enumerate(((mix[mix.arm == fam], None, FAMILY_COLORS[fam]),
                                                   (homo[(homo.arm == fam) & (homo.num_agents == 8)], "////", "white"))):
                v = np.sort(df[df.task_order == order].resources.to_numpy())
                pos = g + (f - 1) * 2.4 * width + (s - .5) * 1.1 * width
                ax.boxplot(v, positions=[pos], widths=width, patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=face, edgecolor=FAMILY_COLORS[fam], hatch=hatch, linewidth=1.2),
                           medianprops=dict(color="#222222", linewidth=1.6), whiskerprops=dict(color="#666666"), capprops=dict(color="#666666"))
                ax.scatter(pos + np.linspace(-width / 3, width / 3, len(v)), v, s=9, color=FAMILY_COLORS[fam],
                           edgecolors="white", linewidths=.4, zorder=3, alpha=.8)
    ax.set_xticks(range(3), [ORDER_LABELS[o] for o in TASK_ORDERS])
    ax.set_ylabel("Cumulative resources per agent at round 10\n(axis starts at 60)")
    ax.set_ylim(60, 82)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.22)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Patch(facecolor=FAMILY_COLORS[f], edgecolor=FAMILY_COLORS[f], label=f) for f in FAMILIES]
    handles += [Patch(facecolor="#bbbbbb", edgecolor="#666666", label="mixed population (filled, left)"),
                Patch(facecolor="white", edgecolor="#666666", hatch="////", label="own-family population (hatched, right)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.2), ncol=5, frameon=False, fontsize=9)
    ax.set_title("Frontier mixed populations (2 Gemini 3.1 Pro + 3 GPT-6 Sol + 3 Opus 5.5) vs each family on its own\n"
                 "8 agents · informed negative-only noise · no defectors · 5 runs per cell · each dot is one agent",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT / "mixed_vs_homogeneous.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_grid(homo, mix):
    import matplotlib.pyplot as plt
    configure_matplotlib()
    per_run = pd.concat([homo.assign(panel=homo.arm), mix.assign(panel="Mixed")])
    per_run = per_run.groupby(["panel", "num_agents", "task_order", "path"])["resources"].mean().reset_index()
    panels = [("Opus 5.5", "8 Opus 5.5", "2 Opus 5.5"), ("Gemini 3.1 Pro", "8 Gemini 3.1 Pro", "2 Gemini 3.1 Pro"),
              ("GPT-6 Sol", "8 GPT-6 Sol", "2 GPT-6 Sol"), ("Mixed", "2 Gemini 3.1 Pro + 3 GPT-6 Sol + 3 Opus 5.5", None)]
    labels = [ORDER_LABELS[o] for o in TASK_ORDERS]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharey=True, squeeze=False)
    for row, n_agents in enumerate((2, 8)):
        for ax, (panel, title8, title2) in zip(axes[row], panels):
            title = title2 if n_agents == 2 else title8
            ax.set_xticks([1, 2, 3], labels, fontsize=8)
            ax.set_xlim(.5, 3.5)
            ax.set_ylim(0, 80)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=.22)
            ax.spines[["top", "right"]].set_visible(False)
            if title is None:
                ax.set_title("Mixed dyads\nnot run for this model set", fontsize=11, fontweight="bold", pad=8, color="#777777")
                ax.text(2, 40, "not run", ha="center", va="center", fontsize=11, color="#999999")
                continue
            n_runs = 0
            for pos, order in enumerate(TASK_ORDERS, 1):
                sel = per_run[(per_run.panel == panel) & (per_run.num_agents == n_agents) & (per_run.task_order == order)]
                v = np.sort(sel.resources.to_numpy())
                n_runs = max(n_runs, len(v))
                ax.boxplot(v, positions=[pos], widths=.52, patch_artist=True, showfliers=False, whis=1.5,
                           boxprops=dict(facecolor=BOX_COLORS[pos - 1], edgecolor="#666666"),
                           medianprops=dict(color="#222222", linewidth=1.6),
                           whiskerprops=dict(color="#666666"), capprops=dict(color="#666666"))
                ax.scatter(pos + np.linspace(-.1, .1, len(v)), v, s=30, c=DOT_COLORS[pos - 1], edgecolors="white", linewidths=.6, zorder=3)
            ax.set_xticks([1, 2, 3], labels, fontsize=8)  # boxplot() resets the ticks
            ax.set_xlim(.5, 3.5)
            kind = "mixed" if panel == "Mixed" else "homogeneous"
            ax.set_title(f"{title}\n{kind} · n = {n_runs}", fontsize=11, fontweight="bold", pad=8)
        axes[row][0].set_ylabel(f"{n_agents} agents", fontsize=12, fontweight="bold", labelpad=14)
    fig.suptitle("Final cumulative resources per agent (all agents)\nFrontier update 2026-09-28 (Opus 5.5, Gemini 3.1 Pro, GPT-6 Sol) · "
                 "Informed negative-only noise · No defectors · Round 10", fontsize=14, fontweight="bold")
    fig.text(.5, .012, "Each dot = one run (mean over its agents) · Box = middle 50% · Line = median · Whiskers = up to 1.5 × IQR",
             ha="center", fontsize=9, color="#444444")
    fig.supylabel("Cumulative resources per agent", fontsize=12, x=.006)
    fig.tight_layout(rect=(.02, .05, 1, .91), h_pad=2.2, w_pad=1.6)
    fig.savefig(OUTPUT / "resources_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    homo, mix, frontier, mixed = load()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cells = cell_summary(homo)
    cells.to_csv(OUTPUT / "cell_summary.csv", index=False)
    fam = mixed_summary(homo, mix)
    fam.to_csv(OUTPUT / "mixed_family_summary.csv", index=False)
    plot(homo, mix)
    plot_grid(homo, mix)
    outputs = [p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "provenance.json"]
    frontier_used = [p.resolve() for p in frontier]
    mixed_used = [p.resolve() for p in mixed]
    document = output_provenance(frontier_used + mixed_used, outputs, ALLOWED, output_root=OUTPUT,
                                 pools={"homogeneous": frontier_used, "mixed": mixed_used}, pool_reason=POOL_REASON)
    (OUTPUT / "provenance.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(cells.pivot_table(index=["num_agents", "task_order"], columns="arm", values="resources", aggfunc="first").to_string())
    print()
    print(fam.to_string(index=False))
    print(f"provenance: {document['n_runs']} runs")


if __name__ == "__main__":
    main()
