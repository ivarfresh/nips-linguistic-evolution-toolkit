"""Phase 7 plots — decoder asymmetry / behavioral uptake / gowith transplant.

Inputs (produced by phase7_decoder_asymmetry.py):
  data/phase7/decoder_asymmetry_results.json          extract mode, 5 pools
  data/phase7/decoder_behavioral_results.json         behavioral mode, 5 pools + baseline
  data/phase7/decoder_asymmetry_results_gowith.json   extract mode, gowith pool
  data/phase7/decoder_behavioral_results_gowith.json  behavioral mode, gowith pool

Outputs:
  data/phase7/plots/01_behavioral_uptake.png   pools x reader models, round-1 send
  data/phase7/plots/02_stated_vs_enacted.png   Sonnet reader: extracted vs played send
  data/phase7/plots/03_gowith_vs_original.png  s_end+ vs its gowith translation
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyses._shared import configure_matplotlib

DATA = Path("data/phase7")
PLOTS = DATA / "plots"
ENDOWMENT = 5

# Reader series palette validated with dataviz six-checks (CVD/contrast pass in
# this order; cyan's contrast WARN is relieved by direct value labels).
READERS = [("sonnet", "Sonnet 4.5", "#2ca02c"),
           ("gemini", "Gemini 3.1 flash-lite", "#17a2b8"),
           ("gpt", "GPT-5-nano", "#d9640e")]

# Pool colors follow the established phase 5/6 figure identities.
POOL_STYLE = [
    # (key, label, color)
    ("baseline", "Baseline\n(no seed)", "#7f7f7f"),
    ("s_end_minus", "S-end−\n(low-coop)", "#d62728"),
    ("s_end_plus_gpt", "S-end+ GPT\n(round-10)", "#ff7f0e"),
    ("s_start", "S-start\n(round-1)", "#1f77b4"),
    ("s_end_plus_gemini", "S-end+ Gemini\n(round-10)", "#17becf"),
    ("s_end_plus", "S-end+ Sonnet\n(round-10)", "#2ca02c"),
    ("s_end_plus_gowith", "S-end+ gowith\n(translated)", "#a3552e"),
]


def load_cells(path, value_key):
    """-> {(pool, decoder): [values]}"""
    cells = defaultdict(list)
    if not Path(path).exists():
        return cells
    for r in json.loads(Path(path).read_text())["results"]:
        parsed = r.get("parsed") or {}
        val = parsed.get(value_key)
        if value_key == "send" and parsed.get("has_strategy") is False:
            continue
        if val is not None:
            cells[(r["pool"], r["decoder"])].append(float(val))
    return cells


def merge(*cell_dicts):
    out = defaultdict(list)
    for d in cell_dicts:
        for k, v in d.items():
            out[k].extend(v)
    return out


def mean_sd(vals):
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def bar_label(ax, x, top, mean, sd):
    ax.text(x, top + 0.08, f"{mean:.2f}\n(±{sd:.2f})", ha="center", va="bottom", fontsize=8)


def plot_behavioral(behav):
    fig, ax = plt.subplots(figsize=(14, 6.5))
    x = np.arange(len(POOL_STYLE))
    width = 0.26
    for j, (rkey, rlabel, rcolor) in enumerate(READERS):
        for i, (pkey, _, _) in enumerate(POOL_STYLE):
            m, sd = mean_sd(behav.get((pkey, rkey), []))
            if m is None:
                continue
            xpos = x[i] + (j - 1) * width
            ax.bar(xpos, m, width=width - 0.03, yerr=sd, capsize=3,
                   color=rcolor, edgecolor="black", linewidth=0.6, alpha=0.88,
                   label=rlabel if i == 0 else None)
            bar_label(ax, xpos, m + sd, m, sd)
    ax.set_xticks(x)
    ax.set_xticklabels([p[1] for p in POOL_STYLE], fontsize=9.5)
    ax.set_ylabel(f"Round-1 send (0–{ENDOWMENT}), seed in chat memory")
    ax.set_ylim(0, ENDOWMENT + 2.2)
    ax.axhline(ENDOWMENT, color="red", linestyle="--", alpha=0.5, linewidth=1.2)
    ax.legend(loc="upper left", fontsize=10, title="Reader (game-playing model)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        "Phase 7 — behavioral uptake by reader model\n"
        "Round-1 send with the seed myth injected in the Phase 3 shape · 5 seeds/pool × 3 samples · "
        "Gemini reads at ceiling regardless; GPT-written myths move no reader (incl. GPT itself)",
        fontsize=11,
    )
    out = PLOTS / "01_behavioral_uptake.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_stated_vs_enacted(extract, behav):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    pools = POOL_STYLE
    x = np.arange(len(pools))
    width = 0.36
    baseline_m, _ = mean_sd(behav.get(("baseline", "sonnet"), []))
    for i, (pkey, _, pcolor) in enumerate(pools):
        sm, ssd = mean_sd(extract.get((pkey, "sonnet"), []))
        em, esd = mean_sd(behav.get((pkey, "sonnet"), []))
        if sm is not None:
            ax.bar(x[i] - width / 2, sm, width=width - 0.03, yerr=ssd, capsize=3,
                   facecolor="white", edgecolor=pcolor, linewidth=2.2, hatch="//",
                   label="Stated (extracted recipe)" if i == 1 else None)
            bar_label(ax, x[i] - width / 2, sm + ssd, sm, ssd)
        if em is not None:
            ax.bar(x[i] + width / 2, em, width=width - 0.03, yerr=esd, capsize=3,
                   color=pcolor, edgecolor="black", linewidth=0.6, alpha=0.9,
                   label="Enacted (round-1 send)" if i == 1 else None)
            bar_label(ax, x[i] + width / 2, em + esd, em, esd)
    if baseline_m is not None:
        ax.axhline(baseline_m, color="#7f7f7f", linestyle=":", linewidth=1.5, alpha=0.8,
                   label=f"Unseeded baseline send ({baseline_m:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels([p[1] for p in pools], fontsize=9.5)
    ax.set_ylabel(f"Send amount (0–{ENDOWMENT})")
    ax.set_ylim(0, ENDOWMENT + 1.4)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        "Phase 7 — legible to all, binding to some (Sonnet 4.5 reader)\n"
        "Hatched outline = recipe Sonnet EXTRACTS from the myth when asked · solid = what it PLAYS with the myth in memory\n"
        "GPT-written myths: stated send 5.00 but enacted 3.13 ≈ baseline — the transfer failure is bindingness, not legibility",
        fontsize=11,
    )
    out = PLOTS / "02_stated_vs_enacted.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_gowith(extract, behav):
    fig, ax = plt.subplots(figsize=(9, 6))
    entities = [("s_end_plus", "S-end+ original", "#2ca02c"),
                ("s_end_plus_gowith", "S-end+ gowith translation", "#a3552e")]
    groups = [("Stated (extracted)", extract), ("Enacted (round-1 send)", behav)]
    x = np.arange(len(groups))
    width = 0.32
    baseline_m, _ = mean_sd(behav.get(("baseline", "sonnet"), []))
    for j, (ekey, elabel, ecolor) in enumerate(entities):
        for i, (_, cells) in enumerate(groups):
            m, sd = mean_sd(cells.get((ekey, "sonnet"), []))
            if m is None:
                continue
            xpos = x[i] + (j - 0.5) * width
            ax.bar(xpos, m, width=width - 0.03, yerr=sd, capsize=4,
                   color=ecolor, edgecolor="black", linewidth=0.7, alpha=0.9,
                   label=elabel if i == 0 else None)
            bar_label(ax, xpos, m + sd, m, sd)
    if baseline_m is not None:
        ax.axhline(baseline_m, color="#7f7f7f", linestyle=":", linewidth=1.5, alpha=0.8,
                   label=f"Unseeded baseline send ({baseline_m:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=11)
    ax.set_ylabel(f"Send amount (0–{ENDOWMENT})")
    ax.set_ylim(0, ENDOWMENT + 1.2)
    ax.legend(loc="upper center", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        "Phase 7 — the recipe survives the gowith grammar transplant (Sonnet 4.5 reader)\n"
        "Subject-less relational grammar, numbers pinned · extraction unchanged · ~85% of the behavioral lift retained",
        fontsize=11,
    )
    out = PLOTS / "03_gowith_vs_original.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    configure_matplotlib()
    PLOTS.mkdir(parents=True, exist_ok=True)
    extract = merge(load_cells(DATA / "decoder_asymmetry_results.json", "send"),
                    load_cells(DATA / "decoder_asymmetry_results_gowith.json", "send"))
    behav = merge(load_cells(DATA / "decoder_behavioral_results.json", "send"),
                  load_cells(DATA / "decoder_behavioral_results_gowith.json", "send"))
    plot_behavioral(behav)
    plot_stated_vs_enacted(extract, behav)
    plot_gowith(extract, behav)


if __name__ == "__main__":
    main()
