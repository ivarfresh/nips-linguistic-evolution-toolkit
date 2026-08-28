"""Four-panel figure for the exposure-contrast / rewiring-null analysis.

Visualizes data/analysis/meme_transmission_null_<date>/:

  A. Observed exposure effect vs the rewiring null (does it beat coincidence?)
  B. Observed effect vs future/sham negative controls (is it confounded?)
  C. Adoption rates exposed vs unexposed (the raw contrast behind A)
  D. Prompt elicitation (which families the experiment itself injects)

Panels A-C show 8-agent runs, where the rewiring null and the sham control
are defined; 2-agent contrasts are in exposure_contrast.csv.

Usage (from repo root):
    python analyses/plot_meme_transmission_null.py \
        --input-dir data/analysis/meme_transmission_null_2026_08_28 \
        --output-dir data/plots/meme_transmission_null_2026_08_28
"""

import argparse
import os

import numpy as np
import pandas as pd

from _shared import configure_matplotlib

configure_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402

from plot_meme_evolution import (  # noqa: E402
    BASELINE,
    GRIDLINE,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    PRETTY,
    SURFACE,
    style_axis,
)

OBSERVED = "#2a78d6"
FUTURE = "#eda100"
SHAM = "#1baf7a"
NULL_BAND = "#c3c2b7"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",
                        default="data/analysis/meme_transmission_null_2026_08_28")
    parser.add_argument("--output-dir",
                        default="data/plots/meme_transmission_null_2026_08_28")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    contrast = pd.read_csv(os.path.join(args.input_dir, "exposure_contrast.csv"))
    perm = pd.read_csv(os.path.join(args.input_dir, "permutation_results.csv"))
    elicit = pd.read_csv(os.path.join(args.input_dir, "prompt_elicitation.csv"))

    eight = contrast[contrast["group"] == "8-agent"].set_index("meme_family")
    perm = perm.set_index("meme_family")
    perm["excess"] = perm["observed_diff"] - perm["null_mean"]
    families = list(perm.sort_values("excess").index)  # bottom-to-top ascending
    y = np.arange(len(families))
    labels = [PRETTY.get(f, f) for f in families]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2))
    fig.patch.set_facecolor(SURFACE)
    (ax_null, ax_ctrl), (ax_rates, ax_elicit) = axes
    for ax in axes.flat:
        style_axis(ax)

    # --- A: observed vs rewiring null ------------------------------------
    for yi, f in zip(y, families):
        lo = (perm.loc[f, "null_mean"] - 2 * perm.loc[f, "null_sd"]) * 100
        hi = (perm.loc[f, "null_mean"] + 2 * perm.loc[f, "null_sd"]) * 100
        ax_null.plot([lo, hi], [yi, yi], color=NULL_BAND, linewidth=7,
                     solid_capstyle="round", zorder=2)
        obs = perm.loc[f, "observed_diff"] * 100
        ax_null.plot([obs], [yi], "o", color=OBSERVED, markersize=8, zorder=4)
        ax_null.annotate(f"p={perm.loc[f, 'p_holm']:.3f}",
                         (max(obs, hi) + 1.2, yi), va="center", fontsize=8,
                         color=INK_SECONDARY)
    ax_null.axvline(0, color=BASELINE, linewidth=1, zorder=1)
    ax_null.set_yticks(y)
    ax_null.set_yticklabels(labels, fontsize=9, color=INK)
    ax_null.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax_null.set_xlabel("Adoption difference, exposed − unexposed (pp)",
                       color=INK_SECONDARY, fontsize=9.5)
    ax_null.set_title("A · Observed exposure effect vs rewiring null\n"
                      "(band = null mean ± 2 sd; p Holm-corrected)",
                      color=INK, fontsize=10.5, loc="left")
    ax_null.plot([], [], "o", color=OBSERVED, label="Observed")
    ax_null.plot([], [], "-", color=NULL_BAND, linewidth=7, label="Rewiring null")
    ax_null.legend(frameon=True, facecolor=SURFACE, edgecolor="none",
                   framealpha=1, fontsize=8, labelcolor=INK_SECONDARY,
                   loc="lower right")

    # --- B: observed vs negative controls --------------------------------
    for yi, f in zip(y, families):
        obs = eight.loc[f, "exposure_diff_mean"] * 100
        fut = eight.loc[f, "future_control_diff_mean"] * 100
        sham = eight.loc[f, "sham_control_diff_mean"] * 100
        ax_ctrl.plot([min(obs, fut, sham), max(obs, fut, sham)], [yi, yi],
                     color=GRIDLINE, linewidth=1, zorder=1)
        ax_ctrl.plot([obs], [yi], "o", color=OBSERVED, markersize=8, zorder=4)
        ax_ctrl.plot([fut], [yi], "D", color=FUTURE, markersize=7, zorder=3)
        ax_ctrl.plot([sham], [yi], "s", color=SHAM, markersize=7, zorder=3)
    ax_ctrl.axvline(0, color=BASELINE, linewidth=1, zorder=1)
    ax_ctrl.set_yticks(y)
    ax_ctrl.set_yticklabels([], fontsize=9)
    ax_ctrl.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax_ctrl.set_xlabel("Adoption difference (pp)", color=INK_SECONDARY,
                       fontsize=9.5)
    ax_ctrl.set_title("B · Same effect vs negative controls\n"
                      "(a control ≈ observed means confounded)",
                      color=INK, fontsize=10.5, loc="left")
    ax_ctrl.plot([], [], "o", color=OBSERVED, label="Observed (visible myth)")
    ax_ctrl.plot([], [], "D", color=FUTURE, label="Future myth (never seen)")
    ax_ctrl.plot([], [], "s", color=SHAM, label="Sham myth (same round, not seen)")
    ax_ctrl.legend(frameon=True, facecolor=SURFACE, edgecolor="none",
                   framealpha=1, fontsize=8, labelcolor=INK_SECONDARY,
                   loc="lower right")

    # --- C: adoption exposed vs unexposed --------------------------------
    for yi, f in zip(y, families):
        unexp = eight.loc[f, "adoption_unexposed_mean"] * 100
        exp = eight.loc[f, "adoption_exposed_mean"] * 100
        ax_rates.plot([unexp, exp], [yi, yi], color=NULL_BAND, linewidth=2,
                      zorder=2)
        ax_rates.plot([unexp], [yi], "o", markerfacecolor=SURFACE,
                      markeredgecolor=OBSERVED, markeredgewidth=1.6,
                      markersize=8, zorder=3)
        ax_rates.plot([exp], [yi], "o", color=OBSERVED, markersize=8, zorder=4)
    ax_rates.set_yticks(y)
    ax_rates.set_yticklabels(labels, fontsize=9, color=INK)
    ax_rates.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax_rates.set_xlim(0, 100)
    ax_rates.set_xlabel("Share of child responses carrying the norm (%)",
                        color=INK_SECONDARY, fontsize=9.5)
    ax_rates.set_title("C · Adoption when a visible partner myth\n"
                       "carries the norm vs when none does",
                       color=INK, fontsize=10.5, loc="left")
    ax_rates.plot([], [], "o", color=OBSERVED, label="Exposed")
    ax_rates.plot([], [], "o", markerfacecolor=SURFACE,
                  markeredgecolor=OBSERVED, markeredgewidth=1.6,
                  label="Unexposed")
    ax_rates.legend(frameon=True, facecolor=SURFACE, edgecolor="none",
                    framealpha=1, fontsize=8, labelcolor=INK_SECONDARY,
                    loc="lower right")

    # --- D: prompt elicitation -------------------------------------------
    elicit = elicit.set_index("meme_family").reindex(families)
    rates = elicit["parent_free_rate"].to_numpy() * 100
    ax_elicit.barh(y, rates, height=0.62, color=NULL_BAND, zorder=3)
    for yi, f, rate in zip(y, families, rates):
        note = f"{rate:.0f}%"
        if elicit.loc[f, "system_prompt_hit_capsules"] > 0:
            note += "  (also in every system prompt)"
        ax_elicit.annotate(note, (rate + 1.2, yi), va="center", fontsize=8.5,
                           color=INK_SECONDARY)
    ax_elicit.set_yticks(y)
    ax_elicit.set_yticklabels([], fontsize=9)
    ax_elicit.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax_elicit.set_xlim(0, 100)
    ax_elicit.set_xlabel("Zero-history responses already carrying the norm (%)",
                         color=INK_SECONDARY, fontsize=9.5)
    ax_elicit.set_title("D · Spontaneous emergence with no exposure\n"
                        "(the floor any transmission claim must beat)",
                        color=INK, fontsize=10.5, loc="left")

    fig.text(
        0.01, 0.005,
        "Corpus: 60 informed-noise runs; panels A–C use the 20 8-agent runs "
        "(the rewiring null and sham control are undefined for dyads). "
        "Exposure = norm present in a partner myth visible in the child's "
        "prompt. Null: partner myths replaced by same-run, same-round myths "
        "of other agents (B=10,000). Statistics: run-level means; "
        "see data/analysis/meme_transmission_null_2026_08_28/report.md.",
        fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out = os.path.join(args.output_dir, "meme_transmission_null.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
