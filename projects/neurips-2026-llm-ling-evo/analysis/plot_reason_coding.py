#!/usr/bin/env python3
"""Reason-coding asymmetry figure: Claude vs GPT-5-Nano on the share of
game-response prose that references the agent's own preceding myth.

Visualises the §4.5 finding: Claude threads myth vocabulary into 78–82%
of game responses; GPT-5-Nano emits no reasoning prose at all and so
the metric is undecidable for that model (shown as gray bars with
explanatory annotation).

Output: analysis/figures/reason_coding.png + .pdf
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).parent
SUMMARIES = ANALYSIS_DIR / "cell_summaries"
FIG_DIR = ANALYSIS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

NOISE_ORDER = ["positive", "negative_5", "bootstrap", "deterministic_max"]
NOISE_LABELS = {
    "positive": "positive",
    "negative_5": "negative",
    "bootstrap": "bootstrap",
    "deterministic_max": "det. max",
}


def main():
    df = pd.read_csv(SUMMARIES / "reason_coding_summary.csv")
    df["noise_rank"] = df["noise_label"].map(
        {n: i for i, n in enumerate(NOISE_ORDER)}
    ).fillna(99)
    df["task_rank"] = df["task_order"].map(
        {"game_myth": 0, "myth_game": 1}
    ).fillna(99)
    df = df.sort_values(["model", "noise_rank", "informed", "task_rank"])

    df["row_label"] = df.apply(
        lambda r: f"{NOISE_LABELS.get(r['noise_label'], r['noise_label'])}"
                  + (" (inf)" if r["informed"] else "")
                  + f" | {r['task_order']}",
        axis=1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=False)

    for ax, model, color in zip(
        axes,
        ["claude-sonnet-4.5", "gpt-5-nano"],
        ["#d97706", "#0ea5e9"],
    ):
        sub = df[df["model"] == model].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        y = np.arange(len(sub))
        share_theme = sub["share_theme_hit"].values
        share_own = sub["share_own_myth_hit"].values

        # Two-bar group: theme + own-myth
        bar_h = 0.35
        ax.barh(y - bar_h / 2, share_theme, bar_h,
                color=color, alpha=0.4,
                label="theme-lexicon hit (story / spirit / elder / ...)")
        ax.barh(y + bar_h / 2, share_own, bar_h,
                color=color, alpha=0.95,
                label="own-myth vocabulary hit")

        # Annotate values on the right of each bar
        for i, (t, o) in enumerate(zip(share_theme, share_own)):
            ax.text(t + 0.01, i - bar_h / 2, f"{t:.2f}",
                    va="center", fontsize=7, color="gray")
            ax.text(o + 0.01, i + bar_h / 2, f"{o:.2f}",
                    va="center", fontsize=7, color="black")

        ax.set_yticks(y)
        ax.set_yticklabels(sub["row_label"], fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Share of round-level game responses with hit", fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, frameon=True)

        title = model.split("-")[0] + (" (Claude Sonnet 4.5)" if "claude" in model else " (GPT-5-Nano)")
        ax.set_title(title, fontsize=11, fontweight="bold", color=color)

        if "gpt" in model:
            ax.text(
                0.5, 0.97,
                "All cells = 0.00.\nGPT-5-Nano emits only the JSON action\n"
                "(`{\"send\": 3}`, mean ~12 chars).\nNo prose to code →\nmetric is "
                "undecidable from\nvisible output, not absent.",
                ha="center", va="top", transform=ax.transAxes,
                fontsize=9, color="#444444",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef3c7",
                          edgecolor="#d97706", alpha=0.9),
            )

    fig.suptitle(
        "Does myth content enter game-response prose?\n"
        "Lexical proxy on `game_responses[ag].content` against own preceding myth chain",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    out = FIG_DIR / "reason_coding.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(FIG_DIR / "reason_coding.pdf", bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
