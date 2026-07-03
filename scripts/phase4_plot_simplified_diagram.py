"""Simplified conditioning diagram for Phase 3 + Phase 4 (Option A).

Collapses the three-task-order view (which showed them as distinct conditions)
into the actual reduced design: there are only TWO conditions — game-only and
game-plus-myth — because under Option A the temporal order of myth vs game
within a round doesn't change what either call sees.

Output: data/phase4/plots/03_simplified_diagram.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyses._shared import configure_matplotlib


OUT_PATH = Path("data/phase4/plots/03_simplified_diagram.png")

COLOR_PERSISTENT = "#bccfe6"
COLOR_FRESH = "#f4c98a"
COLOR_RESPONSE = "#c6e3c6"
COLOR_DISCARDED = "#e6e6e6"
COLOR_SAVED = "#fff4b8"


def rounded_rect(ax, x, y, w, h, color, edge="black", linestyle="-", linewidth=1.0):
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=color,
        linestyle=linestyle,
    )
    ax.add_patch(rect)


def draw_call_block(ax, x, y_top, w, kind="game"):
    """Draw an LLM call with messages [0:3] = persistent prefix, [3] = task prompt.

    kind: 'game' or 'myth'
    """
    # Title
    title = "GAME call" if kind == "game" else "MYTH call"
    ax.text(x + w / 2, y_top, title, ha="center", va="top",
            fontsize=11, fontweight="bold")

    # Persistent prefix box (collapsed view — single box covering [0:3])
    box_top = y_top - 0.6
    prefix_h = 1.6
    rounded_rect(ax, x, box_top - prefix_h, w, prefix_h, COLOR_PERSISTENT)
    ax.text(
        x + w / 2,
        box_top - prefix_h / 2,
        "[0:3]  PERSISTENT PREFIX\n(system + seed_user + seed_myth)\nbyte-identical every round, every call",
        ha="center", va="center", fontsize=8.5,
    )

    # Fresh task prompt box
    gap = 0.18
    fresh_h = 1.05
    fresh_top = box_top - prefix_h - gap
    rounded_rect(ax, x, fresh_top - fresh_h, w, fresh_h, COLOR_FRESH)
    if kind == "game":
        fresh_text = "[3]  GAME PROMPT\nRound N, balance, role, partner"
    else:
        fresh_text = "[3]  MYTH PROMPT\nround-1 template, every round"
    ax.text(
        x + w / 2,
        fresh_top - fresh_h / 2,
        fresh_text,
        ha="center", va="center", fontsize=9,
    )

    # Arrow
    arrow_top = fresh_top - fresh_h - 0.18
    arrow_bot = arrow_top - 0.55
    ax.annotate(
        "",
        xy=(x + w / 2, arrow_bot),
        xytext=(x + w / 2, arrow_top),
        arrowprops=dict(arrowstyle="->", linewidth=1.6, color="#444"),
    )
    ax.text(x + w / 2 + 0.2, (arrow_top + arrow_bot) / 2, "LLM",
            fontsize=8, color="#444", style="italic")

    # Response
    resp_h = 0.9
    resp_top = arrow_bot - 0.05
    rounded_rect(ax, x, resp_top - resp_h, w, resp_h, COLOR_RESPONSE)
    resp_text = "game response\n(send/return + reasoning)" if kind == "game" else "myth response\n(200-word story)"
    ax.text(x + w / 2, resp_top - resp_h / 2, resp_text,
            ha="center", va="center", fontsize=8.5)

    # Fate
    fate_top = resp_top - resp_h - 0.18
    fate_h = 0.95
    fate_color = COLOR_DISCARDED if kind == "game" else COLOR_SAVED
    rounded_rect(ax, x, fate_top - fate_h, w, fate_h, fate_color,
                 edge="#666", linestyle="--", linewidth=0.8)
    fate_text = (
        "NOT in chat memory.\nDrives ledger →\nupdates balance scalar"
        if kind == "game"
        else "NOT in chat memory.\nSaved to sim_data\nfor later analysis"
    )
    ax.text(x + w / 2, fate_top - fate_h / 2, fate_text,
            ha="center", va="center", fontsize=8, style="italic")

    return fate_top - fate_h  # bottom y


def main():
    configure_matplotlib()

    fig, ax = plt.subplots(figsize=(13, 11))

    # === Top section: the persistent state and how rounds use it ===
    ax.text(0.4, 11.0,
            "Phase 3 + Phase 4 (Option A) — simplified",
            fontsize=14, fontweight="bold", va="top")
    ax.text(0.4, 10.4,
            "Because no LLM-call response is appended to chat memory and the seed is re-injected every round,\n"
            "the order of myth vs game within a round is informationally irrelevant. Only TWO conditions exist:",
            fontsize=10.5, color="#444", style="italic", va="top")

    # === Two conditions ===
    # Condition 1: GAME ONLY (1 call/round)
    ax.text(2.55, 9.3, 'CONDITION 1: game-only',
            fontsize=13, fontweight="bold", va="top", ha="center")
    ax.text(2.55, 8.85, '`["game"]`  —  1 LLM call / round',
            fontsize=10, color="#444", style="italic", va="top", ha="center")
    draw_call_block(ax, 0.6, 8.1, 4.3, kind="game")

    # Vertical divider
    ax.plot([6.5, 6.5], [-0.5, 9.5], color="#bbb", linewidth=1.0, linestyle=":")

    # Condition 2: GAME + MYTH (2 calls/round, ORDER DOESN'T MATTER)
    ax.text(11.65, 9.3, 'CONDITION 2: game + myth',
            fontsize=13, fontweight="bold", va="top", ha="center")
    ax.text(11.65, 8.85,
            '`["myth","game"]`  ≡  `["game","myth"]`   —   2 LLM calls / round (order is informationally irrelevant)',
            fontsize=10, color="#444", style="italic", va="top", ha="center")

    draw_call_block(ax, 7.0, 8.1, 4.3, kind="game")
    draw_call_block(ax, 12.0, 8.1, 4.3, kind="myth")

    # "either before or after" annotation between the two calls
    ax.annotate("", xy=(12.0, 3.0), xytext=(11.3, 3.0),
                arrowprops=dict(arrowstyle="<->", linewidth=1.4, color="#888"))
    ax.text(11.65, 3.4, "either order\n(no info flow)",
            ha="center", va="bottom", fontsize=8.5, color="#666", style="italic")

    # === Bottom: the empirical confirmation ===
    ax.text(0.4, -0.0,
            "Empirical confirmation (n=5 per cell, joint balance ± sd):",
            fontsize=10.5, fontweight="bold", va="top")

    table_lines = [
        ("",                 'Phase 3 ["game"]',  'P4 ["myth","game"]',  'P4 ["game","myth"]'),
        ("Baseline",         "$437.4 (±$5.5)",    "$440.4 (±$5.8)",       "$440.8 (±$2.9)"),
        ("S-start",          "$499.0 (±$11.5)",   "$511.6 (±$16.5)",      "$503.8 (±$14.0)"),
        ("S-end+",           "$600.0 (±$0.0)",    "$598.4 (±$3.6)",       "$599.2 (±$1.8)"),
    ]
    col_xs = [0.4, 3.6, 7.4, 11.4]
    base_y = -0.6
    line_h = 0.45
    for r, row in enumerate(table_lines):
        y = base_y - r * line_h
        for c, txt in enumerate(row):
            weight = "bold" if r == 0 or c == 0 else "normal"
            ax.text(col_xs[c], y, txt, fontsize=9, va="top", fontweight=weight)

    ax.text(0.4, -3.3,
            "All three columns are statistically indistinguishable per row → consistent with the design property at top.",
            fontsize=10, color="#444", style="italic", va="top")

    # Legend in upper right corner of figure
    legend_handles = [
        mpatches.Patch(facecolor=COLOR_PERSISTENT, edgecolor="black",
                       label="Persistent prefix (same every round, every call)"),
        mpatches.Patch(facecolor=COLOR_FRESH, edgecolor="black",
                       label="Fresh task prompt (round-specific scalar info)"),
        mpatches.Patch(facecolor=COLOR_RESPONSE, edgecolor="black",
                       label="LLM response"),
        mpatches.Patch(facecolor=COLOR_DISCARDED, edgecolor="#666", linestyle="--",
                       label="Game response: updates balance scalar; NOT in chat memory"),
        mpatches.Patch(facecolor=COLOR_SAVED, edgecolor="#666", linestyle="--",
                       label="Myth response: saved to sim_data; NOT in chat memory"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.38), ncol=2, fontsize=9, frameon=True)

    ax.set_xlim(0, 17)
    ax.set_ylim(-5.5, 11.5)
    ax.axis("off")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
