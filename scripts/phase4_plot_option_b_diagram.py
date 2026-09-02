"""Option B conditioning diagram for Phase 4.

Shows how the two task orders genuinely diverge under Option B by tracking
the chat-memory state evolution within round 5 (an arbitrary mid-run round).

Output: data/phase4/plots/04_option_b_diagram.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyses._shared import configure_matplotlib


OUT_PATH = Path("data/phase4/plots/04_option_b_diagram.png")

# Colors
C_SEED = "#bccfe6"           # [0:3] persistent prefix
C_PREV_MYTH = "#d1b3e6"       # [3:5] R(N-1) own myth — carried in
C_NEW_MYTH = "#a87bdc"        # [3:5] R(N) own myth — just written this round
C_GAME = "#9ecbe6"            # game-call-related
C_MYTH = "#fff4b8"            # myth-call-related


def rrect(ax, x, y, w, h, color, edge="black", style="-", lw=1.0):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04",
        linewidth=lw, edgecolor=edge, facecolor=color, linestyle=style,
    ))


def state_pill(ax, x, y, w, h, label, messages_summary, has_prev_myth):
    """Draw a chat-memory state pill summarizing what's in agent.messages."""
    # outer box
    rrect(ax, x, y, w, h, "#fafafa", edge="#888", style="--", lw=1.0)
    ax.text(x + 0.15, y + h - 0.15, label, fontsize=9.5, fontweight="bold", va="top")

    # message sub-boxes
    sub_h = 0.45
    cur_y = y + h - 0.7

    # persistent prefix (collapsed into one box)
    rrect(ax, x + 0.2, cur_y - sub_h, w - 0.4, sub_h, C_SEED)
    ax.text(x + w / 2, cur_y - sub_h / 2,
            "[0:3]  system + seed_user + seed_myth   (persistent, 3 msgs)",
            ha="center", va="center", fontsize=8)
    cur_y -= sub_h + 0.07

    # own myth slot (carried-in or fresh)
    if has_prev_myth == "prev":
        rrect(ax, x + 0.2, cur_y - sub_h, w - 0.4, sub_h, C_PREV_MYTH)
        ax.text(x + w / 2, cur_y - sub_h / 2,
                "[3:5]  R(N-1) own myth user+text   (carried from previous round, 2 msgs)",
                ha="center", va="center", fontsize=8)
    elif has_prev_myth == "fresh":
        rrect(ax, x + 0.2, cur_y - sub_h, w - 0.4, sub_h, C_NEW_MYTH)
        ax.text(x + w / 2, cur_y - sub_h / 2,
                "[3:5]  R(N) own myth user+text   (just written this round, 2 msgs)",
                ha="center", va="center", fontsize=8)
    else:
        # round 1, no own myth yet
        rrect(ax, x + 0.2, cur_y - sub_h, w - 0.4, sub_h, "#eeeeee", edge="#bbb", style=":")
        ax.text(x + w / 2, cur_y - sub_h / 2,
                "(round 1: no own myth slot yet)",
                ha="center", va="center", fontsize=8, color="#888", style="italic")

    # message count summary
    cur_y -= sub_h + 0.12
    ax.text(x + w / 2, cur_y, messages_summary,
            ha="center", va="center", fontsize=8.5, color="#444", style="italic")


def llm_call(ax, x, y, w, h, kind, sees_text, response_text, fate_text, remember):
    """Draw an LLM call: sees / response / fate, with the remember flag."""
    color = C_GAME if kind == "game" else C_MYTH
    rrect(ax, x, y, w, h, color, lw=1.2)
    title = ("GAME call" if kind == "game" else "MYTH call") + \
            ("  (remember=False)" if not remember else "  (remember=True)")
    ax.text(x + w / 2, y + h - 0.18, title,
            ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(x + 0.15, y + h - 0.55, "sees:", fontsize=8.5, fontweight="bold", va="top", color="#222")
    ax.text(x + 0.65, y + h - 0.55, sees_text, fontsize=8, va="top")
    ax.text(x + 0.15, y + h - 1.20, "outputs:", fontsize=8.5, fontweight="bold", va="top", color="#222")
    ax.text(x + 0.78, y + h - 1.20, response_text, fontsize=8, va="top")
    ax.text(x + 0.15, y + h - 1.65, "fate:", fontsize=8.5, fontweight="bold", va="top", color="#222")
    ax.text(x + 0.58, y + h - 1.65, fate_text, fontsize=8, va="top", style="italic")


def vert_arrow(ax, x, y_top, y_bot, label=""):
    ax.annotate("", xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle="->", linewidth=1.6, color="#444"))
    if label:
        ax.text(x + 0.15, (y_top + y_bot) / 2, label,
                fontsize=8.5, color="#444", style="italic", va="center")


def draw_panel(ax, x_left, y_top, w, task_order):
    """Draw a column showing the chat-memory evolution within round 5 under Option B."""
    is_myth_first = task_order == "myth_game"
    label = '`["myth","game"]`' if is_myth_first else '`["game","myth"]`'
    color_title = "#1f77b4" if is_myth_first else "#2ca02c"
    note = ("Game decision sees the SAME-ROUND myth\n(written 1 LLM call earlier in this round)"
            if is_myth_first
            else "Game decision sees the PREVIOUS-ROUND myth\n(R4, carried into this round)")

    ax.text(x_left + w / 2, y_top, label, ha="center", va="top",
            fontsize=14, fontweight="bold", color=color_title)
    ax.text(x_left + w / 2, y_top - 0.55, note, ha="center", va="top",
            fontsize=10, color=color_title, style="italic")

    # State pill 1 — at round 5 start
    pill_h = 2.0
    y = y_top - 1.5
    state_pill(ax, x_left, y - pill_h, w, pill_h,
               label="Chat memory at round 5 START",
               messages_summary="Total: 5 messages",
               has_prev_myth="prev")
    y_after_pill1 = y - pill_h

    if is_myth_first:
        # Myth call fires first
        vert_arrow(ax, x_left + w / 2, y_after_pill1 - 0.15, y_after_pill1 - 0.7,
                   label="myth call fires")
        call_h = 2.1
        y_call = y_after_pill1 - 0.85 - call_h
        llm_call(ax, x_left, y_call, w, call_h, kind="myth",
                 sees_text="5 state msgs + myth prompt (6 msgs total)\nincluding R4 own myth (carried in)",
                 response_text="R5 own myth (200-word story)",
                 fate_text="APPENDED to chat memory.\nReplaces 'last_own_myth' slot →\nR4 slot is overwritten by R5.",
                 remember=True)
        # Updated state
        pill2_y = y_call - 0.4 - pill_h
        state_pill(ax, x_left, pill2_y, w, pill_h,
                   label="State after myth call",
                   messages_summary="Total: 5 messages — R5 myth now in slot",
                   has_prev_myth="fresh")
        y_after_pill2 = pill2_y

        # Game call fires second
        vert_arrow(ax, x_left + w / 2, y_after_pill2 - 0.15, y_after_pill2 - 0.7,
                   label="game call fires")
        y_call2 = y_after_pill2 - 0.85 - call_h
        llm_call(ax, x_left, y_call2, w, call_h, kind="game",
                 sees_text="5 state msgs + game prompt (6 msgs total)\nincluding R5 own myth (just written)",
                 response_text="R5 game decision",
                 fate_text="NOT in chat memory.\nUpdates balance scalar for next round.\nDecision conditioned on R5 own myth.",
                 remember=False)
        # Final state for round 5 end
        pill3_y = y_call2 - 0.4 - pill_h
        state_pill(ax, x_left, pill3_y, w, pill_h,
                   label="State at round 5 END  →  becomes R6 start",
                   messages_summary="Total: 5 messages — R5 myth carries to R6",
                   has_prev_myth="fresh")
    else:
        # Game call fires first
        vert_arrow(ax, x_left + w / 2, y_after_pill1 - 0.15, y_after_pill1 - 0.7,
                   label="game call fires")
        call_h = 2.1
        y_call = y_after_pill1 - 0.85 - call_h
        llm_call(ax, x_left, y_call, w, call_h, kind="game",
                 sees_text="5 state msgs + game prompt (6 msgs total)\nincluding R4 own myth (previous round)",
                 response_text="R5 game decision",
                 fate_text="NOT in chat memory.\nUpdates balance scalar for next round.\nDecision conditioned on R4 own myth.",
                 remember=False)
        # State unchanged (game didn't write)
        pill2_y = y_call - 0.4 - pill_h
        state_pill(ax, x_left, pill2_y, w, pill_h,
                   label="State after game call (unchanged)",
                   messages_summary="Total: 5 messages — still has R4 myth",
                   has_prev_myth="prev")
        y_after_pill2 = pill2_y

        # Myth call fires second
        vert_arrow(ax, x_left + w / 2, y_after_pill2 - 0.15, y_after_pill2 - 0.7,
                   label="myth call fires")
        y_call2 = y_after_pill2 - 0.85 - call_h
        llm_call(ax, x_left, y_call2, w, call_h, kind="myth",
                 sees_text="5 state msgs + myth prompt (6 msgs total)\nincluding R4 own myth (carried in)",
                 response_text="R5 own myth (200-word story)",
                 fate_text="APPENDED to chat memory.\nReplaces 'last_own_myth' slot →\nR4 overwritten by R5. Carries to R6.",
                 remember=True)
        # Final state
        pill3_y = y_call2 - 0.4 - pill_h
        state_pill(ax, x_left, pill3_y, w, pill_h,
                   label="State at round 5 END  →  becomes R6 start",
                   messages_summary="Total: 5 messages — R5 myth carries to R6",
                   has_prev_myth="fresh")


def main():
    configure_matplotlib()

    fig, ax = plt.subplots(figsize=(20, 17))

    # Header
    ax.text(0.5, 17.5,
            "Phase 4 Option B — chat memory carries the agent's own most recent myth",
            fontsize=15, fontweight="bold", va="top")
    ax.text(0.5, 16.95,
            "Round N starts with chat memory = [system, seed_user, seed_myth, R(N-1)_own_myth_user, R(N-1)_own_myth_text] = 5 msgs (or 3 in round 1).\n"
            "Myth call has remember=True (response is appended, replacing the previous-round myth slot).\n"
            "Game call has remember=False (decision drives the ledger but doesn't enter chat memory).",
            fontsize=10.5, color="#444", style="italic", va="top")

    # Draw the two side-by-side panels (example: round 5)
    panel_w = 9.0
    draw_panel(ax, 0.5, 16.0, panel_w, "myth_game")
    draw_panel(ax, 10.5, 16.0, panel_w, "game_myth")

    # Summary table at bottom
    table_y = -0.5
    ax.text(0.5, table_y, "Per-LLM-call message counts — Option B",
            fontsize=12, fontweight="bold", va="top")

    table = [
        ("",                "Round",  "Myth call sees",  "Game call sees",  "Game decision conditioned on"),
        ('["myth","game"]', "1",      "4 msgs",          "6 msgs",          "R1 own myth (just written this round)"),
        ('["myth","game"]', "N ≥ 2",  "6 msgs",          "6 msgs",          "R(N) own myth (just written this round)"),
        ('["game","myth"]', "1",      "4 msgs",          "4 msgs",          "no own myth (none exists yet)"),
        ('["game","myth"]', "N ≥ 2",  "6 msgs",          "6 msgs",          "R(N-1) own myth (previous round)"),
    ]
    col_xs = [0.5, 5.5, 7.5, 11.5, 15.5]
    line_h = 0.45
    base_y = table_y - 0.55
    for r, row in enumerate(table):
        y = base_y - r * line_h
        for c, txt in enumerate(row):
            weight = "bold" if r == 0 else "normal"
            ax.text(col_xs[c], y, txt, fontsize=10, va="top", fontweight=weight)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=C_SEED, edgecolor="black",
                       label="Persistent prefix [0:3]  (system + seed_user + seed_myth)"),
        mpatches.Patch(facecolor=C_PREV_MYTH, edgecolor="black",
                       label="R(N−1) own myth [3:5]  — carried in from last round"),
        mpatches.Patch(facecolor=C_NEW_MYTH, edgecolor="black",
                       label="R(N) own myth [3:5]  — just written this round"),
        mpatches.Patch(facecolor=C_GAME, edgecolor="black",
                       label="GAME call"),
        mpatches.Patch(facecolor=C_MYTH, edgecolor="black",
                       label="MYTH call"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.35),
              ncol=5, fontsize=9, frameon=True)

    ax.set_xlim(0, 20)
    ax.set_ylim(-4.5, 18.0)
    ax.axis("off")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
