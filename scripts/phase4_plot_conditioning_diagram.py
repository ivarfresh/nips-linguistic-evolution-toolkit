"""Conditioning diagram for the Phase 3 + Phase 4 regime.

For each of the three task orders (`["game"]`, `["myth","game"]`, `["game","myth"]`),
shows exactly what each LLM call's messages_sent contains and what its response
is conditioned on. Highlights the design property: per-LLM-call conditioning is
identical across task orders.

Output: data/phase4/plots/02_conditioning_diagram.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyses._shared import configure_matplotlib


OUT_PATH = Path("data/phase4/plots/02_conditioning_diagram.png")


# Color scheme
COLOR_PERSISTENT = "#bccfe6"   # locked across all rounds & both tasks
COLOR_FRESH = "#f4c98a"        # changes each round (current task prompt)
COLOR_RESPONSE = "#c6e3c6"     # response box
COLOR_DISCARDED = "#e6e6e6"    # response NOT appended to chat memory
COLOR_SAVED = "#fff4b8"        # response saved to sim_data


# Each "call" is a dict: {label, messages: [(name, kind), ...], response_name, response_fate}
# kind: "persistent" or "fresh"
# fate: "discarded" (game) or "saved_to_sim" (myth) — both are NOT in chat memory
ROW_DEFS = [
    {
        "title": '["game"]   —   one LLM call per round',
        "calls": [
            {
                "label": "GAME call (round N)",
                "messages": [
                    ("[0] system\n(game rules)", "persistent"),
                    ("[1] user\n(seed user prompt)", "persistent"),
                    ("[2] assistant\n(SEED MYTH)", "persistent"),
                    ("[3] user\n(game prompt:\nRound N, balance,\nrole, partner)", "fresh"),
                ],
                "response": "game response\n(send/return + reasoning)",
                "fate": "drives ledger →\nupdates balance scalar\nfor next round",
                "fate_color": COLOR_DISCARDED,
            },
        ],
    },
    {
        "title": '["myth","game"]   —   two LLM calls per round (myth first, then game)',
        "calls": [
            {
                "label": "MYTH call (round N)",
                "messages": [
                    ("[0] system", "persistent"),
                    ("[1] user\n(seed user prompt)", "persistent"),
                    ("[2] assistant\n(SEED MYTH)", "persistent"),
                    ("[3] user\n(myth prompt:\nround-1 template\nevery round)", "fresh"),
                ],
                "response": "myth response\n(200-word story)",
                "fate": "saved to sim_data\n(for later analysis)\nNOT in chat memory",
                "fate_color": COLOR_SAVED,
            },
            {
                "label": "GAME call (round N)",
                "messages": [
                    ("[0] system", "persistent"),
                    ("[1] user\n(seed user prompt)", "persistent"),
                    ("[2] assistant\n(SEED MYTH)", "persistent"),
                    ("[3] user\n(game prompt:\nRound N, balance,\nrole, partner)", "fresh"),
                ],
                "response": "game response\n(send/return + reasoning)",
                "fate": "drives ledger →\nupdates balance scalar\nfor next round",
                "fate_color": COLOR_DISCARDED,
            },
        ],
    },
    {
        "title": '["game","myth"]   —   two LLM calls per round (game first, then myth)',
        "calls": [
            {
                "label": "GAME call (round N)",
                "messages": [
                    ("[0] system", "persistent"),
                    ("[1] user\n(seed user prompt)", "persistent"),
                    ("[2] assistant\n(SEED MYTH)", "persistent"),
                    ("[3] user\n(game prompt:\nRound N, balance,\nrole, partner)", "fresh"),
                ],
                "response": "game response\n(send/return + reasoning)",
                "fate": "drives ledger →\nupdates balance scalar\nfor next round",
                "fate_color": COLOR_DISCARDED,
            },
            {
                "label": "MYTH call (round N)",
                "messages": [
                    ("[0] system", "persistent"),
                    ("[1] user\n(seed user prompt)", "persistent"),
                    ("[2] assistant\n(SEED MYTH)", "persistent"),
                    ("[3] user\n(myth prompt:\nround-1 template\nevery round)", "fresh"),
                ],
                "response": "myth response\n(200-word story)",
                "fate": "saved to sim_data\n(for later analysis)\nNOT in chat memory",
                "fate_color": COLOR_SAVED,
            },
        ],
    },
]


def draw_call(ax, x, y, call, call_width=4.8, msg_h=0.85, gap=0.18, label_height=0.5):
    """Draw a single LLM call as: 4 message boxes (stacked), then arrow → response box.

    x, y: top-left of the call group.
    """
    # Call label
    ax.text(x + call_width / 2, y, call["label"], ha="center", va="top",
            fontsize=10, fontweight="bold")

    # Messages box (stacked)
    msg_top = y - label_height
    for i, (name, kind) in enumerate(call["messages"]):
        color = COLOR_PERSISTENT if kind == "persistent" else COLOR_FRESH
        rect = mpatches.FancyBboxPatch(
            (x, msg_top - (i + 1) * msg_h - i * gap),
            call_width,
            msg_h,
            boxstyle="round,pad=0.04",
            linewidth=1.0,
            edgecolor="black",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(
            x + call_width / 2,
            msg_top - i * (msg_h + gap) - msg_h / 2,
            name,
            ha="center",
            va="center",
            fontsize=8.5,
        )

    # Bottom of messages stack
    msgs_bottom = msg_top - len(call["messages"]) * msg_h - (len(call["messages"]) - 1) * gap

    # Arrow to response
    arrow_top = msgs_bottom - 0.20
    arrow_bot = arrow_top - 0.55
    ax.annotate(
        "",
        xy=(x + call_width / 2, arrow_bot),
        xytext=(x + call_width / 2, arrow_top),
        arrowprops=dict(arrowstyle="->", linewidth=1.6, color="#444"),
    )
    ax.text(
        x + call_width / 2 + 0.18,
        (arrow_top + arrow_bot) / 2,
        "LLM",
        fontsize=8,
        color="#444",
        style="italic",
    )

    # Response box
    resp_h = 1.05
    resp = mpatches.FancyBboxPatch(
        (x, arrow_bot - resp_h),
        call_width,
        resp_h,
        boxstyle="round,pad=0.04",
        linewidth=1.0,
        edgecolor="black",
        facecolor=COLOR_RESPONSE,
    )
    ax.add_patch(resp)
    ax.text(
        x + call_width / 2,
        arrow_bot - resp_h / 2,
        call["response"],
        ha="center",
        va="center",
        fontsize=8.5,
    )

    # Fate annotation below response
    fate_top = arrow_bot - resp_h - 0.18
    fate_h = 1.05
    fate_box = mpatches.FancyBboxPatch(
        (x, fate_top - fate_h),
        call_width,
        fate_h,
        boxstyle="round,pad=0.04",
        linewidth=0.8,
        edgecolor="#666",
        facecolor=call["fate_color"],
        linestyle="--",
    )
    ax.add_patch(fate_box)
    ax.text(
        x + call_width / 2,
        fate_top - fate_h / 2,
        call["fate"],
        ha="center",
        va="center",
        fontsize=8,
        style="italic",
    )

    return fate_top - fate_h  # bottom-y of the whole call group


def draw_row(ax, y_top, row_def, x_start=0.5, row_width=18.0):
    """Draw one task order row."""
    # Title
    ax.text(x_start, y_top, row_def["title"], ha="left", va="top",
            fontsize=12, fontweight="bold")

    calls = row_def["calls"]
    n = len(calls)
    call_width = 4.8
    inter_gap = 1.4

    if n == 1:
        x_offset = x_start + (row_width - call_width) / 2 - x_start
    else:
        # center the n calls
        total_w = n * call_width + (n - 1) * inter_gap
        x_offset = (row_width - total_w) / 2

    for i, call in enumerate(calls):
        cx = x_start + x_offset + i * (call_width + inter_gap)
        cy = y_top - 0.6  # below title
        draw_call(ax, cx, cy, call, call_width=call_width)

        # Arrow between consecutive calls
        if i < n - 1:
            arrow_y = y_top - 5.0
            ax.annotate(
                "",
                xy=(cx + call_width + 0.05, arrow_y),
                xytext=(cx + call_width + inter_gap - 0.05, arrow_y),
                arrowprops=dict(arrowstyle="<-", linewidth=1.4, color="#888"),
            )
            ax.text(
                cx + call_width + inter_gap / 2,
                arrow_y + 0.25,
                "then",
                ha="center",
                fontsize=9,
                color="#666",
            )


def main():
    configure_matplotlib()

    fig, ax = plt.subplots(figsize=(18, 22))

    # Each row needs ~7.5 vertical units. 3 rows + headers.
    row_height = 7.5
    row_gap = 1.0
    y_starts = [22.0, 22.0 - row_height - row_gap, 22.0 - 2 * (row_height + row_gap)]

    for y_top, row_def in zip(y_starts, ROW_DEFS):
        draw_row(ax, y_top, row_def)

    # Legend
    legend_y = -2.5
    legend_handles = [
        mpatches.Patch(facecolor=COLOR_PERSISTENT, edgecolor="black",
                       label="Persistent across all rounds and both tasks\n(messages [0:3] of every LLM call)"),
        mpatches.Patch(facecolor=COLOR_FRESH, edgecolor="black",
                       label="Fresh each round (message [3] of every LLM call)\nConditioned only on round number, role, balance scalar"),
        mpatches.Patch(facecolor=COLOR_RESPONSE, edgecolor="black",
                       label="LLM response"),
        mpatches.Patch(facecolor=COLOR_DISCARDED, edgecolor="#666", linestyle="--",
                       label="Response NOT in chat memory; only effect is updating the balance scalar"),
        mpatches.Patch(facecolor=COLOR_SAVED, edgecolor="#666", linestyle="--",
                       label="Response saved to sim_data for analysis; NOT in chat memory"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=10, frameon=True)

    # Top header note
    ax.text(
        0.5,
        24.0,
        "What each LLM call is conditioned on   ·   Phase 3 + Phase 4 (Option A) regime, seeded condition",
        ha="left", va="bottom", fontsize=13, fontweight="bold",
    )
    ax.text(
        0.5,
        23.0,
        "Key property: messages [0:3] are byte-identical across rounds and across both tasks. Only message [3] (and the response that uses it) changes.",
        ha="left", va="bottom", fontsize=10.5, style="italic", color="#444",
    )

    ax.set_xlim(0, 19)
    ax.set_ylim(-4.5, 25)
    ax.axis("off")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
