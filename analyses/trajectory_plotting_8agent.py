#!/usr/bin/env python3
"""Per-run trajectory plots for 8-agent (multi-dyad) simulations.

Same 2x2 layout as trajectory_plotting.py (Transaction Flow, Payoffs per
Round, Cumulative Balances by agent and by role), adapted to rounds with 4
dyads: transaction and payoff panels show dyad means; the agent-balance panel
shows all 8 agents with the original role markers (^ investor, o trustee);
the role-balance panel shows the mean balance of that round's investors and
trustees.

Usage:
    python analyses/trajectory_plotting_8agent.py <input.json> <output_dir> [title]
    # or via env vars like the other analyses:
    ANALYSIS_INPUT_FILE=... ANALYSIS_OUTPUT_DIR=... python analyses/trajectory_plotting_8agent.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_numerical_trajectory_8agent(
    conversation_history, save_path=None, title="Trust Game Trajectory"
):
    game_rounds = [r for r in conversation_history if r.get("dyads")]
    if not game_rounds:
        print("No game rounds with dyads to plot")
        return

    rounds = [r["round"] for r in game_rounds]
    min_round, max_round = min(rounds), max(rounds)
    xticks = list(range(min_round, max_round + 1))

    def dyad_mean(r, key):
        vals = [d[key] for d in r["dyads"] if d.get(key) is not None]
        return float(np.mean(vals)) if vals else 0.0

    sent = [dyad_mean(r, "sent") for r in game_rounds]
    received = [dyad_mean(r, "received") for r in game_rounds]
    returned = [dyad_mean(r, "returned") for r in game_rounds]
    investor_payoff = [dyad_mean(r, "investor_payoff") for r in game_rounds]
    trustee_payoff = [dyad_mean(r, "trustee_payoff") for r in game_rounds]

    agent_ids = sorted(game_rounds[0].get("balances", {}).keys())
    agent_balances = {
        a: [float(r["balances"][a]) for r in game_rounds] for a in agent_ids
    }
    roles_per_round = [r.get("roles", {}) for r in game_rounds]

    investor_role_balance = []
    trustee_role_balance = []
    for r in game_rounds:
        roles = r.get("roles", {})
        inv = [float(r["balances"][a]) for a, role in roles.items() if role == "investor"]
        tru = [float(r["balances"][a]) for a, role in roles.items() if role == "trustee"]
        investor_role_balance.append(float(np.mean(inv)) if inv else 0.0)
        trustee_role_balance.append(float(np.mean(tru)) if tru else 0.0)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Plot 1: Transaction amounts (sent, received, returned) — dyad means
    ax1 = axes[0, 0]
    for values, label in [(sent, "Sent"), (received, "Received"), (returned, "Returned")]:
        ax1.plot(rounds, values, label=label, linewidth=2, marker="o", markersize=6)
    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_ylabel("Amount (dyad mean)", fontsize=12)
    ax1.set_title("Transaction Flow", fontsize=14, fontweight="bold")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(xticks)
    ax1.set_xlim(min_round, max_round)
    ax1.set_ylim(bottom=0)

    # Plot 2: Payoffs per round — dyad means
    ax2 = axes[0, 1]
    ax2.plot(rounds, investor_payoff, label="Investor Payoff", linewidth=2,
             color="green", marker="o", markersize=6)
    ax2.plot(rounds, trustee_payoff, label="Trustee Payoff", linewidth=2,
             color="orange", marker="o", markersize=6)
    ax2.set_xlabel("Round", fontsize=12)
    ax2.set_ylabel("Payoff (dyad mean)", fontsize=12)
    ax2.set_title("Payoffs per Round", fontsize=14, fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(xticks)
    ax2.set_xlim(min_round, max_round)
    ax2.set_ylim(bottom=0)

    # Plot 3: Cumulative balances per agent, role markers as in the original
    ax3 = axes[1, 0]
    cmap = plt.cm.tab10(np.linspace(0, 1, len(agent_ids)))
    for a, color in zip(agent_ids, cmap):
        values = agent_balances[a]
        line = ax3.plot(rounds, values, label=f"{a} Balance", linewidth=2, color=color)
        line_color = line[0].get_color()
        for i, r in enumerate(game_rounds):
            role = roles_per_round[i].get(a)
            marker = "^" if role == "investor" else "o" if role == "trustee" else "x"
            ax3.scatter(rounds[i], values[i], marker=marker, color=line_color, s=50, zorder=5)
    ax3.set_xlabel("Round", fontsize=12)
    ax3.set_ylabel("Cumulative Balance", fontsize=12)
    ax3.set_title("Cumulative Balances (Agents)", fontsize=14, fontweight="bold")
    legend_line_handles, legend_line_labels = ax3.get_legend_handles_labels()
    role_marker_handles = [
        plt.Line2D([], [], marker="^", linestyle="None", color="gray", markersize=8, label="Investor (^)"),
        plt.Line2D([], [], marker="o", linestyle="None", color="gray", markersize=8, label="Trustee (o)"),
    ]
    ax3.legend(
        handles=legend_line_handles + role_marker_handles,
        labels=legend_line_labels + ["Investor (^)", "Trustee (o)"],
        loc="best",
        fontsize=8,
        ncol=2,
    )
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(xticks)
    ax3.set_xlim(min_round, max_round)
    ax3.set_ylim(bottom=0)

    # Plot 4: Cumulative balances by role (mean over that round's 4 holders)
    ax4 = axes[1, 1]
    ax4.plot(rounds, investor_role_balance, label="Investor Role Balance (mean)",
             linewidth=2, color="blue", marker="^", markersize=6)
    ax4.plot(rounds, trustee_role_balance, label="Trustee Role Balance (mean)",
             linewidth=2, color="red", marker="o", markersize=6)
    ax4.set_xlabel("Round", fontsize=12)
    ax4.set_ylabel("Cumulative Balance", fontsize=12)
    ax4.set_title("Cumulative Balances (Roles)", fontsize=14, fontweight="bold")
    ax4.legend(loc="best")
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(xticks)
    ax4.set_xlim(min_round, max_round)
    ax4.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved trajectory plot to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ANALYSIS_INPUT_FILE")
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("ANALYSIS_OUTPUT_DIR")
    title = sys.argv[3] if len(sys.argv) > 3 else "Trajectory 1 - Numerical Choices"

    if not input_file or not output_dir:
        print(__doc__)
        sys.exit(1)

    with open(input_file) as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / (Path(input_file).stem + "_trajectory.png")
    plot_numerical_trajectory_8agent(
        data["conversation_history"], save_path=str(save_path), title=title
    )


if __name__ == "__main__":
    main()
