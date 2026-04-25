"""Shared helpers used across analysis scripts.

Keep this module minimal — only helpers duplicated in 2+ places belong here.
"""

import json
from typing import Dict, Optional

import numpy as np


def configure_matplotlib() -> None:
    """Set the non-interactive Agg backend.

    Must be called before any `import matplotlib.pyplot`. Safe to call more
    than once.
    """
    import matplotlib
    matplotlib.use("Agg")


def load_simulation_data(filepath: str) -> Dict:
    """Load a simulation state JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_game_metrics(data: Dict, endowment: Optional[float] = None) -> Optional[Dict]:
    """Extract cooperation metrics from a simulation JSON file.

    Args:
        data: parsed simulation JSON (the dict returned by `load_simulation_data`).
        endowment: initial per-round endowment. If None, it is derived from the
            first game round as `sent[0] + investor_payoff[0]`.

    Returns:
        Metrics dict, or None if the simulation has no valid game rounds.
    """
    history = data.get("conversation_history", [])

    game_rounds = [
        r for r in history
        if r.get("sent") is not None and r.get("returned") is not None
    ]

    if not game_rounds:
        return None

    sent = np.array([r["sent"] for r in game_rounds])
    received = np.array([r["received"] for r in game_rounds])
    returned = np.array([r["returned"] for r in game_rounds])
    investor_payoff = np.array([r["investor_payoff"] for r in game_rounds])
    trustee_payoff = np.array([r["trustee_payoff"] for r in game_rounds])

    agent_1_balances = []
    agent_2_balances = []
    for r in game_rounds:
        balances = r.get("balances", {})
        agent_1_balances.append(balances.get("Agent_1", 0))
        agent_2_balances.append(balances.get("Agent_2", 0))

    if endowment is None:
        endowment = float(sent[0] + investor_payoff[0]) if len(sent) > 0 else 10.0

    trust_ratios = sent / endowment if endowment > 0 else sent * 0
    return_ratios = np.where(received > 0, returned / received, 0)

    return {
        "num_rounds": len(game_rounds),
        "mean_sent": float(np.mean(sent)),
        "std_sent": float(np.std(sent)),
        "mean_returned": float(np.mean(returned)),
        "std_returned": float(np.std(returned)),
        "mean_trust_ratio": float(np.mean(trust_ratios)),
        "std_trust_ratio": float(np.std(trust_ratios)),
        "mean_return_ratio": float(np.mean(return_ratios)),
        "std_return_ratio": float(np.std(return_ratios)),
        "mean_investor_payoff": float(np.mean(investor_payoff)),
        "mean_trustee_payoff": float(np.mean(trustee_payoff)),
        "final_investor_payoff": float(investor_payoff[-1]),
        "final_trustee_payoff": float(trustee_payoff[-1]),
        "cooperation_stability": float(np.std(return_ratios)),
        "agent_1_balances": agent_1_balances,
        "agent_2_balances": agent_2_balances,
    }
