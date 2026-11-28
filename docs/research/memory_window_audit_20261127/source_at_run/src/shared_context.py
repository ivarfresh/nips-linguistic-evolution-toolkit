"""Shared public context blocks for cross-task prompt interventions."""


def _format_amount(value):
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if value.is_integer():
        return f"${int(value)}"
    return f"${value:.2f}"


def _string_or_empty(value):
    if value is None:
        return ""
    return str(value).strip()


def build_previous_round_shared_context(agent_id, sim_data, turn):
    """Build an agent-relative public context block from the prior round only.

    The block intentionally uses ledger outcomes, not hidden pre-noise decisions
    or the other agent's private reasoning.
    """
    if sim_data is None or turn <= 1:
        return ""

    previous = None
    for entry in reversed(sim_data.conversation_history):
        if entry.get("round") == turn - 1:
            previous = entry
            break
    if not previous:
        return ""

    roles = previous.get("roles") or {}
    partner_id = next((candidate for candidate in roles if candidate != agent_id), "")
    myths = previous.get("myths") or {}
    own_myth = _string_or_empty(myths.get(agent_id))
    partner_myth = _string_or_empty(myths.get(partner_id))

    has_game = previous.get("sent") is not None
    has_myth = bool(own_myth or partner_myth)
    if not has_game and not has_myth:
        return ""

    lines = ["Shared context from the previous round:"]

    if has_game:
        investor_id = next(
            (candidate for candidate, role in roles.items() if role == "investor"),
            "the investor",
        )
        trustee_id = next(
            (candidate for candidate, role in roles.items() if role == "trustee"),
            "the trustee",
        )
        lines.append(
            "- Game outcome: "
            f"{investor_id} was investor and {trustee_id} was trustee; "
            f"the investor sent {_format_amount(previous.get('sent'))}, "
            f"the trustee received {_format_amount(previous.get('received'))}, "
            f"and the trustee returned {_format_amount(previous.get('returned'))}."
        )

        agent_role = roles.get(agent_id)
        if agent_role == "investor":
            payoff = previous.get("investor_payoff")
        elif agent_role == "trustee":
            payoff = previous.get("trustee_payoff")
        else:
            payoff = None
        if agent_role and payoff is not None:
            lines.append(
                f"- Your previous game role/payoff: {agent_role}, "
                f"payoff {_format_amount(payoff)}."
            )

        balances = previous.get("balances") or {}
        if agent_id in balances:
            lines.append(
                "- Your cumulative earnings after that round: "
                f"{_format_amount(balances[agent_id])}."
            )

    if own_myth:
        lines.append(f'- Your previous myth:\n"{own_myth}"')
    if partner_myth:
        lines.append(f'- Your partner\'s previous myth:\n"{partner_myth}"')

    return "\n".join(lines) + "\n\n"
