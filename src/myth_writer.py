# ============================================================================
# MYTH WRITING
# ============================================================================

class MythWriter:
    """Handles myth writing functionality, separate from game logic"""

    def __init__(self, myth_topic, round1_template=None, later_rounds_template=None):
        self.myth_topic = myth_topic
        self.round1_template = round1_template
        self.later_rounds_template = later_rounds_template
    
    def get_myth_prompt_round_1(self, agent_id, turn, sim_data):
        """Generate prompt for myth writing"""
        if not self.round1_template:
            raise ValueError(
                "No prompt provided. Provide prompt in config/experiments.yaml under "
                "prompt_templates, named 'myth_writing_default'"
            )
        return self.round1_template.format(myth_topic=self.myth_topic)

    def get_myth_prompt_round_later(self, agent_id, turn, sim_data):
        """Generate prompt for myth writing with the agent's previous myth"""
        if not self.later_rounds_template:
            raise ValueError(
                "No prompt provided. Provide prompt in config/experiments.yaml under "
                "prompt_templates, named 'myth_writing_later_rounds'"
            )

        # Get this agent's myth from previous round
        last_myth = ""
        other_agent_myth = ""

        for entry in sim_data.conversation_history:
            if entry["round"] == turn - 1 and "myths" in entry:
                myths = entry["myths"]
                if agent_id in myths:
                    last_myth = myths[agent_id]

                opponent_id = self._get_paired_opponent_id(entry, agent_id)
                if opponent_id and opponent_id in myths:
                    other_agent_myth = myths[opponent_id]
                else:
                    # Fallback for older myth-only state without pairing metadata.
                    for other_agent_id, myth in myths.items():
                        if other_agent_id != agent_id:
                            other_agent_myth = myth
                            break
                break

        if not last_myth:
            raise ValueError(f"NO SELF MYTH ERROR: No previous myth found for {agent_id} (you/self agent) in round {turn - 1}. Cannot generate later round prompt.")

        if not other_agent_myth:
            raise ValueError(f"OTHER AGENT MYTH ERROR: No previous myth found for {agent_id} (other agent) in round {turn - 1}. Cannot generate later round prompt.")

        game_behavior_summary = self._get_game_behavior_summary(agent_id, turn, sim_data)

        return self.later_rounds_template.format(
            last_myth=last_myth,
            other_agent_myth=other_agent_myth,
            game_behavior_summary=game_behavior_summary
        ).lstrip()

    def _get_game_behavior_summary(self, agent_id, turn, sim_data):
        """Summarize the most recent prior game round from this agent's view.

        Current-turn game data is intentionally ignored so game->myth runs do
        not add an explicit same-round outcome summary to the myth prompt.
        """
        last_game_round = None
        last_dyad = None
        for entry in reversed(sim_data.conversation_history):
            if entry.get("round", 0) >= turn:
                continue
            for dyad in self._iter_completed_dyads(entry):
                if agent_id in (dyad.get("agents") or []):
                    last_game_round = entry
                    last_dyad = dyad
                    break
            if last_dyad:
                last_game_round = entry
                break

        if not last_game_round or not last_dyad:
            return ""

        role = (last_dyad.get("roles") or {}).get(agent_id)
        balances = (
            last_dyad.get("balances_communicated")
            or last_dyad.get("balances")
            or last_game_round.get("balances_communicated")
            or last_game_round.get("balances")
            or {}
        )
        balance = balances.get(agent_id)

        if role == "investor":
            sent = last_dyad.get("sent")
            received = last_dyad.get("received")
            returned = last_dyad.get("returned_communicated", last_dyad.get("returned"))
            payoff = self._visible_investor_payoff(last_dyad, sent, returned)
            return (
                "In the most recent previous game round, you were the sender: "
                f"you sent {self._money(sent)} to the other agent, it became {self._money(received)} "
                f"for them, and you saw them return {self._money(returned)} to you. "
                f"Your visible payoff for that round was {self._money(payoff)}"
                f"{self._balance_clause(balance)}."
            )

        if role == "trustee":
            sent = last_dyad.get("sent_communicated", last_dyad.get("sent"))
            received = last_dyad.get("received_communicated", last_dyad.get("received"))
            returned = last_dyad.get("returned")
            payoff = self._visible_trustee_payoff(received, returned, last_dyad)
            return (
                "In the most recent previous game round, you were the receiver: "
                f"you saw the other agent send {self._money(sent)} to you, it became {self._money(received)}, "
                f"and you returned {self._money(returned)}. "
                f"Your visible payoff for that round was {self._money(payoff)}"
                f"{self._balance_clause(balance)}."
            )

        return ""

    def _get_paired_opponent_id(self, entry, agent_id):
        for dyad in entry.get("dyads") or entry.get("pairings") or []:
            agents = dyad.get("agents") or []
            if agent_id in agents:
                for other_agent_id in agents:
                    if other_agent_id != agent_id:
                        return other_agent_id

        roles = entry.get("roles") or {}
        if agent_id in roles:
            for other_agent_id in roles:
                if other_agent_id != agent_id:
                    return other_agent_id
        return None

    def _iter_completed_dyads(self, entry):
        dyads = entry.get("dyads") or []
        if dyads:
            for dyad in dyads:
                if dyad.get("sent") is not None and dyad.get("returned") is not None:
                    yield dyad
            return

        roles = entry.get("roles") or {}
        investor_id = next((aid for aid, role in roles.items() if role == "investor"), None)
        trustee_id = next((aid for aid, role in roles.items() if role == "trustee"), None)
        if investor_id and trustee_id and entry.get("sent") is not None and entry.get("returned") is not None:
            yield {
                "agents": [investor_id, trustee_id],
                "investor": investor_id,
                "trustee": trustee_id,
                "roles": roles,
                "sent": entry.get("sent"),
                "sent_communicated": entry.get("sent_communicated"),
                "received": entry.get("received"),
                "received_communicated": entry.get("received_communicated"),
                "returned": entry.get("returned"),
                "returned_communicated": entry.get("returned_communicated"),
                "investor_payoff": entry.get("investor_payoff"),
                "trustee_payoff": entry.get("trustee_payoff"),
                "balances": entry.get("balances"),
                "balances_communicated": entry.get("balances_communicated"),
            }

    def _visible_investor_payoff(self, entry, sent, returned):
        if sent is None or returned is None:
            return entry.get("investor_payoff")

        actual_sent = entry.get("sent")
        actual_returned = entry.get("returned")
        actual_payoff = entry.get("investor_payoff")
        if actual_sent is None or actual_returned is None or actual_payoff is None:
            return actual_payoff

        endowment = actual_payoff + actual_sent - actual_returned
        return endowment - sent + returned

    def _visible_trustee_payoff(self, received, returned, entry):
        if received is None or returned is None:
            return entry.get("trustee_payoff")
        return received - returned

    def _balance_clause(self, balance):
        if balance is None:
            return ""
        return f", and your cumulative visible balance was {self._money(balance)}"

    def _money(self, amount):
        if amount is None:
            return "$unknown"
        if isinstance(amount, int) or (isinstance(amount, float) and amount.is_integer()):
            return f"${int(amount)}"
        return f"${amount:.2f}".rstrip("0").rstrip(".")

    def process_myths(self, turn, agent_myths, sim_data):
        """Store the myths written by agents

        Args:
            turn: Current round number
            agent_myths: Dict mapping agent_id to response_data (dict with 'content', 'reasoning', 'usage')
            sim_data: Simulation data object
        """
        # Find the pre-created entry for this turn and fill in myths
        # Extract just the content from structured response data for storage
        myths_content = {}
        for agent_id, response_data in agent_myths.items():
            # Handle both old string format and new dict format for backward compatibility
            if isinstance(response_data, str):
                myths_content[agent_id] = response_data
            else:
                myths_content[agent_id] = response_data.get("content", "")

        for entry in sim_data.conversation_history:
            if entry["round"] == turn:
                entry["myths"] = myths_content
                break
