# ============================================================================
# MYTH WRITING
# ============================================================================

import re

from src.shared_context import build_previous_round_shared_context


class InvalidMythResponseError(ValueError):
    """Raised when a myth response crosses the myth/game task boundary."""


_GAME_PROMPT_MARKERS = {
    "round header": re.compile(r"(?im)^\s*Round\s+\d+\s*$"),
    "visible earnings": re.compile(
        r"(?im)^\s*Your total visible earnings across all rounds are\s+\$"
    ),
    "assigned game role": re.compile(
        r"(?im)^\s*This round,\s+you are the\s+(?:SENDER|RECEIVER)\b"
    ),
    "game decision question": re.compile(
        r"(?i)\bHow much do you (?:send|return)(?:\s+to the sender)?\?"
    ),
    "JSON decision instruction": re.compile(
        r"(?im)^\s*Respond exactly as JSON:\s*"
    ),
    "myth decision instruction": re.compile(
        r"(?im)^\s*Take any myths written in this session into account "
        r"when making your decision\.\s*$"
    ),
}

_GAME_DECISION_ONLY = re.compile(
    r"""
    ^\s*
    (?:```(?:json)?\s*)?
    (?:
        \{\s*[\"']?(?:send|return)[\"']?\s*:\s*
        \$?\s*-?(?:\d+(?:\.\d*)?|\.\d+)\s*\}
        |
        \$?\s*-?(?:\d+(?:\.\d*)?|\.\d+)
    )
    \s*(?:```)?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def validate_myth_response(content):
    """Reject likely prompt-shaped continuations appended to a generated myth.

    A myth may naturally mention rounds, senders, receivers, or decisions. To
    avoid rejecting ordinary story content, require at least two independent
    prompt markers. The observed spillover contained five markers.
    """
    if not isinstance(content, str) or not content.strip():
        raise InvalidMythResponseError("Myth response is empty.")

    if _GAME_DECISION_ONLY.fullmatch(content):
        raise InvalidMythResponseError(
            "Myth response is a game decision rather than a story."
        )

    matched_markers = [
        label
        for label, pattern in _GAME_PROMPT_MARKERS.items()
        if pattern.search(content)
    ]
    if len(matched_markers) >= 2:
        markers = ", ".join(matched_markers)
        raise InvalidMythResponseError(
            "Myth response appears to continue into a game prompt "
            f"(matched markers: {markers})."
        )


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
        shared_context_block = build_previous_round_shared_context(
            agent_id, sim_data, turn
        )
        return self.round1_template.format(
            myth_topic=self.myth_topic,
            shared_context_block=shared_context_block,
            topic_instruction=self._get_topic_instruction(),
        )

    def _get_topic_instruction(self):
        """Render the myth topic without making the default "anything" literal."""
        topic = (self.myth_topic or "").strip()
        if not topic or topic.lower() == "anything":
            return "You may choose any mythic setting, characters, or symbols."
        return f"Use this topic: {topic}."

    @staticmethod
    def validate_response(content):
        """Validate a raw LLM response before it is stored as a myth."""
        validate_myth_response(content)

    @staticmethod
    def get_retry_prompt(original_prompt, error, retry_number):
        """Clarify task boundaries after a rejected myth response.

        The original substantive prompt remains intact. This suffix says only
        how to satisfy the requested output boundary, and every rejected and
        accepted attempt remains explicit in the interaction audit.
        """
        return (
            f"{original_prompt}\n\n"
            "Your previous response was rejected because it did not contain "
            "only the requested myth. Complete the original myth-writing task "
            "now. Write the story in prose and start it with 'Myth:'. Do not "
            "answer with a send/return decision, JSON, or any part of a game "
            f"prompt. (Correction attempt {retry_number} of 2.)"
        )

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

        previous_entry = None
        original_author_id = None
        presented_author_id = None
        substitution_applied = False

        for entry in sim_data.conversation_history:
            if entry["round"] == turn - 1 and "myths" in entry:
                previous_entry = entry
                myths = entry["myths"]
                if agent_id in myths:
                    last_myth = myths[agent_id]

                opponent_id = self._get_paired_opponent_id(entry, agent_id)
                if opponent_id and opponent_id in myths:
                    original_author_id = opponent_id
                    presented_author_id = opponent_id
                    other_agent_myth = myths[opponent_id]
                else:
                    # Fallback for older myth-only state without pairing metadata.
                    for other_agent_id, myth in myths.items():
                        if other_agent_id != agent_id:
                            original_author_id = other_agent_id
                            presented_author_id = other_agent_id
                            other_agent_myth = myth
                            break

                (
                    other_agent_myth,
                    presented_author_id,
                    substitution_applied,
                ) = self._apply_defector_myth_policy(
                    agent_id=agent_id,
                    turn=turn,
                    sim_data=sim_data,
                    previous_entry=entry,
                    original_author_id=original_author_id,
                    original_myth=other_agent_myth,
                )
                break

        if not last_myth:
            raise ValueError(f"NO SELF MYTH ERROR: No previous myth found for {agent_id} (you/self agent) in round {turn - 1}. Cannot generate later round prompt.")

        if not other_agent_myth:
            raise ValueError(f"OTHER AGENT MYTH ERROR: No previous myth found for {agent_id} (other agent) in round {turn - 1}. Cannot generate later round prompt.")

        self._record_myth_exposure(
            sim_data=sim_data,
            turn=turn,
            agent_id=agent_id,
            previous_entry=previous_entry,
            original_author_id=original_author_id,
            presented_author_id=presented_author_id,
            substitution_applied=substitution_applied,
        )

        game_behavior_summary = self._get_game_behavior_summary(agent_id, turn, sim_data)

        return self.later_rounds_template.format(
            myth_topic=self.myth_topic,
            last_myth=last_myth,
            other_agent_myth=other_agent_myth,
            shared_context_block=build_previous_round_shared_context(
                agent_id, sim_data, turn
            ),
            game_behavior_summary=game_behavior_summary,
        ).lstrip()

    def _apply_defector_myth_policy(
        self,
        *,
        agent_id,
        turn,
        sim_data,
        previous_entry,
        original_author_id,
        original_myth,
    ):
        """Resolve the prior myth shown under the configured circulation arm.

        ``standard_substitute`` affects only transmission from a defector
        author to an ordinary target. It substitutes a real ordinary-authored
        myth from the same prior population-round, without disclosing the
        author or policy in the prompt. Defectors still generate and retain
        their own myths normally.
        """
        metadata = getattr(sim_data, "run_metadata", {}) or {}
        game_data = getattr(sim_data, "game_data", {}) or {}
        policy = metadata.get(
            "defector_myth_policy",
            game_data.get("defector_myth_policy", "normal"),
        )
        agent_types = (
            (previous_entry or {}).get("agent_types")
            or game_data.get("agent_types")
            or {}
        )
        target_type = agent_types.get(agent_id, "standard")
        original_type = agent_types.get(original_author_id, "standard")
        if not (
            policy == "standard_substitute"
            and target_type == "standard"
            and original_type == "defector"
        ):
            return original_myth, original_author_id, False

        myths = (previous_entry or {}).get("myths") or {}
        candidates = sorted(
            candidate_id
            for candidate_id, myth in myths.items()
            if candidate_id != agent_id
            and candidate_id != original_author_id
            and myth
            and agent_types.get(candidate_id, "standard") == "standard"
        )
        if not candidates:
            raise ValueError(
                "DEFECTOR MYTH SUBSTITUTION ERROR: no ordinary-authored myth "
                f"is available for {agent_id} in round {turn - 1}."
            )

        ordered_agent_ids = sorted(myths)
        target_index = ordered_agent_ids.index(agent_id)
        candidate_id = candidates[(turn + target_index) % len(candidates)]
        return myths[candidate_id], candidate_id, True

    @staticmethod
    def _record_myth_exposure(
        *,
        sim_data,
        turn,
        agent_id,
        previous_entry,
        original_author_id,
        presented_author_id,
        substitution_applied,
    ):
        metadata = getattr(sim_data, "run_metadata", {}) or {}
        game_data = getattr(sim_data, "game_data", {}) or {}
        agent_types = (
            (previous_entry or {}).get("agent_types")
            or game_data.get("agent_types")
            or {}
        )
        policy = metadata.get(
            "defector_myth_policy",
            game_data.get("defector_myth_policy", "normal"),
        )
        record = {
            "policy": policy,
            "source_round": turn - 1,
            "original_author_id": original_author_id,
            "original_author_type": agent_types.get(
                original_author_id,
                "standard",
            ),
            "presented_author_id": presented_author_id,
            "presented_author_type": agent_types.get(
                presented_author_id,
                "standard",
            ),
            "substitution_applied": bool(substitution_applied),
        }
        for entry in sim_data.conversation_history:
            if entry.get("round") == turn:
                entry.setdefault("myth_exposures", {})[agent_id] = record
                return
        raise ValueError(
            f"MYTH EXPOSURE AUDIT ERROR: current round {turn} is unavailable."
        )

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
                content = response_data
            else:
                content = response_data.get("content", "")
            # Defense in depth: the normal simulation path validates before
            # memory insertion and retries, while this prevents any alternate
            # caller from storing a contaminated myth directly.
            validate_myth_response(content)
            myths_content[agent_id] = content

        for entry in sim_data.conversation_history:
            if entry["round"] == turn:
                entry["myths"] = myths_content
                break
