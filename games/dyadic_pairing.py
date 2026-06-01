import random


class DyadicPairingMixin:
    """Shared pairing/name helpers for games played in two-agent dyads."""

    requires_even_agents = True

    def _init_dyadic_agents(self):
        self.agent_ids = ["Agent_1", "Agent_2"]
        self.agent_names = {agent_id: agent_id for agent_id in self.agent_ids}
        self._round_pairings = {}
        self._sync_agent_aliases()

    def configure_agents(self, agent_ids, agent_names=None):
        if len(agent_ids) < 2:
            raise ValueError("Dyadic games require at least two agents.")
        if len(agent_ids) % 2 != 0:
            raise ValueError(
                f"Dyadic games require an even number of agents; got {len(agent_ids)}."
            )

        self.agent_ids = list(agent_ids)
        self.agent_names = {
            agent_id: str((agent_names or {}).get(agent_id, agent_id))
            for agent_id in self.agent_ids
        }
        if len(set(self.agent_names.values())) != len(self.agent_names):
            raise ValueError("Agent display names must be unique.")

        self._round_pairings = {}
        self._sync_agent_aliases()

    def _sync_agent_aliases(self):
        self.agent_1_id = self.agent_ids[0] if self.agent_ids else "Agent_1"
        self.agent_2_id = self.agent_ids[1] if len(self.agent_ids) > 1 else "Agent_2"

    def get_agent_display_name(self, agent_id):
        return self.agent_names.get(agent_id, agent_id)

    def get_round_pairings(self, turn, sim_data=None):
        if turn in self._round_pairings:
            return self._round_pairings[turn]

        existing = self._get_existing_pairings(turn, sim_data)
        if existing:
            self._round_pairings[turn] = existing
            return existing

        if len(self.agent_ids) == 2:
            if turn % 2 == 1:
                raw_pairs = [(self.agent_1_id, self.agent_2_id)]
            else:
                raw_pairs = [(self.agent_2_id, self.agent_1_id)]
        else:
            investor_ids, trustee_ids = self._role_cohorts_for_turn(turn, sim_data)
            random.shuffle(investor_ids)
            random.shuffle(trustee_ids)
            raw_pairs = list(zip(investor_ids, trustee_ids))

        pairings = []
        for idx, (investor_id, trustee_id) in enumerate(raw_pairs, start=1):
            pairings.append(self._pairing_record(turn, idx, investor_id, trustee_id))

        self._round_pairings[turn] = pairings
        return pairings

    def _role_cohorts_for_turn(self, turn, sim_data=None):
        if turn <= 1:
            return self._initial_role_cohorts()

        previous_roles = self._get_previous_roles(turn, sim_data)
        if not previous_roles:
            return self._initial_role_cohorts()

        investor_ids = []
        trustee_ids = []
        for agent_id in self.agent_ids:
            previous_role = previous_roles.get(agent_id)
            if previous_role == "investor":
                trustee_ids.append(agent_id)
            elif previous_role == "trustee":
                investor_ids.append(agent_id)
            else:
                raise ValueError(
                    f"Cannot alternate roles for {agent_id}; no previous role found before round {turn}."
                )

        if len(investor_ids) != len(trustee_ids):
            raise ValueError(
                "Cannot alternate roles because previous round did not have equal "
                "numbers of senders and receivers."
            )

        return investor_ids, trustee_ids

    def _initial_role_cohorts(self):
        shuffled = list(self.agent_ids)
        random.shuffle(shuffled)
        midpoint = len(shuffled) // 2
        return shuffled[:midpoint], shuffled[midpoint:]

    def _get_previous_roles(self, turn, sim_data):
        if sim_data is None:
            return None

        for entry in reversed(sim_data.conversation_history):
            if entry.get("round", 0) >= turn:
                continue
            roles = entry.get("roles") or {}
            if all(agent_id in roles for agent_id in self.agent_ids):
                return roles

            pairings = entry.get("pairings") or entry.get("dyads") or []
            pairing_roles = {}
            for pairing in pairings:
                pairing_roles.update(pairing.get("roles") or {})
            if all(agent_id in pairing_roles for agent_id in self.agent_ids):
                return pairing_roles

        return None

    def _get_existing_pairings(self, turn, sim_data):
        if sim_data is None:
            return None
        for entry in sim_data.conversation_history:
            if entry.get("round") != turn:
                continue
            pairings = entry.get("pairings")
            if pairings:
                return [self._normalize_pairing(turn, idx, p) for idx, p in enumerate(pairings, start=1)]
            dyads = entry.get("dyads")
            if dyads:
                return [self._normalize_pairing(turn, idx, d) for idx, d in enumerate(dyads, start=1)]
            roles = entry.get("roles") or {}
            investor_id = next((aid for aid, role in roles.items() if role == "investor"), None)
            trustee_id = next((aid for aid, role in roles.items() if role == "trustee"), None)
            if investor_id and trustee_id:
                return [self._pairing_record(turn, 1, investor_id, trustee_id)]
        return None

    def _normalize_pairing(self, turn, idx, pairing):
        investor_id = pairing.get("investor")
        trustee_id = pairing.get("trustee")
        if not investor_id or not trustee_id:
            agents = pairing.get("agents") or []
            roles = pairing.get("roles") or {}
            investor_id = investor_id or next(
                (aid for aid in agents if roles.get(aid) == "investor"), None
            )
            trustee_id = trustee_id or next(
                (aid for aid in agents if roles.get(aid) == "trustee"), None
            )
        return self._pairing_record(
            turn,
            idx,
            investor_id,
            trustee_id,
            dyad_id=pairing.get("dyad_id"),
        )

    def _pairing_record(self, turn, idx, investor_id, trustee_id, dyad_id=None):
        dyad_id = dyad_id or f"dyad_{idx}"
        agents = [investor_id, trustee_id]
        return {
            "round": turn,
            "dyad_id": dyad_id,
            "agents": agents,
            "investor": investor_id,
            "trustee": trustee_id,
            "roles": {
                investor_id: "investor",
                trustee_id: "trustee",
            },
            "agent_names": {
                agent_id: self.get_agent_display_name(agent_id)
                for agent_id in agents
            },
        }

    def get_roles_for_round(self, turn, sim_data=None):
        pairings = self.get_round_pairings(turn, sim_data)
        if len(pairings) == 1:
            return {
                "investor": pairings[0]["investor"],
                "trustee": pairings[0]["trustee"],
            }
        return self.get_roles_by_agent_for_round(turn, sim_data)

    def get_roles_by_agent_for_round(self, turn, sim_data=None):
        roles = {}
        for pairing in self.get_round_pairings(turn, sim_data):
            roles.update(pairing["roles"])
        return roles

    def get_move_order(self, turn, sim_data):
        move_order = []
        for pairing in self.get_round_pairings(turn, sim_data):
            move_order.extend([pairing["investor"], pairing["trustee"]])
        return move_order

    def get_pairing_for_agent(self, agent_id, turn, sim_data=None):
        for pairing in self.get_round_pairings(turn, sim_data):
            if agent_id in pairing["agents"]:
                return pairing
        raise ValueError(f"No pairing found for {agent_id} in round {turn}.")

    def get_pairing_by_dyad_id(self, dyad_id, turn, sim_data=None):
        for pairing in self.get_round_pairings(turn, sim_data):
            if pairing["dyad_id"] == dyad_id:
                return pairing
        raise ValueError(f"No dyad {dyad_id} found in round {turn}.")

    def get_opponent_id(self, agent_id, turn, sim_data=None):
        pairing = self.get_pairing_for_agent(agent_id, turn, sim_data)
        return pairing["trustee"] if agent_id == pairing["investor"] else pairing["investor"]

    def with_prompt_context(self, prompt, agent_id, opponent_id=None):
        if len(self.agent_ids) <= 2:
            return prompt

        lines = [
            f"Your name in this experiment is {self.get_agent_display_name(agent_id)}."
        ]
        if opponent_id:
            lines.append(
                "Your opponent this round is "
                f"{self.get_agent_display_name(opponent_id)} ({opponent_id})."
            )
        return "\n".join(lines) + "\n\n" + prompt

    def with_system_context(self, prompt, agent_id):
        if len(self.agent_ids) <= 2:
            return prompt

        return (
            prompt
            + "\n\nMULTI-AGENT SETTING:\n"
            + f"- There are {len(self.agent_ids)} agents in this run.\n"
            + "- Each round, agents are randomly paired into dyads. Pairings may repeat.\n"
            + "- You play one sender-receiver game with your paired opponent each round.\n"
            + "- You alternate roles each round: if you are sender in one round, "
            + "you are receiver in the next, and vice versa.\n"
            + "- You will be told your opponent's name in each round.\n"
            + f"- Your name in this experiment is {self.get_agent_display_name(agent_id)}."
        )

    def iter_completed_dyads(self, entry):
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
                "round": entry.get("round"),
                "dyad_id": "dyad_1",
                "agents": [investor_id, trustee_id],
                "investor": investor_id,
                "trustee": trustee_id,
                "roles": roles,
                "sent": entry.get("sent"),
                "received": entry.get("received"),
                "returned": entry.get("returned"),
                "investor_payoff": entry.get("investor_payoff"),
                "trustee_payoff": entry.get("trustee_payoff"),
                "payoffs": {
                    investor_id: entry.get("investor_payoff"),
                    trustee_id: entry.get("trustee_payoff"),
                },
                "balances": entry.get("balances"),
                "actions": entry.get("actions"),
            }

    def find_last_completed_dyad_for_agent(self, agent_id, turn, sim_data):
        for entry in reversed(sim_data.conversation_history):
            if entry.get("round", 0) >= turn:
                continue
            for dyad in self.iter_completed_dyads(entry):
                if agent_id in (dyad.get("agents") or []):
                    return dyad
        return None

    def get_opponent_from_dyad(self, dyad, agent_id):
        agents = dyad.get("agents") or []
        for other_agent_id in agents:
            if other_agent_id != agent_id:
                return other_agent_id
        return None
