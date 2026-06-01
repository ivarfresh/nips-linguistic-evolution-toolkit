import random
import re

from games.base_game import Game
from games.dyadic_pairing import DyadicPairingMixin


OTHER_PLAYER_NAMES = {
    "default": {"investor": "The sender", "trustee": "the receiver"},
    "prosocial": {"investor": "Prometheus", "trustee": "Orpheus"},
    "trickster": {"investor": "Coyote", "trustee": "Anansi"},
    "exchange": {"investor": "Persephone", "trustee": "Yggdrasil"},
    "neutral_myth": {"investor": "The Wanderer", "trustee": "The Guardian"},
}


class TrustGameNoisy(DyadicPairingMixin, Game):
    """Trust/Investment game with amount noise and optional asymmetric naming."""

    def __init__(
        self,
        endowment,
        multiplier,
        system_prompt_template=None,
        personas=None,
        round1_investor_template=None,
        round1_trustee_template=None,
        later_investor_template=None,
        later_trustee_template=None,
        noise_config=None,
        other_player_names="default",
    ):
        super().__init__()
        self.endowment = endowment
        self.multiplier = multiplier
        self.system_prompt_template = system_prompt_template
        self.round1_investor_template = round1_investor_template
        self.round1_trustee_template = round1_trustee_template
        self.later_investor_template = later_investor_template
        self.later_trustee_template = later_trustee_template
        self.personas = personas or {}
        self.noise_config = noise_config or {}

        if isinstance(other_player_names, str):
            self.other_player_names = OTHER_PLAYER_NAMES.get(
                other_player_names, OTHER_PLAYER_NAMES["default"]
            )
        else:
            self.other_player_names = other_player_names

        self._init_dyadic_agents()

    def _apply_noise(self, actual_amount, max_amount):
        if not self.noise_config:
            return actual_amount, actual_amount

        noise_type = self.noise_config.get("type", "none")
        if noise_type == "uniform":
            noise_range = self.noise_config.get("range", 0)
            direction = self.noise_config.get("direction", "both")
            if direction == "negative":
                noise = random.uniform(-noise_range, 0)
            elif direction == "positive":
                noise = random.uniform(0, noise_range)
            else:
                noise = random.uniform(-noise_range, noise_range)
            communicated = actual_amount + noise
            communicated = max(0, min(communicated, max_amount))
        elif noise_type == "probabilistic":
            prob = self.noise_config.get("probability", 0)
            if random.random() < prob:
                replacement = self.noise_config.get("replacement", "random")
                if replacement == "random":
                    communicated = random.uniform(0, max_amount)
                elif replacement == "zero":
                    communicated = 0
                elif replacement == "max":
                    communicated = max_amount
                else:
                    communicated = actual_amount
            else:
                communicated = actual_amount
        else:
            communicated = actual_amount

        return round(communicated, 2), actual_amount

    def _should_apply_noise_to(self, action_type):
        applies_to = self.noise_config.get("applies_to", "sent")
        return applies_to == "both" or applies_to == action_type

    def _role_display_name(self, pairing, role):
        agent_id = pairing[role]
        if len(self.agent_ids) > 2:
            return self.get_agent_display_name(agent_id)
        return self.other_player_names[role]

    def get_system_prompt(self, agent_id, agent):
        if not self.system_prompt_template:
            raise ValueError(
                "No prompt provided. Provide prompt in config/experiments_noisy.yaml under "
                "prompt_templates, named 'trust_game_default' (or your custom template name)"
            )

        base_prompt = self.system_prompt_template.format(
            endowment=self.endowment,
            multiplier=self.multiplier,
        )

        if agent_id in self.personas and self.personas[agent_id].get("system_addition"):
            base_prompt += f"\n\n{self.personas[agent_id]['system_addition']}"

        if self.noise_config.get("inform_agents", False):
            base_prompt += (
                "\n\nNOTE: There may be communication noise in this game.\n"
                "The amounts you see may differ slightly from what was actually sent/returned.\n"
                "This is part of the experimental design."
            )

        return self.with_system_context(base_prompt, agent_id)

    def get_game_prompt_round_1(self, agent_id, agent, turn):
        pairing = self.get_pairing_for_agent(agent_id, turn)
        opponent_id = self.get_opponent_id(agent_id, turn)

        if agent_id == pairing["investor"]:
            if not self.round1_investor_template:
                raise ValueError(
                    "No prompt provided. Provide prompt in config/experiments_noisy.yaml under "
                    "prompt_templates, named 'trust_game_round1_investor'"
                )
            prompt = self.round1_investor_template.format(
                endowment=self.endowment,
                agent_name=self.get_agent_display_name(agent_id),
                opponent_name=self.get_agent_display_name(opponent_id),
                investor_name=self._role_display_name(pairing, "investor"),
                trustee_name=self._role_display_name(pairing, "trustee"),
            )
            return self.with_prompt_context(prompt, agent_id, opponent_id)

        sent_actual = self.sim_data_ref.game_data.get("pending_sents", {}).get(pairing["dyad_id"])
        if sent_actual is None:
            raise ValueError("pending_sent not found in game_data. Investor should have responded first.")

        sent_communicated = self._communicated_sent_for_prompt(
            sim_data=self.sim_data_ref,
            pairing=pairing,
            sent_actual=sent_actual,
        )
        received = sent_communicated * self.multiplier
        percentage = sent_communicated / self.endowment * 100

        if not self.round1_trustee_template:
            raise ValueError(
                "No prompt provided. Provide prompt in config/experiments_noisy.yaml under "
                "prompt_templates, named 'trust_game_round1_trustee'"
            )
        prompt = self.round1_trustee_template.format(
            sent=sent_communicated,
            percentage=percentage,
            received=received,
            agent_name=self.get_agent_display_name(agent_id),
            opponent_name=self.get_agent_display_name(opponent_id),
            investor_name=self._role_display_name(pairing, "investor"),
            trustee_name=self._role_display_name(pairing, "trustee"),
        )
        return self.with_prompt_context(prompt, agent_id, opponent_id)

    def get_game_prompt_later_round(self, agent_id, turn, sim_data, last_responses):
        if len(self.agent_ids) > 2:
            return self._get_multi_agent_later_prompt(agent_id, turn, sim_data)

        pairing = self.get_pairing_for_agent(agent_id, turn, sim_data)
        opponent_id = self.get_opponent_id(agent_id, turn, sim_data)
        roles = self.get_roles_for_round(turn, sim_data)
        last_round = self.find_last_completed_dyad_for_agent(agent_id, turn, sim_data)
        if last_round is None:
            return self.get_game_prompt_round_1(agent_id, None, turn)

        sent_actual = sim_data.game_data["pending_sents"].get(pairing["dyad_id"])
        if sent_actual is None:
            sent_actual = sim_data.game_data.get("pending_sent", 0)
        sent_communicated = self._communicated_sent_for_prompt(sim_data, pairing, sent_actual)
        received = sent_communicated * self.multiplier

        balances = sim_data.game_data.get("balances_communicated") or sim_data.game_data["balances"]
        agent_balance = balances[agent_id]

        if agent_id == roles["investor"]:
            lr_sent = last_round.get("sent_communicated", last_round["sent"])
            lr_received = lr_sent * self.multiplier
            lr_returned = last_round["returned"]
            lr_sent_pct = lr_sent / self.endowment * 100
            lr_trustee_payoff = lr_received - lr_returned

            if not self.later_investor_template:
                raise ValueError(
                    "No prompt provided. Provide prompt in config/experiments_noisy.yaml under "
                    "prompt_templates, named 'trust_game_later_investor'"
                )
            prompt = self.later_investor_template.format(
                turn=turn,
                last_round_sent=lr_sent,
                last_round_sent_percentage=lr_sent_pct,
                last_round_received=lr_received,
                last_round_returned=lr_returned,
                last_round_trustee_payoff=lr_trustee_payoff,
                agent_balance=agent_balance,
                endowment=self.endowment,
                agent_name=self.get_agent_display_name(agent_id),
                opponent_name=self.get_agent_display_name(opponent_id),
                investor_name=self._role_display_name(pairing, "investor"),
                trustee_name=self._role_display_name(pairing, "trustee"),
            )
            return self.with_prompt_context(prompt, agent_id, opponent_id)

        lr_sent = last_round["sent"]
        lr_received = last_round["sent"] * self.multiplier
        lr_returned = last_round.get("returned_communicated", last_round["returned"])
        lr_sent_pct = lr_sent / self.endowment * 100
        lr_investor_payoff = self.endowment - lr_sent + lr_returned

        if not self.later_trustee_template:
            raise ValueError(
                "No prompt provided. Provide prompt in config/experiments_noisy.yaml under "
                "prompt_templates, named 'trust_game_later_trustee'"
            )
        prompt = self.later_trustee_template.format(
            turn=turn,
            last_round_sent=lr_sent,
            last_round_sent_percentage=lr_sent_pct,
            last_round_received=lr_received,
            last_round_returned=lr_returned,
            last_round_investor_payoff=lr_investor_payoff,
            agent_balance=agent_balance,
            current_round_sent=sent_communicated,
            current_round_sent_percentage=sent_communicated / self.endowment * 100,
            received=received,
            agent_name=self.get_agent_display_name(agent_id),
            opponent_name=self.get_agent_display_name(opponent_id),
            investor_name=self._role_display_name(pairing, "investor"),
            trustee_name=self._role_display_name(pairing, "trustee"),
        )
        return self.with_prompt_context(prompt, agent_id, opponent_id)

    def _get_multi_agent_later_prompt(self, agent_id, turn, sim_data):
        pairing = self.get_pairing_for_agent(agent_id, turn, sim_data)
        opponent_id = self.get_opponent_id(agent_id, turn, sim_data)
        role = pairing["roles"][agent_id]
        balances = sim_data.game_data.get("balances_communicated") or sim_data.game_data["balances"]
        agent_balance = balances[agent_id]
        last_dyad = self.find_last_completed_dyad_for_agent(agent_id, turn, sim_data)

        history = "No previous game round involving you has been completed."
        if last_dyad:
            last_opponent_id = self.get_opponent_from_dyad(last_dyad, agent_id)
            last_opponent_name = self.get_agent_display_name(last_opponent_id)
            last_role = last_dyad["roles"].get(agent_id)
            if last_role == "investor":
                lr_returned = last_dyad.get("returned_communicated", last_dyad["returned"])
                lr_payoff = self.endowment - last_dyad["sent"] + lr_returned
                history = (
                    f"In your most recent previous game round, you were the SENDER "
                    f"against {last_opponent_name}. You sent ${last_dyad['sent']}, "
                    f"it became ${last_dyad['received']}, and you saw them return "
                    f"${lr_returned} to you. Your visible payoff was ${lr_payoff}."
                )
            elif last_role == "trustee":
                lr_sent = last_dyad.get("sent_communicated", last_dyad["sent"])
                lr_received = last_dyad.get("received_communicated", lr_sent * self.multiplier)
                lr_payoff = lr_received - last_dyad["returned"]
                history = (
                    f"In your most recent previous game round, you were the RECEIVER "
                    f"against {last_opponent_name}. You saw them send ${lr_sent}, "
                    f"it became ${lr_received}, and you returned ${last_dyad['returned']}. "
                    f"Your visible payoff was ${lr_payoff}."
                )

        if role == "investor":
            prompt = (
                f"Round {turn}\n\n"
                f"{history}\n"
                f"Your total visible earnings across all rounds are ${agent_balance}.\n\n"
                f"This round, you are the SENDER against {self.get_agent_display_name(opponent_id)}. "
                f"You have ${self.endowment}. How much do you send? (0-{self.endowment})"
            )
        else:
            sent_actual = sim_data.game_data["pending_sents"].get(pairing["dyad_id"])
            if sent_actual is None:
                raise ValueError("pending_sent not found in game_data. Investor should have responded first.")
            sent_communicated = self._communicated_sent_for_prompt(sim_data, pairing, sent_actual)
            received = sent_communicated * self.multiplier
            prompt = (
                f"Round {turn}\n\n"
                f"{history}\n"
                f"Your total visible earnings across all rounds are ${agent_balance}.\n\n"
                f"This round, you are the RECEIVER against {self.get_agent_display_name(opponent_id)}. "
                f"They sent you ${sent_communicated}, so you received ${received}. "
                f"How much do you return? (0-{received})"
            )

        return self.with_prompt_context(prompt, agent_id, opponent_id)

    def _communicated_sent_for_prompt(self, sim_data, pairing, sent_actual):
        dyad_id = pairing["dyad_id"]
        pending = sim_data.game_data.setdefault("pending_sents_communicated", {})
        if dyad_id in pending:
            return pending[dyad_id]
        if self._should_apply_noise_to("sent"):
            sent_communicated, _ = self._apply_noise(sent_actual, self.endowment)
        else:
            sent_communicated = sent_actual
        pending[dyad_id] = sent_communicated
        sim_data.game_data["pending_sent_communicated"] = sent_communicated
        return sent_communicated

    def process_intermediate_response(self, agent_id, response, turn, sim_data):
        pairing = self.get_pairing_for_agent(agent_id, turn, sim_data)
        if agent_id != pairing["investor"]:
            return

        sent_amount = self._extract_amount(response, "send")
        sim_data.game_data.setdefault("pending_sents", {})[pairing["dyad_id"]] = sent_amount
        sim_data.game_data["pending_sent"] = sent_amount
        self.sim_data_ref = sim_data

    def process_turn(self, turn, agent_responses, sim_data):
        self._ensure_balances(sim_data)
        pairings = self.get_round_pairings(turn, sim_data)

        dyads = []
        round_actions = {}
        round_payoffs = {}
        last_responses = {}

        for pairing in pairings:
            investor_id = pairing["investor"]
            trustee_id = pairing["trustee"]
            sent_actual = self._extract_amount(agent_responses[investor_id], "send")
            returned_actual = self._extract_amount(agent_responses[trustee_id], "return")

            sent_communicated = sim_data.game_data.get("pending_sents_communicated", {}).get(
                pairing["dyad_id"],
                sent_actual,
            )
            received_actual = sent_actual * self.multiplier
            received_communicated = sent_communicated * self.multiplier

            if self._should_apply_noise_to("returned"):
                returned_communicated, _ = self._apply_noise(returned_actual, received_actual)
            else:
                returned_communicated = returned_actual

            investor_payoff = (self.endowment - sent_actual) + returned_actual
            trustee_payoff = received_actual - returned_actual
            investor_payoff_communicated = (self.endowment - sent_actual) + returned_communicated
            trustee_payoff_communicated = received_communicated - returned_actual

            sim_data.game_data["balances"][investor_id] += investor_payoff
            sim_data.game_data["balances"][trustee_id] += trustee_payoff
            sim_data.game_data["balances_communicated"][investor_id] += investor_payoff_communicated
            sim_data.game_data["balances_communicated"][trustee_id] += trustee_payoff_communicated

            noise_applied = (
                sent_communicated != sent_actual
                or returned_communicated != returned_actual
            )
            if noise_applied:
                stats = sim_data.game_data["noise_stats"]
                stats["rounds_with_noise"] += 1
                stats["total_sent_noise"] += abs(sent_communicated - sent_actual)
                stats["total_returned_noise"] += abs(returned_communicated - returned_actual)

            dyad = {
                **pairing,
                "sent": sent_actual,
                "sent_communicated": sent_communicated,
                "received": received_actual,
                "received_communicated": received_communicated,
                "returned": returned_actual,
                "returned_communicated": returned_communicated,
                "investor_payoff": investor_payoff,
                "trustee_payoff": trustee_payoff,
                "investor_payoff_communicated": investor_payoff_communicated,
                "trustee_payoff_communicated": trustee_payoff_communicated,
                "payoffs": {
                    investor_id: investor_payoff,
                    trustee_id: trustee_payoff,
                },
                "payoffs_communicated": {
                    investor_id: investor_payoff_communicated,
                    trustee_id: trustee_payoff_communicated,
                },
                "balances": dict(sim_data.game_data["balances"]),
                "balances_communicated": dict(sim_data.game_data["balances_communicated"]),
                "noise_applied": noise_applied,
                "other_player_names": self.other_player_names,
                "actions": {
                    investor_id: {
                        "action": "sent",
                        "amount": sent_actual,
                        "communicated": sent_communicated,
                    },
                    trustee_id: {
                        "action": "returned",
                        "amount": returned_actual,
                        "communicated": returned_communicated,
                    },
                },
            }
            dyads.append(dyad)
            round_actions.update(dyad["actions"])
            round_payoffs.update(dyad["payoffs"])
            last_responses[investor_id] = {
                "sent": sent_actual,
                "sent_communicated": sent_communicated,
            }
            last_responses[trustee_id] = {
                "returned": returned_actual,
                "returned_communicated": returned_communicated,
            }

        sim_data.game_data["pending_sents"] = {}
        sim_data.game_data["pending_sents_communicated"] = {}
        sim_data.game_data["pending_sent"] = 0
        sim_data.game_data["pending_sent_communicated"] = 0
        self._fill_round_entry(turn, sim_data, dyads, round_actions, round_payoffs)
        return last_responses

    def _ensure_balances(self, sim_data):
        if "balances" not in sim_data.game_data:
            sim_data.game_data["balances"] = {agent_id: 0 for agent_id in self.agent_ids}
        else:
            for agent_id in self.agent_ids:
                sim_data.game_data["balances"].setdefault(agent_id, 0)

        if "balances_communicated" not in sim_data.game_data:
            sim_data.game_data["balances_communicated"] = {
                agent_id: 0 for agent_id in self.agent_ids
            }
        else:
            for agent_id in self.agent_ids:
                sim_data.game_data["balances_communicated"].setdefault(agent_id, 0)

        sim_data.game_data.setdefault("pending_sents", {})
        sim_data.game_data.setdefault("pending_sents_communicated", {})
        sim_data.game_data.setdefault("pending_sent", 0)
        sim_data.game_data.setdefault("pending_sent_communicated", 0)
        sim_data.game_data.setdefault(
            "noise_stats",
            {
                "rounds_with_noise": 0,
                "total_sent_noise": 0,
                "total_returned_noise": 0,
            },
        )
        self.sim_data_ref = sim_data

    def _fill_round_entry(self, turn, sim_data, dyads, round_actions, round_payoffs):
        for entry in sim_data.conversation_history:
            if entry.get("round") != turn:
                continue

            entry["dyads"] = dyads
            entry["pairings"] = [
                {
                    "round": dyad["round"],
                    "dyad_id": dyad["dyad_id"],
                    "agents": dyad["agents"],
                    "investor": dyad["investor"],
                    "trustee": dyad["trustee"],
                    "roles": dyad["roles"],
                    "agent_names": dyad["agent_names"],
                }
                for dyad in dyads
            ]
            entry["roles"] = self.get_roles_by_agent_for_round(turn, sim_data)
            entry["payoffs"] = round_payoffs
            entry["balances"] = dict(sim_data.game_data["balances"])
            entry["balances_communicated"] = dict(sim_data.game_data["balances_communicated"])
            entry["actions"] = round_actions
            entry["other_player_names"] = self.other_player_names

            if len(dyads) == 1:
                dyad = dyads[0]
                for key in [
                    "sent",
                    "sent_communicated",
                    "received",
                    "received_communicated",
                    "returned",
                    "returned_communicated",
                    "investor_payoff",
                    "trustee_payoff",
                    "investor_payoff_communicated",
                    "trustee_payoff_communicated",
                    "noise_applied",
                ]:
                    entry[key] = dyad[key]
            else:
                for key in [
                    "sent",
                    "sent_communicated",
                    "received",
                    "received_communicated",
                    "returned",
                    "returned_communicated",
                    "investor_payoff",
                    "trustee_payoff",
                    "investor_payoff_communicated",
                    "trustee_payoff_communicated",
                    "noise_applied",
                ]:
                    entry[key] = None
            break

    def _extract_amount(self, response_data, key):
        if isinstance(response_data, str):
            content = response_data
        else:
            content = response_data.get("content", "")

        pattern = rf"'{key}':\s*(\d+\.?\d*)|" + rf'"{key}":\s*(\d+\.?\d*)'
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return float(match.group(1) or match.group(2))
        raise ValueError(f"Could not extract {key} from: {content[:200]}")

    def print_turn_summary(self, turn, agent_responses, sim_data):
        entry = sim_data.conversation_history[-1]
        dyads = list(self.iter_completed_dyads(entry))

        print(f"\n{'*' * 80}")
        print(f"ROUND {turn} COMPLETE")
        for dyad in dyads:
            investor_id = dyad["investor"]
            trustee_id = dyad["trustee"]
            print(
                f"  {dyad['dyad_id']}: "
                f"{investor_id} ({self.get_agent_display_name(investor_id)}) = Sender, "
                f"{trustee_id} ({self.get_agent_display_name(trustee_id)}) = Receiver"
            )
            print(
                f"    Sent: ${dyad['sent']} -> Received: ${dyad['received']} "
                f"-> Returned: ${dyad['returned']}"
            )
            if dyad.get("noise_applied"):
                print(
                    f"    [NOISE] Communicated sent=${dyad['sent_communicated']:.2f}, "
                    f"returned=${dyad['returned_communicated']:.2f}"
                )
            print(
                f"    Payoffs: {investor_id} ${dyad['investor_payoff']}, "
                f"{trustee_id} ${dyad['trustee_payoff']}"
            )

        balances = entry.get("balances") or {}
        cumulative = ", ".join(
            f"{agent_id} ${balances.get(agent_id, 0)}" for agent_id in self.agent_ids
        )
        print(f"  Cumulative: {cumulative}")
        print(f"{'*' * 80}")

    def print_game_summary(self, sim_data):
        total_rounds = max(
            (entry.get("round", 0) for entry in sim_data.conversation_history),
            default=0,
        )
        game_dyads = [
            dyad
            for entry in sim_data.conversation_history
            for dyad in self.iter_completed_dyads(entry)
        ]
        actual_game_dyads = len(game_dyads)
        total_sent = sum(dyad["sent"] for dyad in game_dyads)
        total_returned = sum(dyad["returned"] for dyad in game_dyads)
        avg_sent = total_sent / actual_game_dyads if actual_game_dyads > 0 else 0
        avg_returned = total_returned / actual_game_dyads if actual_game_dyads > 0 else 0

        print("\n" + "=" * 80)
        print("GAME SUMMARY (NOISY)")
        print("=" * 80)
        print(f"\nTotal rounds: {total_rounds}")
        print(f"Dyadic games played: {actual_game_dyads}")
        print(f"Avg sent: ${avg_sent:.2f}/{self.endowment}")
        print(f"Avg returned: ${avg_returned:.2f}")

        balances = sim_data.game_data.get("balances", {})
        final_earnings = ", ".join(
            f"{agent_id} ${balances.get(agent_id, 0)}" for agent_id in self.agent_ids
        )
        print(f"Final earnings: {final_earnings}")

        noise_stats = sim_data.game_data.get("noise_stats", {})
        if noise_stats.get("rounds_with_noise", 0) > 0:
            print("\n--- Noise Statistics ---")
            print(f"  Dyads with noise: {noise_stats['rounds_with_noise']}/{actual_game_dyads}")
            print(f"  Total sent noise: ${noise_stats['total_sent_noise']:.2f}")
            print(f"  Total returned noise: ${noise_stats['total_returned_noise']:.2f}")

        print("\nRole distribution:")
        for agent_id in self.agent_ids:
            investor_rounds = sum(
                1 for dyad in game_dyads if dyad.get("roles", {}).get(agent_id) == "investor"
            )
            trustee_rounds = sum(
                1 for dyad in game_dyads if dyad.get("roles", {}).get(agent_id) == "trustee"
            )
            print(
                f"  {agent_id}: {investor_rounds} rounds as sender, "
                f"{trustee_rounds} rounds as receiver"
            )
