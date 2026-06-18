"""
Trust Game with Environmental Noise and Asymmetric Naming Support

This module extends the standard trust game with:
1. Amount noise: Environmental perturbations on actual transferred amounts
2. Asymmetric naming: The OTHER player's role is presented with a mythological name

Noise types:
- uniform: Add random noise from uniform distribution
- probabilistic: With probability p, replace amount with random/zero/max value
- decay: Information becomes less accurate over time (future extension)

Asymmetric naming:
- YOUR role stays generic ("You are the INVESTOR")
- The OTHER player's role can be mythological ("Prometheus sent you $3...")
"""

import glob
import hashlib
import json
import os
import random
import re
from games.base_game import Game
from src.control_text_pool import get_filler_text
from src.shared_context import build_previous_round_shared_context


VALID_MYTH_INJECTION_MODES = ("partner", "own", "shuffled", "filler")


# ============================================================================
# OTHER PLAYER NAMES - Mythological names for the OTHER player
# ============================================================================

OTHER_PLAYER_NAMES = {
    "default": {"investor": "The investor", "trustee": "the trustee"},
    "prosocial": {"investor": "Prometheus", "trustee": "Orpheus"},
    "trickster": {"investor": "Coyote", "trustee": "Anansi"},
    "exchange": {"investor": "Persephone", "trustee": "Yggdrasil"},
    "neutral_myth": {"investor": "The Wanderer", "trustee": "The Guardian"},
}


# ============================================================================
# TRUST GAME WITH NOISE IMPLEMENTATION
# ============================================================================

class TrustGameNoisy(Game):
    """Trust/Investment game with noise and asymmetric naming support"""

    def __init__(self, endowment, multiplier, system_prompt_template=None, personas=None,
                 round1_investor_template=None, round1_trustee_template=None,
                 later_investor_template=None, later_trustee_template=None,
                 noise_config=None, other_player_names="default",
                 myth_injection_mode="partner", shuffled_myth_pool_path=None,
                 run_seed=None):
        """
        Initialize Trust Game with noise support.

        Args:
            endowment: Starting amount for investor each round
            multiplier: Multiplier applied to sent amount
            system_prompt_template: System prompt template string
            personas: Dict mapping agent_id to persona config
            round1_investor_template: Template for first round investor prompt
            round1_trustee_template: Template for first round trustee prompt
            later_investor_template: Template for later round investor prompt
            later_trustee_template: Template for later round trustee prompt
            noise_config: Dict with noise configuration:
                {
                    "type": "uniform" | "probabilistic" | "decay",
                    "range": float,           # for uniform: +/- range
                    "probability": float,     # for probabilistic: chance of noise
                    "replacement": str,       # for probabilistic: "random" | "zero" | "max"
                    "applies_to": str,        # "sent" | "returned" | "both"
                    "inform_agents": bool     # whether to tell agents about noise
                }
            other_player_names: Key from OTHER_PLAYER_NAMES dict for asymmetric naming
            myth_injection_mode: Which text to bind to {other_agent_last_myth_block}
                in A1-style game prompts. One of:
                  "partner" — partner's most recent myth (default A1 behaviour)
                  "own"     — agent's own most recent myth (self-anchoring control)
                  "shuffled" — partner-position myth sampled from a different
                              dyad's run on disk (cross-dyad control)
                  "filler"  — length-matched non-myth reference paragraph
                              (any-text control)
            shuffled_myth_pool_path: Glob or directory used to build the pool of
                external myth strings when myth_injection_mode == "shuffled".
            run_seed: Optional integer used as part of the deterministic
                seed_key for "shuffled" and "filler" sampling so that the same
                experiment seed selects the same control texts on re-runs.
        """
        super().__init__()
        self.endowment = endowment
        self.multiplier = multiplier
        self.system_prompt_template = system_prompt_template
        self.round1_investor_template = round1_investor_template
        self.round1_trustee_template = round1_trustee_template
        self.later_investor_template = later_investor_template
        self.later_trustee_template = later_trustee_template
        self.personas = personas or {}

        # Noise configuration
        self.noise_config = noise_config or {}

        # Asymmetric naming: get name set for the OTHER player
        if isinstance(other_player_names, str):
            self.other_player_names = OTHER_PLAYER_NAMES.get(
                other_player_names, OTHER_PLAYER_NAMES["default"]
            )
        else:
            # Allow passing a custom dict directly
            self.other_player_names = other_player_names

        # Agent IDs (fixed)
        self.agent_1_id = "Agent_1"
        self.agent_2_id = "Agent_2"

        # Myth-injection mode (default "partner" preserves the original A1
        # behaviour). Validate and stash; load shuffled pool lazily / once.
        if myth_injection_mode not in VALID_MYTH_INJECTION_MODES:
            raise ValueError(
                f"Unknown myth_injection_mode {myth_injection_mode!r}. "
                f"Must be one of {VALID_MYTH_INJECTION_MODES}."
            )
        self.myth_injection_mode = myth_injection_mode
        self.shuffled_myth_pool_path = shuffled_myth_pool_path
        self.run_seed = run_seed
        self._shuffled_myth_pool = None
        if myth_injection_mode == "shuffled":
            self._shuffled_myth_pool = self._load_shuffled_myth_pool(
                shuffled_myth_pool_path
            )

    @staticmethod
    def _load_shuffled_myth_pool(pool_path):
        """Load the cross-dyad myth pool used by mode=='shuffled'.

        pool_path may be either a directory (recursively scanned for *.json)
        or a glob expression. Only main run files are considered: checkpoint,
        results, and error files are skipped. Each non-empty per-round myth
        string is added to the pool exactly as it appears in
        conversation_history[*].myths[agent_id]. Returns the pool as a tuple
        so it is hashable and shareable across worker processes.
        """
        if not pool_path:
            raise ValueError(
                "myth_injection_mode='shuffled' requires shuffled_myth_pool_path."
            )

        candidates = []
        if os.path.isdir(pool_path):
            for root, _dirs, files in os.walk(pool_path):
                for name in files:
                    if not name.endswith(".json"):
                        continue
                    if any(skip in name for skip in (".checkpoint.", ".results.", ".error.")):
                        continue
                    candidates.append(os.path.join(root, name))
        else:
            for path in glob.glob(pool_path, recursive=True):
                if not path.endswith(".json"):
                    continue
                if any(skip in path for skip in (".checkpoint.", ".results.", ".error.")):
                    continue
                candidates.append(path)

        pool = []
        for fp in candidates:
            try:
                with open(fp, "r") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            for entry in data.get("conversation_history", []) or []:
                myths = entry.get("myths") or {}
                for myth in myths.values():
                    if isinstance(myth, str) and myth.strip():
                        pool.append(myth.strip())

        if not pool:
            raise ValueError(
                f"Shuffled myth pool at {pool_path!r} is empty. "
                "Check the path and that the runs contain non-empty myths."
            )
        return tuple(pool)

    def _make_seed_key(self, agent_id, sim_data, turn):
        """Build a deterministic seed key for sampling control texts."""
        seed_parts = [
            str(self.run_seed) if self.run_seed is not None else "no_seed",
            agent_id,
            str(turn),
        ]
        # Include the run's task order if accessible so two runs that differ
        # only in task order still pick different filler/shuffled selections.
        task_order = getattr(sim_data, "task_order", None) if sim_data else None
        if task_order:
            seed_parts.append("_".join(task_order))
        return "|".join(seed_parts)

    @staticmethod
    def _clamp_amount(amount, max_amount):
        """Clamp a transfer to the available budget/pot."""
        return round(max(0, min(float(amount), float(max_amount))), 2)

    def _apply_noise(self, decision_amount, max_amount):
        """
        Apply environmental noise to a chosen transfer amount.

        Args:
            decision_amount: The amount chosen by the agent
            max_amount: Maximum valid amount (for clamping)

        Returns:
            tuple: (perturbed_amount, raw_noise_delta)
        """
        if not self.noise_config:
            return self._clamp_amount(decision_amount, max_amount), 0.0

        noise_type = self.noise_config.get("type", "none")
        raw_noise = 0.0

        if noise_type == "uniform":
            noise_range = self.noise_config.get("range", 0)
            direction = self.noise_config.get("direction", "both")
            if direction == "negative":
                raw_noise = random.uniform(-noise_range, 0)
            elif direction == "positive":
                raw_noise = random.uniform(0, noise_range)
            else:
                raw_noise = random.uniform(-noise_range, noise_range)
            perturbed = decision_amount + raw_noise

        elif noise_type == "probabilistic":
            prob = self.noise_config.get("probability", 0)
            if random.random() < prob:
                replacement = self.noise_config.get("replacement", "random")
                if replacement == "random":
                    perturbed = random.uniform(0, max_amount)
                elif replacement == "zero":
                    perturbed = 0
                elif replacement == "max":
                    perturbed = max_amount
                else:
                    perturbed = decision_amount
                raw_noise = perturbed - decision_amount
            else:
                perturbed = decision_amount

        elif noise_type == "decay":
            # Future extension: decay based on rounds_ago
            # For now, treat as no noise
            perturbed = decision_amount

        else:
            perturbed = decision_amount

        return self._clamp_amount(perturbed, max_amount), round(raw_noise, 2)

    def _apply_action_noise(self, action_type, decision_amount, max_amount):
        """Return the actual ledger amount after any configured action noise."""
        if self._should_apply_noise_to(action_type):
            return self._apply_noise(decision_amount, max_amount)
        return self._clamp_amount(decision_amount, max_amount), 0.0

    def _should_apply_noise_to(self, action_type):
        """Check if noise should be applied to this action type."""
        if not self.noise_config:
            return False
        applies_to = self.noise_config.get("applies_to", "sent")
        if applies_to == "both":
            return True
        return applies_to == action_type

    def get_roles_for_round(self, turn):
        """Determine role assignments for this round (roles swap each round)"""
        if turn % 2 == 1:  # Odd rounds
            return {
                "investor": self.agent_1_id,
                "trustee": self.agent_2_id
            }
        else:  # Even rounds
            return {
                "investor": self.agent_2_id,
                "trustee": self.agent_1_id
            }

    def get_move_order(self, turn, sim_data):
        """Define who moves in what order this turn (investor first, then trustee)"""
        roles = self.get_roles_for_round(turn)
        return [roles["investor"], roles["trustee"]]

    def get_system_prompt(self, agent_id, agent):
        """Role-agnostic system prompt that covers both roles"""
        if not self.system_prompt_template:
            raise ValueError(
                "No prompt provided. Provide prompt in config/experiments.yaml under "
                "prompt_templates, named 'trust_game_default' (or your custom template name)"
            )

        base_prompt = self.system_prompt_template.format(
            endowment=self.endowment,
            multiplier=self.multiplier
        )

        # Add persona if specified for this agent
        if agent_id in self.personas and self.personas[agent_id].get('system_addition'):
            base_prompt += f"\n\n{self.personas[agent_id]['system_addition']}"

        # Optionally inform agents about environmental perturbations
        if self.noise_config.get("inform_agents", False):
            noise_info = """

NOTE: Transfer amounts may be perturbed by the environment after you choose them.
Your earnings are based on the amounts that actually arrive."""
            base_prompt += noise_info

        return base_prompt

    def _get_other_agent_last_myth(self, agent_id, sim_data, turn=None):
        """Return (text, block) bound into A1-style game prompts.

        Behaviour depends on ``self.myth_injection_mode``:
          - "partner"  — partner's most recent myth from this run (default).
          - "own"      — agent's own most recent myth from this run.
          - "shuffled" — a myth sampled from the cross-dyad pool.
          - "filler"   — a length-matched non-myth reference paragraph.

        The returned wrapper text ("Your partner's most recent story:") is
        identical across modes so all four conditions remain prompt-shape
        comparable. ``text`` is the raw inner string and ``block`` is the
        wrapped form (or both empty when no source myth exists yet, e.g.
        round 1 of game→myth).
        """
        if sim_data is None and self.myth_injection_mode in ("partner", "own"):
            return "", ""

        text = ""
        if self.myth_injection_mode == "partner":
            partner_id = (
                self.agent_2_id if agent_id == self.agent_1_id else self.agent_1_id
            )
            text = self._latest_myth_for(partner_id, sim_data)
        elif self.myth_injection_mode == "own":
            text = self._latest_myth_for(agent_id, sim_data)
        elif self.myth_injection_mode == "shuffled":
            text = self._sample_shuffled_myth(agent_id, sim_data, turn)
        elif self.myth_injection_mode == "filler":
            text = self._sample_filler(agent_id, sim_data, turn)

        if not text:
            return "", ""
        block = "Your partner's most recent story:\n" f'"{text}"\n\n'
        return text, block

    @staticmethod
    def _latest_myth_for(target_agent_id, sim_data):
        if sim_data is None:
            return ""
        for entry in reversed(sim_data.conversation_history):
            myths = entry.get("myths") or {}
            myth = myths.get(target_agent_id)
            if isinstance(myth, str) and myth.strip():
                return myth.strip()
        return ""

    def _sample_shuffled_myth(self, agent_id, sim_data, turn):
        if not self._shuffled_myth_pool:
            return ""
        if turn is None:
            turn = self._infer_turn(sim_data)
        seed_key = self._make_seed_key(agent_id, sim_data, turn)
        digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:8], "big") % len(self._shuffled_myth_pool)
        return self._shuffled_myth_pool[idx]

    def _sample_filler(self, agent_id, sim_data, turn):
        if turn is None:
            turn = self._infer_turn(sim_data)
        seed_key = self._make_seed_key(agent_id, sim_data, turn)
        return get_filler_text(seed_key)

    @staticmethod
    def _infer_turn(sim_data):
        """Best-effort turn inference when callers don't pass it explicitly."""
        if sim_data is None:
            return 0
        history = getattr(sim_data, "conversation_history", []) or []
        if not history:
            return 1
        return history[-1].get("round", len(history))

    def get_game_prompt_round_1(self, agent_id, agent, turn):
        """First turn"""
        roles = self.get_roles_for_round(turn)

        sim_data = getattr(self, "sim_data_ref", None)
        other_agent_last_myth, other_agent_last_myth_block = (
            self._get_other_agent_last_myth(agent_id, sim_data, turn)
        )
        shared_context_block = build_previous_round_shared_context(
            agent_id, sim_data, turn
        )

        if agent_id == roles["investor"]:
            if not self.round1_investor_template:
                raise ValueError(
                    "No prompt provided. Provide prompt in config/experiments_noisy.yaml under "
                    "prompt_templates, named 'trust_game_round1_investor'"
                )
            return self.round1_investor_template.format(
                endowment=self.endowment,
                other_agent_last_myth=other_agent_last_myth,
                other_agent_last_myth_block=other_agent_last_myth_block,
                shared_context_block=shared_context_block,
            )
        else:
            # Trustee gets prompted after investor, using the real perturbed transfer.
            sent = self.sim_data_ref.game_data.get("pending_sent")
            if sent is None:
                raise ValueError("pending_sent not found in game_data. Investor should have responded first.")

            received = sent * self.multiplier
            percentage = (sent / self.endowment * 100)

            # Get asymmetric name for the investor (the OTHER player from trustee's perspective)
            investor_display_name = self.other_player_names["investor"]

            if not self.round1_trustee_template:
                raise ValueError(
                    "No prompt provided. Provide prompt in config/experiments_noisy.yaml under "
                    "prompt_templates, named 'trust_game_round1_trustee'"
                )

            # Format with investor_name for asymmetric naming
            template = self.round1_trustee_template

            # Check if template uses {investor_name} placeholder
            if "{investor_name}" in template:
                return template.format(
                    sent=sent,
                    percentage=percentage,
                    received=received,
                    investor_name=investor_display_name,
                    other_agent_last_myth=other_agent_last_myth,
                    other_agent_last_myth_block=other_agent_last_myth_block,
                    shared_context_block=shared_context_block,
                )
            else:
                # Fallback to standard template
                return template.format(
                    sent=sent,
                    percentage=percentage,
                    received=received,
                    other_agent_last_myth=other_agent_last_myth,
                    other_agent_last_myth_block=other_agent_last_myth_block,
                    shared_context_block=shared_context_block,
                )

    def get_game_prompt_later_round(self, agent_id, turn, sim_data, last_responses):
        """Subsequent turns with noise and asymmetric naming"""
        roles = self.get_roles_for_round(turn)

        # M1 memory-transplant ablation: when the agent's memory is wiped to the
        # seed exchange each round, the round-N user prompt also must not leak
        # last-round game state. Dispatch to the round-1 template every round.
        # See docs/memory_transplant_ablation_design.md §6.
        agent = sim_data.agents.get(agent_id)
        if agent is not None and getattr(agent, "memory_mode", "normal") == "m1":
            # Keep sim_data_ref fresh so get_game_prompt_round_1 can read
            # pending_sent for the trustee path.
            self.sim_data_ref = sim_data
            return self.get_game_prompt_round_1(agent_id, agent, turn)

        # Find the last round that contains actual game data (sent is not None)
        last_round = None
        for entry in reversed(sim_data.conversation_history):
            if entry.get('sent') is not None:
                last_round = entry
                break

        if last_round is None:
            # No previous game data found, treat as first round
            return self.get_game_prompt_round_1(agent_id, None, turn)

        agent_balance = sim_data.game_data["balances"][agent_id]

        other_agent_last_myth, other_agent_last_myth_block = (
            self._get_other_agent_last_myth(agent_id, sim_data, turn)
        )
        shared_context_block = build_previous_round_shared_context(
            agent_id, sim_data, turn
        )

        # Get asymmetric names
        investor_display_name = self.other_player_names["investor"]
        trustee_display_name = self.other_player_names["trustee"]

        if agent_id == roles["investor"]:
            # Use the real ledger from the previous round.
            lr_sent = last_round['sent']
            lr_received = last_round['received']
            lr_returned = last_round['returned']
            lr_sent_pct = (lr_sent / self.endowment * 100)
            lr_trustee_payoff = last_round['trustee_payoff']

            if not self.later_investor_template:
                raise ValueError(
                    "No prompt provided. Provide prompt in config/experiments_noisy.yaml under "
                    "prompt_templates, named 'trust_game_later_investor'"
                )
            return self.later_investor_template.format(
                turn=turn,
                last_round_sent=lr_sent,
                last_round_sent_percentage=lr_sent_pct,
                last_round_received=lr_received,
                last_round_returned=lr_returned,
                last_round_trustee_payoff=lr_trustee_payoff,
                agent_balance=agent_balance,
                endowment=self.endowment,
                investor_name=investor_display_name,
                trustee_name=trustee_display_name,
                other_agent_last_myth=other_agent_last_myth,
                other_agent_last_myth_block=other_agent_last_myth_block,
                shared_context_block=shared_context_block,
            )
        else:
            # Current round: pending_sent is the real perturbed transfer.
            sent = sim_data.game_data["pending_sent"]
            received = sent * self.multiplier

            # Use the real ledger from the previous round.
            lr_sent = last_round['sent']
            lr_received = last_round['received']
            lr_returned = last_round['returned']
            lr_sent_pct = (lr_sent / self.endowment * 100)
            lr_investor_payoff = last_round['investor_payoff']

            if not self.later_trustee_template:
                raise ValueError(
                    "No prompt provided. Provide prompt in config/experiments_noisy.yaml under "
                    "prompt_templates, named 'trust_game_later_trustee'"
                )
            return self.later_trustee_template.format(
                turn=turn,
                last_round_sent=lr_sent,
                last_round_sent_percentage=lr_sent_pct,
                last_round_received=lr_received,
                last_round_returned=lr_returned,
                last_round_investor_payoff=lr_investor_payoff,
                agent_balance=agent_balance,
                received=received,
                investor_name=investor_display_name,
                trustee_name=trustee_display_name,
                other_agent_last_myth=other_agent_last_myth,
                other_agent_last_myth_block=other_agent_last_myth_block,
                shared_context_block=shared_context_block,
            )

    def process_intermediate_response(self, agent_id, response, turn, sim_data):
        """Called after investor responds, before trustee"""
        roles = self.get_roles_for_round(turn)
        if agent_id == roles["investor"]:
            sent_decision = self._extract_amount(response, "send")
            sent, sent_noise_draw = self._apply_action_noise("sent", sent_decision, self.endowment)
            sim_data.game_data["pending_sent_decision"] = sent_decision
            sim_data.game_data["pending_sent"] = sent
            sim_data.game_data["pending_sent_noise_draw"] = sent_noise_draw
            # Store reference for get_round_1_prompt to use
            self.sim_data_ref = sim_data

    def process_turn(self, turn, agent_responses, sim_data):
        """Process complete turn (both moves done) with noise tracking"""
        # Initialize on first turn (don't overwrite pending values set during prompt generation)
        if "balances" not in sim_data.game_data:
            sim_data.game_data["balances"] = {self.agent_1_id: 0, self.agent_2_id: 0}
            sim_data.game_data.setdefault("pending_sent", 0)
            sim_data.game_data.setdefault("pending_sent_decision", 0)
            sim_data.game_data.setdefault("pending_sent_noise_draw", 0)
            self.sim_data_ref = sim_data

        # Get roles for this round
        roles = self.get_roles_for_round(turn)
        investor_id = roles["investor"]
        trustee_id = roles["trustee"]

        # Extract decisions and use the real perturbed ledger amounts.
        sent_decision = sim_data.game_data.get(
            "pending_sent_decision",
            self._extract_amount(agent_responses[investor_id], "send")
        )
        sent = sim_data.game_data.get("pending_sent")
        if sent is None:
            sent, sent_noise_draw = self._apply_action_noise("sent", sent_decision, self.endowment)
        else:
            sent_noise_draw = sim_data.game_data.get("pending_sent_noise_draw", round(sent - sent_decision, 2))

        received = sent * self.multiplier

        try:
            returned_decision = self._extract_amount(agent_responses[trustee_id], "return")
        except ValueError:
            if received == 0:
                returned_decision = 0.0
            else:
                raise
        returned, returned_noise_draw = self._apply_action_noise("returned", returned_decision, received)

        # Calculate payoffs using the real perturbed amounts.
        investor_payoff = (self.endowment - sent) + returned
        trustee_payoff = received - returned

        # Update balances with actual payoffs
        sim_data.game_data["balances"][investor_id] += investor_payoff
        sim_data.game_data["balances"][trustee_id] += trustee_payoff

        # Track noise statistics
        if "noise_stats" not in sim_data.game_data:
            sim_data.game_data["noise_stats"] = {
                "rounds_with_noise": 0,
                "total_sent_noise": 0,
                "total_returned_noise": 0,
                "total_sent_noise_draw": 0,
                "total_returned_noise_draw": 0,
            }

        sent_noise = round(sent - sent_decision, 2)
        returned_noise = round(returned - returned_decision, 2)
        noise_applied = (sent_noise != 0 or returned_noise != 0)
        if noise_applied:
            sim_data.game_data["noise_stats"]["rounds_with_noise"] += 1
            sim_data.game_data["noise_stats"]["total_sent_noise"] += abs(sent_noise)
            sim_data.game_data["noise_stats"]["total_returned_noise"] += abs(returned_noise)
            sim_data.game_data["noise_stats"]["total_sent_noise_draw"] += abs(sent_noise_draw)
            sim_data.game_data["noise_stats"]["total_returned_noise_draw"] += abs(returned_noise_draw)

        # Fill in the pre-created entry for this round with game data
        for entry in sim_data.conversation_history:
            if entry["round"] == turn:
                # Store real ledger amounts plus unperturbed decisions.
                entry["sent"] = sent
                entry["sent_decision"] = sent_decision
                entry["sent_noise"] = sent_noise
                entry["sent_noise_draw"] = sent_noise_draw
                entry["received"] = received
                entry["returned"] = returned
                entry["returned_decision"] = returned_decision
                entry["returned_noise"] = returned_noise
                entry["returned_noise_draw"] = returned_noise_draw
                entry["investor_payoff"] = investor_payoff
                entry["trustee_payoff"] = trustee_payoff
                entry["balances"] = dict(sim_data.game_data["balances"])
                entry["noise_applied"] = noise_applied
                entry["other_player_names"] = self.other_player_names
                entry["actions"] = {
                    investor_id: {
                        "action": "sent",
                        "decision": sent_decision,
                        "amount": sent,
                        "noise": sent_noise,
                        "noise_draw": sent_noise_draw,
                    },
                    trustee_id: {
                        "action": "returned",
                        "decision": returned_decision,
                        "amount": returned,
                        "noise": returned_noise,
                        "noise_draw": returned_noise_draw,
                    }
                }
                break

        return {
            investor_id: {"sent": sent, "sent_decision": sent_decision},
            trustee_id: {"returned": returned, "returned_decision": returned_decision}
        }

    def _extract_amount(self, response_data, key):
        """Extract number from JSON response in structured response data"""
        # Handle both old string format and new dict format for backward compatibility
        if isinstance(response_data, str):
            content = response_data
        else:
            content = response_data.get("content", "")

        # Parse JSON format: {'send': 5} or {"send": 5}
        pattern = rf"'{key}':\s*(\d+\.?\d*)|" + rf'"{key}":\s*(\d+\.?\d*)'
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return float(match.group(1) or match.group(2))
        raise ValueError(f"Could not extract {key} from: {content[:200]}")

    def print_turn_summary(self, turn, agent_responses, sim_data):
        """Print round summary with noise information"""
        entry = sim_data.conversation_history[-1]
        roles = entry["roles"]

        # Determine which agent had which role
        investor_id = [aid for aid, role in roles.items() if role == "investor"][0]
        trustee_id = [aid for aid, role in roles.items() if role == "trustee"][0]

        print(f"\n{'*' * 80}")
        print(f"ROUND {turn} COMPLETE")
        print(f"  Roles: {investor_id} = Investor, {trustee_id} = Trustee")
        print(f"  Sent: ${entry['sent']} → Received: ${entry['received']} → Returned: ${entry['returned']}")

        # Show noise if applied
        if entry.get('noise_applied'):
            print(f"  [NOISE] Decisions: sent=${entry['sent_decision']:.2f}, "
                  f"returned=${entry['returned_decision']:.2f}")
            print(f"  [NOISE] Applied: sent={entry['sent_noise']:+.2f}, "
                  f"returned={entry['returned_noise']:+.2f}")

        # Show asymmetric naming if non-default
        if self.other_player_names != OTHER_PLAYER_NAMES["default"]:
            print(f"  [FRAMING] Other player names: {self.other_player_names}")

        print(f"  Payoffs: {investor_id} ${entry['investor_payoff']}, {trustee_id} ${entry['trustee_payoff']}")
        print(f"  Cumulative: {self.agent_1_id} ${entry['balances'][self.agent_1_id]}, "
              f"{self.agent_2_id} ${entry['balances'][self.agent_2_id]}")
        print(f"{'*' * 80}")

    def print_game_summary(self, sim_data):
        """Final summary with noise statistics"""
        # Calculate total simulation rounds from the highest round number
        total_rounds = max((entry.get("round", 0) for entry in sim_data.conversation_history), default=0)

        print("\n" + "=" * 80)
        print("GAME SUMMARY (NOISY)")
        print("=" * 80)

        # Filter for entries that contain actual game data (sent is not None)
        game_rounds = [r for r in sim_data.conversation_history if r.get("sent") is not None]

        total_sent = sum(r["sent"] for r in game_rounds)
        total_returned = sum(r["returned"] for r in game_rounds)
        actual_game_rounds = len(game_rounds)
        avg_sent = total_sent / actual_game_rounds if actual_game_rounds > 0 else 0
        avg_returned = total_returned / actual_game_rounds if actual_game_rounds > 0 else 0

        print(f"\nTotal rounds: {total_rounds}")
        print(f"Game rounds played: {actual_game_rounds}")
        print(f"Avg sent: ${avg_sent:.2f}/{self.endowment}")
        print(f"Avg returned: ${avg_returned:.2f}")
        print(f"Final earnings: {self.agent_1_id} ${sim_data.game_data['balances'][self.agent_1_id]}, "
              f"{self.agent_2_id} ${sim_data.game_data['balances'][self.agent_2_id]}")

        # Noise statistics
        noise_stats = sim_data.game_data.get("noise_stats", {})
        if noise_stats.get("rounds_with_noise", 0) > 0:
            print(f"\n--- Noise Statistics ---")
            print(f"  Rounds with noise: {noise_stats['rounds_with_noise']}/{actual_game_rounds}")
            print(f"  Total sent noise: ${noise_stats['total_sent_noise']:.2f}")
            print(f"  Total returned noise: ${noise_stats['total_returned_noise']:.2f}")

        # Asymmetric naming info
        if self.other_player_names != OTHER_PLAYER_NAMES["default"]:
            print(f"\n--- Asymmetric Naming ---")
            print(f"  Investor shown as: {self.other_player_names['investor']}")
            print(f"  Trustee shown as: {self.other_player_names['trustee']}")

        # Show role distribution
        agent_1_investor_rounds = sum(1 for r in game_rounds if "roles" in r and r["roles"][self.agent_1_id] == "investor")
        agent_2_investor_rounds = sum(1 for r in game_rounds if "roles" in r and r["roles"][self.agent_2_id] == "investor")
        print(f"\nRole distribution:")
        print(f"  {self.agent_1_id}: {agent_1_investor_rounds} rounds as investor, "
              f"{actual_game_rounds - agent_1_investor_rounds} rounds as trustee")
        print(f"  {self.agent_2_id}: {agent_2_investor_rounds} rounds as investor, "
              f"{actual_game_rounds - agent_2_investor_rounds} rounds as trustee")
