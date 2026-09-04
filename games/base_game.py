import re


class InvalidGameResponseError(ValueError):
    """Raised when an LLM game decision is not in the requested JSON shape."""


class Game:
    """Base class for all games"""
    
    def __init__(self):
        pass
    
    def get_system_prompt(self, agent_id, agent):
        """Return the system prompt for an agent (game rules/context)"""
        raise NotImplementedError
    
    def get_round_1_prompt(self, agent_id, agent):
        """Return the initial user prompt for first turn"""
        raise NotImplementedError
    
    def get_later_round_prompt(self, agent_id, turn, sim_data, last_responses):
        """Return the prompt for an agent on subsequent turns"""
        raise NotImplementedError
    
    def process_turn(self, turn, agent_responses, sim_data):
        """Process the responses from all agents and update game state"""
        raise NotImplementedError

    def validate_game_response(self, content, role):
        """Validate a trust-game decision before committing it to chat memory.

        Amount bounds remain the responsibility of the concrete game's action
        processor. This boundary only requires the explicitly requested,
        quoted JSON key so malformed or wrong-role continuations can be retried
        instead of aborting (or silently entering) a run.
        """
        key_by_role = {"investor": "send", "trustee": "return"}
        key = key_by_role.get(role)
        if key is None:
            raise InvalidGameResponseError(f"Unknown game role: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise InvalidGameResponseError("Game response is empty.")

        number = r"\$?\s*(-?\d+(?:\.\d*)?|-?\.\d+)"
        expected = re.compile(
            rf"(?:'{key}'|\"{key}\")\s*:\s*{number}",
            re.IGNORECASE,
        )
        if expected.search(content) is None:
            raise InvalidGameResponseError(
                f"Expected a quoted '{key}' JSON decision; got: {content[:200]}"
            )

    GAME_RETRY_ATTEMPTS = 2

    @staticmethod
    def get_game_retry_prompt(original_prompt, role, error, retry_number):
        """Corrective prompt after a rejected game decision.

        Mirrors the myth retry (src/myth_writer.py::get_retry_prompt): the
        substantive prompt is unchanged, a suffix names the role and the
        required key, and both the rejected and accepted attempts stay in the
        interaction audit. Added 2026-09-04 after JSON-only Claude replies
        answered with the other role's key; a plain repeat of the same prompt
        reproduced the mistake and killed every run.
        """
        key_by_role = {"investor": "send", "trustee": "return"}
        key = key_by_role.get(role, "send")
        role_word = {"investor": "SENDER", "trustee": "RECEIVER"}.get(role, str(role))
        return (
            f"{original_prompt}\n\n"
            "Your previous reply was rejected because it did not contain the "
            f"required decision key. This round you are the {role_word}, so "
            f"the JSON object must use the key '{key}': {{'{key}': <amount>}}. "
            f"(Correction attempt {retry_number} of "
            f"{Game.GAME_RETRY_ATTEMPTS}.)"
        )
    
    def print_turn_summary(self, turn, agent_responses, sim_data):
        """Print summary of what happened this turn"""
        pass
    
    def print_game_summary(self, sim_data):
        """Print final game summary"""
        pass
