import contextlib
import io
import threading
import unittest
from unittest.mock import patch

from games.trust_game import TrustGame
from src.myth_writer import (
    InvalidMythResponseError,
    MythWriter,
    validate_myth_response,
)
from src.simulation import run_simulation


CONTAMINATED_MYTH = """Myth: A complete story about trust and reciprocity.

Round 3

Your total visible earnings across all rounds are $7.97.

This round, you are the RECEIVER. The sender sent $3.00. You received $9.00.
How much do you return to the sender? (0-9.00)

Take any myths written in this session into account when making your decision.
"""


def build_game():
    return TrustGame(
        endowment=5,
        multiplier=3,
        system_prompt_template="Game rules",
        round1_investor_template="Send an amount.",
        round1_trustee_template="Return an amount.",
        later_investor_template="Send an amount.",
        later_trustee_template="Return an amount.",
    )


class MythResponseValidatorTests(unittest.TestCase):
    def test_rejects_observed_prompt_shaped_continuation(self):
        with self.assertRaisesRegex(
            InvalidMythResponseError,
            "continue into a game prompt",
        ):
            validate_myth_response(CONTAMINATED_MYTH)

    def test_allows_game_themed_story_language(self):
        validate_myth_response(
            "Myth: The receiver asked how much generosity should be returned, "
            "and the sender answered with trust rather than calculation."
        )

    def test_simulation_retries_without_keeping_rejected_response_in_memory(self):
        myth_writer = MythWriter(
            myth_topic="anything",
            round1_template="Write a myth. {topic_instruction}",
            later_rounds_template="Rewrite your myth.",
        )
        calls_by_conversation = {}
        calls_lock = threading.Lock()

        def fake_call_llm(client, model, temperature, messages):
            conversation_id = id(messages)
            with calls_lock:
                attempt = calls_by_conversation.get(conversation_id, 0) + 1
                calls_by_conversation[conversation_id] = attempt
            content = (
                CONTAMINATED_MYTH
                if attempt == 1
                else "Myth: A valid replacement story about reciprocal trust."
            )
            return {"content": content, "reasoning": None, "usage": None}

        with patch("src.simulation.create_llm_client", return_value=object()), patch(
            "src.agents.call_llm",
            side_effect=fake_call_llm,
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                sim_data = run_simulation(
                    game=build_game(),
                    model="mock/model",
                    temperature=0,
                    num_turns=1,
                    num_agents=2,
                    memory_capacity=6,
                    agent_biases="",
                    myth_writer=myth_writer,
                    task_order=["myth"],
                    chat_memory_mode="memory_primary",
                )

        for agent_id, agent in sim_data.agents.items():
            self.assertEqual(len(agent.interaction_history), 2)
            rejected, accepted = agent.interaction_history
            self.assertEqual(
                rejected["error"]["type"],
                "InvalidMythResponseError",
            )
            self.assertIn("Round 3", rejected["response"]["content"])
            self.assertNotIn(
                CONTAMINATED_MYTH,
                [message.get("content") for message in agent.messages],
            )
            self.assertNotIn("error", accepted)
            self.assertEqual(
                sim_data.conversation_history[0]["myths"][agent_id],
                "Myth: A valid replacement story about reciprocal trust.",
            )


if __name__ == "__main__":
    unittest.main()
