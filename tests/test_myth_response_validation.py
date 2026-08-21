import contextlib
import io
import threading
import unittest
from unittest.mock import patch

from games.trust_game import TrustGame
from scripts.audit_v2_protocol import classify_interactions
from src.agents import Agent
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

    def test_rejects_game_decision_instead_of_a_myth(self):
        for response in (
            '{"send": 5}',
            "{'return': 7.0}",
            "```json\n{\"send\": 3}\n```",
            "4.5",
        ):
            with self.subTest(response=response):
                with self.assertRaisesRegex(
                    InvalidMythResponseError,
                    "game decision rather than a story",
                ):
                    validate_myth_response(response)

    def test_allows_game_themed_story_language(self):
        validate_myth_response(
            "Myth: The receiver asked how much generosity should be returned, "
            "and the sender answered with trust rather than calculation."
        )

    def test_provider_failure_rolls_back_unanswered_prompt(self):
        agent = Agent(
            agent_id="Agent_1",
            model="mock/model",
            temperature=0,
            client=object(),
            memory_capacity=3,
            initial_bias=None,
        )
        agent.messages.append({"role": "system", "content": "Game rules"})

        with patch(
            "src.agents.call_llm",
            side_effect=RuntimeError("temporary provider failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "provider failure"):
                agent.respond("Write a myth.")

        self.assertEqual(
            agent.messages,
            [{"role": "system", "content": "Game rules"}],
        )
        self.assertEqual(
            agent.interaction_history[0]["error"]["type"],
            "RuntimeError",
        )

    def test_audit_classifies_a_rejected_attempt_as_a_recovered_retry(self):
        metadata = {"round": 3, "task": "myth"}
        rejected = {
            "metadata": metadata,
            "error": {"type": "InvalidMythResponseError"},
        }
        accepted = {"metadata": metadata, "response": {"content": "Myth: ok"}}

        accepted_events, retries, unrecovered = classify_interactions(
            [rejected, accepted]
        )

        self.assertEqual(accepted_events, [accepted])
        self.assertEqual(retries, [rejected])
        self.assertEqual(unrecovered, [])

    def test_audit_flags_a_retry_without_a_later_accepted_response(self):
        rejected = {
            "metadata": {"round": 3, "task": "myth"},
            "error": {"type": "InvalidMythResponseError"},
        }

        _, retries, unrecovered = classify_interactions([rejected])

        self.assertEqual(retries, [rejected])
        self.assertEqual(unrecovered, [rejected])

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
