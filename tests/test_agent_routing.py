import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.agent_hub import _select_agents, aggregate_answer, web_agent


class AgentRoutingTests(unittest.TestCase):
    def test_general_movie_question_uses_graph_and_web(self):
        self.assertEqual(
            _select_agents("Who directed The Matrix?", has_upload=False),
            ["neo4j", "web"],
        )

    def test_time_sensitive_question_uses_web(self):
        self.assertEqual(
            _select_agents(
                "What is the latest box office news for Gladiator 2?",
                has_upload=False,
            ),
            ["web"],
        )

    def test_script_question_uses_uploaded_document(self):
        self.assertEqual(
            _select_agents(
                "Summarize this script.",
                has_upload=True,
            ),
            ["rag"],
        )

    @patch("core.agent_hub._build_llm")
    @patch("core.agent_hub._invoke_with_retry")
    def test_aggregate_answer_returns_plain_text(self, invoke, build_llm):
        invoke.return_value = SimpleNamespace(
            content=[
                {
                    "type": "text",
                    "text": "The Matrix was directed by the Wachowskis.",
                    "extras": {"signature": "provider-only metadata"},
                }
            ]
        )

        result = aggregate_answer(
            {
                "question": "Who directed The Matrix?",
                "selected_agents": ["neo4j", "web"],
                "web_context": "Source: https://example.com",
            }
        )

        self.assertEqual(
            result["answer"],
            "The Matrix was directed by the Wachowskis.",
        )
        self.assertIsNone(result["error"])
        build_llm.assert_called_once()

    @patch("core.agent_hub._search_with_bing", return_value=["Search result"])
    @patch("core.agent_hub.config.TAVILY_API_KEY", "")
    def test_successful_fallback_search_is_not_reported_as_warning(
        self,
        search_with_bing,
    ):
        result = web_agent(
            {
                "question": "Who directed The Matrix?",
                "selected_agents": ["web"],
            }
        )

        self.assertEqual(result["web_context"], "Search result")
        self.assertEqual(result["web_error"], "")
        search_with_bing.assert_called_once()


if __name__ == "__main__":
    unittest.main()
