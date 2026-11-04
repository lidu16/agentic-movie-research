import unittest

from core.llm_utils import normalize_llm_text


class NormalizeLlmTextTests(unittest.TestCase):
    def test_extracts_text_without_provider_metadata(self):
        response = [
            {
                "type": "text",
                "text": "The Matrix was directed by the Wachowskis.",
                "extras": {"signature": "provider-only metadata"},
            }
        ]

        self.assertEqual(
            normalize_llm_text(response),
            "The Matrix was directed by the Wachowskis.",
        )

    def test_joins_text_blocks_and_ignores_non_text_blocks(self):
        response = [
            {"type": "text", "text": "First sentence."},
            {"type": "image", "data": "not display text"},
            {"type": "text", "text": "Second sentence."},
        ]

        self.assertEqual(
            normalize_llm_text(response),
            "First sentence.\nSecond sentence.",
        )

    def test_preserves_plain_text_and_handles_empty_content(self):
        self.assertEqual(normalize_llm_text("Answer"), "Answer")
        self.assertEqual(normalize_llm_text(None), "")
        self.assertEqual(normalize_llm_text([]), "")


if __name__ == "__main__":
    unittest.main()
