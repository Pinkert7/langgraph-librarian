"""Unit tests for langgraph_librarian.utils."""

from langgraph_librarian.utils import (
    SelectionResult,
    estimate_tokens,
    extract_message_text,
    parse_selection_response,
)


class TestEstimateTokens:
    def test_basic(self):
        assert estimate_tokens("hello world") == 2  # 11 chars / 4 = 2

    def test_empty(self):
        assert estimate_tokens("") == 1  # min 1

    def test_long_text(self):
        text = "a" * 400
        assert estimate_tokens(text) == 100


class TestParseSelectionResponse:
    def test_structured_response(self):
        response = (
            "Reasoning: Messages 0 and 2 are about the login feature.\n"
            "IDs: [0, 2]"
        )
        result = parse_selection_response(response)
        assert result.ids == [0, 2]
        assert "login" in result.reasoning

    def test_ids_only(self):
        response = "IDs: [1, 3, 5]"
        result = parse_selection_response(response)
        assert result.ids == [1, 3, 5]

    def test_fallback_array(self):
        response = "I think messages [0, 4, 7] are relevant."
        result = parse_selection_response(response)
        assert result.ids == [0, 4, 7]

    def test_malformed_response(self):
        response = "I don't know which messages to select."
        result = parse_selection_response(response)
        assert result.ids == []

    def test_negative_ids_filtered(self):
        response = "IDs: [-1, 0, 2, -3]"
        result = parse_selection_response(response)
        assert result.ids == [0, 2]

    def test_mixed_types(self):
        response = 'IDs: [0, "1", 2]'
        result = parse_selection_response(response)
        assert result.ids == [0, 1, 2]

    def test_empty_array(self):
        response = "IDs: []"
        result = parse_selection_response(response)
        assert result.ids == []


class TestExtractMessageText:
    def test_string_content(self):
        class MockMsg:
            content = "hello world"

        assert extract_message_text(MockMsg()) == "hello world"

    def test_list_content(self):
        class MockMsg:
            content = [{"text": "hello"}, {"text": "world"}]

        assert extract_message_text(MockMsg()) == "hello\nworld"

    def test_mixed_list_content(self):
        class MockMsg:
            content = ["hello", {"text": "world"}]

        assert extract_message_text(MockMsg()) == "hello\nworld"
