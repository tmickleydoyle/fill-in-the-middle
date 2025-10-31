from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from fim import CodeCompleter, InferenceConfig


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    return model


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    tokenizer.decode.return_value = "self.name = name"
    return tokenizer


@pytest.fixture
def completer(mock_model, mock_tokenizer, inference_config):
    return CodeCompleter(mock_model, mock_tokenizer, inference_config)


class TestCodeCompleter:
    def test_format_output_markdown_enabled(self, completer: CodeCompleter):
        result = completer._format_output("self.name = name", "Python")
        assert result == "```Python\nself.name = name\n```"

    def test_format_output_markdown_disabled(self, completer: CodeCompleter):
        completer.config.format_output = False
        result = completer._format_output("self.name = name", "Python")
        assert result == "self.name = name"

    def test_format_output_plain_format(self, completer: CodeCompleter):
        completer.config.output_format = "plain"
        result = completer._format_output("self.name = name", "Python")
        assert result == "self.name = name"

    def test_format_output_strips_whitespace(self, completer: CodeCompleter):
        result = completer._format_output("  self.name = name  \n", "Python")
        assert result == "```Python\nself.name = name\n```"

    def test_format_output_empty_text(self, completer: CodeCompleter):
        result = completer._format_output("", "Python")
        assert result == ""

    def test_format_output_already_has_markdown(self, completer: CodeCompleter):
        input_text = "```Python\nself.name = name\n```"
        result = completer._format_output(input_text, "Python")
        assert result == input_text

    def test_complete_non_streaming(self, completer: CodeCompleter):
        result = completer.complete(
            prefix="class User:\n    def __init__(self, name):\n        ",
            suffix="\n",
            language="Python",
            stream=False,
        )
        assert result == "```Python\nself.name = name\n```"
        completer.model.generate.assert_called_once()

    def test_complete_non_streaming_plain_format(self, completer: CodeCompleter):
        completer.config.format_output = False
        result = completer.complete(
            prefix="class User:\n    def __init__(self, name):\n        ",
            suffix="\n",
            language="Python",
            stream=False,
        )
        assert result == "self.name = name"

    def test_build_messages_without_context(self, completer: CodeCompleter):
        messages = completer._build_messages(
            prefix="def foo():",
            suffix="",
            language="Python",
            context=None,
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "code completion assistant" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "<FILL_HERE>" in messages[1]["content"]
        assert "```Python" in messages[1]["content"]

    def test_build_messages_with_context(self, completer: CodeCompleter):
        context = ["def helper():\n    return 42"]
        messages = completer._build_messages(
            prefix="def foo():",
            suffix="",
            language="Python",
            context=context,
        )
        assert len(messages) == 2
        assert "Relevant code context" in messages[1]["content"]
        assert "def helper()" in messages[1]["content"]

    def test_complete_with_context(self, completer: CodeCompleter):
        context = ["def helper():\n    return 42"]
        result = completer.complete(
            prefix="def foo():",
            suffix="",
            language="Python",
            context=context,
            stream=False,
        )
        assert result == "```Python\nself.name = name\n```"
        completer.model.generate.assert_called_once()

    @patch("fim.inference.MarkdownTextStreamer")
    def test_complete_streaming_markdown(self, mock_streamer_class, completer: CodeCompleter):
        result = completer.complete(
            prefix="def foo():",
            suffix="",
            language="Python",
            stream=True,
        )
        assert result == ""
        mock_streamer_class.assert_called_once()
        call_kwargs = mock_streamer_class.call_args
        assert call_kwargs[0][0] == completer.tokenizer
        assert call_kwargs[0][1] == "Python"

    @patch("fim.inference.TextStreamer")
    def test_complete_streaming_plain(self, mock_streamer_class, completer: CodeCompleter):
        completer.config.format_output = False
        result = completer.complete(
            prefix="def foo():",
            suffix="",
            language="Python",
            stream=True,
        )
        assert result == ""
        mock_streamer_class.assert_called_once()
