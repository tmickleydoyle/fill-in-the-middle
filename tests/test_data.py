from unittest.mock import Mock

import pytest

from fim import DataConfig, DatasetFormatter


class TestDatasetFormatter:
    @pytest.fixture
    def formatter(self, data_config: DataConfig) -> DatasetFormatter:
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template = Mock(return_value="formatted_text")
        return DatasetFormatter(mock_tokenizer, data_config)

    def test_format_valid_example(self, formatter: DatasetFormatter, sample_fim_example: dict):
        result = formatter.format_example(sample_fim_example)
        assert result == "formatted_text"
        formatter.tokenizer.apply_chat_template.assert_called_once()

    def test_format_empty_middle_returns_empty(self, formatter: DatasetFormatter):
        example = {"lang": "Python", "prefix": "test", "suffix": "", "middle": ""}
        result = formatter.format_example(example)
        assert result == ""

    def test_format_no_context_returns_empty(self, formatter: DatasetFormatter):
        example = {"lang": "Python", "prefix": "", "suffix": "", "middle": "test"}
        result = formatter.format_example(example)
        assert result == ""

    def test_context_section_building(self, formatter: DatasetFormatter):
        context_items = [
            {"file_path": "utils.py", "content": "def helper(): pass"},
            {"file_path": "main.py", "content": "import utils"},
        ]
        result = formatter._build_context_section(context_items, "Python")
        assert "utils.py" in result
        assert "main.py" in result

    def test_batch_formatting(self, formatter: DatasetFormatter, sample_fim_example: dict):
        examples = {key: [value] for key, value in sample_fim_example.items()}
        result = formatter(examples)
        assert "text" in result
        assert len(result["text"]) == 1
