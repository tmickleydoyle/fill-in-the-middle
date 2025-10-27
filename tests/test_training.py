import pytest

from fim.training import MaskingStrategy


class TestMaskingStrategy:
    def test_detect_chatml_with_newlines(self):
        sample = "text <|im_start|>user\n content <|im_start|>assistant\n response"
        result = MaskingStrategy.detect(sample)
        assert result is not None
        user_pat, asst_pat, desc = result
        assert user_pat == "<|im_start|>user\n"
        assert asst_pat == "<|im_start|>assistant\n"
        assert "ChatML with newlines" in desc

    def test_detect_chatml_without_newlines(self):
        sample = "text <|im_start|>user content <|im_start|>assistant response"
        result = MaskingStrategy.detect(sample)
        assert result is not None
        user_pat, asst_pat, desc = result
        assert user_pat == "<|im_start|>user"
        assert asst_pat == "<|im_start|>assistant"

    def test_detect_harmony_format(self):
        sample = "text <|start|>user<|message|> content <|start|>assistant<|message|> response"
        result = MaskingStrategy.detect(sample)
        assert result is not None
        user_pat, asst_pat, desc = result
        assert "<|start|>user<|message|>" in user_pat
        assert "<|start|>assistant<|message|>" in asst_pat

    def test_detect_no_pattern(self):
        sample = "random text without any patterns"
        result = MaskingStrategy.detect(sample)
        assert result is None
