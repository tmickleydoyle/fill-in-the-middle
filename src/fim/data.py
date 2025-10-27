import logging
from typing import Any

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

from .config import DataConfig

logger = logging.getLogger("fim")


class DatasetFormatter:
    def __init__(self, tokenizer: PreTrainedTokenizer, config: DataConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.skipped_count = 0

    def format_example(self, example: dict[str, Any]) -> str:
        lang = example.get("lang", "Unknown")
        prefix = example.get("prefix", "")
        suffix = example.get("suffix", "")
        middle = example.get("middle", "")
        context_items = example.get("context_items", [])

        if not middle or not middle.strip():
            return ""

        if not prefix.strip() and not suffix.strip():
            return ""

        context_section = self._build_context_section(context_items, lang)
        user_content = (
            f"{context_section}### Complete the following {lang} code:\n\n"
            f"```{lang}\n{prefix}<FILL_HERE>{suffix}\n```"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an expert code completion assistant. "
                    f"Complete the code at the <FILL_HERE> marker with syntactically correct "
                    f"{lang} code that fits naturally with the surrounding context."
                )
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": middle}
        ]

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception as e:
            logger.warning(f"Failed to format example: {e}")
            return ""

    def _build_context_section(self, context_items: list[dict], lang: str) -> str:
        if not context_items:
            return ""

        sections = ["### Relevant code context from repository:\n"]
        for item in context_items[:self.config.max_context_items]:
            if not item or "content" not in item:
                continue

            file_path = item.get("file_path", "unknown")
            content = item["content"]
            sections.append(f"```{lang}\n// From: {file_path}\n{content}\n```\n")

        return "\n".join(sections) if len(sections) > 1 else ""

    def __call__(self, examples: dict[str, list[Any]]) -> dict[str, list[str]]:
        texts = []
        for i in range(len(examples["prefix"])):
            example = {key: examples[key][i] for key in examples}
            formatted = self.format_example(example)

            if formatted:
                texts.append(formatted)
            else:
                texts.append("")
                self.skipped_count += 1

        if self.skipped_count > 0 and self.skipped_count % 100 == 0:
            logger.info(f"Skipped {self.skipped_count} invalid examples")

        return {"text": texts}


def load_and_format_dataset(
    tokenizer: PreTrainedTokenizer,
    config: DataConfig,
) -> Dataset:
    logger.info(f"Loading dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name, split=config.split)

    formatter = DatasetFormatter(tokenizer, config)
    logger.info("Formatting dataset")
    dataset = dataset.map(formatter, batched=True, num_proc=6)

    initial_size = len(dataset)
    dataset = dataset.filter(lambda x: x["text"] and len(x["text"].strip()) > 0)
    final_size = len(dataset)

    logger.info(f"Dataset: {initial_size} -> {final_size} examples after filtering")
    return dataset
