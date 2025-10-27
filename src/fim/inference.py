import logging
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer

from .config import InferenceConfig

logger = logging.getLogger("fim")


class CodeCompleter:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: InferenceConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def complete(
        self,
        prefix: str,
        suffix: str,
        language: str = "Python",
        context: Optional[list[str]] = None,
        stream: bool = False,
    ) -> str:
        messages = self._build_messages(prefix, suffix, language, context)
        inputs = self._prepare_inputs(messages)

        streamer = TextStreamer(self.tokenizer) if stream else None

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                streamer=streamer,
            )

        if stream:
            return ""

        response_ids = outputs[0][len(inputs["input_ids"][0]):]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)

    def _build_messages(
        self,
        prefix: str,
        suffix: str,
        language: str,
        context: Optional[list[str]],
    ) -> list[dict[str, str]]:
        context_section = ""
        if context:
            context_section = "### Relevant code context:\n\n"
            for ctx in context:
                context_section += f"```{language}\n{ctx}\n```\n\n"

        user_content = (
            f"{context_section}### Complete the following {language} code:\n\n"
            f"```{language}\n{prefix}<FILL_HERE>{suffix}\n```"
        )

        return [
            {
                "role": "system",
                "content": (
                    f"You are an expert code completion assistant. "
                    f"Complete the code at the <FILL_HERE> marker with syntactically correct "
                    f"{language} code that fits naturally with the surrounding context."
                )
            },
            {"role": "user", "content": user_content},
        ]

    def _prepare_inputs(self, messages: list[dict[str, str]]) -> dict:
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=self.config.enable_thinking,
        ).to("cuda")
