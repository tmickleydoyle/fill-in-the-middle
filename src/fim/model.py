import logging
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from unsloth import FastLanguageModel

from .config import LoRAConfig, ModelConfig

logger = logging.getLogger("fim")


class ModelLoader:
    def __init__(self, model_config: ModelConfig, lora_config: LoRAConfig):
        self.model_config = model_config
        self.lora_config = lora_config

    def load(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        logger.info(f"Loading model: {self.model_config.name}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config.name,
            dtype=self._get_dtype(),
            max_seq_length=self.model_config.max_seq_length,
            load_in_4bit=self.model_config.load_in_4bit,
            full_finetuning=False,
        )

        logger.info("Applying LoRA adapters")
        model = self._apply_lora(model)

        return model, tokenizer

    def _get_dtype(self) -> Optional[torch.dtype]:
        if self.model_config.dtype is None:
            return None

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.model_config.dtype)

    def _apply_lora(self, model: PreTrainedModel) -> PreTrainedModel:
        return FastLanguageModel.get_peft_model(
            model,
            r=self.lora_config.r,
            target_modules=self.lora_config.target_modules,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            use_gradient_checkpointing=self.lora_config.use_gradient_checkpointing,
            random_state=3407,
            use_rslora=self.lora_config.use_rslora,
            loftq_config=None,
        )
