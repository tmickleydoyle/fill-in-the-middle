import logging
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only

from .config import DataConfig, HuggingFaceConfig, TrainingConfig

logger = logging.getLogger("fim")


class MaskingStrategy:
    CHATML_PATTERNS = [
        ("<|im_start|>user\n", "<|im_start|>assistant\n", "ChatML with newlines"),
        ("<|im_start|>user", "<|im_start|>assistant", "ChatML without newlines"),
    ]

    HARMONY_PATTERNS = [
        (
            "<|start|>user<|message|>",
            "<|start|>assistant<|channel|>final<|message|>",
            "Harmony with channels"
        ),
        (
            "<|start|>user<|message|>",
            "<|start|>assistant<|message|>",
            "Harmony without channels"
        ),
    ]

    @classmethod
    def detect(cls, sample_text: str) -> Optional[tuple[str, str, str]]:
        for user_pat, asst_pat, desc in cls.CHATML_PATTERNS + cls.HARMONY_PATTERNS:
            if user_pat in sample_text and asst_pat in sample_text:
                return user_pat, asst_pat, desc
        return None


class Trainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset,
        training_config: TrainingConfig,
        data_config: DataConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.training_config = training_config
        self.data_config = data_config

    def train(self) -> dict[str, float]:
        trainer = self._create_trainer()
        trainer = self._apply_masking(trainer)
        self._validate_data(trainer)

        logger.info("Starting training")
        result = trainer.train()

        self._log_training_stats(result)
        return result.metrics

    def _create_trainer(self) -> SFTTrainer:
        self.training_config.output_dir.mkdir(parents=True, exist_ok=True)

        # Handle num_train_epochs and max_steps logic
        if self.training_config.max_steps and self.training_config.max_steps > 0:
            # Using max_steps: set epochs to 1 (will be overridden by max_steps)
            num_epochs = 1
            max_steps = self.training_config.max_steps
            logger.info(f"Training mode: {max_steps} steps")
        elif self.training_config.num_train_epochs:
            # Using epochs: disable max_steps
            num_epochs = self.training_config.num_train_epochs
            max_steps = -1
            logger.info(f"Training mode: {num_epochs} epochs")
        else:
            # Default: 1 epoch
            num_epochs = 1
            max_steps = -1
            logger.info("Training mode: 1 epoch (default)")

        args = SFTConfig(
            output_dir=str(self.training_config.output_dir),
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            warmup_ratio=self.training_config.warmup_ratio,
            max_grad_norm=self.training_config.max_grad_norm,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            optim=self.training_config.optim,
            weight_decay=self.training_config.weight_decay,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            seed=self.training_config.seed,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            report_to="none",
        )

        return SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            args=args,
        )

    def _apply_masking(self, trainer: SFTTrainer) -> SFTTrainer:
        if not self.data_config.enable_response_masking:
            logger.info("Response masking disabled - training on full sequences")
            return trainer

        sample_text = self.dataset[0]["text"]
        pattern = MaskingStrategy.detect(sample_text)

        if pattern is None:
            logger.warning("Could not detect masking patterns - skipping masking")
            return trainer

        user_pat, asst_pat, desc = pattern
        logger.info(f"Masking enabled: {desc}")
        logger.info(f"User pattern: {repr(user_pat)}")
        logger.info(f"Assistant pattern: {repr(asst_pat)}")

        return train_on_responses_only(
            trainer,
            instruction_part=user_pat,
            response_part=asst_pat,
        )

    def _validate_data(self, trainer: SFTTrainer) -> None:
        try:
            input_ids = trainer.train_dataset[0]["input_ids"]
            logger.info(f"Data validation: {len(input_ids)} tokens in first example")

            if "labels" in trainer.train_dataset[0]:
                labels = trainer.train_dataset[0]["labels"]
                non_masked = sum(1 for label in labels if label != -100)
                logger.info(f"Training on {non_masked} non-masked tokens")

                if non_masked == 0:
                    logger.error("All labels are masked! Training will fail.")
        except Exception as e:
            logger.error(f"Data validation failed: {e}")

    def _log_training_stats(self, result: Any) -> None:
        gpu_stats = torch.cuda.get_device_properties(0)
        peak_memory = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024

        logger.info(f"Training completed in {result.metrics['train_runtime']:.2f}s")
        logger.info(f"GPU: {gpu_stats.name}, Peak memory: {peak_memory:.2f} GB")

    def save(self, path: Path) -> None:
        logger.info(f"Saving model to {path}")
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))

    def push_to_hub(self, hf_config: HuggingFaceConfig, local_path: Path) -> None:
        import os

        repo_id = f"{hf_config.username}/{hf_config.repo_name}"
        logger.info(f"Pushing model to Hugging Face Hub: {repo_id}")

        token = hf_config.token or os.environ.get("HF_TOKEN")
        if not token:
            logger.error("No Hugging Face token found. Set HF_TOKEN environment variable or FIM_HF_TOKEN.")
            return

        try:
            logger.info(f"Uploading model (private={hf_config.private})...")
            self.model.push_to_hub(
                repo_id=repo_id,
                token=token,
                private=hf_config.private,
            )

            logger.info("Uploading tokenizer...")
            self.tokenizer.push_to_hub(
                repo_id=repo_id,
                token=token,
                private=hf_config.private,
            )

            logger.info(f"Successfully pushed to https://huggingface.co/{repo_id}")

        except Exception as e:
            logger.error(f"Failed to push to Hugging Face Hub: {e}")
            logger.info("Model is still saved locally at {local_path}")
