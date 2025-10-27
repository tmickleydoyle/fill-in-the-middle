from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FIM_MODEL_")

    name: str = "unsloth/Qwen3-8B-bnb-4bit"
    max_seq_length: int = 1024
    load_in_4bit: bool = True
    dtype: Optional[str] = None


class LoRAConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FIM_LORA_")

    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list[str] = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    use_rslora: bool = False


class DataConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FIM_DATA_")

    dataset_name: str = "sourcegraph/context-aware-fim-code-completions"
    split: str = "train"
    max_context_items: int = 3
    enable_response_masking: bool = False


class TrainingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FIM_TRAIN_")

    output_dir: Path = Path("outputs")
    num_train_epochs: Optional[int] = 1
    max_steps: Optional[int] = None
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.5
    logging_steps: int = 10
    save_steps: int = 500
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    fp16: bool = False
    bf16: bool = False


class InferenceConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FIM_INFER_")

    max_new_tokens: int = 128
    temperature: float = 0.7
    do_sample: bool = True
    enable_thinking: bool = False


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
