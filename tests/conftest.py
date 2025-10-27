import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fim import Config, DataConfig, InferenceConfig, LoRAConfig, ModelConfig, TrainingConfig


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig()


@pytest.fixture
def lora_config() -> LoRAConfig:
    return LoRAConfig()


@pytest.fixture
def data_config() -> DataConfig:
    return DataConfig()


@pytest.fixture
def training_config() -> TrainingConfig:
    return TrainingConfig()


@pytest.fixture
def inference_config() -> InferenceConfig:
    return InferenceConfig()


@pytest.fixture
def sample_fim_example() -> dict:
    return {
        "lang": "Python",
        "prefix": "def calculate_average(numbers):\n    if not numbers:\n        return 0\n    ",
        "suffix": "\n    return total / count",
        "middle": "total = sum(numbers)\n    count = len(numbers)",
        "context_items": [],
    }
