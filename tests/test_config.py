import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fim import Config, DataConfig, TrainingConfig


def test_training_config_reads_from_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FIM_TRAIN_MAX_STEPS=250\n")

    monkeypatch.chdir(tmp_path)

    config = TrainingConfig()

    assert config.max_steps == 250


def test_training_config_reads_multiple_values_from_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "FIM_TRAIN_MAX_STEPS=500\n"
        "FIM_TRAIN_LEARNING_RATE=2e-5\n"
        "FIM_TRAIN_LOGGING_STEPS=25\n"
    )

    monkeypatch.chdir(tmp_path)

    config = TrainingConfig()

    assert config.max_steps == 500
    assert config.learning_rate == 2e-5
    assert config.logging_steps == 25


def test_data_config_reads_from_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FIM_DATA_ENABLE_RESPONSE_MASKING=true\n")

    monkeypatch.chdir(tmp_path)

    config = DataConfig()

    assert config.enable_response_masking is True


def test_nested_configs_in_parent_read_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "FIM_TRAIN_MAX_STEPS=300\n"
        "FIM_DATA_ENABLE_RESPONSE_MASKING=true\n"
        "FIM_LOG_LEVEL=DEBUG\n"
    )

    monkeypatch.chdir(tmp_path)

    config = Config()

    assert config.training.max_steps == 300
    assert config.data.enable_response_masking is True
    assert config.log_level == "DEBUG"


def test_environment_variables_override_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FIM_TRAIN_MAX_STEPS=100\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FIM_TRAIN_MAX_STEPS", "500")

    config = TrainingConfig()

    assert config.max_steps == 500


def test_default_values_when_no_env_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    config = TrainingConfig()

    assert config.max_steps == 100
    assert config.learning_rate == 1e-5
    assert config.logging_steps == 10


def test_colab_setup_scenario(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "FIM_TRAIN_MAX_STEPS=250\n"
        "FIM_DATA_ENABLE_RESPONSE_MASKING=false\n"
        "FIM_LOG_LEVEL=INFO\n"
    )

    monkeypatch.chdir(tmp_path)

    config = Config()

    assert config.training.max_steps == 250
    assert config.data.enable_response_masking is False
    assert config.log_level == "INFO"
