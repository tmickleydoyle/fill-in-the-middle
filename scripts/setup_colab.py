#!/usr/bin/env python3
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def setup_environment(
    max_steps: int = 250,
    enable_response_masking: bool = False,
    log_level: str = "INFO",
    use_env_file: bool = True,
) -> None:
    if use_env_file:
        _create_env_file(max_steps, enable_response_masking, log_level)
    else:
        _set_environment_variables(max_steps, enable_response_masking, log_level)


def _create_env_file(
    max_steps: int,
    enable_response_masking: bool,
    log_level: str,
) -> None:
    env_path = Path(".env")

    config_lines = [
        f"FIM_TRAIN_MAX_STEPS={max_steps}",
        f"FIM_DATA_ENABLE_RESPONSE_MASKING={str(enable_response_masking).lower()}",
        f"FIM_LOG_LEVEL={log_level}",
    ]

    env_path.write_text("\n".join(config_lines) + "\n")

    print(f"Created .env file at: {env_path.absolute()}")
    print(f"Contents:")
    print(env_path.read_text())


def _set_environment_variables(
    max_steps: int,
    enable_response_masking: bool,
    log_level: str,
) -> None:
    os.environ["FIM_TRAIN_MAX_STEPS"] = str(max_steps)
    os.environ["FIM_DATA_ENABLE_RESPONSE_MASKING"] = str(enable_response_masking).lower()
    os.environ["FIM_LOG_LEVEL"] = log_level

    print("Set environment variables:")
    print(f"  FIM_TRAIN_MAX_STEPS={max_steps}")
    print(f"  FIM_DATA_ENABLE_RESPONSE_MASKING={str(enable_response_masking).lower()}")
    print(f"  FIM_LOG_LEVEL={log_level}")


def verify_configuration() -> None:
    from fim import Config

    config = Config()

    print("\nConfiguration loaded:")
    print(f"  Max steps: {config.training.max_steps}")
    print(f"  Response masking: {config.data.enable_response_masking}")
    print(f"  Log level: {config.log_level}")
    print(f"  Output dir: {config.training.output_dir}")

    env_file = Path(".env")
    if env_file.exists():
        print(f"\n.env file exists at: {env_file.absolute()}")
    else:
        print("\n.env file not found - using environment variables or defaults")


if __name__ == "__main__":
    print("Setting up Google Colab environment for FIM training\n")

    setup_environment(
        max_steps=250,
        enable_response_masking=False,
        log_level="INFO",
        use_env_file=True,
    )

    print("\nVerifying configuration...")
    verify_configuration()
