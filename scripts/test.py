#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fim import CodeCompleter, Config, ModelLoader, setup_logger


TEST_CASES = [
    {
        "name": "Python - Calculate Average",
        "prefix": "def calculate_average(numbers):\n    total = sum(numbers)\n    count = len(numbers)\n    ",
        "suffix": "",
        "language": "Python",
    },
    {
        "name": "Python - List Comprehension",
        "prefix": "def filter_even_numbers(numbers):\n    return [",
        "suffix": "]",
        "language": "Python",
    },
    {
        "name": "JavaScript - Array Map",
        "prefix": "function doubleValues(arr) {\n    return arr.map(",
        "suffix": ");\n}",
        "language": "JavaScript",
    },
]


def main() -> None:
    config = Config()
    logger = setup_logger(level=config.log_level)

    # Check if fine-tuned model exists
    model_path = config.training.output_dir / "final_model"
    if model_path.exists():
        logger.info(f"Loading fine-tuned model from {model_path}")
        config.model.name = str(model_path)
    else:
        logger.info(f"Fine-tuned model not found at {model_path}, using base model")

    logger.info(f"Model: {config.model.name}")

    model_loader = ModelLoader(config.model, config.lora)
    model, tokenizer = model_loader.load()

    completer = CodeCompleter(model, tokenizer, config.inference)

    print("\n" + "=" * 70)
    print("TESTING FINE-TUNED MODEL")
    print("=" * 70)

    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n[Test {i}/{len(TEST_CASES)}] {test_case['name']}")
        print("-" * 70)
        print("PREFIX:")
        print(test_case['prefix'])
        if test_case['suffix']:
            print(f"SUFFIX: {test_case['suffix']}")
        print("-" * 70)

        result = completer.complete(
            prefix=test_case['prefix'],
            suffix=test_case['suffix'],
            language=test_case['language'],
        )

        print("COMPLETION:")
        print(result)
        print("=" * 70)

    logger.info("Testing complete")


if __name__ == "__main__":
    main()
