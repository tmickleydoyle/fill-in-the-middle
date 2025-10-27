#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fim import CodeCompleter, Config, ModelLoader, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run code completion inference")
    parser.add_argument("--prefix", required=True, help="Code before completion")
    parser.add_argument("--suffix", default="", help="Code after completion")
    parser.add_argument("--language", default="Python", help="Programming language")
    parser.add_argument("--context", nargs="*", help="Additional context snippets")
    parser.add_argument("--stream", action="store_true", help="Stream output")
    parser.add_argument("--model-path", help="Path to model (default: auto-detect outputs/final_model or base model)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()
    logger = setup_logger(level=config.log_level)

    # Auto-detect fine-tuned model or use specified path
    if args.model_path:
        logger.info(f"Loading model from {args.model_path}")
        config.model.name = args.model_path
    else:
        default_path = config.training.output_dir / "final_model"
        if default_path.exists():
            logger.info(f"Fine-tuned model found, loading from {default_path}")
            config.model.name = str(default_path)
        else:
            logger.info(f"No fine-tuned model at {default_path}, using base model: {config.model.name}")

    model_loader = ModelLoader(config.model, config.lora)
    model, tokenizer = model_loader.load()

    completer = CodeCompleter(model, tokenizer, config.inference)

    logger.info("Running inference")
    result = completer.complete(
        prefix=args.prefix,
        suffix=args.suffix,
        language=args.language,
        context=args.context,
        stream=args.stream,
    )

    if not args.stream:
        print("\n" + "=" * 50)
        print("COMPLETION:")
        print("=" * 50)
        print(result)


if __name__ == "__main__":
    main()
