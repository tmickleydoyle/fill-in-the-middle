#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fim import (
    Config,
    ModelLoader,
    Trainer,
    load_and_format_dataset,
    setup_logger,
)


def main() -> None:
    config = Config()
    logger = setup_logger(level=config.log_level)

    logger.info("Initializing training pipeline")
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Dataset: {config.data.dataset_name}")

    model_loader = ModelLoader(config.model, config.lora)
    model, tokenizer = model_loader.load()

    dataset = load_and_format_dataset(tokenizer, config.data)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        training_config=config.training,
        data_config=config.data,
    )

    metrics = trainer.train()
    logger.info(f"Training metrics: {metrics}")

    save_path = config.training.output_dir / "final_model"
    trainer.save(save_path)

    if config.huggingface.push_to_hub:
        trainer.push_to_hub(config.huggingface, save_path)

    logger.info("Training complete")


if __name__ == "__main__":
    main()
