# Fill-in-the-Middle Fine-Tuning

Production-grade Python application for fine-tuning Qwen3-8B on code completion tasks using fill-in-the-middle training.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python scripts/train.py
```

### Inference
```bash
python scripts/infer.py \
  --prefix "def calculate_average(numbers):" \
  --suffix "return total / count" \
  --language Python
```

## Configuration

Configure via environment variables or `.env` file:

```bash
FIM_MODEL_NAME=unsloth/Qwen3-8B-bnb-4bit
FIM_DATA_DATASET_NAME=sourcegraph/context-aware-fim-code-completions
FIM_DATA_ENABLE_RESPONSE_MASKING=false
FIM_TRAIN_NUM_TRAIN_EPOCHS=1
FIM_TRAIN_LEARNING_RATE=1e-5
FIM_LOG_LEVEL=INFO
```

## Project Structure

```
.
├── src/fim/              # Core library
│   ├── config.py         # Configuration management
│   ├── data.py           # Dataset pipeline
│   ├── model.py          # Model loading
│   ├── training.py       # Training orchestration
│   ├── inference.py      # Inference utilities
│   └── logger.py         # Centralized logging
├── scripts/              # CLI entry points
│   ├── train.py          # Training script
│   └── infer.py          # Inference script
├── tests/                # Test suite
├── archive/              # Original implementation
└── pyproject.toml        # Project metadata
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM)
- 16GB+ RAM

## Development

### Setup
```bash
pip install -e ".[dev]"
```

### Testing
```bash
pytest tests/
pytest --cov=src/fim tests/
```

### Code Quality
```bash
ruff check src/ tests/
ruff format src/ tests/
mypy src/
```

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation and development guidelines.

## License

See LICENSE file for details.
