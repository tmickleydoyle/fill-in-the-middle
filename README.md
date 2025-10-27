# Fill-in-the-Middle Fine-Tuning

Production-grade Python application for fine-tuning Qwen3-8B on code completion tasks using fill-in-the-middle training.

## Quick Start

### Google Colab (Recommended)

**Enable GPU:** Runtime → Change runtime type → T4 GPU

```python
# Clone repository
!git clone https://github.com/tmickleydoyle/fill-in-the-middle.git
%cd fill-in-the-middle

# Verify GPU
!nvidia-smi

# Install dependencies (skip triton for Colab)
!pip install -q setuptools wheel torch>=2.8.0 transformers==4.56.2 datasets>=2.14.0 \
  trl==0.22.2 pydantic>=2.0.0 pydantic-settings>=2.0.0 bitsandbytes>=0.41.0 accelerate>=0.24.0

# Install unsloth
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Create minimal test configuration (10 steps for quick testing)
# Remove .env or set FIM_TRAIN_MAX_STEPS=100 for full training
!echo "FIM_TRAIN_MAX_STEPS=10" > .env
!echo "FIM_TRAIN_LOGGING_STEPS=1" >> .env
!echo "FIM_DATA_ENABLE_RESPONSE_MASKING=false" >> .env
!echo "FIM_LOG_LEVEL=INFO" >> .env

# Run training
!python scripts/train.py

# Test fine-tuned model with sample prompts
!python scripts/test.py
```

### Local Installation

**Requirements:** NVIDIA/AMD/Intel GPU (Unsloth does not support Apple Silicon)

```bash
pip install -r requirements.txt
```

### Training
```bash
python scripts/train.py
```

### Testing
Test the fine-tuned model with predefined prompts:
```bash
python scripts/test.py
```

### Custom Inference
Run inference with custom prompts (automatically uses fine-tuned model if available):
```bash
# Uses fine-tuned model from outputs/final_model (or base model if not found)
python scripts/infer.py \
  --prefix "def calculate_average(numbers):" \
  --suffix "return total / count" \
  --language Python

# Override with specific model path
python scripts/infer.py \
  --model-path path/to/custom/model \
  --prefix "def process_data(items):" \
  --language Python
```

## Configuration

Configure via environment variables or `.env` file. See `.env.example` for all options.

**Default:** 100 training steps (configurable via `FIM_TRAIN_MAX_STEPS`)

**Common Settings:**
```bash
FIM_MODEL_NAME=unsloth/Qwen3-8B-bnb-4bit
FIM_DATA_DATASET_NAME=sourcegraph/context-aware-fim-code-completions
FIM_DATA_ENABLE_RESPONSE_MASKING=false
FIM_TRAIN_MAX_STEPS=100  # Default: 100 steps
FIM_TRAIN_LEARNING_RATE=1e-5
FIM_LOG_LEVEL=INFO
```

**Alternative Training Modes:**
```bash
# Use epochs instead of steps
FIM_TRAIN_NUM_TRAIN_EPOCHS=1
FIM_TRAIN_MAX_STEPS=-1  # Disable max_steps

# Minimal testing (10 steps)
FIM_TRAIN_MAX_STEPS=10
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
│   ├── test.py           # Test fine-tuned model
│   └── infer.py          # Custom inference script
├── tests/                # Test suite
├── archive/              # Original implementation
└── pyproject.toml        # Project metadata
```

## Requirements

- Python 3.10+
- GPU: NVIDIA/AMD/Intel with 8GB+ VRAM (Colab T4 GPU works)
- **Note:** Unsloth does not support Apple Silicon (use Colab instead)
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
