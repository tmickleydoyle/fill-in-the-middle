# Fill-in-the-Middle Fine-Tuning

Python application for fine-tuning Qwen3-8B on code completion tasks using fill-in-the-middle training.

## Quick Start

### Google Colab (Recommended)

**Step 1: Enable GPU**
- Go to: Runtime → Change runtime type → T4 GPU

**Step 2: Setup (in first cell)**
```python
# Clone repository
!git clone https://github.com/tmickleydoyle/fill-in-the-middle.git
%cd fill-in-the-middle

# Install dependencies (skip triton for Colab)
!pip install -q setuptools wheel torch>=2.8.0 transformers==4.56.2 datasets>=2.14.0 \
  trl==0.22.2 pydantic>=2.0.0 pydantic-settings>=2.0.0 bitsandbytes>=0.41.0 accelerate>=0.24.0

# Install unsloth
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**Step 3: Configure Training (in second cell)**
```python
# Option A: Quick test (10 steps, ~2 minutes)
!echo "FIM_TRAIN_MAX_STEPS=10" > .env
!echo "FIM_TRAIN_LOGGING_STEPS=1" >> .env
!echo "FIM_DATA_ENABLE_RESPONSE_MASKING=false" >> .env
!echo "FIM_LOG_LEVEL=INFO" >> .env

# Option B: Full training (100 steps)
# Just delete the .env file to use defaults
# !rm .env

# Optional: Enable Hugging Face upload after training
# Requires HF_TOKEN to be set in Colab environment
# !echo "FIM_HF_PUSH_TO_HUB=true" >> .env
# !echo "FIM_HF_REPO_NAME=graft-fim" >> .env
# !echo "FIM_HF_USERNAME=tmickleydoyle" >> .env
```

**Step 4: Train Model (in third cell)**
```python
# Trains on Sourcegraph FIM dataset
# Default: 100 steps (or 10 if configured above)
# Saves to: outputs/final_model
!python scripts/train.py
```

**Step 5: Test Fine-Tuned Model (in fourth cell)**
```python
# Test with predefined examples (Python & JavaScript)
!python scripts/test.py

# Or test with your own custom prompt
!python scripts/infer.py \
  --prefix "def my_function(x, y):\n    " \
  --language Python
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

**Hugging Face Upload:**
```bash
# Enable automatic upload to Hugging Face after training
# Requires HF_TOKEN environment variable to be set
FIM_HF_PUSH_TO_HUB=true
FIM_HF_REPO_NAME=graft-fim           # Repository name
FIM_HF_USERNAME=tmickleydoyle        # Your HF username
FIM_HF_PRIVATE=false                 # Public repository

# Model will be uploaded to: https://huggingface.co/tmickleydoyle/graft-fim
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
- GPU: Colab T4 GPU works
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

## License

See LICENSE file for details.
