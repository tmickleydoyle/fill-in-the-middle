# Google Colab Setup Guide

This guide shows how to run the FIM training pipeline in Google Colab.

## Quick Start

### Method 1: Using setup_colab.py (Recommended)

```python
# Clone repository
!git clone https://github.com/tmickleydoyle/fill-in-the-middle.git
%cd fill-in-the-middle

# Verify GPU
!nvidia-smi

# Install dependencies
!pip install -q setuptools wheel torch>=2.8.0 transformers==4.56.2 datasets>=2.14.0 \
    trl==0.22.2 pydantic>=2.0.0 pydantic-settings>=2.0.0 bitsandbytes>=0.41.0 accelerate>=0.24.0
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Setup configuration (creates .env file)
!python scripts/setup_colab.py

# Run training
!python scripts/train.py
```

### Method 2: Using IPython magic commands

```python
# Clone and setup (same as above)
!git clone https://github.com/tmickleydoyle/fill-in-the-middle.git
%cd fill-in-the-middle

# Install dependencies (same as above)

# Set environment variables directly
%env FIM_TRAIN_MAX_STEPS=250
%env FIM_DATA_ENABLE_RESPONSE_MASKING=false
%env FIM_LOG_LEVEL=INFO

# Run training
!python scripts/train.py
```

### Method 3: Using Python os.environ

```python
import os

# Clone and setup (same as above)

# Set environment variables in Python
os.environ['FIM_TRAIN_MAX_STEPS'] = '250'
os.environ['FIM_DATA_ENABLE_RESPONSE_MASKING'] = 'false'
os.environ['FIM_LOG_LEVEL'] = 'INFO'

# Run training
!python scripts/train.py
```

## Custom Configuration

### Using setup_colab.py with custom settings

```python
# Import the setup module
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "scripts"))

from setup_colab import setup_environment, verify_configuration

# Configure with custom settings
setup_environment(
    max_steps=500,              # Train for 500 steps
    enable_response_masking=True,  # Only train on completions
    log_level="DEBUG",          # Verbose logging
    use_env_file=True           # Create .env file
)

# Verify configuration loaded correctly
verify_configuration()
```

### Creating .env file manually

```bash
# Create .env file with custom settings
cat > .env << 'EOF'
FIM_TRAIN_MAX_STEPS=250
FIM_DATA_ENABLE_RESPONSE_MASKING=false
FIM_LOG_LEVEL=INFO

# Model settings
FIM_MODEL_NAME=unsloth/Qwen3-8B-bnb-4bit
FIM_MODEL_MAX_SEQ_LENGTH=1024

# Training settings
FIM_TRAIN_LEARNING_RATE=1e-5
FIM_TRAIN_WARMUP_RATIO=0.1
FIM_TRAIN_LOGGING_STEPS=10
EOF

# Verify file was created
cat .env
```

## Troubleshooting

### Configuration not being applied

If your configuration changes aren't taking effect:

1. **Verify .env file exists:**
   ```bash
   !ls -la .env
   !cat .env
   ```

2. **Check current working directory:**
   ```python
   import os
   print(f"Current directory: {os.getcwd()}")
   print(f".env exists: {os.path.exists('.env')}")
   ```

3. **Verify configuration loads correctly:**
   ```python
   !python scripts/setup_colab.py
   ```

4. **Check for duplicate entries:**
   If using `echo >> .env` multiple times, you may have duplicates. Use `>` to overwrite:
   ```bash
   !echo "FIM_TRAIN_MAX_STEPS=250" > .env  # Overwrites
   !cat .env  # Verify
   ```

### Still training for default steps?

The configuration is loaded when the script starts. If you change `.env` or environment variables:

1. **Restart the Python kernel** (Runtime > Restart runtime)
2. **Re-run the training script** from a fresh cell
3. **Verify configuration** before training:
   ```python
   from setup_colab import verify_configuration
   verify_configuration()
   ```

### Memory issues

If you run out of GPU memory:

```python
# Reduce batch size or sequence length
os.environ['FIM_TRAIN_PER_DEVICE_TRAIN_BATCH_SIZE'] = '1'
os.environ['FIM_MODEL_MAX_SEQ_LENGTH'] = '512'
```

## Configuration Reference

### Environment Variables

All configuration uses the `FIM_` prefix. Common settings:

**Training:**
- `FIM_TRAIN_MAX_STEPS` - Number of training steps (default: 100)
- `FIM_TRAIN_NUM_TRAIN_EPOCHS` - Number of epochs (if max_steps not set)
- `FIM_TRAIN_LEARNING_RATE` - Learning rate (default: 1e-5)
- `FIM_TRAIN_LOGGING_STEPS` - Log every N steps (default: 10)
- `FIM_TRAIN_SAVE_STEPS` - Save checkpoint every N steps (default: 500)

**Data:**
- `FIM_DATA_ENABLE_RESPONSE_MASKING` - Train only on completions (default: false)
- `FIM_DATA_MAX_CONTEXT_ITEMS` - Number of context examples (default: 3)

**Model:**
- `FIM_MODEL_NAME` - Model to load (default: unsloth/Qwen3-8B-bnb-4bit)
- `FIM_MODEL_MAX_SEQ_LENGTH` - Max sequence length (default: 1024)

**Logging:**
- `FIM_LOG_LEVEL` - DEBUG, INFO, WARNING, or ERROR (default: INFO)

See `.env` file for complete reference.

## Example Colab Notebook

Complete notebook example:

```python
# Cell 1: Clone and install
!git clone https://github.com/tmickleydoyle/fill-in-the-middle.git
%cd fill-in-the-middle
!nvidia-smi

# Cell 2: Install dependencies
!pip install -q setuptools wheel torch>=2.8.0 transformers==4.56.2 datasets>=2.14.0 \
    trl==0.22.2 pydantic>=2.0.0 pydantic-settings>=2.0.0 bitsandbytes>=0.41.0 accelerate>=0.24.0
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Cell 3: Configure training
!python scripts/setup_colab.py

# Cell 4: Verify configuration
from scripts.setup_colab import verify_configuration
verify_configuration()

# Cell 5: Run training
!python scripts/train.py

# Cell 6 (optional): Push to Hugging Face
import os
os.environ['FIM_HF_PUSH_TO_HUB'] = 'true'
os.environ['FIM_HF_REPO_NAME'] = 'my-fim-model'
os.environ['HF_TOKEN'] = 'your_token_here'
!python scripts/train.py
```

## Next Steps

After training completes:
- Model saved to `outputs/final_model/`
- Use `scripts/infer.py` for code completion
- Push to Hugging Face Hub with `FIM_HF_PUSH_TO_HUB=true`

See main README.md for inference examples.
