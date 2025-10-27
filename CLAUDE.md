# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Production-grade Python application for fine-tuning Qwen3-8B on fill-in-the-middle code completion using the Sourcegraph dataset. Built with modern engineering practices: modular architecture, type safety, comprehensive testing, and centralized configuration.

## Architecture

### Module Responsibilities

**src/fim/config.py** - Configuration Management
- Pydantic models for type-safe configuration
- Environment variable support with FIM_ prefix
- Validation at application startup
- No magic strings or hardcoded values

**src/fim/logger.py** - Centralized Logging
- Structured logging with consistent format
- Configurable log levels
- File and console output support
- Used throughout application (no print statements)

**src/fim/data.py** - Data Pipeline
- Dataset loading from Hugging Face
- FIM format to ChatML conversion
- Validation and filtering
- Context-aware prompt building

**src/fim/model.py** - Model Management
- Unsloth FastLanguageModel integration
- LoRA adapter configuration
- Dtype handling
- Clean model/tokenizer loading interface

**src/fim/training.py** - Training Orchestration
- SFTTrainer configuration
- Automatic masking pattern detection
- Training metrics and validation
- Model checkpointing

**src/fim/inference.py** - Inference Utilities
- Code completion interface
- Streaming support
- Context injection
- Generation parameter management

### Entry Points

**scripts/train.py** - Training CLI
```bash
python scripts/train.py
```
Reads configuration from environment variables (FIM_* prefix) or .env file.

**scripts/infer.py** - Inference CLI
```bash
python scripts/infer.py \
  --prefix "def calculate_average(numbers):" \
  --suffix "return total / count" \
  --language Python \
  --stream
```

## Development Workflow

### Setup
```bash
pip install -r requirements.txt
pip install -e ".[dev]"  # Development dependencies
```

### Environment Configuration
Create `.env` file:
```bash
FIM_MODEL_NAME=unsloth/Qwen3-8B-bnb-4bit
FIM_DATA_ENABLE_RESPONSE_MASKING=false
FIM_TRAIN_NUM_TRAIN_EPOCHS=1
FIM_LOG_LEVEL=INFO
```

### Running Tests
```bash
pytest tests/                    # All tests
pytest tests/test_data.py       # Specific module
pytest --cov=src/fim tests/     # With coverage
```

### Code Quality
```bash
ruff check src/ tests/          # Linting
ruff format src/ tests/         # Formatting
mypy src/                       # Type checking
```

## Key Design Patterns

### Configuration Pattern
- Pydantic BaseSettings for env vars
- Nested config models for organization
- Type validation at startup
- Default values in code, overrides in env

### Dependency Injection
- ModelLoader receives config objects
- Trainer receives all dependencies
- Easy testing with mocks
- Clear dependency graph

### Strategy Pattern
- MaskingStrategy for pattern detection
- Supports ChatML and Harmony formats
- Extensible for new formats
- Automatic fallback handling

## Training Configuration

### GPU Requirements
- **Minimum**: T4 (14.7 GB) or 8GB+ VRAM
- **Memory**: ~5-6 GB with 4-bit quantization
- **Recommended**: T4, V100, or A100

### Key Hyperparameters
- Learning rate: 1e-5 (conservative for stability)
- Batch size: 1 with gradient accumulation 4
- Warmup ratio: 0.1
- Max gradient norm: 0.5 (tight clipping)
- Optimizer: adamw_8bit

### Masking Strategy
Set `FIM_DATA_ENABLE_RESPONSE_MASKING=true` to train only on code completions. When disabled, trains on full sequences (safer for debugging NaN loss).

## Dataset Structure

- **Source**: sourcegraph/context-aware-fim-code-completions
- **Size**: 13.5k examples across 3 languages
- **Format**: Fill-in-the-middle with context
- **Fields**: prefix, suffix, middle, context_items, lang

## Testing Strategy

### Unit Tests
- Mock external dependencies (HuggingFace, Unsloth)
- Test business logic in isolation
- Fast execution
- pytest fixtures for common objects

### Integration Points
- Data loading and formatting
- Model initialization
- Training loop
- Inference pipeline

### Coverage Goals
- Core logic: >90%
- Configuration: 100%
- Utilities: >80%

## Common Development Tasks

### Adding New Configuration
1. Add field to relevant config class in config.py
2. Use FIM_{SECTION}_{FIELD} env var pattern
3. Add validation if needed
4. Update .env.example

### Extending Masking Patterns
1. Add pattern to MaskingStrategy.CHATML_PATTERNS or HARMONY_PATTERNS
2. Add test case in test_training.py
3. Document in CLAUDE.md

### Adding New Inference Mode
1. Extend InferenceConfig in config.py
2. Update CodeCompleter.complete() method
3. Add CLI argument to scripts/infer.py
4. Add test case

## Original Implementation

The original monolithic script is preserved in `archive/fill-in-the-middle-fine-tune.py` for reference. It contains the initial Colab notebook implementation before refactoring.

## Best Practices Enforced

- **Type Hints**: 100% coverage with mypy validation
- **No Inline Comments**: Self-documenting code via clear naming
- **Centralized Logging**: logger.info/warning/error (never print)
- **Configuration**: Environment variables, no hardcoded values
- **Error Handling**: Specific exceptions with context
- **Testing**: pytest with fixtures and mocks
- **Code Quality**: ruff linting and formatting
- **Documentation**: Minimal but clear docstrings
