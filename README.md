# StudioEval

**StudioEval** is a comprehensive, lightweight evaluation framework for benchmarking Large Language Models (LLMs) using [LM Studio](https://lmstudio.ai) as the inference backend. It supports local GGUF models and integrates seamlessly with MLX, llama.cpp, and other inference engines through LM Studio's unified API.

> **Inspired by [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)**: StudioEval was created to address lm-eval's limitations with GGUF file handling and lack of MLX support. While it shares a subset of the Hugging Face datasets that lm-eval uses, StudioEval focuses specifically on local inference through LM Studio and implements only a subset of lm-eval's testing features, optimized for this use case.

## ✨ Key Features

- **🏗️ Clean Architecture**: Refactored with separation of concerns - orchestrator, model manager, dataset registry
- **📊 Multiple Evaluation Types**: Supports MCQ, math reasoning, grammar checking, and custom evaluations
- **🔧 Flexible Configuration**: JSON-based dataset configurations with per-dataset parameter overrides
- **🎯 Smart Parameter Precedence**: CLI args override config files, config files override defaults
- **🛡️ Robust Error Handling**: Graceful handling of missing files, malformed configs, and model errors
- **🧪 Comprehensive Testing**: 100+ tests covering unit, integration, and end-to-end scenarios
- **📈 Rich Reporting**: Detailed accuracy metrics, performance stats, and CSV output
- **🔍 Advanced Model Selection**: Support for glob patterns, architecture filters, and model sets

## ⚠️ Current Limitations

**StudioEval currently supports a focused subset of evaluation types:**

- **📝 Exact Answer Matching**: Questions with single, precise answers (e.g., math problems, factual queries)
- **🎯 Multiple Choice Questions**: MCQ datasets with A/B/C/D answer options
- **🚫 Not Supported**: Open-ended generation tasks, complex reasoning chains, subjective evaluations, or tasks requiring human judgment

This design choice ensures reliable, automated scoring while maintaining compatibility with LM Studio's inference capabilities. Future versions may expand to include additional evaluation paradigms.

## 🏗️ Architecture

StudioEval follows a modular, testable architecture:

```
studioeval/
├── studioeval.py             # CLI entry point
├── core/                     # Core business logic
│   ├── evaluator.py          # Main orchestration
│   ├── model_manager.py      # Model lifecycle management
│   └── dataset_registry.py   # Dataset loading and evaluation
├── interfaces/               # Abstract interfaces
│   └── model_client.py       # Model client interface
├── implementations/          # Concrete implementations
│   ├── lmstudio_client.py    # LM Studio API client
│   └── mock_client.py        # Testing mock client
├── eval_datasets/            # Dataset loaders
│   ├── custom/               # Custom evaluation datasets
│   │   ├── grammar.py
│   │   ├── math.py
│   │   └── custom_mcq.py
│   ├── huggingface/          # Hugging Face datasets
│   │   ├── arc.py
│   │   ├── gsm8k.py
│   │   ├── mmlu.py
│   │   ├── commonsense_qa.py
│   │   ├── logiqa.py
│   │   └── truthfulqa.py
│   └── default_evals.py      # Default evaluation configurations
├── utils/                    # Utility functions
│   ├── text_parsing.py       # Text parsing and extraction
│   ├── params.py             # Parameter merging logic
│   ├── timing_utils.py       # Duration formatting
│   └── data_loading.py       # Data loading helpers
├── config/                   # Configuration management
├── models/                   # Model metadata and loading
├── examples/                 # Example configurations
├── tests/                    # Comprehensive test suite
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd StudioEval
pip install -e .
```

### 2. Start LM Studio

1. Install and launch [LM Studio](https://lmstudio.ai)
2. Load your desired models
3. Start the local server (default: http://localhost:1234)

### 3. Basic Usage

```bash
# Evaluate a specific model
python studioeval.py --model "your-model-name"

# Evaluate all loaded models
python studioeval.py --all

# Use a custom dataset configuration
python studioeval.py --model "your-model" --datasets-config examples/run_base_benchmarks.json

# Filter models by pattern
python studioeval.py --model-filter "llama*" --sample-size 50
```

## 📋 Supported Datasets

### 🤗 Hugging Face Datasets
- **ARC** (AI2 Reasoning Challenge) - Science reasoning (Default tinyArc)
- **GSM8K** - Grade school math word problems   (Default tinyGSM8K)
- **MMLU** (Massive Multitask Language Understanding) - Knowledge across 57 subjects (Default tinyMMLU)
- **TruthfulQA** - Truthfulness evaluation - Single Answer (Default tinyTruthfulQA)
- **WinoGrande** - Winograd Schema Challenge  - Fill-in-a-blank task with binary options (Default tinyWinoGrande)
- **HellaSwag** - Commonsense sentence completion (Default tinyHellaswag)
- **CommonSenseQA** - Commonsense reasoning
- **LogiQA** - Logical reasoning

### 🎯 Custom Datasets
- **Grammar** - Grammar correction and validation (Exact sentence answer - 100 Questions)
- **Math** - Separate Elementary and high school mathematics (Exact numeric answer - 100 Questions each)
- **Custom MCQ** - Multiple choice questions for various domains - Proof of Concept Examples
  - ***Creative Writing*** - Writing Mechanics and editing (120 Questions)
  - ***Coding*** - Realistic high technical cross topics (110 Questions)
  - ***Data Analysis*** - Comprehensive data analysis topics (135 Questions)

#### 📝 Adding Your Own Custom MCQ Dataset

StudioEval supports custom Multiple Choice Question (MCQ) datasets in JSONL format. You can easily drop in your own dataset by following the required format and configuration steps.

**Required JSONL Format:**
Each line in your `.jsonl` file must be a valid JSON object with these required fields:

```json
{"id": "q_001", "question": "Your question text here?", "choices": {"A": "First option", "B": "Second option", "C": "Third option", "D": "Fourth option"}, "answer": "B", "difficulty": "medium"}
```

**Required Fields:**
- `id`: Unique identifier for the question (string)
- `question`: The question text (string)
- `choices`: Object with exactly 4 options labeled "A", "B", "C", "D" (object)
- `answer`: The correct answer letter - must be one of "A", "B", "C", or "D" (string)
- `difficulty`: Question difficulty level - "easy", "medium", or "hard" (string)

**Setup Steps:**
1. Create your JSONL file following the format above
2. Place it in `eval_datasets/custom/data/`
3. Add configuration to your dataset config JSON:

```json
{
  "eval_type": "custom_mcq",
  "dataset_path": "eval_datasets/custom/data/your_dataset.jsonl", 
  "dataset_name": "your_dataset_name",
  "sample_size": null
}
```

**Example Usage:**
```bash
# Evaluate using your custom MCQ dataset
python studioeval.py --model "your-model" --datasets-config your_custom_config.json
```

## ⚙️ Configuration

### Dataset Configuration

Create JSON files to specify which datasets to run:

```json
[
  {
    "eval_type": "mmlu",
    "dataset_path": "tinyBenchmarks/tinyMMLU",
    "dataset_name": "tinyMMLU",
    "split": null,
    "seed": 42,
    "sample_size": 100
  },
  {
    "eval_type": "gsm8k",
    "dataset_path": "tinyBenchmarks/tinyGSM8k",
    "dataset_name": "tinyGSM8k",
    "sample_size": null  # Uses CLI default
  }
]
```

### Communication Configuration

Configure LM Studio communication parameters:

```json
{
  "timeout": 120,
  "GENERATION_PARAMS": {
    "temperature": 0.0,
    "topPSampling": 1.0,
    "topKSampling": 0,
    "repeatPenalty": 1.00,
    "maxTokens": 2000
  }
}
```

## 🎛️ Command Line Options

### Model Selection
```bash
--model MODEL                 # Specific model ID
--model-filter PATTERN        # Glob pattern (e.g., 'llama*', '*7B*')
--model-arch ARCH            # Architecture filter (e.g., 'qwen*')
--model-set FILE             # File with model IDs (one per line)
--all                        # Evaluate all loaded models
```

### Evaluation Control
```bash
--datasets-config FILE       # Custom dataset configuration
--sample-size N              # Samples per dataset (0 = all)
--seed N                     # Random seed for reproducibility
```

### Output & Logging
```bash
--log-level LEVEL            # DEBUG, INFO, WARNING, ERROR
--log-file FILE              # Log to file instead of console
--raw-duration               # Output seconds instead of human-readable time
```

### Configuration
```bash
--comm-config FILE           # LM Studio communication settings
```

## 📊 Output & Results

StudioEval provides detailed evaluation results:

```
--- Evaluating llama-3-8b ---
✅ tinyMMLU accuracy: 65.2% (326/500)
📊 tinyMMLU average response tokens/sec: 42.5

✅ tinyGSM8k accuracy: 73.8% (369/500) 
📊 tinyGSM8k average response tokens/sec: 38.2

📊 Took 4m 23.45s to evaluate llama-3-8b
📊 Overall accuracy: 69.5%
📊 Overall average response tokens per second: 40.35
📄 Results saved to evaluation_summary.csv
```

### CSV Output
Results are automatically saved to `evaluation_summary.csv`:

| dataset | correct | total | skipped | accuracy | tok_per_sec |
|---------|---------|-------|---------|----------|-------------|
| tinyMMLU | 326 | 500 | 0 | 65.2 | 42.5 |
| tinyGSM8k | 369 | 500 | 0 | 73.8 | 38.2 |

## 🧪 Testing

StudioEval includes a comprehensive test suite:

```bash
# Run all tests
pytest

# Run specific test categories  
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m e2e           # End-to-end tests only

# Run with coverage
pytest --cov --cov-report=html
```

## 🔧 Development

### Adding New Datasets

1. Create a new dataset loader in `eval_datasets/custom/` or `eval_datasets/huggingface/`
2. Implement the required interface methods
3. Add the dataset to `default_evals.py` if desired
4. Write tests for the new dataset

### Extending Model Support

1. Implement the `ModelClient` interface in `implementations/`
2. Update the `ModelManager` to support the new client
3. Add configuration options as needed

### Version Management

```bash
# Check current version
python studioeval.py --version
# or programmatically:
python -c "from __version__ import __version__; print(__version__)"

# Quick version bump (version only)
python scripts/bump_version.py patch   # 0.5.0 → 0.5.1
python scripts/bump_version.py minor   # 0.5.0 → 0.6.0  
python scripts/bump_version.py major   # 0.5.0 → 1.0.0

# Full release (version + changelog)
python scripts/release.py patch        # Updates version + CHANGELOG.md
```

Versioning follows [Semantic Versioning](https://semver.org):
- **Major**: Breaking changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

### Changelog Management

**Manual Process (Recommended)**: Update `CHANGELOG.md` as you develop:

```markdown
## [Unreleased]

### Added
- New dataset support for CustomEval
- Enhanced error handling for timeouts

### Fixed
- Bug in model selection when using glob patterns

### Changed
- Improved performance for large datasets
```

**Release Process**:
1. Update changelog throughout development
2. Use `scripts/release.py` to prepare release (auto-moves `[Unreleased]` to version)
3. Review changes with `git diff`
4. Commit, tag, and push

## 🎯 Parameter Precedence

StudioEval uses a clear parameter precedence system:

1. **Explicit CLI arguments** (highest priority)
2. **Dataset configuration values** (medium priority)  
3. **CLI defaults** (lowest priority)

Example:
```bash
# CLI --sample-size 100 overrides everything
python studioeval.py --model "test" --sample-size 100 --datasets-config config.json

# Config values override CLI defaults when CLI not explicit
python studioeval.py --model "test" --datasets-config config.json  # Uses config sample_size
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for your changes
4. Ensure all tests pass
5. Submit a pull request

## 📜 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for pioneering comprehensive LLM evaluation and inspiring this project
- [LM Studio](https://lmstudio.ai) for providing excellent local LLM inference
- [Hugging Face](https://huggingface.co) for dataset access and tools
- The open source LLM community for inspiration and datasets
