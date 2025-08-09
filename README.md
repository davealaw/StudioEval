# StudioEval

**StudioEval** is a lightweight evaluation framework for benchmarking LLMs using [LM Studio](https://lmstudio.ai) as the model backend. It supports custom GGUF models and works seamlessly with MLX and llama.cpp.

## 🚀 Features

- Modular dataset loaders (custom and Hugging Face)
- Dynamic dataset discovery and invocation
- Supports GGUF, MLX, and other local models via LM Studio API
- Clear scoring and prompt templating utilities
- Easily extensible for new datasets or evaluation strategies

## 📁 Project Structure

```
studioeval/
├── studioeval.py              # Main entry point (formerly llm_lm_eval_lmstudio.py)
├── eval_datasets/             # All dataset loaders
│   ├── loader.py              # Dynamic loading logic
│   ├── custom/
│   └── huggingface/
├── models/                    # Model metadata, endpoint configs, etc.
├── utils/                     # Prompt formatting, scoring, helpers
├── config/                    # Shared constants
├── README.md
└── pyproject.toml
```

## ✅ Getting Started

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install -e .
   ```

3. Run an evaluation:

   ```bash
   python studioeval.py --dataset truthfulqa --model your-model-name
   ```

## 🛠️ License

MIT License
