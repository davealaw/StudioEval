# Changelog

All notable changes to StudioEval will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 

### Changed
- 

### Fixed
- 

## [0.5.0] - 2025-01-10

### Added
- Initial release of StudioEval
- Comprehensive evaluation framework for LLMs via LM Studio
- Support for multiple dataset types (HuggingFace and custom)
- CLI with flexible model selection and filtering options
- JSON-based configuration system for datasets and communication
- Parameter precedence system (CLI → config → defaults)
- Robust error handling and logging
- Comprehensive test suite (unit, integration, e2e)
- CSV output for evaluation results
- Version management system with `--version` support
- Documentation and examples

### Datasets Supported
- **HuggingFace**: ARC, GSM8K, MMLU, CommonSenseQA, LogiQA, TruthfulQA
- **Custom**: Grammar, Math, Custom MCQ

### Features
- Clean modular architecture with separation of concerns
- Mock client for testing without LM Studio dependency
- Advanced model selection (glob patterns, architecture filters)
- Performance metrics and timing statistics
- Reproducible evaluations with seed support

[Unreleased]: https://github.com/your-org/studioeval/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/your-org/studioeval/releases/tag/v0.5.0
