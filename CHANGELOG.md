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

## [0.6.0] - 2025-01-11

### Added
- **Comprehensive Unit Tests for CommonsenseQA Evaluator**:
  - Basic evaluation functionality with correct/incorrect model responses
  - Mixed results handling (partial correct answers)
  - Malformed dataset item detection and skipping
  - Unparseable model response handling
  - Custom parameter passing validation
  - Empty dataset handling
  - Prompt formatting verification
  - Extract letterToE (A-E) support validation
- **Comprehensive Unit Tests for LogiQA Evaluator**:
  - Basic evaluation functionality with correct/incorrect model responses
  - Mixed results handling (partial correct answers)
  - Malformed dataset item detection and skipping (invalid option counts, out-of-range indices)
  - Unparseable model response handling
  - Custom parameter passing validation
  - Empty dataset handling
  - Prompt formatting verification
  - Extract letter (A-D) support validation
  - Correct option numeric-to-letter conversion testing
- **Enhanced Test Coverage**: Both evaluators now have 100% test coverage
- **Edge Case Testing**: Comprehensive coverage of malformed data, empty datasets, and error conditions

### Fixed
- **CommonsenseQA KeyError Bug**: Fixed evaluator crash when dataset items are missing the 'choices' key
- **LogiQA Unparseable Response Test**: Fixed test that was incorrectly passing due to accidental letter extraction from mock response text
- **Robust Error Handling**: Both evaluators now gracefully handle and skip malformed dataset items instead of crashing

### Changed
- **Improved Test Reliability**: All CommonsenseQA and LogiQA tests now pass consistently
- **Better Error Recovery**: Enhanced malformed data handling in both evaluators
- Removed 'skip' command line option as no longer relevant

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

[Unreleased]: https://github.com/your-org/studioeval/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/your-org/studioeval/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/your-org/studioeval/releases/tag/v0.5.0
