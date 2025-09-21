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

## [0.7.0] - 2025-01-21

### Added
- **Comprehensive Unit Tests for HellaSwag Evaluator**:
  - Basic evaluation functionality with correct/incorrect model responses  
  - Mixed results handling (partial correct answers)
  - Malformed dataset item detection and skipping (empty context, insufficient endings, invalid labels)
  - Prompt formatting verification
  - Unparseable model response handling
  - Custom parameter passing validation
  - Empty dataset handling
  - Integer and string label format support
  - 9 comprehensive test methods covering all edge cases
- **Comprehensive Unit Tests for WinoGrande Evaluator**:
  - Basic evaluation functionality with binary choice format
  - Wrong answers and mixed results handling
  - Malformed dataset item detection and skipping (empty fields, missing data, invalid answers)
  - Answer format handling for different formats ("1"/"2", "A"/"B", "a"/"b")
  - Prompt formatting verification
  - Unparseable model response handling
  - Custom parameter passing validation
  - Empty dataset handling
  - Binary choice extraction using `extract_mcq_letter`
  - 10 comprehensive test methods covering all edge cases
- **Enhanced Test Coverage**: Both HellaSwag and WinoGrande evaluators now have comprehensive test coverage
- **Test Suite Expansion**: Added 19 new comprehensive test methods across both evaluators

### Fixed
- **Integration Test Compatibility**: Verified all config error handling integration tests work correctly with new fail-fast orchestrator behavior
- **Test Framework Robustness**: All 392 tests now pass consistently with 82% overall test coverage
- **Evaluator Test Coverage Gap**: Filled missing test coverage for HellaSwag and WinoGrande evaluators

### Changed
- **Improved Test Reliability**: All HellaSwag and WinoGrande tests now pass consistently
- **Enhanced Error Handling Testing**: Comprehensive coverage of malformed data, network issues, and parsing failures
- **Test Pattern Consistency**: New tests follow established patterns and conventions from existing evaluator tests

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

[Unreleased]: https://github.com/davealaw/StudioEval/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/davealaw/StudioEval/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/davealaw/StudioEval/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/davealaw/StudioEval/releases/tag/v0.5.0
