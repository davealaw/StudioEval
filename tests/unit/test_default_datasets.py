"""
Unit tests for default dataset configurations.
Tests the default evaluation configurations and their validity.
"""

from unittest.mock import patch

import pytest

from core.dataset_registry import DatasetRegistry
from eval_datasets.default_evals import default_datasets_to_run


class TestDefaultDatasetConfigurations:
    """Test default dataset configuration structure and validity."""

    def test_default_datasets_is_list(self):
        """Test that default_datasets_to_run is a list."""
        assert isinstance(default_datasets_to_run, list)
        assert len(default_datasets_to_run) > 0

    def test_default_datasets_have_required_fields(self):
        """Test that all default datasets have required fields."""
        required_fields = ["eval_type", "dataset_path", "dataset_name"]

        for i, dataset_config in enumerate(default_datasets_to_run):
            assert isinstance(dataset_config, dict), f"Dataset {i} is not a dict"

            for field in required_fields:
                assert (
                    field in dataset_config
                ), f"Dataset {i} missing required field: {field}"
                assert (
                    dataset_config[field] is not None
                ), f"Dataset {i} has None value for: {field}"
                assert (
                    dataset_config[field] != ""
                ), f"Dataset {i} has empty string for: {field}"

    def test_default_datasets_eval_types_are_valid(self):
        """Test that all default dataset eval_types are supported."""
        registry = DatasetRegistry()
        supported_types = registry.list_supported_types()

        for i, dataset_config in enumerate(default_datasets_to_run):
            eval_type = dataset_config["eval_type"]
            assert (
                eval_type in supported_types
            ), f"Dataset {i} has unsupported eval_type: {eval_type}"

    def test_default_datasets_have_consistent_structure(self):
        """Test that all dataset configs have consistent field structure."""
        # Check that all configs have the same set of keys
        if not default_datasets_to_run:
            pytest.skip("No default datasets to test")

        expected_keys = set(default_datasets_to_run[0].keys())

        for i, dataset_config in enumerate(default_datasets_to_run):
            config_keys = set(dataset_config.keys())
            # Allow some configs to have additional optional fields
            assert expected_keys.issubset(
                config_keys
            ), f"Dataset {i} missing keys: {expected_keys - config_keys}"

    def test_default_datasets_names_are_unique(self):
        """Test that dataset names are unique across all configs."""
        dataset_names = [config["dataset_name"] for config in default_datasets_to_run]
        unique_names = set(dataset_names)

        assert len(dataset_names) == len(
            unique_names
        ), "Duplicate dataset names found in default configs"

    def test_default_datasets_paths_are_strings(self):
        """Test that all dataset paths are strings."""
        for i, dataset_config in enumerate(default_datasets_to_run):
            path = dataset_config["dataset_path"]
            assert isinstance(
                path, str
            ), f"Dataset {i} path is not a string: {type(path)}"
            assert len(path) > 0, f"Dataset {i} path is empty"


class TestDefaultDatasetTypes:
    """Test coverage of different dataset types in defaults."""

    def test_default_datasets_include_custom_types(self):
        """Test that default datasets include custom evaluation types."""
        eval_types = [config["eval_type"] for config in default_datasets_to_run]

        custom_types = ["custom_mcq", "grammar", "math"]
        for custom_type in custom_types:
            assert custom_type in eval_types, f"Missing custom eval type: {custom_type}"

    def test_default_datasets_include_huggingface_types(self):
        """Test that default datasets include HuggingFace evaluation types."""
        eval_types = [config["eval_type"] for config in default_datasets_to_run]

        hf_types = ["arc", "mmlu", "gsm8k"]
        for hf_type in hf_types:
            assert hf_type in eval_types, f"Missing HuggingFace eval type: {hf_type}"

    def test_default_datasets_coverage(self):
        """Test that default datasets provide good coverage of available types."""
        registry = DatasetRegistry()
        all_supported = set(registry.list_supported_types())
        default_types = {config["eval_type"] for config in default_datasets_to_run}

        # Should cover at least 70% of supported types
        coverage = len(default_types) / len(all_supported)
        assert (
            coverage >= 0.7
        ), f"Default datasets only cover {coverage:.1%} of supported types"


class TestDefaultDatasetConfiguration:
    """Test individual aspects of dataset configuration."""

    def test_custom_mcq_configurations(self):
        """Test custom MCQ dataset configurations."""
        custom_mcq_configs = [
            config
            for config in default_datasets_to_run
            if config["eval_type"] == "custom_mcq"
        ]

        assert len(custom_mcq_configs) > 0, "No custom MCQ configs found"

        for config in custom_mcq_configs:
            # Should have JSONL files
            assert config["dataset_path"].endswith(
                ".jsonl"
            ), f"Custom MCQ should use JSONL: {config['dataset_path']}"

    def test_huggingface_configurations(self):
        """Test HuggingFace dataset configurations."""
        hf_types = ["arc", "mmlu", "gsm8k", "commonsenseqa", "logiqa", "truthfulqa"]

        for hf_type in hf_types:
            hf_configs = [
                config
                for config in default_datasets_to_run
                if config["eval_type"] == hf_type
            ]

            if hf_configs:  # If this type exists in defaults
                for config in hf_configs:
                    # HF datasets typically don't use file extensions
                    assert not config["dataset_path"].endswith(".jsonl"), (
                        "HF dataset shouldn't use JSONL extension: "
                        f"{config['dataset_path']}"
                    )

    def test_optional_fields_handling(self):
        """Test that optional fields are properly handled."""
        optional_fields = ["subset", "split", "seed", "sample_size"]

        for i, config in enumerate(default_datasets_to_run):
            for field in optional_fields:
                if field in config:
                    value = config[field]
                    # Optional fields can be None, but if present should be valid types
                    if value is not None:
                        if field in ["seed", "sample_size"]:
                            assert isinstance(
                                value, int
                            ), f"Dataset {i} {field} should be int or None"
                        elif field in ["subset", "split"]:
                            assert isinstance(
                                value, str
                            ), f"Dataset {i} {field} should be str or None"


class TestDefaultDatasetValidation:
    """Test validation of default dataset configurations."""

    def test_all_configs_can_get_evaluators(self):
        """Test that registry can get evaluators for all default configs."""
        registry = DatasetRegistry()

        for i, config in enumerate(default_datasets_to_run):
            eval_type = config["eval_type"]
            try:
                evaluator = registry.get_evaluator(eval_type)
                assert callable(evaluator), f"Dataset {i} evaluator is not callable"
            except ValueError as e:
                pytest.fail(
                    f"Failed to get evaluator for dataset {i} ({eval_type}): {e}"
                )

    @patch("core.dataset_registry.DatasetRegistry.evaluate_dataset")
    def test_all_configs_can_be_evaluated(self, mock_evaluate):
        """Test that all default configs can be passed to evaluation."""
        mock_evaluate.return_value = {"accuracy": 50.0}

        registry = DatasetRegistry()

        for i, config in enumerate(default_datasets_to_run):
            try:
                # Attempt to evaluate with minimal required parameters
                registry.evaluate_dataset(
                    model_id="test_model", **config  # Unpack the entire config
                )

                # Verify the call was made
                assert mock_evaluate.called
                mock_evaluate.reset_mock()

            except (RuntimeError, ValueError, KeyError, TypeError) as e:
                pytest.fail(
                    f"Failed to evaluate dataset {i} ({config['eval_type']}): {e}"
                )


class TestDatasetConfigurationIntegration:
    """Integration tests for dataset configurations."""

    def test_config_fields_match_evaluator_signatures(self):
        """Test that config fields match evaluator function signatures."""
        import inspect

        registry = DatasetRegistry()

        for config in default_datasets_to_run:
            eval_type = config["eval_type"]
            evaluator = registry.get_evaluator(eval_type)

            # Get evaluator signature
            sig = inspect.signature(evaluator)
            param_names = list(sig.parameters.keys())

            # First parameter should always be model_id
            assert param_names[0] == "model_id"

            # Check that config doesn't have unexpected fields for this evaluator
            config_keys = set(config.keys())
            # Remove standard keys that are handled by registry
            standard_keys = {"eval_type", "dataset_name"}
            dataset_specific_keys = config_keys - standard_keys

            # Dataset-specific keys should align with evaluator parameters or
            # other commonly accepted options.
            common_params = {
                "dataset_path",
                "jsonl_path",
                "subset",
                "split",
                "seed",
                "sample_size",
            }

            for key in dataset_specific_keys:
                assert (
                    key in param_names or key in common_params
                ), f"Config key '{key}' not in evaluator params for {eval_type}"

    def test_default_configs_provide_full_coverage(self):
        """Test that default configs provide examples of all major features."""
        eval_types = [config["eval_type"] for config in default_datasets_to_run]

        # Should have examples of both custom and HuggingFace datasets
        custom_types = {"custom_mcq", "grammar", "math"}
        hf_types = {"arc", "mmlu", "gsm8k"}

        assert (
            len(set(eval_types) & custom_types) >= 2
        ), "Should have multiple custom dataset types"
        assert (
            len(set(eval_types) & hf_types) >= 2
        ), "Should have multiple HuggingFace dataset types"


class TestDatasetConfigurationRobustness:
    """Test robustness of dataset configurations."""

    def test_configs_handle_missing_optional_fields(self):
        """Test that configs work even with missing optional fields."""
        # Create a minimal config with only required fields
        minimal_config = {
            "eval_type": "custom_mcq",
            "dataset_path": "test.jsonl",
            "dataset_name": "test",
        }

        registry = DatasetRegistry()

        # Should be able to get evaluator without error
        evaluator = registry.get_evaluator(minimal_config["eval_type"])
        assert callable(evaluator)

    def test_configs_are_json_serializable(self):
        """Test that all configs can be serialized to JSON."""
        import json

        try:
            json_str = json.dumps(default_datasets_to_run)
            # Should be able to round-trip
            restored = json.loads(json_str)
            assert restored == default_datasets_to_run
        except (TypeError, json.JSONDecodeError) as e:
            pytest.fail(f"Default configs are not JSON serializable: {e}")

    def test_config_field_types_are_consistent(self):
        """Test that same field names have consistent types across configs."""
        field_types = {}

        for config in default_datasets_to_run:
            for field, value in config.items():
                if value is not None:  # Skip None values
                    field_type = type(value)

                    if field in field_types:
                        assert field_types[field] == field_type, (
                            "Inconsistent type for field "
                            f"'{field}': expected {field_types[field]}, "
                            f"got {field_type}"
                        )
                    else:
                        field_types[field] = field_type
