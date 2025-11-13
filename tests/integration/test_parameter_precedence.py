"""
Test parameter precedence between CLI args and dataset config values.

This tests the bug where dataset-specific config values (like sample_size)
should override global CLI parameters, but currently don't.
"""

import json
from unittest.mock import patch

import pytest

from core.dataset_registry import DatasetRegistry
from core.evaluator import EvaluationOrchestrator
from core.model_manager import ModelManager
from implementations.mock_client import MockModelClient


@pytest.fixture
def mock_client():
    """Mock client for parameter tests."""
    return MockModelClient(
        responses={"test-model": "Answer: A"},
        available_models=["test-model"],
        server_running=True,
    )


@pytest.fixture
def orchestrator(mock_client):
    """Orchestrator with mocked dependencies."""
    return EvaluationOrchestrator(
        model_manager=ModelManager(mock_client), dataset_registry=DatasetRegistry()
    )


class TestParameterPrecedence:
    """Test parameter precedence between config and CLI."""

    def test_config_sample_size_overrides_cli_default(self, orchestrator, tmp_path):
        """Test that dataset config sample_size overrides CLI default."""

        # Create config with specific sample_size for one dataset
        config_data = [
            {
                "eval_type": "mmlu",
                "dataset_path": "tinyBenchmarks/tinyMMLU",
                "dataset_name": "tinyMMLU",
                "split": None,
                "seed": None,
                "sample_size": None,  # Should use CLI default
            },
            {
                "eval_type": "commonsenseqa",
                "dataset_path": "tau/commonsense_qa",
                "dataset_name": "commonsenseqa",
                "split": None,
                "seed": 42,
                "sample_size": 100,  # Should override CLI default
            },
        ]

        config_file = tmp_path / "test_config.json"
        config_file.write_text(json.dumps(config_data, indent=2))

        # Track the parameters passed to evaluate_dataset
        eval_calls = []

        def mock_evaluate_dataset(**kwargs):
            eval_calls.append(kwargs.copy())
            return {
                "dataset": kwargs.get("dataset_name", "unknown"),
                "correct": 1,
                "total": 1,
                "skipped": 0,
                "accuracy": 100.0,
                "tok_per_sec": 10.0,
            }

        with patch.object(
            orchestrator.dataset_registry,
            "evaluate_dataset",
            side_effect=mock_evaluate_dataset,
        ):
            # Run with CLI sample_size=50; config should override where values
            # are provided.
            orchestrator.run_evaluation(
                model="test-model",
                datasets_config=str(config_file),
                sample_size=50,  # CLI default
                seed=123,  # CLI default
            )

        # Verify two datasets were evaluated
        assert len(eval_calls) == 2

        # First dataset (tinyMMLU) should use CLI defaults since config has null
        mmlu_call = next(
            call for call in eval_calls if call["dataset_name"] == "tinyMMLU"
        )
        assert mmlu_call["sample_size"] == 50  # CLI default used (config is null)
        assert mmlu_call["seed"] == 123  # CLI default used (config is null)

        # Second dataset (commonsenseqa) should use config values (CLI not explicit)
        cqa_call = next(
            call for call in eval_calls if call["dataset_name"] == "commonsenseqa"
        )
        assert cqa_call["sample_size"] == 100  # Config overrides CLI default
        assert cqa_call["seed"] == 42  # Config overrides CLI default

    def test_config_null_values_use_cli_defaults(self, orchestrator, tmp_path):
        """Test that null config values fall back to CLI defaults."""

        config_data = [
            {
                "eval_type": "mmlu",
                "dataset_path": "tinyBenchmarks/tinyMMLU",
                "dataset_name": "tinyMMLU",
                "split": None,
                "seed": None,  # Should use CLI default
                "sample_size": None,  # Should use CLI default
            }
        ]

        config_file = tmp_path / "test_config.json"
        config_file.write_text(json.dumps(config_data, indent=2))

        eval_calls = []

        def mock_evaluate_dataset(**kwargs):
            eval_calls.append(kwargs.copy())
            return {
                "dataset": kwargs.get("dataset_name", "unknown"),
                "correct": 1,
                "total": 1,
                "skipped": 0,
                "accuracy": 100.0,
                "tok_per_sec": 10.0,
            }

        with patch.object(
            orchestrator.dataset_registry,
            "evaluate_dataset",
            side_effect=mock_evaluate_dataset,
        ):
            orchestrator.run_evaluation(
                model="test-model",
                datasets_config=str(config_file),
                sample_size=75,
                seed=999,
            )

        assert len(eval_calls) == 1
        call = eval_calls[0]
        assert call["sample_size"] == 75  # CLI default used
        assert call["seed"] == 999  # CLI default used

    def test_config_zero_values_are_preserved(self, orchestrator, tmp_path):
        """Test that zero/falsy config values are preserved, not treated as null."""

        config_data = [
            {
                "eval_type": "mmlu",
                "dataset_path": "tinyBenchmarks/tinyMMLU",
                "dataset_name": "tinyMMLU",
                "split": None,
                "seed": 0,  # Zero should be preserved
                "sample_size": 0,  # Zero should be preserved (means all samples)
            }
        ]

        config_file = tmp_path / "test_config.json"
        config_file.write_text(json.dumps(config_data, indent=2))

        eval_calls = []

        def mock_evaluate_dataset(**kwargs):
            eval_calls.append(kwargs.copy())
            return {
                "dataset": kwargs.get("dataset_name", "unknown"),
                "correct": 1,
                "total": 1,
                "skipped": 0,
                "accuracy": 100.0,
                "tok_per_sec": 10.0,
            }

        with patch.object(
            orchestrator.dataset_registry,
            "evaluate_dataset",
            side_effect=mock_evaluate_dataset,
        ):
            orchestrator.run_evaluation(
                model="test-model",
                datasets_config=str(config_file),
                sample_size=100,  # CLI default
                seed=42,  # CLI default
            )

        assert len(eval_calls) == 1
        call = eval_calls[0]
        assert call["sample_size"] == 0  # Config zero preserved
        assert call["seed"] == 0  # Config zero preserved

    def test_explicit_cli_args_override_config(self, orchestrator, tmp_path):
        """Test that explicitly set CLI args override config values."""

        config_data = [
            {
                "eval_type": "mmlu",
                "dataset_path": "tinyBenchmarks/tinyMMLU",
                "dataset_name": "tinyMMLU",
                "split": None,
                "seed": 42,  # Config value
                "sample_size": 100,  # Config value
            }
        ]

        config_file = tmp_path / "test_config.json"
        config_file.write_text(json.dumps(config_data, indent=2))

        eval_calls = []

        def mock_evaluate_dataset(**kwargs):
            eval_calls.append(kwargs.copy())
            return {
                "dataset": kwargs.get("dataset_name", "unknown"),
                "correct": 1,
                "total": 1,
                "skipped": 0,
                "accuracy": 100.0,
                "tok_per_sec": 10.0,
            }

        with patch.object(
            orchestrator.dataset_registry,
            "evaluate_dataset",
            side_effect=mock_evaluate_dataset,
        ):
            # Simulate explicit CLI args that should override config
            orchestrator.run_evaluation(
                model="test-model",
                datasets_config=str(config_file),
                sample_size=200,  # Should override config value of 100
                seed=999,  # Should override config value of 42
                cli_explicit_args={"sample_size", "seed"},  # These were explicitly set
            )

        assert len(eval_calls) == 1
        call = eval_calls[0]
        assert call["sample_size"] == 200  # Explicit CLI overrides config
        assert call["seed"] == 999  # Explicit CLI overrides config
