"""
Integration tests for dataset loading and evaluation pipeline.
Tests the interaction between dataset registry, evaluators, and data loading.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.dataset_registry import DatasetRegistry
from core.evaluator import EvaluationOrchestrator
from core.model_manager import ModelManager
from implementations.mock_client import MockModelClient
from tests.fixtures.mock_datasets import MockDatasetLoader


@pytest.fixture
def dataset_pipeline_client():
    """Mock client optimized for dataset pipeline testing."""
    responses = {
        "test-model": "Answer: B",
        "grammar-model": "This sentence is grammatically correct.",
        "math-model": "#### 42",
        "mcq-model": "B",
    }
    return MockModelClient(
        responses=responses,
        available_models=["test-model", "grammar-model", "math-model", "mcq-model"],
        server_running=True,
    )


class TestDatasetRegistryIntegration:
    """Test dataset registry integration with evaluators."""

    def test_registry_evaluator_mapping(self):
        """Test that registry correctly maps types to evaluators."""
        registry = DatasetRegistry()

        # Test all supported types have evaluators
        supported_types = registry.list_supported_types()
        for eval_type in supported_types:
            evaluator = registry.get_evaluator(eval_type)
            assert callable(evaluator), f"Evaluator for {eval_type} should be callable"

    @patch("eval_datasets.custom.grammar.query_model")
    @patch("eval_datasets.custom.grammar.load_json_dataset_with_config")
    def test_registry_evaluation_dispatch(
        self, mock_load_dataset, mock_query_model, dataset_pipeline_client
    ):
        """Test evaluation dispatch through registry."""
        registry = DatasetRegistry()

        fixtures_dir = Path(__file__).resolve().parent.parent / "fixtures" / "data"
        dataset_path = fixtures_dir / "mock_grammar_dataset.jsonl"
        mock_load_dataset.return_value = [
            {
                "question": "Fix: he go to store.",
                "answer": "Corrected: He goes to the store.",
            }
        ]
        mock_query_model.return_value = (
            "Corrected: Test",
            {"tokens_per_second": 10.0},
        )

        # Dispatch should use the real client but still complete without
        # hanging.
        result = registry.evaluate_dataset(
            eval_type="grammar",
            model_id="test-model",
            dataset_path=str(dataset_path),
            dataset_name="test_grammar",
        )

        # The evaluation should complete and return a result (even with server errors)
        assert result is not None
        assert "dataset" in result
        assert "accuracy" in result
        # Verify the dispatch worked by checking expected result structure
        assert result["dataset"] == "test_grammar"

    def test_registry_unknown_type_handling(self):
        """Test registry handling of unknown evaluation types."""
        registry = DatasetRegistry()

        with pytest.raises(ValueError, match="Unknown eval_type"):
            registry.get_evaluator("definitely_unknown_type")

    def test_registry_with_invalid_parameters(self, dataset_pipeline_client):
        """Test registry evaluation with invalid parameters."""
        registry = DatasetRegistry()

        # Test with missing required parameters
        with pytest.raises(TypeError):
            registry.evaluate_dataset(
                eval_type="grammar",
                # Missing model_id, dataset_path, dataset_name
            )


class TestDatasetLoadingIntegration:
    """Test integration between dataset loading and evaluation."""

    def test_dataset_loading_grammar_integration(self, dataset_pipeline_client):
        """Test grammar dataset loading and evaluation integration."""
        expected_result = {
            "dataset": "grammar_test",
            "correct": 8,
            "total": 10,
            "skipped": 1,
            "accuracy": 80.0,
            "tok_per_sec": 12.5,
        }

        # Mock the registry's evaluate_dataset method
        with patch.object(DatasetRegistry, "evaluate_dataset") as mock_eval:
            mock_eval.return_value = expected_result

            orchestrator = EvaluationOrchestrator(
                model_manager=ModelManager(dataset_pipeline_client),
                dataset_registry=DatasetRegistry(),
            )

            result = orchestrator._evaluate_single_dataset(
                "grammar-model",
                {
                    "eval_type": "grammar",
                    "dataset_path": "test.jsonl",
                    "dataset_name": "grammar_test",
                },
            )

        assert result is not None
        assert result["dataset"] == "grammar_test"
        mock_eval.assert_called_once()

    def test_dataset_loading_mcq_integration(self, dataset_pipeline_client):
        """Test MCQ dataset loading and evaluation integration."""
        expected_result = {
            "dataset": "mcq_test",
            "correct": 7,
            "total": 10,
            "skipped": 0,
            "accuracy": 70.0,
            "tok_per_sec": 11.0,
        }

        # Mock the registry's evaluate_dataset method
        with patch.object(DatasetRegistry, "evaluate_dataset") as mock_eval:
            mock_eval.return_value = expected_result

            orchestrator = EvaluationOrchestrator(
                model_manager=ModelManager(dataset_pipeline_client),
                dataset_registry=DatasetRegistry(),
            )

            result = orchestrator._evaluate_single_dataset(
                "mcq-model",
                {
                    "eval_type": "custom_mcq",
                    "dataset_path": "test_mcq.jsonl",
                    "dataset_name": "mcq_test",
                },
            )

        assert result is not None
        assert result["dataset"] == "mcq_test"
        mock_eval.assert_called_once()

    def test_dataset_loading_math_integration(self, dataset_pipeline_client):
        """Test math dataset loading and evaluation integration."""
        expected_result = {
            "dataset": "gsm8k_test",
            "correct": 6,
            "total": 8,
            "skipped": 2,
            "accuracy": 75.0,
            "tok_per_sec": 10.0,
        }

        # Mock the registry's evaluate_dataset method
        with patch.object(DatasetRegistry, "evaluate_dataset") as mock_eval:
            mock_eval.return_value = expected_result

            orchestrator = EvaluationOrchestrator(
                model_manager=ModelManager(dataset_pipeline_client),
                dataset_registry=DatasetRegistry(),
            )

            result = orchestrator._evaluate_single_dataset(
                "math-model",
                {
                    "eval_type": "gsm8k",
                    "dataset_path": "gsm8k.jsonl",
                    "dataset_name": "gsm8k_test",
                },
            )

        assert result is not None
        assert result["dataset"] == "gsm8k_test"
        mock_eval.assert_called_once()

    def test_dataset_loading_failure_handling(self, dataset_pipeline_client):
        """Test handling of dataset loading failures."""
        # Mock the registry's evaluate_dataset method to simulate failure
        with patch.object(DatasetRegistry, "evaluate_dataset") as mock_eval:
            mock_eval.side_effect = FileNotFoundError("Dataset file not found")

            orchestrator = EvaluationOrchestrator(
                model_manager=ModelManager(dataset_pipeline_client),
                dataset_registry=DatasetRegistry(),
            )

            result = orchestrator._evaluate_single_dataset(
                "test-model",
                {
                    "eval_type": "grammar",
                    "dataset_path": "nonexistent.jsonl",
                    "dataset_name": "missing_test",
                },
            )

        # Should handle loading failure gracefully
        assert result is None
        mock_eval.assert_called_once()


class TestModelDatasetPipelineIntegration:
    """Test full pipeline integration between models and datasets."""

    @patch("models.model_handling.query_model")
    @patch("utils.data_loading.load_json_dataset_with_config")
    def test_model_switching_between_datasets(
        self, mock_load, mock_query_model, dataset_pipeline_client
    ):
        """Test model loading/unloading when switching between datasets."""

        def dataset_side_effect(dataset_name, file_path=None):
            if "grammar" in file_path:
                return MockDatasetLoader.MOCK_DATA["grammar"]
            elif "math" in file_path:
                return MockDatasetLoader.MOCK_DATA["math"]
            return []

        mock_load.side_effect = dataset_side_effect
        mock_query_model.return_value = ("Test response", {"tokens_per_second": 12.0})

        orchestrator = EvaluationOrchestrator(
            model_manager=ModelManager(dataset_pipeline_client),
            dataset_registry=DatasetRegistry(),
        )

        # Track model loading calls
        load_calls = []
        unload_calls = []

        original_load = dataset_pipeline_client.load_model
        original_unload = dataset_pipeline_client.unload_model

        def track_load(model_id):
            load_calls.append(model_id)
            return original_load(model_id)

        def track_unload(model_id=None):
            unload_calls.append(model_id)
            return original_unload(model_id)

        dataset_pipeline_client.load_model = track_load
        dataset_pipeline_client.unload_model = track_unload

        # Evaluate multiple datasets with same model
        datasets = [
            {
                "eval_type": "grammar",
                "dataset_path": "grammar.jsonl",
                "dataset_name": "grammar_test",
            },
            {
                "eval_type": "gsm8k",
                "dataset_path": "math.jsonl",
                "dataset_name": "math_test",
            },
        ]

        with (
            patch(
                "eval_datasets.custom.grammar.evaluate_grammar_dataset"
            ) as mock_grammar,
            patch(
                "eval_datasets.huggingface.gsm8k.evaluate_gsm8k_dataset"
            ) as mock_math,
        ):
            mock_grammar.return_value = {"dataset": "grammar_test", "accuracy": 80.0}
            mock_math.return_value = {"dataset": "math_test", "accuracy": 70.0}

            result = orchestrator.run_evaluation(
                model="test-model", datasets_config=datasets
            )

        assert result is True

        # Should load model once and unload once
        assert len(load_calls) == 1
        assert load_calls[0] == "test-model"
        assert len(unload_calls) == 1
        assert unload_calls[0] == "test-model"

    @patch("models.model_handling.query_model")
    @patch("utils.data_loading.load_json_dataset_with_config")
    def test_multiple_models_same_dataset(
        self, mock_load, mock_query_model, dataset_pipeline_client
    ):
        """Test multiple models evaluating same dataset."""
        mock_load.return_value = MockDatasetLoader.MOCK_DATA["grammar"]
        mock_query_model.return_value = ("Test response", {"tokens_per_second": 12.0})

        orchestrator = EvaluationOrchestrator(
            model_manager=ModelManager(dataset_pipeline_client),
            dataset_registry=DatasetRegistry(),
        )

        # Track model operations
        model_operations = []

        # Store original methods
        original_load = dataset_pipeline_client.load_model
        original_unload = dataset_pipeline_client.unload_model

        def track_load(model_id):
            model_operations.append(("load", model_id))
            return original_load(model_id)

        def track_unload(model_id=None):
            model_operations.append(("unload", model_id))
            return original_unload(model_id)

        dataset_pipeline_client.load_model = track_load
        dataset_pipeline_client.unload_model = track_unload

        with patch(
            "eval_datasets.custom.grammar.evaluate_grammar_dataset"
        ) as mock_eval:
            mock_eval.return_value = {"dataset": "grammar_test", "accuracy": 90.0}

            result = orchestrator.run_evaluation(
                models=["grammar-model", "test-model"],
                datasets_config=[
                    {
                        "eval_type": "grammar",
                        "dataset_path": "grammar.jsonl",
                        "dataset_name": "grammar_test",
                    }
                ],
            )

        assert result is True

        # Verify proper model loading/unloading sequence
        load_operations = [op for op in model_operations if op[0] == "load"]
        unload_operations = [op for op in model_operations if op[0] == "unload"]

        assert len(load_operations) == 2  # Two models loaded
        assert len(unload_operations) == 2  # Two models unloaded


class TestEvaluationResultPipelineIntegration:
    """Test integration of evaluation results through the pipeline."""

    @patch("eval_datasets.custom.grammar.query_model")
    @patch("eval_datasets.custom.grammar.load_json_dataset_with_config")
    def test_result_aggregation_pipeline(
        self, mock_load, mock_query, dataset_pipeline_client
    ):
        """Test that results are properly aggregated through pipeline."""
        mock_load.return_value = MockDatasetLoader.MOCK_DATA["grammar"]
        mock_query.return_value = ("correct", {"tokens_per_second": 10.0})

        expected_result = {
            "dataset": "grammar_test",
            "model": "grammar-model",
            "correct": 8,
            "total": 10,
            "skipped": 1,
            "accuracy": 80.0,
            "tok_per_sec": 12.5,
            "eval_time": 5.2,
        }

        # Mock the registry's evaluate_dataset method to return our expected result
        with patch.object(DatasetRegistry, "evaluate_dataset") as mock_registry_eval:
            mock_registry_eval.return_value = expected_result

            orchestrator = EvaluationOrchestrator(
                model_manager=ModelManager(dataset_pipeline_client),
                dataset_registry=DatasetRegistry(),
            )

            result = orchestrator._evaluate_single_dataset(
                "grammar-model",
                {
                    "eval_type": "grammar",
                    "dataset_path": "grammar.jsonl",
                    "dataset_name": "grammar_test",
                },
            )

        # Verify result propagation
        assert result == expected_result
        assert result["accuracy"] == 80.0
        assert result["tok_per_sec"] == 12.5
        mock_registry_eval.assert_called_once()

    def test_result_validation_pipeline(self, dataset_pipeline_client):
        """Test that invalid results are handled in pipeline."""
        # Mock evaluator returning invalid result
        invalid_result = {"incomplete": "result"}

        # Mock the registry's evaluate_dataset method
        with patch.object(DatasetRegistry, "evaluate_dataset") as mock_eval:
            mock_eval.return_value = invalid_result

            orchestrator = EvaluationOrchestrator(
                model_manager=ModelManager(dataset_pipeline_client),
                dataset_registry=DatasetRegistry(),
            )

            result = orchestrator._evaluate_single_dataset(
                "grammar-model",
                {
                    "eval_type": "grammar",
                    "dataset_path": "grammar.jsonl",
                    "dataset_name": "grammar_test",
                },
            )

        # Should still return the result even if incomplete
        # (validation happens at display/output level)
        assert result == invalid_result
        mock_eval.assert_called_once()


class TestPipelineConfigurationIntegration:
    """Test integration of configuration loading with pipeline execution."""

    def test_datasets_config_file_integration(self, dataset_pipeline_client, tmp_path):
        """Test loading datasets config from file and executing pipeline."""
        # Create config file
        config_data = [
            {
                "eval_type": "grammar",
                "dataset_path": "grammar.jsonl",
                "dataset_name": "grammar_benchmark",
            },
            {
                "eval_type": "custom_mcq",
                "dataset_path": "logic.jsonl",
                "dataset_name": "logic_benchmark",
            },
        ]

        config_file = tmp_path / "datasets.json"
        config_file.write_text(json.dumps(config_data, indent=2))

        orchestrator = EvaluationOrchestrator(
            model_manager=ModelManager(dataset_pipeline_client),
            dataset_registry=DatasetRegistry(),
        )

        # Test config loading
        loaded_config = orchestrator._load_datasets_config(str(config_file))

        assert loaded_config == config_data
        assert len(loaded_config) == 2
        assert loaded_config[0]["eval_type"] == "grammar"
        assert loaded_config[1]["eval_type"] == "custom_mcq"

    def test_default_datasets_config_integration(self, dataset_pipeline_client):
        """Test integration with default datasets configuration."""
        orchestrator = EvaluationOrchestrator(
            model_manager=ModelManager(dataset_pipeline_client),
            dataset_registry=DatasetRegistry(),
        )

        # Mock default datasets
        mock_default = [
            {
                "eval_type": "grammar",
                "dataset_path": "default_grammar.jsonl",
                "dataset_name": "default_grammar",
            }
        ]

        with patch("core.evaluator.default_datasets_to_run", mock_default):
            config = orchestrator._load_datasets_config()

            assert config == mock_default
            assert config[0]["dataset_name"] == "default_grammar"

    def test_invalid_config_pipeline_integration(
        self, dataset_pipeline_client, tmp_path
    ):
        """Test pipeline behavior with invalid configuration."""
        # Create invalid config file
        invalid_config = tmp_path / "invalid.json"
        invalid_config.write_text('{"invalid": json, "syntax"}')

        orchestrator = EvaluationOrchestrator(
            model_manager=ModelManager(dataset_pipeline_client),
            dataset_registry=DatasetRegistry(),
        )

        # Should raise JSON decode error
        with pytest.raises(json.JSONDecodeError):
            orchestrator._load_datasets_config(str(invalid_config))
