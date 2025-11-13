"""
Unit tests for DatasetRegistry functionality.
Tests the registry that manages dataset evaluation types and their implementations.
"""

from unittest.mock import Mock, patch

import pytest

from core.dataset_registry import DatasetRegistry


class TestDatasetRegistryInitialization:
    """Test DatasetRegistry initialization."""

    def test_registry_initializes_with_evaluators(self):
        """Test that registry initializes with expected evaluators."""
        registry = DatasetRegistry()

        expected_types = [
            "grammar",
            "custom_mcq",
            "math",
            "gsm8k",
            "arc",
            "mmlu",
            "commonsenseqa",
            "logiqa",
            "truthfulqa",
        ]

        for eval_type in expected_types:
            assert eval_type in registry._evaluators
            assert callable(registry._evaluators[eval_type])

    def test_registry_evaluators_are_functions(self):
        """Test that all registered evaluators are callable."""
        registry = DatasetRegistry()

        for eval_type, evaluator in registry._evaluators.items():
            assert callable(evaluator), f"Evaluator for {eval_type} is not callable"


class TestDatasetRegistryGetters:
    """Test DatasetRegistry getter methods."""

    def test_get_evaluator_valid_type(self):
        """Test getting evaluator for valid type."""
        registry = DatasetRegistry()

        evaluator = registry.get_evaluator("grammar")
        assert callable(evaluator)

        # Test another type
        evaluator = registry.get_evaluator("arc")
        assert callable(evaluator)

    def test_get_evaluator_invalid_type(self):
        """Test getting evaluator for invalid type raises ValueError."""
        registry = DatasetRegistry()

        with pytest.raises(ValueError, match="Unknown eval_type: invalid_type"):
            registry.get_evaluator("invalid_type")

    def test_get_evaluator_case_sensitive(self):
        """Test that evaluator lookup is case sensitive."""
        registry = DatasetRegistry()

        with pytest.raises(ValueError, match="Unknown eval_type: GRAMMAR"):
            registry.get_evaluator("GRAMMAR")

    def test_list_supported_types(self):
        """Test listing supported evaluation types."""
        registry = DatasetRegistry()

        supported = registry.list_supported_types()
        assert isinstance(supported, list)
        assert len(supported) > 0

        # Check some expected types are present
        assert "grammar" in supported
        assert "custom_mcq" in supported
        assert "arc" in supported

    def test_list_supported_types_returns_copy(self):
        """Test that list_supported_types returns a copy, not reference."""
        registry = DatasetRegistry()

        supported1 = registry.list_supported_types()
        supported2 = registry.list_supported_types()

        # Modify one list
        supported1.append("fake_type")

        # Other list should be unchanged
        assert "fake_type" not in supported2

    def test_is_supported_valid_types(self):
        """Test is_supported for valid types."""
        registry = DatasetRegistry()

        assert registry.is_supported("grammar") is True
        assert registry.is_supported("custom_mcq") is True
        assert registry.is_supported("arc") is True
        assert registry.is_supported("mmlu") is True

    def test_is_supported_invalid_types(self):
        """Test is_supported for invalid types."""
        registry = DatasetRegistry()

        assert registry.is_supported("invalid_type") is False
        assert registry.is_supported("GRAMMAR") is False  # Case sensitive
        assert registry.is_supported("") is False
        assert registry.is_supported("arc_challenge") is False


class TestDatasetRegistryEvaluation:
    """Test DatasetRegistry evaluation functionality."""

    def test_evaluate_dataset_calls_correct_evaluator(self):
        """Test that evaluate_dataset calls the correct evaluator function."""
        registry = DatasetRegistry()

        # Mock the grammar evaluator
        mock_evaluator = Mock(return_value={"accuracy": 85.0})
        registry._evaluators["grammar"] = mock_evaluator

        # Call evaluate_dataset
        result = registry.evaluate_dataset(
            eval_type="grammar",
            model_id="test_model",
            dataset_path="test_path.jsonl",
            dataset_name="test_grammar",
            seed=42,
        )

        # Verify the evaluator was called with correct parameters
        mock_evaluator.assert_called_once_with(
            "test_model", "test_path.jsonl", dataset_name="test_grammar", seed=42
        )
        assert result == {"accuracy": 85.0}

    def test_evaluate_dataset_passes_kwargs(self):
        """Test that evaluate_dataset passes additional kwargs."""
        registry = DatasetRegistry()

        mock_evaluator = Mock(return_value={"total": 100})
        registry._evaluators["custom_mcq"] = mock_evaluator

        registry.evaluate_dataset(
            eval_type="custom_mcq",
            model_id="test_model",
            dataset_path="test.jsonl",
            dataset_name="test_mcq",
            sample_size=50,
            timeout=120,
        )

        mock_evaluator.assert_called_once_with(
            "test_model",
            "test.jsonl",
            dataset_name="test_mcq",
            sample_size=50,
            timeout=120,
        )

    def test_evaluate_dataset_invalid_type_raises_error(self):
        """Test that evaluate_dataset with invalid type raises ValueError."""
        registry = DatasetRegistry()

        with pytest.raises(ValueError, match="Unknown eval_type: invalid"):
            registry.evaluate_dataset(
                eval_type="invalid",
                model_id="test_model",
                dataset_path="test.jsonl",
                dataset_name="test",
            )

    def test_evaluate_dataset_evaluator_exception_propagates(self):
        """Test that exceptions from evaluators are propagated."""
        registry = DatasetRegistry()

        # Mock evaluator that raises exception
        mock_evaluator = Mock(side_effect=RuntimeError("Evaluation failed"))
        registry._evaluators["grammar"] = mock_evaluator

        with pytest.raises(RuntimeError, match="Evaluation failed"):
            registry.evaluate_dataset(
                eval_type="grammar",
                model_id="test_model",
                dataset_path="test.jsonl",
                dataset_name="test",
            )


class TestDatasetRegistryIntegration:
    """Integration-style tests for DatasetRegistry."""

    def test_all_registered_evaluators_importable(self):
        """Test that all registered evaluators can be imported and called."""
        registry = DatasetRegistry()

        # This tests that the imports in dataset_registry.py work correctly
        for eval_type in registry.list_supported_types():
            evaluator = registry.get_evaluator(eval_type)
            assert callable(evaluator)

            # Test that the function has the expected signature
            # (all evaluators should accept model_id as first param)
            import inspect

            sig = inspect.signature(evaluator)
            params = list(sig.parameters.keys())
            assert len(params) > 0, f"Evaluator {eval_type} has no parameters"
            assert (
                params[0] == "model_id"
            ), f"Evaluator {eval_type} first param is not 'model_id'"

    @patch("core.dataset_registry.evaluate_grammar_dataset")
    def test_registry_uses_actual_imports(self, mock_grammar_eval):
        """Test that registry uses actual imported functions."""
        mock_grammar_eval.return_value = {"test": "result"}

        registry = DatasetRegistry()
        result = registry.evaluate_dataset(
            eval_type="grammar",
            model_id="test_model",
            dataset_path="test.jsonl",
            dataset_name="test",
        )

        # Verify the actual imported function was called
        mock_grammar_eval.assert_called_once_with(
            "test_model", "test.jsonl", dataset_name="test"
        )
        assert result == {"test": "result"}


class TestDatasetRegistryEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string_eval_type(self):
        """Test handling of empty string eval_type."""
        registry = DatasetRegistry()

        with pytest.raises(ValueError, match="Unknown eval_type: "):
            registry.get_evaluator("")

    def test_none_eval_type(self):
        """Test handling of None eval_type."""
        registry = DatasetRegistry()

        with pytest.raises((ValueError, TypeError)):
            registry.get_evaluator(None)

    def test_whitespace_eval_type(self):
        """Test handling of whitespace-only eval_type."""
        registry = DatasetRegistry()

        with pytest.raises(ValueError):
            registry.get_evaluator("   ")

    def test_evaluate_dataset_with_no_kwargs(self):
        """Test evaluate_dataset with minimal parameters."""
        registry = DatasetRegistry()

        mock_evaluator = Mock(return_value={})
        registry._evaluators["math"] = mock_evaluator

        registry.evaluate_dataset(
            eval_type="math",
            model_id="test_model",
            dataset_path="test.jsonl",
            dataset_name="test_math",
        )

        mock_evaluator.assert_called_once_with(
            "test_model", "test.jsonl", dataset_name="test_math"
        )
