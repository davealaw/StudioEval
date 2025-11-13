"""
Test error handling for configuration file issues.
Specifically tests the scenario where an invalid config file path is provided.
"""

import json

import pytest

from core.dataset_registry import DatasetRegistry
from core.evaluator import EvaluationOrchestrator
from core.model_manager import ModelManager
from implementations.mock_client import MockModelClient


@pytest.fixture
def test_mock_client():
    """Mock client for error handling tests."""
    return MockModelClient(
        responses={"test-model": "Answer: A"},
        available_models=["test-model"],
        server_running=True,
    )


@pytest.fixture
def orchestrator(test_mock_client):
    """Orchestrator with mock dependencies."""
    return EvaluationOrchestrator(
        model_manager=ModelManager(test_mock_client), dataset_registry=DatasetRegistry()
    )


class TestConfigFileErrorHandling:
    """Test error handling for configuration file issues."""

    def test_nonexistent_config_file_graceful_handling(self, orchestrator, caplog):
        """Nonexistent config files should trigger fail-fast handling."""
        # This should NOT raise an exception but handle it gracefully
        result = orchestrator.run_evaluation(
            model="test-model", datasets_config="nonexistent_config.json"
        )

        # Should fail fast when config is invalid (correct behavior)
        assert result is False

        # Should log appropriate error messages
        assert "Configuration file not found: nonexistent_config.json" in caplog.text
        assert "Configuration validation failed" in caplog.text
        assert "Cannot proceed with evaluation" in caplog.text

    def test_malformed_json_config_file_graceful_handling(
        self, orchestrator, caplog, tmp_path
    ):
        """Malformed JSON config files should trigger fail-fast handling."""
        # Create malformed JSON file
        malformed_config = tmp_path / "malformed.json"
        malformed_config.write_text('{"invalid": json, "missing": quotes}')

        result = orchestrator.run_evaluation(
            model="test-model", datasets_config=str(malformed_config)
        )

        # Should fail fast when JSON is invalid (correct behavior)
        assert result is False

        # Should log appropriate error messages
        assert "Failed to parse JSON configuration file" in caplog.text
        assert "Configuration validation failed" in caplog.text
        assert "Cannot proceed with evaluation" in caplog.text

    def test_valid_config_file_works_normally(self, orchestrator, tmp_path):
        """Test that valid config files work normally."""
        # Create valid config file
        config_data = [
            {
                "eval_type": "grammar",
                "dataset_path": "test.jsonl",
                "dataset_name": "test_grammar",
            }
        ]

        valid_config = tmp_path / "valid.json"
        valid_config.write_text(json.dumps(config_data, indent=2))

        # Mock the dataset evaluation to avoid actual file I/O
        from unittest.mock import patch

        with patch(
            "core.dataset_registry.DatasetRegistry.evaluate_dataset"
        ) as mock_eval:
            mock_eval.return_value = {
                "dataset": "test_grammar",
                "correct": 1,
                "total": 1,
                "skipped": 0,
                "accuracy": 100.0,
                "tok_per_sec": 10.0,
            }

            result = orchestrator.run_evaluation(
                model="test-model", datasets_config=str(valid_config)
            )

            assert result is True
            mock_eval.assert_called_once()

    def test_empty_config_file_handling(self, orchestrator, tmp_path):
        """Test handling of empty config files with fail-fast behavior."""
        empty_config = tmp_path / "empty.json"
        empty_config.write_text("")

        result = orchestrator.run_evaluation(
            model="test-model", datasets_config=str(empty_config)
        )

        # Should fail fast when config is empty (correct behavior)
        assert result is False

    def test_config_with_permission_error(self, orchestrator, tmp_path):
        """Test handling of config files with permission issues."""
        import os

        # Create config file and remove read permissions (on Unix systems)
        restricted_config = tmp_path / "restricted.json"
        restricted_config.write_text('{"test": "data"}')

        try:
            # Remove read permissions
            os.chmod(str(restricted_config), 0o000)

            result = orchestrator.run_evaluation(
                model="test-model", datasets_config=str(restricted_config)
            )

            # Fail fast when permissions prevent reading config (expected behavior)
            assert result is False

        finally:
            # Restore permissions for cleanup
            os.chmod(str(restricted_config), 0o644)

    def test_model_unloading_after_config_error(self, test_mock_client, orchestrator):
        """Test that models are not loaded when config validation fails early."""
        # Track model loading/unloading
        load_calls = []
        unload_calls = []

        original_load = test_mock_client.load_model
        original_unload = test_mock_client.unload_model

        def mock_load(model_id):
            load_calls.append(model_id)
            return original_load(model_id)

        def mock_unload(model_id=None):
            unload_calls.append(model_id)
            return original_unload(model_id)

        test_mock_client.load_model = mock_load
        test_mock_client.unload_model = mock_unload

        # Run evaluation with nonexistent config
        result = orchestrator.run_evaluation(
            model="test-model", datasets_config="nonexistent.json"
        )

        # Should fail fast before any model loading
        assert result is False

        # Verify that no models were loaded (config validation happens first)
        assert len(load_calls) == 0, "No models should be loaded when config is invalid"
        assert (
            len(unload_calls) == 0
        ), "No models should need unloading when none were loaded"

    def test_multiple_models_with_config_error(self, orchestrator, caplog):
        """Test that config errors prevent all model evaluation (fail-fast behavior)."""
        # Add more models to the mock client
        orchestrator.model_manager.client.available_models.extend(
            ["model-2", "model-3"]
        )
        orchestrator.model_manager.client.responses.update(
            {"model-2": "Answer: B", "model-3": "Answer: C"}
        )

        result = orchestrator.run_evaluation(
            all_models=True, datasets_config="nonexistent.json"
        )

        # Should fail fast for all models when config is invalid (correct behavior)
        assert result is False

        # Should log single configuration validation failure (fail-fast)
        assert "Configuration validation failed" in caplog.text
        assert "Cannot proceed with evaluation of 3 model(s)" in caplog.text
