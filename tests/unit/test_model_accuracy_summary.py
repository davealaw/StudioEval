"""
Unit tests for model accuracy summary functionality.
"""

import csv
import os
import tempfile
from unittest.mock import Mock

import pytest

from core.evaluator import EvaluationOrchestrator


class TestModelAccuracySummary:
    """Test cases for model accuracy summary CSV generation."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mocked dependencies."""
        mock_model_manager = Mock()
        mock_dataset_registry = Mock()
        return EvaluationOrchestrator(mock_model_manager, mock_dataset_registry)

    @pytest.fixture
    def sample_results(self):
        """Sample evaluation results."""
        return [
            {
                "dataset": "tinyMMLU",
                "correct": 76,
                "total": 100,
                "skipped": 0,
                "accuracy": 76.0,
                "tok_per_sec": 26.02,
            },
            {
                "dataset": "commonsenseqa",
                "correct": 85,
                "total": 100,
                "skipped": 0,
                "accuracy": 85.0,
                "tok_per_sec": 26.17,
            },
        ]

    def test_save_model_accuracy_summary_new_file(self, orchestrator, sample_results):
        """Test creating a new model accuracy summary file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                # Test data
                model_id = "test-model-1"
                elapsed_time = 125.5
                avg_tokens_per_sec = 26.095

                # Call the method
                orchestrator._save_model_accuracy_summary(
                    model_id=model_id,
                    results=sample_results,
                    elapsed_time=elapsed_time,
                    avg_tokens_per_sec=avg_tokens_per_sec,
                    raw_duration=False,
                )

                # Verify file was created
                assert os.path.exists("model_accuracy_summary.csv")

                # Read and verify content
                with open("model_accuracy_summary.csv") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                assert len(rows) == 1
                row = rows[0]

                assert row["model"] == model_id
                assert row["commonsenseqa"] == "85.0"
                assert row["tinyMMLU"] == "76.0"
                assert row["overall"] == "80.5"  # (76+85)/2 = 80.5
                assert "tokens" in row
                assert float(row["tokens"]) == 26.09  # rounded to 26.09

            finally:
                os.chdir(original_cwd)

    def test_save_model_accuracy_summary_update_existing(
        self, orchestrator, sample_results
    ):
        """Test updating existing model accuracy summary file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                # Create initial file with existing model
                initial_data = [
                    ["model", "tinyMMLU", "commonsenseqa", "overall", "time", "tokens"],
                    ["existing-model", "70.0", "80.0", "75.0", "2m 30s", "25.5"],
                ]

                with open("model_accuracy_summary.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(initial_data)

                # Add new model
                model_id = "test-model-2"
                orchestrator._save_model_accuracy_summary(
                    model_id=model_id,
                    results=sample_results,
                    elapsed_time=125.5,
                    avg_tokens_per_sec=26.095,
                    raw_duration=False,
                )

                # Verify both models exist
                with open("model_accuracy_summary.csv") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                assert len(rows) == 2

                # Check existing model is preserved
                existing_model = next(r for r in rows if r["model"] == "existing-model")
                assert existing_model["tinyMMLU"] == "70.0"
                assert existing_model["commonsenseqa"] == "80.0"

                # Check new model was added
                new_model = next(r for r in rows if r["model"] == "test-model-2")
                assert new_model["tinyMMLU"] == "76.0"
                assert new_model["commonsenseqa"] == "85.0"

            finally:
                os.chdir(original_cwd)

    def test_save_model_accuracy_summary_different_datasets(self, orchestrator):
        """Test handling models evaluated on different dataset subsets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                # First model with dataset A and B
                results_1 = [
                    {
                        "dataset": "datasetA",
                        "correct": 80,
                        "total": 100,
                        "accuracy": 80.0,
                        "tok_per_sec": 25.0,
                    },
                    {
                        "dataset": "datasetB",
                        "correct": 70,
                        "total": 100,
                        "accuracy": 70.0,
                        "tok_per_sec": 24.0,
                    },
                ]

                orchestrator._save_model_accuracy_summary(
                    model_id="model-1",
                    results=results_1,
                    elapsed_time=120.0,
                    avg_tokens_per_sec=24.5,
                    raw_duration=True,
                )

                # Second model with dataset B and C
                results_2 = [
                    {
                        "dataset": "datasetB",
                        "correct": 90,
                        "total": 100,
                        "accuracy": 90.0,
                        "tok_per_sec": 26.0,
                    },
                    {
                        "dataset": "datasetC",
                        "correct": 85,
                        "total": 100,
                        "accuracy": 85.0,
                        "tok_per_sec": 27.0,
                    },
                ]

                orchestrator._save_model_accuracy_summary(
                    model_id="model-2",
                    results=results_2,
                    elapsed_time=110.0,
                    avg_tokens_per_sec=26.5,
                    raw_duration=True,
                )

                # Verify dynamic columns are handled correctly
                with open("model_accuracy_summary.csv") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    fieldnames = reader.fieldnames

                assert len(rows) == 2

                # Check that all datasets are columns
                expected_columns = {
                    "model",
                    "datasetA",
                    "datasetB",
                    "datasetC",
                    "overall",
                    "time",
                    "tokens",
                }
                assert set(fieldnames) == expected_columns

                # Check model 1 data
                model1 = next(r for r in rows if r["model"] == "model-1")
                assert model1["datasetA"] == "80.0"
                assert model1["datasetB"] == "70.0"
                assert model1["datasetC"] == ""  # Empty for missing dataset

                # Check model 2 data
                model2 = next(r for r in rows if r["model"] == "model-2")
                assert model2["datasetA"] == ""  # Empty for missing dataset
                assert model2["datasetB"] == "90.0"
                assert model2["datasetC"] == "85.0"

            finally:
                os.chdir(original_cwd)

    def test_load_existing_model_summary_nonexistent(self, orchestrator):
        """Test loading non-existent model summary file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                result = orchestrator._load_existing_model_summary("nonexistent.csv")

                assert result == {"models": [], "all_datasets": set()}

            finally:
                os.chdir(original_cwd)

    def test_update_model_summary_data_new_model(self, orchestrator):
        """Test updating model summary data with new model."""
        existing_data = {"models": [], "all_datasets": set()}
        new_model_row = {
            "model": "test-model",
            "datasetA": 80.0,
            "datasetB": 75.0,
            "overall": 77.5,
            "time": "2m 30s",
            "tokens": 25.5,
        }

        orchestrator._update_model_summary_data(existing_data, new_model_row)

        assert len(existing_data["models"]) == 1
        assert existing_data["models"][0] == new_model_row
        assert existing_data["all_datasets"] == {"datasetA", "datasetB"}

    def test_update_model_summary_data_existing_model(self, orchestrator):
        """Test updating model summary data with existing model."""
        existing_data = {
            "models": [
                {
                    "model": "test-model",
                    "datasetA": 70.0,
                    "overall": 70.0,
                    "time": "2m 00s",
                    "tokens": 24.0,
                }
            ],
            "all_datasets": {"datasetA"},
        }

        updated_model_row = {
            "model": "test-model",
            "datasetA": 80.0,
            "datasetB": 85.0,
            "overall": 82.5,
            "time": "2m 15s",
            "tokens": 26.0,
        }

        orchestrator._update_model_summary_data(existing_data, updated_model_row)

        assert len(existing_data["models"]) == 1
        assert existing_data["models"][0] == updated_model_row
        assert existing_data["all_datasets"] == {"datasetA", "datasetB"}
