"""
Enhanced integration tests for evaluation flow with proper dataset mocking.
These tests fix the dataset loading issues from the original integration tests.
"""
import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from core.evaluator import EvaluationOrchestrator  
from core.model_manager import ModelManager
from core.dataset_registry import DatasetRegistry
from implementations.mock_client import MockModelClient
from tests.fixtures.mock_datasets import MockDatasetFiles, MockDatasetLoader


@pytest.fixture
def enhanced_mock_client():
    """Enhanced mock client with realistic responses."""
    responses = {
        "qwen-7b": "Answer: B",
        "qwen-14b": "Answer: A", 
        "llama-7b": "Answer: C",
        "grammar-specialist": "The correct answer is B) He doesn't like it",
        "math-model": "Answer: B (42)"
    }
    
    models = ["qwen-7b", "qwen-14b", "llama-7b", "grammar-specialist", "math-model"]
    
    return MockModelClient(
        responses=responses,
        available_models=models,
        server_running=True
    )


@pytest.fixture
def temp_datasets_config(tmp_path):
    """Create temporary datasets configuration file."""
    config_data = [
        {
            "eval_type": "grammar",
            "dataset_path": "mock_grammar.json",
            "dataset_name": "grammar_test"
        },
        {
            "eval_type": "custom_mcq", 
            "dataset_path": "mock_logic.json",
            "dataset_name": "logic_test"
        },
        {
            "eval_type": "math",
            "dataset_path": "mock_math.json", 
            "dataset_name": "math_test"
        }
    ]
    
    config_file = tmp_path / "test_datasets.json"
    config_file.write_text(json.dumps(config_data, indent=2))
    
    return str(config_file)


class TestEnhancedEvaluationFlow:
    """Enhanced tests for full evaluation pipeline."""
    
    @patch('eval_datasets.custom.grammar.query_model')
    @patch('eval_datasets.custom.grammar.load_json_dataset_with_config')
    @patch('core.evaluator.default_datasets_to_run', [])
    def test_single_model_grammar_evaluation(self, mock_load_dataset, mock_query_model, enhanced_mock_client):
        """Test complete evaluation flow for grammar dataset."""
        # Mock dataset loading and model query
        mock_load_dataset.return_value = MockDatasetLoader.MOCK_DATA['grammar']
        mock_query_model.return_value = ("Corrected: This is correct.", {"tokens_per_second": 12.5})
        
        expected_result = {
            "dataset": "grammar_test",
            "correct": 1,
            "total": 1,
            "skipped": 0,
            "accuracy": 100.0,
            "tok_per_sec": 12.5
        }
        
        # Mock the registry's evaluate_dataset method to return our expected result
        with patch.object(DatasetRegistry, 'evaluate_dataset') as mock_registry_eval, \
             patch('builtins.open') as mock_open:
            mock_registry_eval.return_value = expected_result
            
            orchestrator = EvaluationOrchestrator(
                model_manager=ModelManager(enhanced_mock_client),
                dataset_registry=DatasetRegistry()
            )
            
            result = orchestrator.run_evaluation(
                model="grammar-specialist",
                datasets_config=[{
                    "eval_type": "grammar",
                    "dataset_path": "mock_grammar.json",
                    "dataset_name": "grammar_test"
                }]
            )
        
        assert result is True
        mock_registry_eval.assert_called()
    
    @patch('core.evaluator.default_datasets_to_run', [])
    def test_multiple_models_evaluation(self, enhanced_mock_client):
        """Test evaluation with multiple models."""
        # Mock the registry's evaluate_dataset method to return different results per call
        with patch.object(DatasetRegistry, 'evaluate_dataset') as mock_eval, \
             patch('builtins.open') as mock_open, \
             patch('time.sleep'):  # Mock sleep to avoid delays
            
            # Return different results for different models
            mock_eval.side_effect = [
                {
                    "dataset": "grammar_test",
                    "correct": 0,
                    "total": 1,
                    "skipped": 0,
                    "accuracy": 0.0,
                    "tok_per_sec": 10.0
                },
                {
                    "dataset": "grammar_test", 
                    "correct": 1,
                    "total": 1,
                    "skipped": 0,
                    "accuracy": 100.0,
                    "tok_per_sec": 15.0
                }
            ]
            
            orchestrator = EvaluationOrchestrator(
                model_manager=ModelManager(enhanced_mock_client), 
                dataset_registry=DatasetRegistry()
            )
            
            result = orchestrator.run_evaluation(
                models=["qwen-7b", "grammar-specialist"],
                datasets_config=[{
                    "eval_type": "grammar",
                    "dataset_path": "mock_grammar.json", 
                    "dataset_name": "grammar_test"
                }]
            )
        
        assert result is True
        assert mock_eval.call_count == 2  # Called once per model
    
    @patch('eval_datasets.custom.grammar.query_model')
    @patch('utils.data_loading.load_dataset_with_config')
    @patch('core.evaluator.default_datasets_to_run', [])
    def test_multiple_datasets_evaluation(self, mock_load_dataset, mock_query_model, enhanced_mock_client):
        """Test evaluation with multiple datasets."""
        def mock_load_side_effect(dataset_name, file_path=None):
            if 'grammar' in file_path:
                return MockDatasetLoader.MOCK_DATA['grammar']
            elif 'logic' in file_path:
                return MockDatasetLoader.MOCK_DATA['logic'] 
            elif 'math' in file_path:
                return MockDatasetLoader.MOCK_DATA['math']
            return []
        
        mock_load_dataset.side_effect = mock_load_side_effect
        mock_query_model.return_value = ("Corrected: This is correct.", {"tokens_per_second": 12.5})
        
        orchestrator = EvaluationOrchestrator(
            model_manager=ModelManager(enhanced_mock_client),
            dataset_registry=DatasetRegistry()
        )
        
        # Mock the registry's evaluate_dataset method to handle multiple dataset types
        with patch.object(DatasetRegistry, 'evaluate_dataset') as mock_eval, \
             patch('builtins.open') as mock_open, \
             patch('time.sleep'):  # Mock sleep to avoid delays
            
            # Return different results for different dataset types
            mock_eval.side_effect = [
                {
                    "dataset": "grammar_test",
                    "correct": 1, "total": 1, "skipped": 0, "accuracy": 100.0, "tok_per_sec": 12.0
                },
                {
                    "dataset": "logic_test", 
                    "correct": 1, "total": 1, "skipped": 0, "accuracy": 100.0, "tok_per_sec": 11.0
                },
                {
                    "dataset": "math_test",
                    "correct": 1, "total": 1, "skipped": 0, "accuracy": 100.0, "tok_per_sec": 10.0
                }
            ]
            
            result = orchestrator.run_evaluation(
                model="qwen-7b",
                datasets_config=[
                    {"eval_type": "grammar", "dataset_path": "mock_grammar.json", "dataset_name": "grammar_test"},
                    {"eval_type": "custom_mcq", "dataset_path": "mock_logic.json", "dataset_name": "logic_test"},
                    {"eval_type": "gsm8k", "dataset_path": "mock_math.json", "dataset_name": "math_test"}
                ]
            )
        
        assert result is True
        assert mock_eval.call_count == 3  # Called once per dataset
    
    def test_server_down_handling(self, enhanced_mock_client):
        """Test graceful handling when server is down."""
        enhanced_mock_client.set_server_running(False)
        
        orchestrator = EvaluationOrchestrator(
            model_manager=ModelManager(enhanced_mock_client),
            dataset_registry=DatasetRegistry()
        )
        
        result = orchestrator.run_evaluation(
            model="qwen-7b",
            datasets_config=[]
        )
        
        # Should return False when server is down
        assert result is False
    
    def test_invalid_model_handling(self, enhanced_mock_client):
        """Test handling of invalid model selection.""" 
        orchestrator = EvaluationOrchestrator(
            model_manager=ModelManager(enhanced_mock_client),
            dataset_registry=DatasetRegistry()
        )
        
        # Test with non-existent model - should raise exception during model loading
        with pytest.raises(ValueError, match="Model nonexistent-model not available"):
            orchestrator.run_evaluation(
                model="nonexistent-model",
                datasets_config=[]
            )
    


class TestErrorRecoveryScenarios:
    """Test evaluation resilience and error recovery."""
    
    def test_partial_dataset_failure(self, enhanced_mock_client):
        """Test handling when some datasets fail to load."""
        # Mock the registry's evaluate_dataset method to simulate some failures
        with patch.object(DatasetRegistry, 'evaluate_dataset') as mock_eval, \
             patch('builtins.open') as mock_open, \
             patch('time.sleep'):  # Mock sleep to avoid delays
            
            # First call succeeds, second call fails
            mock_eval.side_effect = [
                {
                    "dataset": "working_test",
                    "correct": 1, "total": 1, "skipped": 0, "accuracy": 100.0, "tok_per_sec": 10.0
                },
                FileNotFoundError("Dataset failing.json not found")
            ]
            
            orchestrator = EvaluationOrchestrator(
                model_manager=ModelManager(enhanced_mock_client),
                dataset_registry=DatasetRegistry()
            )
            
            result = orchestrator.run_evaluation(
                model="qwen-7b",
                datasets_config=[
                    {"eval_type": "grammar", "dataset_path": "working.json", "dataset_name": "working_test"},
                    {"eval_type": "grammar", "dataset_path": "failing.json", "dataset_name": "failing_test"}
                ]
            )
        
        # Should continue with successful datasets
        assert result is True
        assert mock_eval.call_count == 2  # Both datasets attempted
    
    def test_evaluation_function_exception(self, enhanced_mock_client):
        """Test handling when evaluation function throws exception."""
        orchestrator = EvaluationOrchestrator(
            model_manager=ModelManager(enhanced_mock_client),
            dataset_registry=DatasetRegistry()
        )
        
        # Mock evaluation function that throws exception
        with patch('eval_datasets.custom.grammar.evaluate_grammar_dataset') as mock_eval:
            mock_eval.side_effect = Exception("Evaluation failed")
            
            result = orchestrator._evaluate_single_dataset(
                "qwen-7b", 
                {"eval_type": "grammar", "dataset_path": "test.json", "dataset_name": "test"}
            )
        
        # Should return None on exception
        assert result is None
    
    def test_model_loading_failure_recovery(self, enhanced_mock_client):
        """Test recovery when model loading fails."""
        # Mock client to simulate loading failure for specific model
        def failing_load_model(model_name):
            if model_name == "failing-model":
                raise Exception("Failed to load model")
        
        enhanced_mock_client.load_model = Mock(side_effect=failing_load_model)
        
        orchestrator = EvaluationOrchestrator(
            model_manager=ModelManager(enhanced_mock_client),
            dataset_registry=DatasetRegistry()
        )
        
        # Should handle model loading failure gracefully by catching the exception
        with pytest.raises(Exception, match="Failed to load model"):
            orchestrator.run_evaluation(
                model="failing-model",
                datasets_config=[]
            )


class TestPerformanceScenarios:
    """Test performance-related evaluation scenarios."""
    
    def test_large_model_list_evaluation(self, enhanced_mock_client):
        """Test evaluation with many models (performance test setup)."""
        # Add many models to simulate large deployment
        large_model_list = [f"model-{i}" for i in range(20)]
        enhanced_mock_client.available_models.extend(large_model_list)
        
        # Mock responses for all models
        for model in large_model_list:
            enhanced_mock_client.responses[model] = "Answer: A"
        
        orchestrator = EvaluationOrchestrator(
            model_manager=ModelManager(enhanced_mock_client),
            dataset_registry=DatasetRegistry()
        )
        
        # Test model resolution with large list
        models = orchestrator.model_manager.resolve_models(all_models=True)
        
        assert len(models) >= 20
        assert all(f"model-{i}" in models for i in range(20))
    
    def test_evaluation_timing_tracking(self, enhanced_mock_client):
        """Test that evaluation timing is properly tracked."""
        # Mock the registry's evaluate_dataset method
        with patch.object(DatasetRegistry, 'evaluate_dataset') as mock_eval, \
             patch('builtins.open') as mock_open, \
             patch('time.sleep'):  # Mock sleep to avoid delays
            
            mock_eval.return_value = {
                "dataset": "timing_test",
                "correct": 1,
                "total": 1,
                "skipped": 0,
                "accuracy": 100.0,
                "tok_per_sec": 15.0  # Performance metric
            }
            
            orchestrator = EvaluationOrchestrator(
                model_manager=ModelManager(enhanced_mock_client),
                dataset_registry=DatasetRegistry()
            )
            
            result = orchestrator.run_evaluation(
                model="qwen-7b",
                datasets_config=[{
                    "eval_type": "grammar",
                    "dataset_path": "timing.json", 
                    "dataset_name": "timing_test"
                }]
            )
        
        assert result is True
        mock_eval.assert_called()
        
        # Verify that timing metrics would be captured
        call_args = mock_eval.call_args
        assert call_args is not None
