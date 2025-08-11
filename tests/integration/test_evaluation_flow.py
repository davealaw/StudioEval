"""
Integration tests for evaluation flow using the new refactored architecture.
"""
import pytest
import json
from unittest.mock import Mock, patch
from typing import Dict, Any

from core.evaluator import EvaluationOrchestrator  
from core.model_manager import ModelManager
from core.dataset_registry import DatasetRegistry
from implementations.mock_client import MockModelClient


@pytest.fixture
def mock_model_client():
    """Create a mock model client with predefined responses."""
    responses = {
        "test-model": "Answer: A",
        "grammar-model": "Corrected: This is the corrected sentence.",
        "math-model": "Answer: 42"
    }
    return MockModelClient(responses=responses, available_models=["test-model", "grammar-model", "math-model"])


@pytest.fixture  
def model_manager(mock_model_client):
    """Create model manager with mock client."""
    return ModelManager(client=mock_model_client)


@pytest.fixture
def dataset_registry():
    """Create dataset registry."""
    return DatasetRegistry()


@pytest.fixture
def orchestrator(model_manager, dataset_registry):
    """Create evaluation orchestrator with mocked dependencies."""
    return EvaluationOrchestrator(model_manager=model_manager, dataset_registry=dataset_registry)


class TestModelManager:
    """Test ModelManager functionality."""
    
    def test_resolve_models_single(self, model_manager):
        """Test resolving single model."""
        models = model_manager.resolve_models(model="test-model")
        assert models == ["test-model"]
        
    def test_resolve_models_all(self, model_manager):
        """Test resolving all models."""
        models = model_manager.resolve_models(all_models=True)
        assert "test-model" in models
        assert "grammar-model" in models
        assert "math-model" in models
        
    def test_resolve_models_filter(self, model_manager):
        """Test resolving models with filter."""
        models = model_manager.resolve_models(model_filter="test*")
        assert models == ["test-model"]
        
    def test_resolve_models_no_criteria(self, model_manager):
        """Test that no criteria raises ValueError."""
        with pytest.raises(ValueError, match="Must specify model selection criteria"):
            model_manager.resolve_models()
            

class TestDatasetRegistry:
    """Test DatasetRegistry functionality."""
    
    def test_list_supported_types(self, dataset_registry):
        """Test listing supported evaluation types."""
        types = dataset_registry.list_supported_types()
        assert "grammar" in types
        assert "custom_mcq" in types
        assert "gsm8k" in types
        assert len(types) > 5  # Should have multiple types
        
    def test_is_supported(self, dataset_registry):
        """Test checking if types are supported."""
        assert dataset_registry.is_supported("grammar")
        assert dataset_registry.is_supported("custom_mcq")
        assert not dataset_registry.is_supported("unknown_type")
        
    def test_get_evaluator_valid(self, dataset_registry):
        """Test getting valid evaluator."""
        evaluator = dataset_registry.get_evaluator("grammar")
        assert callable(evaluator)
        
    def test_get_evaluator_invalid(self, dataset_registry):
        """Test getting invalid evaluator raises error."""
        with pytest.raises(ValueError, match="Unknown eval_type"):
            dataset_registry.get_evaluator("invalid_type")


class TestEvaluationOrchestrator:
    """Test EvaluationOrchestrator integration."""
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes correctly."""
        assert isinstance(orchestrator.model_manager, ModelManager)
        assert isinstance(orchestrator.dataset_registry, DatasetRegistry)
    
    @patch('core.evaluator.default_datasets_to_run', [])  # Use empty list to avoid actual evaluations
    def test_run_evaluation_server_not_running(self, mock_model_client):
        """Test evaluation when server is not running."""
        mock_model_client.set_server_running(False)
        model_manager = ModelManager(client=mock_model_client)
        orchestrator = EvaluationOrchestrator(model_manager=model_manager)
        
        result = orchestrator.run_evaluation(model="test-model")
        assert not result
    
    @patch('core.evaluator.default_datasets_to_run', [])
    def test_run_evaluation_no_models(self, mock_model_client):
        """Test evaluation with no model criteria."""
        model_manager = ModelManager(client=mock_model_client)
        orchestrator = EvaluationOrchestrator(model_manager=model_manager)
        
        result = orchestrator.run_evaluation()  # No model criteria
        assert not result
    
    @patch('core.evaluator.default_datasets_to_run', [])
    def test_run_evaluation_success(self, orchestrator):
        """Test successful evaluation run."""
        result = orchestrator.run_evaluation(model="test-model")
        assert result is True


class TestEvaluationIntegration:
    """Test full evaluation integration with mocks."""
    
    def test_load_datasets_config_default(self, orchestrator):
        """Test loading default datasets config."""
        with patch('core.evaluator.default_datasets_to_run', [{"test": "config"}]):
            config = orchestrator._load_datasets_config()
            assert config == [{"test": "config"}]
    
    def test_load_datasets_config_from_file(self, orchestrator, temp_config_file):
        """Test loading datasets config from file."""
        config_data = [{"eval_type": "test", "dataset_name": "test"}]
        config_path = temp_config_file(config_data)
        
        loaded_config = orchestrator._load_datasets_config(config_path)
        assert loaded_config == config_data
    
    def test_load_datasets_config_invalid_json(self, orchestrator, tmp_path):
        """Test loading invalid JSON raises appropriate error."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text('{"invalid": json}')
        
        with pytest.raises(json.JSONDecodeError):
            orchestrator._load_datasets_config(str(invalid_file))
    
    def test_load_datasets_config_file_not_found(self, orchestrator):
        """Test loading non-existent file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            orchestrator._load_datasets_config("/nonexistent/file.json")


class TestModelManagerIntegration:
    """Integration tests for ModelManager with MockModelClient."""
    
    def test_query_model_integration(self, model_manager):
        """Test model querying through manager."""
        response, stats = model_manager.query_model("Test prompt", "test-model")
        
        assert response == "Answer: A"
        assert "tokens_per_second" in stats
        assert stats["tokens_per_second"] == 15.0
        
    def test_model_loading_integration(self, model_manager):
        """Test model loading through manager."""
        # Should not raise exception for available model
        model_manager.load_model("test-model")
        
        # Should raise exception for unavailable model
        with pytest.raises(ValueError):
            model_manager.load_model("nonexistent-model")


@pytest.mark.integration
class TestFullEvaluationMocking:
    """Test full evaluation with comprehensive mocking."""
    
    def test_evaluate_single_dataset_grammar(self, orchestrator):
        """Test single dataset evaluation for grammar type."""
        # Setup mock by directly replacing the evaluator in the registry
        mock_eval = Mock(return_value={
            "dataset": "test_grammar",
            "correct": 8,
            "total": 10,
            "skipped": 0,
            "accuracy": 80.0,
            "tok_per_sec": 12.5
        })
        
        # Replace the evaluator function in the registry
        original_evaluator = orchestrator.dataset_registry._evaluators["grammar"]
        orchestrator.dataset_registry._evaluators["grammar"] = mock_eval
        
        try:
            # Test
            dataset_config = {
                "eval_type": "grammar",
                "dataset_path": "test_path.jsonl",
                "dataset_name": "test_grammar"
            }
            
            result = orchestrator._evaluate_single_dataset("test-model", dataset_config)
            
            # Assertions
            assert result["accuracy"] == 80.0
            assert result["correct"] == 8
            assert result["total"] == 10
            mock_eval.assert_called_once()
        finally:
            # Restore original evaluator
            orchestrator.dataset_registry._evaluators["grammar"] = original_evaluator
        
    def test_evaluate_single_dataset_mcq(self, orchestrator):
        """Test single dataset evaluation for MCQ type."""
        # Setup mock by directly replacing the evaluator in the registry
        mock_eval = Mock(return_value={
            "dataset": "test_mcq", 
            "correct": 15,
            "total": 20,
            "skipped": 1,
            "accuracy": 75.0,
            "tok_per_sec": 10.0
        })
        
        # Replace the evaluator function in the registry
        original_evaluator = orchestrator.dataset_registry._evaluators["custom_mcq"]
        orchestrator.dataset_registry._evaluators["custom_mcq"] = mock_eval
        
        try:
            dataset_config = {
                "eval_type": "custom_mcq",
                "dataset_path": "test_mcq.jsonl", 
                "dataset_name": "test_mcq"
            }
            
            result = orchestrator._evaluate_single_dataset("test-model", dataset_config)
            
            assert result["accuracy"] == 75.0
            assert result["skipped"] == 1
            mock_eval.assert_called_once()
        finally:
            # Restore original evaluator
            orchestrator.dataset_registry._evaluators["custom_mcq"] = original_evaluator
        
    def test_evaluate_single_dataset_unknown_type(self, orchestrator):
        """Test evaluation with unknown dataset type."""
        dataset_config = {
            "eval_type": "unknown_type",
            "dataset_path": "test.jsonl",
            "dataset_name": "test"
        }
        
        result = orchestrator._evaluate_single_dataset("test-model", dataset_config)
        assert result is None  # Should fail gracefully
