"""
Unit tests for individual dataset evaluators.
Tests the specific evaluation functions for each dataset type.
"""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from tqdm import tqdm

# Import evaluators to test
from eval_datasets.huggingface.arc import evaluate_arc
from eval_datasets.huggingface.mmlu import evaluate_mmlu  
from eval_datasets.huggingface.gsm8k import evaluate_gsm8k_dataset
from eval_datasets.custom.custom_mcq import evaluate_cumstom_mcq
from eval_datasets.custom.grammar import evaluate_grammar_dataset
from eval_datasets.custom.math import evaluate_math_dataset


class TestARCEvaluator:
    """Test ARC (AI2 Reasoning Challenge) evaluator."""
    
    @patch('eval_datasets.huggingface.arc.load_dataset_with_config')
    @patch('eval_datasets.huggingface.arc.query_model')
    @patch('eval_datasets.huggingface.arc.time.sleep')  # Speed up tests
    def test_arc_evaluator_basic(self, mock_sleep, mock_query, mock_load):
        """Test basic ARC evaluation."""
        # Setup mock dataset
        mock_dataset = [
            {
                "question": "What is 2+2?",
                "choices": {"label": ["A", "B", "C", "D"], "text": ["3", "4", "5", "6"]},
                "answerKey": "B"
            },
            {
                "question": "What color is the sky?", 
                "choices": {"label": ["A", "B", "C", "D"], "text": ["red", "blue", "green", "yellow"]},
                "answerKey": "B"
            }
        ]
        mock_load.return_value = mock_dataset
        
        # Setup mock model responses
        mock_query.side_effect = [
            ("Answer: B", {"tokens_per_second": 10.0}),
            ("Answer: B", {"tokens_per_second": 12.0})
        ]
        
        # Run evaluation
        result = evaluate_arc("test_model", dataset_name="test_arc")
        
        # Verify results
        assert result["dataset"] == "test_arc"
        assert result["total"] == 2
        assert result["correct"] == 2
        assert result["skipped"] == 0
        assert result["accuracy"] == 100.0
        assert result["tok_per_sec"] == 11.0  # Average
        
        # Verify load_dataset was called correctly
        mock_load.assert_called_once_with(
            "tinyBenchmarks/tinyAI2_arc", 
            subset=None, 
            split="test", 
            seed=42, 
            sample_size=0
        )
        
        # Verify model was queried twice
        assert mock_query.call_count == 2
    
    @patch('eval_datasets.huggingface.arc.load_dataset_with_config')
    @patch('eval_datasets.huggingface.arc.query_model')
    @patch('eval_datasets.huggingface.arc.time.sleep')
    def test_arc_evaluator_handles_wrong_answers(self, mock_sleep, mock_query, mock_load):
        """Test ARC evaluation with incorrect answers."""
        mock_dataset = [
            {
                "question": "Test question",
                "choices": {"label": ["A", "B"], "text": ["wrong", "right"]},
                "answerKey": "B"
            }
        ]
        mock_load.return_value = mock_dataset
        mock_query.return_value = ("Answer: A", {"tokens_per_second": 5.0})
        
        result = evaluate_arc("test_model")
        
        assert result["correct"] == 0
        assert result["total"] == 1
        assert result["accuracy"] == 0.0
    
    @patch('eval_datasets.huggingface.arc.load_dataset_with_config')
    @patch('eval_datasets.huggingface.arc.query_model')
    @patch('eval_datasets.huggingface.arc.time.sleep')
    def test_arc_evaluator_skips_malformed_items(self, mock_sleep, mock_query, mock_load):
        """Test ARC evaluation skips malformed dataset items."""
        mock_dataset = [
            {"question": "Test", "choices": {"label": ["A"], "text": ["answer"]}, "answerKey": "A"},  # Valid
            {"question": "Test", "choices": "malformed"},  # Invalid choices - missing answerKey
            {"question": "Test"},  # Missing answerKey
            # NOTE: The current ARC evaluator doesn't handle missing 'question' gracefully - it would crash
            # This is a potential improvement area for the actual implementation
        ]
        mock_load.return_value = mock_dataset
        mock_query.return_value = ("Answer: A", {"tokens_per_second": 5.0})
        
        result = evaluate_arc("test_model")
        
        assert result["total"] == 1  # Only 1 valid item processed
        assert result["skipped"] == 2  # 2 items skipped (both missing answerKey)
        assert mock_query.call_count == 1  # Only called once for valid item


class TestCustomMCQEvaluator:
    """Test Custom MCQ evaluator."""
    
    @patch('eval_datasets.custom.custom_mcq.load_json_dataset_with_config')
    @patch('eval_datasets.custom.custom_mcq.query_model')
    @patch('eval_datasets.custom.custom_mcq.time.sleep')
    def test_custom_mcq_evaluator_basic(self, mock_sleep, mock_query, mock_load):
        """Test basic custom MCQ evaluation."""
        mock_dataset = [
            {
                "question": "What is Python?",
                "choices": {"A": "Snake", "B": "Language", "C": "Food", "D": "Animal"},
                "answer": "B"
            }
        ]
        mock_load.return_value = mock_dataset
        mock_query.return_value = ("Answer: B", {"tokens_per_second": 8.0})
        
        result = evaluate_cumstom_mcq("test_model", "test.jsonl")
        
        assert result["dataset"] == "coding_mcq"  # Default name
        assert result["total"] == 1
        assert result["correct"] == 1
        assert result["accuracy"] == 100.0
        
        # Verify dataset loading
        mock_load.assert_called_once_with("test.jsonl", seed=42, sample_size=0)
    
    @patch('eval_datasets.custom.custom_mcq.load_json_dataset_with_config')
    def test_custom_mcq_evaluator_handles_dataset_failure(self, mock_load):
        """Test custom MCQ evaluation when dataset loading fails."""
        mock_load.return_value = None
        
        result = evaluate_cumstom_mcq("test_model", "bad.jsonl")
        
        assert result is None
    
    @patch('eval_datasets.custom.custom_mcq.load_json_dataset_with_config')
    @patch('eval_datasets.custom.custom_mcq.query_model') 
    @patch('eval_datasets.custom.custom_mcq.time.sleep')
    def test_custom_mcq_evaluator_skips_invalid_items(self, mock_sleep, mock_query, mock_load):
        """Test custom MCQ evaluation skips invalid items."""
        mock_dataset = [
            # Valid item
            {
                "question": "Valid question?", 
                "choices": {"A": "1", "B": "2", "C": "3", "D": "4"},
                "answer": "A"
            },
            # Invalid items
            {"question": "", "choices": {"A": "1", "B": "2", "C": "3", "D": "4"}, "answer": "A"},  # Empty question
            {"question": "Test", "choices": {"A": "1", "B": "2"}, "answer": "A"},  # Not 4 choices
            {"question": "Test", "choices": {"A": "1", "B": "2", "C": "3", "D": "4"}, "answer": "E"},  # Invalid answer
            {"question": "Test", "choices": {"A": "1", "B": "2", "C": "3", "D": "4"}},  # Missing answer
        ]
        mock_load.return_value = mock_dataset
        mock_query.return_value = ("Answer: A", {"tokens_per_second": 5.0})
        
        result = evaluate_cumstom_mcq("test_model", "test.jsonl")
        
        assert result["total"] == 1  # Only 1 valid item
        assert result["skipped"] == 4  # 4 invalid items
        assert mock_query.call_count == 1


class TestGrammarEvaluator:
    """Test Grammar dataset evaluator."""
    
    @patch('eval_datasets.custom.grammar.load_json_dataset_with_config')
    @patch('eval_datasets.custom.grammar.query_model')
    @patch('eval_datasets.custom.grammar.time.sleep')
    def test_grammar_evaluator_basic(self, mock_sleep, mock_query, mock_load):
        """Test basic grammar evaluation."""
        mock_dataset = [
            {
                "question": "Me and him went to store",
                "answer": "He and I went to the store"
            }
        ]
        mock_load.return_value = mock_dataset
        mock_query.return_value = ("Corrected: He and I went to the store", {"tokens_per_second": 7.0})
        
        result = evaluate_grammar_dataset("test_model", "grammar.jsonl")
        
        assert result["dataset"] == "grammar"  # Default name
        assert result["total"] == 1
        assert result["correct"] == 1
        assert result["accuracy"] == 100.0
    
    @patch('eval_datasets.custom.grammar.load_json_dataset_with_config')
    def test_grammar_evaluator_handles_dataset_failure(self, mock_load):
        """Test grammar evaluation when dataset loading fails."""
        # When dataset loading returns None, tqdm will fail on None iteration
        # This is the actual behavior - the evaluator doesn't handle this gracefully
        mock_load.return_value = None
        
        # The evaluator should raise an exception when trying to iterate None
        with pytest.raises(TypeError):  # 'NoneType' object is not iterable
            evaluate_grammar_dataset("test_model", "bad.jsonl")


class TestMathEvaluator:
    """Test Math dataset evaluator."""
    
    @patch('eval_datasets.custom.math.load_json_dataset_with_config')
    @patch('eval_datasets.custom.math.query_model')
    def test_math_evaluator_basic(self, mock_query, mock_load):
        """Test basic math evaluation."""
        mock_dataset = [
            {
                "question": "What is 5 + 3?",
                "answer": 8
            }
        ]
        mock_load.return_value = mock_dataset
        mock_query.return_value = ("Answer: 8", {"tokens_per_second": 6.0})
        
        result = evaluate_math_dataset("test_model", "math.jsonl", sleep=False)
        
        assert result["dataset"] == "basic math"  # Default name
        assert result["total"] == 1
        assert result["correct"] == 1
        assert result["accuracy"] == 100.0
    
    @patch('eval_datasets.custom.math.load_json_dataset_with_config')
    @patch('eval_datasets.custom.math.query_model')
    def test_math_evaluator_handles_float_answers(self, mock_query, mock_load):
        """Test math evaluation with float answers."""
        mock_dataset = [
            {
                "question": "What is 1/2 as decimal?", 
                "answer": 0.5
            }
        ]
        mock_load.return_value = mock_dataset
        mock_query.return_value = ("Answer: 0.5", {"tokens_per_second": 6.0})
        
        result = evaluate_math_dataset("test_model", "math.jsonl", sleep=False)
        
        assert result["correct"] == 1
        assert result["accuracy"] == 100.0


class TestGSM8KEvaluator:
    """Test GSM8K dataset evaluator."""
    
    @patch('eval_datasets.huggingface.gsm8k.load_dataset_with_config')
    @patch('eval_datasets.huggingface.gsm8k.query_model')
    @patch('eval_datasets.huggingface.gsm8k.time.sleep')
    def test_gsm8k_evaluator_basic(self, mock_sleep, mock_query, mock_load):
        """Test basic GSM8K evaluation."""
        mock_dataset = [
            {
                "question": "Tom has 5 apples. He gives 2 away. How many does he have left?",
                "answer": "#### 3"  # GSM8K format includes ####
            }
        ]
        mock_load.return_value = mock_dataset
        mock_query.return_value = ("Let me solve step by step... #### 3", {"tokens_per_second": 9.0})
        
        result = evaluate_gsm8k_dataset("test_model")
        
        assert result["dataset"] == "tinyGSM8k"  # Default name
        assert result["total"] == 1
        assert result["correct"] == 1
        assert result["accuracy"] == 100.0
    
    @patch('eval_datasets.huggingface.gsm8k.load_dataset_with_config')
    @patch('eval_datasets.huggingface.gsm8k.query_model')
    @patch('eval_datasets.huggingface.gsm8k.time.sleep')
    def test_gsm8k_evaluator_wrong_answer(self, mock_sleep, mock_query, mock_load):
        """Test GSM8K evaluation with wrong answer."""
        mock_dataset = [
            {
                "question": "Simple math problem",
                "answer": "5"
            }
        ]
        mock_load.return_value = mock_dataset  
        mock_query.return_value = ("Wrong calculation #### 7", {"tokens_per_second": 9.0})
        
        result = evaluate_gsm8k_dataset("test_model")
        
        assert result["correct"] == 0
        assert result["accuracy"] == 0.0


class TestMMLUEvaluator:
    """Test MMLU dataset evaluator."""
    
    @patch('eval_datasets.huggingface.mmlu.load_dataset_with_config')
    @patch('eval_datasets.huggingface.mmlu.query_model')
    @patch('eval_datasets.huggingface.mmlu.time.sleep')
    def test_mmlu_evaluator_basic(self, mock_sleep, mock_query, mock_load):
        """Test basic MMLU evaluation."""
        mock_dataset = [
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Berlin", "Paris", "Madrid"], 
                "answer": 2  # Paris (0-indexed)
            }
        ]
        mock_load.return_value = mock_dataset
        mock_query.return_value = ("Answer: C", {"tokens_per_second": 11.0})
        
        result = evaluate_mmlu("test_model", "tinyBenchmarks/tinyMMLU", "tinyMMLU")
        
        assert result["dataset"] == "tinyMMLU"  # Default name
        assert result["total"] == 1
        assert result["correct"] == 1
        assert result["accuracy"] == 100.0


class TestEvaluatorCommonPatterns:
    """Test common patterns across evaluators."""
    
    def test_evaluators_handle_custom_dataset_names(self):
        """Test that evaluators accept custom dataset names."""
        with patch('eval_datasets.custom.grammar.load_json_dataset_with_config') as mock_load:
            with patch('eval_datasets.custom.grammar.query_model'):
                with patch('eval_datasets.custom.grammar.time.sleep'):
                    mock_load.return_value = []  # Empty dataset
                    
                    result = evaluate_grammar_dataset("model", "test.jsonl", dataset_name="custom_grammar")
                    
                    assert result["dataset"] == "custom_grammar"
    
    def test_evaluators_handle_sample_size_parameter(self):
        """Test that evaluators pass sample_size to dataset loaders."""
        with patch('eval_datasets.custom.custom_mcq.load_json_dataset_with_config') as mock_load:
            mock_load.return_value = []
            
            evaluate_cumstom_mcq("model", "test.jsonl", sample_size=10)
            
            mock_load.assert_called_once_with("test.jsonl", seed=42, sample_size=10)
    
    def test_evaluators_handle_seed_parameter(self):
        """Test that evaluators pass seed to dataset loaders."""
        with patch('eval_datasets.huggingface.arc.load_dataset_with_config') as mock_load:
            mock_load.return_value = []
            
            evaluate_arc("model", seed=123)
            
            # Check that seed was passed correctly
            call_args = mock_load.call_args
            assert call_args[1]['seed'] == 123


class TestEvaluatorErrorHandling:
    """Test error handling across evaluators."""
    
    @patch('eval_datasets.custom.custom_mcq.load_json_dataset_with_config')
    @patch('eval_datasets.custom.custom_mcq.query_model')
    def test_evaluator_handles_model_query_exception(self, mock_query, mock_load):
        """Test evaluator behavior when model query raises exception."""
        mock_dataset = [
            {
                "question": "Test",
                "choices": {"A": "1", "B": "2", "C": "3", "D": "4"},
                "answer": "A"
            }
        ]
        mock_load.return_value = mock_dataset
        mock_query.side_effect = RuntimeError("Model query failed")
        
        # The evaluator should let the exception propagate
        with pytest.raises(RuntimeError, match="Model query failed"):
            evaluate_cumstom_mcq("test_model", "test.jsonl")
    
    def test_evaluators_handle_empty_datasets(self):
        """Test evaluator behavior with empty datasets."""
        with patch('eval_datasets.custom.grammar.load_json_dataset_with_config') as mock_load:
            with patch('eval_datasets.custom.grammar.query_model'):
                mock_load.return_value = []  # Empty dataset
                
                result = evaluate_grammar_dataset("model", "empty.jsonl")
                
                assert result["total"] == 0
                assert result["correct"] == 0
                assert result["accuracy"] == 0.0
