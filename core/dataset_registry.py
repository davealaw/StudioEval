"""
Dataset registry for managing evaluation types and their implementations.
"""
import logging
from typing import Dict, Any, Callable, List

from eval_datasets.custom.custom_mcq import evaluate_cumstom_mcq
from eval_datasets.custom.grammar import evaluate_grammar_dataset  
from eval_datasets.custom.math import evaluate_math_dataset
from eval_datasets.huggingface.arc import evaluate_arc
from eval_datasets.huggingface.mmlu import evaluate_mmlu
from eval_datasets.huggingface.truthfulqa import evaluate_tiny_truthfulqa
from eval_datasets.huggingface.gsm8k import evaluate_gsm8k_dataset
from eval_datasets.huggingface.commonsense_qa import evaluate_commonsense_qa
from eval_datasets.huggingface.logiqa import evaluate_logiqa

logger = logging.getLogger(__name__)


class DatasetRegistry:
    """Registry for dataset evaluation functions."""
    
    def __init__(self):
        """Initialize registry with available evaluators."""
        self._evaluators: Dict[str, Callable] = {
            "grammar": evaluate_grammar_dataset,
            "custom_mcq": evaluate_cumstom_mcq,
            "math": evaluate_math_dataset,
            "gsm8k": evaluate_gsm8k_dataset,
            "arc": evaluate_arc,
            "mmlu": evaluate_mmlu,
            "commonsenseqa": evaluate_commonsense_qa,
            "logiqa": evaluate_logiqa,
            "truthfulqa": evaluate_tiny_truthfulqa
        }
    
    def get_evaluator(self, eval_type: str) -> Callable:
        """
        Get evaluator function for the given type.
        
        Args:
            eval_type: Type of evaluation (e.g., "grammar", "custom_mcq")
            
        Returns:
            Evaluation function
            
        Raises:
            ValueError: If eval_type is not supported
        """
        if eval_type not in self._evaluators:
            available = list(self._evaluators.keys())
            raise ValueError(f"Unknown eval_type: {eval_type}. Available: {available}")
        
        return self._evaluators[eval_type]
    
    def list_supported_types(self) -> List[str]:
        """List all supported evaluation types."""
        return list(self._evaluators.keys())
    
    def is_supported(self, eval_type: str) -> bool:
        """Check if an evaluation type is supported."""
        return eval_type in self._evaluators
    
    def evaluate_dataset(self, eval_type: str, model_id: str, dataset_path: str, 
                        dataset_name: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a dataset using the appropriate evaluator.
        
        Args:
            eval_type: Type of evaluation
            model_id: Model identifier
            dataset_path: Path to dataset
            dataset_name: Name for logging/results
            **kwargs: Additional evaluation parameters
            
        Returns:
            Evaluation results dictionary
        """
        evaluator = self.get_evaluator(eval_type)
        return evaluator(model_id, dataset_path, dataset_name=dataset_name, **kwargs)
