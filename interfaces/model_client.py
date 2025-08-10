"""
Abstract interface for model clients to enable dependency injection and testing.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any


class ModelClient(ABC):
    """Abstract interface for model clients."""
    
    @abstractmethod
    def is_server_running(self) -> bool:
        """Check if the model server is running and accessible."""
        pass
    
    @abstractmethod
    def query_model(self, prompt: str, model_id: str, current: int = 0) -> Tuple[str, Dict[str, Any]]:
        """
        Query a model with a prompt.
        
        Args:
            prompt: The input prompt to send to the model
            model_id: Identifier for the model to query
            current: Current question index for logging purposes
            
        Returns:
            Tuple of (response_text, statistics_dict)
        """
        pass
    
    @abstractmethod
    def load_model(self, model_id: str) -> None:
        """Load a model by ID."""
        pass
    
    @abstractmethod
    def unload_model(self, model_id: str = None) -> None:
        """Unload a model. If model_id is None, unloads current model."""
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """List all available model IDs."""
        pass
    
    @abstractmethod
    def list_models_with_arch(self) -> Dict[str, str]:
        """List models with their architectures."""
        pass


class EvaluationResult:
    """Standard result format for evaluations."""
    
    def __init__(self, dataset: str, correct: int, total: int, skipped: int = 0, tok_per_sec: float = 0.0):
        self.dataset = dataset
        self.correct = correct
        self.total = total
        self.skipped = skipped
        self.tok_per_sec = tok_per_sec
        
    @property
    def accuracy(self) -> float:
        """Calculate accuracy percentage."""
        return round((self.correct / self.total * 100), 2) if self.total > 0 else 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for CSV output."""
        return {
            "dataset": self.dataset,
            "correct": self.correct,
            "total": self.total,
            "skipped": self.skipped,
            "accuracy": self.accuracy,
            "tok_per_sec": self.tok_per_sec
        }
