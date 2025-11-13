"""
Mock implementation of ModelClient for testing without LM Studio dependency.
"""

import logging
from typing import Any, Optional

from interfaces.model_client import ModelClient

logger = logging.getLogger(__name__)


class MockModelClient(ModelClient):
    """Mock implementation of ModelClient for testing."""

    def __init__(
        self,
        responses: Optional[dict[str, str]] = None,
        available_models: Optional[list[str]] = None,
        server_running: bool = True,
    ):
        """
        Initialize mock client.

        Args:
            responses: Dict mapping model_id -> response text
            available_models: List of available model IDs
            server_running: Whether the mock server is running
        """
        self.responses: dict[str, str] = responses or {
            "test-model": "Answer: A",
            "grammar-model": "Corrected: The corrected text.",
            "math-model": "Answer: 42",
        }
        self.available_models: list[str] = available_models or [
            "test-model",
            "grammar-model",
            "math-model",
        ]
        self.loaded_models: set[str] = set()
        self.call_log: list[dict[str, Any]] = []  # Track all queries for testing
        self.is_running = server_running

    def is_server_running(self) -> bool:
        """Mock server is always running unless explicitly set."""
        return self.is_running

    def query_model(
        self, prompt: str, model_id: str, current: int = 0
    ) -> tuple[str, dict[str, Any]]:
        """Return mock response for the given model."""
        self.call_log.append(
            {"prompt": prompt, "model_id": model_id, "current": current}
        )

        response = self.responses.get(model_id, "Mock response")
        stats = {
            "tokens_per_second": 15.0,
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(response.split()),
            "total_tokens": len(prompt.split()) + len(response.split()),
            "stop_reason": "eos_token",
            "structured": False,
        }

        return response, stats

    def load_model(self, model_id: str) -> None:
        """Mock model loading."""
        if model_id not in self.available_models:
            raise ValueError(f"Model {model_id} not available")
        self.loaded_models.add(model_id)
        logger.info(f"Mock: Loaded model {model_id}")

    def unload_model(self, model_id: Optional[str] = None) -> None:
        """Mock model unloading."""
        if model_id is not None:
            self.loaded_models.discard(model_id)
            logger.info(f"Mock: Unloaded model {model_id}")
        else:
            self.loaded_models.clear()
            logger.info("Mock: Unloaded all models")

    def list_models(self) -> list[str]:
        """Return list of available models."""
        return self.available_models.copy()

    def list_models_with_arch(self) -> dict[str, str]:
        """Return models with mock architectures."""
        return {
            model: f"mock_arch_{model.split('-')[0]}" for model in self.available_models
        }

    def set_server_running(self, running: bool) -> None:
        """Helper method to simulate server down scenarios."""
        self.is_running = running

    def add_response(self, model_id: str, response: str) -> None:
        """Helper method to add mock responses during testing."""
        self.responses[model_id] = response

    def get_call_count(self, model_id: Optional[str] = None) -> int:
        """Get number of queries made to a specific model or all models."""
        if model_id is not None:
            return len([call for call in self.call_log if call["model_id"] == model_id])
        return len(self.call_log)

    def reset_call_log(self) -> None:
        """Reset the call log for testing."""
        self.call_log.clear()
