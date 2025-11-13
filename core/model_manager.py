"""
Model management wrapper providing higher-level operations.
"""

import fnmatch
import logging
from typing import Any, Optional

from implementations.lmstudio_client import LMStudioClient
from interfaces.model_client import ModelClient

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model operations through the ModelClient interface."""

    def __init__(self, client: Optional[ModelClient] = None):
        """
        Initialize with a ModelClient implementation.

        Args:
            client: ModelClient implementation (defaults to LMStudioClient)
        """
        self.client = client if client is not None else LMStudioClient()

    def is_server_running(self) -> bool:
        """Check if the model server is accessible."""
        return self.client.is_server_running()

    def resolve_models(
        self,
        model: Optional[str] = None,
        model_filter: Optional[str] = None,
        model_set: Optional[list[str]] = None,
        model_arch: Optional[str] = None,
        all_models: bool = False,
    ) -> list[str]:
        """
        Resolve which models to evaluate based on various selection criteria.

        Args:
            model: Specific model ID
            model_filter: Glob pattern for model filtering
            model_set: List of specific model IDs
            model_arch: Architecture pattern filter
            all_models: Whether to use all available models

        Returns:
            List of model IDs to evaluate

        Raises:
            ValueError: If no models match criteria or invalid configuration
        """
        if model:
            return [model]

        available = self.client.list_models()
        if not available:
            raise ValueError("No models found on server")

        if model_filter:
            models = [m for m in available if fnmatch.fnmatch(m, model_filter)]
            if not models:
                raise ValueError(f"No models matched filter: {model_filter}")
            return models

        if model_set:
            available_set = set(available)
            unknown = [m for m in model_set if m not in available_set]
            if unknown:
                logger.warning(f"Models not available on server: {unknown}")
            models = [m for m in model_set if m in available_set]
            if not models:
                raise ValueError(
                    "None of the models in the set are available on server"
                )
            return models

        if model_arch:
            arch_map = self.client.list_models_with_arch()
            pattern = model_arch.strip().lower()
            models = [
                key
                for key, arch in arch_map.items()
                if fnmatch.fnmatch((arch or "").lower(), pattern)
            ]
            if not models:
                raise ValueError(f"No models match architecture pattern: {pattern}")
            return models

        if all_models:
            return available

        raise ValueError("Must specify model selection criteria")

    def load_model(self, model_id: str) -> None:
        """Load a model."""
        self.client.load_model(model_id)

    def unload_model(self, model_id: Optional[str] = None) -> None:
        """Unload a model."""
        self.client.unload_model(model_id)

    def query_model(
        self, prompt: str, model_id: str, current: int = 0
    ) -> tuple[str, dict[str, Any]]:
        """Query a model with a prompt."""
        return self.client.query_model(prompt, model_id, current)
