"""
LM Studio implementation of the ModelClient interface.
This wraps the existing model_handling logic.
"""

import logging
from typing import Any, Optional

from interfaces.model_client import ModelClient
from models.model_handling import (
    is_lm_studio_server_running,
)
from models.model_handling import (
    list_models as _list_models,
)
from models.model_handling import (
    list_models_with_arch as _list_models_with_arch,
)
from models.model_handling import (
    load_model as _load_model,
)
from models.model_handling import (
    query_model as _query_model,
)
from models.model_handling import (
    unload_model as _unload_model,
)

logger = logging.getLogger(__name__)


class LMStudioClient(ModelClient):
    """LM Studio implementation of ModelClient interface."""

    def is_server_running(self) -> bool:
        """Check if LM Studio server is running."""
        return is_lm_studio_server_running()

    def query_model(
        self, prompt: str, model_id: str, current: int = 0
    ) -> tuple[str, dict[str, Any]]:
        """Query a model using LM Studio."""
        return _query_model(prompt, model_key=model_id, current=current)

    def load_model(self, model_id: str) -> None:
        """Load a model in LM Studio."""
        _load_model(model_id)

    def unload_model(self, model_id: Optional[str] = None) -> None:
        """Unload a model from LM Studio."""
        _unload_model(model_id)

    def list_models(self) -> list[str]:
        """List all available models in LM Studio."""
        return _list_models()

    def list_models_with_arch(self) -> dict[str, str]:
        """List models with their architectures from LM Studio."""
        return _list_models_with_arch()
