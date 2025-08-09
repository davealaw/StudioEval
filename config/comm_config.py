import json
import logging
from copy import deepcopy
from typing import Any, Dict

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "timeout": 180,  # seconds
    "GENERATION_PARAMS": {
        "temperature": 0.0,
        "topPSampling": 1.0,
        "topKSampling": 0,
        "repeatPenalty": 1.00,
        "maxTokens": 2000 # default 512
    }
}

_config: Dict[str, Any] = deepcopy(DEFAULT_CONFIG)


def _validate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and coerce types; fallback to defaults with warnings when invalid."""
    out = deepcopy(DEFAULT_CONFIG)

    # LMSTUDIO_URL
    url = cfg.get("LMSTUDIO_URL", out["LMSTUDIO_URL"])
    if not isinstance(url, str) or not url.strip():
        logger.warning("Invalid LMSTUDIO_URL in comm config; using default.")
    else:
        out["LMSTUDIO_URL"] = url.strip()

    # timeout
    timeout = cfg.get("timeout", out["timeout"])
    if not isinstance(timeout, int) and not isinstance(timeout, float):
        logger.warning("Invalid timeout in comm config; using default.")
    else:
        timeout = int(timeout)
        if timeout <= 0:
            logger.warning("Non-positive timeout in comm config; using default.")
        else:
            out["timeout"] = timeout

    # GENERATION_PARAMS
    gen = cfg.get("GENERATION_PARAMS", {})
    if not isinstance(gen, dict):
        logger.warning("GENERATION_PARAMS must be an object; using defaults.")
    else:
        merged = deepcopy(out["GENERATION_PARAMS"])
        for k, v in gen.items():
            if k not in merged:
                # Unknown keys are allowed (forward-compat) but warn once.
                logger.debug(f"GENERATION_PARAMS: unknown key '{k}' will be included.")
                merged[k] = v
                continue
            # Basic numeric validation for known fields
            if isinstance(v, (int, float)):
                merged[k] = v
            else:
                logger.warning(f"GENERATION_PARAMS['{k}'] must be numeric; keeping default.")
        out["GENERATION_PARAMS"] = merged

    return out


def load_comm_config(path: str) -> None:
    """
    Load communication config from JSON file, merge with defaults, and validate.
    Safe to call multiple times; last successful load wins.
    """
    global _config
    try:
        with open(path, "r") as f:
            raw = json.load(f)
        _config = _validate(raw)
        logger.info(f"Loaded communication config from {path}.")
    except FileNotFoundError:
        logger.warning(f"Communication config not found at '{path}'. Using defaults.")
        _config = deepcopy(DEFAULT_CONFIG)
    except json.JSONDecodeError as e:
        logger.warning(
            f"Invalid JSON in communication config '{path}': {e}. Using defaults."
        )
        _config = deepcopy(DEFAULT_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to load communication config '{path}': {e}. Using defaults.")
        _config = deepcopy(DEFAULT_CONFIG)


def get_comm_config() -> Dict[str, Any]:
    """Return the effective, validated config (defaults if not loaded)."""
    return _config
