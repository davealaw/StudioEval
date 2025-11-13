import logging

import lmstudio

from config.comm_config import get_comm_config

logger = logging.getLogger(__name__)

def _get_lmstudio_error_cls() -> type[Exception]:
    try:
        return lmstudio.LMStudioError  # type: ignore[attr-defined]
    except AttributeError:

        class _LMStudioError(Exception):
            """Fallback LM Studio error type when the client does not expose one."""

            pass

        return _LMStudioError


LMStudioError = _get_lmstudio_error_cls()


def log_all_fields(label, obj):
    """
    Logs all accessible fields of an object, excluding private attributes.
    This is useful for debugging and inspecting model objects.

    Args:
        label (str): A label to identify the object being inspected.
        obj (object): The object whose fields are to be logged.

    Returns:
        None
    """
    logger.info(f"🔍 Inspecting {label}")
    for attr in dir(obj):
        if not attr.startswith("_"):
            try:
                value = getattr(obj, attr)
                logger.info(f"{label}.{attr} = {value}")
            except AttributeError as e:
                logger.warning(f"{label}.{attr} could not be accessed: {e}")


def is_lm_studio_server_running():
    """
    Checks if LM Studio is running by listing currently loaded models via IPC.
    """
    try:
        _ = lmstudio.list_loaded_models()
        return True
    except (LMStudioError, OSError) as e:
        logger.error(f"LM Studio IPC check failed: {e}")
        return False


def query_model(prompt, model_key="local-model", current=0):
    """
    Queries a model using the lmstudio-python client.
    Args:
        prompt (str): The input prompt to send to the model.
        model_key (str): The key of the model to query.
        current (int): The current question index for logging purposes.
    Returns:
        tuple: A tuple containing the model's response and statistics.
    """
    cfg = get_comm_config()
    logger.debug(f"📤 Prompt sent to model (question {current + 1}):\n{prompt}\n")

    try:
        model = lmstudio.llm(model_key)

        # Not thinking improves performance for some Qwen3 thinking models.
        # However, results are worse with all but Grammar and Creative writing
        # evaluations, so the option remains disabled by default.
        # if (model.get_info().architecture.startswith('qwen3')):
        #    prompt += " /no_think"

        system_prompt = cfg.get("SYSTEM_PROMPT", "You are a helpful assistant.")
        chat = lmstudio.Chat(system_prompt)
        chat.add_user_message(prompt)

        gen_params = cfg.get("GENERATION_PARAMS", {})

        response = model.respond(chat, config={**gen_params})

        output = response.content.strip()

        stats = {
            "tokens_per_second": response.stats.tokens_per_second,
            "prompt_tokens": response.stats.prompt_tokens_count,
            "completion_tokens": response.stats.predicted_tokens_count,
            "total_tokens": response.stats.total_tokens_count,
            "stop_reason": response.stats.stop_reason,
            "structured": response.structured,
        }

        return output, stats

    except (LMStudioError, OSError) as e:
        logger.error(f"Error querying model via lmstudio-python: {e}")
        return "", {
            "tokens_per_second": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "stop_reason": "error",
            "structured": False,
        }


def list_models():
    """
    This function uses the LM Studio Client to list models via IPC.
    It retrieves the list of downloaded models and extracts their model keys.
    If an error occurs, it logs the error and returns an empty list.

    Returns:
        list: A list of model keys for all downloaded LLMs.
        If an error occurs, an empty list is returned.
    """
    try:
        models = lmstudio.list_downloaded_models("llm")
        _ = models[0].info.architecture  # Trigger loading of model info
        return [m.model_key for m in models if hasattr(m, "model_key")]
    except (LMStudioError, OSError, IndexError) as e:
        logger.error(f"Failed to list downloaded models via IPC: {e}")
        return []


def list_models_with_arch():
    """
    This function uses the LM Studio Client to list models via IPC.
    It retrieves the list of downloaded models and extracts their model keys
    and architectures. If an error occurs, it logs the error and returns an
    empty dict.

    Returns:
        dict: Mapping of model keys to their architectures for all downloaded
            LLMs. If an error occurs, an empty dict is returned.
    """
    try:
        models = lmstudio.list_downloaded_models("llm")
        return {
            m.model_key: getattr(m.info, "architecture", "").strip().lower()
            for m in models
            if hasattr(m, "model_key") and hasattr(m, "info")
        }
    except (LMStudioError, OSError) as e:
        logger.error(f"Failed to list downloaded models via IPC: {e}")
        return {}


def list_loaded_models():
    """
    This function uses the LM Studio Client to list models via IPC.
    If an error occurs, it logs the error and returns an empty list.

    Returns:
        list: A list of currently loaded model keys.
        If an error occurs, an empty list is returned.
    """
    try:
        return lmstudio.list_loaded_models("llm")
    except (LMStudioError, OSError) as e:
        logger.error(f"Failed to list loaded models: {e}")
        return []


def load_model(model_key):
    """
    Loads a model using its model_key via LM Studio Client.

    Args:
        model_key (str): The key of the model to load.
    Returns:
        None
    """
    try:
        logger.info(f"🔄 Loading model: {model_key}...")
        lmstudio.llm(model_key)
        logger.info(f"✅ Loaded model: {model_key}")
    except (LMStudioError, OSError) as e:
        logger.error(f"Failed to load model {model_key}: {e}")


def unload_model(model_key=None):
    """
    Unloads the currently loaded model via LM Studio Client.
    If model_key is provided, it unloads that specific model.

    Args:
        model_key (str, optional): Key of the model to unload. If None, the
            active model is unloaded.

    Returns:
        None
    """
    try:
        model = lmstudio.llm()
        model.unload()
        logger.info(
            f"✅ Unloaded model: {model_key if model_key else '[active model]'}"
        )
    except (LMStudioError, OSError) as e:
        logger.error(f"Failed to unload model: {e}")
