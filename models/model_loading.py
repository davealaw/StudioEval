import json
import logging

logger = logging.getLogger(__name__)


def load_model_set_file(path: str):
    """
    Returns a list of model IDs from a file.
    The file can be in plain text or JSON format.
    - Plain text: one model ID per line, '#' for comments, blanks ignored
    - JSON: either {"models": [...]} or a top-level JSON list ["id1", "id2"]

    If the file is not found or has invalid content, returns an empty list.

    Args:
        path (str): Path to the model set file.

    Returns:
        list: List of model IDs as strings.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the JSON content is invalid.

    Example:
        load_model_set_file("models.txt")
        load_model_set_file("models.json")

    Example:
        # models.txt
        model1
        model2  # This is a comment

        # models.json
        {"models": ["model1", "model2"]}
        ["model1", "model2"]
    """
    try:
        with open(path) as f:
            text = f.read().strip()
        # JSON?
        if path.endswith(".json"):
            data = json.loads(text)
            if (
                isinstance(data, dict)
                and "models" in data
                and isinstance(data["models"], list)
            ):
                return [str(m).strip() for m in data["models"] if str(m).strip()]
            if isinstance(data, list):
                return [str(m).strip() for m in data if str(m).strip()]
            logger.error(
                "Invalid JSON structure in model-set file. Use "
                "{'models': [...]} or a JSON list."
            )
            return []
        # Plain text
        lines = [ln.strip() for ln in text.splitlines()]
        return [ln for ln in lines if ln and not ln.startswith("#")]
    except FileNotFoundError:
        logger.error(f"Model set file not found: {path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON in model set file: {path} ({e})")
        return []
