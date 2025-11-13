from typing import Any, Optional


def merge_eval_kwargs(
    config: dict[str, Any],
    cli_args: Any,
    keys: list[str],
    cli_explicit_args: Optional[set[str]] = None,
) -> dict[str, Any]:
    """
    Merges evaluation parameters with proper precedence:
    1. Explicitly set CLI args (highest priority)
    2. Config file values (medium priority)
    3. CLI default values (lowest priority)

    Args:
        config (dict): Dataset-specific parameters.
        cli_args (argparse.Namespace | object): CLI arguments containing
            parameters.
        keys (list): List of keys to merge from both sources.
        cli_explicit_args (set | None): CLI argument names explicitly set by
            the user. If None, treats all non-None CLI values as explicit.

    Returns:
        dict: Merged dictionary with parameters from both sources.
    """
    kwargs = {}
    for key in keys:
        cli_value = getattr(cli_args, key, None)
        config_value = config.get(key)

        # Determine if CLI arg was explicitly set
        cli_was_explicit = (
            cli_explicit_args is None or key in cli_explicit_args
            if cli_explicit_args
            else False
        )

        if cli_was_explicit and cli_value is not None:
            # Explicit CLI args have highest priority
            kwargs[key] = cli_value
        elif config_value is not None:
            # Config values have medium priority
            kwargs[key] = config_value
        elif cli_value is not None:
            # CLI defaults have lowest priority
            kwargs[key] = cli_value

    return kwargs
