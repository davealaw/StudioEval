from typing import Any, Dict, List, Optional, Set, Union


def merge_eval_kwargs(config: Dict[str, Any], cli_args: Any, keys: List[str], 
                     cli_explicit_args: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    Merges evaluation parameters with proper precedence:
    1. Explicitly set CLI args (highest priority)
    2. Config file values (medium priority)  
    3. CLI default values (lowest priority)

    Args:
        config (dict): Configuration dictionary containing dataset-specific parameters.
        cli_args (argparse.Namespace or object): CLI arguments containing parameters.
        keys (list): List of keys to merge from both sources.
        cli_explicit_args (set, optional): Set of CLI argument names that were explicitly set by user.
                                          If None, treats all non-None CLI values as explicit.

    Returns:
        dict: Merged dictionary with parameters from both sources.
    """
    kwargs = {}
    for key in keys:
        cli_value = getattr(cli_args, key, None)
        config_value = config.get(key)
        
        # Determine if CLI arg was explicitly set
        cli_was_explicit = (
            cli_explicit_args is None or 
            key in cli_explicit_args if cli_explicit_args else False
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
