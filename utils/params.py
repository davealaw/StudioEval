def merge_eval_kwargs(config, cli_args, keys):
    """
    Merges evaluation parameters from a configuration dictionary and command line arguments.

    Args:
        config (dict): Configuration dictionary containing default parameters.
        cli_args (argparse.Namespace): Command line arguments containing user-specified parameters.
        keys (list): List of keys to merge from both sources.

    Returns:
        dict: Merged dictionary with parameters from both sources.
    """
    kwargs = {}
    for key in keys:
        if config.get(key) is not None:
            kwargs[key] = config[key]
        elif getattr(cli_args, key, None) is not None:
            kwargs[key] = getattr(cli_args, key)
    return kwargs
