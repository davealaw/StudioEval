def format_duration(seconds: float) -> str:
    """
    Convert a duration in seconds to a human-readable string.
    
    Args:
        seconds (float): Duration in seconds.
    Returns:
        str: Formatted duration string.

    Examples:
        3723.45 → "1h 2m 3.45s"
        75.8    → "1m 15.80s"
        9.21    → "9.21s"
    """
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes >= 1:
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"
