import logging

def setup_logging(log_level="INFO", log_file=None):
    """
    Sets up logging configuration for the application.
    
    Args:
        log_level (str): The logging level to set (DEBUG, INFO, WARNING, ERROR).
        log_file (str): Optional file path to write logs to. If None, logs will be printed to console.
    """
    log_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove default handlers if re-running
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
