import logging
from typing import Optional

class DropWsNoise(logging.Filter):
    """    
    Custom logging filter to drop websocket-related noise from logs.
    This filter ignores log messages that contain specific websocket-related keywords
    to reduce clutter in the logs.
    """
    NEEDLES = (
        "Websocket handling thread started",
        "Websocket handling task started",
        "Websocket session established",
        "Websocket session disconnected",
        "Websocket closed",
        "Websocket demultiplexing task terminated",
        "Websocket task terminated",
        'Websocket thread terminated',
        'HTTP Request: GET ws://',
    )
    def filter(self, record):
        msg = record.getMessage()
        return not any(n in msg for n in self.NEEDLES)

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None,
                 console_format: str = '%(message)s',
                 file_format: str = '%(asctime)s - %(levelname)s - %(message)s') -> None:
    """
    Sets up logging configuration for the application.
    
    Args:
        log_level (str): The logging level (DEBUG, INFO, WARNING, ERROR)
        log_file (str, optional): File path to write logs to. If None, console only
        console_format (str): Format string for console output
        file_format (str): Format string for file output
        
    Raises:
        ValueError: If log_level is invalid
    """
    # Validate and convert log level
    try:
        numeric_level = getattr(logging, log_level.upper())
    except AttributeError:
        raise ValueError(f"Invalid log level: {log_level}. Must be DEBUG, INFO, WARNING, or ERROR")

    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler - always present
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(console_format))
    if numeric_level != logging.DEBUG:
        console_handler.addFilter(DropWsNoise())
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(file_format))
            logger.addHandler(file_handler)
        except (OSError, IOError) as e:
            logger.error(f"Failed to create log file '{log_file}': {e}")
