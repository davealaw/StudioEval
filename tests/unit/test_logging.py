"""
Unit tests for logging utilities.
"""

import logging
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from utils.logging import setup_logging


class TestLoggingSetup:
    """Test logging setup functionality."""

    def test_setup_logging_default(self):
        """Test default logging setup."""
        setup_logging()

        logger = logging.getLogger()
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1  # Console handler only

    def test_setup_logging_debug_level(self):
        """Test logging setup with DEBUG level."""
        setup_logging(log_level="DEBUG")

        logger = logging.getLogger()
        assert logger.level == logging.DEBUG

    def test_setup_logging_error_level(self):
        """Test logging setup with ERROR level."""
        setup_logging(log_level="ERROR")

        logger = logging.getLogger()
        assert logger.level == logging.ERROR

    def test_setup_logging_invalid_level(self):
        """Test logging setup with invalid level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log level: INVALID"):
            setup_logging(log_level="INVALID")

    def test_setup_logging_case_insensitive(self):
        """Test that log levels are case insensitive."""
        setup_logging(log_level="info")

        logger = logging.getLogger()
        assert logger.level == logging.INFO

        setup_logging(log_level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            log_file_path = temp_file.name

        try:
            setup_logging(log_level="INFO", log_file=log_file_path)

            logger = logging.getLogger()
            assert len(logger.handlers) == 2  # Console + file handlers

            # Test that logging works
            logger.info("Test log message")

            # Verify file was created and has content
            assert os.path.exists(log_file_path)
            with open(log_file_path) as f:
                content = f.read()
                assert "Test log message" in content

        finally:
            if os.path.exists(log_file_path):
                os.unlink(log_file_path)

    def test_setup_logging_file_permission_error(self):
        """Test handling of file permission errors."""
        # Try to write to an invalid path
        invalid_path = "/root/cannot_write_here.log"

        # Should not raise exception, but should log error
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            mock_logger.hasHandlers.return_value = False
            mock_logger.handlers = []

            setup_logging(log_level="INFO", log_file=invalid_path)

            # Should have called logger.error for permission issue
            mock_logger.error.assert_called()

    def test_setup_logging_clears_existing_handlers(self):
        """Test that existing handlers are cleared when re-setting up."""
        # First setup
        setup_logging(log_level="INFO")
        logger = logging.getLogger()
        initial_handler_count = len(logger.handlers)

        # Second setup should clear and re-add handlers
        setup_logging(log_level="DEBUG")
        final_handler_count = len(logger.handlers)

        # Should have same number of handlers, but logger level should change
        assert final_handler_count == initial_handler_count
        assert logger.level == logging.DEBUG

    def test_setup_logging_custom_formats(self):
        """Test custom console and file formats."""
        custom_console = "CONSOLE: %(message)s"
        custom_file = "FILE: %(levelname)s - %(message)s"

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            log_file_path = temp_file.name

        try:
            setup_logging(
                log_level="INFO",
                log_file=log_file_path,
                console_format=custom_console,
                file_format=custom_file,
            )

            logger = logging.getLogger()
            assert len(logger.handlers) == 2

            # Check that formatters are applied. It's hard to assert the exact
            # output, but the call should not fail.
            logger.info("Test message with custom formats")

            # Verify file has content
            with open(log_file_path) as f:
                content = f.read()
                assert "Test message with custom formats" in content

        finally:
            if os.path.exists(log_file_path):
                os.unlink(log_file_path)

    def test_setup_logging_utf8_encoding(self):
        """Test that file logging uses UTF-8 encoding."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, encoding="utf-8"
        ) as temp_file:
            log_file_path = temp_file.name

        try:
            setup_logging(log_level="INFO", log_file=log_file_path)

            logger = logging.getLogger()

            # Test logging unicode characters
            unicode_message = "Test with unicode: 🎉 测试 éñ"
            logger.info(unicode_message)

            # Verify unicode was written correctly
            with open(log_file_path, encoding="utf-8") as f:
                content = f.read()
                assert unicode_message in content

        finally:
            if os.path.exists(log_file_path):
                os.unlink(log_file_path)


class TestLoggingEdgeCases:
    """Test edge cases and error conditions."""

    def test_logging_level_validation(self):
        """Test various invalid log levels."""
        invalid_levels = ["INVALID", "", "123", None]

        for invalid_level in invalid_levels:
            # All invalid levels should raise ValueError (None gets converted to string)
            with pytest.raises(ValueError):
                setup_logging(log_level=invalid_level)

    def test_file_handler_creation_edge_cases(self):
        """Test edge cases in file handler creation."""
        # Test with directory that doesn't exist
        nonexistent_dir = "/path/that/does/not/exist/test.log"

        # Should handle gracefully and log error
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            mock_logger.hasHandlers.return_value = False
            mock_logger.handlers = []

            setup_logging(log_file=nonexistent_dir)

            # Should have attempted to log the error
            mock_logger.error.assert_called()

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    def test_all_valid_log_levels(self, level):
        """Test all valid log levels work correctly."""
        setup_logging(log_level=level)

        logger = logging.getLogger()
        expected_level = getattr(logging, level)
        assert logger.level == expected_level
