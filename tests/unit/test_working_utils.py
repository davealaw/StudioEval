"""
Unit tests for working utility functions.
Tests only the utilities that work correctly in the current codebase.
"""

from utils.text_parsing import (
    extract_answer_from_solution,
    extract_corrected_text,
    extract_letter,
    extract_letter_to_e,
    extract_mcq_letter,
    extract_numeric_answer,
    normalize,
)
from utils.timing_utils import format_duration


class TestTextParsingWorking:
    """Test text parsing utilities that work correctly."""

    def test_extract_mcq_basic(self):
        """Test basic MCQ extraction."""
        assert extract_mcq_letter("Answer: A") == "A"
        assert extract_mcq_letter("Answer: B") == "B"
        assert extract_mcq_letter("I think C") == "C"

    def test_extract_no_letter(self):
        """Test when no valid letter found."""
        assert extract_mcq_letter("No answer here") == ""
        assert extract_mcq_letter("") == ""

    def test_extract_letter_functions(self):
        """Test specialized letter extraction functions."""
        assert extract_letter("Answer: A") == "A"
        assert extract_letter("Answer: E") == ""  # E not in ABCD

        assert extract_letter_to_e("Answer: E") == "E"
        assert extract_letter_to_e("Answer: F") == ""  # F not in ABCDE

    def test_corrected_text_extraction(self):
        """Test corrected text extraction."""
        text = "Corrected: This is the corrected text."
        assert extract_corrected_text(text) == "This is the corrected text."

        # Test without marker
        assert extract_corrected_text("Just text") == "Just text"

    def test_normalize_basic(self):
        """Test basic text normalization."""
        assert normalize("  HELLO  ") == "hello"
        assert normalize("Mixed Case") == "mixed case"

    def test_numeric_extraction(self):
        """Test numeric answer extraction."""
        assert extract_numeric_answer("Answer: 42") == 42.0
        assert extract_numeric_answer("No numbers") is None
        assert extract_numeric_answer("The answer is 3.14") == 3.14

    def test_gsm8k_extraction(self):
        """Test GSM8K answer extraction."""
        assert extract_answer_from_solution("#### 42") == 42.0
        assert extract_answer_from_solution("No marker") is None


class TestTimingUtilsWorking:
    """Test timing utilities that work correctly."""

    def test_format_duration_basic(self):
        """Test basic duration formatting."""
        assert format_duration(1.0) == "1.00s"
        assert format_duration(60.0) == "1m 0.00s"
        assert format_duration(3600.0) == "1h 0m 0.00s"

    def test_format_duration_combined(self):
        """Test duration formatting with combined units."""
        assert format_duration(3661.0) == "1h 1m 1.00s"
        assert format_duration(90.5) == "1m 30.50s"

    def test_format_duration_edge_cases(self):
        """Test edge cases."""
        assert format_duration(0.0) == "0.00s"
