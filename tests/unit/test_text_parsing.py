"""
Unit tests for text parsing utilities.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from utils.text_parsing import (
    extract_answer_from_solution,
    extract_corrected_text,
    extract_letter,
    extract_letter_to_e,
    extract_mcq_letter,
    extract_numeric_answer,
    normalize,
)


class TestMCQExtraction:
    """Test MCQ letter extraction functionality."""

    def test_extract_simple_answer(self):
        """Test basic letter extraction."""
        text = "I think the answer is B."
        assert extract_mcq_letter(text) == "B"

    def test_extract_with_marker(self):
        """Test extraction with 'Answer:' marker."""
        text = "Let me think... Answer: A"
        assert extract_mcq_letter(text) == "A"

    def test_extract_with_formatting(self):
        """Test extraction with various formatting."""
        assert extract_mcq_letter("Answer: (C)") == "C"
        assert extract_mcq_letter("Answer: [B]") == "B"
        assert extract_mcq_letter("Answer: A.") == "A"
        assert extract_mcq_letter("Answer: D,") == "D"

    def test_extract_case_insensitive(self):
        """Test case insensitive extraction."""
        assert extract_mcq_letter("answer: a") == "A"
        assert extract_mcq_letter("ANSWER: b") == "B"

    def test_no_valid_letter(self):
        """Test when no valid letter is found."""
        text = "I don't know the answer"
        assert extract_mcq_letter(text) == ""

    def test_invalid_choices(self):
        """Test with letters outside valid choices."""
        text = "Answer: Z"
        assert extract_mcq_letter(text, choices="ABCD") == ""

    def test_multiple_answers_last_wins(self):
        """Test that the last occurrence wins."""
        text = "First I thought A, but actually B, no wait, definitely C"
        assert extract_mcq_letter(text) == "C"

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("The answer is definitely A!", "A"),
            ("I choose option B.", "B"),
            ("Answer: D) This is correct", "D"),
            ("Multiple A B letters, last B wins", "B"),
            ("Answer: A\nBut actually B", "A"),  # First line after marker wins
            ("No marker here, just C somewhere", "C"),
            ("Answer is (A)", "A"),
            ("Final answer: B", "B"),
        ],
    )
    def test_various_formats(self, text, expected):
        """Test various text formats."""
        assert extract_mcq_letter(text) == expected

    def test_custom_choices(self):
        """Test with custom choice sets."""
        assert extract_mcq_letter("Answer: E", choices="ABCDE") == "E"
        assert extract_mcq_letter("Answer: F", choices="ABCDEF") == "F"

    def test_custom_marker(self):
        """Test with custom markers."""
        assert extract_mcq_letter("Final choice: B", marker="Final choice:") == "B"
        assert extract_mcq_letter("Result: C", marker="Result:") == "C"


class TestSpecializedExtractors:
    """Test specialized extractor functions."""

    def test_extract_letter_default(self):
        """Test default letter extractor (A-D)."""
        assert extract_letter("Answer: A") == "A"
        assert extract_letter("Answer: E") == ""  # E not in ABCD

    def test_extract_letter_to_e(self):
        """Test A-E letter extractor."""
        assert extract_letter_to_e("Answer: E") == "E"
        assert extract_letter_to_e("Answer: A") == "A"
        assert extract_letter_to_e("Answer: F") == ""  # F not in ABCDE


class TestCorrectedTextExtraction:
    """Test corrected text extraction for grammar evaluation."""

    def test_extract_with_corrected_marker(self):
        """Test extraction with 'Corrected:' marker."""
        text = "The sentence has issues. Corrected: This sentence is correct."
        expected = "This sentence is correct."
        assert extract_corrected_text(text) == expected

    def test_extract_case_insensitive(self):
        """Test case insensitive marker matching."""
        text = "CORRECTED: Fixed sentence here."
        assert extract_corrected_text(text) == "Fixed sentence here."

    def test_extract_without_marker(self):
        """Test fallback when no marker present."""
        text = "  This is the full response.  "
        assert extract_corrected_text(text) == "This is the full response."

    def test_extract_last_occurrence(self):
        """Test that last 'Corrected:' occurrence wins."""
        text = "Corrected: Wrong. Actually, Corrected: Right answer."
        assert extract_corrected_text(text) == "Right answer."

    def test_extract_multiline(self):
        """Test extraction with multiline responses."""
        text = "Let me think.\nCorrected: The final corrected sentence."
        assert extract_corrected_text(text) == "The final corrected sentence."


class TestTextNormalization:
    """Test text normalization functionality."""

    def test_normalize_basic(self):
        """Test basic normalization."""
        assert normalize("  Hello World!  ") == "hello world!"
        assert normalize("UPPERCASE") == "uppercase"

    def test_normalize_unicode_quotes(self):
        """Test Unicode quote normalization."""
        assert normalize('"smart quotes"') == '"smart quotes"'
        assert normalize("'apostrophe'") == "'apostrophe'"

    def test_normalize_unicode_dashes(self):
        """Test Unicode dash normalization."""
        assert normalize("em—dash") == "em-dash"
        assert normalize("en–dash") == "en-dash"  # noqa: RUF001
        assert normalize("ellipsis…") == "ellipsis..."

    def test_normalize_unicode_spaces(self):
        """Test Unicode space normalization."""
        # Test with actual non-breaking space character
        assert normalize("non\u00a0breaking\u00a0space") == "non breaking space"

    def test_normalize_nfkc(self):
        """Test Unicode NFKC normalization."""
        # Test that normalization handles composed characters
        text_with_composed = "café"  # é as composed character
        normalized = normalize(text_with_composed)
        assert "café" in normalized.lower()

    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("  Mixed   Spacing  ", "mixed   spacing"),
            ('Quote"Test"', 'quote"test"'),
            ("Multiple—Different–Dashes", "multiple-different-dashes"),  # noqa: RUF001
            ("", ""),
            ("NoChangesNeeded", "nochangesneeded"),
        ],
    )
    def test_normalize_various_cases(self, input_text, expected):
        """Test normalization with various inputs."""
        assert normalize(input_text) == expected


class TestNumericExtraction:
    """Test numeric answer extraction."""

    def test_extract_with_answer_marker(self):
        """Test extraction with 'Answer:' marker."""
        assert extract_numeric_answer("Answer: 42") == 42.0
        assert extract_numeric_answer("Answer: 3.14") == 3.14
        assert extract_numeric_answer("Answer: -5") == -5.0

    def test_extract_fallback_first_number(self):
        """Test fallback to first number found."""
        assert extract_numeric_answer("The result is 123 and then 456") == 123.0
        assert extract_numeric_answer("No marker, just 99.5 here") == 99.5

    def test_extract_no_number(self):
        """Test when no number is found."""
        assert extract_numeric_answer("No numbers here at all") is None
        assert extract_numeric_answer("") is None

    def test_extract_decimal_numbers(self):
        """Test decimal number extraction."""
        assert extract_numeric_answer("Answer: 0.5") == 0.5
        assert extract_numeric_answer("Answer: 100.00") == 100.0

    def test_extract_negative_numbers(self):
        """Test negative number extraction."""
        assert extract_numeric_answer("Answer: -42") == -42.0
        assert extract_numeric_answer("The deficit is -15.5") == -15.5

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Answer: 1", 1.0),
            ("Answer: 0", 0.0),
            ("Answer: 999", 999.0),
            ("Answer: 3.14159", 3.14159),
            ("Calculate: 2+2=4", 2.0),  # First number
            ("No Answer: but 7 is here", 7.0),
        ],
    )
    def test_numeric_extraction_cases(self, text, expected):
        """Test various numeric extraction cases."""
        assert extract_numeric_answer(text) == expected


class TestGSM8KExtraction:
    """Test GSM8K-specific answer extraction."""

    def test_extract_gsm8k_basic(self):
        """Test basic GSM8K answer extraction."""
        solution = "Step 1: ... Step 2: ... #### 42"
        assert extract_answer_from_solution(solution) == 42.0

    def test_extract_gsm8k_decimal(self):
        """Test GSM8K decimal answer extraction."""
        solution = "Calculate... #### 3.14"
        assert extract_answer_from_solution(solution) == 3.14

    def test_extract_gsm8k_negative(self):
        """Test GSM8K negative answer extraction."""
        solution = "The deficit is #### -25"
        assert extract_answer_from_solution(solution) == -25.0

    def test_extract_gsm8k_no_marker(self):
        """Test when no #### marker found."""
        solution = "No marker in this solution"
        assert extract_answer_from_solution(solution) is None

    def test_extract_gsm8k_with_spacing(self):
        """Test GSM8K extraction with various spacing."""
        assert extract_answer_from_solution("####42") == 42.0
        assert extract_answer_from_solution("#### 42") == 42.0
        assert extract_answer_from_solution("####    42") == 42.0


# Property-based testing for robustness
class TestTextParsingProperties:
    """Property-based tests for text parsing robustness."""

    @given(st.text())
    def test_normalize_never_crashes(self, text):
        """Test that normalize never crashes on arbitrary input."""
        result = normalize(text)
        assert isinstance(result, str)

    @given(st.text())
    def test_extract_corrected_never_crashes(self, text):
        """Test that extract_corrected_text never crashes."""
        result = extract_corrected_text(text)
        assert isinstance(result, str)

    @given(st.text())
    def test_extract_mcq_letter_robustness(self, text):
        """Test MCQ extraction robustness."""
        result = extract_mcq_letter(text)
        assert isinstance(result, str)
        if result:  # If not empty, should be valid choice
            assert result in "ABCD"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_strings(self):
        """Test behavior with empty strings."""
        assert extract_mcq_letter("") == ""
        assert extract_corrected_text("") == ""
        assert normalize("") == ""
        assert extract_numeric_answer("") is None

    def test_whitespace_only(self):
        """Test behavior with whitespace-only strings."""
        # Test with actual whitespace characters
        whitespace_string = "   \n\t  "
        assert extract_mcq_letter(whitespace_string) == ""
        # extract_corrected_text strips all leading/trailing whitespace
        assert extract_corrected_text(whitespace_string) == ""
        assert normalize(whitespace_string) == ""

    def test_very_long_strings(self):
        """Test behavior with very long strings."""
        long_text = "x" * 10000 + " Answer: B " + "y" * 10000
        assert extract_mcq_letter(long_text) == "B"

    def test_unicode_edge_cases(self):
        """Test Unicode edge cases."""
        # Test with various Unicode characters
        assert normalize("Héllo Wörld") == "héllo wörld"
        assert extract_mcq_letter("Ответ: A") == "A"  # Cyrillic text
