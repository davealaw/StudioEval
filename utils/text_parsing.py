import re
import unicodedata
from typing import Optional

# Precompiled regex for splitting off LM Studio-style control tokens
_CONTROL_SPLIT = re.compile(r"<\|")


def extract_mcq_letter(text: str, choices: str = "ABCD", marker: str = "Answer:") -> str:
    """
    Extract the MCQ letter (A-D/E) from model output.
    
    Args:
        text (str): The model output text
        choices (str): Valid choices, e.g., "ABCD" or "ABCDE"
        marker (str): The marker to look for, e.g., "Answer:" 
    Returns:
        str: The extracted letter, uppercased.
             If no valid letter is found, returns an empty string.
    """
    # 1) Find the last 'Answer:' (case-insensitive)
    idx = text.lower().rfind(marker.lower())
    if idx != -1:
        tail = text[idx + len(marker):]

        # 2) Stop at control tokens and take just the first line
        tail = _CONTROL_SPLIT.split(tail, 1)[0]
        line = tail.splitlines()[0] if "\n" in tail else tail
        line = line.strip()

        # 3) Prefer a clean single-letter on that line (allow () [] and trailing punctuation)
        m = re.match(fr"^[\(\[]?\s*([{choices}])\s*[\)\].,:;-]?$", line, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        # 4) Fallback within the tail: first occurrence of a valid letter
        m = re.search(fr"[{choices}]", tail, re.IGNORECASE)
        if m:
            return m.group(0).upper()

    # 5) Global fallback if no marker: last standalone letter anywhere
    matches = re.findall(fr"\b([{choices}])\b", text, re.IGNORECASE)
    return matches[-1].upper() if matches else ""

def extract_letter(text: str) -> str:
    """ 
    Extracts a single-letter answer from model output.  
    This is a generic version that defaults to choices A-D and marker 'Answer:'. 
    """
    return extract_mcq_letter(text, choices="ABCD", marker="Answer:")

def extract_letterToE(text: str) -> str:
    """ 
    Extracts a single-letter answer from model output, allowing choices A-E.
    This is a generic version that defaults to choices A-E and marker 'Answer:'. 
    """
    return extract_mcq_letter(text, choices="ABCDE", marker="Answer:")

def extract_corrected_text(text: str) -> str:
    """
    Extracts the corrected sentence from model output.

    Args:
        text (str): The model output text
    Returns:
        str: The extracted corrected text, stripped of leading/trailing whitespace.
             If 'Corrected:' is not found, returns the original text stripped.
    """
    marker = "corrected:"
    idx = text.lower().rfind(marker)
    if idx != -1:
        result = text[idx + len(marker):].strip()
        return result
    return text.strip()

def normalize(text: str) -> str:
    """
    Normalizes text by:
    - Stripping leading/trailing whitespace
    - Lowercasing
    - Replacing various punctuation with standard forms
    Args:
        text (str): The text to normalize
    Returns:
        str: The normalized text
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("…", "...").replace("–", "-").replace("—", "-")
    text = text.replace("\u00A0", " ")
    return text.strip().lower()

def extract_numeric_answer(text: str) -> Optional[float]:
    """
    Extracts a numeric answer from model output.
    Prefers a value after 'Answer:' if present, otherwise falls back to first numeric found.

    Args:
        text (str): The model output text
    Returns:
        float or None: The extracted numeric value, or None if no numeric value is found
    """
    # Try to match specifically after "Answer:"
    match = re.search(r"Answer:\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Fallback: first numeric in the text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(nums[0]) if nums else None

# Only used in evaluate_gsm8k_dataset
def extract_answer_from_solution(solution_text: str) -> Optional[float]:
    """
    Extract final numeric answer after '####'.
    Args:
        solution_text (str): The solution text containing the answer
    Returns:
        float or None: The extracted numeric answer, or None if not found
    """
    match = re.search(r"####\s*([-+]?[0-9]*\.?[0-9]+)", solution_text)
    return float(match.group(1)) if match else None
