# hellaswag.py
import logging
import time

from tqdm import tqdm

from models.model_handling import query_model
from utils.data_loading import load_dataset_with_config
from utils.text_parsing import extract_letter

logger = logging.getLogger(__name__)


def evaluate_hellaswag(
    model_id,
    dataset_path="tinyBenchmarks/tinyHellaswag",
    dataset_name="tinyHellaSwag",
    subset=None,
    split="validation",
    seed=42,
    sample_size=0,
):
    """
    Evaluate HellaSwag dataset with a language model.

    HellaSwag tests commonsense reasoning by asking models to choose the most
    likely continuation of a given context from 4 options. It requires models
    to understand everyday scenarios and predict plausible next steps.

    Dataset structure expected:
      - ctx: str                 (context/stem)
      - endings: list[str]       (4 answer options; HellaSwag uses 4)
      - label: str (0-based)     (correct option index as string "0"-"3")

    Model must respond strictly with: 'Answer: A'/'B'/'C'/'D' (letter only extraction).
    Other fields (activity_label, source_id, split, etc.) are ignored.

    Args:
        model_id (str): Identifier for the model to query.
        dataset_path (str): Path to the HellaSwag dataset (default: tinyHellaSwag).
        dataset_name (str): Name of the dataset for logging.
        subset (str): Optional subset of the dataset to evaluate.
        split (str): Dataset split to evaluate (default is "validation").
        seed (int): Random seed for reproducibility.
        sample_size (int): Number of samples to evaluate, 0 means all.

    Returns:
        dict: Evaluation results including accuracy and tokens per second.
    """
    dataset = load_dataset_with_config(
        dataset_path, subset=subset, split=split, seed=seed, sample_size=sample_size
    )

    total = 0
    correct = 0
    skipped = 0
    tokens_per_second_total = 0.0

    for item in tqdm(dataset, desc=f"⏳ Evaluating {dataset_name}"):
        # ---- Validate & extract core fields ----
        try:
            ctx = str(item.get("ctx", "")).strip()
            endings = item.get("endings", [])
            raw_label = item.get("label", None)

            if (
                not ctx
                or not isinstance(endings, list)
                or len(endings) < 2
                or raw_label is None
            ):
                skipped += 1
                continue

            # HellaSwag is 4-way; we keep generic in case of variants
            n_choices = len(endings)

            # Handle string or int labels (HellaSwag typically uses strings)
            try:
                label = int(raw_label)
            except (ValueError, TypeError):
                logger.debug(f"Invalid label format: {raw_label}, skipping item")
                skipped += 1
                continue

            # Ensure label is within valid range
            if not (0 <= label < n_choices):
                logger.debug(
                    f"Label {label} out of range for {n_choices} choices, skipping item"
                )
                skipped += 1
                continue

        except (KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error processing item: {e}, skipping")
            skipped += 1
            continue

        # ---- Expected answer as a letter (A,B,C,...) ----
        expected_letter = chr(ord("A") + label)

        # ---- Format choices as "A. ..., B. ..., ..." ----
        letters = [chr(ord("A") + i) for i in range(n_choices)]
        formatted_choices = "  ".join(
            f"{letters[i]}. {str(endings[i]).strip()}" for i in range(n_choices)
        )

        # ---- Deterministic, short prompt (matches your style/parsers) ----
        full_prompt = (
            "Choose the single best continuation for the context.\n"
            "Only respond with the letter prefixed with 'Answer:' and nothing else.\n\n"
            f"Context: {ctx}\n"
            f"Choices: {formatted_choices}\n"
            "Answer:"
        )

        # ---- Query model ----
        model_output, stats = query_model(
            full_prompt, model_key=model_id, current=total
        )
        tokens_per_second_total += stats.get("tokens_per_second", 0.0)

        # ---- Parse model letter and score ----
        predicted = extract_letter(model_output)  # expects a single letter like A/B/C/D
        is_correct = predicted == expected_letter
        logger.debug(
            "✅ Question %s - Expected: %s, Predicted: %s - %s",
            total + 1,
            expected_letter,
            predicted,
            "Correct" if is_correct else "Incorrect",
        )

        if is_correct:
            correct += 1
        total += 1
        time.sleep(0.1)

    return {
        "dataset": dataset_name,
        "correct": correct,
        "total": total,
        "skipped": skipped,
        "accuracy": round((correct / total * 100), 2) if total > 0 else 0.0,
        "tok_per_sec": (tokens_per_second_total / total) if total > 0 else 0.0,
    }
