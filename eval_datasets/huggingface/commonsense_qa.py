import logging
import time

from tqdm import tqdm

from models.model_handling import query_model
from utils.data_loading import load_dataset_with_config
from utils.text_parsing import extract_letter_to_e

logger = logging.getLogger(__name__)


def evaluate_commonsense_qa(
    model_id,
    dataset_path="tau/commonsense_qa",
    dataset_name="commonsense_qa",
    subset=None,
    split="validation",
    seed=142,
    sample_size=100,
):
    """
    Evaluate the Commonsense QA dataset with a language model.

    Args:
        model_id (str): Identifier for the model to query.
        dataset_path (str): Path to the Commonsense QA dataset.
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
    tokens_per_second_total = 0

    for item in tqdm(dataset, desc=f"⏳ Evaluating {dataset_name}"):

        # Extract Question details
        correct_answer = None
        if "answerKey" in item:
            correct_answer = item["answerKey"].strip().upper()
        else:
            skipped += 1
            continue

        question = item["question"].strip()

        # Check if choices exists and is properly formatted
        if "choices" not in item:
            skipped += 1
            continue

        choices = item["choices"]

        if isinstance(choices, dict) and "label" in choices and "text" in choices:
            formatted_choices = ", ".join(
                [f"{lbl}. {txt}" for lbl, txt in zip(choices["label"], choices["text"])]
            )
        else:
            skipped += 1
            continue

        full_prompt = (
            "Answer the following multiple-choice question. "
            "Only respond with the letter (A, B, C, D, or E) prefixed with "
            "'Answer:'. Do not explain your answer.\n\n"
            f"Question: {question}\n"
            + "Choices: "
            + formatted_choices.strip()
            + "\nAnswer:"
        )

        # Query model
        model_output, stats = query_model(
            full_prompt, model_key=model_id, current=total
        )
        tokens_per_second_total += stats["tokens_per_second"]

        # Gather results
        predicted = extract_letter_to_e(model_output)
        is_correct = predicted == correct_answer
        logger.debug(
            "✅ Question %s - Expected: %s, Predicted: %s - %s",
            total + 1,
            correct_answer,
            predicted,
            "Correct" if is_correct else "Incorrect",
        )

        if is_correct:
            correct += 1
        total += 1

        time.sleep(0.1)  # Avoid overload

    return {
        "dataset": dataset_name,
        "correct": correct,
        "total": total,
        "skipped": skipped,
        "accuracy": round((correct / total * 100), 2) if total > 0 else 0.0,
        "tok_per_sec": tokens_per_second_total / total if total > 0 else 0,
    }
