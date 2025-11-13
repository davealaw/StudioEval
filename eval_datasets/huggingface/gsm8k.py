import logging
import time

from tqdm import tqdm

from models.model_handling import query_model
from utils.data_loading import load_dataset_with_config
from utils.text_parsing import extract_answer_from_solution, extract_numeric_answer

logger = logging.getLogger(__name__)


def evaluate_gsm8k_dataset(
    model_id,
    dataset_path="tinyBenchmarks/tinyGSM8k",
    dataset_name="tinyGSM8k",
    subset=None,
    split="test",
    seed=42,
    sample_size=0,
):
    """
    Evaluate the GSM8K (Grade School Math 8K) dataset with a language model.

    Args:
        model_id (str): Identifier for the model to query.
        dataset_path (str): Path to the GSM8K dataset.
        dataset_name (str): Name of the dataset for logging.
        subset (str): Optional subset of the dataset to evaluate.
        split (str): Dataset split to evaluate (default is "test").
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
        question = item.get("question", "").strip()
        expected_solution = item.get("answer", "").strip()
        expected_answer = extract_answer_from_solution(expected_solution)

        if not question or expected_answer is None:
            skipped += 1
            continue

        # Prompt for model
        full_prompt = (
            "Solve the following math problem. Show your work, and end your "
            "answer with only the final numeric value, using this format:\n"
            "Answer: numeric result\n"
            f"Problem: {question}\n"
        )

        # Query model
        model_output, stats = query_model(
            full_prompt, model_key=model_id, current=total
        )
        model_output = model_output.strip()
        tokens_per_second_total += stats["tokens_per_second"]

        # Gather results
        predicted_string = extract_numeric_answer(model_output)

        # Quick manual cleanup (keeping example for reference)
        # predicted_string = (
        #     predicted_string.replace("Answer:", "")
        #     .replace("%", "")
        #     .strip()
        # )

        try:
            if predicted_string is None:
                logger.warning(
                    "⚠️ Could not extract numeric answer for question: '%s'",
                    question,
                )
                is_correct = False
            else:
                predicted = float(predicted_string)
                is_correct = abs(predicted - expected_answer) < 1e-3

            if not is_correct:
                logger.debug(
                    "❌ Incorrect answer for question '%s': expected %s, got %s",
                    question,
                    expected_answer,
                    predicted_string,
                )

        except ValueError:
            is_correct = False
            # May just be thinking model not finished
            logger.debug(
                "⚠️ Invalid number format: '%s' for question: '%s'",
                predicted_string,
                question,
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
        "tok_per_sec": tokens_per_second_total / total if total > 0 else 0,
    }
