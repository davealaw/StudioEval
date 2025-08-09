import time
import logging
from tqdm import tqdm
from models.model_handling import query_model
from utils.data_loading import load_json_dataset_with_config
from utils.text_parsing import extract_letter

logger = logging.getLogger(__name__)

def evaluate_cumstom_mcq(model_id, jsonl_path="coding_mcq_dataset.jsonl", dataset_name="coding_mcq", seed=42, sample_size=0, **kwargs):
    """
    Evaluate a custom multiple-choice question dataset using a language model.
    
    Args:
        model_id (str): Identifier for the model to query.
        jsonl_path (str): Path to the JSONL dataset file.
        dataset_name (str): Name of the dataset for logging.
        seed (int): Random seed for reproducibility.
        sample_size (int): Number of samples to evaluate, 0 means all.
        **kwargs: Additional keyword arguments.     
    
    Returns:
        dict: Evaluation results including accuracy and tokens per second.
    """
    dataset = load_json_dataset_with_config(jsonl_path, seed=seed, sample_size=sample_size)        

    if dataset is None:
        logger.error("Aborting evaluation due to dataset loading failure.")
        return

    correct = 0
    total = 0
    skipped = 0
    tokens_per_second_total = 0

    for item in tqdm(dataset, desc=f"⏳ Evaluating {dataset_name}"):
        # Extract Question details
        question = item.get("question", "").strip()
        choices = item.get("choices", {})
        expected = item.get("answer", "").strip().upper()

        if not question or not expected or expected not in choices or len(choices) != 4:
            skipped += 1
            continue

        formatted_choices = "\n".join([f"{k}. {v}" for k, v in choices.items()])

        full_prompt = (
            f"Answer the following multiple-choice question.\n"
            f"Only respond with the letter (A, B, C, or D) prefixed with 'Answer:'. Do not explain your answer.\n\n"
            f"Question:\n{question}\n\nChoices:\n{formatted_choices}\n\nAnswer:"
        )

        # Query model
        model_output, stats = query_model(full_prompt, model_key=model_id, current = total)
        tokens_per_second_total += stats["tokens_per_second"]

        # Gather results
        predicted = extract_letter(model_output)
        is_correct = predicted == expected
        logger.debug(f"✅ Question {total + 1} - Expected: {expected}, Predicted: {predicted} - {'Correct' if is_correct else 'Incorrect'}")

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
        "tok_per_sec": tokens_per_second_total / total if total > 0 else 0
    }
