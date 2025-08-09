import time
import logging
from tqdm import tqdm
from utils.data_loading import load_json_dataset_with_config
from utils.text_parsing import extract_numeric_answer
from models.model_handling import query_model

logger = logging.getLogger(__name__)

def evaluate_math_dataset(model_id, jsonl_path="elementary_math_dataset.jsonl", dataset_name="basic math", seed=42, sample_size=0, **kwargs):
    """
    Evaluate a custom exact answer math dataset with the model. 

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
  
    total = 0
    correct = 0
    skipped = 0
    tokens_per_second_total = 0

    for item in tqdm(dataset, desc=f"⏳ Evaluating {dataset_name}"):
        # Extract Question details
        question = item.get("question", "").strip()
        expected_answer = item.get("answer")

        if not question or expected_answer is None:
            skipped += 1
            continue

        # Construct simple math prompt
        full_prompt = (
            "Solve the following math problem. Return only the final numeric answer prefixed with 'Answer:'. "
            "Do not explain your steps.\n\n"
            f"Problem: {question}\nAnswer:"
        )

        # Query model
        model_output, stats = query_model(full_prompt, model_key=model_id, current = total)
        model_output = model_output.strip()
        tokens_per_second_total += stats["tokens_per_second"]

        # Gather results
        try:
            predicted = extract_numeric_answer(model_output)
            
            try:
                expected = float(expected_answer)
            except (TypeError, ValueError):
                expected = None

            if predicted is None or expected is None:
                is_correct = False
            else:
                is_correct = abs(predicted - expected) < 1e-3  # Tolerance for rounding

            if not is_correct:
                logger.debug(f"❌ Incorrect answer for question '{question}': expected {expected}, got {predicted}")

        except ValueError:
            is_correct = False  # Model didn't return a parseable number
            logger.debug(f"⚠️ Invalid number format from model: '{predicted}' for question '{question}'")

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
