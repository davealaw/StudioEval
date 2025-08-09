from models.model_handling import query_model
from utils.data_loading import load_dataset_with_config
from utils.text_parsing import extract_letter
import time
import tqdm
import logging

logger = logging.getLogger(__name__)

def evaluate_tiny_truthfulqa(model_id, dataset_path="tinyBenchmarks/tinyTruthfulQA", dataset_name="tiny_truthfulqa", subset=None, split="test", seed=42, sample_size=0):
    """
    Evaluate the Tiny TruthfulQA dataset with a language model. 
    
    Args:
        model_id (str): Identifier for the model to query.
        dataset_path (str): Path to the Tiny TruthfulQA dataset.
        dataset_name (str): Name of the dataset for logging.
        subset (str): Optional subset of the dataset to evaluate.
        split (str): Dataset split to evaluate (default is "test").
        seed (int): Random seed for reproducibility.        
        sample_size (int): Number of samples to evaluate, 0 means all.      
    
    Returns:
        dict: Evaluation results including accuracy and tokens per second.
    """
    dataset = load_dataset_with_config(dataset_path, subset=subset, split=split, seed=seed, sample_size=sample_size)

    correct = 0
    total = 0
    skipped = 0
    tokens_per_second_total = 0

    for item in tqdm(dataset, desc=f"⏳ Evaluating {dataset_name}"):
        # Extract Question details
        question = item.get("question", "").strip()
        options = item.get("options", [])
        expected = item.get("answer", "").strip().upper()

        option_labels = ["A", "B", "C", "D"]
        if not question or not options or expected not in option_labels:
            skipped += 1
            continue

        # Format A–D options
        formatted_choices = "\n".join([f"{label}. {opt.strip()}" for label, opt in zip(option_labels, options)])

        full_prompt = (
            f"Answer the following multiple-choice question. "
            f"Only respond with the letter (A, B, C, or D) prefixed with 'Answer:'. Do not explain your answer.\n\n"
            f"Question: {question}\n"
            f"Choices:\n{formatted_choices}\n\n"
            f"Answer:"
        )

        # Query model
        model_output, stats = query_model(full_prompt, model_key=model_id, current = total)
        model_output = model_output.strip()
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
