import time
import logging
from tqdm import tqdm
from models.model_handling import query_model
from utils.data_loading import load_json_dataset_with_config
from utils.text_parsing import extract_corrected_text, normalize, is_accepted

logger = logging.getLogger(__name__)

def evaluate_grammar_dataset(model_id, jsonl_path="grammar_dataset.jsonl", dataset_name="grammar", seed=42, sample_size=0, sleep=True, **kwargs):
    """
    Evaluate a custom grammar correction dataset using a language model.
    
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
        expected_answer = item.get("answer", "").strip()

        if not question or not expected_answer:
            skipped += 1
            continue

        # Construct simple grammar prompt
        full_prompt = (
            "Correct the following sentence for grammar, punctuation, and clarity, following Modern and inclusive American English conventions. "
            "Only return the corrected version prefixed by 'Corrected:'. Do not explain your answer.\n\n"
            f"Sentence: {question}\nCorrected:"
        )

        # Query model
        model_output, stats = query_model(full_prompt, model_key=model_id, current = total)
        model_output = model_output.strip()
        tokens_per_second_total += stats["tokens_per_second"]

        logger.debug(f"Model Output: {model_output}")

        # Gather results
        predicted = extract_corrected_text(model_output).strip()
        is_correct = is_accepted(expected_answer, predicted)
        logger.debug(f"✅ Question {total + 1} - Expected: {normalize(expected_answer)}, Predicted: {normalize(predicted)} - {'Correct' if is_correct else 'Incorrect'}")  

        if is_correct:
            correct += 1
        total += 1

        if sleep:
            time.sleep(0.1)

    return {
        "dataset": dataset_name,
        "correct": correct,
        "total": total,
        "skipped": skipped,
        "accuracy": round((correct / total * 100), 2) if total > 0 else 0.0,
        "tok_per_sec": tokens_per_second_total / total if total > 0 else 0
    }
