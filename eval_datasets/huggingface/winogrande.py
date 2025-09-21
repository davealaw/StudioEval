import time
import logging
from tqdm import tqdm
from models.model_handling import query_model
from utils.data_loading import load_dataset_with_config
from utils.text_parsing import extract_letter, extract_mcq_letter

logger = logging.getLogger(__name__)

def evaluate_winogrande(model_id, dataset_path="tinyBenchmarks/tinyWinogrande", dataset_name="tinyWinogrande", subset=None, split="validation", seed=42, sample_size=0):
    """
    Evaluate Winogrande dataset with a language model.
    
    Winogrande tests commonsense reasoning through pronoun resolution in sentences.
    Models must choose which of two options correctly fills a blank (indicated by '_')
    to resolve ambiguous pronoun references. This requires understanding context,
    world knowledge, and logical reasoning.

    Dataset structure expected:
      - sentence: str   (contains '_' as the blank to be filled)
      - option1: str    (first choice for filling the blank)
      - option2: str    (second choice for filling the blank)  
      - answer: str     ("1" or "2" indicating correct option)

    Model must respond strictly with: 'Answer: A' or 'Answer: B'.

    Args:
        model_id (str): Identifier for the model to query.
        dataset_path (str): Path to the Winogrande dataset (default: tinyWinogrande).
        dataset_name (str): Name of the dataset for logging.
        subset (str): Subset of the dataset to evaluate (default: None for tinyBenchmarks).
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
        # Validate required fields
        try:
            sentence = str(item.get("sentence", "")).strip()
            option1 = str(item.get("option1", "")).strip()
            option2 = str(item.get("option2", "")).strip()
            raw_answer = str(item.get("answer", "")).strip()
            if not sentence or not option1 or not option2 or not raw_answer:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        # Convert expected answer to A/B (Winogrande uses "1"/"2" format)
        if raw_answer in {"1", "A", "a"}:
            correct_answer = "A"
        elif raw_answer in {"2", "B", "b"}:
            correct_answer = "B"
        else:
            # Unknown label; skip safely with debug info
            logger.debug(f"Unknown answer format: {raw_answer}, skipping item")
            skipped += 1
            continue

        # Build a tight, deterministic prompt
        full_prompt = (
            "You are given a sentence with a blank indicated by an underscore \"_\".\n"
            "Choose the option (A or B) that best fills the blank so the sentence makes sense.\n"
            "Only respond with the letter (A or B) prefixed with 'Answer:' and nothing else.\n\n"
            f"Sentence: {sentence}\n"
            f"Choices: A. {option1}  B. {option2}\n"
            "Answer:"
        )

        # Query model
        model_output, stats = query_model(full_prompt, model_key=model_id, current=total)
        tokens_per_second_total += stats.get("tokens_per_second", 0.0)

        # Use binary choice extraction for better precision (A/B only)
        predicted = extract_mcq_letter(model_output, choices="AB", marker="Answer:")
        is_correct = (predicted == correct_answer)
        logger.debug(f"✅ Question {total + 1} - Expected: {correct_answer}, Predicted: {predicted} - {'Correct' if is_correct else 'Incorrect'}")

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
