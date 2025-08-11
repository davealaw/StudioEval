import random
from string import ascii_uppercase
from models.model_handling import query_model
from utils.data_loading import load_dataset_with_config
from utils.text_parsing import extract_letter, extract_mcq_letter
import time
import tqdm
import logging

logger = logging.getLogger(__name__)

def extract_mc1_fields(item):
    """
    Extracts multiple-choice question fields from an item.

    Args:
        item (dict): The item from the dataset.

    Returns:
        tuple: (options, flags, correct_idx)
        - options (list[str]): List of option texts.
        - flags (list[bool]|None): List of flags indicating correct options, or None if not available.
        - correct_idx (int|None): Index of the correct option, or None if not available.

    Supports both shapes:
      A) { "mc1_targets": {"choices":[...], "labels":[0/1,...]} }
      B) { "choices":[...], "labels":[0/1,...] }
      C) { "mc1_targets":[...]}  # first is correct fallback
    """
    options, flags, correct_idx = [], None, item.get("mc1_idx")

    mc1 = item.get("mc1_targets")
    if isinstance(mc1, dict):
        options = list(mc1.get("choices") or [])
        raw = mc1.get("labels")
        if isinstance(raw, list) and len(raw) == len(options):
            flags = [bool(x) for x in raw]
    elif isinstance(mc1, list):
        options = mc1  # no flags; assume first correct unless mc1_idx present

    if not options:
        options = list(item.get("choices") or [])
    if flags is None:
        raw = item.get("labels") or item.get("mc1_correct")
        if isinstance(raw, list) and len(raw) == len(options):
            flags = [bool(x) for x in raw]

    return options, flags, correct_idx

def prepare_mc1_item(item, seed=42, qkey=None, shuffle=True):
    """
    Prepare a multiple-choice question item for evaluation.
    Args:
        item (dict): The item from the dataset.
        seed (int): Random seed for reproducibility.
        qkey (str|int): Optional key for the question, used for shuffling.
        shuffle (bool): Whether to shuffle the options.
    Returns:
        tuple: (question, choices_block, letters, gold_letter, labeled)
    """
    question = (item.get("question") or "").strip()
    if question == "":
        logger.debug("MC1: skipped item with empty question")
        return None 

    options, flags, correct_idx = extract_mc1_fields(item)
    if len(options) < 2:
        logger.debug("MC1: skipped item with <2 options")
        return None

    if isinstance(flags, list) and len(flags) == len(options):
        corr = flags
    elif isinstance(correct_idx, int) and 0 <= correct_idx < len(options):
        corr = [i == correct_idx for i in range(len(options))]
    else:
        corr = [i == 0 for i in range(len(options))]  # fallback: first is correct

    if sum(bool(x) for x in corr) != 1:
        logger.warning("MC1: expected exactly one correct option; found %s", sum(bool(x) for x in corr))

    pairs = [{"text": str(opt).strip(), "is_correct": bool(corr[i])} for i, opt in enumerate(options)]
    if shuffle:
        rng = random.Random(f"{seed}-{qkey or item.get('id') or question[:50]}")
        rng.shuffle(pairs)

    letters = ascii_uppercase[:len(pairs)]
    labeled = [{"letter": letters[i], **pairs[i]} for i in range(len(pairs))]
    expected_letter = next((p["letter"] for p in labeled if p["is_correct"]), letters[0])

    choices_block = "\n".join(f"{x['letter']}. {x['text']}" for x in labeled)

    if any(x["text"] in {"choices", "labels"} for x in labeled):
        logger.error("MC1: schema error — iterated dict keys. Check extract_mc1_fields(). Item=%r", item)

    return question, choices_block, letters, expected_letter

def evaluate_tiny_truthfulqa(model_id, dataset_path="tinyBenchmarks/tinyTruthfulQA", dataset_name="tiny_truthfulqa", subset=None, split="validation", seed=42, sample_size=0):
    """
    Evaluate the Tiny TruthfulQA dataset with a language model. 
    
    Args:
        model_id (str): Identifier for the model to query.
        dataset_path (str): Path to the Tiny TruthfulQA dataset.
        dataset_name (str): Name of the dataset for logging.
        subset (str): Optional subset of the dataset to evaluate.
        split (str): Dataset split to evaluate (default is "validation").
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

    for idx, item in enumerate(tqdm.tqdm(dataset, desc=f"⏳ Evaluating {dataset_name}")):
        pkg = prepare_mc1_item(item, seed=seed, qkey=idx, shuffle=True)
        if not pkg:
            skipped += 1
            continue

        question, choices_block, letters, expected_letter = pkg
        # full_prompt = build_mc1_prompt(question, choices_block, letters)
       
        valid =  ", ".join(list(letters))
        full_prompt = (
            "Answer the multiple-choice question.\n"
            f"Respond with the letter ({valid}) prefixed with 'Answer:'. Do not explain your answer.\n\n"
            f"Question: {question}\nChoices:\n{choices_block}\n\nAnswer:"
        )

        logger.debug(f"\nFull prompt: {full_prompt}")  # Debugging line

        # Query model
        model_output, stats = query_model(full_prompt, model_key=model_id, current=total)
        model_output = model_output.strip()
        tokens_per_second_total += stats["tokens_per_second"]

        # Gather results
        pred_letter = extract_mcq_letter(model_output, choices=letters)
        is_correct = (pred_letter == expected_letter)
        logger.debug(f"✅ Question {total + 1} - Expected: {expected_letter}, Predicted: {pred_letter} - {'Correct' if is_correct else 'Incorrect'}")

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
