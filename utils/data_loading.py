
import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)

def load_dataset_with_config(dataset_path, subset=None, split="train", seed=None, sample_size=0, revision=None):
    """
    Load a dataset from Hugging Face or local path with optional subset (config name), split, and sample size.

    Args:
        dataset_path (str): Path or HF identifier for the dataset.
        subset (str or None): Optional configuration name (e.g., "ARC-Easy").
        split (str): Which split to load ("train", "test", etc.).
        seed (int or None): Seed for shuffling (used only if sample_size > 0).
        sample_size (int): Number of samples to select; 0 means full dataset.
        revision (str or None): Optional revision to load specific version of the dataset.

    Returns:
        Dataset: A Hugging Face Dataset object.
    """
    load_args = {"split": split}
    if subset:
        load_args["name"] = subset
    if revision:
        load_args["revision"] = revision

    dataset = load_dataset(dataset_path, **load_args)

    if sample_size > 0:
        if sample_size > len(dataset):
            logging.warning(f"⚠️ sample_size={sample_size} exceeds dataset size ({len(dataset)}). Truncating to full dataset.")
        else:
            if seed is None:
                seed = 42
            dataset = dataset.shuffle(seed=seed).select(range(sample_size))

    return dataset

def load_json_dataset_with_config(jsonl_path, seed=None, sample_size=0):
    """
    Load a dataset from Hugging Face or local path with optional subset (config name), split, and sample size.

    Args:
        jsonl_path (str): Path for JSONL dataset.
        seed (int or None): Seed for shuffling (used only if sample_size > 0).
        sample_size (int): Number of samples to select; 0 means full dataset.

    Returns:
        Dataset: A Hugging Face Dataset object.
    """
    try:
        dataset = load_dataset("json", data_files=jsonl_path, split="train")
    except Exception as e:
        logger.error(f"Failed to load dataset from {jsonl_path}")
        logger.error(f"{type(e).__name__}: {e}")
        return None

    if sample_size > 0:
        if sample_size > len(dataset):
            logging.warning(f"⚠️ sample_size={sample_size} exceeds dataset size ({len(dataset)}). Truncating to full dataset.")
        else:
            if seed is None:
                seed = 42
            dataset = dataset.shuffle(seed=seed).select(range(sample_size))

    return dataset
