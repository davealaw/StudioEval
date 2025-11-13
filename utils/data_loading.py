import logging
from typing import Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)

# Default seed for reproducibility
DEFAULT_SEED = 42


def _apply_sampling(dataset, sample_size: int, seed: Optional[int] = None):
    """
    Apply sampling to a dataset if sample_size > 0.

    Args:
        dataset: The dataset to sample from
        sample_size (int): Number of samples to select; 0 means full dataset
        seed (int, optional): Seed for shuffling

    Returns:
        Dataset: Sampled dataset or original if sample_size <= 0
    """
    if sample_size <= 0:
        return dataset

    if sample_size > len(dataset):
        logger.warning(
            "⚠️ sample_size=%s exceeds dataset size (%s). Using full dataset.",
            sample_size,
            len(dataset),
        )
        return dataset

    if seed is None:
        seed = DEFAULT_SEED

    return dataset.shuffle(seed=seed).select(range(sample_size))


def load_dataset_with_config(
    dataset_path: str,
    subset: Optional[str] = None,
    split: str = "train",
    seed: Optional[int] = None,
    sample_size: int = 0,
    revision: Optional[str] = None,
):
    """
    Load a dataset from Hugging Face with optional configuration.

    Args:
        dataset_path (str): HF identifier for the dataset
        subset (str, optional): Configuration name (e.g., "ARC-Easy")
        split (str): Split to load ("train", "test", etc.)
        seed (int, optional): Seed for shuffling (used only if sample_size > 0)
        sample_size (int): Number of samples to select; 0 means full dataset
        revision (str, optional): Specific version/revision of the dataset

    Returns:
        Dataset: A Hugging Face Dataset object

    Raises:
        Exception: If dataset loading fails
    """
    load_args = {"split": split}
    if subset:
        load_args["name"] = subset
    if revision:
        load_args["revision"] = revision

    try:
        dataset = load_dataset(dataset_path, **load_args)
        return _apply_sampling(dataset, sample_size, seed)
    except Exception as e:
        logger.error(
            f"Failed to load dataset '{dataset_path}': {type(e).__name__}: {e}"
        )
        raise


def load_json_dataset_with_config(
    jsonl_path: str, seed: Optional[int] = None, sample_size: int = 0
):
    """
    Load a dataset from a local JSONL file.

    Args:
        jsonl_path (str): Path to JSONL dataset file
        seed (int, optional): Seed for shuffling (used only if sample_size > 0)
        sample_size (int): Number of samples to select; 0 means full dataset

    Returns:
        Dataset: A Hugging Face Dataset object

    Raises:
        Exception: If dataset loading fails
    """
    try:
        dataset = load_dataset("json", data_files=jsonl_path, split="train")
        return _apply_sampling(dataset, sample_size, seed)
    except Exception as e:
        logger.error(
            f"Failed to load JSON dataset from '{jsonl_path}': {type(e).__name__}: {e}"
        )
        raise
