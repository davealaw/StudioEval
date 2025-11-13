"""
Pytest configuration and shared fixtures for StudioEval tests.
"""

import json
from unittest.mock import Mock

import pytest


@pytest.fixture
def sample_mcq_data():
    """Sample MCQ dataset entries for testing."""
    return [
        {
            "question": "What is the capital of France?",
            "choices": ["A) London", "B) Berlin", "C) Paris", "D) Madrid"],
            "answer": "C",
        },
        {
            "question": "What is 2 + 2?",
            "choices": ["A) 3", "B) 4", "C) 5", "D) 6"],
            "answer": "B",
        },
        {
            "question": "Which programming language is this project written in?",
            "choices": ["A) JavaScript", "B) Python", "C) Go", "D) Rust"],
            "answer": "B",
        },
    ]


@pytest.fixture
def sample_grammar_data():
    """Sample grammar dataset entries for testing."""
    return [
        {
            "question": "Me and him went to the store",
            "answer": "He and I went to the store",
        },
        {"question": "Their going to be late", "answer": "They're going to be late"},
        {
            "question": "The data shows that they're results are correct",
            "answer": "The data shows that their results are correct",
        },
    ]


@pytest.fixture
def sample_math_data():
    """Sample math dataset entries for testing."""
    return [
        {
            "question": (
                "If John has 5 apples and gives away 2, how many does he have " "left?"
            ),
            "answer": 3,
        },
        {"question": "What is 15% of 200?", "answer": 30},
        {
            "question": "A rectangle has length 8 and width 6. What is its area?",
            "answer": 48,
        },
    ]


@pytest.fixture
def mock_model_responses():
    """Mock responses for different model queries."""
    return {
        "test-model-a": "Answer: A",
        "test-model-b": "Answer: B",
        "test-model-c": "Answer: C",
        "grammar-model": "Corrected: The corrected sentence is here.",
        "math-model": "Answer: 42",
        "error-model": "",  # Empty response for error testing
    }


@pytest.fixture
def mock_stats():
    """Mock statistics returned by model queries."""
    return {
        "tokens_per_second": 15.5,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "stop_reason": "max_tokens",
        "structured": False,
    }


@pytest.fixture
def temp_jsonl_file(tmp_path):
    """Create a temporary JSONL file for testing."""

    def _create_jsonl(data):
        file_path = tmp_path / "test_dataset.jsonl"
        with open(file_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return str(file_path)

    return _create_jsonl


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for testing."""

    def _create_config(config_data):
        file_path = tmp_path / "test_config.json"
        with open(file_path, "w") as f:
            json.dump(config_data, f, indent=2)
        return str(file_path)

    return _create_config


@pytest.fixture
def sample_comm_config():
    """Sample communication configuration for testing."""
    return {
        "timeout": 120,
        "GENERATION_PARAMS": {
            "temperature": 0.1,
            "topPSampling": 0.9,
            "topKSampling": 5,
            "repeatPenalty": 1.05,
            "maxTokens": 1000,
        },
    }


@pytest.fixture
def sample_datasets_config():
    """Sample datasets configuration for testing."""
    return [
        {
            "eval_type": "custom_mcq",
            "dataset_path": "test_data/sample_mcq.jsonl",
            "dataset_name": "test_mcq",
            "seed": 42,
            "sample_size": 10,
        },
        {
            "eval_type": "grammar",
            "dataset_path": "test_data/sample_grammar.jsonl",
            "dataset_name": "test_grammar",
            "seed": 42,
            "sample_size": 5,
        },
    ]


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before each test."""
    import logging

    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Reset to basic config
    logging.basicConfig(level=logging.WARNING, force=True)


@pytest.fixture
def mock_lmstudio_client():
    """Mock LM Studio client for testing without actual LM Studio dependency."""
    mock_client = Mock()
    mock_client.query_model.return_value = (
        "Answer: A",
        {
            "tokens_per_second": 10.0,
            "prompt_tokens": 50,
            "completion_tokens": 10,
            "total_tokens": 60,
            "stop_reason": "eos_token",
            "structured": False,
        },
    )
    mock_client.load_model.return_value = None
    mock_client.unload_model.return_value = None
    mock_client.list_models.return_value = ["model1", "model2", "test-model"]
    mock_client.is_running.return_value = True
    return mock_client
