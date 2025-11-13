"""
Mock dataset fixtures for testing evaluation flows without real dataset files.
"""

import json
import os
import tempfile
from typing import Any, ClassVar

import pytest


class MockDatasetFiles:
    """Creates temporary mock dataset files for testing."""

    def __init__(self):
        self.temp_dir = None
        self.dataset_files = {}

    def setup(self):
        """Create temporary directory and mock dataset files."""
        self.temp_dir = tempfile.mkdtemp()

        # Mock grammar dataset
        grammar_data = [
            {
                "question": "Which sentence is grammatically correct?",
                "options": [
                    "A) He don't like it",
                    "B) He doesn't like it",
                    "C) He not like it",
                    "D) He no like it",
                ],
                "answer": "B",
            },
            {
                "question": "Choose the correct form:",
                "options": [
                    "A) I has done it",
                    "B) I have done it",
                    "C) I had did it",
                    "D) I done it",
                ],
                "answer": "B",
            },
        ]

        # Mock logic dataset
        logic_data = [
            {
                "question": (
                    "If all cats are animals, and Fluffy is a cat, what can we "
                    "conclude?"
                ),
                "options": [
                    "A) Fluffy is an animal",
                    "B) Fluffy is not an animal",
                    "C) Cannot determine",
                    "D) Fluffy is a dog",
                ],
                "answer": "A",
            },
            {
                "question": "What comes next in the sequence: 2, 4, 8, 16, ?",
                "options": ["A) 20", "B) 24", "C) 32", "D) 64"],
                "answer": "C",
            },
        ]

        # Mock math dataset
        math_data = [
            {
                "question": "What is 15 + 27?",
                "options": ["A) 32", "B) 42", "C) 52", "D) 62"],
                "answer": "B",
            },
            {
                "question": "Solve: 3x + 5 = 14",
                "options": ["A) x = 2", "B) x = 3", "C) x = 4", "D) x = 5"],
                "answer": "B",
            },
        ]

        # Create dataset files
        datasets = {"grammar": grammar_data, "logic": logic_data, "math": math_data}

        for dataset_name, data in datasets.items():
            file_path = os.path.join(self.temp_dir, f"{dataset_name}.json")
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            self.dataset_files[dataset_name] = file_path

        return self.temp_dir

    def get_dataset_path(self, dataset_name):
        """Get path to mock dataset file."""
        return self.dataset_files.get(dataset_name)

    def teardown(self):
        """Clean up temporary files."""
        if self.temp_dir:
            import shutil

            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
            self.dataset_files.clear()


@pytest.fixture
def mock_dataset_files():
    """Pytest fixture providing mock dataset files."""
    mock_files = MockDatasetFiles()
    mock_files.setup()

    yield mock_files

    mock_files.teardown()


@pytest.fixture
def mock_datasets_config(mock_dataset_files):
    """Pytest fixture providing datasets config using mock files."""
    return {
        "grammar": {
            "file": mock_dataset_files.get_dataset_path("grammar"),
            "type": "json",
        },
        "logic": {"file": mock_dataset_files.get_dataset_path("logic"), "type": "json"},
        "math": {"file": mock_dataset_files.get_dataset_path("math"), "type": "json"},
    }


class MockDatasetLoader:
    """Mock dataset loader that returns predefined data without file I/O."""

    MOCK_DATA: ClassVar[dict[str, list[dict[str, Any]]]] = {
        "grammar": [
            {
                "question": "Which sentence is grammatically correct?",
                "options": [
                    "A) He don't like it",
                    "B) He doesn't like it",
                    "C) He not like it",
                    "D) He no like it",
                ],
                "answer": "B",
            }
        ],
        "logic": [
            {
                "question": (
                    "If all cats are animals, and Fluffy is a cat, what can we "
                    "conclude?"
                ),
                "options": [
                    "A) Fluffy is an animal",
                    "B) Fluffy is not an animal",
                    "C) Cannot determine",
                    "D) Fluffy is a dog",
                ],
                "answer": "A",
            }
        ],
        "math": [
            {
                "question": "What is 15 + 27?",
                "options": ["A) 32", "B) 42", "C) 52", "D) 62"],
                "answer": "B",
            }
        ],
    }

    @classmethod
    def load_dataset(cls, dataset_name, file_path=None):
        """Mock dataset loading that returns predefined data."""
        if dataset_name in cls.MOCK_DATA:
            return cls.MOCK_DATA[dataset_name]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
