"""
Main evaluation orchestrator containing the core business logic.
This replaces the monolithic main() function with testable components.
"""

import csv
import json
import logging
import os
import time
from typing import Any, Optional, Union

from core.dataset_registry import DatasetRegistry
from core.model_manager import ModelManager
from eval_datasets.default_evals import default_datasets_to_run
from models.model_loading import load_model_set_file
from utils.params import merge_eval_kwargs
from utils.timing_utils import format_duration

logger = logging.getLogger(__name__)


class EvaluationOrchestrator:
    """Orchestrates the full evaluation pipeline."""

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        dataset_registry: Optional[DatasetRegistry] = None,
    ):
        """
        Initialize orchestrator with dependencies.

        Args:
            model_manager: ModelManager instance (defaults to new instance)
            dataset_registry: DatasetRegistry instance (defaults to new instance)
        """
        self.model_manager = model_manager or ModelManager()
        self.dataset_registry = dataset_registry or DatasetRegistry()

    def run_evaluation(
        self,
        model: Optional[str] = None,
        models: Optional[list[str]] = None,
        model_filter: Optional[str] = None,
        model_set: Optional[str] = None,
        model_arch: Optional[str] = None,
        all_models: bool = False,
        datasets_config: Optional[Union[str, list[dict[str, Any]]]] = None,
        sample_size: int = 0,
        seed: int = 42,
        raw_duration: bool = False,
        cli_explicit_args: Optional[set[str]] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Run complete evaluation pipeline.

        Args:
            model: Specific model to evaluate
            models: List of specific models to evaluate
            model_filter: Glob pattern for model filtering
            model_set: Path to file containing model IDs
            model_arch: Architecture pattern filter
            all_models: Evaluate all available models
            datasets_config: Path to datasets configuration file
            sample_size: Number of samples per dataset (0 for all)
            seed: Random seed for reproducibility
            raw_duration: Output raw seconds vs human-readable duration
            cli_explicit_args: CLI args explicitly set by user
            **kwargs: Additional parameters

        Returns:
            True if evaluation completed successfully
        """
        logger.info("-" * 41)
        logger.info("Starting evaluation...\n")

        # Check server availability
        if not self.model_manager.is_server_running():
            logger.error("Model server is not running")
            return False

        # Resolve models to evaluate
        resolved_models: list[str]
        try:
            # If models list is provided directly, use it
            if models:
                resolved_models = models
            else:
                # Handle model set file loading
                model_set_list: Optional[list[str]] = None
                if model_set:
                    model_set_list = load_model_set_file(model_set)
                    if not model_set_list:
                        logger.error("Model set file produced no model IDs")
                        return False

                resolved_models = self.model_manager.resolve_models(
                    model=model,
                    model_filter=model_filter,
                    model_set=model_set_list,
                    model_arch=model_arch,
                    all_models=all_models,
                )
        except ValueError as e:
            logger.error(str(e))
            return False

        # Load and validate dataset configuration once (before model loop)
        try:
            datasets_to_run: list[dict[str, Any]] = self._load_datasets_config(
                datasets_config
            )
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            PermissionError,
            ValueError,
        ) as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            logger.error(
                "🚫 Cannot proceed with evaluation of "
                f"{len(resolved_models)} model(s) due to invalid configuration"
            )
            return False

        logger.info("✅ Configuration validated successfully")
        logger.info(
            "📊 Will evaluate "
            f"{len(resolved_models)} model(s) on {len(datasets_to_run)} dataset(s)\n"
        )

        # Evaluate each model
        for model_id in resolved_models:
            self._evaluate_single_model(
                model_id=model_id,
                datasets_to_run=datasets_to_run,  # Pass pre-loaded config
                sample_size=sample_size,
                seed=seed,
                raw_duration=raw_duration,
                cli_explicit_args=cli_explicit_args,
                **kwargs,
            )

        return True

    def _evaluate_single_model(
        self,
        model_id: str,
        datasets_to_run: list[dict[str, Any]],
        sample_size: int = 0,
        seed: int = 42,
        raw_duration: bool = False,
        cli_explicit_args: Optional[set[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Evaluate a single model on configured datasets.

        Args:
            model_id: Model identifier
            datasets_to_run: Pre-loaded list of dataset configurations
            sample_size: Number of samples per dataset
            seed: Random seed
            raw_duration: Output format for duration
            cli_explicit_args: CLI args explicitly set by user
            **kwargs: Additional parameters
        """
        logger.info(f"--- Evaluating {model_id} ---")

        # Load model
        self.model_manager.load_model(model_id)

        start_time = time.time()
        results: list[dict[str, Any]] = []
        total_tokens_per_sec: float = 0.0

        # Evaluate each dataset (configuration already validated at orchestrator level)
        for dataset_config in datasets_to_run:
            result = self._evaluate_single_dataset(
                model_id=model_id,
                dataset_config=dataset_config,
                sample_size=sample_size,
                seed=seed,
                cli_explicit_args=cli_explicit_args,
                **kwargs,
            )

            if result:
                results.append(result)
                total_tokens_per_sec += result["tok_per_sec"]

                logger.info(
                    "✅ "
                    f"{result['dataset']} accuracy: {result['accuracy']}% "
                    f"({result['correct']}/{result['total']})"
                )
                logger.info(
                    "📊 "
                    f"{result['dataset']} average response tokens/sec: "
                    f"{result['tok_per_sec']:.2f}\n"
                )

        # Calculate metrics for summaries
        elapsed_time = time.time() - start_time
        avg_tokens_per_sec = total_tokens_per_sec / len(results) if results else 0.0

        # Save results and log summary
        self._save_results(results)
        self._save_model_accuracy_summary(
            model_id=model_id,
            results=results,
            elapsed_time=elapsed_time,
            avg_tokens_per_sec=avg_tokens_per_sec,
            raw_duration=raw_duration,
        )
        self._log_evaluation_summary(
            model_id=model_id,
            results=results,
            start_time=start_time,
            total_tokens_per_sec=total_tokens_per_sec,
            raw_duration=raw_duration,
        )

        # Always unload model
        self.model_manager.unload_model(model_id)
        logger.info("-" * 41 + "\n")
        time.sleep(2)

    def _load_datasets_config(
        self,
        datasets_config: Optional[Union[str, list[dict[str, Any]]]] = None,
    ) -> list[dict[str, Any]]:
        """Load datasets configuration from file, list, or use defaults."""
        if datasets_config:
            # If it's already a list, return it directly
            if isinstance(datasets_config, list):
                return self._validate_config_structure(
                    datasets_config, "<provided list>"
                )
            # Otherwise treat as file path
            try:
                with open(datasets_config) as f:
                    content = f.read().strip()
                    if not content:
                        logger.error(
                            f"❌ Configuration file is empty: {datasets_config}"
                        )
                        logger.error(
                            "💡 Hint: Add a valid JSON array of dataset configurations"
                        )
                        raise ValueError("Empty configuration file")

                    data = json.loads(content)
                    return self._validate_config_structure(data, datasets_config)

            except json.JSONDecodeError as e:
                self._handle_json_parse_error(e, datasets_config)
                raise
            except FileNotFoundError:
                logger.error(f"❌ Configuration file not found: {datasets_config}")
                logger.error(
                    "💡 Hint: Check the file path or create the configuration file"
                )
                logger.error(
                    "   Example working files: "
                    "examples/poc_arc.json, examples/run_base_benchmarks.json"
                )
                raise
            except PermissionError:
                logger.error(f"❌ Permission denied reading file: {datasets_config}")
                logger.error(
                    f"💡 Hint: Check file permissions with: ls -la {datasets_config}"
                )
                raise
        else:
            return default_datasets_to_run

    def _validate_config_structure(
        self, data: Any, source: str
    ) -> list[dict[str, Any]]:
        """Validate and fix configuration structure with helpful error messages."""

        # Check if data is a dictionary (common mistake - should be array)
        if isinstance(data, dict):
            logger.warning(
                f"⚠️  Configuration in {source} is a single object, "
                "but should be an array"
            )
            logger.info("🔧 Auto-fixing: Wrapping single configuration in array")
            logger.info("💡 Recommendation: Update your JSON file to use array format:")
            logger.info('   Current:  { "eval_type": "...", ... }')
            logger.info('   Correct: [{ "eval_type": "...", ... }]')
            data = [data]  # Auto-fix by wrapping in array

        # Check if data is a list
        if not isinstance(data, list):
            logger.error(
                "❌ Configuration must be a JSON array, "
                f"got {type(data).__name__}: {source}"
            )
            logger.error('💡 Expected format: [ { "eval_type": "arc", ... }, { ... } ]')
            raise ValueError(
                f"Configuration must be an array, got {type(data).__name__}"
            )

        # Check if list is empty
        if not data:
            logger.error(f"❌ Configuration array is empty: {source}")
            logger.error("💡 Add at least one dataset configuration")
            raise ValueError("Empty configuration array")

        # Validate each dataset configuration
        for i, config in enumerate(data):
            self._validate_dataset_config(config, i, source)

        return data

    def _validate_dataset_config(self, config: Any, index: int, source: str) -> None:
        """Validate individual dataset configuration."""
        if not isinstance(config, dict):
            logger.error(
                "❌ Dataset configuration "
                f"#{index + 1} must be an object, "
                f"got {type(config).__name__}: {source}"
            )
            logger.error(
                '💡 Expected: { "eval_type": "arc", "dataset_path": "...", ... }'
            )
            raise ValueError(f"Dataset config #{index + 1} must be a dictionary")

        # Check required fields
        required_fields = ["eval_type", "dataset_path", "dataset_name"]
        missing_fields = [field for field in required_fields if field not in config]

        if missing_fields:
            logger.error(
                "❌ Dataset configuration "
                f"#{index + 1} missing required fields: {missing_fields}"
            )
            logger.error(f"   Available fields: {list(config.keys())}")
            logger.error(f"💡 Required fields: {required_fields}")
            logger.error("   Example: {")
            logger.error('     "eval_type": "arc",')
            logger.error('     "dataset_path": "allenai/ai2_arc",')
            logger.error('     "dataset_name": "ai2_arc_easy",')
            logger.error('     "subset": "ARC-Easy",')
            logger.error('     "split": "validation",')
            logger.error('     "sample_size": 10')
            logger.error("   }")
            raise ValueError(
                "Missing required fields in dataset config "
                f"#{index + 1}: {missing_fields}"
            )

        # Check eval_type is supported
        eval_type = config["eval_type"]
        if not self.dataset_registry.is_supported(eval_type):
            available_types = self.dataset_registry.list_supported_types()
            logger.error(
                "❌ Unsupported eval_type "
                f"'{eval_type}' in dataset configuration #{index + 1}"
            )
            logger.error(f"💡 Supported eval_types: {available_types}")
            raise ValueError(f"Unsupported eval_type: {eval_type}")

    def _handle_json_parse_error(self, e: json.JSONDecodeError, file_path: str) -> None:
        """Provide detailed guidance for JSON parsing errors."""
        logger.error(f"❌ Failed to parse JSON configuration file: {file_path}")
        logger.error(
            f"   Error: {e.msg} at line {e.lineno}, column {e.colno} "
            f"(character {e.pos})"
        )

        # Try to read the file and show the problematic line
        try:
            with open(file_path) as f:
                lines = f.readlines()
                if e.lineno <= len(lines):
                    problem_line = lines[e.lineno - 1].rstrip()
                    logger.error(f"   Problem line {e.lineno}: {problem_line}")

                    # Show a pointer to the error position
                    if e.colno > 0:
                        pointer = " " * (e.colno - 1) + "^"
                        logger.error(f"   {' ' * (len(str(e.lineno)) + 16)}{pointer}")
        except OSError:
            pass  # If we can't read the file for context, just skip this part

        # Provide specific guidance based on common JSON errors
        error_msg_lower = e.msg.lower()
        if "expecting ',' delimiter" in error_msg_lower:
            logger.error("💡 Hint: Missing comma between JSON elements")
            logger.error("   Check if you need a comma after the previous line")
        elif "expecting ':' delimiter" in error_msg_lower:
            logger.error("💡 Hint: Missing colon ':' after a key")
            logger.error('   Keys must be followed by a colon, e.g., "key": "value"')
        elif "expecting value" in error_msg_lower:
            logger.error("💡 Hint: Missing or invalid value")
            logger.error("   Check for trailing commas or incomplete values")
        elif "expecting property name" in error_msg_lower:
            logger.error("💡 Hint: Missing or invalid property name")
            logger.error("   Property names must be quoted strings")
        elif "unterminated string" in error_msg_lower:
            logger.error("💡 Hint: Unclosed string - missing closing quote")
        elif "invalid escape sequence" in error_msg_lower:
            logger.error("💡 Hint: Invalid escape sequence in string")
            logger.error("   Use \\\\ for backslashes or \\/ for forward slashes")
        else:
            logger.error(
                "💡 Hint: Check for missing commas, quotes, or mismatched "
                "brackets/braces"
            )

        logger.error(
            "📖 For reference, check working examples in: examples/poc_arc.json"
        )

    def _evaluate_single_dataset(
        self,
        model_id: str,
        dataset_config: dict[str, Any],
        sample_size: int = 0,
        seed: int = 42,
        cli_explicit_args: Optional[set[str]] = None,
        **cli_kwargs: Any,
    ) -> Optional[dict[str, Any]]:
        """
        Evaluate a single dataset.

        Args:
            model_id: Model identifier
            dataset_config: Dataset configuration dictionary
            sample_size: Number of samples to evaluate
            seed: Random seed
            cli_explicit_args: CLI args explicitly set by user
            **cli_kwargs: Additional CLI arguments

        Returns:
            Evaluation result dictionary or None if failed
        """
        eval_type = dataset_config["eval_type"]
        dataset_path = dataset_config["dataset_path"]
        dataset_name = dataset_config["dataset_name"]

        # Mock CLI args for parameter merging
        class MockArgs:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        cli_args = MockArgs(seed=seed, sample_size=sample_size)
        kwargs = merge_eval_kwargs(
            dataset_config,
            cli_args,
            ["seed", "sample_size", "split", "subset"],
            cli_explicit_args,
        )

        logger.debug(f"▶️ Evaluating {dataset_name} [{eval_type}]...")

        try:
            return self.dataset_registry.evaluate_dataset(
                eval_type=eval_type,
                model_id=model_id,
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                **kwargs,
            )
        except (RuntimeError, ValueError, OSError, KeyError, TypeError) as e:
            logger.error(f"Failed to evaluate {dataset_name}: {e}")
            return None

    def _save_results(self, results: list[dict[str, Any]]) -> None:
        """Save evaluation results to CSV."""
        if not results:
            return

        with open("evaluation_summary.csv", "w", newline="") as csvfile:
            fieldnames = [
                "dataset",
                "correct",
                "total",
                "skipped",
                "accuracy",
                "tok_per_sec",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        logger.info("📄 Results saved to evaluation_summary.csv")

    def _log_evaluation_summary(
        self,
        model_id: str,
        results: list[dict[str, Any]],
        start_time: float,
        total_tokens_per_sec: float,
        raw_duration: bool = False,
    ) -> None:
        """Log evaluation summary statistics."""
        if not results:
            logger.warning("No results to summarize")
            return

        # Calculate timing
        elapsed = time.time() - start_time
        if raw_duration:
            logger.info(f"📊 Took {elapsed:.2f} seconds to evaluate {model_id}")
        else:
            logger.info(f"📊 Took {format_duration(elapsed)} to evaluate {model_id}")

        # Calculate overall metrics
        total_correct = sum(r["correct"] for r in results)
        total_questions = sum(r["total"] for r in results)
        overall_accuracy = (
            round((total_correct / total_questions * 100), 2)
            if total_questions > 0
            else 0.0
        )
        avg_tokens_per_sec = total_tokens_per_sec / len(results) if results else 0.0

        logger.info(f"📊 Overall accuracy: {overall_accuracy}%")
        logger.info(
            f"📊 Overall average response tokens per second: {avg_tokens_per_sec:.2f}"
        )

    def _save_model_accuracy_summary(
        self,
        model_id: str,
        results: list[dict[str, Any]],
        elapsed_time: float,
        avg_tokens_per_sec: float,
        raw_duration: bool = False,
    ) -> None:
        """Save or update model accuracy summary CSV with dynamic columns."""
        if not results:
            return

        # Calculate overall accuracy
        total_correct = sum(r["correct"] for r in results)
        total_questions = sum(r["total"] for r in results)
        overall_accuracy = (
            round((total_correct / total_questions * 100), 2)
            if total_questions > 0
            else 0.0
        )

        # Format duration
        if raw_duration:
            time_str = f"{elapsed_time:.2f}s"
        else:
            time_str = format_duration(elapsed_time)

        # Load existing data or create new structure
        model_summary_file = "model_accuracy_summary.csv"
        existing_data = self._load_existing_model_summary(model_summary_file)

        # Prepare new model row
        model_row = {
            "model": model_id,
            "overall": overall_accuracy,
            "time": time_str,
            "tokens": round(avg_tokens_per_sec, 2),
        }

        # Add dataset accuracies
        for result in results:
            model_row[result["dataset"]] = result["accuracy"]

        # Update existing data or add new model
        self._update_model_summary_data(existing_data, model_row)

        # Save updated data
        self._write_model_summary_csv(model_summary_file, existing_data)

        logger.info(f"📄 Model accuracy summary updated in {model_summary_file}")

    def _load_existing_model_summary(self, file_path: str) -> dict[str, Any]:
        """Load existing model accuracy summary data."""
        if not os.path.exists(file_path):
            return {"models": [], "all_datasets": set()}

        try:
            with open(file_path, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                models = list(reader)

                # Extract all dataset columns (excluding fixed columns)
                fixed_cols = {"model", "overall", "time", "tokens"}
                all_datasets: set[str] = set()

                if models:
                    fieldnames = reader.fieldnames or []
                    all_datasets = set(fieldnames) - fixed_cols

                return {"models": models, "all_datasets": all_datasets}
        except (OSError, csv.Error) as e:
            logger.warning(f"Could not read existing model summary file: {e}")
            return {"models": [], "all_datasets": set()}

    def _update_model_summary_data(
        self, existing_data: dict[str, Any], new_model_row: dict[str, Any]
    ) -> None:
        """Update existing data with new model results."""
        # Update the set of all datasets
        model_datasets = set(new_model_row.keys()) - {
            "model",
            "overall",
            "time",
            "tokens",
        }
        existing_data["all_datasets"].update(model_datasets)

        # Check if model already exists and update or append
        model_id = new_model_row["model"]
        existing_model_idx = None

        for i, existing_model in enumerate(existing_data["models"]):
            if existing_model.get("model") == model_id:
                existing_model_idx = i
                break

        if existing_model_idx is not None:
            # Update existing model data
            existing_data["models"][existing_model_idx].update(new_model_row)
        else:
            # Add new model
            existing_data["models"].append(new_model_row)

    def _write_model_summary_csv(self, file_path: str, data: dict[str, Any]) -> None:
        """Write model summary data to CSV with dynamic columns."""
        if not data["models"]:
            return

        # Sort datasets alphabetically for consistent column order
        dataset_columns = sorted(data["all_datasets"])

        # Define column order: model, datasets (alphabetical), overall, time, tokens
        fieldnames = [
            "model",
            *dataset_columns,
            "overall",
            "time",
            "tokens",
        ]

        with open(file_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Write each model row, filling missing dataset columns with empty strings
            for model_data in data["models"]:
                row = {col: model_data.get(col, "") for col in fieldnames}
                writer.writerow(row)
