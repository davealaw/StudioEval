"""
Main evaluation orchestrator containing the core business logic.
This replaces the monolithic main() function with testable components.
"""
import csv
import json
import time
import logging
from typing import List, Dict, Any, Optional

from interfaces.model_client import EvaluationResult
from core.model_manager import ModelManager
from core.dataset_registry import DatasetRegistry
from eval_datasets.default_evals import default_datasets_to_run
from models.model_loading import load_model_set_file
from utils.params import merge_eval_kwargs
from utils.timing_utils import format_duration

logger = logging.getLogger(__name__)


class EvaluationOrchestrator:
    """Orchestrates the full evaluation pipeline."""
    
    def __init__(self, model_manager: ModelManager = None, dataset_registry: DatasetRegistry = None):
        """
        Initialize orchestrator with dependencies.
        
        Args:
            model_manager: ModelManager instance (defaults to new instance)
            dataset_registry: DatasetRegistry instance (defaults to new instance)
        """
        self.model_manager = model_manager or ModelManager()
        self.dataset_registry = dataset_registry or DatasetRegistry()
    
    def run_evaluation(self, 
                      model: str = None,
                      models: List[str] = None,
                      model_filter: str = None,
                      model_set: str = None,
                      model_arch: str = None,
                      all_models: bool = False,
                      datasets_config: str = None,
                      skip_thinking_models: bool = False,
                      sample_size: int = 0,
                      seed: int = 42,
                      raw_duration: bool = False,
                      cli_explicit_args: set = None,
                      **kwargs) -> bool:
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
            skip_thinking_models: Skip known thinking models
            sample_size: Number of samples per dataset (0 for all)
            seed: Random seed for reproducibility
            raw_duration: Output raw seconds vs human-readable duration
            cli_explicit_args: Set of CLI argument names that were explicitly set by user
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
        try:
            # If models list is provided directly, use it
            if models:
                resolved_models = models
            else:
                # Handle model set file loading
                model_set_list = None
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
                    all_models=all_models
                )
        except ValueError as e:
            logger.error(str(e))
            return False
        
        # logger.info("-" * 41 + "\n")
        
        # Evaluate each model
        for model_id in resolved_models:
            self._evaluate_single_model(
                model_id=model_id,
                datasets_config=datasets_config,
                skip_thinking_models=skip_thinking_models,
                sample_size=sample_size,
                seed=seed,
                raw_duration=raw_duration,
                cli_explicit_args=cli_explicit_args,
                **kwargs
            )
        
        return True
    
    def _evaluate_single_model(self, 
                              model_id: str,
                              datasets_config: str = None,
                              skip_thinking_models: bool = False,
                              sample_size: int = 0,
                              seed: int = 42,
                              raw_duration: bool = False,
                              cli_explicit_args: set = None,
                              **kwargs) -> None:
        """
        Evaluate a single model on configured datasets.
        
        Args:
            model_id: Model identifier
            datasets_config: Path to datasets configuration
            skip_thinking_models: Skip known thinking models
            sample_size: Number of samples per dataset
            seed: Random seed
            raw_duration: Output format for duration
            cli_explicit_args: Set of CLI argument names that were explicitly set by user
            **kwargs: Additional parameters
        """
        logger.info(f"--- Evaluating {model_id} ---")
        
        # Check if model should be skipped
        if self.model_manager.should_skip_model(model_id, skip_thinking_models):
            logger.info(f"Skipping {model_id} as it is a thinking model.")
            return
        
        # Load model
        self.model_manager.load_model(model_id)
        
        start_time = time.time()
        results = []
        total_tokens_per_sec = 0
        
        try:
            # Load dataset configuration
            datasets_to_run = self._load_datasets_config(datasets_config)
            
            # Evaluate each dataset
            for dataset_config in datasets_to_run:
                result = self._evaluate_single_dataset(
                    model_id=model_id,
                    dataset_config=dataset_config,
                    sample_size=sample_size,
                    seed=seed,
                    cli_explicit_args=cli_explicit_args,
                    **kwargs
                )
                
                if result:
                    results.append(result)
                    total_tokens_per_sec += result["tok_per_sec"]
                    
                    logger.info(f"✅ {result['dataset']} accuracy: {result['accuracy']}% ({result['correct']}/{result['total']})")
                    logger.info(f"📊 {result['dataset']} average response tokens/sec: {result['tok_per_sec']:.2f}\n")
            
            # Save results and log summary
            self._save_results(results)
            self._log_evaluation_summary(
                model_id=model_id,
                results=results, 
                start_time=start_time,
                total_tokens_per_sec=total_tokens_per_sec,
                raw_duration=raw_duration
            )
            
        except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
            logger.error(f"Configuration error for model {model_id}: {e}")
            logger.error("Skipping evaluation for this model due to configuration issues.")
            return  # Exit gracefully without results
            
        finally:
            # Always unload model
            self.model_manager.unload_model(model_id)
            logger.info("-" * 41 + "\n")
            time.sleep(2)
    
    def _load_datasets_config(self, datasets_config = None) -> List[Dict[str, Any]]:
        """Load datasets configuration from file, list, or use defaults."""
        if datasets_config:
            # If it's already a list, return it directly
            if isinstance(datasets_config, list):
                return datasets_config
            # Otherwise treat as file path
            try:
                with open(datasets_config, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON config file: {datasets_config}")
                logger.error(f"{e.msg} at line {e.lineno}, column {e.colno} (char {e.pos})")
                logger.error("Hint: Check for missing commas, quotes, or mismatched brackets.")
                raise
            except FileNotFoundError:
                logger.error(f"Config file not found: {datasets_config}")
                raise
        else:
            return default_datasets_to_run
    
    def _evaluate_single_dataset(self, 
                                model_id: str,
                                dataset_config: Dict[str, Any],
                                sample_size: int = 0,
                                seed: int = 42,
                                cli_explicit_args: set = None,
                                **cli_kwargs) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single dataset.
        
        Args:
            model_id: Model identifier
            dataset_config: Dataset configuration dictionary
            sample_size: Number of samples to evaluate
            seed: Random seed
            cli_explicit_args: Set of CLI argument names that were explicitly set by user
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
        kwargs = merge_eval_kwargs(dataset_config, cli_args, ["seed", "sample_size", "split", "subset"], cli_explicit_args)
        
        logger.debug(f"▶️ Evaluating {dataset_name} [{eval_type}]...")
        
        try:
            return self.dataset_registry.evaluate_dataset(
                eval_type=eval_type,
                model_id=model_id,
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to evaluate {dataset_name}: {e}")
            return None
    
    def _save_results(self, results: List[Dict[str, Any]]) -> None:
        """Save evaluation results to CSV."""
        if not results:
            return
            
        with open("evaluation_summary.csv", "w", newline="") as csvfile:
            fieldnames = ["dataset", "correct", "total", "skipped", "accuracy", "tok_per_sec"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        
        logger.info("📄 Results saved to evaluation_summary.csv")
    
    def _log_evaluation_summary(self, 
                               model_id: str,
                               results: List[Dict[str, Any]], 
                               start_time: float,
                               total_tokens_per_sec: float,
                               raw_duration: bool = False) -> None:
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
        total_correct = sum(r['correct'] for r in results)
        total_questions = sum(r['total'] for r in results)
        overall_accuracy = round((total_correct / total_questions * 100), 2) if total_questions > 0 else 0.0
        avg_tokens_per_sec = total_tokens_per_sec / len(results) if results else 0.0
        
        logger.info(f"📊 Overall accuracy: {overall_accuracy}%")
        logger.info(f"📊 Overall average response tokens per second: {avg_tokens_per_sec:.2f}")
