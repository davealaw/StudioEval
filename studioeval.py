import csv
import fnmatch
import json
import time
import argparse
import logging
from models.model_handling import is_lm_studio_server_running, load_model, unload_model, list_models, list_models_with_arch
from eval_datasets.custom.custom_mcq import evaluate_cumstom_mcq
from eval_datasets.custom.grammar import evaluate_grammar_dataset
from eval_datasets.custom.math import evaluate_math_dataset
from eval_datasets.huggingface.arc import evaluate_arc
from eval_datasets.huggingface.mmlu import evaluate_mmlu
from eval_datasets.huggingface.truthfulqa import evaluate_tiny_truthfulqa
from eval_datasets.huggingface.gsm8k import evaluate_gsm8k_dataset
from eval_datasets.huggingface.commonsense_qa import evaluate_commonsense_qa
from eval_datasets.huggingface.logiqa import evaluate_logiqa
from eval_datasets.default_evals import default_datasets_to_run
from models.model_loading import load_model_set_file
from utils.params import merge_eval_kwargs
from utils.logging import setup_logging
from config.comm_config import load_comm_config
from utils.timing_utils import format_duration

logger = logging.getLogger(__name__)

def main():


    parser = argparse.ArgumentParser(description="Evaluate LLMs via LM Studio")
    parser.add_argument("--model", help="Test a specific model ID")
    parser.add_argument('--model-filter', help="Glob-style filter for models, e.g. 'qwen3*' or '*4B*'")
    parser.add_argument("--model-set", type=str, help="Path to a file listing model IDs to evaluate")
    parser.add_argument("--model-arch", dest="model_arch", default=None, help="Glob-style architecture filter of models, e.g. 'qwen3' or 'qwen*'")
    parser.add_argument("--all", action="store_true", help="Test all loaded models")
    parser.add_argument("--datasets-config", type=str, help="Path to datasets_to_run JSON file")
    parser.add_argument("--skip", action="store_true", help="Skip models that are thinking models or not LLMs")
    parser.add_argument("--sample-size", type=int, default=0, help="Number of samples to evaluate from each dataset (0 for all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log-level", default="INFO", help="Set log level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--log-file", default=None, help="Optional file to write logs instead of console")
    parser.add_argument("--comm-config", type=str, help="Path to communication config JSON")
    parser.add_argument("--raw-duration", action="store_true", help="Output duration in seconds instead of human-readable format")
    args = parser.parse_args()

    setup_logging(log_level=args.log_level, log_file=args.log_file)

    if args.comm_config:
        load_comm_config(args.comm_config)

    logger.info("-----------------------------------------")
    logger.info("Starting evaluation...\n")

    if not is_lm_studio_server_running():
        return

    results = {}

    if args.model:
        models = [args.model]
    elif args.model_filter:
        all_models = list_models()
        if not all_models:
            logger.error("No models found on LM Studio server.")
            return
        models = [m for m in all_models if fnmatch.fnmatch(m, args.model_filter)]
        if not models:
            logger.error(f"No models matched filter: {args.model_filter}")
            return        
    elif args.model_set:
        requested = load_model_set_file(args.model_set)
        if not requested:
            logger.error("Model set file produced no model IDs.")
            return
        available = set(list_models())
        if not available:
            logger.error("No models found on LM Studio server.")
            return
        unknown = [m for m in requested if m not in available]
        if unknown:
            logger.warning(f"The following models are not available on the server and will be skipped: {unknown}")
        models = [m for m in requested if m in available]
        if not models:
            logger.error("None of the models in the model set are available on the server.")
            return        
    elif args.model_arch:   
        pat = args.model_arch.strip().lower()
        arch_map = list_models_with_arch()  # {model_key: architecture}

        models = [key for key, arch in arch_map.items() if fnmatch.fnmatch((arch or "").lower(), pat)]
        if not models:
            logger.error(f"No models match architecture pattern: {pat}")
            return
    elif args.all:
        models = list_models()
        if not models:
            logger.error("No models found on LM Studio server.")
            return
    else:
        logger.error("Please specify a model with --model, use --model-filter for pattern matching, or --all to test all models.")
        return

    logger.info("-----------------------------------------\n")

    for model_id in models:
        logger.info(f"--- Evaluating {model_id} ---")

        if args.skip:
            skip_prefixes = ( "glm", "kimi-dev", "qwq", "qwen/qwen3", "qwen3-128k", "openthinker2", "openai", "xbai", "mradermacher/xbai-o4", "mlx-community/xbai-o4", "gpt" )
            if any(model_id.startswith(prefix) for prefix in skip_prefixes):
                logger.info(f"Skipping {model_id} as it is a thinking model.")
                continue 

        load_model(model_id)
        
        start_time = time.time()
        results = []
        kwargs = {}
        total_datasets = 0
        total_tokens_per_sec = 0

        if args.datasets_config:
            try:
                with open(args.datasets_config, "r") as f:
                    datasets_to_run = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON config file: {args.datasets_config}")
                logger.error(f"{e.msg} at line {e.lineno}, column {e.colno} (char {e.pos})")
                logger.error("Hint: Check for missing commas, quotes, or mismatched brackets.")
                return
            except FileNotFoundError:
                logger.error(f"Config file not found: {args.datasets_config}")
                return
        else:
            datasets_to_run = default_datasets_to_run

        for dataset_config in datasets_to_run:
            eval_type = dataset_config["eval_type"]
            dataset_path = dataset_config["dataset_path"]
            dataset_name = dataset_config["dataset_name"]

            kwargs = merge_eval_kwargs(dataset_config, args, keys=["seed", "sample_size", "split", "subset"])

            logger.debug(f"▶️ Evaluating {dataset_name} [{eval_type}]...")

            if eval_type == "grammar":
                result = evaluate_grammar_dataset(model_id, dataset_path, dataset_name=dataset_name, **kwargs)
            elif eval_type == "custom_mcq":
                result = evaluate_cumstom_mcq(model_id, dataset_path, dataset_name=dataset_name, **kwargs)
            elif eval_type == "math":
                result = evaluate_math_dataset(model_id, dataset_path, dataset_name=dataset_name, **kwargs)  
            elif eval_type == "gsm8k":
                result = evaluate_gsm8k_dataset(model_id, dataset_path, dataset_name=dataset_name, **kwargs)
            elif eval_type == "arc":
                result = evaluate_arc(model_id, dataset_path, dataset_name=dataset_name, **kwargs)
            elif eval_type == "mmlu":
                result = evaluate_mmlu(model_id, dataset_path, dataset_name=dataset_name, **kwargs)
            elif eval_type == "commonsenseqa":
                result = evaluate_commonsense_qa(model_id, dataset_path, dataset_name=dataset_name, **kwargs)
            elif eval_type == "logiqa":
                result = evaluate_logiqa(model_id, dataset_path, dataset_name=dataset_name, **kwargs)
            #elif eval_type == "truthfulqa":
            #    result = evaluate_tiny_truthfulqa(model_id, dataset_path, dataset_name=dataset_name, **kwargs)
            else:
                logger.error(f"Unknown dataset: {dataset_name}\n")
                continue

            logger.info(f"✅ {dataset_name} accuracy: {result['accuracy']}% ({result['correct']}/{result['total']})")
            logger.info(f"📊 {dataset_name} average response tokens/sec: {result['tok_per_sec']:.2f}\n")

            total_datasets += 1
            total_tokens_per_sec += result['tok_per_sec']

            results.append(result)


        # Write results to CSV
        with open("evaluation_summary.csv", "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["dataset", "correct", "total", "skipped", "accuracy", "tok_per_sec", ])
            writer.writeheader()
            for row in results:
               writer.writerow(row)

        logger.info("📄 Results saved to evaluation_summary.csv")
        
        # Finalize testing
        end_time = time.time()
        elapsed = end_time - start_time

        if args.raw_duration:
            logger.info(f"📊 Took {elapsed:.2f} seconds to evaluate {model_id}")
        else:
            logger.info(f"📊 Took {format_duration(elapsed)} to evaluate {model_id}")

        logger.info(f"📊 Overall accuracy: {round((sum(r['correct'] for r in results) / sum(r['total'] for r in results) * 100), 2)}%")
        logger.info(f"📊 Overall average response tokens per second: {total_tokens_per_sec / total_datasets if total_datasets > 0 else 0:.2f}")
        
        unload_model(model_id)
        logger.info("-----------------------------------------\n")

        time.sleep(2)


if __name__ == "__main__":
    main()
