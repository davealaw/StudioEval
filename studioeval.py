import argparse
import logging
from utils.logging import setup_logging
from config.comm_config import load_comm_config
from __version__ import __version__

logger = logging.getLogger(__name__)

def main():
    """Main CLI entry point using refactored evaluation orchestrator."""
    parser = argparse.ArgumentParser(description="Evaluate LLMs via LM Studio")
    parser.add_argument("--version", action="version", version=f"StudioEval {__version__}")
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
    
    # Determine which arguments were explicitly provided by the user
    import sys
    cli_explicit_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            # Convert CLI format to internal format
            arg_name = arg[2:].replace('-', '_')
            cli_explicit_args.add(arg_name)

    # Setup logging and configuration
    setup_logging(log_level=args.log_level, log_file=args.log_file)

    if args.comm_config:
        load_comm_config(args.comm_config)

    # Import orchestrator only when needed (avoids import-time hanging)
    from core.evaluator import EvaluationOrchestrator
    
    # Create evaluation orchestrator and run
    orchestrator = EvaluationOrchestrator()
    
    success = orchestrator.run_evaluation(
        model=args.model,
        model_filter=args.model_filter, 
        model_set=args.model_set,
        model_arch=args.model_arch,
        all_models=args.all,
        datasets_config=args.datasets_config,
        skip_thinking_models=args.skip,
        sample_size=args.sample_size,
        seed=args.seed,
        raw_duration=args.raw_duration,
        cli_explicit_args=cli_explicit_args
    )
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
