#!/usr/bin/env python3
"""
Demo script showing the model accuracy summary functionality.

This creates sample evaluation data to demonstrate how the model_accuracy_summary.csv
file is generated and updated across multiple model evaluations with different datasets.
"""

import os
import sys
import tempfile
from unittest.mock import Mock

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.evaluator import EvaluationOrchestrator


def demo_model_summary():
    """Demonstrate model accuracy summary functionality."""
    print("🚀 Model Accuracy Summary Demo")
    print("=" * 50)
    
    # Create orchestrator with mocked dependencies (for demo purposes)
    mock_model_manager = Mock()
    mock_dataset_registry = Mock()
    orchestrator = EvaluationOrchestrator(mock_model_manager, mock_dataset_registry)
    
    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            print(f"📁 Working in temporary directory: {temp_dir}\n")
            
            # Demo 1: Evaluate first model on grammar and creative_writing
            print("📊 Evaluating Model 1: 'mistralai/magistral-small-2509'")
            model1_results = [
                {
                    'dataset': 'grammar',
                    'correct': 82,
                    'total': 100,
                    'skipped': 0,
                    'accuracy': 82.0,
                    'tok_per_sec': 21.1
                },
                {
                    'dataset': 'creative_writing',
                    'correct': 93,
                    'total': 120,
                    'skipped': 0,
                    'accuracy': 77.5,
                    'tok_per_sec': 26.16
                }
            ]
            
            orchestrator._save_model_accuracy_summary(
                model_id='mistralai/magistral-small-2509',
                results=model1_results,
                elapsed_time=885.53,  # 14m 45.53s
                avg_tokens_per_sec=23.63,
                raw_duration=False
            )
            
            print("   ✅ Model 1 evaluation complete")
            print(f"   📄 model_accuracy_summary.csv created")
            
            # Show current CSV content
            if os.path.exists('model_accuracy_summary.csv'):
                print("\n📋 Current CSV content:")
                with open('model_accuracy_summary.csv', 'r') as f:
                    for line in f:
                        print(f"   {line.strip()}")
            
            print()
            
            # Demo 2: Evaluate second model on different datasets
            print("📊 Evaluating Model 2: 'llama3.1:8b'")
            model2_results = [
                {
                    'dataset': 'creative_writing',
                    'correct': 98,
                    'total': 120,
                    'skipped': 0,
                    'accuracy': 81.7,
                    'tok_per_sec': 28.5
                },
                {
                    'dataset': 'data_analysis_mcq',
                    'correct': 129,
                    'total': 135,
                    'skipped': 0,
                    'accuracy': 95.56,
                    'tok_per_sec': 26.19
                },
                {
                    'dataset': 'elementary_math',
                    'correct': 89,
                    'total': 100,
                    'skipped': 0,
                    'accuracy': 89.0,
                    'tok_per_sec': 24.8
                }
            ]
            
            orchestrator._save_model_accuracy_summary(
                model_id='llama3.1:8b',
                results=model2_results,
                elapsed_time=756.2,  # 12m 36.2s
                avg_tokens_per_sec=26.49,
                raw_duration=False
            )
            
            print("   ✅ Model 2 evaluation complete")
            print("   📄 model_accuracy_summary.csv updated with new model")
            
            # Show updated CSV content
            if os.path.exists('model_accuracy_summary.csv'):
                print("\n📋 Updated CSV content:")
                with open('model_accuracy_summary.csv', 'r') as f:
                    for line in f:
                        print(f"   {line.strip()}")
            
            print()
            
            # Demo 3: Re-evaluate first model with additional datasets
            print("📊 Re-evaluating Model 1 with additional datasets")
            model1_extended_results = [
                {
                    'dataset': 'grammar',
                    'correct': 85,  # Improved score
                    'total': 100,
                    'skipped': 0,
                    'accuracy': 85.0,
                    'tok_per_sec': 22.3
                },
                {
                    'dataset': 'creative_writing',
                    'correct': 96,  # Improved score
                    'total': 120,
                    'skipped': 0,
                    'accuracy': 80.0,
                    'tok_per_sec': 27.1
                },
                {
                    'dataset': 'elementary_math',
                    'correct': 92,
                    'total': 100,
                    'skipped': 0,
                    'accuracy': 92.0,
                    'tok_per_sec': 23.8
                }
            ]
            
            orchestrator._save_model_accuracy_summary(
                model_id='mistralai/magistral-small-2509',
                results=model1_extended_results,
                elapsed_time=920.1,  # 15m 20.1s
                avg_tokens_per_sec=24.4,
                raw_duration=False
            )
            
            print("   ✅ Model 1 re-evaluation complete")
            print("   📄 model_accuracy_summary.csv updated with new results")
            
            # Show final CSV content
            if os.path.exists('model_accuracy_summary.csv'):
                print("\n📋 Final CSV content (note dynamic columns):")
                with open('model_accuracy_summary.csv', 'r') as f:
                    for line in f:
                        print(f"   {line.strip()}")
            
            print("\n🎉 Demo Complete!")
            print("\nKey features demonstrated:")
            print("• ✅ Dynamic column generation based on datasets evaluated")
            print("• ✅ Multiple model tracking in single CSV file") 
            print("• ✅ Model data updates when re-evaluated")
            print("• ✅ Handles models evaluated on different dataset subsets")
            print("• ✅ Automatic overall accuracy calculation")
            print("• ✅ Human-readable duration formatting")
            print("• ✅ Persistent data across evaluation runs")
            
        finally:
            os.chdir(original_cwd)


if __name__ == '__main__':
    demo_model_summary()