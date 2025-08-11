"""
Unit tests for CLI entry point (studioeval.py).
Tests argument parsing, error handling, and integration with orchestrator.
"""
import pytest
import argparse
import sys
from unittest.mock import patch, Mock, MagicMock
from io import StringIO

import studioeval


class TestCLIArgumentParsing:
    """Test command line argument parsing."""
    
    def test_basic_model_argument(self):
        """Test basic model argument parsing."""
        test_args = ['--model', 'test-model']
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging'):
                    mock_instance = Mock()
                    mock_instance.run_evaluation.return_value = True
                    mock_orchestrator.return_value = mock_instance
                    
                    studioeval.main()
                    
                    # Verify orchestrator was called with correct model
                    mock_instance.run_evaluation.assert_called_once()
                    call_kwargs = mock_instance.run_evaluation.call_args[1]
                    assert call_kwargs['model'] == 'test-model'
    
    def test_all_models_argument(self):
        """Test --all models argument."""
        test_args = ['--all']
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging'):
                    mock_instance = Mock()
                    mock_instance.run_evaluation.return_value = True
                    mock_orchestrator.return_value = mock_instance
                    
                    studioeval.main()
                    
                    call_kwargs = mock_instance.run_evaluation.call_args[1]
                    assert call_kwargs['all_models'] is True
    
    def test_model_filter_argument(self):
        """Test model filter argument parsing."""
        test_args = ['--model-filter', 'llama*']
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging'):
                    mock_instance = Mock()
                    mock_instance.run_evaluation.return_value = True
                    mock_orchestrator.return_value = mock_instance
                    
                    studioeval.main()
                    
                    call_kwargs = mock_instance.run_evaluation.call_args[1]
                    assert call_kwargs['model_filter'] == 'llama*'
    
    def test_datasets_config_argument(self):
        """Test datasets config argument parsing."""
        test_args = ['--datasets-config', 'config.json']
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging'):
                    mock_instance = Mock()
                    mock_instance.run_evaluation.return_value = True
                    mock_orchestrator.return_value = mock_instance
                    
                    studioeval.main()
                    
                    call_kwargs = mock_instance.run_evaluation.call_args[1]
                    assert call_kwargs['datasets_config'] == 'config.json'
    
    def test_sample_size_argument(self):
        """Test sample size argument parsing."""
        test_args = ['--sample-size', '100', '--model', 'test']
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging'):
                    mock_instance = Mock()
                    mock_instance.run_evaluation.return_value = True
                    mock_orchestrator.return_value = mock_instance
                    
                    studioeval.main()
                    
                    call_kwargs = mock_instance.run_evaluation.call_args[1]
                    assert call_kwargs['sample_size'] == 100
    
    def test_seed_argument(self):
        """Test seed argument parsing."""
        test_args = ['--seed', '123', '--model', 'test']
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging'):
                    mock_instance = Mock()
                    mock_instance.run_evaluation.return_value = True
                    mock_orchestrator.return_value = mock_instance
                    
                    studioeval.main()
                    
                    call_kwargs = mock_instance.run_evaluation.call_args[1]
                    assert call_kwargs['seed'] == 123
    
    def test_logging_arguments(self):
        """Test logging-related arguments."""
        test_args = ['--log-level', 'DEBUG', '--log-file', 'test.log', '--model', 'test']
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging') as mock_logging:
                    mock_instance = Mock()
                    mock_instance.run_evaluation.return_value = True
                    mock_orchestrator.return_value = mock_instance
                    
                    studioeval.main()
                    
                    # Verify logging setup was called with correct parameters
                    mock_logging.assert_called_once_with(log_level='DEBUG', log_file='test.log')
    
    def test_communication_config_argument(self):
        """Test communication config argument."""
        test_args = ['--comm-config', 'comm.json', '--model', 'test']
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging'):
                    with patch.object(studioeval, 'load_comm_config') as mock_load_config:
                        mock_instance = Mock()
                        mock_instance.run_evaluation.return_value = True
                        mock_orchestrator.return_value = mock_instance
                        
                        studioeval.main()
                        
                        # Verify comm config loading was called
                        mock_load_config.assert_called_once_with('comm.json')
    
class TestCLIExplicitArgumentDetection:
    """Test CLI explicit argument detection logic."""
    
    def test_explicit_sample_size_detected(self):
        """Test that explicitly set sample-size is detected."""
        test_args = ['--sample-size', '50', '--model', 'test']
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging'):
                    mock_instance = Mock()
                    mock_instance.run_evaluation.return_value = True
                    mock_orchestrator.return_value = mock_instance
                    
                    studioeval.main()
                    
                    call_kwargs = mock_instance.run_evaluation.call_args[1]
                    explicit_args = call_kwargs['cli_explicit_args']
                    assert 'sample_size' in explicit_args
                    assert 'model' in explicit_args
    
    def test_explicit_seed_detected(self):
        """Test that explicitly set seed is detected."""
        test_args = ['--seed', '999', '--model', 'test']
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging'):
                    mock_instance = Mock()
                    mock_instance.run_evaluation.return_value = True
                    mock_orchestrator.return_value = mock_instance
                    
                    studioeval.main()
                    
                    call_kwargs = mock_instance.run_evaluation.call_args[1]
                    explicit_args = call_kwargs['cli_explicit_args']
                    assert 'seed' in explicit_args
    
    def test_dashed_arguments_converted(self):
        """Test that dashed arguments are converted to underscore format."""
        test_args = ['--model-filter', 'test*', '--datasets-config', 'test.json']
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging'):
                    mock_instance = Mock()
                    mock_instance.run_evaluation.return_value = True
                    mock_orchestrator.return_value = mock_instance
                    
                    studioeval.main()
                    
                    call_kwargs = mock_instance.run_evaluation.call_args[1]
                    explicit_args = call_kwargs['cli_explicit_args']
                    assert 'model_filter' in explicit_args
                    assert 'datasets_config' in explicit_args
    
    def test_no_explicit_args_empty_set(self):
        """Test that no explicit args results in empty set."""
        # Only use default arguments
        test_args = []
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging'):
                    mock_instance = Mock()
                    mock_instance.run_evaluation.return_value = True
                    mock_orchestrator.return_value = mock_instance
                    
                    studioeval.main()
                    
                    call_kwargs = mock_instance.run_evaluation.call_args[1]
                    explicit_args = call_kwargs['cli_explicit_args']
                    assert len(explicit_args) == 0


class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""
    
    def test_evaluation_failure_exits_with_code_1(self):
        """Test that evaluation failure causes exit with code 1."""
        test_args = ['--model', 'test']
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging'):
                    with patch.object(studioeval, 'exit') as mock_exit:
                        mock_instance = Mock()
                        mock_instance.run_evaluation.return_value = False  # Simulate failure
                        mock_orchestrator.return_value = mock_instance
                        
                        studioeval.main()
                        
                        mock_exit.assert_called_once_with(1)
    
    def test_evaluation_success_no_exit(self):
        """Test that evaluation success doesn't call exit."""
        test_args = ['--model', 'test']
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging'):
                    with patch.object(studioeval, 'exit') as mock_exit:
                        mock_instance = Mock()
                        mock_instance.run_evaluation.return_value = True  # Simulate success
                        mock_orchestrator.return_value = mock_instance
                        
                        studioeval.main()
                        
                        mock_exit.assert_not_called()
    
    def test_orchestrator_exception_propagates(self):
        """Test that orchestrator exceptions are not caught."""
        test_args = ['--model', 'test']
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging'):
                    mock_instance = Mock()
                    mock_instance.run_evaluation.side_effect = RuntimeError("Test error")
                    mock_orchestrator.return_value = mock_instance
                    
                    with pytest.raises(RuntimeError, match="Test error"):
                        studioeval.main()
    
    def test_invalid_sample_size_handled_by_argparse(self):
        """Test that invalid sample size is handled by argparse."""
        test_args = ['--sample-size', 'invalid', '--model', 'test']
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('sys.stderr', new_callable=StringIO):
                with pytest.raises(SystemExit):  # argparse exits on invalid int
                    studioeval.main()
    
    def test_missing_comm_config_file_handled(self):
        """Test handling of missing communication config file."""
        test_args = ['--comm-config', 'missing.json', '--model', 'test']
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging'):
                    with patch.object(studioeval, 'load_comm_config') as mock_load_config:
                        # Simulate load_comm_config raising an exception
                        mock_load_config.side_effect = FileNotFoundError("Config not found")
                        
                        with pytest.raises(FileNotFoundError):
                            studioeval.main()


class TestCLIIntegration:
    """Test CLI integration scenarios."""
    
    def test_full_argument_integration(self):
        """Test integration with all arguments."""
        test_args = [
            '--model', 'test-model',
            '--datasets-config', 'datasets.json',
            '--sample-size', '50',
            '--seed', '123',
            '--log-level', 'DEBUG',
            '--raw-duration'
        ]
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging') as mock_logging:
                    mock_instance = Mock()
                    mock_instance.run_evaluation.return_value = True
                    mock_orchestrator.return_value = mock_instance
                    
                    studioeval.main()
                    
                    # Verify all arguments were passed correctly
                    call_kwargs = mock_instance.run_evaluation.call_args[1]
                    assert call_kwargs['model'] == 'test-model'
                    assert call_kwargs['datasets_config'] == 'datasets.json'
                    assert call_kwargs['sample_size'] == 50
                    assert call_kwargs['seed'] == 123
                    assert call_kwargs['raw_duration'] is True
                    
                    # Verify explicit args detection
                    explicit_args = call_kwargs['cli_explicit_args']
                    expected_explicit = {'model', 'datasets_config', 'sample_size', 'seed', 'raw_duration'}
                    assert expected_explicit.issubset(explicit_args)
                    
                    # Verify logging setup
                    mock_logging.assert_called_once_with(log_level='DEBUG', log_file=None)
    
    def test_defaults_when_no_args_provided(self):
        """Test that defaults are used when no arguments provided."""
        test_args = []
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging') as mock_logging:
                    mock_instance = Mock()
                    mock_instance.run_evaluation.return_value = True
                    mock_orchestrator.return_value = mock_instance
                    
                    studioeval.main()
                    
                    # Verify defaults
                    call_kwargs = mock_instance.run_evaluation.call_args[1]
                    assert call_kwargs['model'] is None
                    assert call_kwargs['sample_size'] == 0  # Default
                    assert call_kwargs['seed'] == 42  # Default
                    assert call_kwargs['raw_duration'] is False
                    
                    # Verify no explicit args
                    explicit_args = call_kwargs['cli_explicit_args']
                    assert len(explicit_args) == 0
                    
                    # Verify default logging
                    mock_logging.assert_called_once_with(log_level='INFO', log_file=None)


class TestCLIEdgeCases:
    """Test CLI edge cases and boundary conditions."""
    
    def test_zero_sample_size(self):
        """Test sample size of zero."""
        test_args = ['--sample-size', '0', '--model', 'test']
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging'):
                    mock_instance = Mock()
                    mock_instance.run_evaluation.return_value = True
                    mock_orchestrator.return_value = mock_instance
                    
                    studioeval.main()
                    
                    call_kwargs = mock_instance.run_evaluation.call_args[1]
                    assert call_kwargs['sample_size'] == 0
    
    def test_negative_seed(self):
        """Test negative seed value."""
        test_args = ['--seed', '-1', '--model', 'test']
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging'):
                    mock_instance = Mock()
                    mock_instance.run_evaluation.return_value = True
                    mock_orchestrator.return_value = mock_instance
                    
                    studioeval.main()
                    
                    call_kwargs = mock_instance.run_evaluation.call_args[1]
                    assert call_kwargs['seed'] == -1
    
    def test_empty_string_arguments(self):
        """Test empty string arguments."""
        test_args = ['--model', '', '--datasets-config', '']
        
        with patch('sys.argv', ['studioeval.py'] + test_args):
            with patch('core.evaluator.EvaluationOrchestrator') as mock_orchestrator:
                with patch.object(studioeval, 'setup_logging'):
                    mock_instance = Mock()
                    mock_instance.run_evaluation.return_value = True
                    mock_orchestrator.return_value = mock_instance
                    
                    studioeval.main()
                    
                    call_kwargs = mock_instance.run_evaluation.call_args[1]
                    assert call_kwargs['model'] == ''
                    assert call_kwargs['datasets_config'] == ''
