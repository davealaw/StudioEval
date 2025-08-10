"""
Unit tests for utility functions.
"""
import pytest
import argparse
from unittest.mock import Mock

from utils.params import merge_eval_kwargs
from utils.timing_utils import format_duration


class TestParamsMerging:
    """Test parameter merging functionality."""
    
    def test_merge_eval_kwargs_config_only(self):
        """Test merging when only config has values."""
        config = {
            "seed": 42,
            "sample_size": 100,
            "split": "train"
        }
        
        # Mock argparse namespace with None values
        cli_args = Mock()
        cli_args.seed = None
        cli_args.sample_size = None
        cli_args.split = None
        
        result = merge_eval_kwargs(config, cli_args, ["seed", "sample_size", "split"])
        
        assert result == {
            "seed": 42,
            "sample_size": 100, 
            "split": "train"
        }
        
    def test_merge_eval_kwargs_cli_only(self):
        """Test merging when only CLI args have values."""
        config = {}  # Empty config
        
        cli_args = Mock()
        cli_args.seed = 123
        cli_args.sample_size = 50
        cli_args.split = "test"
        
        result = merge_eval_kwargs(config, cli_args, ["seed", "sample_size", "split"])
        
        assert result == {
            "seed": 123,
            "sample_size": 50,
            "split": "test"
        }
        
    def test_merge_eval_kwargs_explicit_cli_overrides_config(self):
        """Test that explicitly set CLI args override config values."""
        config = {
            "seed": 42,
            "sample_size": 100,
            "split": "train"
        }
        
        cli_args = Mock()
        cli_args.seed = 999
        cli_args.sample_size = 50
        cli_args.split = "validation"
        
        # Simulate user explicitly setting these CLI args
        cli_explicit_args = {"seed", "sample_size", "split"}
        
        result = merge_eval_kwargs(config, cli_args, ["seed", "sample_size", "split"], cli_explicit_args)
        
        assert result == {
            "seed": 999,  # From CLI (explicit override)
            "sample_size": 50,  # From CLI (explicit override) 
            "split": "validation"  # From CLI (explicit override)
        }
        
    def test_merge_eval_kwargs_neither_has_value(self):
        """Test merging when neither config nor CLI has values."""
        config = {}
        
        cli_args = Mock()
        cli_args.seed = None
        cli_args.sample_size = None
        
        result = merge_eval_kwargs(config, cli_args, ["seed", "sample_size"])
        
        assert result == {}
        
    def test_merge_eval_kwargs_config_overrides_cli_defaults(self):
        """Test that config values override CLI defaults (non-explicit)."""
        config = {
            "seed": 42,
            "sample_size": 100,
            "split": "train"
        }
        
        cli_args = Mock()
        cli_args.seed = 999  # CLI default (not explicitly set)
        cli_args.sample_size = 50  # CLI default (not explicitly set)
        cli_args.split = "validation"  # CLI default (not explicitly set)
        
        # No explicitly set CLI args - these are defaults
        cli_explicit_args = set()
        
        result = merge_eval_kwargs(config, cli_args, ["seed", "sample_size", "split"], cli_explicit_args)
        
        assert result == {
            "seed": 42,  # From config (overrides CLI default)
            "sample_size": 100,  # From config (overrides CLI default)
            "split": "train"  # From config (overrides CLI default)
        }
        
    def test_merge_eval_kwargs_mixed_precedence(self):
        """Test mixed precedence: explicit CLI > config > CLI defaults."""
        config = {
            "seed": None,  # Should use CLI fallback
            "sample_size": 100,  # Should be used unless CLI explicit
            "split": "train"  # Should be used unless CLI explicit
        }
        
        cli_args = Mock()
        cli_args.seed = 42  # CLI default (fallback)
        cli_args.sample_size = 50  # CLI default (should be overridden by config)
        cli_args.split = "validation"  # CLI explicit (should override config)
        
        # Only split was explicitly set by user
        cli_explicit_args = {"split"}
        
        result = merge_eval_kwargs(config, cli_args, ["seed", "sample_size", "split"], cli_explicit_args)
        
        assert result == {
            "seed": 42,  # From CLI default (config was None)
            "sample_size": 100,  # From config (overrides CLI default)
            "split": "validation"  # From explicit CLI (overrides config)
        }
        
    def test_merge_eval_kwargs_zero_values_preserved(self):
        """Test that zero values in config are preserved (not treated as None)."""
        config = {
            "seed": 0,
            "sample_size": 0
        }
        
        cli_args = Mock()
        cli_args.seed = 42  # CLI default
        cli_args.sample_size = 100  # CLI default
        
        # No explicit CLI args
        cli_explicit_args = set()
        
        result = merge_eval_kwargs(config, cli_args, ["seed", "sample_size"], cli_explicit_args)
        
        assert result == {
            "seed": 0,  # From config (zero is preserved)
            "sample_size": 0  # From config (zero is preserved)
        }


class TestTimingUtils:
    """Test timing utility functions."""
    
    def test_format_duration_seconds_only(self):
        """Test formatting duration with seconds only."""
        assert format_duration(5.5) == "5.50s"
        assert format_duration(0.1) == "0.10s"
        assert format_duration(59.99) == "59.99s"
        
    def test_format_duration_minutes_and_seconds(self):
        """Test formatting duration with minutes and seconds."""
        assert format_duration(75.5) == "1m 15.50s"
        assert format_duration(120.0) == "2m 0.00s"
        assert format_duration(3599.99) == "59m 59.99s"
        
    def test_format_duration_hours_minutes_seconds(self):
        """Test formatting duration with hours, minutes, and seconds."""
        assert format_duration(3661.5) == "1h 1m 1.50s"
        assert format_duration(3600.0) == "1h 0m 0.00s"
        assert format_duration(7323.45) == "2h 2m 3.45s"
        
    def test_format_duration_edge_cases(self):
        """Test edge cases for duration formatting."""
        assert format_duration(0.0) == "0.00s"
        assert format_duration(0.001) == "0.00s"  # Rounds to 2 decimal places
        
    def test_format_duration_large_values(self):
        """Test formatting very large durations."""
        # 25 hours
        assert format_duration(90000) == "25h 0m 0.00s"
        
        # Multiple days worth of seconds
        day_in_seconds = 24 * 3600
        assert format_duration(day_in_seconds) == "24h 0m 0.00s"
        
    @pytest.mark.parametrize("seconds,expected", [
        (0.0, "0.00s"),
        (1.0, "1.00s"),
        (30.5, "30.50s"),
        (60.0, "1m 0.00s"),
        (90.25, "1m 30.25s"),
        (3600.0, "1h 0m 0.00s"),
        (3661.0, "1h 1m 1.00s"),
        (3723.45, "1h 2m 3.45s"),
    ])
    def test_format_duration_examples(self, seconds, expected):
        """Test duration formatting with specific examples."""
        assert format_duration(seconds) == expected
        
    def test_format_duration_precision(self):
        """Test that duration formatting maintains 2 decimal precision."""
        # Test various precision scenarios
        assert format_duration(1.234567) == "1.23s"  # Rounded down
        assert format_duration(1.235) == "1.24s"    # Python banker's rounding  
        assert format_duration(1.236) == "1.24s"    # Rounded up
