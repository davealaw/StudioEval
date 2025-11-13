"""
Unit tests for configuration management.
"""

import json
from copy import deepcopy
from unittest.mock import patch

import pytest

from config.comm_config import (
    DEFAULT_CONFIG,
    _validate,
    get_comm_config,
    load_comm_config,
)


class TestConfigValidation:
    """Test configuration validation logic."""

    def test_validate_empty_config(self):
        """Test validation with empty config uses defaults."""
        result = _validate({})
        assert result == DEFAULT_CONFIG

    def test_validate_valid_config(self):
        """Test validation with valid config."""
        valid_config = {
            "timeout": 300,
            "GENERATION_PARAMS": {
                "temperature": 0.5,
                "topPSampling": 0.8,
                "topKSampling": 10,
                "repeatPenalty": 1.1,
                "maxTokens": 1500,
            },
        }
        result = _validate(valid_config)
        assert result["timeout"] == 300
        assert result["GENERATION_PARAMS"]["temperature"] == 0.5
        assert result["GENERATION_PARAMS"]["maxTokens"] == 1500

    def test_validate_invalid_timeout(self):
        """Test validation with invalid timeout falls back to default."""
        invalid_configs = [
            {"timeout": "not_a_number"},
            {"timeout": -1},
            {"timeout": 0},
            {"timeout": None},
        ]

        for config in invalid_configs:
            result = _validate(config)
            assert result["timeout"] == DEFAULT_CONFIG["timeout"]

    def test_validate_invalid_generation_params(self):
        """Test validation with invalid generation params."""
        # Non-dict GENERATION_PARAMS
        result = _validate({"GENERATION_PARAMS": "not_a_dict"})
        assert result["GENERATION_PARAMS"] == DEFAULT_CONFIG["GENERATION_PARAMS"]

        # Invalid numeric values
        invalid_gen_config = {
            "GENERATION_PARAMS": {
                "temperature": "not_numeric",
                "topPSampling": None,
                "maxTokens": "invalid",
            }
        }
        result = _validate(invalid_gen_config)
        # Should keep defaults for invalid fields
        assert (
            result["GENERATION_PARAMS"]["temperature"]
            == DEFAULT_CONFIG["GENERATION_PARAMS"]["temperature"]
        )
        assert (
            result["GENERATION_PARAMS"]["maxTokens"]
            == DEFAULT_CONFIG["GENERATION_PARAMS"]["maxTokens"]
        )

    def test_validate_unknown_generation_params(self):
        """Test that unknown generation params are preserved."""
        config_with_unknown = {
            "GENERATION_PARAMS": {
                "temperature": 0.7,
                "unknown_param": 123,
                "custom_setting": "value",
            }
        }
        result = _validate(config_with_unknown)
        assert result["GENERATION_PARAMS"]["temperature"] == 0.7
        assert result["GENERATION_PARAMS"]["unknown_param"] == 123
        assert result["GENERATION_PARAMS"]["custom_setting"] == "value"

    def test_validate_partial_config(self):
        """Test validation with partial configuration."""
        partial_config = {
            "timeout": 200,
            "GENERATION_PARAMS": {
                "temperature": 0.3
                # Missing other params - should use defaults
            },
        }
        result = _validate(partial_config)
        assert result["timeout"] == 200
        assert result["GENERATION_PARAMS"]["temperature"] == 0.3
        assert (
            result["GENERATION_PARAMS"]["maxTokens"]
            == DEFAULT_CONFIG["GENERATION_PARAMS"]["maxTokens"]
        )


class TestConfigLoading:
    """Test configuration file loading."""

    def test_load_valid_config_file(self, temp_config_file, sample_comm_config):
        """Test loading a valid config file."""
        config_path = temp_config_file(sample_comm_config)

        with patch("config.comm_config._config", deepcopy(DEFAULT_CONFIG)):
            load_comm_config(config_path)
            loaded_config = get_comm_config()

        assert loaded_config["timeout"] == sample_comm_config["timeout"]
        assert (
            loaded_config["GENERATION_PARAMS"]["temperature"]
            == sample_comm_config["GENERATION_PARAMS"]["temperature"]
        )

    def test_load_nonexistent_file(self):
        """Test loading non-existent config file uses defaults."""
        with patch("config.comm_config._config", deepcopy(DEFAULT_CONFIG)):
            load_comm_config("/nonexistent/path/config.json")
            loaded_config = get_comm_config()

        assert loaded_config == DEFAULT_CONFIG

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON file uses defaults."""
        invalid_json_file = tmp_path / "invalid.json"
        invalid_json_file.write_text('{"invalid": json, "missing": quote}')

        with patch("config.comm_config._config", deepcopy(DEFAULT_CONFIG)):
            load_comm_config(str(invalid_json_file))
            loaded_config = get_comm_config()

        assert loaded_config == DEFAULT_CONFIG

    def test_load_config_multiple_times(self, tmp_path):
        """Test loading config multiple times - last one wins."""
        config1 = {"timeout": 100}
        config2 = {"timeout": 200}

        # Create unique config files
        path1 = tmp_path / "config1.json"
        path2 = tmp_path / "config2.json"

        with open(path1, "w") as f:
            json.dump(config1, f)
        with open(path2, "w") as f:
            json.dump(config2, f)

        # Import the module to patch the global variable directly
        import config.comm_config as cc

        # Store original config
        original_config = cc._config

        try:
            # Reset to default state
            cc._config = deepcopy(DEFAULT_CONFIG)

            # Load first config
            load_comm_config(str(path1))
            first_config = get_comm_config()
            assert first_config["timeout"] == 100

            # Load second config - should override
            load_comm_config(str(path2))
            second_config = get_comm_config()
            assert second_config["timeout"] == 200
        finally:
            # Restore original config
            cc._config = original_config


class TestConfigGetters:
    """Test configuration getter functions."""

    def test_get_comm_config_returns_copy(self):
        """Test that get_comm_config returns independent copy."""
        config1 = get_comm_config()
        config2 = get_comm_config()

        # Modify one - shouldn't affect the other
        config1["timeout"] = 999
        assert config2["timeout"] != 999

    def test_get_comm_config_default_structure(self):
        """Test default config has expected structure."""
        config = get_comm_config()

        # Required keys
        assert "timeout" in config
        assert "GENERATION_PARAMS" in config

        # Generation params structure
        gen_params = config["GENERATION_PARAMS"]
        expected_keys = [
            "temperature",
            "topPSampling",
            "topKSampling",
            "repeatPenalty",
            "maxTokens",
        ]
        for key in expected_keys:
            assert key in gen_params

        # Type validation
        assert isinstance(config["timeout"], int)
        assert isinstance(gen_params["temperature"], (int, float))
        assert isinstance(gen_params["maxTokens"], int)


class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_config_structure(self):
        """Test that DEFAULT_CONFIG has expected structure."""
        assert isinstance(DEFAULT_CONFIG, dict)
        assert "timeout" in DEFAULT_CONFIG
        assert "GENERATION_PARAMS" in DEFAULT_CONFIG

    def test_default_config_immutability(self):
        """Test that DEFAULT_CONFIG doesn't get mutated."""
        original = deepcopy(DEFAULT_CONFIG)

        # Simulate some operations that might mutate
        _validate({"timeout": 999})
        load_comm_config("/nonexistent/file")

        # DEFAULT_CONFIG should remain unchanged
        assert original == DEFAULT_CONFIG

    def test_default_generation_params(self):
        """Test default generation parameters."""
        gen_params = DEFAULT_CONFIG["GENERATION_PARAMS"]

        # Test reasonable defaults
        assert 0.0 <= gen_params["temperature"] <= 2.0
        assert 0.0 <= gen_params["topPSampling"] <= 1.0
        assert gen_params["topKSampling"] >= 0
        assert gen_params["repeatPenalty"] >= 1.0
        assert gen_params["maxTokens"] > 0


@pytest.mark.parametrize(
    "invalid_input,expected_default",
    [
        ({"timeout": "string"}, DEFAULT_CONFIG["timeout"]),
        ({"timeout": -100}, DEFAULT_CONFIG["timeout"]),
        ({"GENERATION_PARAMS": []}, DEFAULT_CONFIG["GENERATION_PARAMS"]),
        ({"GENERATION_PARAMS": None}, DEFAULT_CONFIG["GENERATION_PARAMS"]),
    ],
)
def test_config_validation_edge_cases(invalid_input, expected_default):
    """Test config validation with various invalid inputs."""
    result = _validate(invalid_input)

    if "timeout" in invalid_input:
        assert result["timeout"] == expected_default
    if "GENERATION_PARAMS" in invalid_input:
        assert result["GENERATION_PARAMS"] == expected_default
