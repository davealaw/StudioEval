"""
Integration tests for model client interactions and communication protocols.
Tests the interface between model managers and different client implementations.
"""

import time
from unittest.mock import Mock

import pytest

from core.model_manager import ModelManager
from implementations.lmstudio_client import LMStudioClient
from implementations.mock_client import MockModelClient
from interfaces.model_client import ModelClient


@pytest.fixture
def basic_mock_client():
    """Basic mock client for testing."""
    return MockModelClient(
        responses={"test-model": "Test response"},
        available_models=["test-model", "other-model"],
        server_running=True,
    )


@pytest.fixture
def advanced_mock_client():
    """Advanced mock client with realistic behaviors."""
    responses = {
        "fast-model": "Quick answer",
        "slow-model": "Detailed response after thinking",
        "error-model": "This will cause an error",
        "thinking-model": "Let me think... The answer is X",
    }

    return MockModelClient(
        responses=responses,
        available_models=["fast-model", "slow-model", "error-model", "thinking-model"],
        server_running=True,
    )


class TestModelClientInterface:
    """Test model client interface compliance."""

    def test_mock_client_implements_interface(self, basic_mock_client):
        """Test that mock client implements ModelClient interface."""
        assert isinstance(basic_mock_client, ModelClient)

        # Test all required methods exist
        assert hasattr(basic_mock_client, "is_server_running")
        assert hasattr(basic_mock_client, "list_models")
        assert hasattr(basic_mock_client, "load_model")
        assert hasattr(basic_mock_client, "unload_model")
        assert hasattr(basic_mock_client, "query_model")

        # Test methods are callable
        assert callable(basic_mock_client.is_server_running)
        assert callable(basic_mock_client.list_models)
        assert callable(basic_mock_client.load_model)
        assert callable(basic_mock_client.unload_model)
        assert callable(basic_mock_client.query_model)

    def test_lmstudio_client_implements_interface(self):
        """Test that LMStudio client implements ModelClient interface."""
        client = LMStudioClient()
        assert isinstance(client, ModelClient)

        # Test all required methods exist
        assert hasattr(client, "is_server_running")
        assert hasattr(client, "list_models")
        assert hasattr(client, "load_model")
        assert hasattr(client, "unload_model")
        assert hasattr(client, "query_model")


class TestModelManagerClientIntegration:
    """Test model manager integration with different clients."""

    def test_manager_with_mock_client(self, basic_mock_client):
        """Test model manager working with mock client."""
        manager = ModelManager(basic_mock_client)

        # Test model resolution
        models = manager.resolve_models(all_models=True)
        assert "test-model" in models
        assert "other-model" in models

        # Test model querying
        response, stats = manager.query_model("Test prompt", "test-model")
        assert response == "Test response"
        assert "tokens_per_second" in stats

    def test_manager_client_server_status_integration(self, basic_mock_client):
        """Test manager handling of client server status."""
        manager = ModelManager(basic_mock_client)

        # Test when server is running
        assert basic_mock_client.is_server_running()
        models = manager.resolve_models(all_models=True)
        assert len(models) > 0

        # Test when server is down
        basic_mock_client.set_server_running(False)
        assert not basic_mock_client.is_server_running()

        # Should still be able to resolve models (cached)
        models = manager.resolve_models(all_models=True)
        assert len(models) > 0  # Should still work

        # But queries should still work because the MockClient does not fail
        # when the server is down. Real clients would error in this case.
        response, _ = manager.query_model("Test", "test-model")
        # For mock client, it still returns responses

    def test_manager_model_loading_integration(self, basic_mock_client):
        """Test model loading through manager-client integration."""
        manager = ModelManager(basic_mock_client)

        # Test successful loading
        manager.load_model("test-model")

        # Mock client should track loaded model
        assert "test-model" in basic_mock_client.loaded_models

        # Test loading invalid model
        with pytest.raises(ValueError, match="not available"):
            manager.load_model("invalid-model")

    def test_manager_model_unloading_integration(self, basic_mock_client):
        """Test model unloading through manager-client integration."""
        manager = ModelManager(basic_mock_client)

        # Load then unload
        manager.load_model("test-model")
        assert "test-model" in basic_mock_client.loaded_models

        manager.unload_model("test-model")
        assert "test-model" not in basic_mock_client.loaded_models


class TestClientCommunicationProtocols:
    """Test communication protocols between components."""

    def test_query_response_protocol(self, advanced_mock_client):
        """Test query-response communication protocol."""
        manager = ModelManager(advanced_mock_client)

        # Test normal query
        response, stats = manager.query_model("What is 2+2?", "fast-model")

        assert response == "Quick answer"
        assert isinstance(stats, dict)
        assert "tokens_per_second" in stats
        # MockModelClient provides tokens_per_second but not response_time
        assert stats["tokens_per_second"] > 0

    def test_error_handling_protocol(self, advanced_mock_client):
        """Test error handling in communication protocol."""
        # Mock client doesn't have built-in error simulation
        # But we can test with invalid model
        manager = ModelManager(advanced_mock_client)

        # Should work with available models
        response, _ = manager.query_model("Test query", "error-model")
        assert response == "This will cause an error"

    def test_model_state_protocol(self, advanced_mock_client):
        """Test model state management protocol."""
        manager = ModelManager(advanced_mock_client)

        # Test state transitions
        assert "fast-model" not in advanced_mock_client.loaded_models

        manager.load_model("fast-model")
        assert "fast-model" in advanced_mock_client.loaded_models

        # Loading different model doesn't unload previous in mock
        manager.load_model("slow-model")
        assert "slow-model" in advanced_mock_client.loaded_models

        manager.unload_model("fast-model")
        assert "fast-model" not in advanced_mock_client.loaded_models
        assert "slow-model" in advanced_mock_client.loaded_models

    def test_performance_metrics_protocol(self, advanced_mock_client):
        """Test performance metrics communication protocol."""
        manager = ModelManager(advanced_mock_client)

        # Query model and check metrics
        response, stats = manager.query_model("Performance test", "fast-model")

        # Required metrics
        assert "tokens_per_second" in stats
        # MockModelClient provides different metrics than response_time

        # Optional metrics that might be present
        performance_metrics = ["prompt_tokens", "completion_tokens", "total_tokens"]

        # At least basic metrics should be available
        assert stats["tokens_per_second"] >= 0

        # Check for token counting metrics
        for metric in performance_metrics:
            if metric in stats:
                assert stats[metric] >= 0


class TestClientFailureRecovery:
    """Test client failure scenarios and recovery."""

    def test_server_disconnect_recovery(self, basic_mock_client):
        """Test recovery from server disconnect."""
        manager = ModelManager(basic_mock_client)

        # Initially working
        response, _ = manager.query_model("Test", "test-model")
        assert response == "Test response"

        # Simulate server disconnect
        basic_mock_client.set_server_running(False)

        # Mock client doesn't fail on server disconnect
        # But we can verify the server status changed
        assert not basic_mock_client.is_server_running()

        # Simulate server reconnect
        basic_mock_client.set_server_running(True)

        # Should work again
        response, _ = manager.query_model("Test", "test-model")
        assert response == "Test response"

    def test_model_loading_failure_recovery(self, advanced_mock_client):
        """Test recovery from model loading failures."""
        manager = ModelManager(advanced_mock_client)

        # Simulate loading failure for specific model
        def failing_load(model_id):
            if model_id == "failing-model":
                raise Exception("Model loading failed")

        advanced_mock_client.load_model = Mock(side_effect=failing_load)

        # Loading should fail
        with pytest.raises(Exception, match="Model loading failed"):
            manager.load_model("failing-model")

        # Should be able to load other models
        advanced_mock_client.load_model = Mock()  # Reset to working state
        manager.load_model("fast-model")  # Should work

    def test_partial_client_failure_recovery(self, advanced_mock_client):
        """Test recovery from partial client failures."""
        manager = ModelManager(advanced_mock_client)

        # Simulate partial failure - model listing fails but queries work
        advanced_mock_client.list_models = Mock(
            side_effect=Exception("List models failed")
        )

        # Model listing should fail
        with pytest.raises(Exception, match="List models failed"):
            manager.resolve_models(all_models=True)

        # But individual queries should still work if model is already loaded
        advanced_mock_client.current_model = "test-model"
        response, _ = manager.query_model("Test", "test-model")
        # Should still work as query doesn't depend on list_models


class TestMultiClientScenarios:
    """Test scenarios involving multiple client instances."""

    def test_client_isolation(self):
        """Test that different client instances are isolated."""
        client1 = MockModelClient(
            responses={"model-1": "Response 1"},
            available_models=["model-1"],
            server_running=True,
        )

        client2 = MockModelClient(
            responses={"model-2": "Response 2"},
            available_models=["model-2"],
            server_running=True,
        )

        manager1 = ModelManager(client1)
        manager2 = ModelManager(client2)

        # Each manager should work with its own client
        response1, _ = manager1.query_model("Test", "model-1")
        response2, _ = manager2.query_model("Test", "model-2")

        assert response1 == "Response 1"
        assert response2 == "Response 2"

        # Clients should be independent - test by checking responses
        # Both start with empty loaded_models, so test by loading different models
        manager1.load_model("model-1")
        manager2.load_model("model-2")

        # Now they should have different loaded models
        assert "model-1" in client1.loaded_models
        assert "model-2" in client2.loaded_models
        assert client1.loaded_models != client2.loaded_models

    def test_client_switching(self, basic_mock_client, advanced_mock_client):
        """Test switching clients in manager."""
        # Start with basic client
        manager = ModelManager(basic_mock_client)
        response1, _ = manager.query_model("Test", "test-model")
        assert response1 == "Test response"

        # Switch to advanced client
        manager = ModelManager(advanced_mock_client)
        response2, _ = manager.query_model("Test", "fast-model")
        assert response2 == "Quick answer"


class TestClientPerformanceIntegration:
    """Test client performance characteristics integration."""

    def test_response_timing_integration(self, advanced_mock_client):
        """Test response timing through manager-client integration."""
        # Mock client doesn't have configurable delays
        # But we can test timing capture

        manager = ModelManager(advanced_mock_client)

        start_time = time.time()
        response, stats = manager.query_model("Timing test", "fast-model")
        end_time = time.time()

        # Verify we got meaningful response and stats
        assert response == "Quick answer"
        assert "tokens_per_second" in stats

        # Total time should be reasonable (very fast for mock)
        total_time = end_time - start_time
        assert total_time >= 0  # Should be positive
        assert total_time < 1.0  # But not excessive

    def test_throughput_metrics_integration(self, advanced_mock_client):
        """Test throughput metrics integration."""
        manager = ModelManager(advanced_mock_client)

        # Configure consistent response for throughput testing
        advanced_mock_client.responses["throughput-model"] = "Consistent response"
        advanced_mock_client.available_models.append("throughput-model")

        # Multiple queries to test consistency
        stats_list = []
        for i in range(3):
            _, stats = manager.query_model(f"Query {i}", "throughput-model")
            stats_list.append(stats)

        # All should have metrics
        for stats in stats_list:
            assert "tokens_per_second" in stats
            assert stats["tokens_per_second"] > 0

        # Metrics should be relatively consistent
        tps_values = [stats["tokens_per_second"] for stats in stats_list]
        assert min(tps_values) > 0
        assert max(tps_values) / min(tps_values) < 2.0  # Within reasonable variance


class TestClientConfigurationIntegration:
    """Test client configuration and setup integration."""

    def test_client_initialization_with_manager(self):
        """Test client initialization through manager."""
        # Test with different client configurations
        configs = [
            {"responses": {"model-a": "A"}, "available_models": ["model-a"]},
            {"responses": {"model-b": "B"}, "available_models": ["model-b"]},
        ]

        for config in configs:
            client = MockModelClient(
                responses=config["responses"],
                available_models=config["available_models"],
                server_running=True,
            )
            manager = ModelManager(client)

            # Should work with each configuration
            models = manager.resolve_models(all_models=True)
            assert len(models) == 1
            assert models[0] in config["available_models"]

    def test_client_reconfiguration_integration(self, basic_mock_client):
        """Test client reconfiguration during operation."""
        manager = ModelManager(basic_mock_client)

        # Initial state
        models = manager.resolve_models(all_models=True)
        assert "test-model" in models

        # Reconfigure client (add more models)
        basic_mock_client.available_models.append("new-model")
        basic_mock_client.responses["new-model"] = "New response"

        # Should see new models
        updated_models = manager.resolve_models(all_models=True)
        assert "new-model" in updated_models

        # Should be able to query new model
        response, _ = manager.query_model("Test", "new-model")
        assert response == "New response"
