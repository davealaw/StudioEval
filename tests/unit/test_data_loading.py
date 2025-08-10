"""
Unit tests for data loading utilities.
"""
import pytest
import json
from unittest.mock import patch, Mock, MagicMock
from datasets import Dataset

from utils.data_loading import load_dataset_with_config, load_json_dataset_with_config


class TestDatasetLoading:
    """Test dataset loading functionality."""
    
    @patch('utils.data_loading.load_dataset')
    def test_load_dataset_basic(self, mock_load_dataset):
        """Test basic dataset loading."""
        # Setup mock
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load_dataset.return_value = mock_dataset
        
        # Test
        result = load_dataset_with_config("test_dataset")
        
        # Assertions
        mock_load_dataset.assert_called_once_with("test_dataset", split="train")
        assert result == mock_dataset
        
    @patch('utils.data_loading.load_dataset')
    def test_load_dataset_with_subset(self, mock_load_dataset):
        """Test dataset loading with subset."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load_dataset.return_value = mock_dataset
        
        result = load_dataset_with_config("test_dataset", subset="test_subset")
        
        mock_load_dataset.assert_called_once_with("test_dataset", split="train", name="test_subset")
        
    @patch('utils.data_loading.load_dataset')
    def test_load_dataset_with_revision(self, mock_load_dataset):
        """Test dataset loading with revision."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load_dataset.return_value = mock_dataset
        
        result = load_dataset_with_config("test_dataset", revision="refs/convert/parquet")
        
        mock_load_dataset.assert_called_once_with("test_dataset", split="train", revision="refs/convert/parquet")
        
    @patch('utils.data_loading.load_dataset')
    def test_load_dataset_custom_split(self, mock_load_dataset):
        """Test dataset loading with custom split."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=50)
        mock_load_dataset.return_value = mock_dataset
        
        result = load_dataset_with_config("test_dataset", split="test")
        
        mock_load_dataset.assert_called_once_with("test_dataset", split="test")


class TestDatasetSampling:
    """Test dataset sampling functionality."""
    
    @patch('utils.data_loading.load_dataset')
    def test_sample_size_smaller_than_dataset(self, mock_load_dataset):
        """Test sampling when sample_size < dataset size."""
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_shuffled = Mock()
        mock_selected = Mock()
        mock_dataset.shuffle.return_value = mock_shuffled
        mock_shuffled.select.return_value = mock_selected
        mock_load_dataset.return_value = mock_dataset
        
        # Test
        result = load_dataset_with_config("test_dataset", sample_size=10, seed=42)
        
        # Assertions
        mock_dataset.shuffle.assert_called_once_with(seed=42)
        mock_shuffled.select.assert_called_once_with(range(10))
        assert result == mock_selected
        
    @patch('utils.data_loading.load_dataset')
    def test_sample_size_larger_than_dataset(self, mock_load_dataset):
        """Test when sample_size > dataset size."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=5)
        mock_load_dataset.return_value = mock_dataset
        
        # Should return full dataset without sampling
        result = load_dataset_with_config("test_dataset", sample_size=10)
        
        assert result == mock_dataset
        # shuffle/select should not be called
        assert not hasattr(mock_dataset, 'shuffle') or not mock_dataset.shuffle.called
        
    @patch('utils.data_loading.load_dataset')
    def test_sample_size_zero(self, mock_load_dataset):
        """Test with sample_size=0 (full dataset)."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load_dataset.return_value = mock_dataset
        
        result = load_dataset_with_config("test_dataset", sample_size=0)
        
        assert result == mock_dataset
        # No sampling should occur
        assert not hasattr(mock_dataset, 'shuffle') or not mock_dataset.shuffle.called
        
    @patch('utils.data_loading.load_dataset')
    def test_default_seed_when_none(self, mock_load_dataset):
        """Test default seed is used when seed=None and sampling."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_shuffled = Mock()
        mock_selected = Mock()
        mock_dataset.shuffle.return_value = mock_shuffled
        mock_shuffled.select.return_value = mock_selected
        mock_load_dataset.return_value = mock_dataset
        
        result = load_dataset_with_config("test_dataset", sample_size=10, seed=None)
        
        # Should use default seed of 42
        mock_dataset.shuffle.assert_called_once_with(seed=42)


class TestJSONDatasetLoading:
    """Test JSON/JSONL dataset loading."""
    
    @patch('utils.data_loading.load_dataset')
    def test_load_json_dataset_basic(self, mock_load_dataset):
        """Test basic JSON dataset loading."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        mock_load_dataset.return_value = mock_dataset
        
        result = load_json_dataset_with_config("test.jsonl")
        
        mock_load_dataset.assert_called_once_with("json", data_files="test.jsonl", split="train")
        assert result == mock_dataset
        
    @patch('utils.data_loading.load_dataset')
    def test_load_json_dataset_with_sampling(self, mock_load_dataset):
        """Test JSON dataset loading with sampling."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_shuffled = Mock()
        mock_selected = Mock()
        mock_dataset.shuffle.return_value = mock_shuffled
        mock_shuffled.select.return_value = mock_selected
        mock_load_dataset.return_value = mock_dataset
        
        result = load_json_dataset_with_config("test.jsonl", sample_size=5, seed=123)
        
        mock_dataset.shuffle.assert_called_once_with(seed=123)
        mock_shuffled.select.assert_called_once_with(range(5))
        assert result == mock_selected
        
    @patch('utils.data_loading.load_dataset')
    def test_load_json_dataset_error_handling(self, mock_load_dataset):
        """Test JSON dataset loading error handling."""
        mock_load_dataset.side_effect = Exception("Dataset loading failed")
        
        # Should raise exception instead of returning None
        with pytest.raises(Exception, match="Dataset loading failed"):
            load_json_dataset_with_config("nonexistent.jsonl")
        
    @patch('utils.data_loading.load_dataset')
    def test_load_json_dataset_sample_exceeds_size(self, mock_load_dataset):
        """Test JSON dataset when sample size exceeds dataset size."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=3)
        mock_load_dataset.return_value = mock_dataset
        
        # Should return full dataset, no sampling
        result = load_json_dataset_with_config("test.jsonl", sample_size=10)
        
        assert result == mock_dataset
        assert not hasattr(mock_dataset, 'shuffle') or not mock_dataset.shuffle.called


class TestDataLoadingIntegration:
    """Integration-style tests with real data structures."""
    
    def test_load_with_real_jsonl_file(self, temp_jsonl_file, sample_mcq_data):
        """Test loading actual JSONL file (mocked HF datasets)."""
        jsonl_path = temp_jsonl_file(sample_mcq_data)
        
        with patch('utils.data_loading.load_dataset') as mock_load:
            # Mock successful loading
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=len(sample_mcq_data))
            mock_load.return_value = mock_dataset
            
            result = load_json_dataset_with_config(jsonl_path)
            
            mock_load.assert_called_once_with("json", data_files=jsonl_path, split="train")
            assert result == mock_dataset
            
    def test_dataset_config_parameter_combinations(self):
        """Test various parameter combinations."""
        with patch('utils.data_loading.load_dataset') as mock_load:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=100)
            mock_shuffled = Mock()
            mock_selected = Mock()
            mock_dataset.shuffle.return_value = mock_shuffled
            mock_shuffled.select.return_value = mock_selected
            mock_load.return_value = mock_dataset
            
            # Test all parameters
            result = load_dataset_with_config(
                dataset_path="test/dataset",
                subset="subset1", 
                split="validation",
                seed=999,
                sample_size=25,
                revision="v1.0"
            )
            
            mock_load.assert_called_once_with(
                "test/dataset", 
                split="validation", 
                name="subset1", 
                revision="v1.0"
            )
            mock_dataset.shuffle.assert_called_once_with(seed=999)
            mock_shuffled.select.assert_called_once_with(range(25))


class TestDataLoadingEdgeCases:
    """Test edge cases and error conditions."""
    
    @patch('utils.data_loading.load_dataset')
    def test_load_dataset_exception_handling(self, mock_load_dataset):
        """Test that exceptions during HF dataset loading are handled."""
        mock_load_dataset.side_effect = ValueError("Invalid dataset path")
        
        # For regular dataset loading, exception should propagate
        with pytest.raises(ValueError):
            load_dataset_with_config("invalid_dataset")
            
    @patch('utils.data_loading.load_dataset')  
    def test_json_dataset_exception_propagates(self, mock_load_dataset):
        """Test that JSON dataset loading exceptions are properly propagated."""
        mock_load_dataset.side_effect = FileNotFoundError("File not found")
        
        # Should raise exception instead of returning None
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_json_dataset_with_config("nonexistent.jsonl")
        
    def test_sample_size_edge_values(self):
        """Test sample_size edge values."""
        with patch('utils.data_loading.load_dataset') as mock_load:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=100)
            mock_load.return_value = mock_dataset
            
            # Test sample_size = 1
            result = load_dataset_with_config("test", sample_size=1, seed=42)
            mock_dataset.shuffle.assert_called_with(seed=42)
            
            # Test sample_size equals dataset size
            mock_dataset.reset_mock()
            result = load_dataset_with_config("test", sample_size=100)
            # Should still perform sampling since sample_size > 0
            assert mock_dataset.shuffle.called
            
    @pytest.mark.parametrize("split", ["train", "test", "validation", "dev"])
    def test_different_splits(self, split):
        """Test loading different dataset splits."""
        with patch('utils.data_loading.load_dataset') as mock_load:
            mock_dataset = Mock()
            mock_load.return_value = mock_dataset
            
            load_dataset_with_config("test_dataset", split=split)
            
            mock_load.assert_called_once_with("test_dataset", split=split)
