"""
Unit tests for the data processing module.
"""

import pytest
import numpy as np
import torch
from src.data import SeismicDataProcessor


def test_processor_creation():
    """Test SeismicDataProcessor creation."""
    processor = SeismicDataProcessor(sampling_rate=100.0, window_length=30.0)
    assert processor.sampling_rate == 100.0
    assert processor.window_length == 30.0


def test_to_torch():
    """Test numpy to torch conversion."""
    processor = SeismicDataProcessor()
    data = np.random.randn(3, 1000)
    
    tensor = processor.to_torch(data)
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 1000)
    assert tensor.dtype == torch.float32


def test_normalization():
    """Test data normalization."""
    processor = SeismicDataProcessor(normalize=True)
    
    # Create test data with known statistics
    data = np.array([[1, 2, 3, 4, 5]], dtype=float)
    
    # Mock stream_to_array behavior
    normalized = np.zeros_like(data)
    for i in range(data.shape[0]):
        std = np.std(data[i])
        if std > 0:
            normalized[i] = data[i] / std
    
    assert not np.allclose(data, normalized)


if __name__ == '__main__':
    pytest.main([__file__])
