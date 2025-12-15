"""
Unit tests for the models module.
"""

import pytest
import torch
from src.models import SeismicCNN, save_model, load_model
import tempfile
import os


def test_seismic_cnn_creation():
    """Test SeismicCNN model creation."""
    model = SeismicCNN(input_channels=3, num_classes=2)
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_seismic_cnn_forward():
    """Test forward pass of SeismicCNN."""
    model = SeismicCNN(input_channels=3, num_classes=2)
    batch_size = 4
    n_samples = 3000
    
    # Create dummy input
    x = torch.randn(batch_size, 3, n_samples)
    
    # Forward pass
    output = model(x)
    
    assert output.shape == (batch_size, 2)


def test_model_save_load():
    """Test saving and loading model."""
    model = SeismicCNN(input_channels=3, num_classes=2)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
        tmp_path = tmp.name
    
    try:
        # Save model
        save_model(model, tmp_path, epoch=10, loss=0.5)
        
        # Load model
        new_model = SeismicCNN(input_channels=3, num_classes=2)
        info = load_model(new_model, tmp_path)
        
        assert info['epoch'] == 10
        assert info['loss'] == 0.5
        
        # Verify weights match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)
    
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == '__main__':
    pytest.main([__file__])
