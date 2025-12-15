"""
PyTorch models for seismic data analysis.

This module contains neural network architectures for processing
seismic waveforms and detecting debris flows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeismicCNN(nn.Module):
    """
    Convolutional Neural Network for seismic waveform classification.
    
    Args:
        input_channels (int): Number of input channels (typically 3 for Z, N, E)
        num_classes (int): Number of output classes
        dropout (float): Dropout rate for regularization
    """
    
    def __init__(self, input_channels=3, num_classes=2, dropout=0.5):
        super(SeismicCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(dropout)
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


def save_model(model, path, optimizer=None, epoch=None, loss=None):
    """
    Save a PyTorch model checkpoint.
    
    Args:
        model: PyTorch model to save
        path: File path to save the checkpoint
        optimizer: Optional optimizer state
        epoch: Optional epoch number
        loss: Optional loss value
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(model, path, optimizer=None, device='cpu'):
    """
    Load a PyTorch model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        path: File path to load the checkpoint from
        optimizer: Optional optimizer to load state into
        device: Device to load the model to
    
    Returns:
        Dictionary with epoch and loss if available
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    info = {}
    if 'epoch' in checkpoint:
        info['epoch'] = checkpoint['epoch']
    if 'loss' in checkpoint:
        info['loss'] = checkpoint['loss']
    
    print(f"Model loaded from {path}")
    return info
