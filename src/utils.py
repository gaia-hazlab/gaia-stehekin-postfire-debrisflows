"""
Utility functions for visualization and helper operations.

This module provides tools for plotting seismic data, model outputs,
and other visualizations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


def get_device(device='auto'):
    """
    Get the appropriate PyTorch device.
    
    Args:
        device (str): 'auto', 'cpu', 'cuda', or 'mps'
    
    Returns:
        torch.device
    """
    if device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device)


def plot_seismogram(stream, save_path=None, figsize=(12, 8)):
    """
    Plot seismic waveforms.
    
    Args:
        stream: ObsPy Stream object
        save_path (str): Optional path to save the figure
        figsize (tuple): Figure size
    
    Returns:
        matplotlib figure and axes
    """
    n_traces = len(stream)
    fig, axes = plt.subplots(n_traces, 1, figsize=figsize, sharex=True)
    
    if n_traces == 1:
        axes = [axes]
    
    for i, trace in enumerate(stream):
        times = trace.times()
        axes[i].plot(times, trace.data, 'k', linewidth=0.5)
        axes[i].set_ylabel(f'{trace.stats.channel}\nAmplitude')
        axes[i].grid(True, alpha=0.3)
        
        # Add trace info
        info = f'{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}'
        axes[i].text(0.02, 0.95, info, transform=axes[i].transAxes, 
                    verticalalignment='top', fontsize=9)
    
    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title(f'Seismogram - {stream[0].stats.starttime}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes


def plot_predictions(data, predictions, labels=None, save_path=None, figsize=(12, 6)):
    """
    Plot model predictions alongside data.
    
    Args:
        data (np.ndarray): Input data of shape (n_channels, n_samples)
        predictions (np.ndarray): Model predictions
        labels (list): Optional class labels
        save_path (str): Optional path to save the figure
        figsize (tuple): Figure size
    
    Returns:
        matplotlib figure and axes
    """
    n_channels = data.shape[0]
    fig, axes = plt.subplots(n_channels + 1, 1, figsize=figsize, sharex=True)
    
    # Plot waveforms
    for i in range(n_channels):
        axes[i].plot(data[i], 'k', linewidth=0.5)
        axes[i].set_ylabel(f'Channel {i}')
        axes[i].grid(True, alpha=0.3)
    
    # Plot predictions
    if len(predictions.shape) == 1:
        axes[-1].plot(predictions, linewidth=2)
    else:
        for i, pred in enumerate(predictions.T):
            label = labels[i] if labels else f'Class {i}'
            axes[-1].plot(pred, linewidth=2, label=label)
        axes[-1].legend()
    
    axes[-1].set_ylabel('Probability')
    axes[-1].set_xlabel('Sample')
    axes[-1].grid(True, alpha=0.3)
    axes[-1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes


def set_random_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
