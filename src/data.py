"""
Data processing utilities for seismic waveforms.

This module provides tools for loading, preprocessing, and transforming
seismic data following ObsPy conventions.
"""

import numpy as np
import torch
from obspy import read, Stream, UTCDateTime
from obspy.signal.filter import bandpass


class SeismicDataProcessor:
    """
    Processor for seismic data following ObsPy conventions.
    
    Args:
        sampling_rate (float): Target sampling rate in Hz
        window_length (float): Window length in seconds
        normalize (bool): Whether to normalize data
    """
    
    def __init__(self, sampling_rate=100.0, window_length=30.0, normalize=True):
        self.sampling_rate = sampling_rate
        self.window_length = window_length
        self.normalize = normalize
    
    def load_seismic_data(self, filepath):
        """
        Load seismic data using ObsPy.
        
        Args:
            filepath (str): Path to seismic data file
        
        Returns:
            ObsPy Stream object
        """
        stream = read(filepath)
        return stream
    
    def preprocess_stream(self, stream, freqmin=1.0, freqmax=20.0):
        """
        Preprocess seismic stream following standard workflow.
        
        Args:
            stream (Stream): ObsPy Stream object
            freqmin (float): Minimum frequency for bandpass filter
            freqmax (float): Maximum frequency for bandpass filter
        
        Returns:
            Preprocessed ObsPy Stream object
        """
        stream = stream.copy()
        
        # Detrend
        stream.detrend('linear')
        stream.detrend('demean')
        
        # Taper
        stream.taper(max_percentage=0.05, type='cosine')
        
        # Filter
        stream.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)
        
        # Resample
        stream.resample(self.sampling_rate)
        
        # Sort components (Z, N, E order)
        stream.sort(keys=['channel'], reverse=True)
        
        # Merge to handle gaps
        stream.merge(fill_value=0)
        
        return stream
    
    def stream_to_array(self, stream):
        """
        Convert ObsPy Stream to numpy array.
        
        Args:
            stream (Stream): ObsPy Stream object
        
        Returns:
            numpy array of shape (n_channels, n_samples)
        """
        data_list = []
        for trace in stream:
            data_list.append(trace.data)
        
        data = np.array(data_list)
        
        if self.normalize:
            # Normalize each channel independently
            for i in range(data.shape[0]):
                std = np.std(data[i])
                if std > 0:
                    data[i] = data[i] / std
        
        return data
    
    def to_torch(self, data):
        """
        Convert numpy array to PyTorch tensor.
        
        Args:
            data (np.ndarray): Numpy array
        
        Returns:
            torch.Tensor
        """
        return torch.from_numpy(data).float()
    
    def create_windows(self, stream, window_length=None, overlap=0.5):
        """
        Create sliding windows from continuous stream.
        
        Args:
            stream (Stream): ObsPy Stream object
            window_length (float): Window length in seconds (default uses self.window_length)
            overlap (float): Overlap fraction (0 to 1)
        
        Returns:
            List of numpy arrays, each representing a window
        """
        if window_length is None:
            window_length = self.window_length
        
        windows = []
        window_samples = int(window_length * self.sampling_rate)
        step_samples = int(window_samples * (1 - overlap))
        
        data = self.stream_to_array(stream)
        n_samples = data.shape[1]
        
        for start in range(0, n_samples - window_samples + 1, step_samples):
            end = start + window_samples
            window = data[:, start:end]
            windows.append(window)
        
        return windows
