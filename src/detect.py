"""
Event detection utilities for seismic data.

This module provides tools for detecting and analyzing seismic events
using various methods including SeisBench models.
"""

import numpy as np
from scipy.signal import find_peaks


def multi_class_detection(probabilities, threshold=0.5, min_distance=100):
    """
    Detect events from multi-class probability predictions.
    
    Args:
        probabilities (np.ndarray): Array of probabilities (n_samples, n_classes)
        threshold (float): Detection threshold
        min_distance (int): Minimum distance between peaks in samples
    
    Returns:
        List of detected events with timing and class information
    """
    events = []
    
    n_classes = probabilities.shape[1] if len(probabilities.shape) > 1 else 1
    
    for class_idx in range(n_classes):
        if len(probabilities.shape) > 1:
            probs = probabilities[:, class_idx]
        else:
            probs = probabilities
        
        # Find peaks above threshold
        peaks, properties = find_peaks(probs, height=threshold, distance=min_distance)
        
        for peak in peaks:
            events.append({
                'time_index': peak,
                'class': class_idx,
                'probability': probs[peak],
                'max_probability': probs[peak]
            })
    
    # Sort by time
    events = sorted(events, key=lambda x: x['time_index'])
    
    return events


def calculate_event_metrics(probabilities, event):
    """
    Calculate metrics for a detected event.
    
    Args:
        probabilities (np.ndarray): Array of probabilities
        event (dict): Event dictionary
    
    Returns:
        Updated event dictionary with metrics
    """
    # Calculate area under curve around the event
    window = 50  # samples around peak
    start = max(0, event['time_index'] - window)
    end = min(len(probabilities), event['time_index'] + window)
    
    if len(probabilities.shape) > 1:
        probs = probabilities[start:end, event['class']]
    else:
        probs = probabilities[start:end]
    
    event['auc'] = np.trapz(probs)
    event['mean_probability'] = np.mean(probs)
    event['duration'] = len(probs[probs > 0.3])  # Duration above 30% threshold
    
    return event
