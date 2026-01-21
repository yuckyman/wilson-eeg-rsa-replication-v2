"""Representation Method Extraction for RSA Analysis.

This module provides functions to extract neural patterns using different
encoding schemes for testing robustness of imagery vs perception findings.

Representation Methods:
- power_bands: Average power in frequency bands across all channels
- channels: Mean activity per channel (spatial patterns)
- channel_x_band: Power per channel per band (full spatial-frequency matrix)
- time_windows: Representations from different temporal windows
- erp_features: Peak amplitude, mean amplitude, area under curve
- time_frequency: Time-frequency decomposition features

Author: Ian
Date: 2025-01-XX
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import mne
from mne.time_frequency import tfr_morlet

from feature_extraction.extract_features import (
    ERPFeatureExtractor,
    TimeFrequencyExtractor
)


def extract_power_bands(epochs: mne.Epochs,
                       bands: Optional[Dict[str, Tuple[float, float]]] = None) -> np.ndarray:
    """Extract average power in frequency bands across all channels.
    
    This method collapses spatial information and focuses on frequency content.
    Returns a vector of band powers averaged across all channels.
    
    Args:
        epochs: MNE epochs object.
        bands: Dictionary mapping band names to (fmin, fmax) tuples.
               If None, uses standard bands.
    
    Returns:
        Array of shape (n_epochs, n_bands) with average power per band.
    """
    if bands is None:
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40),
        }
    
    tf_extractor = TimeFrequencyExtractor(epochs)
    band_powers = tf_extractor.extract_band_power(bands)
    
    # Average across channels for each band
    representations = []
    for band_name in sorted(bands.keys()):
        # Shape: (n_epochs, n_channels) -> (n_epochs,)
        band_avg = np.mean(band_powers[band_name], axis=1)
        representations.append(band_avg)
    
    # Stack to get (n_epochs, n_bands)
    return np.column_stack(representations)


def extract_channels(epochs: mne.Epochs,
                    time_window: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Extract mean activity per channel (spatial patterns).
    
    This method collapses temporal and frequency information, focusing
    on spatial distribution of activity.
    
    Args:
        epochs: MNE epochs object.
        time_window: (tmin, tmax) time window in seconds. If None, uses all time.
    
    Returns:
        Array of shape (n_epochs, n_channels) with mean activity per channel.
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    
    if time_window is not None:
        tmin_idx = np.argmin(np.abs(epochs.times - time_window[0]))
        tmax_idx = np.argmin(np.abs(epochs.times - time_window[1]))
        data = data[:, :, tmin_idx:tmax_idx]
    
    # Average across time: (n_epochs, n_channels, n_times) -> (n_epochs, n_channels)
    return np.mean(data, axis=2)


def extract_channel_x_band(epochs: mne.Epochs,
                          bands: Optional[Dict[str, Tuple[float, float]]] = None) -> np.ndarray:
    """Extract power per channel per band (full spatial-frequency matrix).
    
    This method preserves both spatial and frequency information,
    creating a high-dimensional representation.
    
    Args:
        epochs: MNE epochs object.
        bands: Dictionary mapping band names to (fmin, fmax) tuples.
               If None, uses standard bands.
    
    Returns:
        Array of shape (n_epochs, n_channels * n_bands) with power per channel-band.
    """
    if bands is None:
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40),
        }
    
    tf_extractor = TimeFrequencyExtractor(epochs)
    band_powers = tf_extractor.extract_band_power(bands)
    
    # Stack channel-band combinations
    # Shape: (n_epochs, n_channels) for each band -> (n_epochs, n_channels * n_bands)
    representations = []
    for band_name in sorted(bands.keys()):
        representations.append(band_powers[band_name])  # (n_epochs, n_channels)
    
    # Concatenate along channel dimension: (n_epochs, n_channels * n_bands)
    return np.concatenate(representations, axis=1)


def extract_time_windows(epochs: mne.Epochs,
                        windows: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
    """Extract representations from different temporal windows.
    
    This method captures temporal dynamics by computing separate
    representations for early vs late time windows.
    
    Args:
        epochs: MNE epochs object.
        windows: List of (tmin, tmax) tuples for each window.
                If None, uses early (0-0.3s) and late (0.3-0.6s) windows.
    
    Returns:
        Array of shape (n_epochs, n_channels * n_windows) with activity per window.
    """
    if windows is None:
        # Default: early and late windows
        windows = [(0.0, 0.3), (0.3, 0.6)]
    
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, _ = data.shape
    
    window_reprs = []
    for tmin, tmax in windows:
        tmin_idx = np.argmin(np.abs(epochs.times - tmin))
        tmax_idx = np.argmin(np.abs(epochs.times - tmax))
        
        # Average across time in this window: (n_epochs, n_channels)
        window_data = np.mean(data[:, :, tmin_idx:tmax_idx], axis=2)
        window_reprs.append(window_data)
    
    # Concatenate windows: (n_epochs, n_channels * n_windows)
    return np.concatenate(window_reprs, axis=1)


def extract_erp_features(epochs: mne.Epochs,
                        time_window: Optional[Tuple[float, float]] = None,
                        channels: Optional[List[str]] = None) -> np.ndarray:
    """Extract ERP features: peak amplitude, mean amplitude, area under curve.
    
    This method extracts classical ERP features that capture different
    aspects of the event-related response.
    
    Args:
        epochs: MNE epochs object.
        time_window: (tmin, tmax) time window in seconds. If None, uses all time.
        channels: List of channel names. If None, uses all channels.
    
    Returns:
        Array of shape (n_epochs, n_channels * 3) with [peak, mean, auc] per channel.
    """
    if time_window is None:
        time_window = (epochs.times[0], epochs.times[-1])
    
    erp_extractor = ERPFeatureExtractor(epochs)
    
    # Extract three types of features
    peak_amp = erp_extractor.extract_peak_amplitude(
        time_window, channels=channels, mode='abs'
    )  # (n_epochs, n_channels)
    
    mean_amp = erp_extractor.extract_mean_amplitude(
        time_window, channels=channels
    )  # (n_epochs, n_channels)
    
    auc = erp_extractor.extract_area_under_curve(
        time_window, channels=channels
    )  # (n_epochs, n_channels)
    
    # Concatenate features: (n_epochs, n_channels * 3)
    return np.concatenate([peak_amp, mean_amp, auc], axis=1)


def extract_time_frequency(epochs: mne.Epochs,
                          freqs: Optional[np.ndarray] = None,
                          time_window: Optional[Tuple[float, float]] = None,
                          average_time: bool = True) -> np.ndarray:
    """Extract time-frequency decomposition features.
    
    This method captures both temporal and frequency dynamics using
    wavelet decomposition.
    
    Args:
        epochs: MNE epochs object.
        freqs: Frequencies of interest. If None, uses 2-40 Hz in 2 Hz steps.
        time_window: (tmin, tmax) time window in seconds. If None, uses all time.
        average_time: If True, average across time. If False, keep time dimension.
    
    Returns:
        If average_time=True: Array of shape (n_epochs, n_channels * n_freqs)
        If average_time=False: Array of shape (n_epochs, n_channels * n_freqs * n_times)
    """
    if freqs is None:
        freqs = np.arange(2, 40, 2)  # 2-40 Hz in 2 Hz steps
    
    tf_extractor = TimeFrequencyExtractor(epochs)
    
    # Extract time-frequency representation
    # Shape: (n_epochs, n_channels, n_freqs, n_times)
    tf_power = tf_extractor.extract_time_frequency(freqs, n_cycles=7.0, method='morlet')
    
    # Optionally crop time window
    if time_window is not None:
        tmin_idx = np.argmin(np.abs(epochs.times - time_window[0]))
        tmax_idx = np.argmin(np.abs(epochs.times - time_window[1]))
        tf_power = tf_power[:, :, :, tmin_idx:tmax_idx]
    
    if average_time:
        # Average across time: (n_epochs, n_channels, n_freqs)
        tf_power = np.mean(tf_power, axis=3)
        # Reshape: (n_epochs, n_channels * n_freqs)
        n_epochs, n_channels, n_freqs = tf_power.shape
        return tf_power.reshape(n_epochs, n_channels * n_freqs)
    else:
        # Keep time dimension: (n_epochs, n_channels * n_freqs * n_times)
        n_epochs, n_channels, n_freqs, n_times = tf_power.shape
        return tf_power.reshape(n_epochs, n_channels * n_freqs * n_times)


def extract_representation_vector(epochs: mne.Epochs,
                                 method: str,
                                 **kwargs) -> np.ndarray:
    """Extract representation vector using specified method.
    
    Main function for extracting representations with different encoding schemes.
    
    Args:
        epochs: MNE epochs object.
        method: Representation method name. Options:
            - 'power_bands': Average power in frequency bands
            - 'channels': Mean activity per channel
            - 'channel_x_band': Power per channel per band
            - 'time_windows': Representations from different temporal windows
            - 'erp_features': Peak, mean, AUC features
            - 'time_frequency': Time-frequency decomposition
        **kwargs: Additional arguments passed to specific extraction function.
    
    Returns:
        Array of shape (n_epochs, n_features) with representation vectors.
    
    Example:
        >>> # Extract power bands representation
        >>> reprs = extract_representation_vector(epochs, 'power_bands')
        >>> 
        >>> # Extract channel representations with time window
        >>> reprs = extract_representation_vector(
        ...     epochs, 'channels', time_window=(0.0, 0.5)
        ... )
    """
    method_map = {
        'power_bands': extract_power_bands,
        'channels': extract_channels,
        'channel_x_band': extract_channel_x_band,
        'time_windows': extract_time_windows,
        'erp_features': extract_erp_features,
        'time_frequency': extract_time_frequency,
    }
    
    if method not in method_map:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Available methods: {list(method_map.keys())}"
        )
    
    extractor_func = method_map[method]
    return extractor_func(epochs, **kwargs)


def get_representation_methods() -> List[str]:
    """Get list of available representation methods.
    
    Returns:
        List of method names.
    """
    return [
        'power_bands',
        'channels',
        'channel_x_band',
        'time_windows',
        'erp_features',
        'time_frequency',
    ]


if __name__ == "__main__":
    # Example usage
    print("Representation Extraction Module")
    print("\nAvailable methods:")
    for method in get_representation_methods():
        print(f"  - {method}")
    
    print("\nUsage example:")
    print("""
    from feature_extraction.extract_representations import extract_representation_vector
    import mne
    
    # Load epochs
    epochs = mne.read_epochs('data/preprocessed/sub01-epo.fif')
    
    # Extract different representations
    power_bands = extract_representation_vector(epochs, 'power_bands')
    channels = extract_representation_vector(epochs, 'channels')
    channel_x_band = extract_representation_vector(epochs, 'channel_x_band')
    
    print(f"Power bands shape: {power_bands.shape}")
    print(f"Channels shape: {channels.shape}")
    print(f"Channel x Band shape: {channel_x_band.shape}")
    """)


