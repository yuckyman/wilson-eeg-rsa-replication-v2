"""Feature Extraction for EEG Data.

This module provides functions for extracting features from preprocessed EEG data,
including time-domain, frequency-domain, and deep learning features.

Feature Types:
- ERP features: Amplitude, latency, area under curve
- Time-frequency: Wavelet decomposition, spectral power
- Connectivity: Phase synchrony, coherence
- Deep learning: Pretrained model embeddings (Braindecode)

Customization Points:
- Select relevant frequency bands for your analysis
- Configure time windows for feature extraction
- Choose appropriate deep learning architectures
- Implement custom feature extraction methods

Integration:
- MNE-Python: Time-frequency analysis, connectivity
- Braindecode: Deep learning feature extraction
- SciPy: Signal processing utilities
- Scikit-learn: Feature scaling and selection

Author: Ian
Date: 2025-10-30
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import mne
from mne.time_frequency import tfr_morlet, psd_array_welch
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from braindecode.models import ShallowFBCSPNet, Deep4Net, EEGNetv4
    BRAINDECODE_AVAILABLE = True
except ImportError:
    BRAINDECODE_AVAILABLE = False
    warnings.warn("Braindecode not available. Install with: pip install braindecode")


class ERPFeatureExtractor:
    """Extract ERP (Event-Related Potential) features.
    
    This class provides methods for extracting classical ERP features
    such as peak amplitudes, latencies, and areas under curve.
    """
    
    def __init__(self, epochs: mne.Epochs):
        """Initialize with epoched data.
        
        Args:
            epochs: Preprocessed epochs object.
        """
        self.epochs = epochs
        self.data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
        self.times = epochs.times
        self.ch_names = epochs.ch_names
        
    def extract_peak_amplitude(self,
                              time_window: Tuple[float, float],
                              channels: Optional[List[str]] = None,
                              mode: str = 'max') -> np.ndarray:
        """Extract peak amplitude in specified time window.
        
        Args:
            time_window: (tmin, tmax) in seconds.
            channels: List of channel names. If None, uses all channels.
            mode: 'max' for positive peak, 'min' for negative, 'abs' for absolute.
        
        Returns:
            Array of shape (n_epochs, n_channels) with peak amplitudes.
            
        CUSTOMIZATION:
            - Use different peak detection methods (e.g., local maxima)
            - Apply smoothing before peak detection
            - Extract multiple peaks within window
        """
        # Get time indices
        tmin_idx = np.argmin(np.abs(self.times - time_window[0]))
        tmax_idx = np.argmin(np.abs(self.times - time_window[1]))
        
        # Select channels
        if channels:
            ch_idx = [self.epochs.ch_names.index(ch) for ch in channels]
            data = self.data[:, ch_idx, tmin_idx:tmax_idx]
        else:
            data = self.data[:, :, tmin_idx:tmax_idx]
        
        # Extract peaks
        if mode == 'max':
            peaks = np.max(data, axis=2)
        elif mode == 'min':
            peaks = np.min(data, axis=2)
        elif mode == 'abs':
            peaks = data[np.arange(len(data))[:, None], 
                        np.arange(data.shape[1]), 
                        np.argmax(np.abs(data), axis=2)]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return peaks
    
    def extract_peak_latency(self,
                            time_window: Tuple[float, float],
                            channels: Optional[List[str]] = None,
                            mode: str = 'max') -> np.ndarray:
        """Extract latency of peak amplitude.
        
        Args:
            time_window: (tmin, tmax) in seconds.
            channels: List of channel names.
            mode: 'max', 'min', or 'abs'.
        
        Returns:
            Array of shape (n_epochs, n_channels) with peak latencies in seconds.
        """
        tmin_idx = np.argmin(np.abs(self.times - time_window[0]))
        tmax_idx = np.argmin(np.abs(self.times - time_window[1]))
        
        if channels:
            ch_idx = [self.epochs.ch_names.index(ch) for ch in channels]
            data = self.data[:, ch_idx, tmin_idx:tmax_idx]
        else:
            data = self.data[:, :, tmin_idx:tmax_idx]
        
        if mode == 'max':
            peak_idx = np.argmax(data, axis=2)
        elif mode == 'min':
            peak_idx = np.argmin(data, axis=2)
        elif mode == 'abs':
            peak_idx = np.argmax(np.abs(data), axis=2)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Convert indices to times
        latencies = self.times[tmin_idx + peak_idx]
        return latencies
    
    def extract_mean_amplitude(self,
                              time_window: Tuple[float, float],
                              channels: Optional[List[str]] = None) -> np.ndarray:
        """Extract mean amplitude in time window.
        
        Args:
            time_window: (tmin, tmax) in seconds.
            channels: List of channel names.
        
        Returns:
            Array of shape (n_epochs, n_channels) with mean amplitudes.
        """
        tmin_idx = np.argmin(np.abs(self.times - time_window[0]))
        tmax_idx = np.argmin(np.abs(self.times - time_window[1]))
        
        if channels:
            ch_idx = [self.epochs.ch_names.index(ch) for ch in channels]
            data = self.data[:, ch_idx, tmin_idx:tmax_idx]
        else:
            data = self.data[:, :, tmin_idx:tmax_idx]
        
        return np.mean(data, axis=2)
    
    def extract_area_under_curve(self,
                                time_window: Tuple[float, float],
                                channels: Optional[List[str]] = None) -> np.ndarray:
        """Extract area under curve using trapezoidal integration.
        
        Args:
            time_window: (tmin, tmax) in seconds.
            channels: List of channel names.
        
        Returns:
            Array of shape (n_epochs, n_channels) with AUC values.
        """
        tmin_idx = np.argmin(np.abs(self.times - time_window[0]))
        tmax_idx = np.argmin(np.abs(self.times - time_window[1]))
        
        if channels:
            ch_idx = [self.epochs.ch_names.index(ch) for ch in channels]
            data = self.data[:, ch_idx, tmin_idx:tmax_idx]
        else:
            data = self.data[:, :, tmin_idx:tmax_idx]
        
        # Integrate using trapezoidal rule
        dt = self.times[1] - self.times[0]
        auc = np.trapz(data, dx=dt, axis=2)
        return auc


class TimeFrequencyExtractor:
    """Extract time-frequency features.
    
    This class provides methods for extracting spectral features using
    wavelet decomposition and other time-frequency analysis techniques.
    
    INTEGRATION:
        - MNE-Python: tfr_morlet, tfr_multitaper
        - SciPy: Welch's method, spectrogram
    """
    
    def __init__(self, epochs: mne.Epochs):
        """Initialize with epoched data.
        
        Args:
            epochs: Preprocessed epochs object.
        """
        self.epochs = epochs
        
    def extract_power_spectrum(self,
                              freqs: Optional[np.ndarray] = None,
                              method: str = 'welch') -> Tuple[np.ndarray, np.ndarray]:
        """Extract power spectral density.
        
        Args:
            freqs: Frequency array. If None, uses default range.
            method: 'welch' or 'multitaper'.
        
        Returns:
            Tuple of (power, freqs) where power has shape 
            (n_epochs, n_channels, n_freqs).
            
        CUSTOMIZATION:
            - Adjust frequency resolution with n_fft
            - Use multitaper for better spectral estimation
            - Apply frequency band averaging
        """
        if freqs is None:
            freqs = np.arange(1, 40, 1)  # 1-40 Hz
        
        if method == 'welch':
            power = self.epochs.compute_psd(
                method='welch',
                fmin=freqs[0],
                fmax=freqs[-1],
                n_fft=int(self.epochs.info['sfreq']),
            ).get_data()
        elif method == 'multitaper':
            power = self.epochs.compute_psd(
                method='multitaper',
                fmin=freqs[0],
                fmax=freqs[-1],
            ).get_data()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return power, freqs
    
    def extract_band_power(self,
                          bands: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, np.ndarray]:
        """Extract power in frequency bands.
        
        Args:
            bands: Dictionary mapping band names to (fmin, fmax) tuples.
                  If None, uses standard bands.
        
        Returns:
            Dictionary mapping band names to power arrays of shape
            (n_epochs, n_channels).
            
        CUSTOMIZATION:
            - Define custom frequency bands for your analysis
            - Use log-transform for better normality
            - Apply baseline normalization
        """
        if bands is None:
            bands = {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 40),
            }
        
        band_powers = {}
        for band_name, (fmin, fmax) in bands.items():
            psd = self.epochs.compute_psd(
                method='welch',
                fmin=fmin,
                fmax=fmax,
            )
            # Average power in band
            band_powers[band_name] = np.mean(psd.get_data(), axis=2)
        
        return band_powers
    
    def extract_time_frequency(self,
                              freqs: np.ndarray,
                              n_cycles: Union[float, np.ndarray] = 7.0,
                              method: str = 'morlet') -> np.ndarray:
        """Extract time-frequency representation.
        
        Args:
            freqs: Frequencies of interest.
            n_cycles: Number of cycles in wavelet. Can be array to vary with freq.
            method: 'morlet' or 'multitaper'.
        
        Returns:
            Time-frequency power array of shape 
            (n_epochs, n_channels, n_freqs, n_times).
            
        CUSTOMIZATION:
            - Increase n_cycles for better frequency resolution
            - Use time-varying n_cycles: n_cycles = freqs / 2
            - Apply baseline correction
        """
        if method == 'morlet':
            power = tfr_morlet(
                self.epochs,
                freqs=freqs,
                n_cycles=n_cycles,
                return_itc=False,
                average=False,
                n_jobs=-1
            )
        else:
            raise NotImplementedError(f"Method {method} not implemented")
        
        return power.data  # Shape: (n_epochs, n_channels, n_freqs, n_times)


class DeepLearningExtractor:
    """Extract features using deep learning models.
    
    This class provides methods for extracting learned representations
    from pretrained or custom deep learning models.
    
    INTEGRATION:
        - Braindecode: Pretrained models for EEG
        - PyTorch: Custom model architectures
        
    CUSTOMIZATION:
        - Fine-tune pretrained models on your data
        - Extract features from intermediate layers
        - Ensemble multiple models
    """
    
    def __init__(self, model_name: str = 'ShallowFBCSPNet'):
        """Initialize with model architecture.
        
        Args:
            model_name: Name of Braindecode model or path to custom model.
        """
        if not BRAINDECODE_AVAILABLE:
            raise ImportError("Braindecode not available. Install with: pip install braindecode")
        
        self.model_name = model_name
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_pretrained_model(self,
                             n_channels: int,
                             n_classes: int,
                             checkpoint_path: Optional[str] = None) -> nn.Module:
        """Load pretrained model.
        
        Args:
            n_channels: Number of EEG channels.
            n_classes: Number of output classes.
            checkpoint_path: Path to model checkpoint. If None, uses random init.
        
        Returns:
            Loaded model.
            
        CUSTOMIZATION:
            - Use different model architectures:
              * ShallowFBCSPNet: Filter-bank CSP
              * Deep4Net: Deep ConvNet
              * EEGNetv4: Compact architecture
              * EEGInception: Multi-scale features
        """
        # Initialize model architecture
        if self.model_name == 'ShallowFBCSPNet':
            self.model = ShallowFBCSPNet(
                n_chans=n_channels,
                n_outputs=n_classes,
                n_times=1000,  # Adjust based on your epoch length
            )
        elif self.model_name == 'Deep4Net':
            self.model = Deep4Net(
                n_chans=n_channels,
                n_outputs=n_classes,
                n_times=1000,
            )
        elif self.model_name == 'EEGNetv4':
            self.model = EEGNetv4(
                n_chans=n_channels,
                n_outputs=n_classes,
                n_times=1000,
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Load checkpoint if provided
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        
        self.model.to(self.device)
        self.model.eval()
        return self.model
    
    def extract_embeddings(self,
                          epochs: mne.Epochs,
                          layer_name: Optional[str] = None) -> np.ndarray:
        """Extract feature embeddings from model.
        
        Args:
            epochs: Preprocessed epochs object.
            layer_name: Name of layer to extract from. If None, uses final layer.
        
        Returns:
            Feature array of shape (n_epochs, n_features).
            
        PLACEHOLDER: Implement extraction from intermediate layers
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_pretrained_model() first.")
        
        # Convert epochs to tensor
        data = epochs.get_data()
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        # Extract features
        embeddings = []
        with torch.no_grad():
            for batch in data_tensor:
                output = self.model(batch.unsqueeze(0))
                embeddings.append(output.cpu().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings


def extract_all_features(epochs: mne.Epochs,
                        feature_types: Optional[List[str]] = None,
                        config: Optional[Dict] = None) -> Dict[str, np.ndarray]:
    """Extract all specified feature types.
    
    This is a convenience function that extracts multiple feature types
    and returns them in a dictionary.
    
    Args:
        epochs: Preprocessed epochs object.
        feature_types: List of feature types to extract.
                      Options: 'erp', 'band_power', 'time_frequency', 'deep_learning'
        config: Configuration dictionary for feature extraction.
    
    Returns:
        Dictionary mapping feature names to arrays.
        
    Example:
        >>> features = extract_all_features(
        ...     epochs,
        ...     feature_types=['erp', 'band_power'],
        ...     config={'erp_window': (0.3, 0.5), 'erp_channels': ['Pz']}
        ... )
    """
    if feature_types is None:
        feature_types = ['erp', 'band_power']
    
    if config is None:
        config = {}
    
    features = {}
    
    # Extract ERP features
    if 'erp' in feature_types:
        print("Extracting ERP features...")
        erp_extractor = ERPFeatureExtractor(epochs)
        
        erp_window = config.get('erp_window', (0.3, 0.5))
        erp_channels = config.get('erp_channels', None)
        
        features['erp_peak_amplitude'] = erp_extractor.extract_peak_amplitude(
            erp_window, erp_channels
        )
        features['erp_mean_amplitude'] = erp_extractor.extract_mean_amplitude(
            erp_window, erp_channels
        )
    
    # Extract band power features
    if 'band_power' in feature_types:
        print("Extracting band power features...")
        tf_extractor = TimeFrequencyExtractor(epochs)
        
        bands = config.get('frequency_bands', None)
        band_powers = tf_extractor.extract_band_power(bands)
        
        for band_name, power in band_powers.items():
            features[f'power_{band_name}'] = power
    
    # Extract time-frequency features
    if 'time_frequency' in feature_types:
        print("Extracting time-frequency features...")
        tf_extractor = TimeFrequencyExtractor(epochs)
        
        freqs = config.get('tf_freqs', np.arange(8, 30, 2))
        tf_power = tf_extractor.extract_time_frequency(freqs)
        features['time_frequency'] = tf_power
    
    # Extract deep learning features
    if 'deep_learning' in feature_types and BRAINDECODE_AVAILABLE:
        print("Extracting deep learning features...")
        # PLACEHOLDER: Implement when model is available
        print("  Note: Requires pretrained model checkpoint")
    
    print(f"Extracted {len(features)} feature types")
    return features


def normalize_features(features: np.ndarray,
                      method: str = 'zscore') -> Tuple[np.ndarray, StandardScaler]:
    """Normalize features for analysis.
    
    Args:
        features: Feature array to normalize.
        method: Normalization method ('zscore', 'minmax', 'robust').
    
    Returns:
        Tuple of (normalized_features, scaler).
        
    CUSTOMIZATION:
        - Apply per-channel normalization
        - Use robust scaling for outliers
        - Apply log-transform before normalization
    """
    if method == 'zscore':
        scaler = StandardScaler()
        normalized = scaler.fit_transform(features)
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(features)
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        normalized = scaler.fit_transform(features)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return normalized, scaler


if __name__ == "__main__":
    # Example usage
    print("Feature Extraction Template")
    print("\nThis script provides feature extraction functions.")
    print("\nUsage example:")
    print("""
    from feature_extraction.extract_features import extract_all_features
    from preprocessing.eeg_preprocessing import load_preprocessed_epochs
    
    # Load preprocessed data
    epochs = load_preprocessed_epochs('data/preprocessed/sub01-epo.fif')
    
    # Configure feature extraction
    config = {
        'erp_window': (0.3, 0.5),
        'erp_channels': ['Pz', 'Cz'],
        'frequency_bands': {
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
        },
    }
    
    # Extract features
    features = extract_all_features(
        epochs,
        feature_types=['erp', 'band_power'],
        config=config
    )
    """)
