"""EEG Preprocessing Pipeline.

This module provides a comprehensive preprocessing pipeline for EEG data,
integrating ERPLAB and MNE-Python functionality for robust signal processing.

The pipeline includes:
- Data loading from various formats (EDF, BDF, EEGLAB .set, etc.)
- Filtering (high-pass, low-pass, notch)
- Artifact rejection (ICA, threshold-based)
- Epoching around stimulus events
- Baseline correction
- Bad channel detection and interpolation
- Re-referencing

Customization Points:
- Adjust filtering parameters based on your signal characteristics
- Modify artifact rejection thresholds for your data quality
- Configure epoching windows for your experimental design
- Select appropriate reference scheme for your montage

Integration:
- ERPLAB: Event binning, artifact detection, ERP computation
- MNE-Python: Core data structures, filtering, ICA, epoching
- AutoReject: Automated artifact rejection (optional)

Author: Ian
Date: 2025-10-30
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import mne
from mne.preprocessing import ICA
try:
    from autoreject import AutoReject, get_rejection_threshold
    AUTOREJECT_AVAILABLE = True
except ImportError:
    AUTOREJECT_AVAILABLE = False
    warnings.warn("AutoReject not available. Install with: pip install autoreject")


class EEGPreprocessor:
    """Complete EEG preprocessing pipeline.
    
    This class encapsulates all preprocessing steps from raw data loading
    to cleaned, epoched data ready for feature extraction.
    
    Attributes:
        raw (mne.io.Raw): Raw EEG data object.
        epochs (mne.Epochs): Epoched EEG data object.
        ica (mne.preprocessing.ICA): ICA decomposition object.
        config (dict): Configuration parameters for preprocessing.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the preprocessor with configuration.
        
        Args:
            config: Dictionary containing preprocessing parameters.
                   If None, uses default configuration.
        """
        self.raw = None
        self.epochs = None
        self.ica = None
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict:
        """Return default preprocessing configuration.
        
        Customize these parameters based on your specific requirements.
        
        Returns:
            Dictionary with default preprocessing parameters.
        """
        return {
            # Filtering parameters
            'highpass_freq': 0.1,  # Hz, high-pass filter cutoff
            'lowpass_freq': 40.0,  # Hz, low-pass filter cutoff
            'notch_freq': 60.0,    # Hz, line noise frequency (50 or 60)
            'notch_width': 2.0,    # Hz, notch filter width
            
            # Epoching parameters
            'tmin': -0.2,          # s, epoch start time relative to event
            'tmax': 0.8,           # s, epoch end time relative to event
            'baseline': (-0.2, 0), # s, baseline correction window
            
            # Artifact rejection parameters
            'reject_peak_to_peak': 150e-6,  # V, max peak-to-peak amplitude
            'use_autoreject': False,         # Use AutoReject for automatic rejection
            'n_ica_components': 20,          # Number of ICA components
            
            # Channel parameters
            'reference': 'average',  # 'average', 'mastoids', or list of channels
            'eog_channels': ['VEOG', 'HEOG'],  # EOG channel names
            'bad_channels': [],      # Manually marked bad channels
            
            # Event parameters
            'event_id': None,        # Dict mapping event names to codes
            'min_event_duration': 0, # Minimum event duration in seconds
        }
    
    def load_data(self, 
                  filepath: Union[str, Path],
                  file_format: str = 'auto') -> mne.io.Raw:
        """Load raw EEG data from file.
        
        Supports multiple file formats through MNE-Python's I/O functions.
        
        Args:
            filepath: Path to the EEG data file.
            file_format: File format ('auto', 'edf', 'bdf', 'set', 'fif', etc.).
                        'auto' attempts to detect format from extension.
        
        Returns:
            Loaded raw EEG data object.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is unsupported.
            
        Example:
            >>> preprocessor = EEGPreprocessor()
            >>> raw = preprocessor.load_data('data/subject_01.edf')
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Auto-detect format from extension
        if file_format == 'auto':
            ext = filepath.suffix.lower()
            format_map = {
                '.edf': 'edf',
                '.bdf': 'bdf',
                '.set': 'eeglab',
                '.fif': 'fif',
                '.vhdr': 'brainvision',
            }
            file_format = format_map.get(ext, 'edf')
        
        # Load data based on format
        # CUSTOMIZATION: Add more formats as needed
        if file_format == 'edf':
            self.raw = mne.io.read_raw_edf(filepath, preload=True)
        elif file_format == 'bdf':
            self.raw = mne.io.read_raw_bdf(filepath, preload=True)
        elif file_format == 'eeglab':
            self.raw = mne.io.read_raw_eeglab(filepath, preload=True)
        elif file_format == 'fif':
            self.raw = mne.io.read_raw_fif(filepath, preload=True)
        elif file_format == 'brainvision':
            self.raw = mne.io.read_raw_brainvision(filepath, preload=True)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        print(f"Loaded data: {self.raw.info['nchan']} channels, "
              f"{self.raw.n_times} samples, "
              f"{self.raw.info['sfreq']} Hz sampling rate")
        
        return self.raw
    
    def set_montage(self, montage_name: str = 'standard_1020') -> None:
        """Set electrode montage for channel locations.
        
        Args:
            montage_name: Name of standard montage or path to custom montage.
                         Common options: 'standard_1020', 'standard_1005',
                         'biosemi64', 'biosemi128', etc.
        
        Example:
            >>> preprocessor.set_montage('biosemi64')
        """
        if self.raw is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        
        # CUSTOMIZATION: Load custom montage file if needed
        montage = mne.channels.make_standard_montage(montage_name)
        self.raw.set_montage(montage, on_missing='warn')
        print(f"Set montage: {montage_name}")
    
    def filter_data(self,
                    highpass: Optional[float] = None,
                    lowpass: Optional[float] = None,
                    notch: Optional[float] = None) -> mne.io.Raw:
        """Apply temporal filters to the data.
        
        Applies high-pass, low-pass, and notch filters to remove drift,
        high-frequency noise, and line noise.
        
        Args:
            highpass: High-pass filter cutoff (Hz). Uses config default if None.
            lowpass: Low-pass filter cutoff (Hz). Uses config default if None.
            notch: Notch filter frequency (Hz). Uses config default if None.
        
        Returns:
            Filtered raw data object.
            
        Note:
            MNE-Python uses zero-phase FIR filters by default for better
            temporal precision.
            
        CUSTOMIZATION:
            - Adjust filter parameters for your signal characteristics
            - Consider using IIR filters for real-time applications
            - Use shorter filter lengths for edge artifact reduction
        """
        if self.raw is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        
        highpass = highpass or self.config['highpass_freq']
        lowpass = lowpass or self.config['lowpass_freq']
        notch = notch or self.config['notch_freq']
        
        # Apply high-pass filter (removes slow drifts)
        if highpass is not None and highpass > 0:
            print(f"Applying high-pass filter at {highpass} Hz...")
            self.raw.filter(l_freq=highpass, h_freq=None, 
                           picks='eeg', method='fir', phase='zero')
        
        # Apply low-pass filter (anti-aliasing, removes high-freq noise)
        if lowpass is not None:
            print(f"Applying low-pass filter at {lowpass} Hz...")
            self.raw.filter(l_freq=None, h_freq=lowpass,
                           picks='eeg', method='fir', phase='zero')
        
        # Apply notch filter (removes line noise)
        if notch is not None:
            print(f"Applying notch filter at {notch} Hz...")
            freqs = np.arange(notch, self.raw.info['sfreq'] / 2, notch)
            self.raw.notch_filter(freqs, picks='eeg', method='fir', phase='zero')
        
        return self.raw
    
    def detect_bad_channels(self, method: str = 'correlation') -> List[str]:
        """Automatically detect bad channels.
        
        Args:
            method: Detection method ('correlation', 'variance', 'manual').
        
        Returns:
            List of bad channel names.
            
        CUSTOMIZATION:
            - Adjust thresholds based on your data quality
            - Implement custom detection algorithms
            - Combine multiple detection methods
        """
        if self.raw is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        
        bad_channels = list(self.config['bad_channels'])
        
        if method == 'correlation':
            # PLACEHOLDER: Implement correlation-based detection
            # Compare each channel to the average of others
            pass
        elif method == 'variance':
            # PLACEHOLDER: Implement variance-based detection
            # Flag channels with unusually high or low variance
            pass
        
        if bad_channels:
            self.raw.info['bads'] = bad_channels
            print(f"Marked {len(bad_channels)} bad channels: {bad_channels}")
        
        return bad_channels
    
    def interpolate_bad_channels(self) -> None:
        """Interpolate bad channels using spherical splines.
        
        Uses MNE's interpolate_bads method which performs spherical
        spline interpolation.
        """
        if self.raw is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        
        if self.raw.info['bads']:
            print(f"Interpolating {len(self.raw.info['bads'])} bad channels...")
            self.raw.interpolate_bads(reset_bads=True)
        else:
            print("No bad channels to interpolate.")
    
    def apply_reference(self, reference: Optional[str] = None) -> None:
        """Apply re-referencing to the data.
        
        Args:
            reference: Reference type ('average', 'mastoids', or channel list).
                      Uses config default if None.
                      
        CUSTOMIZATION:
            - 'average': Common reference (average of all channels)
            - 'mastoids': Average of mastoid channels (e.g., ['M1', 'M2'])
            - ['Cz']: Single channel reference
            - ['TP9', 'TP10']: Multiple channel reference
        """
        if self.raw is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        
        reference = reference or self.config['reference']
        
        if reference == 'average':
            print("Applying average reference...")
            self.raw.set_eeg_reference('average', projection=False)
        elif isinstance(reference, list):
            print(f"Applying reference to channels: {reference}...")
            self.raw.set_eeg_reference(reference, projection=False)
        else:
            print(f"Applying reference: {reference}...")
            self.raw.set_eeg_reference(reference, projection=False)
    
    def run_ica(self, 
                n_components: Optional[int] = None,
                method: str = 'fastica') -> ICA:
        """Run Independent Component Analysis for artifact removal.
        
        ICA can separate independent sources like eye movements, muscle
        artifacts, and heartbeat from brain signals.
        
        Args:
            n_components: Number of ICA components. Uses config default if None.
            method: ICA algorithm ('fastica', 'infomax', 'picard').
        
        Returns:
            Fitted ICA object.
            
        INTEGRATION:
            - Use MNE's ICA implementation
            - Can integrate with ERPLAB's artifact detection
            - Consider using ICLabel for automatic component classification
            
        CUSTOMIZATION:
            - Adjust n_components based on channel count (typically 0.9 * n_channels)
            - Try different ICA algorithms for better convergence
            - Use random_state for reproducibility
        """
        if self.raw is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        
        n_components = n_components or self.config['n_ica_components']
        
        print(f"Running ICA with {n_components} components using {method}...")
        self.ica = ICA(n_components=n_components, method=method, 
                       random_state=42, max_iter=500)
        
        # Fit ICA on filtered data (recommended: 1 Hz high-pass)
        raw_filtered = self.raw.copy().filter(l_freq=1.0, h_freq=None)
        self.ica.fit(raw_filtered, picks='eeg')
        
        print(f"ICA converged after {self.ica.n_iter_} iterations")
        return self.ica
    
    def detect_artifacts_ica(self, 
                            eog_channels: Optional[List[str]] = None,
                            threshold: float = 0.5) -> List[int]:
        """Automatically detect artifact components using EOG correlation.
        
        Args:
            eog_channels: List of EOG channel names. Uses config default if None.
            threshold: Correlation threshold for artifact detection.
        
        Returns:
            List of artifact component indices.
            
        CUSTOMIZATION:
            - Adjust threshold based on your ICA quality
            - Use additional artifact detection methods (ECG, muscle)
            - Manually inspect components before exclusion
        """
        if self.ica is None:
            raise RuntimeError("No ICA fitted. Call run_ica() first.")
        
        eog_channels = eog_channels or self.config['eog_channels']
        artifact_components = []
        
        # Detect EOG artifacts if EOG channels available
        if eog_channels and any(ch in self.raw.ch_names for ch in eog_channels):
            eog_indices, eog_scores = self.ica.find_bads_eog(
                self.raw, ch_name=eog_channels, threshold=threshold
            )
            artifact_components.extend(eog_indices)
            print(f"Found {len(eog_indices)} EOG-related components: {eog_indices}")
        
        # PLACEHOLDER: Add ECG artifact detection
        # ecg_indices, ecg_scores = self.ica.find_bads_ecg(self.raw)
        
        # PLACEHOLDER: Add muscle artifact detection
        # Use frequency analysis or variance measures
        
        self.ica.exclude = artifact_components
        return artifact_components
    
    def apply_ica(self) -> mne.io.Raw:
        """Apply ICA to remove artifact components.
        
        Returns:
            Raw data with artifacts removed.
        """
        if self.ica is None:
            raise RuntimeError("No ICA fitted. Call run_ica() first.")
        
        if not self.ica.exclude:
            print("Warning: No components marked for exclusion.")
        
        print(f"Removing {len(self.ica.exclude)} ICA components...")
        self.raw = self.ica.apply(self.raw.copy())
        return self.raw
    
    def extract_events(self, 
                      stim_channel: Optional[str] = None,
                      min_duration: Optional[float] = None) -> np.ndarray:
        """Extract events from stimulus channel or annotations.
        
        Args:
            stim_channel: Name of stimulus channel. If None, uses annotations.
            min_duration: Minimum event duration (s). Uses config default if None.
        
        Returns:
            Event array (n_events, 3) with [sample, duration, event_id].
            
        INTEGRATION:
            - Compatible with ERPLAB event binning
            - Use MNE's event structure for consistency
            - Can merge with behavioral log files
        """
        if self.raw is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        
        min_duration = min_duration or self.config['min_event_duration']
        
        if stim_channel:
            # Extract events from trigger channel
            events = mne.find_events(self.raw, stim_channel=stim_channel,
                                    min_duration=min_duration, verbose=True)
        else:
            # Extract events from annotations
            events, event_id = mne.events_from_annotations(self.raw)
            if self.config['event_id'] is None:
                self.config['event_id'] = event_id
        
        print(f"Found {len(events)} events")
        return events
    
    def create_epochs(self,
                     events: np.ndarray,
                     event_id: Optional[Dict] = None,
                     tmin: Optional[float] = None,
                     tmax: Optional[float] = None,
                     baseline: Optional[Tuple] = None) -> mne.Epochs:
        """Create epochs around stimulus events.
        
        Args:
            events: Event array from extract_events().
            event_id: Dictionary mapping event names to codes.
            tmin: Epoch start time (s). Uses config default if None.
            tmax: Epoch end time (s). Uses config default if None.
            baseline: Baseline correction window (s). Uses config default if None.
        
        Returns:
            Epoched data object.
            
        CUSTOMIZATION:
            - Adjust time windows for your experimental design
            - Use different baseline windows (e.g., prestimulus only)
            - Set reject parameters for initial artifact rejection
        """
        if self.raw is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        
        event_id = event_id or self.config['event_id']
        tmin = tmin if tmin is not None else self.config['tmin']
        tmax = tmax if tmax is not None else self.config['tmax']
        baseline = baseline if baseline is not None else self.config['baseline']
        
        # Initial rejection criteria (can be refined with AutoReject)
        reject = {'eeg': self.config['reject_peak_to_peak']}
        
        print(f"Creating epochs from {tmin} to {tmax} s...")
        self.epochs = mne.Epochs(
            self.raw, events, event_id=event_id,
            tmin=tmin, tmax=tmax, baseline=baseline,
            reject=reject, preload=True, proj=False
        )
        
        print(f"Created {len(self.epochs)} epochs")
        return self.epochs
    
    def reject_artifacts_epochs(self, method: str = 'threshold') -> mne.Epochs:
        """Reject epochs containing artifacts.
        
        Args:
            method: Rejection method ('threshold', 'autoreject', 'manual').
        
        Returns:
            Clean epochs after artifact rejection.
            
        INTEGRATION:
            - AutoReject: Automated threshold optimization
            - ERPLAB: Can use ERPLAB artifact detection
            - Manual: Interactive plotting for manual rejection
            
        CUSTOMIZATION:
            - Adjust rejection thresholds for your data quality
            - Combine multiple rejection methods
            - Use interpolation for bad channels within epochs
        """
        if self.epochs is None:
            raise RuntimeError("No epochs created. Call create_epochs() first.")
        
        n_epochs_before = len(self.epochs)
        
        if method == 'threshold':
            # Already applied during epoch creation
            print(f"Using threshold-based rejection")
            
        elif method == 'autoreject' and AUTOREJECT_AVAILABLE:
            print("Running AutoReject...")
            ar = AutoReject(n_interpolate=[1, 2, 3, 4], 
                          random_state=42, n_jobs=-1, verbose=False)
            self.epochs, reject_log = ar.fit_transform(self.epochs, return_log=True)
            print(f"AutoReject: {reject_log.bad_epochs.sum()} bad epochs detected")
            
        elif method == 'manual':
            print("Opening interactive plot for manual rejection...")
            # PLACEHOLDER: Implement interactive rejection
            # self.epochs.plot(n_epochs=20, block=True)
            pass
        
        n_epochs_after = len(self.epochs)
        n_rejected = n_epochs_before - n_epochs_after
        print(f"Rejected {n_rejected} epochs ({n_rejected/n_epochs_before*100:.1f}%)")
        
        return self.epochs
    
    def preprocess_pipeline(self,
                           filepath: Union[str, Path],
                           events: Optional[np.ndarray] = None,
                           run_ica: bool = True) -> mne.Epochs:
        """Run complete preprocessing pipeline.
        
        This is a convenience method that runs all preprocessing steps
        in sequence with default parameters.
        
        Args:
            filepath: Path to raw EEG data file.
            events: Event array. If None, extracts from data.
            run_ica: Whether to run ICA artifact removal.
        
        Returns:
            Clean, preprocessed epochs ready for analysis.
            
        Example:
            >>> config = {'highpass_freq': 0.5, 'lowpass_freq': 30.0}
            >>> preprocessor = EEGPreprocessor(config)
            >>> epochs = preprocessor.preprocess_pipeline('data/sub01.set')
        """
        print("="*60)
        print("Starting EEG Preprocessing Pipeline")
        print("="*60)
        
        # 1. Load data
        print("\n[1/9] Loading data...")
        self.load_data(filepath)
        
        # 2. Set montage
        print("\n[2/9] Setting montage...")
        self.set_montage()
        
        # 3. Filter data
        print("\n[3/9] Filtering data...")
        self.filter_data()
        
        # 4. Detect and interpolate bad channels
        print("\n[4/9] Detecting bad channels...")
        self.detect_bad_channels()
        self.interpolate_bad_channels()
        
        # 5. Apply reference
        print("\n[5/9] Applying reference...")
        self.apply_reference()
        
        # 6. Run ICA if requested
        if run_ica:
            print("\n[6/9] Running ICA...")
            self.run_ica()
            self.detect_artifacts_ica()
            self.apply_ica()
        else:
            print("\n[6/9] Skipping ICA...")
        
        # 7. Extract events
        print("\n[7/9] Extracting events...")
        if events is None:
            events = self.extract_events()
        
        # 8. Create epochs
        print("\n[8/9] Creating epochs...")
        self.create_epochs(events)
        
        # 9. Reject artifacts
        print("\n[9/9] Rejecting artifacts...")
        self.reject_artifacts_epochs(
            method='autoreject' if self.config['use_autoreject'] else 'threshold'
        )
        
        print("\n" + "="*60)
        print("Preprocessing Complete!")
        print(f"Final data: {len(self.epochs)} epochs, "
              f"{len(self.epochs.ch_names)} channels")
        print("="*60)
        
        return self.epochs
    
    def save_preprocessed(self, 
                         output_path: Union[str, Path],
                         overwrite: bool = False) -> None:
        """Save preprocessed epochs to file.
        
        Args:
            output_path: Path for output file (FIF format recommended).
            overwrite: Whether to overwrite existing file.
        """
        if self.epochs is None:
            raise RuntimeError("No epochs to save. Run preprocessing first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving preprocessed data to {output_path}...")
        self.epochs.save(output_path, overwrite=overwrite)
        print("Save complete.")


def load_preprocessed_epochs(filepath: Union[str, Path]) -> mne.Epochs:
    """Load preprocessed epochs from file.
    
    Args:
        filepath: Path to saved epochs file.
    
    Returns:
        Loaded epochs object.
    """
    print(f"Loading preprocessed epochs from {filepath}...")
    epochs = mne.read_epochs(filepath, preload=True)
    print(f"Loaded {len(epochs)} epochs with {len(epochs.ch_names)} channels")
    return epochs


if __name__ == "__main__":
    # Example usage
    print("EEG Preprocessing Pipeline Template")
    print("\nThis script provides a complete preprocessing pipeline.")
    print("\nUsage example:")
    print("""
    from preprocessing.eeg_preprocessing import EEGPreprocessor
    
    # Configure preprocessing
    config = {
        'highpass_freq': 0.5,
        'lowpass_freq': 30.0,
        'tmin': -0.2,
        'tmax': 0.8,
        'use_autoreject': True,
    }
    
    # Run pipeline
    preprocessor = EEGPreprocessor(config)
    epochs = preprocessor.preprocess_pipeline('data/subject_01.set')
    preprocessor.save_preprocessed('data/preprocessed/subject_01-epo.fif')
    """)
