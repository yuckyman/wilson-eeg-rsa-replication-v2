# DESIGN NOTES: EEG-RSA Pipeline Technical Architecture

**Project**: Wilson EEG-RSA Replication v2  
**Author**: ian  
**Last Updated**: October 30, 2025  
**Version**: 2.0

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Phase 1: Dataset Acquisition (Papercheck)](#phase-1-dataset-acquisition-papercheck)
4. [Phase 2: ERP Preprocessing (ERPLAB v10)](#phase-2-erp-preprocessing-erplab-v10)
5. [Phase 3: Standard EEG Processing (MNE-Python)](#phase-3-standard-eeg-processing-mne-python)
6. [Phase 4: Deep Learning Analysis (Braindecode)](#phase-4-deep-learning-analysis-braindecode)
7. [Phase 5: RSA Analysis (Scipy/Sklearn)](#phase-5-rsa-analysis-scipysklearn)
8. [Integration Points](#integration-points)
9. [Data Flow Diagram](#data-flow-diagram)
10. [Implementation Guidelines](#implementation-guidelines)
11. [Performance Considerations](#performance-considerations)
12. [Future Enhancements](#future-enhancements)

---

## Overview

This document provides a detailed technical specification of the tools, libraries, and integration strategies employed in the EEG-RSA pipeline for replicating Wilson et al.'s mental imagery versus perception study. The pipeline is designed with modularity, reproducibility, and extensibility as core principles.

### Design Philosophy

- **Modularity**: Each phase is self-contained with clear input/output contracts
- **Reproducibility**: Version-pinned dependencies and deterministic processing
- **Flexibility**: Support for multiple analysis approaches and parameter configurations
- **Scalability**: Efficient processing of multi-subject datasets
- **Interoperability**: Seamless data exchange between different toolboxes

### Technology Stack Summary

| Phase | Primary Tool | Language/Interface | Purpose |
|-------|--------------|-------------------|---------|
| 1 | Papercheck | Python API | Dataset download and validation |
| 2 | ERPLAB v10 | MATLAB/Python wrapper | ERP preprocessing and MVPA |
| 3 | MNE-Python | Python | Standard EEG processing |
| 4 | Braindecode | Python/PyTorch | Deep learning on EEG |
| 5 | Scipy/Sklearn | Python | RSA and statistical analysis |

---

## Pipeline Architecture

### High-Level Workflow

```
Raw Data Acquisition → Preprocessing → Feature Extraction → Analysis → Visualization
       ↓                    ↓                ↓                ↓            ↓
   Papercheck          ERPLAB v10       MNE-Python       Scipy/Sklearn   Matplotlib
                           ↓                ↓                             Seaborn
                      MNE-Python      Braindecode
```

### Data Format Conversions

```
Wilson Dataset → .set/.fif → EEGLAB structure → MNE Raw → NumPy arrays → Results
  (OpenNeuro)     (Raw)        (ERPLAB)         (MNE)      (Analysis)     (Plots)
```

---

## Phase 1: Dataset Acquisition (Papercheck)

### Purpose

Automated download, validation, and organization of the Wilson et al. EEG dataset from public repositories (OpenNeuro, Zenodo, or institutional servers).

### Tool: Papercheck

**Version**: 0.4.0+  
**Repository**: [papercheck GitHub](https://github.com/papercheck/papercheck)  
**Installation**: `pip install papercheck`

### Key Features

- **Automated Dataset Discovery**: Query DataLad, OpenNeuro, and OSF repositories
- **Version Control**: Track dataset versions and updates
- **Integrity Verification**: SHA-256 checksums for downloaded files
- **Metadata Extraction**: Parse BIDS-compliant dataset descriptions
- **Bandwidth Management**: Resume interrupted downloads

### Implementation Details

#### 1. Configuration

```python
# src/data_acquisition/papercheck_config.py

from papercheck import DatasetFetcher

WILSON_DATASET_CONFIG = {
    'source': 'openneuro',
    'dataset_id': 'ds003825',  # Example OpenNeuro ID
    'target_dir': 'data/raw/wilson_dataset/',
    'subjects': 'all',  # or specific: ['sub-01', 'sub-02']
    'sessions': 'all',
    'datatypes': ['eeg'],
    'validate': True,
    'parallel_downloads': 4
}
```

#### 2. Download Script

```python
# src/data_acquisition/download_wilson_data.py

import papercheck
from pathlib import Path
import logging

class WilsonDatasetDownloader:
    """
    Handles downloading and validation of Wilson EEG dataset.
    """
    
    def __init__(self, config):
        self.config = config
        self.fetcher = papercheck.DatasetFetcher(
            source=config['source'],
            dataset_id=config['dataset_id']
        )
        self.logger = logging.getLogger(__name__)
        
    def download_dataset(self):
        """Download complete dataset with validation."""
        self.logger.info(f"Downloading dataset {self.config['dataset_id']}")
        
        # Download with progress tracking
        self.fetcher.download(
            target_dir=self.config['target_dir'],
            subjects=self.config['subjects'],
            validate_checksums=self.config['validate']
        )
        
    def validate_bids_structure(self):
        """Verify BIDS compliance of downloaded data."""
        from bids import BIDSLayout
        
        layout = BIDSLayout(self.config['target_dir'])
        
        # Extract metadata
        subjects = layout.get_subjects()
        sessions = layout.get_sessions()
        tasks = layout.get_tasks()
        
        self.logger.info(f"Found {len(subjects)} subjects")
        self.logger.info(f"Tasks: {tasks}")
        
        return layout
        
    def organize_for_pipeline(self):
        """Reorganize data for pipeline processing."""
        # Create symbolic links or copy to expected locations
        pass
```

#### 3. Metadata Extraction

```python
# Extract experimental parameters from dataset

def extract_experiment_metadata(bids_layout):
    """Parse dataset_description.json and participants.tsv"""
    
    metadata = {
        'n_subjects': len(bids_layout.get_subjects()),
        'sampling_rate': None,  # Extract from EEG file
        'n_channels': None,
        'conditions': ['imagery', 'perception'],
        'n_trials_per_condition': None
    }
    
    # Read first EEG file to get technical specs
    eeg_file = bids_layout.get(subject='01', datatype='eeg', 
                                extension='.set')[0]
    
    # Parse using MNE
    import mne
    raw = mne.io.read_raw_eeglab(eeg_file, preload=False)
    metadata['sampling_rate'] = raw.info['sfreq']
    metadata['n_channels'] = len(raw.ch_names)
    
    return metadata
```

### Integration Points

- **Output Format**: BIDS-compliant directory structure
- **Next Phase**: Raw .set files passed to ERPLAB preprocessing
- **Validation**: Checksum verification, BIDS validation
- **Logging**: Download progress, file counts, errors

### Dependencies

```python
papercheck>=0.4.0
datalad>=0.18.0  # Backend for some repositories
pybids>=0.15.0   # BIDS validation
requests>=2.28.0
tqdm>=4.65.0     # Progress bars
```

---

## Phase 2: ERP Preprocessing (ERPLAB v10)

### Purpose

Event-Related Potential (ERP) preprocessing including filtering, artifact detection, epoching, and preparation for multivariate pattern analysis (MVPA).

### Tool: ERPLAB v10

**Version**: 10.0+  
**Repository**: [ERPLAB GitHub](https://github.com/lucklab/erplab)  
**Interface**: MATLAB with Python wrapper  
**Documentation**: [ERPLAB Manual](https://github.com/lucklab/erplab/wiki)

### Key Features

- **Advanced Filtering**: IIR and FIR filters optimized for ERP analysis
- **Artifact Detection**: Moving window peak-to-peak, step function detection
- **Epoch Extraction**: Event-locked averaging with flexible baselines
- **Channel Operations**: Re-referencing, interpolation, Laplacian transforms
- **MVPA Support**: Export epoched data for pattern analysis
- **Scripting**: Batch processing via MATLAB scripts or Python wrapper

### Implementation Details

#### 1. Python-MATLAB Bridge

```python
# src/preprocessing/erplab_interface.py

import matlab.engine
from pathlib import Path
import numpy as np

class ERPLABProcessor:
    """
    Python interface to ERPLAB MATLAB functions.
    """
    
    def __init__(self, matlab_path='/usr/local/MATLAB/R2023b'):
        """Initialize MATLAB engine with ERPLAB."""
        self.eng = matlab.engine.start_matlab()
        
        # Add ERPLAB to MATLAB path
        erplab_path = Path.home() / 'ERPLAB' / 'erplab10.0'
        self.eng.addpath(str(erplab_path), nargout=0)
        
        # Verify ERPLAB is available
        version = self.eng.eval("erplab('version')", nargout=1)
        print(f"ERPLAB version: {version}")
        
    def load_eeglab_set(self, filepath):
        """Load EEGLAB .set file into MATLAB workspace."""
        filepath = str(Path(filepath).resolve())
        
        self.eng.eval(f"""
            EEG = pop_loadset('filename', '{Path(filepath).name}', ...
                              'filepath', '{Path(filepath).parent}');
            [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, 0);
        """, nargout=0)
        
        return True
        
    def apply_bandpass_filter(self, lowcut=0.1, highcut=40, order=2):
        """Apply ERPLAB's optimal FIR filter."""
        
        self.eng.eval(f"""
            EEG = pop_basicfilter(EEG, 1:EEG.nbchan, ...
                'Cutoff', [{lowcut} {highcut}], ...
                'Design', 'butter', ...
                'Filter', 'bandpass', ...
                'Order', {order}, ...
                'RemoveDC', 'on', ...
                'Boundary', 'boundary');
            [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
        """, nargout=0)
        
    def create_eventlist(self):
        """Generate ERPLAB EVENTLIST structure."""
        
        self.eng.eval("""
            EEG = pop_creabasiceventlist(EEG, ...
                'AlphanumericCleaning', 'on', ...
                'BoundaryNumeric', {-99}, ...
                'BoundaryString', {'boundary'});
            [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
        """, nargout=0)
        
    def bin_epochs(self, bin_descriptor_file):
        """Assign events to bins using BDF file."""
        
        self.eng.eval(f"""
            EEG = pop_binlister(EEG, ...
                'BDF', '{bin_descriptor_file}', ...
                'IndexEL', 1, ...
                'SendEL2', 'EEG', ...
                'Voutput', 'EEG');
            [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
        """, nargout=0)
        
    def extract_epochs(self, tmin=-200, tmax=1000, baseline=(-200, 0)):
        """Extract bin-based epochs."""
        
        self.eng.eval(f"""
            EEG = pop_epochbin(EEG, [{tmin} {tmax}], ...
                'pre', {abs(baseline[0])});
            [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
        """, nargout=0)
        
    def artifact_detection(self, threshold_uv=100):
        """Detect artifacts using moving window analysis."""
        
        self.eng.eval(f"""
            EEG = pop_artmwppth(EEG, ...
                'Channel', 1:EEG.nbchan, ...
                'Flag', 1, ...
                'Threshold', {threshold_uv}, ...
                'Twindow', [tmin tmax], ...
                'Windowsize', 200, ...
                'Windowstep', 50);
            [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
        """, nargout=0)
        
    def export_for_mvpa(self, output_file):
        """Export epoched data for multivariate analysis."""
        
        # Get epoched data from MATLAB
        data = self.eng.workspace['EEG'].data
        times = self.eng.workspace['EEG'].times
        chanlocs = self.eng.workspace['EEG'].chanlocs
        
        # Convert to NumPy and save
        data_np = np.array(data)
        
        np.savez(output_file,
                 data=data_np,
                 times=np.array(times),
                 channels=[ch.labels for ch in chanlocs],
                 sfreq=self.eng.workspace['EEG'].srate)
```

#### 2. Batch Processing Script

```python
# scripts/run_erplab_preprocessing.py

from src.preprocessing.erplab_interface import ERPLABProcessor
from pathlib import Path
import yaml

def preprocess_subject_erplab(subject_id, config):
    """Complete ERPLAB preprocessing for one subject."""
    
    processor = ERPLABProcessor()
    
    # Input/output paths
    raw_file = Path(config['data_dir']) / 'raw' / f'{subject_id}_raw.set'
    output_file = Path(config['data_dir']) / 'erplab_preprocessed' / \
                  f'{subject_id}_erplab.npz'
    
    # Load data
    processor.load_eeglab_set(raw_file)
    
    # Preprocessing steps
    processor.apply_bandpass_filter(
        lowcut=config['filter']['highpass'],
        highcut=config['filter']['lowpass']
    )
    
    processor.create_eventlist()
    
    processor.bin_epochs(
        bin_descriptor_file=config['bin_descriptor']
    )
    
    processor.extract_epochs(
        tmin=config['epoch']['tmin'],
        tmax=config['epoch']['tmax'],
        baseline=config['epoch']['baseline']
    )
    
    processor.artifact_detection(
        threshold_uv=config['artifact']['threshold']
    )
    
    # Export for Python-based MVPA
    processor.export_for_mvpa(output_file)
    
    return output_file
```

#### 3. Bin Descriptor File (BDF)

```matlab
# configs/wilson_bins.txt (ERPLAB BDF format)

bin 1
Mental Imagery - Condition A
.{11}

bin 2
Visual Perception - Condition B
.{21}

bin 3
Mental Imagery - Condition C
.{12}

bin 4
Visual Perception - Condition D
.{22}
```

### Integration Points

- **Input**: BIDS .set files from Papercheck phase
- **Output**: 
  - Preprocessed .set files (ERPLAB format)
  - .npz files (NumPy format for Python pipeline)
- **Next Phase**: Data loaded by MNE-Python for further processing
- **Artifacts**: Artifact detection results exported as rejection matrices

### Dependencies

```python
# Python dependencies
matlab.engine>=0.1.0  # MATLAB Python API
numpy>=1.24.0
pyyaml>=6.0

# MATLAB dependencies
# - MATLAB R2020b or later
# - EEGLAB 2022.0+
# - ERPLAB 10.0+
```

### Performance Notes

- MATLAB engine startup: ~5-10 seconds
- Processing time per subject: ~2-5 minutes (depending on trials)
- Memory usage: ~2-4 GB per subject (64 channels, 1000 trials)
- Batch processing: Process multiple subjects in parallel using separate MATLAB instances

---

## Phase 3: Standard EEG Processing (MNE-Python)

### Purpose

Standard EEG data handling, advanced artifact removal (ICA), time-frequency analysis, source localization preparation, and data quality control.

### Tool: MNE-Python

**Version**: 1.5.0+  
**Repository**: [MNE-Python GitHub](https://github.com/mne-tools/mne-python)  
**Documentation**: [MNE Documentation](https://mne.tools)

### Key Features

- **Comprehensive I/O**: Read 50+ file formats (.fif, .set, .edf, .bdf, etc.)
- **ICA Implementation**: FastICA, Infomax, extended Infomax
- **Time-Frequency**: Morlet wavelets, multitapers, Hilbert transforms
- **Source Space**: Forward modeling, inverse solutions (MNE, dSPM, sLORETA)
- **Visualization**: Interactive plots with `mne.viz`
- **Statistics**: Cluster permutation tests, spatio-temporal statistics
- **Parallel Processing**: Joblib integration for multi-core execution

### Implementation Details

#### 1. Loading ERPLAB Output

```python
# src/preprocessing/mne_loader.py

import mne
import numpy as np
from pathlib import Path

class MNEDataLoader:
    """
    Load ERPLAB-preprocessed data into MNE structures.
    """
    
    def __init__(self, montage='standard_1020'):
        self.montage = mne.channels.make_standard_montage(montage)
        
    def load_from_erplab(self, erplab_file):
        """
        Load .npz file exported from ERPLAB.
        Convert to MNE Epochs object.
        """
        # Load NPZ
        data = np.load(erplab_file)
        eeg_data = data['data']  # Shape: (n_channels, n_samples, n_epochs)
        times = data['times']
        channels = data['channels']
        sfreq = float(data['sfreq'])
        
        # Transpose to MNE format: (n_epochs, n_channels, n_samples)
        eeg_data = np.transpose(eeg_data, (2, 0, 1))
        
        # Create Info structure
        info = mne.create_info(
            ch_names=list(channels),
            sfreq=sfreq,
            ch_types='eeg'
        )
        info.set_montage(self.montage)
        
        # Create Epochs object
        epochs = mne.EpochsArray(eeg_data, info, tmin=times[0]/1000)
        
        return epochs
        
    def load_from_set(self, set_file, preload=True):
        """
        Load EEGLAB .set file directly.
        """
        raw = mne.io.read_raw_eeglab(set_file, preload=preload)
        raw.set_montage(self.montage)
        
        return raw
```

#### 2. ICA Artifact Removal

```python
# src/preprocessing/ica_processing.py

from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
import matplotlib.pyplot as plt

class ICAProcessor:
    """
    Advanced ICA-based artifact removal using MNE.
    """
    
    def __init__(self, n_components=25, method='fastica', random_state=42):
        self.ica = ICA(
            n_components=n_components,
            method=method,
            random_state=random_state,
            max_iter='auto'
        )
        
    def fit_ica(self, epochs, decim=3):
        """
        Fit ICA on epoched data.
        
        Parameters:
        -----------
        epochs : mne.Epochs
            Cleaned epochs
        decim : int
            Decimation factor (speeds up ICA)
        """
        # Filter for ICA (1-40 Hz recommended)
        epochs_ica = epochs.copy().filter(l_freq=1.0, h_freq=None)
        
        self.ica.fit(epochs_ica, decim=decim)
        
        print(f"ICA fitted with {self.ica.n_components_} components")
        
    def detect_artifacts_auto(self, epochs):
        """
        Automatically detect EOG and ECG artifacts.
        """
        # Detect EOG components
        eog_indices, eog_scores = self.ica.find_bads_eog(epochs)
        self.ica.exclude.extend(eog_indices)
        
        print(f"Detected {len(eog_indices)} EOG components: {eog_indices}")
        
        # Detect ECG components if ECG channel present
        if 'ECG' in epochs.ch_names:
            ecg_indices, ecg_scores = self.ica.find_bads_ecg(epochs)
            self.ica.exclude.extend(ecg_indices)
            print(f"Detected {len(ecg_indices)} ECG components: {ecg_indices}")
            
    def plot_components(self, epochs):
        """Interactive component visualization."""
        self.ica.plot_components(inst=epochs)
        self.ica.plot_sources(epochs)
        
    def apply_ica(self, epochs):
        """Remove detected artifact components."""
        epochs_clean = self.ica.apply(epochs.copy())
        
        print(f"Removed {len(self.ica.exclude)} ICA components")
        
        return epochs_clean
```

#### 3. Time-Frequency Analysis

```python
# src/analysis/time_frequency.py

import mne
from mne.time_frequency import tfr_morlet, tfr_multitaper
import numpy as np

class TimeFrequencyAnalyzer:
    """
    Time-frequency decomposition for EEG data.
    """
    
    def __init__(self, freqs=np.arange(4, 40, 1), n_cycles=None):
        """
        Parameters:
        -----------
        freqs : array-like
            Frequencies of interest (Hz)
        n_cycles : array-like or float
            Number of cycles (temporal resolution)
            If None, uses freqs / 2 (constant 2-cycle window)
        """
        self.freqs = freqs
        self.n_cycles = n_cycles if n_cycles is not None else freqs / 2.0
        
    def compute_tfr_morlet(self, epochs, average=True):
        """
        Compute time-frequency representation using Morlet wavelets.
        """
        power = tfr_morlet(
            epochs,
            freqs=self.freqs,
            n_cycles=self.n_cycles,
            use_fft=True,
            return_itc=False,
            decim=2,
            n_jobs=-1,  # Use all cores
            average=average
        )
        
        return power
        
    def compute_tfr_multitaper(self, epochs, bandwidth=4.0):
        """
        Compute TFR using multitaper method (better frequency resolution).
        """
        power = tfr_multitaper(
            epochs,
            freqs=self.freqs,
            n_cycles=self.n_cycles,
            time_bandwidth=bandwidth,
            n_jobs=-1,
            average=True
        )
        
        return power
        
    def baseline_correction(self, power, baseline=(-0.2, 0), mode='percent'):
        """Apply baseline correction to TFR."""
        power.apply_baseline(baseline=baseline, mode=mode)
        return power
```

#### 4. Data Quality Metrics

```python
# src/preprocessing/quality_control.py

import mne
import numpy as np
from scipy import stats

class DataQualityChecker:
    """
    Compute and visualize EEG data quality metrics.
    """
    
    def __init__(self, epochs):
        self.epochs = epochs
        self.metrics = {}
        
    def compute_snr(self):
        """Signal-to-noise ratio per channel."""
        data = self.epochs.get_data()
        
        # SNR = mean(signal) / std(baseline)
        baseline_idx = self.epochs.time_as_index([-0.2, 0])
        signal_idx = self.epochs.time_as_index([0.1, 0.5])
        
        baseline_std = data[:, :, baseline_idx[0]:baseline_idx[1]].std(axis=2).mean(axis=0)
        signal_mean = np.abs(data[:, :, signal_idx[0]:signal_idx[1]].mean(axis=2)).mean(axis=0)
        
        snr = signal_mean / baseline_std
        
        self.metrics['snr'] = dict(zip(self.epochs.ch_names, snr))
        
        return snr
        
    def detect_bad_channels(self, z_threshold=3.0):
        """Detect bad channels based on variance."""
        data = self.epochs.get_data()
        
        # Compute variance per channel
        channel_var = data.var(axis=(0, 2))
        
        # Z-score
        z_scores = np.abs(stats.zscore(channel_var))
        
        bad_channels = [self.epochs.ch_names[i] for i, z in enumerate(z_scores) 
                       if z > z_threshold]
        
        self.metrics['bad_channels'] = bad_channels
        
        return bad_channels
        
    def trial_rejection_summary(self):
        """Summarize rejected trials."""
        n_total = len(self.epochs.selection)
        n_rejected = len(self.epochs.drop_log)
        rejection_rate = n_rejected / n_total * 100
        
        self.metrics['rejection_rate'] = rejection_rate
        
        return rejection_rate
        
    def generate_report(self, output_file):
        """Generate HTML report with QC metrics."""
        report = mne.Report(title='Data Quality Report')
        
        # Add metrics
        report.add_html(f"<h2>Quality Metrics</h2>", title="Metrics")
        report.add_html(f"<p>Rejection rate: {self.metrics['rejection_rate']:.2f}%</p>")
        
        # Add PSD plot
        fig_psd = self.epochs.plot_psd(show=False)
        report.add_figure(fig_psd, title='Power Spectral Density')
        
        # Save report
        report.save(output_file, overwrite=True, open_browser=False)
```

### Integration Points

- **Input**: 
  - ERPLAB .npz files
  - Raw .set files
- **Output**: 
  - MNE Epochs objects (pickled)
  - Cleaned data for RSA (NumPy arrays)
  - Time-frequency representations
- **Next Phase**: 
  - Cleaned epochs → Braindecode for deep learning
  - Cleaned epochs → RSA analysis
- **Visualization**: Interactive plots, HTML reports

### Dependencies

```python
mne>=1.5.0
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
joblib>=1.3.0
scikit-learn>=1.3.0  # For decomposition methods
nibabel>=5.0.0  # For source space operations
```

### Performance Optimization

```python
# Use parallel processing
mne.set_config('MNE_USE_NUMBA', 'true')  # Accelerate with Numba
mne.set_config('MNE_CACHE_DIR', '/tmp/mne_cache')

# Memory mapping for large datasets
epochs = mne.read_epochs(filename, preload=False)  # Memory-mapped

# GPU acceleration for filtering (if CuPy available)
epochs.filter(l_freq=0.1, h_freq=40, n_jobs='cuda')
```

---

## Phase 4: Deep Learning Analysis (Braindecode)

### Purpose

Apply deep learning models to EEG data for automatic feature extraction, classification, and pattern discovery. Complement traditional RSA with learned representations.

### Tool: Braindecode

**Version**: 0.8.0+  
**Repository**: [Braindecode GitHub](https://github.com/braindecode/braindecode)  
**Documentation**: [Braindecode Docs](https://braindecode.org)  
**Backend**: PyTorch

### Key Features

- **EEG-Specific Architectures**: ShallowConvNet, DeepConvNet, EEGNet, TCN
- **MNE Integration**: Direct compatibility with MNE data structures
- **Cropped Training**: Handle variable-length trials efficiently
- **Interpretability**: Layer-wise relevance propagation, attention visualization
- **Transfer Learning**: Pre-trained models on large EEG datasets

### Implementation Details

#### 1. Data Preparation for Braindecode

```python
# src/deep_learning/braindecode_loader.py

from braindecode.datasets import create_from_mne_epochs
from braindecode.preprocessing import Preprocessor, exponential_moving_standardize
from braindecode.preprocessing import preprocess
import mne
import numpy as np

class BraindecodeDataPrep:
    """
    Prepare MNE epochs for Braindecode models.
    """
    
    def __init__(self, epochs_list, subjects, conditions):
        """
        Parameters:
        -----------
        epochs_list : list of mne.Epochs
            List of epoch objects (one per subject)
        subjects : list of str
            Subject identifiers
        conditions : dict
            Mapping of event_id to condition labels
        """
        self.epochs_list = epochs_list
        self.subjects = subjects
        self.conditions = conditions
        
    def create_windows_dataset(self, trial_start_offset=-0.2, trial_stop_offset=1.0):
        """
        Create Braindecode WindowsDataset from MNE epochs.
        """
        from braindecode.datasets import WindowsDataset
        
        # Convert each subject's epochs
        windows_datasets = []
        
        for subject_id, epochs in zip(self.subjects, self.epochs_list):
            # Create dataset
            dataset = create_from_mne_epochs(
                [epochs],
                window_size_samples=None,  # Use full epoch
                window_stride_samples=None,
                drop_last_window=False
            )
            
            # Add subject metadata
            dataset.description['subject'] = subject_id
            
            windows_datasets.append(dataset)
            
        # Concatenate all subjects
        from braindecode.datasets import BaseConcatDataset
        concat_dataset = BaseConcatDataset(windows_datasets)
        
        return concat_dataset
        
    def apply_preprocessing(self, dataset):
        """Apply Braindecode preprocessing transformations."""
        
        preprocessors = [
            # Exponential moving standardization
            Preprocessor('pick_types', eeg=True, meg=False, stim=False),
            Preprocessor(exponential_moving_standardize,
                        factor_new=0.001,
                        init_block_size=1000),
        ]
        
        preprocess(dataset, preprocessors)
        
        return dataset
```

#### 2. Model Architecture

```python
# src/deep_learning/eeg_models.py

import torch
import torch.nn as nn
from braindecode.models import ShallowFBCSPNet, Deep4Net, EEGNetv4

class EEGNetCustom(nn.Module):
    """
    Custom EEGNet architecture for imagery vs. perception classification.
    """
    
    def __init__(self, n_channels=64, n_times=512, n_classes=2):
        super().__init__()
        
        self.model = EEGNetv4(
            n_chans=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            final_conv_length='auto',
            drop_prob=0.5
        )
        
    def forward(self, x):
        return self.model(x)
        
    def extract_features(self, x, layer_name='features'):
        """
        Extract intermediate representations for RSA.
        """
        features = {}
        
        def hook_fn(module, input, output):
            features[layer_name] = output
            
        # Register hook
        if layer_name == 'features':
            handle = self.model.final_layer[0].register_forward_hook(hook_fn)
        
        # Forward pass
        _ = self.forward(x)
        
        handle.remove()
        
        return features[layer_name]


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for EEG sequence modeling.
    """
    
    def __init__(self, n_channels, n_classes, n_filters=32, kernel_size=5):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_channels, n_filters, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(n_filters)
        
        self.conv2 = nn.Conv1d(n_filters, n_filters*2, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(n_filters*2)
        
        self.conv3 = nn.Conv1d(n_filters*2, n_filters*4, kernel_size, padding='same')
        self.bn3 = nn.BatchNorm1d(n_filters*4)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_filters*4, n_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Input: (batch, channels, time)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        
        return x
```

#### 3. Training Pipeline

```python
# src/deep_learning/train_braindecode.py

from braindecode import EEGClassifier
from skorch.callbacks import LRScheduler, EarlyStopping, Checkpoint
from skorch.dataset import ValidSplit
from sklearn.model_selection import cross_val_score
import torch.optim as optim

class BraindecodeTrainer:
    """
    Training pipeline for Braindecode models.
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
    def create_classifier(self, max_epochs=100, lr=0.001):
        """
        Create Skorch-based EEGClassifier wrapper.
        """
        clf = EEGClassifier(
            self.model,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            optimizer__lr=lr,
            batch_size=64,
            max_epochs=max_epochs,
            train_split=ValidSplit(0.2, random_state=42),
            device=self.device,
            callbacks=[
                EarlyStopping(monitor='valid_loss', patience=10),
                LRScheduler(policy='ReduceLROnPlateau', 
                           monitor='valid_loss',
                           patience=5),
                Checkpoint(monitor='valid_acc_best', 
                          fn_prefix='best_model_')
            ],
            verbose=1
        )
        
        return clf
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train model on dataset."""
        clf = self.create_classifier()
        
        if X_val is not None:
            clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            clf.fit(X_train, y_train)
            
        return clf
        
    def cross_validate(self, X, y, cv=5):
        """Cross-validation evaluation."""
        clf = self.create_classifier()
        
        scores = cross_val_score(clf, X, y, cv=cv, 
                                scoring='accuracy', n_jobs=1)
        
        return scores
```

#### 4. Feature Extraction for RSA

```python
# src/deep_learning/feature_extraction.py

import torch
import numpy as np

class DeepFeatureExtractor:
    """
    Extract learned representations from trained models for RSA.
    """
    
    def __init__(self, model, layer_name='penultimate'):
        self.model = model
        self.layer_name = layer_name
        self.features = {}
        
    def _hook_fn(self, module, input, output):
        """Hook function to capture activations."""
        self.features[self.layer_name] = output.detach().cpu().numpy()
        
    def register_hooks(self):
        """Register forward hooks on target layer."""
        # Identify layer
        if self.layer_name == 'penultimate':
            # Last layer before classification
            target_layer = list(self.model.modules())[-2]
        else:
            # Custom layer access
            target_layer = dict(self.model.named_modules())[self.layer_name]
            
        handle = target_layer.register_forward_hook(self._hook_fn)
        
        return handle
        
    def extract_features(self, data_loader):
        """
        Extract features for all trials in data loader.
        
        Returns:
        --------
        features : ndarray, shape (n_trials, n_features)
        labels : ndarray, shape (n_trials,)
        """
        self.model.eval()
        handle = self.register_hooks()
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                # Forward pass
                _ = self.model(batch_X)
                
                # Collect features
                all_features.append(self.features[self.layer_name])
                all_labels.append(batch_y.cpu().numpy())
                
        handle.remove()
        
        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        return features, labels
        
    def extract_time_resolved_features(self, epochs_data):
        """
        Extract features at each time point for time-resolved RSA.
        
        Parameters:
        -----------
        epochs_data : ndarray, shape (n_trials, n_channels, n_times)
        
        Returns:
        --------
        features_timeseries : ndarray, shape (n_times, n_trials, n_features)
        """
        self.model.eval()
        n_trials, n_channels, n_times = epochs_data.shape
        
        # Sliding window extraction
        window_size = 100  # samples
        features_timeseries = []
        
        for t in range(0, n_times - window_size, 10):
            window_data = epochs_data[:, :, t:t+window_size]
            window_tensor = torch.FloatTensor(window_data)
            
            features, _ = self.extract_features(
                [(window_tensor, torch.zeros(n_trials))]
            )
            
            features_timeseries.append(features)
            
        return np.array(features_timeseries)
```

### Integration Points

- **Input**: 
  - MNE Epochs from Phase 3
  - Preprocessed NumPy arrays
- **Output**: 
  - Trained models (PyTorch .pth files)
  - Learned features for RSA (NumPy arrays)
  - Classification accuracies
- **Next Phase**: Deep features passed to RSA analysis
- **Use Cases**:
  - Classification: Imagery vs. Perception
  - Feature learning: Automatic representation discovery
  - Transfer learning: Pre-train on large datasets

### Dependencies

```python
braindecode>=0.8.0
torch>=2.0.0
skorch>=0.12.0  # Scikit-learn wrapper for PyTorch
matplotlib>=3.7.0
scipy>=1.11.0
```

### Training Tips

```python
# GPU acceleration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Mixed precision training (faster)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Data augmentation for EEG
from braindecode.augmentation import FrequencyShift, TimeReverse

augmentations = [FrequencyShift(delta_freq=2), TimeReverse()]
```

---

## Phase 5: RSA Analysis (Scipy/Sklearn)

### Purpose

Compute Representational Similarity Analysis (RSA) to quantify neural pattern similarity across conditions, time points, and feature spaces.

### Tools: Scipy + Scikit-learn

**Scipy Version**: 1.11.0+  
**Sklearn Version**: 1.3.0+  
**Documentation**: [Scipy](https://docs.scipy.org) | [Sklearn](https://scikit-learn.org)

### Key Features

- **Distance Metrics** (Scipy): Euclidean, correlation, Mahalanobis, cosine
- **Statistical Tests** (Scipy): Permutation tests, Spearman correlation, Kendall τ
- **Dimensionality Reduction** (Sklearn): PCA, LDA for noise reduction
- **Cross-Validation** (Sklearn): K-fold CV for robust RDM estimation
- **Clustering** (Sklearn): Hierarchical clustering of representations

### Implementation Details

#### 1. RDM Computation

```python
# src/rsa/rdm_computation.py

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

class RDMComputer:
    """
    Compute Representational Dissimilarity Matrices (RDMs).
    """
    
    def __init__(self, metric='correlation', cv_folds=5):
        """
        Parameters:
        -----------
        metric : str
            Distance metric ('correlation', 'euclidean', 'mahalanobis')
        cv_folds : int
            Number of cross-validation folds for noise estimation
        """
        self.metric = metric
        self.cv_folds = cv_folds
        
    def compute_rdm(self, data, standardize=True):
        """
        Compute RDM from neural patterns.
        
        Parameters:
        -----------
        data : ndarray, shape (n_trials, n_features)
            Neural activity patterns (e.g., channels × timepoints flattened)
            
        Returns:
        --------
        rdm : ndarray, shape (n_trials, n_trials)
            Representational dissimilarity matrix
        """
        # Standardize features
        if standardize:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            
        # Compute pairwise distances
        if self.metric == 'correlation':
            # 1 - Pearson correlation
            distances = pdist(data, metric='correlation')
        elif self.metric == 'euclidean':
            distances = pdist(data, metric='euclidean')
        elif self.metric == 'mahalanobis':
            # Compute covariance
            cov = np.cov(data.T)
            cov_inv = np.linalg.pinv(cov)
            distances = pdist(data, metric='mahalanobis', VI=cov_inv)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
            
        # Convert to square form
        rdm = squareform(distances)
        
        return rdm
        
    def compute_rdm_cv(self, data, labels):
        """
        Compute cross-validated RDM (reduces noise).
        
        Split data into folds, compute RDM on each fold,
        then average across folds.
        """
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        rdms = []
        
        for train_idx, test_idx in kf.split(data):
            # Use training data to compute RDM
            train_data = data[train_idx]
            rdm = self.compute_rdm(train_data)
            rdms.append(rdm)
            
        # Average RDMs
        rdm_mean = np.mean(rdms, axis=0)
        rdm_std = np.std(rdms, axis=0)
        
        return rdm_mean, rdm_std
        
    def compute_condition_rdm(self, data, condition_labels):
        """
        Compute RDM averaged within conditions.
        
        Parameters:
        -----------
        data : ndarray, shape (n_trials, n_features)
        condition_labels : ndarray, shape (n_trials,)
            Condition for each trial
            
        Returns:
        --------
        condition_rdm : ndarray, shape (n_conditions, n_conditions)
        """
        conditions = np.unique(condition_labels)
        n_conditions = len(conditions)
        
        # Average patterns within each condition
        condition_patterns = np.zeros((n_conditions, data.shape[1]))
        
        for i, cond in enumerate(conditions):
            condition_patterns[i] = data[condition_labels == cond].mean(axis=0)
            
        # Compute RDM between condition averages
        condition_rdm = self.compute_rdm(condition_patterns)
        
        return condition_rdm
```

#### 2. Time-Resolved RSA

```python
# src/rsa/time_resolved_rsa.py

import numpy as np
from scipy.stats import spearmanr
from joblib import Parallel, delayed

class TimeResolvedRSA:
    """
    Compute RSA across time for imagery vs. perception comparison.
    """
    
    def __init__(self, rdm_computer):
        self.rdm_computer = rdm_computer
        
    def compute_time_resolved_rdms(self, epochs_data, window_size=50, step_size=10):
        """
        Compute RDMs in sliding time windows.
        
        Parameters:
        -----------
        epochs_data : ndarray, shape (n_trials, n_channels, n_times)
        window_size : int
            Window size in milliseconds
        step_size : int
            Step size in milliseconds
            
        Returns:
        --------
        rdms : list of ndarray
            RDM at each time window
        time_points : ndarray
            Center time of each window
        """
        n_trials, n_channels, n_times = epochs_data.shape
        
        # Convert window sizes from ms to samples (assuming sfreq)
        sfreq = 500  # Hz (should be passed as parameter)
        window_samples = int(window_size * sfreq / 1000)
        step_samples = int(step_size * sfreq / 1000)
        
        rdms = []
        time_points = []
        
        for t_start in range(0, n_times - window_samples, step_samples):
            t_end = t_start + window_samples
            t_center = (t_start + t_end) / 2 / sfreq * 1000  # Convert to ms
            
            # Extract window
            window_data = epochs_data[:, :, t_start:t_end]
            
            # Flatten to (n_trials, n_features)
            window_flat = window_data.reshape(n_trials, -1)
            
            # Compute RDM
            rdm = self.rdm_computer.compute_rdm(window_flat)
            
            rdms.append(rdm)
            time_points.append(t_center)
            
        return rdms, np.array(time_points)
        
    def compare_rdms(self, rdm1, rdm2, method='spearman'):
        """
        Compare two RDMs using correlation.
        
        Parameters:
        -----------
        rdm1, rdm2 : ndarray, shape (n_conditions, n_conditions)
            RDMs to compare
        method : str
            'spearman' or 'kendall' or 'pearson'
            
        Returns:
        --------
        similarity : float
            Correlation between RDMs
        p_value : float
            Statistical significance
        """
        # Extract upper triangular (exclude diagonal)
        triu_idx = np.triu_indices_from(rdm1, k=1)
        rdm1_vec = rdm1[triu_idx]
        rdm2_vec = rdm2[triu_idx]
        
        if method == 'spearman':
            similarity, p_value = spearmanr(rdm1_vec, rdm2_vec)
        elif method == 'kendall':
            similarity, p_value = kendalltau(rdm1_vec, rdm2_vec)
        elif method == 'pearson':
            from scipy.stats import pearsonr
            similarity, p_value = pearsonr(rdm1_vec, rdm2_vec)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return similarity, p_value
        
    def compute_imagery_perception_similarity(self, imagery_epochs, perception_epochs,
                                             sfreq=500):
        """
        Compute time-resolved similarity between imagery and perception RDMs.
        
        Returns:
        --------
        similarity_timeseries : ndarray
            Spearman correlation at each time point
        time_points : ndarray
        """
        # Compute time-resolved RDMs for both conditions
        imagery_rdms, time_points = self.compute_time_resolved_rdms(
            imagery_epochs, window_size=50, step_size=10
        )
        
        perception_rdms, _ = self.compute_time_resolved_rdms(
            perception_epochs, window_size=50, step_size=10
        )
        
        # Compare RDMs at each time point
        similarities = []
        p_values = []
        
        for rdm_img, rdm_per in zip(imagery_rdms, perception_rdms):
            sim, p = self.compare_rdms(rdm_img, rdm_per)
            similarities.append(sim)
            p_values.append(p)
            
        return np.array(similarities), time_points, np.array(p_values)
```

#### 3. Statistical Testing

```python
# src/rsa/statistics.py

import numpy as np
from scipy.stats import ttest_1samp, permutation_test
from joblib import Parallel, delayed

class RSAStatistics:
    """
    Statistical testing for RSA results.
    """
    
    def __init__(self, n_permutations=10000, alpha=0.05):
        self.n_permutations = n_permutations
        self.alpha = alpha
        
    def permutation_test_rdm_similarity(self, rdm1, rdm2, n_permutations=None):
        """
        Permutation test for RDM similarity significance.
        
        H0: RDMs are independent (shuffle one RDM randomly)
        """
        if n_permutations is None:
            n_permutations = self.n_permutations
            
        # Observed similarity
        triu_idx = np.triu_indices_from(rdm1, k=1)
        rdm1_vec = rdm1[triu_idx]
        rdm2_vec = rdm2[triu_idx]
        
        from scipy.stats import spearmanr
        observed_sim, _ = spearmanr(rdm1_vec, rdm2_vec)
        
        # Permutation distribution
        null_sims = []
        
        for _ in range(n_permutations):
            # Shuffle RDM2 (permute rows and columns together)
            perm = np.random.permutation(rdm2.shape[0])
            rdm2_perm = rdm2[perm][:, perm]
            rdm2_perm_vec = rdm2_perm[triu_idx]
            
            null_sim, _ = spearmanr(rdm1_vec, rdm2_perm_vec)
            null_sims.append(null_sim)
            
        null_sims = np.array(null_sims)
        
        # P-value (two-tailed)
        p_value = np.mean(np.abs(null_sims) >= np.abs(observed_sim))
        
        return observed_sim, p_value, null_sims
        
    def cluster_correction_timeseries(self, similarity_timeseries, p_values, 
                                     cluster_threshold=0.05):
        """
        Cluster-based correction for multiple comparisons.
        
        Identify contiguous time windows with p < threshold,
        compute cluster mass statistic.
        """
        # Identify significant time points
        sig_mask = p_values < cluster_threshold
        
        # Find clusters (contiguous True values)
        clusters = []
        cluster_start = None
        
        for i, sig in enumerate(sig_mask):
            if sig and cluster_start is None:
                cluster_start = i
            elif not sig and cluster_start is not None:
                clusters.append((cluster_start, i))
                cluster_start = None
                
        if cluster_start is not None:
            clusters.append((cluster_start, len(sig_mask)))
            
        # Compute cluster mass
        cluster_masses = []
        for start, end in clusters:
            mass = np.sum(similarity_timeseries[start:end])
            cluster_masses.append((start, end, mass))
            
        return clusters, cluster_masses
        
    def bootstrap_confidence_interval(self, rdm1, rdm2, n_bootstrap=1000, ci=95):
        """
        Bootstrap confidence interval for RDM similarity.
        """
        from scipy.stats import spearmanr
        
        triu_idx = np.triu_indices_from(rdm1, k=1)
        rdm1_vec = rdm1[triu_idx]
        rdm2_vec = rdm2[triu_idx]
        
        n_samples = len(rdm1_vec)
        
        bootstrap_sims = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            
            sim, _ = spearmanr(rdm1_vec[idx], rdm2_vec[idx])
            bootstrap_sims.append(sim)
            
        bootstrap_sims = np.array(bootstrap_sims)
        
        # Compute confidence interval
        lower = np.percentile(bootstrap_sims, (100 - ci) / 2)
        upper = np.percentile(bootstrap_sims, 100 - (100 - ci) / 2)
        
        return lower, upper, bootstrap_sims
```

#### 4. Searchlight Analysis

```python
# src/rsa/searchlight.py

import numpy as np
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed

class SearchlightRSA:
    """
    Searchlight analysis for spatially localized RSA.
    """
    
    def __init__(self, rdm_computer, radius=3):
        """
        Parameters:
        -----------
        radius : float
            Searchlight radius (in cm, assuming standard 10-20 system)
        """
        self.rdm_computer = rdm_computer
        self.radius = radius
        
    def get_channel_neighbors(self, ch_idx, channel_positions, radius):
        """
        Find channels within radius of target channel.
        
        Parameters:
        -----------
        ch_idx : int
            Target channel index
        channel_positions : ndarray, shape (n_channels, 3)
            3D positions of channels (x, y, z)
        radius : float
            Radius in same units as positions
            
        Returns:
        --------
        neighbors : list of int
            Indices of neighboring channels
        """
        target_pos = channel_positions[ch_idx]
        
        # Compute distances
        distances = np.linalg.norm(channel_positions - target_pos, axis=1)
        
        # Find neighbors within radius
        neighbors = np.where(distances <= radius)[0]
        
        return neighbors.tolist()
        
    def searchlight_rdm(self, epochs_data, channel_positions, target_rdm):
        """
        Compute searchlight RSA for each channel.
        
        Parameters:
        -----------
        epochs_data : ndarray, shape (n_trials, n_channels, n_times)
        channel_positions : ndarray, shape (n_channels, 3)
        target_rdm : ndarray
            Model RDM to compare against
            
        Returns:
        --------
        similarity_map : ndarray, shape (n_channels,)
            Similarity at each channel
        """
        n_trials, n_channels, n_times = epochs_data.shape
        
        similarity_map = np.zeros(n_channels)
        
        for ch_idx in range(n_channels):
            # Get neighbors
            neighbors = self.get_channel_neighbors(
                ch_idx, channel_positions, self.radius
            )
            
            # Extract data from neighbor channels
            neighbor_data = epochs_data[:, neighbors, :].reshape(n_trials, -1)
            
            # Compute RDM
            local_rdm = self.rdm_computer.compute_rdm(neighbor_data)
            
            # Compare to target RDM
            from scipy.stats import spearmanr
            triu_idx = np.triu_indices_from(local_rdm, k=1)
            sim, _ = spearmanr(local_rdm[triu_idx], target_rdm[triu_idx])
            
            similarity_map[ch_idx] = sim
            
        return similarity_map
```

### Integration Points

- **Input**: 
  - MNE Epochs (NumPy arrays)
  - Braindecode features
  - Channel positions
- **Output**: 
  - RDMs (NumPy arrays, .npy files)
  - Similarity time series
  - Statistical results (CSV, JSON)
  - Searchlight maps
- **Visualization**: 
  - RDM heatmaps (Matplotlib/Seaborn)
  - Time series plots
  - Topographic maps (MNE)

### Dependencies

```python
scipy>=1.11.0
scikit-learn>=1.3.0
numpy>=1.24.0
joblib>=1.3.0  # Parallel processing
```

### Optimization

```python
# Parallel searchlight
from joblib import Parallel, delayed

def parallel_searchlight(epochs_data, channel_positions, target_rdm, n_jobs=-1):
    """Run searchlight in parallel across channels."""
    
    def process_channel(ch_idx):
        # Searchlight for one channel
        pass
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_channel)(ch) for ch in range(n_channels)
    )
    
    return np.array(results)
```

---

## Integration Points

### Cross-Phase Data Flow

```python
# Complete pipeline integration

class WilsonPipeline:
    """
    Integrated pipeline for Wilson EEG-RSA analysis.
    """
    
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def run_full_pipeline(self, subject_id):
        """Execute all phases for one subject."""
        
        # Phase 1: Data acquisition (Papercheck)
        from src.data_acquisition import WilsonDatasetDownloader
        downloader = WilsonDatasetDownloader(self.config['papercheck'])
        raw_file = downloader.get_subject_file(subject_id)
        
        # Phase 2: ERPLAB preprocessing
        from src.preprocessing import ERPLABProcessor
        erplab = ERPLABProcessor()
        erplab.load_eeglab_set(raw_file)
        erplab.apply_bandpass_filter()
        erplab.extract_epochs()
        erplab_output = erplab.export_for_mvpa(f'data/erplab/{subject_id}.npz')
        
        # Phase 3: MNE processing
        from src.preprocessing import MNEDataLoader, ICAProcessor
        loader = MNEDataLoader()
        epochs = loader.load_from_erplab(erplab_output)
        
        ica = ICAProcessor()
        ica.fit_ica(epochs)
        ica.detect_artifacts_auto(epochs)
        epochs_clean = ica.apply_ica(epochs)
        
        # Phase 4: Braindecode feature extraction
        from src.deep_learning import BraindecodeDataPrep, DeepFeatureExtractor
        dataset = BraindecodeDataPrep([epochs_clean], [subject_id], {})
        dataset = dataset.create_windows_dataset()
        
        # Train model (or load pre-trained)
        from src.deep_learning import EEGNetCustom, BraindecodeTrainer
        model = EEGNetCustom()
        trainer = BraindecodeTrainer(model)
        # trainer.train(...)  # Training code
        
        extractor = DeepFeatureExtractor(model)
        deep_features, labels = extractor.extract_features(dataset)
        
        # Phase 5: RSA analysis
        from src.rsa import RDMComputer, TimeResolvedRSA
        rdm_computer = RDMComputer(metric='correlation')
        
        # Separate imagery and perception
        imagery_idx = labels == 0
        perception_idx = labels == 1
        
        imagery_data = epochs_clean.get_data()[imagery_idx]
        perception_data = epochs_clean.get_data()[perception_idx]
        
        # Time-resolved RSA
        tr_rsa = TimeResolvedRSA(rdm_computer)
        similarity, time_points, p_values = tr_rsa.compute_imagery_perception_similarity(
            imagery_data, perception_data
        )
        
        # Statistical testing
        from src.rsa import RSAStatistics
        stats = RSAStatistics()
        clusters, cluster_masses = stats.cluster_correction_timeseries(
            similarity, p_values
        )
        
        # Save results
        results = {
            'subject_id': subject_id,
            'similarity_timeseries': similarity,
            'time_points': time_points,
            'p_values': p_values,
            'significant_clusters': clusters
        }
        
        np.savez(f'data/results/{subject_id}_rsa_results.npz', **results)
        
        return results
```

### Data Format Standards

```python
# Standardized data formats for inter-phase communication

# 1. After Papercheck: BIDS structure
"""
data/raw/
  sub-01/
    eeg/
      sub-01_task-imagery_eeg.set
      sub-01_task-imagery_eeg.fdt
"""

# 2. After ERPLAB: NPZ format
"""
{
    'data': (n_channels, n_samples, n_epochs),
    'times': (n_samples,),
    'channels': list of str,
    'sfreq': float,
    'events': (n_epochs, 3),  # onset, duration, event_id
    'event_id': dict
}
"""

# 3. After MNE: MNE Epochs object (pickle or .fif)
epochs.save('data/preprocessed/sub-01-epo.fif', overwrite=True)

# 4. After Braindecode: Features array
"""
{
    'features': (n_trials, n_features),
    'labels': (n_trials,),
    'feature_names': list of str
}
"""

# 5. After RSA: Results dictionary
"""
{
    'rdm_imagery': (n_conditions, n_conditions),
    'rdm_perception': (n_conditions, n_conditions),
    'similarity_timeseries': (n_timepoints,),
    'time_points': (n_timepoints,),
    'statistics': dict
}
"""
```

---

## Data Flow Diagram

```
┌──────────────────┐
│   Papercheck     │  Download Wilson dataset from OpenNeuro
│  (Data Acq.)     │  → BIDS-compliant .set files
└────────┬─────────┘
         │
         ├─────────────────────────────────┐
         │                                 │
         ▼                                 ▼
┌──────────────────┐              ┌──────────────────┐
│   ERPLAB v10     │              │   MNE-Python     │
│ (ERP Preproc.)   │              │  (Direct load)   │
│  • Filtering     │              │                  │
│  • Epoching      │              └────────┬─────────┘
│  • Artifact      │                       │
│    detection     │                       │
└────────┬─────────┘                       │
         │                                 │
         │ (.npz)                          │ (.fif)
         ▼                                 │
┌──────────────────┐                       │
│   MNE-Python     │◄──────────────────────┘
│ (Standard Proc.) │
│  • ICA           │
│  • Time-freq     │
│  • QC            │
└────────┬─────────┘
         │
         ├───────────────────┬─────────────┐
         │                   │             │
         ▼                   ▼             ▼
┌──────────────────┐  ┌─────────────┐  ┌────────────────┐
│   Braindecode    │  │  RSA Direct │  │   Searchlight  │
│  (Deep Learning) │  │  Analysis   │  │      RSA       │
│  • EEGNet        │  │             │  │                │
│  • Feature       │  │             │  │                │
│    extraction    │  │             │  │                │
└────────┬─────────┘  └──────┬──────┘  └───────┬────────┘
         │                   │                 │
         │                   │                 │
         └───────────────────┴─────────────────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │  Scipy/Sklearn   │
                   │   (RSA Analysis) │
                   │  • RDM compute   │
                   │  • Statistics    │
                   │  • Visualization │
                   └──────────────────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │     Results      │
                   │  • Figures       │
                   │  • Statistics    │
                   │  • Report        │
                   └──────────────────┘
```

---

## Implementation Guidelines

### 1. Environment Setup

```bash
# Create conda environment with all tools
conda create -n eeg-rsa python=3.10
conda activate eeg-rsa

# Install core packages
pip install mne>=1.5.0
pip install braindecode>=0.8.0
pip install scipy>=1.11.0
pip install scikit-learn>=1.3.0
pip install torch>=2.0.0
pip install papercheck>=0.4.0

# Install MATLAB engine (requires MATLAB installation)
cd /usr/local/MATLAB/R2023b/extern/engines/python
python setup.py install

# Add ERPLAB to MATLAB path (manual step)
# In MATLAB: addpath('/path/to/erplab10.0')
```

### 2. Configuration Management

```yaml
# configs/pipeline_config.yaml

pipeline:
  data_dir: "data/"
  results_dir: "data/results/"
  parallel_jobs: -1  # Use all cores
  
papercheck:
  dataset_id: "ds003825"
  target_dir: "data/raw/"
  
erplab:
  filter:
    highpass: 0.1
    lowpass: 40
  epoch:
    tmin: -0.2
    tmax: 1.0
    baseline: [-0.2, 0]
  artifact:
    threshold: 100  # µV
    
mne:
  ica:
    n_components: 25
    method: 'fastica'
  reference: 'average'
  
braindecode:
  model: 'EEGNet'
  batch_size: 64
  max_epochs: 100
  lr: 0.001
  
rsa:
  metric: 'correlation'
  window_size: 50  # ms
  step_size: 10    # ms
  n_permutations: 10000
```

### 3. Logging and Monitoring

```python
# src/utils/logging_config.py

import logging
from pathlib import Path

def setup_logging(log_dir='logs', level=logging.INFO):
    """Configure logging for pipeline."""
    
    Path(log_dir).mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/pipeline.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)
```

### 4. Error Handling

```python
# Robust error handling across phases

class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass

class DataAcquisitionError(PipelineError):
    """Error during Papercheck download."""
    pass

class PreprocessingError(PipelineError):
    """Error during ERPLAB/MNE preprocessing."""
    pass

class AnalysisError(PipelineError):
    """Error during RSA analysis."""
    pass

# Usage in pipeline
try:
    results = pipeline.run_full_pipeline(subject_id)
except DataAcquisitionError as e:
    logger.error(f"Data acquisition failed: {e}")
    # Retry logic or skip subject
except PreprocessingError as e:
    logger.error(f"Preprocessing failed: {e}")
    # Save intermediate results
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

### 5. Testing Strategy

```python
# tests/test_integration.py

import pytest
from src.pipeline import WilsonPipeline

def test_phase1_papercheck():
    """Test Papercheck download."""
    # Use mock dataset
    pass

def test_phase2_erplab():
    """Test ERPLAB preprocessing."""
    # Use synthetic EEG data
    pass

def test_phase3_mne():
    """Test MNE processing."""
    pass

def test_phase4_braindecode():
    """Test Braindecode training."""
    pass

def test_phase5_rsa():
    """Test RSA computation."""
    pass

def test_full_pipeline():
    """Integration test with sample data."""
    pipeline = WilsonPipeline('configs/test_config.yaml')
    results = pipeline.run_full_pipeline('test-01')
    
    assert 'similarity_timeseries' in results
    assert len(results['time_points']) > 0
```

---

## Performance Considerations

### Computational Resources

| Phase | CPU Cores | RAM | GPU | Time per Subject |
|-------|-----------|-----|-----|------------------|
| Papercheck | 4 | 2 GB | No | 5-10 min (download) |
| ERPLAB | 1 (MATLAB) | 4 GB | No | 2-5 min |
| MNE-Python | 8 | 8 GB | No | 5-10 min |
| Braindecode | 4 | 16 GB | Yes (recommended) | 20-60 min (training) |
| RSA | 16 | 8 GB | No | 5-15 min |

### Optimization Strategies

```python
# 1. Parallel subject processing
from joblib import Parallel, delayed

def process_all_subjects(subject_ids, n_jobs=8):
    """Process multiple subjects in parallel."""
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(pipeline.run_full_pipeline)(subj_id)
        for subj_id in subject_ids
    )
    
    return results

# 2. Memory-efficient data loading
epochs = mne.read_epochs(filename, preload=False)  # Memory-mapped
data = epochs.get_data()[:, :, ::2]  # Downsample on-the-fly

# 3. GPU acceleration (Braindecode)
model = model.to('cuda')
data_loader = DataLoader(dataset, batch_size=128, num_workers=4, pin_memory=True)

# 4. Caching intermediate results
from joblib import Memory
memory = Memory('cache', verbose=0)

@memory.cache
def compute_rdm_cached(data, metric):
    return compute_rdm(data, metric)
```

### Disk Space Requirements

- Raw data: ~500 MB per subject
- Preprocessed data: ~200 MB per subject
- Models: ~50 MB per trained model
- Results: ~100 MB per subject
- **Total**: ~1 GB per subject

---

## Future Enhancements

### 1. Cloud Integration

```python
# AWS S3 integration for large-scale processing
import boto3

class S3DataManager:
    """Manage EEG data on AWS S3."""
    
    def __init__(self, bucket_name):
        self.s3 = boto3.client('s3')
        self.bucket = bucket_name
        
    def upload_results(self, local_file, s3_key):
        """Upload results to S3."""
        self.s3.upload_file(local_file, self.bucket, s3_key)
```

### 2. Real-Time Processing

```python
# Online RSA for real-time neurofeedback
class OnlineRSA:
    """Compute RSA in real-time as data streams in."""
    
    def __init__(self, reference_rdm):
        self.reference_rdm = reference_rdm
        self.buffer = []
        
    def update(self, new_epoch):
        """Update RSA with new epoch."""
        self.buffer.append(new_epoch)
        
        if len(self.buffer) >= 10:  # Minimum epochs
            current_rdm = compute_rdm(np.array(self.buffer))
            similarity = compare_rdms(current_rdm, self.reference_rdm)
            
            return similarity
```

### 3. Advanced Visualization

```python
# Interactive dashboards
import plotly.graph_objects as go

def create_interactive_rdm(rdm, condition_labels):
    """Create interactive RDM visualization."""
    
    fig = go.Figure(data=go.Heatmap(
        z=rdm,
        x=condition_labels,
        y=condition_labels,
        colorscale='Viridis'
    ))
    
    fig.update_layout(title='Representational Dissimilarity Matrix')
    
    return fig
```

### 4. Additional Analysis Methods

- **Dimensionality Reduction**: t-SNE/UMAP visualization of neural representations
- **Granger Causality**: Directed connectivity between brain regions
- **Phase Synchrony**: Functional connectivity via phase-locking value
- **Source Localization**: Project RSA results to source space

---

## Summary

This design document provides a comprehensive technical specification for integrating five major tools (Papercheck, ERPLAB, MNE-Python, Braindecode, Scipy/Sklearn) into a unified EEG-RSA analysis pipeline. Key design principles include:

1. **Modularity**: Each phase is independent with clear interfaces
2. **Flexibility**: Support multiple analysis approaches
3. **Reproducibility**: Version control and deterministic processing
4. **Scalability**: Efficient multi-subject processing
5. **Extensibility**: Easy to add new methods

The pipeline enables comprehensive analysis of the Wilson et al. dataset, from raw data acquisition to publication-ready RSA results, with state-of-the-art preprocessing, deep learning, and statistical methods.

---

**For questions or contributions, please refer to the main README.md or open an issue on GitHub.**
