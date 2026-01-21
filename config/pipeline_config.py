"""Pipeline Configuration

Central configuration file for the Wilson EEG RSA replication pipeline.
Defines parameters for preprocessing, feature extraction, RDM construction, and analysis.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'

# Preprocessing parameters
PREPROCESSING = {
    'sampling_rate': 500,  # Hz
    'bandpass_low': 0.1,   # Hz
    'bandpass_high': 40,    # Hz
    'notch_freq': 60,       # Hz (power line frequency)
    'epoch_tmin': -0.2,     # seconds
    'epoch_tmax': 0.8,      # seconds
    'baseline': (-0.2, 0),  # seconds
    'reject_threshold': 100e-6,  # V (100 µV)
}

# Feature extraction parameters
FEATURE_EXTRACTION = {
    'method': 'erp',  # Options: 'erp', 'power', 'time_frequency'
    'time_windows': [(0.08, 0.12), (0.15, 0.18), (0.3, 0.5)],  # N1, P1, P3 windows
    'frequency_bands': {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40),
    },
}

# Representation method parameters for RSA analysis
REPRESENTATION_METHODS = {
    'methods': [
        'power_bands',
        'channels',
        'channel_x_band',
        'time_windows',
        'erp_features',
        'time_frequency',
    ],
    'frequency_bands': {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40),
    },
    'time_windows': [
        (0.0, 0.3),   # Early window
        (0.3, 0.6),   # Late window
    ],
    'erp_time_window': (0.0, 0.6),  # Full epoch for ERP features
    'time_frequency': {
        'freqs': None,  # None = auto (2-40 Hz in 2 Hz steps)
        'time_window': None,  # None = all time
        'average_time': True,
    },
}

# RDM parameters
RDM_CONFIG = {
    'distance_metric': 'correlation',  # Options: 'correlation', 'euclidean', 'mahalanobis'
    'normalization': True,
}

# Analysis parameters
ANALYSIS = {
    'n_permutations': 10000,
    'n_bootstrap': 10000,
    'alpha': 0.05,
    'correction_method': 'fdr_bh',  # False discovery rate
}

# Visualization parameters
VIZUALIZATION = {
    'dpi': 300,
    'figure_format': 'png',
    'colormap': 'viridis',
}

# Random seed for reproducibility
RANDOM_SEED = 42
