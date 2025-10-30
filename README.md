# EEG-RSA Pipeline: Wilson et al. Study Replication (v2)

A comprehensive electroencephalography (EEG) analysis pipeline implementing Representational Similarity Analysis (RSA) to replicate and extend findings from Wilson et al.'s study on mental imagery versus perception.

## Table of Contents

- [Project Overview](#project-overview)
- [Background](#background)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Analysis Pipeline](#analysis-pipeline)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Results and Outputs](#results-and-outputs)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)

## Project Overview

This project provides an end-to-end computational pipeline for analyzing EEG data using Representational Similarity Analysis (RSA) to investigate neural representations during mental imagery compared to perceptual experiences. The pipeline replicates the methodology from Wilson et al.'s seminal work while incorporating modern best practices and extended analytical capabilities.

### Key Objectives

- **Replicate** Wilson et al.'s findings on imagery vs. perception neural signatures
- **Implement** robust EEG preprocessing and artifact rejection procedures
- **Apply** RSA techniques to quantify neural representational similarity
- **Visualize** spatiotemporal dynamics of imagery and perception processes
- **Provide** reproducible and extensible analysis framework

## Background

Representational Similarity Analysis (RSA) is a powerful multivariate technique that characterizes neural representations by comparing patterns of activity across different conditions. This approach allows researchers to:

- Examine how neural representations evolve over time
- Compare representations across different brain regions
- Relate neural patterns to behavioral or computational models
- Investigate shared and distinct neural codes for imagery and perception

The original Wilson et al. study demonstrated that mental imagery and visual perception share overlapping neural representations, particularly in posterior regions associated with visual processing, while also exhibiting distinct temporal dynamics.

## Features

- **Automated EEG Preprocessing Pipeline**
  - Bandpass filtering with customizable frequency ranges
  - Independent Component Analysis (ICA) for artifact removal
  - Bad channel detection and interpolation
  - Re-referencing (average or custom reference)
  - Epoching with flexible time windows

- **Advanced RSA Implementation**
  - Multiple distance metrics (Euclidean, correlation, Mahalanobis)
  - Time-resolved RSA with sliding windows
  - Searchlight analysis for spatial specificity
  - Cross-condition comparisons (imagery vs. perception)

- **Statistical Analysis**
  - Permutation testing for significance assessment
  - Cluster-based correction for multiple comparisons
  - Bootstrap confidence intervals
  - Effect size calculations

- **Comprehensive Visualization**
  - Representational Dissimilarity Matrices (RDMs)
  - Time-series plots of representational similarity
  - Topographic maps showing spatial distributions
  - Statistical overlay visualizations

## Directory Structure

```
wilson-eeg-rsa-replication-v2/
├── data/                       # Data directory (not tracked by git)
│   ├── raw/                   # Raw EEG files
│   ├── preprocessed/          # Preprocessed EEG data
│   ├── behavioral/            # Behavioral data files
│   └── results/               # Analysis outputs
├── src/                       # Source code
│   ├── preprocessing/         # EEG preprocessing modules
│   │   ├── filtering.py
│   │   ├── artifact_removal.py
│   │   ├── epoching.py
│   │   └── bad_channels.py
│   ├── rsa/                   # RSA analysis modules
│   │   ├── distance_metrics.py
│   │   ├── rdm_generation.py
│   │   ├── time_resolved_rsa.py
│   │   └── searchlight.py
│   ├── statistics/            # Statistical testing
│   │   ├── permutation_tests.py
│   │   ├── cluster_correction.py
│   │   └── effect_sizes.py
│   ├── visualization/         # Plotting and visualization
│   │   ├── plot_rdm.py
│   │   ├── plot_timeseries.py
│   │   └── plot_topomaps.py
│   └── utils/                 # Utility functions
│       ├── io.py
│       ├── logging.py
│       └── validators.py
├── notebooks/                 # Jupyter notebooks for analysis
│   ├── 01_preprocessing.ipynb
│   ├── 02_rsa_analysis.ipynb
│   ├── 03_statistics.ipynb
│   └── 04_visualization.ipynb
├── configs/                   # Configuration files
│   ├── preprocessing_config.yaml
│   ├── rsa_config.yaml
│   └── experiment_params.yaml
├── tests/                     # Unit tests
│   ├── test_preprocessing.py
│   ├── test_rsa.py
│   └── test_statistics.py
├── scripts/                   # Executable scripts
│   ├── run_preprocessing.py
│   ├── run_rsa_pipeline.py
│   └── generate_figures.py
├── docs/                      # Documentation
│   ├── methodology.md
│   ├── api_reference.md
│   └── tutorial.md
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment specification
├── setup.py                   # Package installation
├── .gitignore
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- 8GB RAM minimum (16GB recommended)
- Git

### Option 1: pip installation

```bash
# Clone the repository
git clone https://github.com/yuckyman/wilson-eeg-rsa-replication-v2.git
cd wilson-eeg-rsa-replication-v2

# Create virtual environment
python -m venv eeg_rsa_env
source eeg_rsa_env/bin/activate  # On Windows: eeg_rsa_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Option 2: Conda installation

```bash
# Clone the repository
git clone https://github.com/yuckyman/wilson-eeg-rsa-replication-v2.git
cd wilson-eeg-rsa-replication-v2

# Create conda environment
conda env create -f environment.yml
conda activate eeg-rsa

# Install package
pip install -e .
```

## Usage

### Quick Start

```python
from src.preprocessing import preprocess_eeg
from src.rsa import compute_rdm, time_resolved_rsa
from src.visualization import plot_rdm, plot_similarity_timeseries

# 1. Load and preprocess EEG data
eeg_data = preprocess_eeg(
    filepath='data/raw/subject_01.fif',
    highpass=0.1,
    lowpass=40,
    apply_ica=True
)

# 2. Compute Representational Dissimilarity Matrix
rdm_imagery = compute_rdm(
    eeg_data['imagery'],
    metric='correlation',
    time_window=(0, 0.5)
)

rdm_perception = compute_rdm(
    eeg_data['perception'],
    metric='correlation',
    time_window=(0, 0.5)
)

# 3. Time-resolved RSA
similarity_timeseries = time_resolved_rsa(
    rdm_imagery,
    rdm_perception,
    sliding_window=50,
    step_size=10
)

# 4. Visualize results
plot_rdm(rdm_imagery, title='Mental Imagery RDM')
plot_similarity_timeseries(similarity_timeseries)
```

### Running the Complete Pipeline

```bash
# Step 1: Preprocess all subjects
python scripts/run_preprocessing.py --config configs/preprocessing_config.yaml

# Step 2: Run RSA analysis
python scripts/run_rsa_pipeline.py --config configs/rsa_config.yaml --subjects all

# Step 3: Generate figures
python scripts/generate_figures.py --output data/results/figures/
```

### Interactive Analysis with Jupyter

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/

# Follow notebooks in order:
# 1. 01_preprocessing.ipynb
# 2. 02_rsa_analysis.ipynb
# 3. 03_statistics.ipynb
# 4. 04_visualization.ipynb
```

## Methodology

### EEG Preprocessing Pipeline

1. **Data Import and Validation**
   - Load raw EEG data (supports .fif, .set, .bdf formats)
   - Check channel locations and impedances
   - Verify sampling rates and recording parameters

2. **Filtering**
   - High-pass filter: 0.1 Hz (remove slow drifts)
   - Low-pass filter: 40 Hz (anti-aliasing)
   - Notch filter: 50/60 Hz (line noise removal, optional)

3. **Bad Channel Detection**
   - Statistical outlier detection
   - Cross-correlation analysis
   - Manual inspection capability

4. **Re-referencing**
   - Average reference (default)
   - Mastoid reference (optional)
   - Custom reference options

5. **Artifact Removal**
   - Independent Component Analysis (ICA)
   - Automatic component classification
   - Reject ocular, cardiac, and muscle artifacts

6. **Epoching**
   - Extract trials relative to stimulus onset
   - Baseline correction
   - Trial rejection based on amplitude thresholds

7. **Quality Control**
   - Generate preprocessing reports
   - Visualize data quality metrics
   - Track rejected trials and channels

### RSA Implementation

#### Representational Dissimilarity Matrix (RDM) Construction

For each time point t and condition c:

1. Extract neural patterns across electrodes: **X**<sub>c,t</sub> ∈ ℝ<sup>n×e</sup>
   - n = number of trials
   - e = number of electrodes

2. Compute pairwise dissimilarities between trials:
   - **D**<sub>c,t</sub>(i,j) = distance(**x**<sub>i</sub>, **x**<sub>j</sub>)

3. Distance metrics available:
   - **Correlation distance**: 1 - Pearson correlation
   - **Euclidean distance**: ||**x**<sub>i</sub> - **x**<sub>j</sub>||<sub>2</sub>
   - **Mahalanobis distance**: accounts for covariance structure

#### Time-Resolved RSA

1. Compute RDMs in sliding time windows
2. Compare RDMs between imagery and perception conditions
3. Quantify similarity using:
   - Spearman correlation between RDMs
   - Kendall's τ for rank-based comparison
   - Matrix regression coefficients

#### Statistical Testing

1. **Permutation Testing**
   - Shuffle condition labels
   - Generate null distribution (10,000 permutations)
   - Compute p-values for observed effects

2. **Cluster-Based Correction**
   - Identify spatiotemporal clusters
   - Threshold at p < 0.05
   - Correct for multiple comparisons using cluster mass statistic

3. **Effect Size Estimation**
   - Cohen's d for between-condition comparisons
   - Bootstrap confidence intervals (95%)

### Key Hypotheses Tested

1. **H1**: Mental imagery and perception share representational structure
   - Prediction: Positive correlation between imagery and perception RDMs
   
2. **H2**: Temporal dynamics differ between imagery and perception
   - Prediction: Imagery shows delayed onset compared to perception
   
3. **H3**: Representational similarity is strongest in posterior electrodes
   - Prediction: Occipital and parietal regions show highest correlations

## Analysis Pipeline

### Stage 1: Data Quality Assessment

- Inspect raw data for artifacts
- Assess electrode impedances
- Check for common recording issues

### Stage 2: Preprocessing

- Apply temporal filters
- Remove artifacts
- Epoch data
- Perform quality control

### Stage 3: RSA Computation

- Generate RDMs for each condition
- Compute time-resolved similarity
- Apply searchlight analysis (optional)

### Stage 4: Statistical Analysis

- Run permutation tests
- Apply multiple comparison corrections
- Calculate effect sizes

### Stage 5: Visualization and Reporting

- Generate publication-quality figures
- Create summary statistics tables
- Compile analysis report

## Dependencies

### Core Libraries

- **mne** (≥1.5.0): EEG/MEG data processing
- **numpy** (≥1.24.0): Numerical computations
- **scipy** (≥1.11.0): Scientific computing and statistics
- **pandas** (≥2.0.0): Data manipulation
- **matplotlib** (≥3.7.0): Plotting
- **seaborn** (≥0.12.0): Statistical visualization

### Additional Packages

- **scikit-learn** (≥1.3.0): Machine learning utilities
- **nilearn** (≥0.10.0): Neuroimaging analysis
- **joblib** (≥1.3.0): Parallel processing
- **pyyaml** (≥6.0): Configuration file parsing
- **tqdm** (≥4.65.0): Progress bars
- **pytest** (≥7.4.0): Testing framework

## Configuration

Configuration files are stored in `configs/` directory in YAML format.

### Preprocessing Configuration (`preprocessing_config.yaml`)

```yaml
filtering:
  highpass: 0.1
  lowpass: 40
  notch: [50, 60]

ica:
  n_components: 25
  method: 'fastica'
  random_state: 42

epoching:
  tmin: -0.2
  tmax: 1.0
  baseline: [-0.2, 0]
  reject:
    eeg: 100e-6  # 100 µV

reference:
  type: 'average'
```

### RSA Configuration (`rsa_config.yaml`)

```yaml
rdm:
  metric: 'correlation'
  cv_folds: 5

time_resolved:
  window_size: 50  # ms
  step_size: 10    # ms
  
statistics:
  n_permutations: 10000
  alpha: 0.05
  correction_method: 'cluster'
```

## Results and Outputs

### Generated Files

- **Preprocessed Data**: `data/preprocessed/sub-XX_preprocessed.fif`
- **RDM Matrices**: `data/results/rdms/sub-XX_condition-{imagery,perception}_rdm.npy`
- **Statistical Results**: `data/results/stats/group_statistics.csv`
- **Figures**: `data/results/figures/*.png`
- **Analysis Report**: `data/results/analysis_report.html`

### Expected Findings

Based on Wilson et al.'s original study, expect to observe:

1. **Shared Representations**: Significant positive correlation between imagery and perception RDMs (r ≈ 0.6-0.8)
2. **Temporal Dynamics**: Perception peaks around 100-150ms; imagery peaks around 200-300ms
3. **Spatial Distribution**: Strongest effects in occipital (O1, O2, Oz) and parietal (P3, P4, Pz) regions
4. **Individual Differences**: Variability in imagery strength correlates with behavioral vividness ratings

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{wilson_eeg_rsa_v2,
  author = {Ian},
  title = {EEG-RSA Pipeline: Wilson et al. Study Replication v2},
  year = {2025},
  url = {https://github.com/yuckyman/wilson-eeg-rsa-replication-v2},
  version = {2.0}
}
```

And cite the original Wilson et al. study:

```bibtex
@article{wilson_mental_imagery,
  author = {Wilson, K.D. and others},
  title = {Mental imagery and visual perception share representational structure},
  journal = {Journal of Cognitive Neuroscience},
  year = {20XX},
  volume = {XX},
  pages = {XXX-XXX}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Add tests** for new functionality
5. **Ensure all tests pass** (`pytest tests/`)
6. **Commit your changes** (`git commit -m 'Add amazing feature'`)
7. **Push to the branch** (`git push origin feature/amazing-feature`)
8. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Check code style
flake8 src/
black src/ --check

# Generate documentation
cd docs/
make html
```

## Acknowledgments

- Original methodology by Wilson et al.
- MNE-Python developers for excellent EEG analysis tools
- RSA Toolbox contributors
- OpenNeuro for providing open EEG datasets

## Contact

For questions, issues, or collaboration inquiries:

- **GitHub Issues**: [Create an issue](https://github.com/yuckyman/wilson-eeg-rsa-replication-v2/issues)
- **Maintainer**: ian

---

**Last Updated**: October 30, 2025

*This pipeline is under active development. Check back for updates and new features!*
