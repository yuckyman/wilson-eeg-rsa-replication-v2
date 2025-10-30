"""Representational Dissimilarity Matrix (RDM) Computation.

This module provides functions for computing RDMs from neural data features.
RDMs quantify the pairwise dissimilarity between neural response patterns
for different stimuli or conditions.

Dissimilarity Metrics:
- Correlation distance (1 - Pearson correlation)
- Euclidean distance
- Mahalanobis distance
- Cosine distance
- Cross-validated distance estimators

Customization Points:
- Choose appropriate distance metric for your data
- Apply distance correction methods (e.g., cross-validation)
- Implement custom dissimilarity measures
- Handle within-subject vs. across-subject RDMs

Integration:
- RSA Toolbox: Compatible RDM format
- SciPy: Distance computations
- Scikit-learn: Distance metrics and preprocessing

Author: Ian
Date: 2025-10-30
"""

from typing import Dict, List, Optional, Tuple, Union, Callable
import warnings

import numpy as np
from scipy.spatial.distance import pdist, squareform, correlation, euclidean
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


class RDM:
    """Representational Dissimilarity Matrix.
    
    This class encapsulates an RDM and provides methods for
    visualization, comparison, and analysis.
    
    Attributes:
        matrix (np.ndarray): Dissimilarity matrix (n_conditions x n_conditions).
        labels (List[str]): Condition labels.
        metric (str): Distance metric used.
        metadata (Dict): Additional metadata.
    """
    
    def __init__(self,
                 matrix: np.ndarray,
                 labels: Optional[List[str]] = None,
                 metric: str = 'correlation',
                 metadata: Optional[Dict] = None):
        """Initialize RDM.
        
        Args:
            matrix: Square dissimilarity matrix.
            labels: Condition labels for each row/column.
            metric: Name of distance metric used.
            metadata: Additional metadata (subject ID, session, etc.).
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square")
        
        self.matrix = matrix
        self.n_conditions = matrix.shape[0]
        self.labels = labels or [f"Cond_{i+1}" for i in range(self.n_conditions)]
        self.metric = metric
        self.metadata = metadata or {}
        
        # Get upper triangle (excluding diagonal) as vector
        self.vector = self.matrix[np.triu_indices(self.n_conditions, k=1)]
    
    def plot(self,
            title: Optional[str] = None,
            cmap: str = 'viridis',
            figsize: Tuple[int, int] = (8, 7)) -> plt.Figure:
        """Plot RDM as heatmap.
        
        Args:
            title: Plot title.
            cmap: Colormap name.
            figsize: Figure size (width, height).
        
        Returns:
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(self.matrix, cmap=cmap, aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(self.n_conditions))
        ax.set_yticks(np.arange(self.n_conditions))
        ax.set_xticklabels(self.labels, rotation=45, ha='right')
        ax.set_yticklabels(self.labels)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label=f'{self.metric.capitalize()} Distance')
        
        # Set title
        if title is None:
            title = f"RDM ({self.metric})"
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def get_vector(self) -> np.ndarray:
        """Get upper triangle as vector for correlation analysis.
        
        Returns:
            1D array of dissimilarity values.
        """
        return self.vector
    
    def save(self, filepath: str) -> None:
        """Save RDM to file.
        
        Args:
            filepath: Output filepath (.npz format).
        """
        np.savez(filepath,
                matrix=self.matrix,
                labels=self.labels,
                metric=self.metric,
                metadata=self.metadata)
        print(f"Saved RDM to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'RDM':
        """Load RDM from file.
        
        Args:
            filepath: Input filepath (.npz format).
        
        Returns:
            Loaded RDM object.
        """
        data = np.load(filepath, allow_pickle=True)
        return cls(
            matrix=data['matrix'],
            labels=list(data['labels']),
            metric=str(data['metric']),
            metadata=data['metadata'].item() if 'metadata' in data else {}
        )


def compute_rdm(features: np.ndarray,
               metric: str = 'correlation',
               labels: Optional[List[str]] = None,
               **kwargs) -> RDM:
    """Compute RDM from feature matrix.
    
    Args:
        features: Feature matrix of shape (n_conditions, n_features) or
                 (n_conditions, n_trials, n_features) for averaging.
        metric: Distance metric ('correlation', 'euclidean', 'mahalanobis',
               'cosine', 'custom').
        labels: Condition labels.
        **kwargs: Additional arguments for distance computation.
    
    Returns:
        RDM object.
        
    CUSTOMIZATION:
        - For trial-level data, average within conditions first
        - Apply feature selection before computing distances
        - Use cross-validated distance estimators
        - Implement custom distance functions
        
    Example:
        >>> features = np.random.randn(10, 100)  # 10 conditions, 100 features
        >>> rdm = compute_rdm(features, metric='correlation')
        >>> rdm.plot()
    """
    # Handle 3D input (conditions x trials x features)
    if features.ndim == 3:
        print(f"Averaging {features.shape[1]} trials per condition...")
        features = np.mean(features, axis=1)
    
    n_conditions = features.shape[0]
    
    # Compute dissimilarity matrix
    if metric == 'correlation':
        # 1 - Pearson correlation
        distances = pdist(features, metric='correlation')
    elif metric == 'euclidean':
        distances = pdist(features, metric='euclidean')
    elif metric == 'cosine':
        distances = pdist(features, metric='cosine')
    elif metric == 'mahalanobis':
        # Requires covariance matrix
        cov = np.cov(features.T)
        distances = pdist(features, metric='mahalanobis', VI=np.linalg.inv(cov))
    elif metric == 'custom' and 'distance_fn' in kwargs:
        # Use custom distance function
        distance_fn = kwargs['distance_fn']
        distances = pdist(features, metric=distance_fn)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Convert to square matrix
    matrix = squareform(distances)
    
    # Create RDM object
    rdm = RDM(matrix, labels=labels, metric=metric)
    
    print(f"Computed RDM: {n_conditions} conditions, {metric} distance")
    return rdm


def compute_rdm_crossvalidated(features: np.ndarray,
                              labels: np.ndarray,
                              metric: str = 'correlation',
                              n_splits: int = 5) -> RDM:
    """Compute cross-validated RDM.
    
    This method computes distances between average patterns from
    independent data splits to reduce noise bias.
    
    Args:
        features: Feature matrix of shape (n_trials, n_features).
        labels: Condition labels for each trial (length n_trials).
        metric: Distance metric.
        n_splits: Number of cross-validation splits.
    
    Returns:
        Cross-validated RDM.
        
    CUSTOMIZATION:
        - Use stratified splits for imbalanced conditions
        - Apply leave-one-out cross-validation
        - Implement more sophisticated CV schemes
        
    Reference:
        Walther et al. (2016). Reliability of dissimilarity measures
        for multi-voxel pattern analysis. NeuroImage.
    """
    from sklearn.model_selection import KFold
    
    unique_labels = np.unique(labels)
    n_conditions = len(unique_labels)
    
    # Initialize array to store distances
    cv_distances = []
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(features)):
        # Average features within each condition for each split
        train_patterns = np.array([
            np.mean(features[train_idx][labels[train_idx] == label], axis=0)
            for label in unique_labels
        ])
        test_patterns = np.array([
            np.mean(features[test_idx][labels[test_idx] == label], axis=0)
            for label in unique_labels
        ])
        
        # Compute cross-split distances
        fold_distances = []
        for i in range(n_conditions):
            for j in range(i+1, n_conditions):
                if metric == 'correlation':
                    dist = 1 - pearsonr(train_patterns[i], test_patterns[j])[0]
                elif metric == 'euclidean':
                    dist = euclidean(train_patterns[i], test_patterns[j])
                else:
                    raise NotImplementedError(f"Metric {metric} not implemented for CV")
                fold_distances.append(dist)
        
        cv_distances.append(fold_distances)
    
    # Average across folds
    avg_distances = np.mean(cv_distances, axis=0)
    matrix = squareform(avg_distances)
    
    rdm = RDM(matrix, labels=list(unique_labels), metric=f'{metric}_cv')
    print(f"Computed cross-validated RDM: {n_splits} folds")
    return rdm


def compute_rdm_per_subject(features_list: List[np.ndarray],
                           metric: str = 'correlation',
                           labels: Optional[List[str]] = None) -> List[RDM]:
    """Compute RDMs for multiple subjects.
    
    Args:
        features_list: List of feature matrices, one per subject.
        metric: Distance metric.
        labels: Condition labels.
    
    Returns:
        List of RDM objects, one per subject.
        
    Example:
        >>> rdms = compute_rdm_per_subject(
        ...     [subj1_features, subj2_features, subj3_features],
        ...     metric='correlation'
        ... )
    """
    rdms = []
    for i, features in enumerate(features_list):
        rdm = compute_rdm(features, metric=metric, labels=labels)
        rdm.metadata['subject'] = i + 1
        rdms.append(rdm)
    
    print(f"Computed {len(rdms)} subject RDMs")
    return rdms


def average_rdms(rdms: List[RDM]) -> RDM:
    """Average multiple RDMs.
    
    Args:
        rdms: List of RDM objects to average.
    
    Returns:
        Average RDM.
        
    Note:
        RDMs should have the same dimensions and use the same metric.
    """
    # Check consistency
    n_conditions = rdms[0].n_conditions
    metric = rdms[0].metric
    labels = rdms[0].labels
    
    for rdm in rdms[1:]:
        if rdm.n_conditions != n_conditions:
            raise ValueError("RDMs have different dimensions")
        if rdm.metric != metric:
            warnings.warn("RDMs use different metrics")
    
    # Average matrices
    matrices = np.array([rdm.matrix for rdm in rdms])
    avg_matrix = np.mean(matrices, axis=0)
    
    avg_rdm = RDM(avg_matrix, labels=labels, metric=f'{metric}_avg')
    avg_rdm.metadata['n_subjects'] = len(rdms)
    
    print(f"Averaged {len(rdms)} RDMs")
    return avg_rdm


def compare_rdms(rdm1: RDM,
                rdm2: RDM,
                method: str = 'spearman') -> Tuple[float, float]:
    """Compare two RDMs.
    
    Args:
        rdm1: First RDM.
        rdm2: Second RDM (e.g., model prediction).
        method: Comparison method ('spearman', 'pearson', 'kendall').
    
    Returns:
        Tuple of (correlation, p-value).
        
    CUSTOMIZATION:
        - Use rank correlation (Spearman) for non-linear relationships
        - Apply permutation tests for significance
        - Compute confidence intervals via bootstrap
    """
    if rdm1.n_conditions != rdm2.n_conditions:
        raise ValueError("RDMs have different dimensions")
    
    # Get upper triangle vectors
    vec1 = rdm1.get_vector()
    vec2 = rdm2.get_vector()
    
    # Compute correlation
    if method == 'spearman':
        corr, pval = spearmanr(vec1, vec2)
    elif method == 'pearson':
        corr, pval = pearsonr(vec1, vec2)
    elif method == 'kendall':
        from scipy.stats import kendalltau
        corr, pval = kendalltau(vec1, vec2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"{method.capitalize()} correlation: r = {corr:.3f}, p = {pval:.4f}")
    return corr, pval


def create_model_rdm(model_features: np.ndarray,
                    labels: Optional[List[str]] = None,
                    metric: str = 'correlation') -> RDM:
    """Create RDM from model predictions or representations.
    
    This function is useful for creating RDMs from computational models,
    deep learning representations, or theoretical predictions.
    
    Args:
        model_features: Model feature matrix (n_conditions, n_features).
        labels: Condition labels.
        metric: Distance metric.
    
    Returns:
        Model RDM.
        
    CUSTOMIZATION:
        - Extract features from different layers of deep networks
        - Use semantic embeddings (word2vec, BERT, etc.)
        - Implement theoretical models (e.g., category structure)
        
    Example:
        >>> # Create categorical model (within-category = 0, between = 1)
        >>> n_conditions = 10
        >>> categories = np.array([0,0,0,0,0,1,1,1,1,1])  # 2 categories
        >>> model_rdm = np.zeros((n_conditions, n_conditions))
        >>> for i in range(n_conditions):
        ...     for j in range(n_conditions):
        ...         model_rdm[i,j] = float(categories[i] != categories[j])
    """
    rdm = compute_rdm(model_features, metric=metric, labels=labels)
    rdm.metadata['type'] = 'model'
    return rdm


def noise_ceiling(rdms: List[RDM]) -> Tuple[float, float]:
    """Estimate noise ceiling for RSA.
    
    The noise ceiling represents the maximum correlation achievable
    given the noise in the data, estimated from inter-subject correlations.
    
    Args:
        rdms: List of subject RDMs.
    
    Returns:
        Tuple of (lower_bound, upper_bound) for noise ceiling.
        
    Reference:
        Nili et al. (2014). A toolbox for representational similarity
        analysis. PLoS Computational Biology.
    """
    n_subjects = len(rdms)
    
    # Get RDM vectors
    vectors = np.array([rdm.get_vector() for rdm in rdms])
    
    # Lower bound: average correlation of each subject with mean of others
    lower_bound = 0
    for i in range(n_subjects):
        others_mean = np.mean(vectors[np.arange(n_subjects) != i], axis=0)
        corr, _ = spearmanr(vectors[i], others_mean)
        lower_bound += corr
    lower_bound /= n_subjects
    
    # Upper bound: correlation of average with each subject, then averaged
    group_mean = np.mean(vectors, axis=0)
    upper_bound = np.mean([spearmanr(vectors[i], group_mean)[0] 
                          for i in range(n_subjects)])
    
    print(f"Noise ceiling: [{lower_bound:.3f}, {upper_bound:.3f}]")
    return lower_bound, upper_bound


def visualize_rdm_comparison(rdm1: RDM,
                            rdm2: RDM,
                            rdm1_name: str = 'Neural RDM',
                            rdm2_name: str = 'Model RDM') -> plt.Figure:
    """Visualize comparison between two RDMs.
    
    Args:
        rdm1: First RDM (typically neural data).
        rdm2: Second RDM (typically model prediction).
        rdm1_name: Name for first RDM.
        rdm2_name: Name for second RDM.
    
    Returns:
        Matplotlib figure with comparison plots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot first RDM
    im1 = axes[0].imshow(rdm1.matrix, cmap='viridis')
    axes[0].set_title(rdm1_name)
    plt.colorbar(im1, ax=axes[0])
    
    # Plot second RDM
    im2 = axes[1].imshow(rdm2.matrix, cmap='viridis')
    axes[1].set_title(rdm2_name)
    plt.colorbar(im2, ax=axes[1])
    
    # Scatter plot of dissimilarities
    vec1 = rdm1.get_vector()
    vec2 = rdm2.get_vector()
    corr, pval = spearmanr(vec1, vec2)
    
    axes[2].scatter(vec1, vec2, alpha=0.5)
    axes[2].set_xlabel(f'{rdm1_name} Dissimilarity')
    axes[2].set_ylabel(f'{rdm2_name} Dissimilarity')
    axes[2].set_title(f'Correlation: r = {corr:.3f}, p = {pval:.4f}')
    
    # Add regression line
    z = np.polyfit(vec1, vec2, 1)
    p = np.poly1d(z)
    axes[2].plot(vec1, p(vec1), 'r--', alpha=0.8)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example usage
    print("RDM Computation Template")
    print("\nThis script provides RDM computation functions.")
    print("\nUsage example:")
    print("""
    from rdm.compute_dissimilarity import compute_rdm, compare_rdms
    
    # Compute neural RDM
    neural_features = np.random.randn(10, 100)  # 10 conditions, 100 features
    neural_rdm = compute_rdm(neural_features, metric='correlation')
    
    # Create model RDM
    model_features = np.random.randn(10, 50)
    model_rdm = compute_rdm(model_features, metric='correlation')
    
    # Compare RDMs
    corr, pval = compare_rdms(neural_rdm, model_rdm, method='spearman')
    
    # Visualize
    fig = neural_rdm.plot()
    plt.show()
    """)
