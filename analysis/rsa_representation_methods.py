"""RSA Analysis with Multiple Representation Methods.

This module provides functions for testing imagery vs perception similarity
across different representation encoding methods to assess robustness.

Analysis Components:
- Collect representations for each method across subjects/conditions
- Compute RDMs for imagery and perception conditions separately
- Compare imagery vs perception RDMs using Spearman correlation
- Test significance with permutation tests
- Compare results across representation methods

Author: Ian
Date: 2025-01-XX
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import mne
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

from feature_extraction.extract_representations import (
    extract_representation_vector,
    get_representation_methods
)
from rdm.compute_dissimilarity import compute_rdm, RDM, compare_rdms


def collect_representations_by_condition(epochs,
                                        method: str,
                                        condition_key: str = 'condition',
                                        **kwargs) -> Dict[str, np.ndarray]:
    """Collect representations for each condition separately.
    
    Args:
        epochs: MNE epochs object with condition metadata.
        method: Representation method name.
        condition_key: Key in epochs.metadata to identify conditions.
                      Alternatively, can use event_id if condition_key is None.
        **kwargs: Additional arguments for extract_representation_vector.
    
    Returns:
        Dictionary mapping condition names to representation arrays.
        Each array has shape (n_epochs_in_condition, n_features).
    """
    # Get condition labels
    if condition_key and hasattr(epochs, 'metadata') and epochs.metadata is not None:
        if condition_key in epochs.metadata.columns:
            conditions = epochs.metadata[condition_key].values
            unique_conditions = np.unique(conditions)
        else:
            # Fall back to event_id
            unique_conditions = list(epochs.event_id.keys())
            conditions = np.array([
                epochs.event_id[epochs.events[i, 2]] 
                for i in range(len(epochs))
            ])
    else:
        # Use event_id
        unique_conditions = list(epochs.event_id.keys())
        conditions = np.array([
            list(epochs.event_id.keys())[list(epochs.event_id.values()).index(epochs.events[i, 2])]
            for i in range(len(epochs))
        ])
    
    representations = {}
    
    for condition in unique_conditions:
        # Select epochs for this condition
        if condition_key and hasattr(epochs, 'metadata') and epochs.metadata is not None:
            if condition_key in epochs.metadata.columns:
                cond_mask = epochs.metadata[condition_key] == condition
                cond_epochs = epochs[cond_mask]
            else:
                # Use event_id
                cond_epochs = epochs[condition]
        else:
            cond_epochs = epochs[condition]
        
        # Extract representations
        reprs = extract_representation_vector(cond_epochs, method, **kwargs)
        representations[condition] = reprs
    
    return representations


def compute_rdms_per_method(epochs,
                           methods: Optional[List[str]] = None,
                           condition_key: str = 'condition',
                           metric: str = 'correlation',
                           **kwargs) -> Dict[str, Dict[str, RDM]]:
    """Compute RDMs for each representation method and condition.
    
    Args:
        epochs: MNE epochs object.
        methods: List of representation methods to test. If None, uses all methods.
        condition_key: Key to identify conditions in epochs.
        metric: Distance metric for RDM computation.
        **kwargs: Additional arguments for representation extraction.
    
    Returns:
        Nested dictionary: {method: {condition: RDM}}
    """
    import mne
    
    if methods is None:
        methods = get_representation_methods()
    
    all_rdms = {}
    
    for method in methods:
        print(f"\nComputing RDMs for method: {method}")
        
        # Collect representations by condition
        reprs_by_cond = collect_representations_by_condition(
            epochs, method, condition_key=condition_key, **kwargs
        )
        
        # Compute RDM for each condition
        method_rdms = {}
        for condition, reprs in reprs_by_cond.items():
            # Average across epochs within condition
            # Shape: (n_epochs, n_features) -> (n_features,)
            condition_mean = np.mean(reprs, axis=0)
            
            # For RDM, we need multiple conditions. If only one condition,
            # we need to handle this differently (e.g., use trial-level RDMs)
            # For now, assume we have multiple conditions
            if len(reprs_by_cond) > 1:
                # Stack all condition means: (n_conditions, n_features)
                all_means = np.array([np.mean(reprs_by_cond[c], axis=0) 
                                     for c in sorted(reprs_by_cond.keys())])
                
                # Compute RDM
                rdm = compute_rdm(all_means, metric=metric, 
                                 labels=list(sorted(reprs_by_cond.keys())))
                method_rdms[condition] = rdm
            else:
                # Single condition - create placeholder or use trial-level
                warnings.warn(f"Only one condition found for {method}. "
                            "Cannot compute RDM. Skipping.")
                continue
        
        all_rdms[method] = method_rdms
    
    return all_rdms


def compute_imagery_perception_rdms(epochs_imagery,
                                   epochs_perception,
                                   method: str,
                                   metric: str = 'correlation',
                                   **kwargs) -> Tuple[RDM, RDM]:
    """Compute separate RDMs for imagery and perception conditions.
    
    This function assumes imagery and perception epochs have the same
    condition structure (e.g., same stimuli/categories).
    
    Args:
        epochs_imagery: Epochs from imagery condition.
        epochs_perception: Epochs from perception condition.
        method: Representation method name.
        metric: Distance metric for RDM computation.
        **kwargs: Additional arguments for representation extraction.
    
    Returns:
        Tuple of (imagery_rdm, perception_rdm).
    """
    # Extract representations for imagery
    reprs_imagery = extract_representation_vector(epochs_imagery, method, **kwargs)
    
    # Extract representations for perception
    reprs_perception = extract_representation_vector(epochs_perception, method, **kwargs)
    
    # Get condition labels (assuming same structure)
    # Try to get from event_id or metadata
    if hasattr(epochs_imagery, 'event_id') and epochs_imagery.event_id:
        conditions = sorted(epochs_imagery.event_id.keys())
    else:
        # Infer from number of unique event codes
        unique_events = np.unique(epochs_imagery.events[:, 2])
        conditions = [f'condition_{i+1}' for i in range(len(unique_events))]
    
    # Average within each condition for imagery
    imagery_means = []
    for i, cond in enumerate(conditions):
        if hasattr(epochs_imagery, 'event_id') and epochs_imagery.event_id:
            cond_epochs = epochs_imagery[cond]
        else:
            # Use event code
            event_code = unique_events[i]
            cond_mask = epochs_imagery.events[:, 2] == event_code
            cond_epochs = epochs_imagery[cond_mask]
        
        cond_reprs = extract_representation_vector(cond_epochs, method, **kwargs)
        imagery_means.append(np.mean(cond_reprs, axis=0))
    
    # Average within each condition for perception
    perception_means = []
    for i, cond in enumerate(conditions):
        if hasattr(epochs_perception, 'event_id') and epochs_perception.event_id:
            cond_epochs = epochs_perception[cond]
        else:
            # Use event code
            event_code = unique_events[i]
            cond_mask = epochs_perception.events[:, 2] == event_code
            cond_epochs = epochs_perception[cond_mask]
        
        cond_reprs = extract_representation_vector(cond_epochs, method, **kwargs)
        perception_means.append(np.mean(cond_reprs, axis=0))
    
    # Compute RDMs
    imagery_means = np.array(imagery_means)
    perception_means = np.array(perception_means)
    
    imagery_rdm = compute_rdm(imagery_means, metric=metric, labels=conditions)
    perception_rdm = compute_rdm(perception_means, metric=metric, labels=conditions)
    
    return imagery_rdm, perception_rdm


def compare_imagery_perception(imagery_rdm: RDM,
                              perception_rdm: RDM,
                              method: str = 'spearman') -> Tuple[float, float]:
    """Compare imagery and perception RDMs.
    
    Args:
        imagery_rdm: RDM from imagery condition.
        perception_rdm: RDM from perception condition.
        method: Correlation method ('spearman' or 'pearson').
    
    Returns:
        Tuple of (correlation, p_value).
    """
    return compare_rdms(imagery_rdm, perception_rdm, method=method)


def permutation_test_imagery_perception(imagery_rdm: RDM,
                                       perception_rdm: RDM,
                                       n_permutations: int = 10000,
                                       method: str = 'spearman') -> Dict:
    """Permutation test for imagery-perception RDM similarity.
    
    Args:
        imagery_rdm: RDM from imagery condition.
        perception_rdm: RDM from perception condition.
        n_permutations: Number of permutations.
        method: Correlation method.
    
    Returns:
        Dictionary with test results including observed correlation,
        p-value, and null distribution.
    """
    # Observed correlation
    observed_corr, _ = compare_imagery_perception(imagery_rdm, perception_rdm, method)
    
    # Permutation test: permute condition labels in one RDM
    n_conditions = imagery_rdm.n_conditions
    null_corrs = []
    
    for _ in range(n_permutations):
        # Permute condition labels
        perm_idx = np.random.permutation(n_conditions)
        perm_matrix = perception_rdm.matrix[perm_idx][:, perm_idx]
        
        # Create temporary RDM with permuted matrix
        perm_rdm = RDM(perm_matrix, labels=perception_rdm.labels, 
                      metric=perception_rdm.metric)
        
        # Compute correlation
        perm_corr, _ = compare_imagery_perception(imagery_rdm, perm_rdm, method)
        null_corrs.append(perm_corr)
    
    null_corrs = np.array(null_corrs)
    
    # Compute p-value (two-tailed)
    p_value = np.mean(np.abs(null_corrs) >= np.abs(observed_corr))
    
    return {
        'observed_correlation': observed_corr,
        'p_value': p_value,
        'null_distribution': null_corrs,
        'null_mean': np.mean(null_corrs),
        'null_std': np.std(null_corrs),
        'n_permutations': n_permutations
    }


def test_all_methods(epochs_imagery,
                    epochs_perception,
                    methods: Optional[List[str]] = None,
                    n_permutations: int = 10000,
                    metric: str = 'correlation',
                    **kwargs) -> Dict[str, Dict]:
    """Test imagery-perception similarity across all representation methods.
    
    Args:
        epochs_imagery: Epochs from imagery condition.
        epochs_perception: Epochs from perception condition.
        methods: List of methods to test. If None, uses all methods.
        n_permutations: Number of permutations for significance testing.
        metric: Distance metric for RDM computation.
        **kwargs: Additional arguments for representation extraction.
    
    Returns:
        Dictionary mapping method names to results dictionaries.
        Each result dict contains:
        - 'imagery_rdm': RDM object
        - 'perception_rdm': RDM object
        - 'correlation': Correlation between RDMs
        - 'p_value': P-value from permutation test
        - 'permutation_results': Full permutation test results
    """
    import mne
    
    if methods is None:
        methods = get_representation_methods()
    
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Testing method: {method}")
        print(f"{'='*60}")
        
        try:
            # Compute RDMs
            imagery_rdm, perception_rdm = compute_imagery_perception_rdms(
                epochs_imagery, epochs_perception, method, metric=metric, **kwargs
            )
            
            # Compare RDMs
            corr, p_corr = compare_imagery_perception(imagery_rdm, perception_rdm)
            
            # Permutation test
            perm_results = permutation_test_imagery_perception(
                imagery_rdm, perception_rdm, n_permutations=n_permutations
            )
            
            results[method] = {
                'imagery_rdm': imagery_rdm,
                'perception_rdm': perception_rdm,
                'correlation': corr,
                'p_value_correlation': p_corr,
                'permutation_results': perm_results,
                'p_value_permutation': perm_results['p_value']
            }
            
            print(f"Correlation: {corr:.4f}")
            print(f"P-value (permutation): {perm_results['p_value']:.4f}")
            
        except Exception as e:
            warnings.warn(f"Error testing method {method}: {e}")
            results[method] = {'error': str(e)}
    
    return results


def plot_method_comparison(results: Dict[str, Dict],
                          figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """Plot comparison of imagery-perception similarity across methods.
    
    Args:
        results: Results dictionary from test_all_methods().
        figsize: Figure size.
    
    Returns:
        Matplotlib figure.
    """
    # Extract correlations and p-values
    methods = []
    correlations = []
    p_values = []
    
    for method, result in results.items():
        if 'error' not in result:
            methods.append(method)
            correlations.append(result['correlation'])
            p_values.append(result['p_value_permutation'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Bar plot of correlations
    ax = axes[0, 0]
    colors = ['green' if p < 0.05 else 'gray' for p in p_values]
    bars = ax.bar(methods, correlations, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Imagery-Perception Correlation', fontsize=12)
    ax.set_xlabel('Representation Method', fontsize=12)
    ax.set_title('Imagery-Perception Similarity by Method', fontsize=14, fontweight='bold')
    ax.set_xticklabels(methods, rotation=45, ha='right')
    
    # Add significance stars
    for bar, p in zip(bars, p_values):
        height = bar.get_height()
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                sig, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Scatter plot: correlation vs p-value
    ax = axes[0, 1]
    ax.scatter(correlations, p_values, s=100, alpha=0.7, edgecolors='black')
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='p=0.05')
    ax.set_xlabel('Correlation', fontsize=12)
    ax.set_ylabel('P-value', fontsize=12)
    ax.set_title('Correlation vs Significance', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    
    # Add method labels
    for method, corr, p in zip(methods, correlations, p_values):
        ax.annotate(method, (corr, p), fontsize=8, alpha=0.7)
    
    # 3. Example RDM comparison (first method)
    ax = axes[1, 0]
    first_method = methods[0]
    if 'imagery_rdm' in results[first_method]:
        imagery_rdm = results[first_method]['imagery_rdm']
        im = ax.imshow(imagery_rdm.matrix, cmap='viridis', aspect='auto')
        ax.set_title(f'Imagery RDM: {first_method}', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Dissimilarity')
    
    # 4. Example RDM comparison (perception)
    ax = axes[1, 1]
    if 'perception_rdm' in results[first_method]:
        perception_rdm = results[first_method]['perception_rdm']
        im = ax.imshow(perception_rdm.matrix, cmap='viridis', aspect='auto')
        ax.set_title(f'Perception RDM: {first_method}', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Dissimilarity')
    
    plt.tight_layout()
    return fig


def plot_rdm_comparison(imagery_rdm: RDM,
                       perception_rdm: RDM,
                       method_name: str = '',
                       figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """Plot side-by-side comparison of imagery and perception RDMs.
    
    Args:
        imagery_rdm: RDM from imagery condition.
        perception_rdm: RDM from perception condition.
        method_name: Name of representation method (for title).
        figsize: Figure size.
    
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot imagery RDM
    im1 = axes[0].imshow(imagery_rdm.matrix, cmap='viridis', aspect='auto')
    axes[0].set_title(f'Imagery RDM\n{method_name}', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='Dissimilarity')
    
    # Plot perception RDM
    im2 = axes[1].imshow(perception_rdm.matrix, cmap='viridis', aspect='auto')
    axes[1].set_title(f'Perception RDM\n{method_name}', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='Dissimilarity')
    
    # Scatter plot of dissimilarities
    vec1 = imagery_rdm.get_vector()
    vec2 = perception_rdm.get_vector()
    corr, pval = spearmanr(vec1, vec2)
    
    axes[2].scatter(vec1, vec2, alpha=0.6, s=50)
    axes[2].set_xlabel('Imagery Dissimilarity', fontsize=12)
    axes[2].set_ylabel('Perception Dissimilarity', fontsize=12)
    axes[2].set_title(f'RDM Comparison\nr={corr:.3f}, p={pval:.4f}', 
                     fontsize=12, fontweight='bold')
    
    # Add regression line
    z = np.polyfit(vec1, vec2, 1)
    p = np.poly1d(z)
    axes[2].plot(vec1, p(vec1), 'r--', alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("RSA Representation Methods Analysis")
    print("\nThis module provides functions for testing imagery vs perception")
    print("similarity across different representation encoding methods.")
    print("\nUsage example:")
    print("""
    from analysis.rsa_representation_methods import test_all_methods
    import mne
    
    # Load epochs
    epochs_imagery = mne.read_epochs('data/preprocessed/imagery-epo.fif')
    epochs_perception = mne.read_epochs('data/preprocessed/perception-epo.fif')
    
    # Test all methods
    results = test_all_methods(epochs_imagery, epochs_perception)
    
    # Plot comparison
    from analysis.rsa_representation_methods import plot_method_comparison
    fig = plot_method_comparison(results)
    plt.show()
    """)

