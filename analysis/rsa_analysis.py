"""Representational Similarity Analysis (RSA).

This module provides the main RSA analysis functions for comparing
neural representations with model predictions.

Analysis Components:
- RDM computation and comparison
- Statistical testing (permutation, bootstrap)
- Searchlight analysis (spatial/temporal)
- Model comparison and selection
- Group-level inference

Customization Points:
- Implement custom statistical tests
- Add multiple comparison corrections
- Extend to multivariate pattern analysis (MVPA)
- Integrate with representational connectivity analysis

Integration:
- Compatible with standard RSA toolboxes
- Uses MNE-Python data structures
- Integrates with scikit-learn for ML analyses

Author: Ian
Date: 2025-10-30
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from scipy.stats import spearmanr, pearsonr, ttest_1samp, ttest_rel
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from mne.stats import permutation_cluster_1samp_test, fdr_correction
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    warnings.warn("MNE-Python not available for advanced statistics")


class RSAAnalysis:
    """Main RSA Analysis class.
    
    This class implements a complete RSA workflow including RDM computation,
    model comparison, statistical testing, and visualization.
    
    Attributes:
        neural_rdms (List): List of neural RDMs (one per subject).
        model_rdms (Dict): Dictionary of model RDMs.
        results (Dict): Analysis results.
    """
    
    def __init__(self):
        """Initialize RSA analysis."""
        self.neural_rdms = []
        self.model_rdms = {}
        self.results = {}
    
    def add_neural_rdm(self, rdm, subject_id: Optional[str] = None) -> None:
        """Add neural RDM from a subject.
        
        Args:
            rdm: RDM object from neural data.
            subject_id: Subject identifier.
        """
        if subject_id:
            rdm.metadata['subject_id'] = subject_id
        self.neural_rdms.append(rdm)
        print(f"Added neural RDM (total: {len(self.neural_rdms)})")
    
    def add_model_rdm(self, rdm, model_name: str) -> None:
        """Add model RDM.
        
        Args:
            rdm: RDM object from model.
            model_name: Name/identifier for the model.
        """
        self.model_rdms[model_name] = rdm
        print(f"Added model RDM: {model_name}")
    
    def compare_single_model(self,
                           model_name: str,
                           method: str = 'spearman') -> Dict[str, np.ndarray]:
        """Compare neural RDMs with a single model.
        
        Args:
            model_name: Name of model to compare.
            method: Correlation method ('spearman', 'pearson').
        
        Returns:
            Dictionary with correlation values and p-values per subject.
            
        CUSTOMIZATION:
            - Use partial correlation to control for confounds
            - Apply Fisher z-transformation for averaging
            - Implement rank-based methods for robustness
        """
        if model_name not in self.model_rdms:
            raise ValueError(f"Model {model_name} not found")
        
        model_rdm = self.model_rdms[model_name]
        model_vec = model_rdm.get_vector()
        
        correlations = []
        pvalues = []
        
        for neural_rdm in self.neural_rdms:
            neural_vec = neural_rdm.get_vector()
            
            if method == 'spearman':
                corr, pval = spearmanr(neural_vec, model_vec)
            elif method == 'pearson':
                corr, pval = pearsonr(neural_vec, model_vec)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            correlations.append(corr)
            pvalues.append(pval)
        
        correlations = np.array(correlations)
        pvalues = np.array(pvalues)
        
        # Store results
        self.results[model_name] = {
            'correlations': correlations,
            'pvalues': pvalues,
            'mean_correlation': np.mean(correlations),
            'sem_correlation': np.std(correlations) / np.sqrt(len(correlations))
        }
        
        print(f"\nModel: {model_name}")
        print(f"Mean correlation: {np.mean(correlations):.3f} ± {np.std(correlations)/np.sqrt(len(correlations)):.3f}")
        print(f"Individual subjects: {correlations}")
        
        return self.results[model_name]
    
    def compare_all_models(self, method: str = 'spearman') -> pd.DataFrame:
        """Compare neural RDMs with all models.
        
        Args:
            method: Correlation method.
        
        Returns:
            DataFrame with results for all models.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for this function")
        
        results_list = []
        
        for model_name in self.model_rdms.keys():
            result = self.compare_single_model(model_name, method)
            
            results_list.append({
                'model': model_name,
                'mean_r': result['mean_correlation'],
                'sem_r': result['sem_correlation'],
                'n_subjects': len(result['correlations'])
            })
        
        return pd.DataFrame(results_list)
    
    def test_significance(self,
                         model_name: str,
                         test: str = 'ttest',
                         n_permutations: int = 10000) -> Dict:
        """Test statistical significance of model fit.
        
        Args:
            model_name: Name of model to test.
            test: Statistical test ('ttest', 'permutation', 'bootstrap').
            n_permutations: Number of permutations for permutation test.
        
        Returns:
            Dictionary with test results.
            
        CUSTOMIZATION:
            - Use one-sample t-test against zero
            - Apply permutation test for non-parametric inference
            - Compute confidence intervals via bootstrap
        """
        if model_name not in self.results:
            self.compare_single_model(model_name)
        
        correlations = self.results[model_name]['correlations']
        
        if test == 'ttest':
            # One-sample t-test against 0
            t_stat, p_value = ttest_1samp(correlations, 0)
            
            result = {
                'test': 'one-sample t-test',
                'statistic': t_stat,
                'p_value': p_value,
                'df': len(correlations) - 1
            }
            
        elif test == 'permutation':
            # Permutation test
            result = self._permutation_test(
                self.neural_rdms,
                self.model_rdms[model_name],
                n_permutations
            )
            
        elif test == 'bootstrap':
            # Bootstrap confidence interval
            result = self._bootstrap_ci(correlations)
            
        else:
            raise ValueError(f"Unknown test: {test}")
        
        self.results[model_name]['significance'] = result
        
        print(f"\nSignificance test for {model_name}:")
        print(f"Test: {result['test']}")
        print(f"p-value: {result['p_value']:.4f}")
        
        return result
    
    def _permutation_test(self,
                         neural_rdms: List,
                         model_rdm,
                         n_permutations: int = 10000) -> Dict:
        """Perform permutation test.
        
        Args:
            neural_rdms: List of neural RDMs.
            model_rdm: Model RDM.
            n_permutations: Number of permutations.
        
        Returns:
            Dictionary with test results.
        """
        # Compute observed correlations
        model_vec = model_rdm.get_vector()
        observed = np.array([
            spearmanr(rdm.get_vector(), model_vec)[0]
            for rdm in neural_rdms
        ])
        observed_mean = np.mean(observed)
        
        # Permutation distribution
        null_distribution = []
        
        for _ in range(n_permutations):
            # Permute model RDM
            n_conditions = model_rdm.n_conditions
            perm_idx = np.random.permutation(n_conditions)
            perm_matrix = model_rdm.matrix[perm_idx][:, perm_idx]
            perm_vec = perm_matrix[np.triu_indices(n_conditions, k=1)]
            
            # Compute correlations
            perm_corrs = np.array([
                spearmanr(rdm.get_vector(), perm_vec)[0]
                for rdm in neural_rdms
            ])
            null_distribution.append(np.mean(perm_corrs))
        
        null_distribution = np.array(null_distribution)
        
        # Compute p-value
        p_value = np.mean(null_distribution >= observed_mean)
        
        return {
            'test': 'permutation test',
            'observed': observed_mean,
            'null_mean': np.mean(null_distribution),
            'null_std': np.std(null_distribution),
            'p_value': p_value,
            'n_permutations': n_permutations
        }
    
    def _bootstrap_ci(self,
                     correlations: np.ndarray,
                     n_bootstrap: int = 10000,
                     alpha: float = 0.05) -> Dict:
        """Compute bootstrap confidence interval.
        
        Args:
            correlations: Array of correlation values.
            n_bootstrap: Number of bootstrap samples.
            alpha: Significance level.
        
        Returns:
            Dictionary with confidence interval.
        """
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(correlations, size=len(correlations), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Compute confidence interval
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return {
            'test': 'bootstrap CI',
            'mean': np.mean(correlations),
            'ci_lower': lower,
            'ci_upper': upper,
            'alpha': alpha,
            'p_value': np.mean(bootstrap_means <= 0) * 2  # Two-tailed
        }
    
    def compare_models(self,
                      model1_name: str,
                      model2_name: str,
                      test: str = 'paired_ttest') -> Dict:
        """Compare two models statistically.
        
        Args:
            model1_name: Name of first model.
            model2_name: Name of second model.
            test: Statistical test ('paired_ttest', 'wilcoxon').
        
        Returns:
            Dictionary with comparison results.
            
        CUSTOMIZATION:
            - Use Williams' test for dependent correlations
            - Apply Steiger's test for correlation differences
            - Implement model selection criteria (AIC, BIC)
        """
        if model1_name not in self.results:
            self.compare_single_model(model1_name)
        if model2_name not in self.results:
            self.compare_single_model(model2_name)
        
        corr1 = self.results[model1_name]['correlations']
        corr2 = self.results[model2_name]['correlations']
        
        if test == 'paired_ttest':
            t_stat, p_value = ttest_rel(corr1, corr2)
            result = {
                'test': 'paired t-test',
                'statistic': t_stat,
                'p_value': p_value,
                'mean_diff': np.mean(corr1 - corr2)
            }
        elif test == 'wilcoxon':
            from scipy.stats import wilcoxon
            stat, p_value = wilcoxon(corr1, corr2)
            result = {
                'test': 'Wilcoxon signed-rank',
                'statistic': stat,
                'p_value': p_value,
                'median_diff': np.median(corr1 - corr2)
            }
        else:
            raise ValueError(f"Unknown test: {test}")
        
        print(f"\nComparing {model1_name} vs {model2_name}:")
        print(f"Test: {result['test']}")
        print(f"p-value: {result['p_value']:.4f}")
        
        return result
    
    def searchlight_analysis(self,
                           features: np.ndarray,
                           labels: np.ndarray,
                           model_rdm,
                           window_size: int = 10,
                           step_size: int = 1) -> np.ndarray:
        """Perform temporal searchlight RSA.
        
        This method computes RSA correlation in sliding time windows.
        
        Args:
            features: Feature array (n_trials, n_channels, n_times).
            labels: Condition labels for each trial.
            model_rdm: Model RDM to compare against.
            window_size: Size of time window in samples.
            step_size: Step size for sliding window.
        
        Returns:
            Array of correlations over time.
            
        CUSTOMIZATION:
            - Extend to spatial searchlight (across channels/sensors)
            - Use cross-validation within searchlight
            - Apply cluster-based permutation tests
            
        INTEGRATION:
            - MNE-Python: Use with epochs.get_data()
            - Compatible with time-resolved analyses
        """
        from rdm.compute_dissimilarity import compute_rdm
        
        n_trials, n_channels, n_times = features.shape
        model_vec = model_rdm.get_vector()
        
        # Initialize results
        n_windows = (n_times - window_size) // step_size + 1
        correlations = np.zeros(n_windows)
        time_points = np.zeros(n_windows)
        
        # Slide window across time
        for i, start_idx in enumerate(range(0, n_times - window_size + 1, step_size)):
            end_idx = start_idx + window_size
            time_points[i] = start_idx + window_size // 2
            
            # Extract features in window
            window_features = features[:, :, start_idx:end_idx]
            
            # Average across time and compute RDM
            avg_features = np.mean(window_features, axis=2)  # (n_trials, n_channels)
            
            # Average within conditions
            unique_labels = np.unique(labels)
            condition_features = np.array([
                np.mean(avg_features[labels == label], axis=0)
                for label in unique_labels
            ])
            
            # Compute RDM
            neural_rdm = compute_rdm(condition_features, metric='correlation')
            neural_vec = neural_rdm.get_vector()
            
            # Correlate with model
            correlations[i] = spearmanr(neural_vec, model_vec)[0]
        
        print(f"Searchlight: {n_windows} time windows")
        return correlations, time_points
    
    def plot_results(self,
                    figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Plot RSA results for all models.
        
        Args:
            figsize: Figure size.
        
        Returns:
            Matplotlib figure.
        """
        if not self.results:
            raise RuntimeError("No results to plot. Run comparisons first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        models = list(self.results.keys())
        means = [self.results[m]['mean_correlation'] for m in models]
        sems = [self.results[m]['sem_correlation'] for m in models]
        
        x = np.arange(len(models))
        ax.bar(x, means, yerr=sems, capsize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Correlation with Neural Data')
        ax.set_title('RSA Model Comparison')
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_results(self, filepath: str) -> None:
        """Save analysis results.
        
        Args:
            filepath: Output filepath (.npz format).
        """
        np.savez(filepath,
                results=self.results,
                n_subjects=len(self.neural_rdms),
                model_names=list(self.model_rdms.keys()))
        print(f"Saved results to {filepath}")
    
    @classmethod
    def load_results(cls, filepath: str) -> 'RSAAnalysis':
        """Load analysis results.
        
        Args:
            filepath: Input filepath (.npz format).
        
        Returns:
            RSAAnalysis object with loaded results.
        """
        data = np.load(filepath, allow_pickle=True)
        analysis = cls()
        analysis.results = data['results'].item()
        print(f"Loaded results from {filepath}")
        return analysis


def run_group_rsa(neural_rdms: List,
                 model_rdms: Dict,
                 method: str = 'spearman') -> Dict:
    """Run group-level RSA analysis.
    
    Convenience function for complete RSA workflow.
    
    Args:
        neural_rdms: List of neural RDMs (one per subject).
        model_rdms: Dictionary of model RDMs.
        method: Correlation method.
    
    Returns:
        Dictionary with results for all models.
        
    Example:
        >>> results = run_group_rsa(
        ...     neural_rdms=[rdm1, rdm2, rdm3],
        ...     model_rdms={'semantic': sem_rdm, 'visual': vis_rdm},
        ...     method='spearman'
        ... )
    """
    analysis = RSAAnalysis()
    
    # Add neural RDMs
    for i, rdm in enumerate(neural_rdms):
        analysis.add_neural_rdm(rdm, subject_id=f'S{i+1:02d}')
    
    # Add model RDMs
    for model_name, rdm in model_rdms.items():
        analysis.add_model_rdm(rdm, model_name)
    
    # Compare all models
    for model_name in model_rdms.keys():
        analysis.compare_single_model(model_name, method)
        analysis.test_significance(model_name, test='ttest')
    
    return analysis.results


if __name__ == "__main__":
    # Example usage
    print("RSA Analysis Template")
    print("\nThis script provides RSA analysis functions.")
    print("\nUsage example:")
    print("""
    from analysis.rsa_analysis import RSAAnalysis
    from rdm.compute_dissimilarity import compute_rdm
    
    # Initialize analysis
    analysis = RSAAnalysis()
    
    # Add neural RDMs from multiple subjects
    for subject in subjects:
        neural_rdm = compute_rdm(subject.features, metric='correlation')
        analysis.add_neural_rdm(neural_rdm, subject_id=subject.id)
    
    # Add model RDMs
    semantic_rdm = compute_rdm(semantic_features, metric='correlation')
    analysis.add_model_rdm(semantic_rdm, 'semantic')
    
    visual_rdm = compute_rdm(visual_features, metric='correlation')
    analysis.add_model_rdm(visual_rdm, 'visual')
    
    # Compare models
    analysis.compare_all_models(method='spearman')
    
    # Test significance
    analysis.test_significance('semantic', test='permutation')
    
    # Compare models
    analysis.compare_models('semantic', 'visual', test='paired_ttest')
    
    # Visualize
    fig = analysis.plot_results()
    plt.show()
    """)
