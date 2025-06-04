#!/usr/bin/env python3
"""
Advanced Feature Selection Utilities

This module provides utilities for advanced feature selection using various methods:
1. Feature importance based selection
2. Feature clustering for correlation handling
3. Recursive feature elimination with cross-validation
4. Permutation importance
5. SHAP value based selection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
import os
from sklearn.feature_selection import RFE, RFECV, SelectFromModel, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import xgboost as xgb
import lightgbm as lgb
import time
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedFeatureSelector:
    """Advanced feature selection using multiple methods."""
    
    def __init__(self, output_dir: str = "reports/feature_selection", 
                random_state: int = 42, n_jobs: int = -1):
        """
        Initialize the feature selector.
        
        Args:
            output_dir: Directory to save reports and visualizations
            random_state: Random state for reproducibility
            n_jobs: Number of CPU cores to use
        """
        self.output_dir = output_dir
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results storage
        self.importance_scores = {}
        self.selected_features = {}
        self.feature_rankings = {}
        self.correlation_clusters = None
    
    def importance_based_selection(self, X: pd.DataFrame, y: pd.Series, 
                                 method: str = 'lightgbm', 
                                 n_features: Optional[int] = None,
                                 threshold: Optional[float] = None,
                                 verbose: bool = True) -> List[str]:
        """
        Select features based on importance scores from tree-based models.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Model to use ('xgboost', 'lightgbm', 'random_forest', 'gradient_boosting')
            n_features: Number of top features to select (if None, uses threshold)
            threshold: Importance threshold (if None, uses n_features)
            verbose: Whether to log progress
            
        Returns:
            List of selected feature names
        """
        if verbose:
            logger.info(f"Running importance-based selection using {method}...")
            start_time = time.time()
        
        # Initialize model based on method
        if method == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=100, 
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif method == 'lightgbm':
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1
            )
        elif method == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        elif method == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Fit model
        model.fit(X, y)
        
        # Get feature importances
        if method in ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting']:
            importances = model.feature_importances_
        else:
            importances = model.coef_[0]
        
        # Create DataFrame with importances
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Store importances
        self.importance_scores[method] = importance_df
        
        # Select features
        if n_features is not None:
            selected = importance_df.head(n_features)['feature'].tolist()
        elif threshold is not None:
            selected = importance_df[importance_df['importance'] > threshold]['feature'].tolist()
        else:
            # Default: select features with importance > mean
            mean_importance = importance_df['importance'].mean()
            selected = importance_df[importance_df['importance'] > mean_importance]['feature'].tolist()
        
        # Store selected features
        self.selected_features[f"{method}_importance"] = selected
        
        if verbose:
            duration = time.time() - start_time
            logger.info(f"Selected {len(selected)} features using {method} in {duration:.2f} seconds")
            
            # Calculate reduction percentage
            reduction = (1 - len(selected) / len(X.columns)) * 100
            logger.info(f"Reduced feature set by {reduction:.2f}%")
        
        return selected
    
    def correlation_based_selection(self, X: pd.DataFrame, method: str = 'spearman',
                                  threshold: float = 0.8, n_clusters: Optional[int] = None,
                                  verbose: bool = True) -> List[str]:
        """
        Select features by clustering correlated features and picking representatives.
        
        Args:
            X: Feature DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            threshold: Correlation threshold for clustering
            n_clusters: Number of clusters to form (if None, determined from threshold)
            verbose: Whether to log progress
            
        Returns:
            List of selected feature names
        """
        if verbose:
            logger.info(f"Running correlation-based selection using {method}...")
            start_time = time.time()
        
        # Calculate correlation matrix
        corr = X.corr(method=method)
        
        # Convert to distance matrix
        distance = 1 - abs(corr)
        
        # Cluster features
        if n_clusters is None:
            # Use hierarchical clustering with distance threshold
            condensed_dist = squareform(distance)
            z = hierarchy.linkage(condensed_dist, method='average')
            clusters = hierarchy.fcluster(z, t=1-threshold, criterion='distance')
        else:
            # Use AgglomerativeClustering with specified number of clusters
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                linkage='average'
            )
            clusters = clustering.fit_predict(distance)
        
        # Store cluster information
        self.correlation_clusters = pd.DataFrame({
            'feature': X.columns,
            'cluster': clusters
        })
        
        # For each cluster, select the most important feature
        # Or the feature with highest variance if no importance scores
        selected_features = []
        for cluster_id in np.unique(clusters):
            cluster_features = self.correlation_clusters[self.correlation_clusters['cluster'] == cluster_id]['feature'].tolist()
            
            # If we have importance scores, use them
            if hasattr(self, 'importance_scores') and len(self.importance_scores) > 0:
                # Use the first available importance method
                method_name = list(self.importance_scores.keys())[0]
                importance_df = self.importance_scores[method_name]
                
                # Filter to only cluster features and sort by importance
                cluster_importance = importance_df[importance_df['feature'].isin(cluster_features)]
                if not cluster_importance.empty:
                    best_feature = cluster_importance.iloc[0]['feature']
                    selected_features.append(best_feature)
                    continue
            
            # Otherwise, use the feature with highest variance
            variances = X[cluster_features].var()
            best_feature = variances.idxmax()
            selected_features.append(best_feature)
        
        # Store selected features
        self.selected_features[f"{method}_correlation"] = selected_features
        
        if verbose:
            duration = time.time() - start_time
            logger.info(f"Selected {len(selected_features)} features using correlation clustering in {duration:.2f} seconds")
            
            # Calculate reduction percentage
            reduction = (1 - len(selected_features) / len(X.columns)) * 100
            logger.info(f"Reduced feature set by {reduction:.2f}%")
        
        return selected_features
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series,
                                    cv: int = 5, step: float = 0.1,
                                    estimator: Optional[Any] = None,
                                    verbose: bool = True) -> List[str]:
        """
        Select features using recursive feature elimination with cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            cv: Number of cross-validation folds
            step: Step size for feature elimination (percentage or number)
            estimator: Base estimator (default: RandomForestClassifier)
            verbose: Whether to log progress
            
        Returns:
            List of selected feature names
        """
        if verbose:
            logger.info("Running recursive feature elimination with cross-validation...")
            start_time = time.time()
        
        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        
        # Convert step to integer if it's a percentage
        if 0 < step < 1:
            step = max(1, int(len(X.columns) * step))
        
        # Initialize RFECV
        rfecv = RFECV(
            estimator=estimator,
            step=step,
            cv=cv,
            scoring='roc_auc',
            n_jobs=self.n_jobs,
            verbose=int(verbose)
        )
        
        # Fit RFECV
        rfecv.fit(X, y)
        
        # Get selected features
        selected_features = X.columns[rfecv.support_].tolist()
        
        # Store selected features
        self.selected_features['rfecv'] = selected_features
        
        # Store rankings
        self.feature_rankings['rfecv'] = pd.DataFrame({
            'feature': X.columns,
            'ranking': rfecv.ranking_
        }).sort_values('ranking')
        
        if verbose:
            duration = time.time() - start_time
            logger.info(f"Selected {len(selected_features)} features using RFECV in {duration:.2f} seconds")
            logger.info(f"Optimal number of features: {rfecv.n_features_}")
            
            # Calculate reduction percentage
            reduction = (1 - len(selected_features) / len(X.columns)) * 100
            logger.info(f"Reduced feature set by {reduction:.2f}%")
        
        return selected_features
    
    def permutation_importance_selection(self, X: pd.DataFrame, y: pd.Series,
                                       model: Optional[Any] = None,
                                       n_repeats: int = 10,
                                       n_features: Optional[int] = None,
                                       threshold: Optional[float] = None,
                                       verbose: bool = True) -> List[str]:
        """
        Select features using permutation importance.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model: Fitted model (default: RandomForestClassifier)
            n_repeats: Number of times to permute each feature
            n_features: Number of top features to select (if None, uses threshold)
            threshold: Importance threshold (if None, uses n_features)
            verbose: Whether to log progress
            
        Returns:
            List of selected feature names
        """
        if verbose:
            logger.info("Running permutation importance selection...")
            start_time = time.time()
        
        if model is None:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            model.fit(X, y)
        
        # Calculate permutation importance
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        # Create DataFrame with importances
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        # Store importances
        self.importance_scores['permutation'] = importance_df
        
        # Select features
        if n_features is not None:
            selected = importance_df.head(n_features)['feature'].tolist()
        elif threshold is not None:
            selected = importance_df[importance_df['importance_mean'] > threshold]['feature'].tolist()
        else:
            # Default: select features with importance > 0
            selected = importance_df[importance_df['importance_mean'] > 0]['feature'].tolist()
        
        # Store selected features
        self.selected_features['permutation_importance'] = selected
        
        if verbose:
            duration = time.time() - start_time
            logger.info(f"Selected {len(selected)} features using permutation importance in {duration:.2f} seconds")
            
            # Calculate reduction percentage
            reduction = (1 - len(selected) / len(X.columns)) * 100
            logger.info(f"Reduced feature set by {reduction:.2f}%")
        
        return selected
    
    def combine_methods(self, min_methods: int = 2, verbose: bool = True) -> List[str]:
        """
        Combine features selected by multiple methods.
        
        Args:
            min_methods: Minimum number of methods that must select a feature
            verbose: Whether to log progress
            
        Returns:
            List of selected feature names
        """
        if verbose:
            logger.info(f"Combining features selected by at least {min_methods} methods...")
        
        if len(self.selected_features) < min_methods:
            logger.warning(f"Only {len(self.selected_features)} selection methods available, "
                         f"but {min_methods} required. Using union of all methods.")
            min_methods = 1
        
        # Count how many methods selected each feature
        all_features = []
        for method, features in self.selected_features.items():
            all_features.extend(features)
        
        feature_counts = pd.Series(all_features).value_counts()
        
        # Select features that appear in at least min_methods methods
        selected = feature_counts[feature_counts >= min_methods].index.tolist()
        
        # Store selected features
        self.selected_features['combined'] = selected
        
        if verbose:
            logger.info(f"Selected {len(selected)} features by combining methods")
            
            # Calculate reduction percentage from the largest feature set
            max_features = max(len(features) for features in self.selected_features.values())
            reduction = (1 - len(selected) / max_features) * 100
            logger.info(f"Reduced feature set by {reduction:.2f}% from largest method")
        
        return selected
    
    def plot_importances(self, method: str, top_n: int = 20) -> None:
        """
        Plot feature importances for a given method.
        
        Args:
            method: Method name in self.importance_scores
            top_n: Number of top features to plot
        """
        if method not in self.importance_scores:
            logger.warning(f"No importance scores found for method: {method}")
            return
        
        importance_df = self.importance_scores[method]
        
        # Get top N features
        plot_df = importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        
        # Check if we have std values
        if 'importance_std' in plot_df.columns:
            # Plot with error bars
            plt.barh(
                plot_df['feature'],
                plot_df['importance_mean'],
                xerr=plot_df['importance_std'],
                color='skyblue'
            )
            plt.xlabel('Mean Importance (± Std)')
        else:
            # Plot without error bars
            plt.barh(
                plot_df['feature'],
                plot_df['importance'],
                color='skyblue'
            )
            plt.xlabel('Importance')
        
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Features by {method} Importance')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f"{method}_importances.png"), dpi=300)
        plt.close()
    
    def plot_correlation_heatmap(self, X: pd.DataFrame, method: str = 'spearman',
                               selected_only: bool = True) -> None:
        """
        Plot correlation heatmap for features.
        
        Args:
            X: Feature DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            selected_only: Whether to plot only selected features
        """
        if selected_only and 'combined' in self.selected_features:
            features = self.selected_features['combined']
            plot_df = X[features]
        else:
            plot_df = X
        
        # Calculate correlation matrix
        corr = plot_df.corr(method=method)
        
        # Plot heatmap
        plt.figure(figsize=(16, 14))
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        
        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            square=True,
            linewidths=0.5,
            annot=False,
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title(f'{method.capitalize()} Correlation Heatmap')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f"correlation_heatmap_{method}.png"), dpi=300)
        plt.close()
    
    def plot_cluster_dendrogram(self, X: pd.DataFrame, method: str = 'spearman',
                              figsize: Tuple[int, int] = (20, 10)) -> None:
        """
        Plot dendrogram of feature clustering based on correlation.
        
        Args:
            X: Feature DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            figsize: Figure size
        """
        # Calculate correlation matrix
        corr = X.corr(method=method)
        
        # Convert to distance matrix
        distance = 1 - abs(corr)
        
        # Calculate linkage
        condensed_dist = squareform(distance)
        z = hierarchy.linkage(condensed_dist, method='average')
        
        # Plot dendrogram
        plt.figure(figsize=figsize)
        hierarchy.dendrogram(
            z,
            labels=X.columns,
            orientation='right',
            leaf_font_size=8
        )
        plt.title('Feature Correlation Dendrogram')
        plt.xlabel('Distance')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f"dendrogram_{method}.png"), dpi=300)
        plt.close()
    
    def save_results(self, filename: str = "feature_selection_results.pkl") -> None:
        """
        Save feature selection results to file.
        
        Args:
            filename: Output filename
        """
        results = {
            'importance_scores': self.importance_scores,
            'selected_features': self.selected_features,
            'feature_rankings': self.feature_rankings,
            'correlation_clusters': self.correlation_clusters
        }
        
        output_path = os.path.join(self.output_dir, filename)
        joblib.dump(results, output_path)
        logger.info(f"Feature selection results saved to {output_path}")
    
    def load_results(self, filename: str = "feature_selection_results.pkl") -> None:
        """
        Load feature selection results from file.
        
        Args:
            filename: Input filename
        """
        input_path = os.path.join(self.output_dir, filename)
        if not os.path.exists(input_path):
            logger.warning(f"Results file not found: {input_path}")
            return
        
        results = joblib.load(input_path)
        
        self.importance_scores = results['importance_scores']
        self.selected_features = results['selected_features']
        self.feature_rankings = results['feature_rankings']
        self.correlation_clusters = results['correlation_clusters']
        
        logger.info(f"Feature selection results loaded from {input_path}")
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive report on feature selection.
        
        Returns:
            Path to the report file
        """
        report_path = os.path.join(self.output_dir, "feature_selection_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# Feature Selection Report\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            
            if 'combined' in self.selected_features:
                combined_features = self.selected_features['combined']
                f.write(f"**Combined Selection**: {len(combined_features)} features\n\n")
                
                # Calculate reduction
                if self.importance_scores:
                    first_method = list(self.importance_scores.keys())[0]
                    total_features = len(self.importance_scores[first_method])
                    reduction = (1 - len(combined_features) / total_features) * 100
                    f.write(f"**Dimension Reduction**: {reduction:.2f}%\n\n")
            
            # Results by method
            f.write("## Results by Method\n\n")
            f.write("| Method | Features Selected | % of Total |\n")
            f.write("|--------|------------------|------------|\n")
            
            if self.importance_scores:
                first_method = list(self.importance_scores.keys())[0]
                total_features = len(self.importance_scores[first_method])
                
                for method, features in self.selected_features.items():
                    percentage = (len(features) / total_features) * 100
                    f.write(f"| {method} | {len(features)} | {percentage:.2f}% |\n")
            
            # Top features by method
            f.write("\n## Top Features by Method\n\n")
            
            for method, features in self.selected_features.items():
                f.write(f"### {method}\n\n")
                
                # Show top 20 features or all if less than 20
                show_features = features[:20] if len(features) > 20 else features
                for i, feature in enumerate(show_features, 1):
                    f.write(f"{i}. {feature}\n")
                
                if len(features) > 20:
                    f.write(f"... and {len(features) - 20} more\n")
                
                f.write("\n")
            
            # Feature importance plots
            f.write("\n## Feature Importance Plots\n\n")
            
            for method in self.importance_scores:
                plot_path = f"{method}_importances.png"
                f.write(f"### {method}\n\n")
                f.write(f"![{method} Feature Importance]({plot_path})\n\n")
            
            # Correlation analysis
            if self.correlation_clusters is not None:
                f.write("\n## Correlation Analysis\n\n")
                f.write(f"Identified {len(np.unique(self.correlation_clusters['cluster']))} feature clusters\n\n")
                
                # Show clusters with multiple features
                cluster_counts = self.correlation_clusters['cluster'].value_counts()
                multi_feature_clusters = cluster_counts[cluster_counts > 1]
                
                if len(multi_feature_clusters) > 0:
                    f.write("### Correlated Feature Clusters\n\n")
                    
                    for cluster_id in multi_feature_clusters.index:
                        cluster_features = self.correlation_clusters[self.correlation_clusters['cluster'] == cluster_id]['feature'].tolist()
                        f.write(f"**Cluster {cluster_id}** ({len(cluster_features)} features):\n\n")
                        for feature in cluster_features:
                            f.write(f"- {feature}\n")
                        f.write("\n")
                
                # Add correlation heatmap reference
                f.write("### Correlation Heatmap\n\n")
                f.write("![Correlation Heatmap](correlation_heatmap_spearman.png)\n\n")
                
                # Add dendrogram reference
                f.write("### Feature Clustering Dendrogram\n\n")
                f.write("![Dendrogram](dendrogram_spearman.png)\n\n")
        
        logger.info(f"Feature selection report generated: {report_path}")
        return report_path


def select_features(X: pd.DataFrame, y: pd.Series, 
                   methods: List[str] = ['lightgbm', 'correlation'],
                   n_features: Optional[int] = None,
                   output_dir: str = "reports/feature_selection",
                   random_state: int = 42,
                   n_jobs: int = -1,
                   verbose: bool = True) -> List[str]:
    """
    Convenience function for feature selection using multiple methods.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        methods: List of methods to use ('lightgbm', 'xgboost', 'random_forest', 
                                       'correlation', 'rfecv', 'permutation')
        n_features: Number of features to select (if None, automatic)
        output_dir: Directory to save reports and visualizations
        random_state: Random state for reproducibility
        n_jobs: Number of CPU cores to use
        verbose: Whether to log progress
        
    Returns:
        List of selected feature names
    """
    # Initialize selector
    selector = AdvancedFeatureSelector(
        output_dir=output_dir,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    # Apply each method
    for method in methods:
        if method == 'lightgbm':
            selector.importance_based_selection(
                X, y, method='lightgbm', n_features=n_features, verbose=verbose
            )
        elif method == 'xgboost':
            selector.importance_based_selection(
                X, y, method='xgboost', n_features=n_features, verbose=verbose
            )
        elif method == 'random_forest':
            selector.importance_based_selection(
                X, y, method='random_forest', n_features=n_features, verbose=verbose
            )
        elif method == 'correlation':
            selector.correlation_based_selection(
                X, method='spearman', verbose=verbose
            )
        elif method == 'rfecv':
            selector.recursive_feature_elimination(
                X, y, verbose=verbose
            )
        elif method == 'permutation':
            selector.permutation_importance_selection(
                X, y, n_features=n_features, verbose=verbose
            )
        else:
            logger.warning(f"Unknown method: {method}")
    
    # Combine methods
    selected_features = selector.combine_methods(
        min_methods=1 if len(methods) == 1 else 2,
        verbose=verbose
    )
    
    # Generate visualizations
    if 'lightgbm' in methods:
        selector.plot_importances('lightgbm')
    elif 'xgboost' in methods:
        selector.plot_importances('xgboost')
    elif 'permutation' in methods:
        selector.plot_importances('permutation')
    
    if 'correlation' in methods:
        selector.plot_correlation_heatmap(X, selected_only=True)
        selector.plot_cluster_dendrogram(X)
    
    # Save results
    selector.save_results()
    
    # Generate report
    selector.generate_report()
    
    return selected_features
