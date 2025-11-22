"""
Feature Selection Module for Disease Gene Detection
Implements statistical and ML-based feature selection techniques
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import logging
from typing import Tuple, List, Dict, Optional
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneFeatureSelector:
    """Comprehensive feature selection for gene expression data"""
    
    def __init__(self, n_features: int = 100):
        """
        Initialize feature selector
        
        Args:
            n_features: Number of top features to select
        """
        self.n_features = n_features
        self.selected_features = None
        self.feature_scores = {}
        self.feature_rankings = {}
        
    def anova_selection(self, X: np.ndarray, y: np.ndarray, 
                       feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        ANOVA F-test for feature selection
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names
            
        Returns:
            Selected features, feature names, and scores
        """
        logger.info(f"Running ANOVA F-test for top {self.n_features} features...")
        
        # Create selector
        selector = SelectKBest(score_func=f_classif, k=self.n_features)
        X_selected = selector.fit_transform(X, y)
        
        # Get scores
        scores = selector.scores_
        selected_indices = selector.get_support(indices=True)
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        selected_names = [feature_names[i] for i in selected_indices]
        
        # Store results
        self.feature_scores['anova'] = dict(zip(feature_names, scores))
        
        logger.info(f"Selected {len(selected_names)} features using ANOVA")
        logger.info(f"Top 5 scores: {sorted(scores, reverse=True)[:5]}")
        
        return X_selected, selected_names, scores[selected_indices]
    
    def chi_square_selection(self, X: np.ndarray, y: np.ndarray,
                            feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Chi-square test for feature selection
        Note: Requires non-negative features
        
        Args:
            X: Feature matrix (non-negative)
            y: Target labels
            feature_names: Optional feature names
            
        Returns:
            Selected features, feature names, and scores
        """
        logger.info(f"Running Chi-square test for top {self.n_features} features...")
        
        # Ensure non-negative values (shift if needed)
        X_positive = X - X.min() + 1e-10 if X.min() < 0 else X
        
        # Create selector
        selector = SelectKBest(score_func=chi2, k=self.n_features)
        X_selected = selector.fit_transform(X_positive, y)
        
        # Get scores
        scores = selector.scores_
        selected_indices = selector.get_support(indices=True)
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        selected_names = [feature_names[i] for i in selected_indices]
        
        # Store results
        self.feature_scores['chi_square'] = dict(zip(feature_names, scores))
        
        logger.info(f"Selected {len(selected_names)} features using Chi-square")
        
        return X_selected, selected_names, scores[selected_indices]
    
    def correlation_selection(self, X: np.ndarray, y: np.ndarray,
                             feature_names: Optional[List[str]] = None,
                             method: str = 'pearson') -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Correlation-based feature selection
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names
            method: 'pearson' or 'spearman'
            
        Returns:
            Selected features, feature names, and correlation scores
        """
        logger.info(f"Running {method.capitalize()} correlation analysis...")
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Calculate correlations
        correlations = []
        for i in range(X.shape[1]):
            if method == 'pearson':
                corr, _ = stats.pearsonr(X[:, i], y)
            else:  # spearman
                corr, _ = stats.spearmanr(X[:, i], y)
            correlations.append(abs(corr))
        
        correlations = np.array(correlations)
        
        # Select top features
        top_indices = np.argsort(correlations)[-self.n_features:][::-1]
        selected_names = [feature_names[i] for i in top_indices]
        
        # Handle both DataFrame and numpy array
        if hasattr(X, 'iloc'):
            # DataFrame
            X_selected = X.iloc[:, top_indices].values
        else:
            # Numpy array
            X_selected = X[:, top_indices]
        
        # Store results
        self.feature_scores[f'{method}_correlation'] = dict(zip(feature_names, correlations))
        
        logger.info(f"Selected {len(selected_names)} features using {method} correlation")
        logger.info(f"Top 5 correlations: {sorted(correlations, reverse=True)[:5]}")
        
        return X_selected, selected_names, correlations[top_indices]
    
    def mutual_information_selection(self, X: np.ndarray, y: np.ndarray,
                                    feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Mutual Information for feature selection
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names
            
        Returns:
            Selected features, feature names, and MI scores
        """
        logger.info(f"Running Mutual Information for top {self.n_features} features...")
        
        # Create selector
        selector = SelectKBest(score_func=mutual_info_classif, k=self.n_features)
        X_selected = selector.fit_transform(X, y)
        
        # Get scores
        scores = selector.scores_
        selected_indices = selector.get_support(indices=True)
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        selected_names = [feature_names[i] for i in selected_indices]
        
        # Store results
        self.feature_scores['mutual_information'] = dict(zip(feature_names, scores))
        
        logger.info(f"Selected {len(selected_names)} features using Mutual Information")
        
        return X_selected, selected_names, scores[selected_indices]
    
    def rfe_selection(self, X: np.ndarray, y: np.ndarray,
                     feature_names: Optional[List[str]] = None,
                     estimator=None) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Recursive Feature Elimination (RFE)
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names
            estimator: Base estimator (default: RandomForest)
            
        Returns:
            Selected features, feature names, and rankings
        """
        logger.info(f"Running RFE for top {self.n_features} features...")
        
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        # Create RFE selector
        selector = RFE(estimator=estimator, n_features_to_select=self.n_features, step=1)
        X_selected = selector.fit_transform(X, y)
        
        # Get rankings
        rankings = selector.ranking_
        selected_indices = selector.get_support(indices=True)
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        selected_names = [feature_names[i] for i in selected_indices]
        
        # Store results
        self.feature_rankings['rfe'] = dict(zip(feature_names, rankings))
        
        logger.info(f"Selected {len(selected_names)} features using RFE")
        
        return X_selected, selected_names, rankings[selected_indices]
    
    def tree_importance_selection(self, X: np.ndarray, y: np.ndarray,
                                  feature_names: Optional[List[str]] = None,
                                  estimator=None) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Tree-based feature importance selection
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names
            estimator: Tree-based estimator (default: RandomForest)
            
        Returns:
            Selected features, feature names, and importance scores
        """
        logger.info(f"Running Tree-based feature importance for top {self.n_features} features...")
        
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Fit estimator
        estimator.fit(X, y)
        
        # Get feature importances
        importances = estimator.feature_importances_
        
        # Select top features
        top_indices = np.argsort(importances)[-self.n_features:][::-1]
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        selected_names = [feature_names[i] for i in top_indices]
        
        # Handle both DataFrame and numpy array
        if hasattr(X, 'iloc'):
            # DataFrame
            X_selected = X.iloc[:, top_indices].values
        else:
            # Numpy array
            X_selected = X[:, top_indices]
        
        # Store results
        self.feature_scores['tree_importance'] = dict(zip(feature_names, importances))
        
        logger.info(f"Selected {len(selected_names)} features using Tree importance")
        logger.info(f"Top 5 importances: {sorted(importances, reverse=True)[:5]}")
        
        return X_selected, selected_names, importances[top_indices]
    
    def ensemble_selection(self, X: np.ndarray, y: np.ndarray,
                          feature_names: Optional[List[str]] = None,
                          methods: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str], Dict]:
        """
        Ensemble feature selection combining multiple methods
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names
            methods: List of methods to combine
            
        Returns:
            Selected features, feature names, and combined scores
        """
        logger.info("Running Ensemble Feature Selection...")
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        if methods is None:
            methods = ['anova', 'mutual_information', 'tree_importance']
        
        # Run each method and collect scores
        all_scores = {}
        
        for method in methods:
            if method == 'anova':
                _, _, scores = self.anova_selection(X, y, feature_names)
            elif method == 'chi_square':
                _, _, scores = self.chi_square_selection(X, y, feature_names)
            elif method == 'pearson':
                _, _, scores = self.correlation_selection(X, y, feature_names, 'pearson')
            elif method == 'spearman':
                _, _, scores = self.correlation_selection(X, y, feature_names, 'spearman')
            elif method == 'mutual_information':
                _, _, scores = self.mutual_information_selection(X, y, feature_names)
            elif method == 'tree_importance':
                _, _, scores = self.tree_importance_selection(X, y, feature_names)
        
        # Normalize and combine scores
        combined_scores = np.zeros(X.shape[1])
        for method_name, scores_dict in self.feature_scores.items():
            if method_name in methods or any(m in method_name for m in methods):
                scores_array = np.array([scores_dict.get(fn, 0) for fn in feature_names])
                # Normalize to 0-1
                if scores_array.max() > 0:
                    scores_array = scores_array / scores_array.max()
                combined_scores += scores_array
        
        # Average scores
        combined_scores /= len(methods)
        
        # Select top features
        top_indices = np.argsort(combined_scores)[-self.n_features:][::-1]
        selected_names = [feature_names[i] for i in top_indices]
        
        # Handle both DataFrame and numpy array
        if hasattr(X, 'iloc'):
            # DataFrame
            X_selected = X.iloc[:, top_indices].values
        else:
            # Numpy array
            X_selected = X[:, top_indices]
        
        # Store results
        ensemble_scores = dict(zip(feature_names, combined_scores))
        self.feature_scores['ensemble'] = ensemble_scores
        
        logger.info(f"Selected {len(selected_names)} features using Ensemble method")
        logger.info(f"Methods used: {methods}")
        
        return X_selected, selected_names, ensemble_scores
    
    def get_feature_summary(self) -> pd.DataFrame:
        """
        Get summary of all feature selection results
        
        Returns:
            DataFrame with feature scores from all methods
        """
        # Combine all scores into a DataFrame
        summary_data = {}
        
        for method, scores in self.feature_scores.items():
            summary_data[method] = scores
        
        df = pd.DataFrame(summary_data)
        
        # Add average score
        df['average_score'] = df.mean(axis=1)
        
        # Sort by average score
        df = df.sort_values('average_score', ascending=False)
        
        return df
    
    def save_selector(self, filepath: str):
        """Save feature selector to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'n_features': self.n_features,
                'selected_features': self.selected_features,
                'feature_scores': self.feature_scores,
                'feature_rankings': self.feature_rankings
            }, f)
        logger.info(f"Feature selector saved to {filepath}")
    
    def load_selector(self, filepath: str):
        """Load feature selector from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.n_features = data['n_features']
            self.selected_features = data['selected_features']
            self.feature_scores = data['feature_scores']
            self.feature_rankings = data['feature_rankings']
        logger.info(f"Feature selector loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(200, 500)  # 200 samples, 500 genes
    y = np.random.randint(0, 3, 200)  # 3 disease classes
    
    feature_names = [f'GENE_{i+1}' for i in range(500)]
    
    # Initialize selector
    selector = GeneFeatureSelector(n_features=50)
    
    # Test different methods
    print("\n=== Testing Feature Selection Methods ===\n")
    
    # ANOVA
    X_anova, names_anova, scores_anova = selector.anova_selection(X, y, feature_names)
    print(f"ANOVA: Selected {len(names_anova)} features")
    
    # Mutual Information
    X_mi, names_mi, scores_mi = selector.mutual_information_selection(X, y, feature_names)
    print(f"Mutual Information: Selected {len(names_mi)} features")
    
    # Tree Importance
    X_tree, names_tree, scores_tree = selector.tree_importance_selection(X, y, feature_names)
    print(f"Tree Importance: Selected {len(names_tree)} features")
    
    # Ensemble
    X_ensemble, names_ensemble, scores_ensemble = selector.ensemble_selection(
        X, y, feature_names, 
        methods=['anova', 'mutual_information', 'tree_importance']
    )
    print(f"Ensemble: Selected {len(names_ensemble)} features")
    
    # Get summary
    summary = selector.get_feature_summary()
    print(f"\nTop 10 genes by average score:")
    print(summary.head(10))
