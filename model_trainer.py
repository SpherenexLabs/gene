"""
Model Training & Classification Module for Disease Gene Detection
Implements multiple ML classifiers with hyperparameter tuning
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import logging
from typing import Dict, Tuple, Optional, Any
import pickle
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiseaseGeneClassifier:
    """Comprehensive model training and evaluation for disease gene detection"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize classifier trainer
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.cv_scores = {}
        self.training_history = {}
        
    def get_model_configs(self) -> Dict[str, Dict]:
        """
        Get default model configurations and hyperparameter grids
        
        Returns:
            Dictionary of model configs
        """
        configs = {
            'svm': {
                'model': SVC(random_state=self.random_state, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'ann': {
                'model': MLPClassifier(random_state=self.random_state, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            },
            'knn': {
                'model': KNeighborsClassifier(n_jobs=-1),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000, n_jobs=-1),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            }
        }
        
        return configs
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                   custom_params: Optional[Dict] = None) -> Any:
        """
        Train a single model with default or custom parameters
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            custom_params: Optional custom parameters
            
        Returns:
            Trained model
        """
        logger.info(f"Training {model_name}...")
        start_time = time.time()
        
        configs = self.get_model_configs()
        
        if model_name not in configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = configs[model_name]['model']
        
        if custom_params:
            model.set_params(**custom_params)
        
        # Train model
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Store model and training info
        self.models[model_name] = model
        self.training_history[model_name] = {
            'training_time': training_time,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1]
        }
        
        logger.info(f"{model_name} trained in {training_time:.2f} seconds")
        
        return model
    
    def cross_validate(self, model_name: str, X: np.ndarray, y: np.ndarray,
                      cv: int = 5, scoring: str = 'accuracy') -> Dict:
        """
        Perform k-fold cross-validation
        
        Args:
            model_name: Name of the model
            X: Feature matrix
            y: Target labels
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Running {cv}-fold cross-validation for {model_name}...")
        
        configs = self.get_model_configs()
        model = configs[model_name]['model']
        
        # Stratified K-Fold for imbalanced datasets
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        
        cv_results = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }
        
        # Store results
        self.cv_scores[model_name] = cv_results
        
        logger.info(f"{model_name} CV Score: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")
        
        return cv_results
    
    def hyperparameter_tuning(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                             method: str = 'grid', cv: int = 5, n_iter: int = 50) -> Tuple[Any, Dict]:
        """
        Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            method: 'grid' or 'random'
            cv: Number of folds
            n_iter: Number of iterations for RandomizedSearch
            
        Returns:
            Best model and best parameters
        """
        logger.info(f"Tuning hyperparameters for {model_name} using {method}Search...")
        start_time = time.time()
        
        configs = self.get_model_configs()
        
        if model_name not in configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = configs[model_name]['model']
        param_grid = configs[model_name]['params']
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Choose search method
        if method == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=skf, scoring='accuracy',
                n_jobs=-1, verbose=1
            )
        else:  # random
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, cv=skf, scoring='accuracy',
                n_jobs=-1, verbose=1, random_state=self.random_state
            )
        
        # Perform search
        search.fit(X_train, y_train)
        
        tuning_time = time.time() - start_time
        
        # Store best model
        self.best_models[model_name] = search.best_estimator_
        
        logger.info(f"Best parameters for {model_name}: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")
        logger.info(f"Tuning completed in {tuning_time:.2f} seconds")
        
        return search.best_estimator_, search.best_params_
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        models: Optional[list] = None, tune: bool = False) -> Dict:
        """
        Train all models
        
        Args:
            X_train: Training features
            y_train: Training labels
            models: List of model names (default: all)
            tune: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary of trained models
        """
        logger.info("=" * 70)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 70)
        
        if models is None:
            models = ['svm', 'random_forest', 'ann', 'knn', 'gradient_boosting']
        
        trained_models = {}
        
        for model_name in models:
            try:
                if tune:
                    model, params = self.hyperparameter_tuning(model_name, X_train, y_train)
                    trained_models[model_name] = model
                else:
                    model = self.train_model(model_name, X_train, y_train)
                    trained_models[model_name] = model
                    
                # Cross-validate
                self.cross_validate(model_name, X_train, y_train)
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
        
        logger.info("=" * 70)
        logger.info("ALL MODELS TRAINED")
        logger.info("=" * 70)
        
        return trained_models
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray,
                      y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels
            y_pred_proba: Optional prediction probabilities
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Get model
        if model_name in self.best_models:
            model = self.best_models[model_name]
        elif model_name in self.models:
            model = self.models[model_name]
        else:
            raise ValueError(f"Model {model_name} not trained")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Probability predictions for ROC-AUC
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }
        
        # ROC-AUC for binary/multiclass
        if y_pred_proba is not None:
            try:
                n_classes = len(np.unique(y_test))
                if n_classes == 2:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, 
                                                       multi_class='ovr', average='weighted')
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
                metrics['roc_auc'] = None
        else:
            metrics['roc_auc'] = None
        
        logger.info(f"{model_name} Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"{model_name} Precision: {metrics['precision']:.4f}")
        logger.info(f"{model_name} Recall: {metrics['recall']:.4f}")
        logger.info(f"{model_name} F1-Score: {metrics['f1']:.4f}")
        if metrics['roc_auc']:
            logger.info(f"{model_name} ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def compare_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Compare all trained models
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with model comparison
        """
        logger.info("Comparing all models...")
        
        comparison = []
        
        # Check both regular and best models
        all_models = {}
        all_models.update(self.models)
        all_models.update(self.best_models)
        
        for model_name, model in all_models.items():
            metrics = self.evaluate_model(model_name, X_test, y_test)
            
            # Ensure all metrics are valid numbers
            roc_auc_value = metrics.get('roc_auc', 0)
            if roc_auc_value is None or np.isnan(roc_auc_value):
                roc_auc_value = 0
            
            comparison.append({
                'Model': model_name,
                'Accuracy': float(metrics['accuracy']),
                'Precision': float(metrics['precision']),
                'Recall': float(metrics['recall']),
                'F1-Score': float(metrics['f1']),
                'ROC-AUC': float(roc_auc_value)
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('Accuracy', ascending=False)
        
        return df
    
    def save_model(self, model_name: str, filepath: str):
        """Save trained model to file"""
        if model_name in self.best_models:
            model = self.best_models[model_name]
        elif model_name in self.models:
            model = self.models[model_name]
        else:
            raise ValueError(f"Model {model_name} not found")
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, filepath: str, model_name: str):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        self.models[model_name] = model
        logger.info(f"Model {model_name} loaded from {filepath}")
        
        return model


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    X_train = np.random.randn(500, 50)
    y_train = np.random.randint(0, 3, 500)
    X_test = np.random.randn(100, 50)
    y_test = np.random.randint(0, 3, 100)
    
    # Initialize trainer
    trainer = DiseaseGeneClassifier()
    
    # Train all models
    models = trainer.train_all_models(X_train, y_train, tune=False)
    
    # Compare models
    comparison = trainer.compare_models(X_test, y_test)
    print("\nModel Comparison:")
    print(comparison)
