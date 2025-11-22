"""
Complete Machine Learning Pipeline for Disease Gene Detection
Integrates all modules: preprocessing, feature selection, training, evaluation, visualization, and export
"""
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from preprocessor import GeneDataPreprocessor
from feature_selector import GeneFeatureSelector
from model_trainer import DiseaseGeneClassifier
from visualization_engine import GeneVisualizationEngine
from results_exporter import ResultsExporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompletePipeline:
    """End-to-end ML pipeline for disease gene detection"""
    
    def __init__(self, output_dir: str = 'pipeline_output'):
        """
        Initialize complete pipeline
        
        Args:
            output_dir: Base output directory
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create subdirectories
        self.models_dir = os.path.join(output_dir, 'models')
        self.results_dir = os.path.join(output_dir, 'results')
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        
        for directory in [self.models_dir, self.results_dir, self.viz_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize components
        self.preprocessor = None
        self.feature_selector = None
        self.trainer = None
        self.viz_engine = GeneVisualizationEngine(output_dir=self.viz_dir)
        self.exporter = ResultsExporter(output_dir=self.results_dir)
        
        # Store results
        self.results = {
            'preprocessing': {},
            'feature_selection': {},
            'models': {},
            'evaluation': {},
            'visualization_paths': []
        }
    
    def run_preprocessing(self, data: pd.DataFrame, label_column: str,
                         config: Optional[Dict] = None) -> Tuple:
        """
        Run preprocessing pipeline
        
        Args:
            data: Input DataFrame
            label_column: Name of label column
            config: Optional preprocessing config
            
        Returns:
            Tuple of processed datasets
        """
        logger.info("="*70)
        logger.info("STEP 1: PREPROCESSING")
        logger.info("="*70)
        
        self.preprocessor = GeneDataPreprocessor(config)
        
        # Run full preprocessing pipeline
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.full_pipeline(
            data, label_column=label_column, include_validation=True
        )
        
        # Save preprocessor
        preprocessor_path = os.path.join(self.models_dir, f'preprocessor_{self.timestamp}.pkl')
        self.preprocessor.save_preprocessor(preprocessor_path)
        
        # Store results
        self.results['preprocessing'] = {
            'original_shape': self.preprocessor.preprocessing_stats.get('original_shape'),
            'cleaned_shape': self.preprocessor.preprocessing_stats.get('cleaned_shape'),
            'train_shape': X_train.shape,
            'val_shape': X_val.shape,
            'test_shape': X_test.shape,
            'preprocessor_path': preprocessor_path,
            'stats': self.preprocessor.preprocessing_stats
        }
        
        logger.info(f"Preprocessing complete. Preprocessor saved to {preprocessor_path}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def run_feature_selection(self, X_train: np.ndarray, y_train: np.ndarray,
                              n_features: int = 100,
                              methods: Optional[List[str]] = None,
                              feature_names: Optional[List[str]] = None) -> Tuple:
        """
        Run feature selection
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_features: Number of features to select
            methods: Feature selection methods
            feature_names: Optional feature names
            
        Returns:
            Selected features and names
        """
        logger.info("="*70)
        logger.info("STEP 2: FEATURE SELECTION")
        logger.info("="*70)
        
        self.feature_selector = GeneFeatureSelector(n_features=n_features)
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
        
        # Run ensemble feature selection - use fewer methods for speed
        if methods is None:
            methods = ['anova', 'mutual_information']  # Reduced from 3 to 2 methods
        
        logger.info(f"Using feature selection methods: {methods}")
        
        X_selected, selected_names, ensemble_scores = self.feature_selector.ensemble_selection(
            X_train, y_train, feature_names, methods=methods
        )
        
        # Save feature selector
        selector_path = os.path.join(self.models_dir, f'feature_selector_{self.timestamp}.pkl')
        self.feature_selector.save_selector(selector_path)
        
        # Get feature summary
        feature_summary = self.feature_selector.get_feature_summary()
        
        # Store results
        self.results['feature_selection'] = {
            'original_features': X_train.shape[1],
            'selected_features': n_features,
            'n_features_selected': len(selected_names),
            'methods_used': methods,
            'method': 'Ensemble (' + ', '.join(methods) + ')',
            'selected_names': selected_names,
            'selector_path': selector_path,
            'feature_summary': feature_summary
        }
        
        logger.info(f"Feature selection complete. {len(selected_names)} features selected.")
        
        # Visualize feature importance
        importance_array = np.array([ensemble_scores[name] for name in selected_names])
        viz_path = os.path.join(self.viz_dir, f'feature_importance_{self.timestamp}.png')
        self.viz_engine.plot_feature_importance(
            selected_names, importance_array,
            title="Top Selected Genes - Ensemble Feature Importance",
            save_name=f'feature_importance_{self.timestamp}.png'
        )
        self.results['visualization_paths'].append(viz_path)
        
        return X_selected, selected_names, feature_summary
    
    def run_model_training(self, X_train: np.ndarray, y_train: np.ndarray,
                          models: Optional[List[str]] = None,
                          tune_hyperparameters: bool = True) -> Dict:
        """
        Train all models
        
        Args:
            X_train: Training features
            y_train: Training labels
            models: List of models to train
            tune_hyperparameters: Whether to tune hyperparameters
            
        Returns:
            Dictionary of trained models
        """
        logger.info("="*70)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("="*70)
        
        self.trainer = DiseaseGeneClassifier()
        
        if models is None:
            # Default to faster models only
            models = ['random_forest', 'gradient_boosting', 'svm']
        
        logger.info(f"Training models: {models}")
        logger.info(f"Hyperparameter tuning: {'Enabled' if tune_hyperparameters else 'Disabled (faster)'}")
        
        # Train all models
        trained_models = self.trainer.train_all_models(
            X_train, y_train, models=models, tune=tune_hyperparameters
        )
        
        # Save models
        for model_name, model in trained_models.items():
            model_path = os.path.join(self.models_dir, f'{model_name}_{self.timestamp}.pkl')
            self.trainer.save_model(model_name, model_path)
        
        # Store results
        self.results['models'] = {
            'trained_models': list(trained_models.keys()),
            'cv_scores': self.trainer.cv_scores,
            'training_history': self.trainer.training_history
        }
        
        logger.info(f"Model training complete. {len(trained_models)} models trained.")
        
        return trained_models
    
    def run_evaluation(self, X_test: np.ndarray, y_test: np.ndarray,
                      class_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Evaluate all trained models
        
        Args:
            X_test: Test features
            y_test: Test labels
            class_names: Optional class names
            
        Returns:
            Model comparison DataFrame
        """
        logger.info("="*70)
        logger.info("STEP 4: MODEL EVALUATION")
        logger.info("="*70)
        
        # Compare all models
        comparison_df = self.trainer.compare_models(X_test, y_test)
        
        # Store results
        self.results['evaluation'] = {
            'comparison': comparison_df,
            'best_model': comparison_df.iloc[0]['Model']
        }
        
        logger.info(f"Best model: {comparison_df.iloc[0]['Model']} "
                   f"(Accuracy: {comparison_df.iloc[0]['Accuracy']:.4f})")
        
        # Visualize model comparison
        viz_path = os.path.join(self.viz_dir, f'model_comparison_{self.timestamp}.png')
        self.viz_engine.plot_model_comparison(
            comparison_df,
            save_name=f'model_comparison_{self.timestamp}.png'
        )
        self.results['visualization_paths'].append(viz_path)
        
        # ROC-AUC Bar Chart
        viz_path = os.path.join(self.viz_dir, f'roc_auc_bars_{self.timestamp}.png')
        self.viz_engine.plot_roc_auc_bar_chart(
            comparison_df,
            save_name=f'roc_auc_bars_{self.timestamp}.png'
        )
        self.results['visualization_paths'].append(viz_path)
        
        # Precision Bar Chart
        viz_path = os.path.join(self.viz_dir, f'precision_bars_{self.timestamp}.png')
        self.viz_engine.plot_precision_bar_chart(
            comparison_df,
            save_name=f'precision_bars_{self.timestamp}.png'
        )
        self.results['visualization_paths'].append(viz_path)
        
        # Get best model for detailed evaluation
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = self.trainer.best_models.get(best_model_name) or self.trainer.models.get(best_model_name)
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
        
        # Confusion matrix
        viz_path = os.path.join(self.viz_dir, f'confusion_matrix_{self.timestamp}.png')
        self.viz_engine.plot_confusion_matrix(
            y_test, y_pred, class_names=class_names,
            title=f"Confusion Matrix - {best_model_name}",
            save_name=f'confusion_matrix_{self.timestamp}.png'
        )
        self.results['visualization_paths'].append(viz_path)
        
        # ROC curve (single best model)
        if y_pred_proba is not None:
            viz_path = os.path.join(self.viz_dir, f'roc_curve_{self.timestamp}.png')
            self.viz_engine.plot_roc_curve(
                y_test, y_pred_proba,
                model_name=best_model_name,
                class_names=class_names,
                save_name=f'roc_curve_{self.timestamp}.png'
            )
            self.results['visualization_paths'].append(viz_path)
            
            # Multi-model ROC curves comparison
            models_roc_data = {}
            for model_name in comparison_df['Model']:
                model = self.trainer.best_models.get(model_name) or self.trainer.models.get(model_name)
                if model and hasattr(model, 'predict_proba'):
                    model_proba = model.predict_proba(X_test)
                    models_roc_data[model_name] = (y_test, model_proba)
            
            if len(models_roc_data) > 1:
                viz_path = os.path.join(self.viz_dir, f'all_models_roc_{self.timestamp}.png')
                self.viz_engine.plot_multi_model_roc_curves(
                    models_roc_data,
                    save_name=f'all_models_roc_{self.timestamp}.png'
                )
                self.results['visualization_paths'].append(viz_path)
            
            # Precision-Recall curve
            viz_path = os.path.join(self.viz_dir, f'precision_recall_{self.timestamp}.png')
            self.viz_engine.plot_precision_recall_curve(
                y_test, y_pred_proba,
                model_name=best_model_name,
                save_name=f'precision_recall_{self.timestamp}.png'
            )
            self.results['visualization_paths'].append(viz_path)
        
        # PCA visualization
        viz_path = os.path.join(self.viz_dir, f'pca_visualization_{self.timestamp}.png')
        self.viz_engine.plot_pca_visualization(
            X_test, y_test, class_names=class_names,
            save_name=f'pca_visualization_{self.timestamp}.png'
        )
        self.results['visualization_paths'].append(viz_path)
        
        return comparison_df
    
    def export_results(self, X_test: np.ndarray, y_test: np.ndarray,
                      class_names: Optional[List[str]] = None) -> Dict:
        """
        Export all results to CSV, Excel, and PDF
        
        Args:
            X_test: Test features
            y_test: Test labels
            class_names: Optional class names
            
        Returns:
            Dictionary of exported file paths
        """
        logger.info("="*70)
        logger.info("STEP 5: EXPORTING RESULTS")
        logger.info("="*70)
        
        # Get best model
        comparison_df = self.results['evaluation']['comparison']
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = self.trainer.best_models.get(best_model_name) or self.trainer.models.get(best_model_name)
        
        # Get predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
        
        # Create gene impact table
        selected_names = self.results['feature_selection']['selected_names']
        feature_summary = self.results['feature_selection']['feature_summary']
        
        # Get importance scores for selected features
        importance_scores = np.array([
            feature_summary.loc[name, 'ensemble'] if name in feature_summary.index else 0
            for name in selected_names
        ])
        
        gene_impact_table = self.exporter.create_gene_impact_table(
            selected_names,
            importance_scores,
            y_pred,
            y_pred_proba if y_pred_proba is not None else np.zeros((len(y_pred), 1)),
            class_names=class_names
        )
        
        # Get model metrics
        best_model_metrics = self.trainer.evaluate_model(best_model_name, X_test, y_test)
        
        # Export complete results
        exported_files = self.exporter.export_complete_results(
            gene_impact_table=gene_impact_table,
            model_comparison=comparison_df,
            model_metrics=best_model_metrics,
            feature_selection=self.results['feature_selection'],
            preprocessing_stats=self.results['preprocessing']['stats'],
            confusion_matrix=best_model_metrics['confusion_matrix'],
            image_paths=self.results['visualization_paths'],
            base_filename=f'disease_gene_analysis_{self.timestamp}'
        )
        
        return exported_files
    
    def run_complete_pipeline(self, data: pd.DataFrame, label_column: str,
                             n_features: int = 100,
                             models: Optional[List[str]] = None,
                             tune_hyperparameters: bool = True,
                             class_names: Optional[List[str]] = None) -> Dict:
        """
        Run complete end-to-end pipeline
        
        Args:
            data: Input DataFrame
            label_column: Name of label column
            n_features: Number of features to select
            models: List of models to train
            tune_hyperparameters: Whether to tune hyperparameters
            class_names: Optional class names
            
        Returns:
            Dictionary with all results
        """
        logger.info("\n" + "="*70)
        logger.info("STARTING COMPLETE DISEASE GENE DETECTION PIPELINE")
        logger.info("="*70 + "\n")
        
        start_time = datetime.now()
        
        # Step 1: Preprocessing
        X_train, X_val, X_test, y_train, y_val, y_test = self.run_preprocessing(
            data, label_column
        )
        
        # Step 2: Feature Selection
        X_train_selected, selected_names, feature_summary = self.run_feature_selection(
            X_train, y_train, n_features=n_features
        )
        
        # Apply same feature selection to validation and test sets
        selected_indices = [i for i, name in enumerate([f'Feature_{i}' for i in range(X_train.shape[1])]) 
                          if name in selected_names]
        X_val_selected = X_val[:, selected_indices] if len(selected_indices) == n_features else X_val[:, :n_features]
        X_test_selected = X_test[:, selected_indices] if len(selected_indices) == n_features else X_test[:, :n_features]
        
        # Step 3: Model Training
        trained_models = self.run_model_training(
            X_train_selected, y_train, models=models, tune_hyperparameters=tune_hyperparameters
        )
        
        # Step 4: Model Evaluation
        comparison_df = self.run_evaluation(X_test_selected, y_test, class_names=class_names)
        
        # Step 5: Export Results
        exported_files = self.export_results(X_test_selected, y_test, class_names=class_names)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*70)
        logger.info(f"Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Best model: {comparison_df.iloc[0]['Model']}")
        logger.info(f"Best accuracy: {comparison_df.iloc[0]['Accuracy']:.4f}")
        logger.info("="*70 + "\n")
        
        # Store final results
        self.results['exported_files'] = exported_files
        self.results['execution_time'] = duration
        self.results['summary'] = {
            'total_samples': data.shape[0],
            'total_features': data.shape[1],
            'selected_features': n_features,
            'models_trained': len(trained_models),
            'best_model': comparison_df.iloc[0]['Model'],
            'best_accuracy': comparison_df.iloc[0]['Accuracy'],
            'execution_time_seconds': duration
        }
        
        return self.results


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("COMPLETE PIPELINE EXAMPLE")
    print("="*70)
    
    # Create sample dataset
    np.random.seed(42)
    n_samples = 500
    n_genes = 200
    
    # Simulate gene expression data
    gene_data = np.random.randn(n_samples, n_genes)
    gene_columns = [f'GENE_{i+1}' for i in range(n_genes)]
    
    # Create disease labels (3 classes)
    disease_labels = np.random.choice(['Breast_Cancer', 'Lung_Cancer', 'Healthy'], n_samples)
    
    # Create DataFrame
    data = pd.DataFrame(gene_data, columns=gene_columns)
    data['disease_type'] = disease_labels
    
    print(f"\nSample dataset created:")
    print(f"  Samples: {n_samples}")
    print(f"  Genes: {n_genes}")
    print(f"  Classes: {data['disease_type'].unique()}")
    
    # Initialize and run pipeline
    pipeline = CompletePipeline(output_dir='pipeline_output')
    
    results = pipeline.run_complete_pipeline(
        data=data,
        label_column='disease_type',
        n_features=50,
        models=['random_forest', 'svm', 'knn'],
        tune_hyperparameters=False,  # Set to True for better results but slower
        class_names=['Breast Cancer', 'Lung Cancer', 'Healthy']
    )
    
    print("\n" + "="*70)
    print("PIPELINE RESULTS SUMMARY")
    print("="*70)
    for key, value in results['summary'].items():
        print(f"  {key}: {value}")
    
    print(f"\n📁 All outputs saved to: {pipeline.output_dir}/")
