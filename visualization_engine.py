"""
Visualization Engine for Disease Gene Detection
Creates comprehensive visualizations for analysis and results
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - prevents pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix
)
from typing import Dict, List, Optional, Tuple
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class GeneVisualizationEngine:
    """Comprehensive visualization for gene analysis and model results"""
    
    def __init__(self, output_dir: str = 'visualizations'):
        """
        Initialize visualization engine
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_correlation_heatmap(self, data: pd.DataFrame, 
                                 title: str = "Gene Expression Correlation Heatmap",
                                 figsize: Tuple[int, int] = (14, 12),
                                 save_name: Optional[str] = None,
                                 cluster: bool = True):
        """
        Create correlation heatmap for gene expression data with optional clustering
        
        Args:
            data: DataFrame with gene expression data
            title: Plot title
            figsize: Figure size
            save_name: Filename to save plot
            cluster: Whether to apply hierarchical clustering
        """
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        if cluster:
            # Create clustered heatmap
            g = sns.clustermap(corr_matrix, annot=False, cmap='coolwarm', 
                             center=0, vmin=-1, vmax=1,
                             cbar_kws={'label': 'Correlation Coefficient'},
                             figsize=figsize, linewidths=0.5)
            g.fig.suptitle(title, fontsize=16, fontweight='bold', y=1.01)
            
            if save_name:
                g.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
                print(f"Saved: {save_name}")
            
            plt.close()  # Save only, no display
        else:
            plt.figure(figsize=figsize)
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', 
                       center=0, vmin=-1, vmax=1,
                       cbar_kws={'label': 'Correlation Coefficient'})
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('Genes', fontsize=12)
            plt.ylabel('Genes', fontsize=12)
            plt.tight_layout()
            
            if save_name:
                plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
                print(f"Saved: {save_name}")
            
            plt.close()  # Save only, no display
    
    def plot_feature_importance(self, feature_names: List[str], 
                               importance_scores: np.ndarray,
                               title: str = "Top Gene Feature Importance",
                               top_n: int = 20,
                               save_name: Optional[str] = None):
        """
        Create bar chart for feature importance
        
        Args:
            feature_names: List of feature names
            importance_scores: Importance scores
            title: Plot title
            top_n: Number of top features to show
            save_name: Filename to save plot
        """
        # Sort by importance
        indices = np.argsort(importance_scores)[-top_n:][::-1]
        top_features = [feature_names[i] for i in indices]
        top_scores = importance_scores[indices]
        
        # Create bar plot
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        bars = plt.barh(range(len(top_features)), top_scores, color=colors)
        
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.ylabel('Genes', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            plt.text(score, i, f' {score:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
            print(f"Saved: {save_name}")
        
        plt.close()  # Save only, no display
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             title: str = "Confusion Matrix",
                             save_name: Optional[str] = None):
        """
        Create confusion matrix heatmap
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class names
            title: Plot title
            save_name: Filename to save plot
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names if class_names else range(len(cm)),
                   yticklabels=class_names if class_names else range(len(cm)),
                   cbar_kws={'label': 'Count'})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
            print(f"Saved: {save_name}")
        
        plt.close()  # Save only, no display
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      model_name: str = "Model",
                      class_names: Optional[List[str]] = None,
                      save_name: Optional[str] = None):
        """
        Create ROC curve (binary or multi-class)
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            model_name: Model name for legend
            class_names: Optional class names
            save_name: Filename to save plot
        """
        n_classes = y_pred_proba.shape[1] if len(y_pred_proba.shape) > 1 else 1
        
        plt.figure(figsize=(10, 8))
        
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
        else:
            # Multi-class classification
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            # Compute ROC curve for each class
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                class_label = class_names[i] if class_names else f'Class {i}'
                plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title(f'ROC Curve - {model_name}', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
            print(f"Saved: {save_name}")
        
        plt.close()  # Save only, no display
    
    def plot_multi_model_roc_curves(self, models_results: Dict[str, Tuple],
                                   title: str = "ROC Curves - All Models",
                                   save_name: Optional[str] = None):
        """
        Plot ROC curves for multiple models on same plot
        
        Args:
            models_results: Dict with model_name: (y_true, y_pred_proba) tuples
            title: Plot title
            save_name: Filename to save plot
        """
        plt.figure(figsize=(12, 9))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(models_results)))
        
        for idx, (model_name, (y_true, y_pred_proba)) in enumerate(models_results.items()):
            # Binary classification (use probability of positive class)
            if len(y_pred_proba.shape) > 1:
                y_scores = y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba[:, -1]
            else:
                y_scores = y_pred_proba
            
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=colors[idx], lw=2.5,
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', 
                label='Random Classifier (AUC = 0.500)', alpha=0.6)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=11, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle=':')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
            print(f"Saved: {save_name}")
        
        plt.close()  # Save only, no display
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   model_name: str = "Model",
                                   save_name: Optional[str] = None):
        """
        Create Precision-Recall curve
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            model_name: Model name for title
            save_name: Filename to save plot
        """
        plt.figure(figsize=(10, 8))
        
        # For binary classification
        if len(y_pred_proba.shape) > 1:
            y_scores = y_pred_proba[:, 1]
        else:
            y_scores = y_pred_proba
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        
        plt.plot(recall, precision, color='blue', lw=2, label=model_name)
        plt.fill_between(recall, precision, alpha=0.2, color='blue')
        
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title(f'Precision-Recall Curve - {model_name}', fontsize=16, fontweight='bold')
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
            print(f"Saved: {save_name}")
        
        plt.close()  # Save only, no display
    
    def plot_roc_auc_bar_chart(self, comparison_df: pd.DataFrame,
                              title: str = "ROC-AUC Score Comparison",
                              save_name: Optional[str] = None):
        """
        Create bar chart specifically for ROC-AUC scores
        
        Args:
            comparison_df: DataFrame with model comparison metrics
            title: Plot title
            save_name: Filename to save plot
        """
        plt.figure(figsize=(12, 6))
        
        # Sort by ROC-AUC
        data = comparison_df.sort_values('ROC-AUC', ascending=True)
        
        # Create color gradient based on scores (avoid division by zero)
        max_score = data['ROC-AUC'].max()
        if max_score > 0:
            colors = plt.cm.RdYlGn(data['ROC-AUC'] / max_score)
        else:
            colors = ['#cccccc'] * len(data)  # Gray if all scores are 0
        
        # Create horizontal bar plot
        bars = plt.barh(data['Model'], data['ROC-AUC'], color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, value in zip(bars, data['ROC-AUC']):
            plt.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.4f}', va='center', fontsize=11, fontweight='bold')
        
        plt.xlabel('ROC-AUC Score', fontsize=13, fontweight='bold')
        plt.ylabel('Model', fontsize=13, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlim([0, 1.1])
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add timestamp to verify fresh generation
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        plt.text(0.99, 0.01, f'Generated: {timestamp}', transform=plt.gcf().transFigure, 
                fontsize=8, color='gray', ha='right', va='bottom')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
            print(f"Saved: {save_name}")
        
        plt.close()  # Save only, no display
    
    def plot_precision_bar_chart(self, comparison_df: pd.DataFrame,
                                title: str = "Precision Score Comparison",
                                save_name: Optional[str] = None):
        """
        Create bar chart specifically for Precision scores
        
        Args:
            comparison_df: DataFrame with model comparison metrics
            title: Plot title
            save_name: Filename to save plot
        """
        plt.figure(figsize=(12, 6))
        
        # Sort by Precision
        data = comparison_df.sort_values('Precision', ascending=True)
        
        # Create color gradient based on scores
        colors = plt.cm.Blues(data['Precision'] / data['Precision'].max())
        
        # Create horizontal bar plot
        bars = plt.barh(data['Model'], data['Precision'], color=colors, edgecolor='navy', linewidth=1.5)
        
        # Add value labels
        for bar, value in zip(bars, data['Precision']):
            plt.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.4f}', va='center', fontsize=11, fontweight='bold')
        
        plt.xlabel('Precision Score', fontsize=13, fontweight='bold')
        plt.ylabel('Model', fontsize=13, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlim([0, 1.1])
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
            print(f"Saved: {save_name}")
        
        plt.close()  # Save only, no display
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                             title: str = "Model Performance Comparison",
                             save_name: Optional[str] = None):
        """
        Create bar chart comparing multiple models
        
        Args:
            comparison_df: DataFrame with model comparison metrics
            title: Plot title
            save_name: Filename to save plot
        """
        metrics = [col for col in comparison_df.columns if col != 'Model']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            if idx < len(axes):
                ax = axes[idx]
                
                # Sort by metric
                data = comparison_df.sort_values(metric, ascending=False)
                
                # Create bar plot
                colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
                bars = ax.barh(data['Model'], data[metric], color=colors)
                
                ax.set_xlabel(metric, fontsize=11, fontweight='bold')
                ax.set_title(f'{metric} by Model', fontsize=12, fontweight='bold')
                ax.invert_yaxis()
                
                # Add value labels
                for bar, value in zip(bars, data[metric]):
                    ax.text(value, bar.get_y() + bar.get_height()/2, 
                           f' {value:.4f}', va='center', fontsize=9)
        
        # Hide unused subplots
        for idx in range(len(metrics), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=18, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
            print(f"Saved: {save_name}")
        
        plt.close()  # Save only, no display
    
    def plot_training_history(self, history: Dict,
                             title: str = "Model Training History",
                             save_name: Optional[str] = None):
        """
        Plot training and validation loss/accuracy over epochs
        
        Args:
            history: Training history dictionary
            title: Plot title
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        if 'loss' in history and 'val_loss' in history:
            axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
            axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
            axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
            axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
            axes[0].legend(loc='best')
            axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        if 'accuracy' in history and 'val_accuracy' in history:
            axes[1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
            axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
            axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
            axes[1].legend(loc='best')
            axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
            print(f"Saved: {save_name}")
        
        plt.close()  # Save only, no display
    
    def plot_feature_distribution(self, data: pd.DataFrame,
                                  feature_name: str,
                                  class_column: str,
                                  title: Optional[str] = None,
                                  save_name: Optional[str] = None):
        """
        Plot feature distribution across different classes
        
        Args:
            data: DataFrame with data
            feature_name: Feature to plot
            class_column: Column with class labels
            title: Plot title
            save_name: Filename to save plot
        """
        plt.figure(figsize=(12, 6))
        
        classes = data[class_column].unique()
        
        for class_label in classes:
            class_data = data[data[class_column] == class_label][feature_name]
            plt.hist(class_data, alpha=0.5, label=f'{class_label}', bins=30)
        
        plt.xlabel(feature_name, fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title(title or f'Distribution of {feature_name}', fontsize=16, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
            print(f"Saved: {save_name}")
        
        plt.close()  # Save only, no display
    
    def plot_pca_visualization(self, X: np.ndarray, y: np.ndarray,
                              class_names: Optional[List[str]] = None,
                              title: str = "PCA Visualization",
                              save_name: Optional[str] = None):
        """
        Create 2D PCA visualization of data
        
        Args:
            X: Feature matrix
            y: Labels
            class_names: Optional class names
            title: Plot title
            save_name: Filename to save plot
        """
        from sklearn.decomposition import PCA
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        
        # Plot each class
        unique_classes = np.unique(y)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_classes)))
        
        for i, class_label in enumerate(unique_classes):
            mask = y == class_label
            label_name = class_names[i] if class_names else f'Class {class_label}'
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[i]], label=label_name, 
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', 
                  fontsize=12, fontweight='bold')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', 
                  fontsize=12, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
            print(f"Saved: {save_name}")
        
        plt.close()  # Save only, no display
    
    def create_comprehensive_report(self, 
                                   comparison_df: pd.DataFrame,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_pred_proba: np.ndarray,
                                   feature_importance: Dict,
                                   class_names: Optional[List[str]] = None,
                                   report_name: str = "model_report",
                                   models_roc_data: Optional[Dict] = None):
        """
        Create comprehensive visualization report with all metrics
        
        Args:
            comparison_df: Model comparison DataFrame
            y_true: True labels
            y_pred: Predictions
            y_pred_proba: Prediction probabilities
            feature_importance: Feature importance dictionary
            class_names: Optional class names
            report_name: Base name for saved files
            models_roc_data: Optional dict for multi-model ROC curves
        """
        print("Creating comprehensive visualization report...")
        
        # 1. Model comparison (all metrics)
        self.plot_model_comparison(comparison_df, save_name=f"{report_name}_comparison.png")
        
        # 2. ROC-AUC Bar Chart
        self.plot_roc_auc_bar_chart(comparison_df, save_name=f"{report_name}_roc_auc_bars.png")
        
        # 3. Precision Bar Chart
        self.plot_precision_bar_chart(comparison_df, save_name=f"{report_name}_precision_bars.png")
        
        # 4. Confusion matrix
        self.plot_confusion_matrix(y_true, y_pred, class_names, 
                                  save_name=f"{report_name}_confusion_matrix.png")
        
        # 5. ROC curve (single model)
        self.plot_roc_curve(y_true, y_pred_proba, class_names=class_names,
                          save_name=f"{report_name}_roc_curve.png")
        
        # 6. Multi-model ROC curves (if data provided)
        if models_roc_data:
            self.plot_multi_model_roc_curves(models_roc_data, 
                                            save_name=f"{report_name}_all_roc_curves.png")
        
        # 7. Precision-Recall curve
        self.plot_precision_recall_curve(y_true, y_pred_proba,
                                        save_name=f"{report_name}_precision_recall.png")
        
        # 8. Feature importance
        if feature_importance:
            for method, (names, scores) in feature_importance.items():
                self.plot_feature_importance(names, scores, 
                                            title=f"Feature Importance - {method}",
                                            save_name=f"{report_name}_importance_{method}.png")
        
        print(f"✅ Comprehensive report saved to {self.output_dir}/")
        print(f"   - Model comparison charts")
        print(f"   - ROC-AUC bar chart")
        print(f"   - Precision bar chart")
        print(f"   - Confusion matrix")
        print(f"   - ROC curves")
        print(f"   - Precision-Recall curve")
        print(f"   - Feature importance plots")


# Example usage
if __name__ == "__main__":
    # Create visualization engine
    viz = GeneVisualizationEngine(output_dir='visualizations')
    
    # Create sample data
    np.random.seed(42)
    
    # Sample gene expression data
    n_samples = 100
    n_genes = 50
    data = pd.DataFrame(np.random.randn(n_samples, n_genes),
                       columns=[f'GENE_{i+1}' for i in range(n_genes)])
    
    # Test visualizations
    print("Creating sample visualizations...")
    
    # 1. Correlation heatmap (smaller subset for visibility)
    viz.plot_correlation_heatmap(data.iloc[:, :20], save_name='sample_heatmap.png')
    
    # 2. Feature importance
    feature_names = [f'GENE_{i+1}' for i in range(20)]
    importance_scores = np.random.rand(20)
    viz.plot_feature_importance(feature_names, importance_scores, save_name='sample_importance.png')
    
    # 3. Model comparison
    comparison_df = pd.DataFrame({
        'Model': ['SVM', 'Random Forest', 'ANN', 'KNN'],
        'Accuracy': [0.85, 0.92, 0.88, 0.83],
        'Precision': [0.84, 0.91, 0.87, 0.82],
        'Recall': [0.83, 0.90, 0.86, 0.81],
        'F1-Score': [0.83, 0.90, 0.86, 0.81],
        'ROC-AUC': [0.88, 0.94, 0.90, 0.85]
    })
    viz.plot_model_comparison(comparison_df, save_name='sample_comparison.png')
    
    print("Sample visualizations created!")
