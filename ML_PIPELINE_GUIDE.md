# Complete Machine Learning Pipeline - Quick Start Guide

## 🎯 New Features Implemented

### ✅ **Feature Selection Module** (`feature_selector.py`)
- **Statistical Methods:**
  - ✓ ANOVA F-test
  - ✓ Chi-square test
  - ✓ Pearson correlation
  - ✓ Spearman correlation
- **ML-based Methods:**
  - ✓ Mutual Information
  - ✓ Recursive Feature Elimination (RFE)
  - ✓ Tree-based feature importance
- **Ensemble Selection:** Combines multiple methods for robust feature selection

### ✅ **Model Training & Classification** (`model_trainer.py`)
- **Classifiers Implemented:**
  - ✓ Support Vector Machine (SVM)
  - ✓ Random Forest
  - ✓ Artificial Neural Networks (ANN/MLP)
  - ✓ K-Nearest Neighbors (KNN)
  - ✓ Gradient Boosting
  - ✓ Logistic Regression
- **Advanced Features:**
  - ✓ K-fold cross-validation (Stratified)
  - ✓ GridSearchCV for hyperparameter tuning
  - ✓ RandomizedSearchCV for faster optimization
  - ✓ Automated model comparison

### ✅ **Evaluation Metrics** (Built into `model_trainer.py`)
- ✓ Accuracy, Precision, Recall, F1-Score
- ✓ ROC-AUC (binary and multi-class)
- ✓ Confusion Matrix
- ✓ Classification Report

### ✅ **Visualization Engine** (`visualization_engine.py`)
- **Comprehensive Visualizations:**
  - ✓ Gene correlation heatmaps
  - ✓ Feature importance bar charts
  - ✓ Confusion matrix heatmaps
  - ✓ ROC curves (binary and multi-class)
  - ✓ Precision-Recall curves
  - ✓ Model comparison charts
  - ✓ Training history plots
  - ✓ PCA visualization
  - ✓ Feature distribution plots

### ✅ **Results Export & Reporting** (`results_exporter.py`)
- **Export Formats:**
  - ✓ CSV (individual tables)
  - ✓ Excel (multiple sheets in one file)
  - ✓ PDF (comprehensive report with ReportLab)
  - ✓ JSON (metadata)
- **Report Contents:**
  - ✓ High-impact genes table
  - ✓ Disease classification probabilities
  - ✓ Model performance comparison
  - ✓ Summary statistics
  - ✓ Embedded visualizations

### ✅ **Complete Pipeline** (`complete_pipeline.py`)
- **End-to-End Workflow:**
  1. Data preprocessing
  2. Feature selection
  3. Model training
  4. Model evaluation
  5. Results export
- **One-Command Execution:** Run entire analysis with single function call

---

## 🚀 Quick Start

### Step 1: Install Additional Dependencies

```bash
pip install matplotlib seaborn reportlab
```

Or install all at once:
```bash
pip install -r requirements.txt
```

### Step 2: Run Complete Pipeline

#### Option A: Use Complete Pipeline (Recommended)

```python
from complete_pipeline import CompletePipeline
import pandas as pd

# Load your data
data = pd.read_csv('your_gene_data.csv')

# Initialize pipeline
pipeline = CompletePipeline(output_dir='my_analysis')

# Run complete analysis
results = pipeline.run_complete_pipeline(
    data=data,
    label_column='disease_type',
    n_features=100,  # Number of top genes to select
    models=['random_forest', 'svm', 'ann'],
    tune_hyperparameters=True,
    class_names=['Breast Cancer', 'Lung Cancer', 'Healthy']
)

print(f"Best Model: {results['summary']['best_model']}")
print(f"Accuracy: {results['summary']['best_accuracy']:.4f}")
```

#### Option B: Step-by-Step Workflow

```python
# 1. Feature Selection
from feature_selector import GeneFeatureSelector
import numpy as np

selector = GeneFeatureSelector(n_features=100)

# Use ensemble method (combines ANOVA, MI, Tree importance)
X_selected, selected_genes, scores = selector.ensemble_selection(
    X_train, y_train, feature_names=gene_names,
    methods=['anova', 'mutual_information', 'tree_importance']
)

print(f"Selected {len(selected_genes)} genes")
print(f"Top 10 genes: {selected_genes[:10]}")

# 2. Train Models
from model_trainer import DiseaseGeneClassifier

trainer = DiseaseGeneClassifier()

# Train all models with hyperparameter tuning
models = trainer.train_all_models(
    X_selected, y_train,
    models=['random_forest', 'svm', 'ann', 'knn'],
    tune=True
)

# Compare models
comparison = trainer.compare_models(X_test_selected, y_test)
print(comparison)

# 3. Visualize Results
from visualization_engine import GeneVisualizationEngine

viz = GeneVisualizationEngine(output_dir='visualizations')

# Plot feature importance
viz.plot_feature_importance(selected_genes, scores)

# Plot confusion matrix
viz.plot_confusion_matrix(y_test, y_pred, class_names=['BC', 'LC', 'Healthy'])

# Plot ROC curve
viz.plot_roc_curve(y_test, y_pred_proba, class_names=['BC', 'LC', 'Healthy'])

# 4. Export Results
from results_exporter import ResultsExporter

exporter = ResultsExporter(output_dir='results')

# Create gene impact table
gene_table = exporter.create_gene_impact_table(
    selected_genes, scores, y_pred, y_pred_proba,
    class_names=['Breast Cancer', 'Lung Cancer', 'Healthy']
)

# Export everything
exporter.export_complete_results(
    gene_impact_table=gene_table,
    model_comparison=comparison,
    model_metrics={'accuracy': 0.92, 'precision': 0.91},
    feature_selection={'original_features': 500, 'selected_features': 100},
    preprocessing_stats={},
    confusion_matrix=confusion_matrix(y_test, y_pred),
    base_filename='my_analysis'
)
```

---

## 📊 Example Output

After running the complete pipeline, you'll get:

### Directory Structure:
```
pipeline_output/
├── models/
│   ├── preprocessor_20251117_120000.pkl
│   ├── feature_selector_20251117_120000.pkl
│   ├── random_forest_20251117_120000.pkl
│   ├── svm_20251117_120000.pkl
│   └── ann_20251117_120000.pkl
│
├── visualizations/
│   ├── feature_importance_20251117_120000.png
│   ├── confusion_matrix_20251117_120000.png
│   ├── roc_curve_20251117_120000.png
│   ├── model_comparison_20251117_120000.png
│   └── pca_visualization_20251117_120000.png
│
└── results/
    ├── disease_gene_analysis_20251117_120000_gene_impact.csv
    ├── disease_gene_analysis_20251117_120000_model_comparison.csv
    ├── disease_gene_analysis_20251117_120000_complete.xlsx
    ├── disease_gene_analysis_20251117_120000_report.pdf
    └── disease_gene_analysis_20251117_120000_metadata.json
```

### Sample Results:

**Top Genes (from CSV):**
| Gene_Name | Importance_Score | Rank | Breast_Cancer_Probability | Lung_Cancer_Probability | Healthy_Probability |
|-----------|-----------------|------|---------------------------|------------------------|---------------------|
| GENE_45   | 0.8523          | 1    | 0.75                      | 0.15                   | 0.10                |
| GENE_122  | 0.8234          | 2    | 0.72                      | 0.18                   | 0.10                |
| GENE_89   | 0.8012          | 3    | 0.68                      | 0.22                   | 0.10                |

**Model Comparison:**
| Model          | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------|----------|-----------|--------|----------|---------|
| Random Forest  | 0.9200   | 0.9100    | 0.9000 | 0.9000   | 0.9400  |
| SVM            | 0.8800   | 0.8700    | 0.8600 | 0.8600   | 0.9000  |
| ANN            | 0.8500   | 0.8400    | 0.8300 | 0.8300   | 0.8800  |

---

## 🔧 Configuration Options

### Feature Selection Configuration

```python
# Choose specific methods
selector = GeneFeatureSelector(n_features=50)

# ANOVA only
X_anova, genes_anova, scores = selector.anova_selection(X, y, feature_names)

# Mutual Information only
X_mi, genes_mi, scores = selector.mutual_information_selection(X, y, feature_names)

# Custom ensemble
X_ensemble, genes, scores = selector.ensemble_selection(
    X, y, feature_names,
    methods=['anova', 'pearson', 'tree_importance']
)
```

### Model Training Configuration

```python
# Quick training (no tuning)
trainer.train_all_models(X_train, y_train, tune=False)

# Hyperparameter tuning with GridSearch
trainer.hyperparameter_tuning('random_forest', X_train, y_train, method='grid')

# Hyperparameter tuning with RandomizedSearch (faster)
trainer.hyperparameter_tuning('svm', X_train, y_train, method='random', n_iter=50)

# Custom model configuration
custom_params = {'n_estimators': 200, 'max_depth': 30}
trainer.train_model('random_forest', X_train, y_train, custom_params=custom_params)
```

### Visualization Configuration

```python
viz = GeneVisualizationEngine(output_dir='my_plots')

# Customize plots
viz.plot_feature_importance(
    feature_names=genes,
    importance_scores=scores,
    title="My Custom Title",
    top_n=30,  # Show top 30 instead of 20
    save_name='my_importance.png'
)

# Create comprehensive report
viz.create_comprehensive_report(
    comparison_df=model_comparison,
    y_true=y_test,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
    feature_importance={'ensemble': (genes, scores)},
    class_names=['Disease1', 'Disease2', 'Healthy'],
    report_name='full_analysis'
)
```

---

## 📈 Performance Tips

### For Large Datasets (> 10,000 samples):

1. **Feature Selection:**
   - Use faster methods first: `['anova', 'mutual_information']`
   - Skip RFE for very large datasets
   - Use tree importance with fewer trees: `n_estimators=50`

2. **Model Training:**
   - Use RandomizedSearchCV instead of GridSearchCV
   - Reduce `n_iter` parameter
   - Train fewer models initially

3. **Hyperparameter Tuning:**
   ```python
   # Fast tuning
   trainer.hyperparameter_tuning(
       'random_forest', X_train, y_train,
       method='random', n_iter=20, cv=3
   )
   ```

### For Small Datasets (< 1,000 samples):

1. **Use more cross-validation folds:** `cv=10`
2. **Enable full hyperparameter tuning**
3. **Use ensemble feature selection** for robust results

---

## 🎯 Real-World Example

```python
# Complete analysis with your datasets
from complete_pipeline import CompletePipeline
import pandas as pd
from data_collector import DataCollector

# 1. Load your disease datasets
collector = DataCollector()

breast_cancer = collector.load_local_file('breast_cancer_data.csv')
lung_cancer = collector.load_local_file('lung_cancer_data.csv')
healthy = collector.load_local_file('healthy_samples.csv')

# 2. Merge datasets
datasets = [breast_cancer, lung_cancer, healthy]
labels = ['breast_cancer', 'lung_cancer', 'healthy']
merged_data = collector.merge_datasets(datasets, labels)

# 3. Run complete pipeline
pipeline = CompletePipeline(output_dir='disease_analysis_2025')

results = pipeline.run_complete_pipeline(
    data=merged_data,
    label_column='disease_type',
    n_features=100,
    models=['random_forest', 'svm', 'gradient_boosting', 'ann'],
    tune_hyperparameters=True,
    class_names=['Breast Cancer', 'Lung Cancer', 'Healthy']
)

# 4. Review results
print(f"\n{'='*60}")
print(f"Analysis Complete!")
print(f"{'='*60}")
print(f"Best Model: {results['summary']['best_model']}")
print(f"Accuracy: {results['summary']['best_accuracy']:.2%}")
print(f"Top 10 Genes: {results['feature_selection']['selected_names'][:10]}")
print(f"\nAll results saved to: disease_analysis_2025/")
print(f"{'='*60}\n")
```

---

## 📝 Next Steps

1. **Install visualization libraries:**
   ```bash
   pip install matplotlib seaborn reportlab
   ```

2. **Test with sample data:**
   ```bash
   python complete_pipeline.py
   ```

3. **Run with your data:**
   - Use the examples above
   - Adjust parameters as needed
   - Review generated reports

4. **Optimize performance:**
   - Start with `tune_hyperparameters=False` for speed
   - Enable tuning for final analysis
   - Select specific models based on dataset size

---

## ✅ All Features Implemented

- [x] Statistical feature selection (ANOVA, Chi-square, Correlation)
- [x] ML-based feature selection (RFE, Mutual Info, Tree importance)
- [x] 6 ML classifiers (SVM, RF, ANN, KNN, GB, LR)
- [x] K-fold cross-validation
- [x] Hyperparameter tuning (Grid & Randomized)
- [x] Comprehensive metrics (Precision, Recall, F1, ROC-AUC)
- [x] Confusion matrix
- [x] 8+ visualization types
- [x] ROC and Precision-Recall curves
- [x] Export to CSV, Excel, PDF
- [x] Automated report generation
- [x] Complete pipeline integration

**Your ML pipeline is ready for production use! 🎉**
