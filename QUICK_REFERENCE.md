# ⚡ Quick Reference Card - Disease Gene Detection ML Pipeline

## 🚀 One-Line Quick Starts

### Run Complete Pipeline:
```python
from complete_pipeline import CompletePipeline
import pandas as pd

pipeline = CompletePipeline('output')
results = pipeline.run_complete_pipeline(pd.read_csv('data.csv'), 'disease_type', n_features=100, models=['random_forest', 'svm'])
```

### Start Web Interface:
```bash
python app.py
```
Open: http://localhost:5000

### Run Tests:
```bash
python test_ml_pipeline.py
```

---

## 📦 Installation (One Command)

```bash
pip install -r requirements.txt
```

Or run:
```bash
install_and_run.bat
```

---

## 🎯 Core Modules Cheat Sheet

### 1. Feature Selection
```python
from feature_selector import GeneFeatureSelector

selector = GeneFeatureSelector(n_features=50)

# Single method
X_sel, genes, scores = selector.anova_selection(X, y, feature_names)

# Ensemble (recommended)
X_sel, genes, scores = selector.ensemble_selection(X, y, feature_names, 
                        methods=['anova', 'mutual_information', 'tree_importance'])
```

### 2. Model Training
```python
from model_trainer import DiseaseGeneClassifier

trainer = DiseaseGeneClassifier()

# Train one model
model = trainer.train_model('random_forest', X_train, y_train)

# Train all models with tuning
models = trainer.train_all_models(X_train, y_train, 
         models=['random_forest', 'svm', 'ann'], tune=True)

# Evaluate
metrics = trainer.evaluate_model('random_forest', X_test, y_test)

# Compare all models
comparison = trainer.compare_models(X_test, y_test, 
             class_names=['BC', 'LC', 'Healthy'])
```

### 3. Visualization
```python
from visualization_engine import GeneVisualizationEngine

viz = GeneVisualizationEngine(output_dir='plots')

# Individual plots
viz.plot_feature_importance(gene_names, importance_scores)
viz.plot_confusion_matrix(y_true, y_pred, class_names)
viz.plot_roc_curve(y_true, y_proba, class_names)

# Comprehensive report (all plots)
viz.create_comprehensive_report(
    comparison_df, y_true, y_pred, y_proba,
    feature_importance, class_names, 'report_name'
)
```

### 4. Results Export
```python
from results_exporter import ResultsExporter

exporter = ResultsExporter(output_dir='results')

# Create gene impact table
gene_table = exporter.create_gene_impact_table(
    gene_names, importance_scores, y_pred, y_proba, class_names
)

# Export everything
exporter.export_complete_results(
    gene_impact_table=gene_table,
    model_comparison=comparison_df,
    model_metrics=metrics_dict,
    base_filename='my_analysis'
)
```

### 5. Complete Pipeline
```python
from complete_pipeline import CompletePipeline

pipeline = CompletePipeline(output_dir='full_analysis')

results = pipeline.run_complete_pipeline(
    data=dataframe,
    label_column='disease_type',
    n_features=100,
    models=['random_forest', 'svm', 'ann', 'knn'],
    tune_hyperparameters=True,
    class_names=['Disease1', 'Disease2', 'Healthy']
)

# Access results
print(f"Best: {results['summary']['best_model']}")
print(f"Accuracy: {results['summary']['best_accuracy']}")
print(f"Top genes: {results['feature_selection']['selected_names'][:10]}")
```

---

## 🔧 Common Configurations

### Fast Mode (No Tuning):
```python
results = pipeline.run_complete_pipeline(
    data=data,
    label_column='disease_type',
    n_features=50,
    models=['random_forest'],
    tune_hyperparameters=False  # Skip tuning for speed
)
```

### High Performance Mode (Full Tuning):
```python
results = pipeline.run_complete_pipeline(
    data=data,
    label_column='disease_type',
    n_features=200,
    models=['random_forest', 'svm', 'gradient_boosting', 'ann'],
    tune_hyperparameters=True,
    cv=10  # More cross-validation folds
)
```

### Binary Classification:
```python
results = pipeline.run_complete_pipeline(
    data=data,
    label_column='disease_type',
    n_features=100,
    models=['svm', 'random_forest'],
    class_names=['Cancer', 'Healthy']
)
```

---

## 📊 Available Models

| Model Name | Code | Best For |
|------------|------|----------|
| Support Vector Machine | `'svm'` | High-dimensional data |
| Random Forest | `'random_forest'` | Feature importance |
| Neural Network | `'ann'` | Complex patterns |
| K-Nearest Neighbors | `'knn'` | Simple classification |
| Gradient Boosting | `'gradient_boosting'` | Best accuracy |
| Logistic Regression | `'logistic_regression'` | Interpretability |

---

## 🎨 Visualization Types

```python
# All available plot functions:
viz.plot_correlation_heatmap(data, feature_names)
viz.plot_feature_importance(names, scores)
viz.plot_confusion_matrix(y_true, y_pred, classes)
viz.plot_roc_curve(y_true, y_proba, classes)
viz.plot_precision_recall_curve(y_true, y_proba, classes)
viz.plot_model_comparison(comparison_df)
viz.plot_training_history(history_dict)
viz.plot_pca_visualization(X, y, classes)
viz.plot_feature_distributions(data, feature_names, y, classes)
```

---

## 📁 Output Files Generated

### Models Directory:
- `preprocessor_TIMESTAMP.pkl` - Data preprocessor
- `feature_selector_TIMESTAMP.pkl` - Feature selector
- `[model_name]_TIMESTAMP.pkl` - Trained models

### Visualizations Directory:
- `feature_importance_TIMESTAMP.png`
- `confusion_matrix_TIMESTAMP.png`
- `roc_curve_TIMESTAMP.png`
- `precision_recall_curve_TIMESTAMP.png`
- `model_comparison_TIMESTAMP.png`
- `pca_visualization_TIMESTAMP.png`

### Results Directory:
- `[name]_gene_impact.csv` - Top genes with scores
- `[name]_model_comparison.csv` - Model performance
- `[name]_complete.xlsx` - All results (Excel)
- `[name]_report.pdf` - Comprehensive PDF report
- `[name]_metadata.json` - Configuration details

---

## ⚙️ Feature Selection Methods

```python
selector = GeneFeatureSelector(n_features=50)

# Statistical Methods:
selector.anova_selection(X, y, names)           # ANOVA F-test
selector.chi_square_selection(X, y, names)      # Chi-square
selector.correlation_selection(X, y, names, 'pearson')   # Pearson
selector.correlation_selection(X, y, names, 'spearman')  # Spearman

# ML Methods:
selector.mutual_information_selection(X, y, names)  # Mutual Info
selector.rfe_selection(X, y, names)                 # RFE
selector.tree_importance_selection(X, y, names)     # Tree-based

# Best Practice:
selector.ensemble_selection(X, y, names,
    methods=['anova', 'mutual_information', 'tree_importance']
)
```

---

## 📈 Evaluation Metrics Returned

```python
metrics = {
    'accuracy': 0.92,
    'precision': 0.91,
    'recall': 0.90,
    'f1': 0.90,
    'roc_auc': 0.94,
    'confusion_matrix': [[...], [...]],
    'classification_report': "..."
}
```

---

## 🎯 Real-World Example (Your Data)

```python
from complete_pipeline import CompletePipeline
from data_collector import DataCollector

# 1. Load datasets
collector = DataCollector()
datasets = [
    collector.load_local_file('breast_cancer_data1.csv'),
    collector.load_local_file('breast_cancer_data2.csv'),
    collector.load_local_file('lung_cancer_data.csv'),
]
labels = ['breast_cancer', 'breast_cancer', 'lung_cancer']
data = collector.merge_datasets(datasets, labels)

# 2. Run pipeline
pipeline = CompletePipeline('cancer_analysis')
results = pipeline.run_complete_pipeline(
    data=data,
    label_column='disease_type',
    n_features=150,
    models=['random_forest', 'svm', 'gradient_boosting'],
    tune_hyperparameters=True,
    class_names=['Breast Cancer', 'Lung Cancer']
)

# 3. View results
print(f"Best Model: {results['summary']['best_model']}")
print(f"Accuracy: {results['summary']['best_accuracy']:.2%}")
print(f"\nTop 10 Genes:")
for i, gene in enumerate(results['feature_selection']['selected_names'][:10], 1):
    print(f"{i}. {gene}")

# All results saved to: cancer_analysis/
```

---

## ⚡ Performance Tips

### Large Datasets (>10,000 samples):
```python
# Use faster methods
results = pipeline.run_complete_pipeline(
    data=data,
    label_column='disease',
    n_features=50,  # Fewer features
    models=['random_forest'],  # One fast model
    tune_hyperparameters=False  # No tuning
)
```

### Small Datasets (<1,000 samples):
```python
# Use more validation
results = pipeline.run_complete_pipeline(
    data=data,
    label_column='disease',
    n_features=100,
    models=['svm', 'random_forest', 'gradient_boosting'],
    tune_hyperparameters=True,
    cv=10  # More cross-validation folds
)
```

---

## 🐛 Troubleshooting

### Import Error:
```bash
pip install -r requirements.txt
```

### Missing Visualization:
```bash
pip install matplotlib seaborn reportlab
```

### Memory Error (Large Dataset):
- Reduce `n_features`
- Use fewer models
- Set `tune_hyperparameters=False`

### Slow Training:
- Set `tune_hyperparameters=False`
- Use `method='random'` for tuning
- Reduce `cv` parameter

---

## 📚 Documentation Files

- **`README.md`** - Overview
- **`ML_PIPELINE_GUIDE.md`** - Detailed guide ⭐
- **`IMPLEMENTATION_SUMMARY.md`** - What was built
- **`GETTING_STARTED.md`** - Tutorial
- **`PROJECT_OVERVIEW.md`** - Architecture

---

## ✅ Checklist for First Use

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run tests: `python test_ml_pipeline.py`
- [ ] Try examples: `python example_usage.py`
- [ ] Upload your data via web: `python app.py`
- [ ] Run pipeline on your data
- [ ] Review results in output directory

---

**Quick Help:** See `ML_PIPELINE_GUIDE.md` for complete examples and detailed explanations.
