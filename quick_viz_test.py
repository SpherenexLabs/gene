"""
Quick test to generate sample visualizations from breast cancer data
"""
import pandas as pd
import numpy as np
from visualization_engine import GeneVisualizationEngine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

print("="*80)
print("QUICK VISUALIZATION TEST - Breast Cancer Data")
print("="*80)

# Load data
try:
    data = pd.read_csv('data/raw/breast_cancer_GSE2034.csv')
    print(f"✅ Data loaded: {data.shape}")
    
    # Prepare data
    if 'disease_type' in data.columns:
        label_col = 'disease_type'
    elif 'label' in data.columns:
        label_col = 'label'
    else:
        label_col = data.columns[-1]
    
    X = data.drop(columns=[label_col])
    y = data[label_col]
    
    # Encode labels if needed
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Limit to first 100 genes for speed
    if X.shape[1] > 100:
        X = X.iloc[:, :100]
    
    print(f"Using {X.shape[1]} genes, {len(y)} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Testing: {X_test.shape[0]} samples")
    
    # Train quick models
    print("\n" + "="*80)
    print("Training models...")
    print("="*80)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    roc_data = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Metrics
        from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'ROC-AUC': auc
        }
        
        roc_data[name] = (y_test, y_proba)
        
        print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, ROC-AUC: {auc:.4f}")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    comparison_df.columns = ['Model', 'Accuracy', 'Precision', 'ROC-AUC']
    
    # Initialize visualization engine
    viz = GeneVisualizationEngine(output_dir='quick_test_viz')
    
    print("\n" + "="*80)
    print("Creating Visualizations...")
    print("="*80)
    
    # 1. ROC-AUC Bar Chart
    print("1. ROC-AUC Bar Chart...")
    viz.plot_roc_auc_bar_chart(comparison_df, save_name='test_roc_auc_bars.png')
    
    # 2. Precision Bar Chart
    print("2. Precision Bar Chart...")
    viz.plot_precision_bar_chart(comparison_df, save_name='test_precision_bars.png')
    
    # 3. Multi-Model ROC Curves
    print("3. Multi-Model ROC Curves...")
    viz.plot_multi_model_roc_curves(roc_data, save_name='test_all_roc_curves.png')
    
    # 4. Confusion Matrix (best model)
    print("4. Confusion Matrix...")
    best_model_name = comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Model']
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test)
    viz.plot_confusion_matrix(y_test, y_pred_best, save_name='test_confusion_matrix.png')
    
    # 5. Gene Correlation Heatmap (subset)
    print("5. Gene Correlation Heatmap...")
    viz.plot_correlation_heatmap(
        X.iloc[:, :20],  # First 20 genes
        cluster=True,
        save_name='test_gene_correlation.png'
    )
    
    print("\n" + "="*80)
    print("✅ VISUALIZATIONS CREATED SUCCESSFULLY!")
    print("="*80)
    print(f"📁 Output directory: quick_test_viz/")
    print(f"   - test_roc_auc_bars.png")
    print(f"   - test_precision_bars.png")
    print(f"   - test_all_roc_curves.png")
    print(f"   - test_confusion_matrix.png")
    print(f"   - test_gene_correlation.png")
    
    print("\n📊 Model Performance:")
    print(comparison_df.to_string(index=False))
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
