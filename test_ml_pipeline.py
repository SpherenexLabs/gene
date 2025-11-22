"""
Test script for ML Pipeline modules
Verifies feature selection, model training, visualization, and export functionality
"""

import sys
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def test_imports():
    """Test if all ML modules can be imported"""
    print("Testing ML module imports...")
    try:
        from feature_selector import GeneFeatureSelector
        from model_trainer import DiseaseGeneClassifier
        from visualization_engine import GeneVisualizationEngine
        from results_exporter import ResultsExporter
        from complete_pipeline import CompletePipeline
        print("✅ All ML modules imported successfully\n")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}\n")
        return False

def test_visualization_packages():
    """Test if visualization packages are installed"""
    print("Testing visualization packages...")
    missing_packages = []
    
    try:
        import matplotlib
        print("  ✅ matplotlib installed")
    except ImportError:
        print("  ❌ matplotlib missing")
        missing_packages.append("matplotlib")
    
    try:
        import seaborn
        print("  ✅ seaborn installed")
    except ImportError:
        print("  ❌ seaborn missing")
        missing_packages.append("seaborn")
    
    try:
        import reportlab
        print("  ✅ reportlab installed")
    except ImportError:
        print("  ❌ reportlab missing")
        missing_packages.append("reportlab")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}\n")
        return False
    else:
        print("✅ All visualization packages installed\n")
        return True

def test_feature_selection():
    """Test feature selection module"""
    print("Testing Feature Selection...")
    try:
        from feature_selector import GeneFeatureSelector
        
        # Create synthetic data
        X, y = make_classification(n_samples=200, n_features=50, 
                                   n_informative=20, n_redundant=10, 
                                   n_classes=3, random_state=42)
        feature_names = [f"GENE_{i}" for i in range(50)]
        
        selector = GeneFeatureSelector(n_features=20)
        
        # Test ANOVA selection
        X_selected, selected_genes, scores = selector.anova_selection(
            X, y, feature_names
        )
        
        assert X_selected.shape[1] == 20, "Should select 20 features"
        assert len(selected_genes) == 20, "Should return 20 gene names"
        assert len(scores) == 20, "Should return 20 scores"
        
        print(f"  ✅ Selected {len(selected_genes)} features")
        print(f"  ✅ Top 5 genes: {selected_genes[:5]}")
        print("✅ Feature selection working correctly\n")
        return True
    except Exception as e:
        print(f"❌ Feature selection test failed: {e}\n")
        return False

def test_model_training():
    """Test model training module"""
    print("Testing Model Training...")
    try:
        from model_trainer import DiseaseGeneClassifier
        from sklearn.model_selection import train_test_split
        
        # Create synthetic data
        X, y = make_classification(n_samples=200, n_features=20, 
                                   n_informative=15, n_classes=3, 
                                   random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        trainer = DiseaseGeneClassifier()
        
        # Test training single model
        model = trainer.train_model('random_forest', X_train, y_train)
        assert model is not None, "Model should be trained"
        
        # Test evaluation - pass model name, not model object
        metrics = trainer.evaluate_model('random_forest', X_test, y_test)
        assert 'accuracy' in metrics, "Should return accuracy"
        assert 'precision' in metrics, "Should return precision"
        
        print(f"  ✅ Model trained successfully")
        print(f"  ✅ Accuracy: {metrics['accuracy']:.4f}")
        print(f"  ✅ Precision: {metrics['precision']:.4f}")
        print("✅ Model training working correctly\n")
        return True
    except Exception as e:
        print(f"❌ Model training test failed: {e}\n")
        return False

def test_complete_pipeline():
    """Test complete pipeline integration"""
    print("Testing Complete Pipeline...")
    try:
        from complete_pipeline import CompletePipeline
        
        # Create synthetic data with disease labels
        X, y = make_classification(n_samples=300, n_features=100, 
                                   n_informative=30, n_redundant=20,
                                   n_classes=3, random_state=42)
        
        # Convert to DataFrame
        gene_names = [f"GENE_{i}" for i in range(100)]
        data = pd.DataFrame(X, columns=gene_names)
        data['disease_type'] = pd.Categorical.from_codes(
            y, categories=['breast_cancer', 'lung_cancer', 'healthy']
        )
        
        # Run pipeline (without tuning for speed)
        pipeline = CompletePipeline(output_dir='test_pipeline_output')
        
        print("  ⏳ Running pipeline (this may take a moment)...")
        results = pipeline.run_complete_pipeline(
            data=data,
            label_column='disease_type',
            n_features=30,
            models=['random_forest', 'svm'],  # Just 2 models for speed
            tune_hyperparameters=False,  # Skip tuning for speed
            class_names=['Breast Cancer', 'Lung Cancer', 'Healthy']
        )
        
        # Verify results
        assert 'summary' in results, "Should return summary"
        assert 'best_model' in results['summary'], "Should have best model"
        assert results['summary']['best_accuracy'] > 0, "Should have accuracy > 0"
        
        print(f"  ✅ Pipeline completed successfully")
        print(f"  ✅ Best Model: {results['summary']['best_model']}")
        print(f"  ✅ Accuracy: {results['summary']['best_accuracy']:.4f}")
        print(f"  ✅ Selected {results['feature_selection']['n_features_selected']} features")
        print("✅ Complete pipeline working correctly\n")
        return True
    except Exception as e:
        print(f"❌ Complete pipeline test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all ML pipeline tests"""
    print("=" * 60)
    print("ML PIPELINE TEST SUITE")
    print("=" * 60)
    print()
    
    results = []
    
    # Test 1: Imports
    results.append(("Module Imports", test_imports()))
    
    # Test 2: Visualization packages
    viz_installed = test_visualization_packages()
    results.append(("Visualization Packages", viz_installed))
    
    # Test 3: Feature selection
    results.append(("Feature Selection", test_feature_selection()))
    
    # Test 4: Model training
    results.append(("Model Training", test_model_training()))
    
    # Test 5: Complete pipeline (only if viz packages installed)
    if viz_installed:
        results.append(("Complete Pipeline", test_complete_pipeline()))
    else:
        print("⚠️  Skipping complete pipeline test (install visualization packages first)\n")
    
    # Print summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! ML pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Upload your gene expression datasets")
        print("2. Run: python complete_pipeline.py")
        print("3. Or use the web interface: python app.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        if not viz_installed:
            print("\n💡 Install missing packages:")
            print("   pip install matplotlib seaborn reportlab")
    
    print("=" * 60)
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
