"""
Demo script to showcase enhanced visualization features
Includes: Precision metrics, ROC-AUC bar charts, Confusion Matrix, Multi-model ROC curves
"""
import pandas as pd
import numpy as np
from complete_pipeline import CompletePipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run complete pipeline with enhanced visualizations"""
    
    print("="*80)
    print("DISEASE GENE DETECTION - ENHANCED VISUALIZATION DEMO")
    print("="*80)
    
    # Load the breast cancer data from GEO
    data_path = 'data/raw/breast_cancer_GSE2034.csv'
    
    try:
        logger.info(f"Loading data from: {data_path}")
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded: {data.shape[0]} samples × {data.shape[1]} features")
        
        # Check for label column
        if 'disease_type' in data.columns:
            label_column = 'disease_type'
        elif 'label' in data.columns:
            label_column = 'label'
        elif 'class' in data.columns:
            label_column = 'class'
        else:
            logger.warning("No standard label column found. Using last column as label.")
            label_column = data.columns[-1]
        
        logger.info(f"Using label column: {label_column}")
        logger.info(f"Classes: {data[label_column].unique()}")
        
        # Initialize pipeline
        pipeline = CompletePipeline(output_dir='demo_output')
        
        # Configure preprocessing with gene selection
        preprocessing_config = {
            'missing_threshold': 0.3,
            'missing_strategy': 'median',
            'outlier_method': 'iqr',
            'normalization': 'zscore',
            'test_size': 0.2,
            'validation_size': 0.1
        }
        
        # Step 1: Preprocessing (with gene selection to top 100 genes)
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.run_preprocessing(
            data, 
            label_column=label_column,
            config=preprocessing_config
        )
        
        # Get feature names
        feature_names = [col for col in data.columns if col != label_column][:X_train.shape[1]]
        
        # Step 2: Feature Selection (reduce to top 50 for faster training)
        X_train_selected, X_val_selected, X_test_selected, selected_names = pipeline.run_feature_selection(
            X_train, y_train, X_val, X_test,
            n_features=50,
            feature_names=feature_names,
            methods=['anova', 'mutual_information']
        )
        
        # Step 3: Train multiple models (faster models for demo)
        models_to_train = ['random_forest', 'svm', 'knn', 'logistic_regression']
        trained_models = pipeline.run_training(
            X_train_selected, y_train,
            models=models_to_train,
            tune=False  # Faster without tuning
        )
        
        # Get class names
        class_names = [str(c) for c in np.unique(y_train)]
        
        # Step 4: Evaluate and create visualizations
        logger.info("\n" + "="*80)
        logger.info("CREATING ENHANCED VISUALIZATIONS")
        logger.info("="*80)
        
        comparison_df = pipeline.run_evaluation(
            X_test_selected, y_test,
            class_names=class_names
        )
        
        # Display results
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        print(comparison_df.to_string(index=False))
        
        # Step 5: Export results
        exported_files = pipeline.export_results(
            X_test_selected, y_test,
            class_names=class_names
        )
        
        # Summary
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)
        print(f"\n📊 Visualizations created ({len(pipeline.results['visualization_paths'])} files):")
        print(f"   ✅ Model comparison (all metrics)")
        print(f"   ✅ ROC-AUC bar chart")
        print(f"   ✅ Precision bar chart")
        print(f"   ✅ Confusion matrix")
        print(f"   ✅ ROC curves (individual)")
        print(f"   ✅ ROC curves (multi-model comparison)")
        print(f"   ✅ Precision-Recall curve")
        print(f"   ✅ Feature importance")
        print(f"   ✅ PCA visualization")
        
        print(f"\n📁 Output directory: {pipeline.output_dir}/")
        print(f"   - Visualizations: {pipeline.viz_dir}/")
        print(f"   - Results: {pipeline.results_dir}/")
        print(f"   - Models: {pipeline.models_dir}/")
        
        print(f"\n📈 Exported files:")
        for file_type, file_path in exported_files.items():
            print(f"   - {file_type}: {file_path}")
        
        return pipeline, comparison_df
        
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        logger.info("Attempting to use sample data instead...")
        
        # Try sample data
        sample_path = 'data/raw/sample_gene_expression_with_labels.csv'
        if pd.io.common.file_exists(sample_path):
            logger.info(f"Loading sample data from: {sample_path}")
            data = pd.read_csv(sample_path)
            # Restart with sample data (recursive call with updated path)
            # ... (implementation continues)
        else:
            logger.error("No data available. Please ensure breast_cancer_GSE2034.csv exists.")
            return None, None
    
    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    pipeline, results = main()
    
    if pipeline and results is not None:
        print("\n✅ Demo completed successfully!")
        print(f"🎨 Check {pipeline.viz_dir}/ for all visualizations")
    else:
        print("\n❌ Demo failed. Check logs above for errors.")
