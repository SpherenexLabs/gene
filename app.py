"""
Web-based Upload Interface for Disease Gene Detection
Real-time data upload and preprocessing
"""
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging

from config import *
from data_collector import DataCollector
from preprocessor import GeneDataPreprocessor
from results_exporter import ResultsExporter

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['UPLOAD_FOLDER'] = UPLOADS_DIR
CORS(app)

# Initialize components
data_collector = DataCollector(RAW_DATA_DIR)
preprocessor = GeneDataPreprocessor(PREPROCESSING_CONFIG)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', diseases=DISEASE_CATEGORIES)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Upload and validate dataset file
    Returns: Validation results and preview
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        disease_type = request.form.get('disease_type', 'unknown')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Allowed: {ALLOWED_EXTENSIONS}'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename = f"{disease_type}_{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)
        
        logger.info(f"File uploaded: {saved_filename}")
        
        # Load and validate
        df = data_collector.load_local_file(filepath, disease_type)
        validation = data_collector.validate_dataset(df)
        format_info = data_collector.auto_detect_format(df)
        
        # Create lightweight preview for large datasets
        is_large = df.shape[1] > 1000
        preview = {
            'head': df.head(5).fillna('').astype(str).to_dict('records')[:5],  # Only 5 rows
            'columns': df.columns.tolist()[:50],  # Only first 50 columns
            'n_columns': int(len(df.columns)),
            'shape': [int(df.shape[0]), int(df.shape[1])],
            'is_large': is_large
        }
        
        # Lightweight validation summary
        validation_summary = {
            'is_valid': validation.get('is_valid', True),
            'n_samples': validation.get('n_samples', 0),
            'n_features': validation.get('n_features', 0),
            'duplicates': validation.get('duplicates', 0),
            'warnings': validation.get('warnings', []),
            'issues': validation.get('issues', []),
            'is_large_dataset': validation.get('is_large_dataset', False)
        }
        
        # Add helpful message for large datasets
        if is_large:
            validation_summary['message'] = f"Large dataset uploaded ({df.shape[0]} samples × {df.shape[1]} features). Processing may take 2-5 minutes."
        
        response = {
            'success': True,
            'filename': saved_filename,
            'filepath': filepath,
            'validation': validation_summary,
            'format_info': format_info,
            'preview': preview,
            'disease_type': disease_type
        }
        
        logger.info(f"File validated successfully: {saved_filename}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    """
    Preprocess uploaded dataset
    Returns: Preprocessing results and statistics
    """
    try:
        data = request.get_json()
        filepath = data.get('filepath')
        label_column = data.get('label_column')
        disease_type = data.get('disease_type')
        
        # Custom preprocessing config
        custom_config = {
            'missing_value_strategy': data.get('missing_strategy', 'mean'),
            'outlier_method': data.get('outlier_method', 'iqr'),
            'normalization_method': data.get('normalization', 'zscore'),
            'test_size': data.get('test_size', 0.2),
            'validation_size': data.get('validation_size', 0.1),
            'random_state': 42
        }
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        if not label_column:
            return jsonify({'error': 'Label column not specified'}), 400
        
        logger.info(f"Preprocessing file: {filepath}")
        logger.info(f"Preprocessing configuration:")
        logger.info(f"  - Missing values: {custom_config['missing_value_strategy']}")
        logger.info(f"  - Outlier detection: {custom_config['outlier_method']}")
        logger.info(f"  - Normalization: {custom_config['normalization_method']}")
        logger.info(f"  - Test size: {custom_config['test_size']}")
        logger.info(f"  - Validation size: {custom_config['validation_size']}")
        
        # Load data
        df = data_collector.load_local_file(filepath)
        
        # Create new preprocessor with custom config
        custom_preprocessor = GeneDataPreprocessor(custom_config)
        
        # Run preprocessing pipeline with gene selection for faster processing
        max_genes = data.get('max_genes', 100)  # Default to 100 genes like sample data
        results = custom_preprocessor.full_pipeline(
            df, 
            label_column=label_column,
            include_validation=True,
            max_genes=max_genes
        )
        
        X_train, X_val, X_test, y_train, y_val, y_test = results
        
        # Save preprocessed data
        processed_dir = os.path.join(PROCESSED_DATA_DIR, disease_type)
        os.makedirs(processed_dir, exist_ok=True)
        
        # Use timestamp with milliseconds for uniqueness
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:20]  # Include microseconds
        
        # Save as numpy arrays
        np.save(os.path.join(processed_dir, f'X_train_{timestamp}.npy'), X_train)
        np.save(os.path.join(processed_dir, f'X_val_{timestamp}.npy'), X_val)
        np.save(os.path.join(processed_dir, f'X_test_{timestamp}.npy'), X_test)
        np.save(os.path.join(processed_dir, f'y_train_{timestamp}.npy'), y_train)
        np.save(os.path.join(processed_dir, f'y_val_{timestamp}.npy'), y_val)
        np.save(os.path.join(processed_dir, f'y_test_{timestamp}.npy'), y_test)
        
        # Save preprocessor
        preprocessor_path = os.path.join(processed_dir, f'preprocessor_{timestamp}.pkl')
        custom_preprocessor.save_preprocessor(preprocessor_path)
        
        # Generate report
        report = custom_preprocessor.get_preprocessing_report()
        
        # Generate quick visualizations
        viz_files = []
        comparison_df = pd.DataFrame()  # Initialize empty dataframe
        try:
            from visualization_engine import GeneVisualizationEngine
            from model_trainer import DiseaseGeneClassifier
            
            # Create visualization directory with unique timestamp
            viz_dir = os.path.join(BASE_DIR, 'static', 'visualizations', timestamp)
            os.makedirs(viz_dir, exist_ok=True)
            
            # Clear any old files in this directory
            import glob
            for old_file in glob.glob(os.path.join(viz_dir, '*.png')):
                try:
                    os.remove(old_file)
                except:
                    pass
            
            viz_engine = GeneVisualizationEngine(output_dir=viz_dir)
            
            # 1. Quick model training for visualization
            logger.info("Training quick models for visualization...")
            logger.info(f"Preprocessing config: {custom_config}")
            logger.info(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
            logger.info(f"Test data shape: X_test={X_test.shape}, y_test={y_test.shape}")
            logger.info(f"Unique classes in y_train: {np.unique(y_train)}")
            logger.info(f"Unique classes in y_test: {np.unique(y_test)}")
            
            # Check for single-class problem
            n_classes_train = len(np.unique(y_train))
            n_classes_test = len(np.unique(y_test))
            
            if n_classes_train < 2 or n_classes_test < 2:
                warning_msg = (
                    f"⚠️ WARNING: Dataset has only {n_classes_train} class(es). "
                    f"ROC-AUC and multi-class metrics require at least 2 classes (e.g., diseased vs healthy). "
                    f"Please use a dataset with both diseased and control/healthy samples. "
                    f"Try the pre-loaded 'sample_gene_expression_with_labels.csv' which has proper class labels."
                )
                logger.warning(warning_msg)
            
            # Use timestamp-based seed for unique model initialization
            import time
            random_seed = int(time.time() * 1000) % 10000
            
            trainer = DiseaseGeneClassifier()
            
            # Train 3 fast models with current preprocessed data
            models_to_train = ['random_forest', 'knn', 'logistic_regression']
            logger.info(f"Training {len(models_to_train)} models with random_seed={random_seed}")
            logger.info(f"Data preprocessed with: Missing={custom_config['missing_value_strategy']}, Outlier={custom_config['outlier_method']}, Norm={custom_config['normalization_method']}")
            logger.info(f"Data statistics after preprocessing: shape={X_train.shape}, mean={float(X_train.mean().mean()):.4f}, std={float(X_train.std().mean()):.4f}")
            
            for model_name in models_to_train:
                try:
                    logger.info(f"Training {model_name} on {X_train.shape[0]} samples...")
                    trainer.train_model(model_name, X_train, y_train)
                    logger.info(f"✓ {model_name} training complete")
                except Exception as e:
                    logger.warning(f"Could not train {model_name}: {e}")
            
            # 2. Evaluate models
            if len(trainer.models) == 0 and len(trainer.best_models) == 0:
                raise Exception("No models were successfully trained. Check data quality and class balance.")
            
            comparison_df = trainer.compare_models(X_test, y_test)
            
            # Replace NaN with 0 and log the actual values
            logger.info(f"Model comparison results:\n{comparison_df.to_string()}")
            
            if comparison_df.empty:
                raise Exception("Model comparison returned empty results")
            
            comparison_df = comparison_df.fillna(0)
            
            # 3. Generate visualizations
            logger.info("Generating visualizations with fresh data...")
            logger.info(f"Models available: {list(trainer.models.keys())}")
            
            # ROC-AUC bar chart
            viz_engine.plot_roc_auc_bar_chart(
                comparison_df,
                save_name='roc_auc_bars.png'
            )
            viz_files.append({'name': 'ROC-AUC Comparison', 'file': 'roc_auc_bars.png'})
            
            # Precision bar chart
            viz_engine.plot_precision_bar_chart(
                comparison_df,
                save_name='precision_bars.png'
            )
            viz_files.append({'name': 'Precision Comparison', 'file': 'precision_bars.png'})
            
            # Confusion matrix
            best_model_name = comparison_df.iloc[0]['Model']
            best_model = trainer.best_models.get(best_model_name) or trainer.models.get(best_model_name)
            y_pred = best_model.predict(X_test)
            
            viz_engine.plot_confusion_matrix(
                y_test, y_pred,
                title=f'Confusion Matrix - {best_model_name}',
                save_name='confusion_matrix.png'
            )
            viz_files.append({'name': 'Confusion Matrix', 'file': 'confusion_matrix.png'})
            
            # Multi-model ROC curves
            models_roc_data = {}
            for model_name in comparison_df['Model']:
                model = trainer.best_models.get(model_name) or trainer.models.get(model_name)
                if model and hasattr(model, 'predict_proba'):
                    model_proba = model.predict_proba(X_test)
                    models_roc_data[model_name] = (y_test, model_proba)
            
            if len(models_roc_data) > 1:
                viz_engine.plot_multi_model_roc_curves(
                    models_roc_data,
                    save_name='all_models_roc.png'
                )
                viz_files.append({'name': 'ROC Curves - All Models', 'file': 'all_models_roc.png'})
            
            # Model comparison table
            viz_engine.plot_model_comparison(
                comparison_df,
                save_name='model_comparison.png'
            )
            viz_files.append({'name': 'Model Performance Comparison', 'file': 'model_comparison.png'})
            
            logger.info(f"Generated {len(viz_files)} visualizations")
            
        except Exception as viz_error:
            logger.error(f"⚠️ Visualization generation failed: {viz_error}")
            import traceback
            traceback.print_exc()
            # Still continue - preprocessing was successful even if viz failed
            viz_files = []
        
        # Log visualization summary
        logger.info(f"Visualization files created: {[v['file'] for v in viz_files]}")
        
        # Generate downloadable reports (CSV, Excel, PDF)
        export_files = {}
        try:
            logger.info("Generating export files...")
            results_dir = os.path.join(BASE_DIR, 'static', 'exports', timestamp)
            os.makedirs(results_dir, exist_ok=True)
            
            exporter = ResultsExporter(output_dir=results_dir)
            
            # Get feature names
            feature_names = [col for col in df.columns if col != label_column][:X_train.shape[1]]
            
            # Create gene impact table
            if viz_files and len(feature_names) > 0:
                # Get feature importance from best model
                best_model_name = comparison_df.iloc[0]['Model']
                best_model = trainer.best_models.get(best_model_name) or trainer.models.get(best_model_name)
                
                # Get importance scores
                if hasattr(best_model, 'feature_importances_'):
                    importance_scores = best_model.feature_importances_
                elif hasattr(best_model, 'coef_'):
                    importance_scores = np.abs(best_model.coef_).flatten()
                else:
                    importance_scores = np.ones(len(feature_names))
                
                # Pad if needed
                if len(importance_scores) < len(feature_names):
                    importance_scores = np.pad(importance_scores, 
                                              (0, len(feature_names) - len(importance_scores)))
                
                # Get predictions
                y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
                y_pred = best_model.predict(X_test)
                
                # Create gene impact table
                gene_impact_table = exporter.create_gene_impact_table(
                    feature_names,
                    importance_scores,
                    y_pred,
                    y_pred_proba if y_pred_proba is not None else np.zeros((len(y_pred), 1)),
                    class_names=None
                )
                
                # Export to CSV
                csv_file = exporter.export_to_csv(
                    gene_impact_table,
                    f'gene_impact_table_{timestamp}.csv'
                )
                export_files['gene_impact_csv'] = f'/static/exports/{timestamp}/gene_impact_table_{timestamp}.csv'
                
                # Export model comparison to CSV
                model_csv = exporter.export_to_csv(
                    comparison_df.fillna(0),
                    f'model_comparison_{timestamp}.csv'
                )
                export_files['model_comparison_csv'] = f'/static/exports/{timestamp}/model_comparison_{timestamp}.csv'
                
                # Export to Excel (multiple sheets)
                excel_data = {
                    'Gene Impact': gene_impact_table,
                    'Model Comparison': comparison_df.fillna(0),
                    'Summary': pd.DataFrame({
                        'Metric': ['Total Samples', 'Training Samples', 'Test Samples', 'Features', 'Best Model', 'Best Accuracy'],
                        'Value': [
                            X_train.shape[0] + X_val.shape[0] + X_test.shape[0],
                            X_train.shape[0],
                            X_test.shape[0],
                            X_train.shape[1],
                            best_model_name,
                            f"{comparison_df.iloc[0]['Accuracy']:.4f}"
                        ]
                    })
                }
                excel_file = exporter.export_to_excel(
                    excel_data,
                    f'complete_results_{timestamp}.xlsx'
                )
                export_files['excel'] = f'/static/exports/{timestamp}/complete_results_{timestamp}.xlsx'
                
                logger.info(f"Generated {len(export_files)} export files")
            
        except Exception as export_error:
            logger.warning(f"Could not generate export files: {export_error}")
            import traceback
            traceback.print_exc()
        
        # Convert all values to JSON-safe types
        response = {
            'success': True,
            'disease_type': disease_type,
            'timestamp': timestamp,
            'config_applied': {
                'missing_values': custom_config['missing_value_strategy'],
                'outlier_method': custom_config['outlier_method'],
                'normalization': custom_config['normalization_method'],
                'test_size': custom_config['test_size'],
                'validation_size': custom_config['validation_size']
            },
            'shapes': {
                'train': [int(X_train.shape[0]), int(X_train.shape[1])],
                'validation': [int(X_val.shape[0]), int(X_val.shape[1])],
                'test': [int(X_test.shape[0]), int(X_test.shape[1])]
            },
            'label_distribution': {
                'train': [int(x) for x in np.bincount(y_train).tolist()],
                'validation': [int(x) for x in np.bincount(y_val).tolist()],
                'test': [int(x) for x in np.bincount(y_test).tolist()]
            },
            'preprocessing_stats': {
                str(k): (float(v) if isinstance(v, (np.integer, np.floating)) else 
                        int(v) if isinstance(v, (int, np.int_)) else str(v))
                for k, v in custom_preprocessor.preprocessing_stats.items()
            },
            'report': str(report),
            'saved_files': {
                'X_train': f'X_train_{timestamp}.npy',
                'X_val': f'X_val_{timestamp}.npy',
                'X_test': f'X_test_{timestamp}.npy',
                'y_train': f'y_train_{timestamp}.npy',
                'y_val': f'y_val_{timestamp}.npy',
                'y_test': f'y_test_{timestamp}.npy',
                'preprocessor': f'preprocessor_{timestamp}.pkl'
            },
            'visualizations': [{
                'name': v['name'],
                'url': f'/static/visualizations/{timestamp}/{v["file"]}'
            } for v in viz_files],
            'model_performance': comparison_df.fillna(0).to_dict('records') if not comparison_df.empty else [],
            'exports': export_files,
            'warning': (
                f'⚠️ Single-Class Dataset Detected!\n\n'
                f'Classes found: {np.unique(y_test)}\n\n'
                f'For proper ROC-AUC calculation, you need BOTH:\n'
                f'  • Diseased samples ({disease_type})\n'
                f'  • Healthy/Control samples\n\n'
                f'SOLUTIONS:\n'
                f'1. Use "sample_gene_expression_with_labels.csv" from Datasets tab\n'
                f'2. Download from GEO (system will auto-create binary classification)\n'
                f'3. Upload your own dataset with both disease and healthy labels'
            ) if len(np.unique(y_test)) < 2 else None
        }
        
        logger.info(f"Preprocessing completed: {disease_type}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/geo_datasets', methods=['GET'])
def get_geo_datasets():
    """
    Get available GEO datasets for each disease type
    Returns: Dictionary of disease types with their available datasets
    """
    try:
        from GEO_DATASETS import GEO_DATASETS
        
        # Format for frontend
        formatted_datasets = {}
        for disease_type, datasets in GEO_DATASETS.items():
            formatted_datasets[disease_type] = [
                {
                    'accession': ds['accession'],
                    'description': ds['description'],
                    'samples': ds['samples']
                }
                for ds in datasets
            ]
        
        return jsonify(formatted_datasets), 200
    except Exception as e:
        logger.error(f"Error loading GEO datasets: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/collect_geo', methods=['POST'])
def collect_from_geo():
    """
    Collect data from GEO repository
    Returns: Downloaded dataset information
    """
    try:
        data = request.get_json()
        geo_accession = data.get('geo_accession')
        disease_type = data.get('disease_type')
        min_samples = data.get('min_samples', 100)
        max_genes = data.get('max_genes', 100)
        
        if not geo_accession or not disease_type:
            return jsonify({'error': 'GEO accession and disease type required'}), 400
        
        logger.info(f"Collecting GEO data: {geo_accession} (min {min_samples} samples, top {max_genes} genes)")
        
        # Collect from GEO with parameters
        try:
            df = data_collector.collect_from_geo(geo_accession, disease_type, 
                                                min_samples=min_samples, 
                                                max_genes=max_genes)
        except Exception as geo_error:
            return jsonify({
                'error': str(geo_error)
            }), 500
        
        if df is None:
            return jsonify({
                'error': f'Failed to download GEO dataset {geo_accession}. Unknown error occurred.'
            }), 500
        
        # Validate
        validation = data_collector.validate_dataset(df)
        
        # Convert to JSON-safe types
        response = {
            'success': True,
            'geo_accession': geo_accession,
            'disease_type': disease_type,
            'shape': [int(df.shape[0]), int(df.shape[1])],
            'validation': validation  # validation is already JSON-safe from validate_dataset
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"GEO collection error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """
    List all uploaded and processed datasets
    Returns: List of available datasets
    """
    try:
        datasets = {
            'uploaded': [],
            'processed': []
        }
        
        # List uploaded files
        if os.path.exists(UPLOADS_DIR):
            for filename in os.listdir(UPLOADS_DIR):
                filepath = os.path.join(UPLOADS_DIR, filename)
                stats = os.stat(filepath)
                datasets['uploaded'].append({
                    'filename': filename,
                    'size': int(stats.st_size),
                    'modified': datetime.fromtimestamp(stats.st_mtime).isoformat()
                })
        
        # List processed datasets
        if os.path.exists(PROCESSED_DATA_DIR):
            for disease in DISEASE_CATEGORIES:
                disease_dir = os.path.join(PROCESSED_DATA_DIR, disease)
                if os.path.exists(disease_dir):
                    for filename in os.listdir(disease_dir):
                        if filename.endswith('.npy') or filename.endswith('.pkl'):
                            filepath = os.path.join(disease_dir, filename)
                            stats = os.stat(filepath)
                            datasets['processed'].append({
                                'disease': disease,
                                'filename': filename,
                                'size': int(stats.st_size),
                                'modified': datetime.fromtimestamp(stats.st_mtime).isoformat()
                            })
        
        return jsonify(datasets), 200
        
    except Exception as e:
        logger.error(f"Dataset listing error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """
    Get overall system statistics
    Returns: Statistics about uploaded and processed data
    """
    try:
        stats = {
            'total_uploaded': 0,
            'total_processed': 0,
            'diseases': {},
            'total_size': 0
        }
        
        # Count uploaded files
        if os.path.exists(UPLOADS_DIR):
            stats['total_uploaded'] = len([f for f in os.listdir(UPLOADS_DIR) 
                                          if os.path.isfile(os.path.join(UPLOADS_DIR, f))])
        
        # Count processed files by disease
        if os.path.exists(PROCESSED_DATA_DIR):
            for disease in DISEASE_CATEGORIES:
                disease_dir = os.path.join(PROCESSED_DATA_DIR, disease)
                if os.path.exists(disease_dir):
                    count = len([f for f in os.listdir(disease_dir)])
                    if count > 0:
                        stats['diseases'][disease] = count
                        stats['total_processed'] += count
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Statistics error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200


@app.route('/static/exports/<timestamp>/<filename>')
def serve_export(timestamp, filename):
    """Serve export files (CSV, Excel, PDF)"""
    try:
        export_dir = os.path.join(BASE_DIR, 'static', 'exports', timestamp)
        file_path = os.path.join(export_dir, filename)
        
        if os.path.exists(file_path):
            # Determine MIME type
            if filename.endswith('.csv'):
                mimetype = 'text/csv'
            elif filename.endswith('.xlsx'):
                mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            elif filename.endswith('.pdf'):
                mimetype = 'application/pdf'
            else:
                mimetype = 'application/octet-stream'
            
            return send_file(file_path, mimetype=mimetype, as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': 'Export file not found'}), 404
    except Exception as e:
        logger.error(f"Error serving export: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/static/visualizations/<timestamp>/<filename>')
def serve_visualization(timestamp, filename):
    """Serve visualization images"""
    try:
        viz_dir = os.path.join(BASE_DIR, 'static', 'visualizations', timestamp)
        file_path = os.path.join(viz_dir, filename)
        
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='image/png')
        else:
            return jsonify({'error': 'Visualization not found'}), 404
    except Exception as e:
        logger.error(f"Error serving visualization: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect_label_column', methods=['POST'])
def detect_label_column():
    """
    Auto-detect the label column in uploaded file
    Returns: Suggested label column name
    """
    try:
        data = request.get_json()
        filepath = data.get('filepath')
        
        if not filepath:
            return jsonify({'error': 'No filepath provided'}), 400
        
        # Construct full path
        full_path = os.path.join(BASE_DIR, filepath)
        
        if not os.path.exists(full_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Load file (just first few rows for detection)
        df = pd.read_csv(full_path, nrows=10, index_col=0)
        
        # Detect potential label column
        label_keywords = ['disease', 'class', 'label', 'diagnosis', 'condition', 'type', 'target', 'outcome']
        excluded_keywords = ['gsm', 'sample', 'id', 'patient', 'subject']  # Exclude sample IDs
        detected_column = None
        
        for col in df.columns:
            col_lower = col.lower()
            # Skip if it looks like a sample ID
            if any(excl in col_lower for excl in excluded_keywords):
                continue
            # Check for label keywords
            if any(keyword in col_lower for keyword in label_keywords):
                detected_column = col
                break
        
        # If no keyword match, look for columns with few unique values (likely categorical)
        if not detected_column:
            for col in df.columns:
                col_lower = col.lower()
                # Skip sample IDs
                if any(excl in col_lower for excl in excluded_keywords):
                    continue
                try:
                    if df[col].dtype == 'object' or (pd.to_numeric(df[col], errors='coerce').isna().sum() > len(df) * 0.5):
                        if df[col].nunique() < 20 and df[col].nunique() > 1:
                            detected_column = col
                            break
                except:
                    pass
        
        # Check if this looks like gene expression data (all numeric, many columns)
        is_gene_expression = len(df.columns) > 50 and df.select_dtypes(include=[np.number]).shape[1] > len(df.columns) * 0.9
        
        response = {
            'success': True,
            'label_column': detected_column,
            'all_columns': df.columns.tolist()[:20],  # Return first 20 columns
            'n_columns': int(len(df.columns)),
            'is_gene_expression_only': is_gene_expression,
            'message': 'No label column detected. This appears to be raw gene expression data without disease labels.' if is_gene_expression and not detected_column else None
        }
        
        logger.info(f"Detected label column: {detected_column}, Gene expression only: {is_gene_expression}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Label detection error: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/run_ml_pipeline', methods=['POST'])
def run_ml_pipeline():
    """
    Run complete ML pipeline: feature selection, training, evaluation, visualization, export
    Returns: Complete analysis results with visualizations and exports
    """
    try:
        from complete_pipeline import CompletePipeline
        
        data = request.get_json()
        filepath = data.get('filepath')
        label_column = data.get('label_column', 'disease_type')
        n_features = int(data.get('n_features', 100))
        models = data.get('models', ['random_forest', 'svm', 'gradient_boosting'])
        tune_hyperparameters = data.get('tune_hyperparameters', False)
        class_names = data.get('class_names', [])
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        logger.info(f"Running ML pipeline on: {filepath}")
        
        # Load data with optimizations - use float32 to save memory
        logger.info("Loading dataset...")
        df = pd.read_csv(filepath, index_col=0, low_memory=False)
        logger.info(f"Dataset loaded: {df.shape}")
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(BASE_DIR, 'ml_results', timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize and run pipeline
        pipeline = CompletePipeline(output_dir=output_dir)
        
        logger.info("Starting complete ML pipeline...")
        logger.info(f"Configuration: {n_features} features, Models: {models}")
        
        # Run pipeline with progress logging
        results = pipeline.run_complete_pipeline(
            data=df,
            label_column=label_column,
            n_features=n_features,
            models=models,
            tune_hyperparameters=tune_hyperparameters,
            class_names=class_names if class_names else None
        )
        
        # Clear memory
        del df
        import gc
        gc.collect()
        
        # Prepare response with file paths
        response = {
            'success': True,
            'timestamp': timestamp,
            'output_directory': output_dir,
            'summary': {
                'best_model': results['summary']['best_model'],
                'best_accuracy': float(results['summary']['best_accuracy']),
                'n_features_selected': int(results['feature_selection']['n_features_selected']),
                'models_trained': len(results['models'])
            },
            'feature_selection': {
                'n_features_selected': int(results['feature_selection']['n_features_selected']),
                'top_10_genes': results['feature_selection']['selected_names'][:10],
                'method': results['feature_selection']['method']
            },
            'model_performance': {},
            'visualizations': [],
            'exports': []
        }
        
        # Add model performance
        for model_name, metrics in results['evaluation'].items():
            response['model_performance'][model_name] = {
                'accuracy': float(metrics.get('accuracy', 0)),
                'precision': float(metrics.get('precision', 0)),
                'recall': float(metrics.get('recall', 0)),
                'f1': float(metrics.get('f1', 0)),
                'roc_auc': float(metrics.get('roc_auc', 0))
            }
        
        # List generated visualizations
        viz_dir = os.path.join(output_dir, 'visualizations')
        if os.path.exists(viz_dir):
            for filename in os.listdir(viz_dir):
                if filename.endswith('.png'):
                    response['visualizations'].append({
                        'name': filename,
                        'path': os.path.join(viz_dir, filename)
                    })
        
        # List exported files
        results_dir = os.path.join(output_dir, 'results')
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                if filename.endswith(('.csv', '.xlsx', '.pdf')):
                    response['exports'].append({
                        'name': filename,
                        'type': filename.split('.')[-1],
                        'path': os.path.join(results_dir, filename)
                    })
        
        logger.info(f"ML Pipeline completed successfully. Results saved to: {output_dir}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"ML Pipeline error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Disease Gene Detection System - Upload Interface")
    logger.info("=" * 60)
    logger.info(f"Upload directory: {UPLOADS_DIR}")
    logger.info(f"Processed data directory: {PROCESSED_DATA_DIR}")
    logger.info(f"Supported diseases: {', '.join(DISEASE_CATEGORIES)}")
    logger.info(f"Allowed file formats: {', '.join(ALLOWED_EXTENSIONS)}")
    logger.info("=" * 60)
    
    # Run Flask app
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG_MODE)
