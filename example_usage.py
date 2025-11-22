"""
Example Usage Script for Disease Gene Detection System
Demonstrates data collection, preprocessing, and real-time workflow
"""
import pandas as pd
import numpy as np
import os
from data_collector import DataCollector
from preprocessor import GeneDataPreprocessor
from config import *

print("=" * 70)
print("DISEASE GENE DETECTION SYSTEM - EXAMPLE USAGE")
print("=" * 70)

# ============================================================================
# EXAMPLE 1: Load and Validate Local Dataset
# ============================================================================
print("\n📁 EXAMPLE 1: Loading Local Dataset")
print("-" * 70)

collector = DataCollector(RAW_DATA_DIR)

# Simulate loading your datasets
# Replace these with your actual file paths
dataset_files = {
    'breast_cancer': [
        # 'path/to/breast_cancer_data1.csv',
        # 'path/to/breast_cancer_data2.csv'
    ],
    'lung_cancer': [
        # 'path/to/lung_cancer_data1.csv',
        # 'path/to/lung_cancer_data2.csv'
    ],
    'alzheimers': [
        # 'path/to/alzheimers_data.csv'
    ],
    'parkinsons': [
        # 'path/to/parkinsons_data.csv'
    ],
    'prostate_cancer': [
        # 'path/to/prostate_cancer_data.csv'
    ]
}

# Example: Create sample data for demonstration
print("\nCreating sample datasets for demonstration...")

sample_datasets = {}
for disease in DISEASE_CATEGORIES:
    # Create synthetic gene expression data
    n_samples = np.random.randint(50, 150)
    n_genes = 100
    
    # Generate random gene expression values
    gene_data = np.random.randn(n_samples, n_genes)
    
    # Create column names (gene names)
    gene_columns = [f'GENE_{i+1}' for i in range(n_genes)]
    
    # Create DataFrame
    df = pd.DataFrame(gene_data, columns=gene_columns)
    
    # Add disease label
    df['disease_type'] = disease
    
    # Add some missing values (5%)
    mask = np.random.random(df.shape) < 0.05
    df = df.mask(mask)
    
    # Add patient ID
    df.insert(0, 'patient_id', [f'P{i:04d}' for i in range(n_samples)])
    
    sample_datasets[disease] = df
    
    # Save to CSV
    output_path = os.path.join(RAW_DATA_DIR, f'{disease}_sample.csv')
    df.to_csv(output_path, index=False)
    
    print(f"✅ Created {disease} dataset: {df.shape}")

# ============================================================================
# EXAMPLE 2: Validate Dataset
# ============================================================================
print("\n\n🔍 EXAMPLE 2: Validating Dataset")
print("-" * 70)

# Load one dataset
sample_file = os.path.join(RAW_DATA_DIR, 'breast_cancer_sample.csv')
df = collector.load_local_file(sample_file, 'breast_cancer')

# Validate
validation = collector.validate_dataset(df)
print(f"\nValidation Results:")
print(f"  Shape: {validation['shape']}")
print(f"  Duplicates: {validation['duplicates']}")
print(f"  Missing values: {sum(validation['missing_values'].values())} total")
if validation['warnings']:
    print(f"  Warnings: {len(validation['warnings'])}")
    for warning in validation['warnings']:
        print(f"    - {warning}")

# Auto-detect format
format_info = collector.auto_detect_format(df)
print(f"\nFormat Detection:")
print(f"  Samples: {format_info['n_samples']}")
print(f"  Features: {format_info['n_features']}")
print(f"  Numeric columns: {len(format_info['numeric_columns'])}")
print(f"  Has labels: {format_info['has_labels']}")
if format_info['potential_label_column']:
    print(f"  Label column: {format_info['potential_label_column']}")

# ============================================================================
# EXAMPLE 3: Merge Multiple Datasets
# ============================================================================
print("\n\n🔗 EXAMPLE 3: Merging Multiple Disease Datasets")
print("-" * 70)

# Merge all sample datasets
all_datasets = []
all_labels = []

for disease, df in sample_datasets.items():
    all_datasets.append(df.drop(columns=['patient_id']))
    all_labels.append(disease)

merged_data = collector.merge_datasets(all_datasets, all_labels)
print(f"\nMerged dataset shape: {merged_data.shape}")
print(f"Disease distribution:\n{merged_data['disease_type'].value_counts()}")

# ============================================================================
# EXAMPLE 4: Preprocessing Pipeline
# ============================================================================
print("\n\n⚙️ EXAMPLE 4: Complete Preprocessing Pipeline")
print("-" * 70)

# Initialize preprocessor
preprocessor = GeneDataPreprocessor(PREPROCESSING_CONFIG)

# Run full pipeline
print("\nRunning preprocessing pipeline...")
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.full_pipeline(
    merged_data,
    label_column='disease_type',
    include_validation=True
)

# Display results
print("\n📊 Preprocessing Results:")
print(f"  Training set: {X_train.shape}")
print(f"  Validation set: {X_val.shape}")
print(f"  Test set: {X_test.shape}")

print(f"\n  Label distribution (training):")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    if preprocessor.label_encoder:
        disease_name = preprocessor.label_encoder.classes_[label]
        print(f"    {disease_name}: {count} samples")

# Print preprocessing report
print(preprocessor.get_preprocessing_report())

# ============================================================================
# EXAMPLE 5: Save Preprocessed Data
# ============================================================================
print("\n💾 EXAMPLE 5: Saving Preprocessed Data")
print("-" * 70)

# Save processed data
for disease in DISEASE_CATEGORIES:
    disease_dir = os.path.join(PROCESSED_DATA_DIR, disease)
    os.makedirs(disease_dir, exist_ok=True)

# Save combined dataset
np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'), X_val)
np.save(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'), y_val)
np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test)

# Save preprocessor
preprocessor_path = os.path.join(PROCESSED_DATA_DIR, 'preprocessor.pkl')
preprocessor.save_preprocessor(preprocessor_path)

print(f"✅ Saved training data: {X_train.shape}")
print(f"✅ Saved validation data: {X_val.shape}")
print(f"✅ Saved test data: {X_test.shape}")
print(f"✅ Saved preprocessor: {preprocessor_path}")

# ============================================================================
# EXAMPLE 6: Real-time Prediction Preprocessing
# ============================================================================
print("\n\n🚀 EXAMPLE 6: Real-time Data Transformation")
print("-" * 70)

# Simulate new incoming data for prediction
print("\nSimulating new patient data...")
new_patient_data = pd.DataFrame({
    f'GENE_{i+1}': [np.random.randn()] for i in range(100)
})

print(f"New data shape: {new_patient_data.shape}")

# Transform using saved preprocessor
transformed_data = preprocessor.transform_new_data(new_patient_data)

print(f"Transformed shape: {transformed_data.shape}")
print(f"✅ Ready for real-time prediction!")

# ============================================================================
# EXAMPLE 7: Different Preprocessing Configurations
# ============================================================================
print("\n\n🔧 EXAMPLE 7: Custom Preprocessing Configuration")
print("-" * 70)

custom_config = {
    'missing_value_strategy': 'median',
    'outlier_method': 'zscore',
    'normalization_method': 'minmax',
    'test_size': 0.15,
    'validation_size': 0.15,
    'random_state': 42
}

custom_preprocessor = GeneDataPreprocessor(custom_config)

print("Using custom configuration:")
for key, value in custom_config.items():
    print(f"  {key}: {value}")

# Run with custom config (using smaller sample for speed)
sample_data = merged_data.sample(n=min(200, len(merged_data)))
X_train_custom, X_test_custom, y_train_custom, y_test_custom = custom_preprocessor.full_pipeline(
    sample_data,
    label_column='disease_type',
    include_validation=False
)

print(f"\nCustom preprocessing complete!")
print(f"  Training: {X_train_custom.shape}")
print(f"  Testing: {X_test_custom.shape}")

# ============================================================================
# Summary
# ============================================================================
print("\n\n" + "=" * 70)
print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 70)

print("\n📝 Next Steps:")
print("  1. Replace sample data with your actual datasets")
print("  2. Adjust preprocessing parameters as needed")
print("  3. Use preprocessed data for machine learning")
print("  4. Deploy the web interface: python app.py")
print("  5. Upload datasets through the web UI")

print("\n🌐 To start the web interface:")
print("  python app.py")
print("  Then open: http://localhost:5000")

print("\n" + "=" * 70)
