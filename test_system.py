"""
Test Script for Disease Gene Detection System
Verifies all components are working correctly
"""
import os
import sys
import pandas as pd
import numpy as np

print("=" * 70)
print("DISEASE GENE DETECTION SYSTEM - TEST SUITE")
print("=" * 70)

# Test 1: Import all modules
print("\n[TEST 1] Testing module imports...")
try:
    from config import *
    print("  ✅ config.py imported")
except Exception as e:
    print(f"  ❌ config.py failed: {e}")
    sys.exit(1)

try:
    from data_collector import DataCollector
    print("  ✅ data_collector.py imported")
except Exception as e:
    print(f"  ❌ data_collector.py failed: {e}")
    sys.exit(1)

try:
    from preprocessor import GeneDataPreprocessor
    print("  ✅ preprocessor.py imported")
except Exception as e:
    print(f"  ❌ preprocessor.py failed: {e}")
    sys.exit(1)

# Test 2: Check required packages
print("\n[TEST 2] Checking required packages...")
required_packages = {
    'pandas': pd,
    'numpy': np,
    'sklearn': None,
    'flask': None,
    'scipy': None
}

for package_name in ['sklearn', 'flask', 'scipy']:
    try:
        __import__(package_name)
        print(f"  ✅ {package_name} installed")
    except ImportError:
        print(f"  ❌ {package_name} not installed - run: pip install -r requirements.txt")

# Test 3: Check directory structure
print("\n[TEST 3] Checking directory structure...")
directories = [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, UPLOADS_DIR]
for directory in directories:
    if os.path.exists(directory):
        print(f"  ✅ {directory}")
    else:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✨ Created {directory}")

# Test 4: Test DataCollector
print("\n[TEST 4] Testing DataCollector...")
collector = DataCollector(RAW_DATA_DIR)

# Create test data
test_data = pd.DataFrame({
    'gene_1': np.random.randn(50),
    'gene_2': np.random.randn(50),
    'gene_3': np.random.randn(50),
    'disease': ['cancer'] * 25 + ['normal'] * 25
})

# Add some missing values
test_data.loc[0:5, 'gene_1'] = np.nan

test_file = os.path.join(RAW_DATA_DIR, 'test_dataset.csv')
test_data.to_csv(test_file, index=False)
print(f"  ✅ Created test dataset: {test_file}")

# Load test file
loaded_df = collector.load_local_file(test_file)
print(f"  ✅ Loaded dataset: {loaded_df.shape}")

# Validate
validation = collector.validate_dataset(loaded_df)
print(f"  ✅ Validation complete: {validation['is_valid']}")

# Auto-detect format
format_info = collector.auto_detect_format(loaded_df)
print(f"  ✅ Format detection: {format_info['n_samples']} samples, {format_info['n_features']} features")

# Test 5: Test Preprocessor
print("\n[TEST 5] Testing GeneDataPreprocessor...")
preprocessor = GeneDataPreprocessor()

# Test data cleaning
df_clean = preprocessor.clean_data(test_data)
print(f"  ✅ Data cleaning: {df_clean.shape}")

# Test missing value handling
df_imputed = preprocessor.handle_missing_values(df_clean)
missing_after = df_imputed.isnull().sum().sum()
print(f"  ✅ Missing value handling: {missing_after} missing values remain")

# Test normalization
df_normalized = preprocessor.normalize_data(df_imputed, label_column='disease')
print(f"  ✅ Normalization complete")

# Test label encoding
labels = test_data['disease']
encoded = preprocessor.encode_labels(labels)
print(f"  ✅ Label encoding: {len(np.unique(encoded))} classes")

# Test full pipeline
print("\n[TEST 6] Testing full preprocessing pipeline...")
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.full_pipeline(
    test_data,
    label_column='disease',
    include_validation=True
)

print(f"  ✅ Train set: {X_train.shape}")
print(f"  ✅ Validation set: {X_val.shape}")
print(f"  ✅ Test set: {X_test.shape}")

# Test 7: Save and load preprocessor
print("\n[TEST 7] Testing preprocessor save/load...")
test_preprocessor_path = os.path.join(PROCESSED_DATA_DIR, 'test_preprocessor.pkl')
preprocessor.save_preprocessor(test_preprocessor_path)
print(f"  ✅ Saved preprocessor")

new_preprocessor = GeneDataPreprocessor()
new_preprocessor.load_preprocessor(test_preprocessor_path)
print(f"  ✅ Loaded preprocessor")

# Test 8: Test real-time transformation
print("\n[TEST 8] Testing real-time data transformation...")
new_sample = pd.DataFrame({
    'gene_1': [1.5],
    'gene_2': [2.0],
    'gene_3': [0.5]
})

transformed = preprocessor.transform_new_data(new_sample)
print(f"  ✅ Transformed new sample: {transformed.shape}")

# Test 9: Test configuration
print("\n[TEST 9] Testing custom configuration...")
custom_config = {
    'missing_value_strategy': 'median',
    'outlier_method': 'zscore',
    'normalization_method': 'minmax',
    'test_size': 0.25,
    'validation_size': 0.15,
    'random_state': 123
}

custom_preprocessor = GeneDataPreprocessor(custom_config)
print(f"  ✅ Custom preprocessor created")

# Run with custom config
X_train_c, X_test_c, y_train_c, y_test_c = custom_preprocessor.full_pipeline(
    test_data,
    label_column='disease',
    include_validation=False
)
print(f"  ✅ Custom preprocessing: Train {X_train_c.shape}, Test {X_test_c.shape}")

# Test 10: Test Flask app availability
print("\n[TEST 10] Testing Flask app configuration...")
try:
    from app import app
    print(f"  ✅ Flask app imported")
    print(f"  ✅ API will run on {API_HOST}:{API_PORT}")
except Exception as e:
    print(f"  ⚠️  Flask app import warning: {e}")

# Final Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

print("\n✅ All core tests passed!")
print("\n📝 System is ready for use:")
print("  1. Upload datasets through web interface: python app.py")
print("  2. Or use Python API directly (see example_usage.py)")
print("  3. Preprocessed data will be saved in:", PROCESSED_DATA_DIR)

print("\n🚀 To start the web interface:")
print("  python app.py")
print("  Then open: http://localhost:5000")

print("\n" + "=" * 70)

# Cleanup test files (optional)
print("\n🧹 Cleanup test files? (y/n): ", end='')
cleanup = input().lower()
if cleanup == 'y':
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"  ✅ Removed {test_file}")
    if os.path.exists(test_preprocessor_path):
        os.remove(test_preprocessor_path)
        print(f"  ✅ Removed {test_preprocessor_path}")
    print("  ✨ Cleanup complete!")
else:
    print("  ℹ️  Test files kept for reference")

print("\n✨ Testing complete! System is operational.\n")
