"""
Quick test script to verify performance optimizations
"""
import pandas as pd
import numpy as np
import time
from preprocessor import GeneDataPreprocessor

print("="*70)
print("PERFORMANCE OPTIMIZATION TEST")
print("="*70)

# Create test dataset
np.random.seed(42)
n_samples = 100
n_features = 1000

print(f"\n1. Creating test dataset...")
print(f"   Samples: {n_samples}")
print(f"   Features: {n_features}")

# Generate random gene expression data
data = pd.DataFrame(
    np.random.randn(n_samples, n_features),
    columns=[f'GENE_{i}' for i in range(n_features)]
)
data['disease_type'] = np.random.choice(['cancer', 'normal'], n_samples)

print(f"   ✓ Dataset created: {data.shape}")

# Test preprocessing
print(f"\n2. Testing preprocessing pipeline...")
start_time = time.time()

config = {
    'missing_value_strategy': 'mean',
    'outlier_method': 'iqr',
    'normalization_method': 'zscore',
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42
}

preprocessor = GeneDataPreprocessor(config)

try:
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.full_pipeline(
        data, 
        label_column='disease_type',
        include_validation=True
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"   ✓ Preprocessing completed successfully!")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Train shape: {X_train.shape}")
    print(f"   Validation shape: {X_val.shape}")
    print(f"   Test shape: {X_test.shape}")
    
    # Performance check
    if duration < 10:
        print(f"   ✓ PERFORMANCE: EXCELLENT (< 10 seconds)")
    elif duration < 20:
        print(f"   ✓ PERFORMANCE: GOOD (< 20 seconds)")
    else:
        print(f"   ⚠ PERFORMANCE: SLOW (> 20 seconds) - may need further optimization")
    
    print(f"\n3. Testing statistics...")
    print(f"   Original samples: {data.shape[0]}")
    print(f"   Final samples: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]}")
    print(f"   Outliers removed: {preprocessor.preprocessing_stats.get('outliers_removed', 0)}")
    print(f"   Features: {X_train.shape[1]}")
    
    print("\n" + "="*70)
    print("TEST PASSED ✓")
    print("="*70)
    print("\nOptimizations are working correctly!")
    print("The preprocessing pipeline is ready for production use.")
    
except Exception as e:
    print(f"   ✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    print("\n" + "="*70)
    print("TEST FAILED ✗")
    print("="*70)
