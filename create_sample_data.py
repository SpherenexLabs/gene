"""
Create a sample gene expression dataset with labels for ML testing
"""
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create sample data
n_samples = 200
n_genes = 100

# Generate gene names
gene_names = [f'GENE_{i}' for i in range(n_genes)]

# Create three classes: Breast Cancer, Lung Cancer, Healthy
classes = ['Breast_Cancer'] * 70 + ['Lung_Cancer'] * 70 + ['Healthy'] * 60

# Generate gene expression data with some pattern
data = []
for i, disease_class in enumerate(classes):
    if disease_class == 'Breast_Cancer':
        # Breast cancer has higher expression in first 30 genes
        sample = np.random.normal(10, 2, n_genes)
        sample[:30] += np.random.normal(5, 1, 30)
    elif disease_class == 'Lung_Cancer':
        # Lung cancer has higher expression in genes 30-60
        sample = np.random.normal(10, 2, n_genes)
        sample[30:60] += np.random.normal(5, 1, 30)
    else:  # Healthy
        # Healthy samples have normal expression
        sample = np.random.normal(10, 2, n_genes)
    
    data.append(sample)

# Create DataFrame
df = pd.DataFrame(data, columns=gene_names)
df['disease_type'] = classes
df['sample_id'] = [f'SAMPLE_{i:03d}' for i in range(n_samples)]

# Reorder columns to put metadata first
df = df[['sample_id', 'disease_type'] + gene_names]

# Save to CSV
output_path = 'data/raw/sample_gene_expression_with_labels.csv'
df.to_csv(output_path, index=False)

print(f"✅ Created sample dataset: {output_path}")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()[:5]}...")
print(f"\nClass distribution:")
print(df['disease_type'].value_counts())
print(f"\nFirst few rows:")
print(df.head())
