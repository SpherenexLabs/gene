"""
Helper script to download and combine multiple GEO datasets with proper labels
This creates a dataset suitable for ML classification
"""
from data_collector import DataCollector
import pandas as pd

def download_and_combine_geo_datasets():
    """
    Download multiple GEO datasets and combine them with different labels
    Example: Cancer samples vs Healthy samples
    """
    collector = DataCollector()
    
    print("=" * 60)
    print("GEO Dataset Download & Label Generator")
    print("=" * 60)
    print()
    
    # Example 1: Download cancer and healthy samples
    print("To create a proper ML dataset, you need:")
    print("1. GEO accession for CANCER samples")
    print("2. GEO accession for HEALTHY/CONTROL samples")
    print()
    print("Example GEO datasets:")
    print("  - GSE15852 (Breast cancer)")
    print("  - GSE10810 (Breast healthy)")
    print()
    
    # Get user input
    cancer_accession = input("Enter GEO accession for CANCER samples (or press Enter to skip): ").strip()
    healthy_accession = input("Enter GEO accession for HEALTHY samples (or press Enter to skip): ").strip()
    
    datasets = []
    labels = []
    
    if cancer_accession:
        print(f"\nDownloading {cancer_accession} (Cancer)...")
        cancer_data = collector.collect_from_geo(cancer_accession, 'cancer')
        if cancer_data is not None:
            datasets.append(cancer_data)
            labels.append('cancer')
            print(f"✓ Downloaded: {cancer_data.shape[0]} samples")
    
    if healthy_accession:
        print(f"\nDownloading {healthy_accession} (Healthy)...")
        healthy_data = collector.collect_from_geo(healthy_accession, 'healthy')
        if healthy_data is not None:
            datasets.append(healthy_data)
            labels.append('healthy')
            print(f"✓ Downloaded: {healthy_data.shape[0]} samples")
    
    if len(datasets) > 1:
        print("\nCombining datasets...")
        # Find common genes (columns)
        common_genes = set(datasets[0].columns)
        for df in datasets[1:]:
            common_genes = common_genes.intersection(set(df.columns))
        
        common_genes.discard('disease_type')  # Remove label column
        common_genes = list(common_genes)
        
        # Select only common genes
        combined_datasets = []
        for df, label in zip(datasets, labels):
            df_subset = df[common_genes + ['disease_type']].copy()
            combined_datasets.append(df_subset)
        
        # Combine
        final_dataset = pd.concat(combined_datasets, ignore_index=True)
        
        # Save
        output_path = 'data/raw/combined_geo_dataset_with_labels.csv'
        final_dataset.to_csv(output_path, index=False)
        
        print(f"\n✓ Combined dataset saved: {output_path}")
        print(f"  Shape: {final_dataset.shape}")
        print(f"  Common genes: {len(common_genes)}")
        print(f"\n  Label distribution:")
        print(final_dataset['disease_type'].value_counts())
        print("\n✓ This dataset is ready for ML analysis!")
        
    elif len(datasets) == 1:
        print("\n⚠ Only one dataset downloaded.")
        print("For ML classification, you need at least 2 different classes.")
        print("Please run again and provide both cancer and healthy accessions.")
    else:
        print("\n✗ No datasets downloaded.")

if __name__ == "__main__":
    download_and_combine_geo_datasets()
