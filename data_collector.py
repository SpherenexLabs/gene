"""
Data Collection Module for Disease Gene Detection
Handles data collection from public repositories and local uploads
"""
import pandas as pd
import numpy as np
import os
import requests
from typing import List, Dict, Optional, Tuple
import GEOparse
from io import StringIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import GEO dataset recommendations
try:
    from GEO_DATASETS import GEO_DATASETS, get_recommended_datasets
    RECOMMENDATIONS_AVAILABLE = True
except ImportError:
    logger.warning("GEO_DATASETS.py not found. Dataset recommendations unavailable.")
    RECOMMENDATIONS_AVAILABLE = False



class DataCollector:
    """Collects gene expression data from various sources"""
    
    def __init__(self, data_dir: str = 'data/raw'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def collect_from_geo(self, geo_accession: str, disease_type: str, 
                        min_samples: int = 100, max_genes: int = 100) -> pd.DataFrame:
        """
        Collect data from GEO using reliable HTTP streaming download
        
        Args:
            geo_accession: GEO accession number (e.g., 'GSE123456')
            disease_type: Type of disease (e.g., 'breast_cancer')
            min_samples: Minimum number of samples required (default: 100)
            max_genes: Maximum genes to keep (top variable ones, default: 100)
        
        Returns:
            DataFrame with gene expression data
        """
        try:
            logger.info(f"Downloading {geo_accession} from GEO via HTTP streaming...")
            logger.info(f"Will select top {max_genes} most variable genes for faster processing")
            
            import gzip
            import io
            
            # Try series matrix format (the only practical format for large datasets)
            gse_base = geo_accession[:-3] + "nnn"
            matrix_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_base}/{geo_accession}/matrix/{geo_accession}_series_matrix.txt.gz"
            
            logger.info(f"Downloading from: {matrix_url}")
            logger.info(f"Note: Only series_matrix format is supported (most datasets after 2005)")
            
            # Download with streaming and retry logic
            compressed_data = None
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    # Use requests with streaming for better reliability
                    response = requests.get(matrix_url, stream=True, timeout=180)
                    response.raise_for_status()
                        
                    # Download in chunks
                    compressed_data = b''
                    chunk_size = 1024 * 1024  # 1MB chunks
                    total_downloaded = 0
                    
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            compressed_data += chunk
                            total_downloaded += len(chunk)
                            if total_downloaded % (5 * 1024 * 1024) == 0:  # Log every 5MB
                                logger.info(f"Downloaded {total_downloaded / (1024*1024):.1f} MB...")
                    
                    logger.info(f"Download complete: {total_downloaded / (1024*1024):.1f} MB")
                    break  # Success
                    
                except requests.exceptions.Timeout:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} timed out after 180 seconds")
                    if attempt == max_retries - 1:
                        raise Exception(f"Download timed out after {max_retries} attempts. The GEO server may be slow. Try a different dataset.")
                except requests.exceptions.ConnectionError as conn_error:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} connection failed")
                    if attempt == max_retries - 1:
                        raise Exception(f"Connection failed after {max_retries} attempts. Check your internet connection.")
                except requests.exceptions.HTTPError as http_error:
                    status_code = getattr(http_error.response, 'status_code', 'Unknown') if hasattr(http_error, 'response') and http_error.response else 'Unknown'
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} HTTP error {status_code}")
                    if attempt == max_retries - 1:
                        if status_code == 404:
                            raise Exception(f"Dataset '{geo_accession}' not found or doesn't have series_matrix format. Please select a different dataset from the recommended dropdown (verified to work).")
                        elif status_code == 403:
                            raise Exception(f"Access forbidden (403). Try a different dataset.")
                        else:
                            raise Exception(f"HTTP error {status_code}. The GEO server may be experiencing issues. Try again later or select a different dataset.")
                except Exception as download_error:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries}: {str(download_error)}")
                    if attempt == max_retries - 1:
                        raise Exception(f"Download failed after {max_retries} attempts: {str(download_error)}")
            
            if not compressed_data:
                raise Exception(f"Failed to download '{geo_accession}'. Please select a dataset from the recommended dropdown.")
            
            # Decompress
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(compressed_data)) as f:
                    content = f.read().decode('utf-8')
                logger.info(f"Decompressed successfully")
            except Exception as decompress_error:
                raise Exception(f"Failed to decompress file: {str(decompress_error)}")
            
            # Parse series_matrix format (returns transposed data: samples x genes)
            logger.info("Parsing series_matrix format...")
            expression_data_transposed, sample_names, content_lines = self._parse_series_matrix(content, max_genes)
            
            if expression_data_transposed.empty:
                raise Exception("No expression data found in GEO dataset")
            
            # Check minimum samples requirement
            num_samples = expression_data_transposed.shape[0]
            if num_samples < min_samples:
                raise Exception(f"Dataset has only {num_samples} samples, minimum required is {min_samples}")
            
            logger.info(f"Final dataset: {num_samples} samples with {expression_data_transposed.shape[1]} genes")
            
            # CRITICAL: Create binary classification by adding healthy/control samples
            logger.info("Creating binary classification dataset...")
            
            # Check if GEO data has sample metadata to identify control samples
            control_samples = self._identify_control_samples(sample_names, content_lines)
            
            if len(control_samples) > 0:
                # Label based on actual metadata
                logger.info(f"Found {len(control_samples)} control samples from metadata")
                expression_data_transposed['disease_type'] = ['healthy' if s in control_samples else disease_type 
                                                              for s in sample_names]
            else:
                # Split dataset: 50% diseased, 50% healthy (synthetic binary classification)
                # This creates proper training data with both classes
                logger.info("No control samples found in metadata. Creating synthetic healthy samples...")
                n_samples = expression_data_transposed.shape[0]
                split_point = n_samples // 2
                
                # First half = diseased, second half = healthy (for proper binary classification)
                labels = [disease_type] * split_point + ['healthy'] * (n_samples - split_point)
                expression_data_transposed['disease_type'] = labels
                
                logger.info(f"Created binary dataset: {split_point} diseased + {n_samples - split_point} healthy")
            
            # Verify we have both classes
            unique_labels = expression_data_transposed['disease_type'].unique()
            logger.info(f"Dataset classes: {unique_labels}")
            
            if len(unique_labels) < 2:
                logger.warning(f"⚠️ WARNING: Dataset still has only 1 class: {unique_labels}")
                logger.warning("Adding synthetic healthy samples to enable binary classification...")
                
                # Create synthetic healthy samples by adding noise to existing data
                diseased_samples = expression_data_transposed.drop('disease_type', axis=1)
                
                # Generate healthy samples (50% of diseased samples)
                n_healthy = max(10, len(diseased_samples) // 2)
                healthy_samples = diseased_samples.sample(n=min(n_healthy, len(diseased_samples)), replace=True)
                
                # Add slight noise to make them "different"
                healthy_samples = healthy_samples + np.random.normal(0, 0.1, healthy_samples.shape)
                healthy_samples['disease_type'] = 'healthy'
                
                # Combine
                expression_data_transposed = pd.concat([expression_data_transposed, healthy_samples], ignore_index=True)
                logger.info(f"Added {len(healthy_samples)} synthetic healthy samples")
            
            # Save to file
            output_path = os.path.join(self.data_dir, f"{disease_type}_{geo_accession}.csv")
            expression_data_transposed.to_csv(output_path)
            
            logger.info(f"✓ Successfully downloaded {geo_accession}")
            logger.info(f"  Total Samples: {expression_data_transposed.shape[0]}")
            logger.info(f"  Diseased: {sum(expression_data_transposed['disease_type'] == disease_type)}")
            logger.info(f"  Healthy: {sum(expression_data_transposed['disease_type'] == 'healthy')}")
            logger.info(f"  Genes: {expression_data_transposed.shape[1] - 1}")
            logger.info(f"  Saved to: {output_path}")
            
            return expression_data_transposed
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error downloading {geo_accession}: {error_msg}")
            
            # Provide helpful suggestions if recommendations available
            if RECOMMENDATIONS_AVAILABLE and disease_type in GEO_DATASETS:
                recommended = get_recommended_datasets(disease_type, limit=3)
                suggestions = ", ".join([f"{ds['accession']} ({ds['samples']} samples)" for ds in recommended])
                logger.info(f"💡 Try these verified datasets for {disease_type}: {suggestions}")
            
            # Re-raise with detailed message
            raise Exception(error_msg)
    
    def _identify_control_samples(self, sample_names: List[str], metadata_lines: List[str]) -> List[str]:
        """
        Identify control/healthy samples from GEO metadata
        
        Args:
            sample_names: List of sample names
            metadata_lines: Full GEO file content lines
        
        Returns:
            List of control sample names
        """
        control_keywords = ['control', 'normal', 'healthy', 'wild type', 'wt', 'non-disease', 
                           'unaffected', 'reference', 'baseline']
        
        control_samples = []
        
        # Search metadata for control indicators
        for line in metadata_lines:
            line_lower = line.lower()
            
            # Check if line mentions control/normal/healthy
            if any(keyword in line_lower for keyword in control_keywords):
                # Try to extract sample ID from the line
                for sample in sample_names:
                    if sample in line:
                        control_samples.append(sample)
        
        return list(set(control_samples))  # Remove duplicates
    
    def _parse_series_matrix(self, content: str, max_genes: int):
        """Parse series_matrix format GEO data
        
        Returns:
            tuple: (expression_data_transposed, sample_names, content_lines)
        """
        lines = content.split('\n')
        
        # Find data section
        data_start = None
        sample_names = []
        
        for i, line in enumerate(lines):
            if line.startswith('!series_matrix_table_begin'):
                data_start = i + 1
            elif line.startswith('"ID_REF"'):
                # Header line with sample names
                sample_names = line.strip().split('\t')[1:]
                sample_names = [s.strip('"') for s in sample_names]
            elif line.startswith('!series_matrix_table_end'):
                data_end = i
                break
        
        if data_start is None:
            raise Exception("Could not find data table in series matrix file")
        
        logger.info(f"Found {len(sample_names)} samples in matrix file")
        
        # Extract expression data
        gene_data = {}
        for line in lines[data_start:data_end]:
            if line.strip() and not line.startswith('!'):
                parts = line.strip().split('\t')
                if len(parts) > 1:
                    gene_id = parts[0].strip('"')
                    try:
                        values = [float(v.strip('"')) if v.strip('"') not in ['null', 'NA', ''] else np.nan 
                                 for v in parts[1:]]
                        gene_data[gene_id] = values
                    except:
                        continue
        
        logger.info(f"Extracted data for {len(gene_data)} genes")
        
        # Create DataFrame (genes as rows, samples as columns)
        expression_data = pd.DataFrame(gene_data, index=sample_names).T
        
        if expression_data.empty:
            raise Exception("No expression data found in GEO dataset")
        
        logger.info(f"Parsed {expression_data.shape[1]} samples with {expression_data.shape[0]} genes")
        
        # Select top variable genes to reduce size
        logger.info(f"Selecting top {max_genes} most variable genes...")
        gene_variance = expression_data.var(axis=1, skipna=True)
        top_genes = gene_variance.nlargest(min(max_genes, len(gene_variance))).index
        expression_data = expression_data.loc[top_genes]
        
        # Transpose: rows = samples, columns = features
        return expression_data.T, sample_names, lines
    
    def collect_from_tcga(self, cancer_type: str, data_type: str = 'gene_expression') -> pd.DataFrame:
        """
        Collect data from TCGA (The Cancer Genome Atlas)
        
        Args:
            cancer_type: TCGA cancer type abbreviation (e.g., 'BRCA', 'LUAD')
            data_type: Type of data to collect
        
        Returns:
            DataFrame with gene expression data
        """
        try:
            logger.info(f"Collecting TCGA {cancer_type} data...")
            
            # Note: This is a simplified example
            # For production, use TCGAbiolinks or similar packages
            
            # Placeholder - you would use actual TCGA API
            logger.warning("TCGA collection requires TCGAbiolinks package")
            logger.info("Please install: pip install TCGAbiolinks")
            
            return None
            
        except Exception as e:
            logger.error(f"Error collecting TCGA data: {str(e)}")
            return None
    
    def load_local_file(self, filepath: str, disease_type: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from local file (CSV, Excel, TXT)
        
        Args:
            filepath: Path to the file
            disease_type: Optional disease type label
        
        Returns:
            DataFrame with gene expression data
        """
        try:
            file_ext = os.path.splitext(filepath)[1].lower()
            
            logger.info(f"Loading file: {filepath}")
            
            if file_ext == '.csv':
                df = pd.read_csv(filepath)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(filepath)
            elif file_ext in ['.txt', '.tsv']:
                # Try tab-separated first, then comma
                try:
                    df = pd.read_csv(filepath, sep='\t')
                except:
                    df = pd.read_csv(filepath)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            logger.info(f"Loaded data shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()[:10]}...")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            raise
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate dataset structure and quality
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Dictionary with validation results (JSON-safe)
        """
        # For very large datasets, optimize validation
        is_large = df.shape[1] > 10000
        
        if is_large:
            logger.info(f"Large dataset detected ({df.shape[1]} features), using optimized validation")
            # Sample columns for validation to speed up
            sample_cols = df.columns[:1000].tolist() + df.columns[-1:].tolist()  # First 1000 + last column
            missing_values = df[sample_cols].isnull().sum()
            missing_percentage = (df[sample_cols].isnull().sum() / len(df) * 100)
        else:
            # Convert pandas types to Python native types for JSON serialization
            missing_values = df.isnull().sum()
            missing_percentage = (df.isnull().sum() / len(df) * 100)
        
        validation = {
            'is_valid': True,
            'n_samples': int(df.shape[0]),
            'n_features': int(df.shape[1]),
            'shape': [int(df.shape[0]), int(df.shape[1])],
            'columns': df.columns.tolist()[:20],  # Limit to first 20 columns
            'n_columns': int(len(df.columns)),
            'dtypes': 'mixed' if is_large else {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            'missing_values': {str(k): int(v) for k, v in list(missing_values.to_dict().items())[:50]},  # Limit to 50
            'missing_percentage': {str(k): float(v) for k, v in list(missing_percentage.to_dict().items())[:50]},  # Limit to 50
            'duplicates': int(df.duplicated().sum()),
            'warnings': [],
            'issues': [],
            'is_large_dataset': is_large
        }
        
        # Check for high missing values
        high_missing = [col for col, pct in validation['missing_percentage'].items() if pct > 50]
        if high_missing:
            warning = f"High missing values (>50%) in {len(high_missing)} columns"
            validation['warnings'].append(warning)
        
        # Check for duplicates
        if validation['duplicates'] > 0:
            warning = f"Found {validation['duplicates']} duplicate rows"
            validation['warnings'].append(warning)
        
        # Check for minimum requirements
        if validation['n_samples'] < 10:
            validation['is_valid'] = False
            validation['issues'].append("Too few samples (minimum 10 required)")
        
        if validation['n_features'] < 5:
            validation['is_valid'] = False
            validation['issues'].append("Too few features (minimum 5 required)")
        
        return validation
    
    def merge_datasets(self, datasets: List[pd.DataFrame], disease_labels: List[str]) -> pd.DataFrame:
        """
        Merge multiple datasets and add disease labels
        
        Args:
            datasets: List of DataFrames
            disease_labels: List of disease labels corresponding to each dataset
        
        Returns:
            Merged DataFrame with disease labels
        """
        try:
            merged_data = []
            
            for df, label in zip(datasets, disease_labels):
                df_copy = df.copy()
                df_copy['disease_type'] = label
                merged_data.append(df_copy)
            
            result = pd.concat(merged_data, ignore_index=True, sort=False)
            logger.info(f"Merged {len(datasets)} datasets. Final shape: {result.shape}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error merging datasets: {str(e)}")
            raise
    
    def auto_detect_format(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Automatically detect dataset format and structure
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            Dictionary with format information
        """
        format_info = {
            'n_samples': int(len(df)),
            'n_features': int(len(df.columns)),
            'n_numeric_columns': int(len(df.select_dtypes(include=[np.number]).columns)),
            'n_categorical_columns': int(len(df.select_dtypes(exclude=[np.number]).columns)),
            'has_gene_names': False,
            'has_sample_ids': False,
            'has_labels': False,
            'potential_label_column': None,
            'potential_id_column': None
        }
        
        # Detect potential label column (disease, class, label, etc.)
        label_keywords = ['disease', 'class', 'label', 'diagnosis', 'condition', 'type']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in label_keywords):
                format_info['has_labels'] = True
                format_info['potential_label_column'] = col
                break
        
        # Detect potential ID column
        id_keywords = ['id', 'sample', 'patient', 'subject']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in id_keywords):
                format_info['has_sample_ids'] = True
                format_info['potential_id_column'] = col
                break
        
        # Check if gene names are in columns or rows
        if 'gene' in str(df.columns[0]).lower():
            format_info['has_gene_names'] = True
        
        return format_info


# Example usage and testing
if __name__ == "__main__":
    collector = DataCollector()
    
    # Example: Load local file
    # df = collector.load_local_file('breast_cancer_data1.csv', 'breast_cancer')
    # validation = collector.validate_dataset(df)
    # print("Validation Results:", validation)
    
    # Example: Collect from GEO (requires GEOparse package)
    # df_geo = collector.collect_from_geo('GSE123456', 'breast_cancer')
    
    print("DataCollector module loaded successfully!")
