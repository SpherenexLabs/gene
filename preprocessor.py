"""
Preprocessing Module for Disease Gene Detection
Handles data cleaning, normalization, splitting, and encoding
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from scipy import stats
import logging
from typing import Tuple, Dict, Optional, List
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneDataPreprocessor:
    """Comprehensive preprocessing pipeline for gene expression data"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Dictionary with preprocessing parameters
        """
        self.config = config or {
            'missing_value_strategy': 'mean',
            'outlier_method': 'iqr',
            'normalization_method': 'zscore',
            'test_size': 0.2,
            'validation_size': 0.1,
            'random_state': 42
        }
        
        self.scaler = None
        self.label_encoder = None
        self.imputer = None
        self.feature_names = None
        self.preprocessing_stats = {}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean dataset: handle missing values, duplicates, and invalid entries
        
        Args:
            df: Input DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        df_clean = df.copy()
        
        # Store original shape
        original_shape = df_clean.shape
        logger.info(f"Original shape: {original_shape}")
        
        # 1. Remove duplicate rows
        duplicates = df_clean.duplicated().sum()
        if duplicates > 0:
            logger.info(f"Removing {duplicates} duplicate rows")
            df_clean = df_clean.drop_duplicates()
        
        # 2. Remove columns with all NaN
        all_nan_cols = df_clean.columns[df_clean.isnull().all()].tolist()
        if all_nan_cols:
            logger.info(f"Removing {len(all_nan_cols)} columns with all NaN values")
            df_clean = df_clean.drop(columns=all_nan_cols)
        
        # 3. Remove rows with all NaN (except label column)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean = df_clean.dropna(how='all', subset=numeric_cols)
        
        # 4. Handle infinite values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"Cleaned shape: {df_clean.shape}")
        logger.info(f"Removed {original_shape[0] - df_clean.shape[0]} rows")
        
        self.preprocessing_stats['original_shape'] = original_shape
        self.preprocessing_stats['cleaned_shape'] = df_clean.shape
        self.preprocessing_stats['duplicates_removed'] = duplicates
        
        return df_clean
    
    def select_top_variable_genes(self, df: pd.DataFrame, label_column: str, 
                                   n_genes: int = 100) -> pd.DataFrame:
        """
        Select top N most variable genes to reduce dimensionality and speed up processing
        
        Args:
            df: Input DataFrame
            label_column: Name of the label column to preserve
            n_genes: Number of top variable genes to keep (default: 100)
        
        Returns:
            DataFrame with only top variable genes and label column
        """
        logger.info(f"Selecting top {n_genes} most variable genes...")
        
        # Get numeric columns (gene expressions) excluding label
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if label_column in numeric_cols:
            numeric_cols.remove(label_column)
        
        original_genes = len(numeric_cols)
        
        # If already fewer genes than target, keep all
        if original_genes <= n_genes:
            logger.info(f"Dataset has {original_genes} genes, keeping all")
            return df
        
        # Calculate variance for each gene
        gene_variance = df[numeric_cols].var()
        
        # Select top N most variable genes
        top_genes = gene_variance.nlargest(n_genes).index.tolist()
        
        # Keep label column and top genes
        columns_to_keep = top_genes + [label_column]
        df_selected = df[columns_to_keep]
        
        logger.info(f"Reduced from {original_genes} to {n_genes} genes ({(n_genes/original_genes)*100:.1f}%)")
        self.preprocessing_stats['original_genes'] = original_genes
        self.preprocessing_stats['selected_genes'] = n_genes
        self.preprocessing_stats['top_gene_names'] = top_genes
        
        return df_selected
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: Optional[str] = None) -> pd.DataFrame:
        """
        Handle missing values using various strategies
        
        Args:
            df: Input DataFrame
            strategy: 'mean', 'median', 'knn', 'drop'
        
        Returns:
            DataFrame with imputed values
        """
        strategy = strategy or self.config['missing_value_strategy']
        logger.info(f"Handling missing values using '{strategy}' strategy")
        
        df_imputed = df.copy()
        
        # Separate numeric and non-numeric columns
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        
        # Count missing values before
        missing_before = df_imputed[numeric_cols].isnull().sum().sum()
        logger.info(f"Missing values before: {missing_before}")
        
        # If no missing values, skip imputation
        if missing_before == 0:
            logger.info("No missing values found, skipping imputation")
            self.preprocessing_stats['missing_values_handled'] = 0
            return df_imputed
        
        if strategy == 'drop':
            df_imputed = df_imputed.dropna()
        elif strategy == 'mean':
            logger.info(f"Imputing {len(numeric_cols)} features with mean strategy...")
            # For very large datasets, use chunked processing
            if len(numeric_cols) > 10000:
                logger.info("Large dataset detected, using chunked imputation")
                chunk_size = 5000
                for i in range(0, len(numeric_cols), chunk_size):
                    chunk_cols = numeric_cols[i:i+chunk_size]
                    logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(numeric_cols)-1)//chunk_size + 1}")
                    imputer = SimpleImputer(strategy='mean')
                    df_imputed[chunk_cols] = imputer.fit_transform(df_imputed[chunk_cols])
            else:
                self.imputer = SimpleImputer(strategy='mean')
                df_imputed[numeric_cols] = self.imputer.fit_transform(df_imputed[numeric_cols])
        elif strategy == 'median':
            logger.info(f"Imputing {len(numeric_cols)} features with median strategy...")
            if len(numeric_cols) > 10000:
                logger.info("Large dataset detected, using chunked imputation")
                chunk_size = 5000
                for i in range(0, len(numeric_cols), chunk_size):
                    chunk_cols = numeric_cols[i:i+chunk_size]
                    logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(numeric_cols)-1)//chunk_size + 1}")
                    imputer = SimpleImputer(strategy='median')
                    df_imputed[chunk_cols] = imputer.fit_transform(df_imputed[chunk_cols])
            else:
                self.imputer = SimpleImputer(strategy='median')
                df_imputed[numeric_cols] = self.imputer.fit_transform(df_imputed[numeric_cols])
        elif strategy == 'knn':
            # KNN is very slow for large datasets - use simpler method instead
            if len(numeric_cols) > 5000:
                logger.warning(f"KNN imputation too slow for {len(numeric_cols)} features, using mean instead")
                self.imputer = SimpleImputer(strategy='mean')
                df_imputed[numeric_cols] = self.imputer.fit_transform(df_imputed[numeric_cols])
            else:
                self.imputer = KNNImputer(n_neighbors=5)
                df_imputed[numeric_cols] = self.imputer.fit_transform(df_imputed[numeric_cols])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Count missing values after
        missing_after = df_imputed[numeric_cols].isnull().sum().sum()
        logger.info(f"Missing values after: {missing_after}")
        
        self.preprocessing_stats['missing_values_handled'] = missing_before - missing_after
        
        return df_imputed
    
    def remove_outliers(self, df: pd.DataFrame, method: Optional[str] = None, 
                        label_column: Optional[str] = None) -> pd.DataFrame:
        """
        Remove outliers using various methods
        
        Args:
            df: Input DataFrame
            method: 'iqr', 'zscore', 'isolation_forest'
            label_column: Column to preserve (disease labels)
        
        Returns:
            DataFrame with outliers removed
        """
        method = method or self.config['outlier_method']
        logger.info(f"Removing outliers using '{method}' method")
        
        df_clean = df.copy()
        original_len = len(df_clean)
        
        # Get numeric columns (excluding label column)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if label_column and label_column in numeric_cols:
            numeric_cols.remove(label_column)
        
        # For large datasets (>10000 features), use sampling for speed
        if len(numeric_cols) > 10000:
            logger.info(f"Large dataset detected ({len(numeric_cols)} features), using optimized outlier detection")
            # Sample features for outlier detection
            sample_cols = np.random.choice(numeric_cols, min(1000, len(numeric_cols)), replace=False).tolist()
        else:
            sample_cols = numeric_cols
        
        if method == 'iqr':
            # IQR method - vectorized for speed
            mask = pd.Series([True] * len(df_clean), index=df_clean.index)
            removed_count = 0
            for col in sample_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                mask &= col_mask
                # Safety check: don't remove more than 30% of data
                if mask.sum() < original_len * 0.7:
                    logger.warning(f"IQR on {col} would remove >30% data, skipping this column")
                    mask |= ~col_mask  # Restore removed samples for this column
            df_clean = df_clean[mask]
        
        elif method == 'zscore':
            # Z-score method (remove if |z| > 3) - optimized
            z_scores = np.abs(stats.zscore(df_clean[sample_cols].fillna(0), axis=0))
            mask = (z_scores < 3).all(axis=1)
            df_clean = df_clean[mask]
        
        elif method == 'isolation_forest':
            # Isolation Forest - use sampling for large datasets
            iso_forest = IsolationForest(contamination=0.1, random_state=self.config['random_state'], n_jobs=-1)
            predictions = iso_forest.fit_predict(df_clean[sample_cols].fillna(0))
            df_clean = df_clean[predictions == 1]
        
        outliers_removed = original_len - len(df_clean)
        logger.info(f"Removed {outliers_removed} outliers ({outliers_removed/original_len*100:.2f}%)")
        
        self.preprocessing_stats['outliers_removed'] = outliers_removed
        
        return df_clean
    
    def normalize_data(self, df: pd.DataFrame, method: Optional[str] = None,
                       label_column: Optional[str] = None) -> pd.DataFrame:
        """
        Normalize/scale numeric features
        
        Args:
            df: Input DataFrame
            method: 'zscore', 'minmax', 'robust'
            label_column: Column to preserve (disease labels)
        
        Returns:
            Normalized DataFrame
        """
        method = method or self.config['normalization_method']
        logger.info(f"Normalizing data using '{method}' method")
        
        df_normalized = df.copy()
        
        # Get numeric columns (excluding label column)
        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
        if label_column and label_column in numeric_cols:
            numeric_cols.remove(label_column)
        
        # Store feature names
        self.feature_names = numeric_cols
        
        # Select scaler
        if method == 'zscore':
            self.scaler = StandardScaler(copy=False)  # In-place to save memory
        elif method == 'minmax':
            self.scaler = MinMaxScaler(copy=False)
        elif method == 'robust':
            self.scaler = RobustScaler(copy=False)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Fit and transform - convert to float32 to save memory
        logger.info(f"Normalizing {len(numeric_cols)} features...")
        
        # For very large datasets, use chunked processing
        if len(numeric_cols) > 10000:
            logger.info("Large dataset detected, using chunked normalization")
            chunk_size = 10000  # Increased from 5000 for faster processing
            for i in range(0, len(numeric_cols), chunk_size):
                chunk_cols = numeric_cols[i:i+chunk_size]
                logger.info(f"Normalizing chunk {i//chunk_size + 1}/{(len(numeric_cols)-1)//chunk_size + 1}")
                scaler = type(self.scaler)(copy=False)
                df_normalized[chunk_cols] = scaler.fit_transform(df_normalized[chunk_cols].astype(np.float32))
        else:
            df_normalized[numeric_cols] = self.scaler.fit_transform(df_normalized[numeric_cols].astype(np.float32))
        
        logger.info(f"Normalized {len(numeric_cols)} features")
        
        return df_normalized
    
    def encode_labels(self, labels: pd.Series, encoding_type: str = 'label') -> np.ndarray:
        """
        Encode disease labels
        
        Args:
            labels: Series with disease labels
            encoding_type: 'label' or 'onehot'
        
        Returns:
            Encoded labels
        """
        logger.info(f"Encoding labels using '{encoding_type}' encoding")
        
        if encoding_type == 'label':
            self.label_encoder = LabelEncoder()
            encoded = self.label_encoder.fit_transform(labels)
            logger.info(f"Label mapping: {dict(enumerate(self.label_encoder.classes_))}")
        elif encoding_type == 'onehot':
            self.label_encoder = OneHotEncoder(sparse=False)
            encoded = self.label_encoder.fit_transform(labels.values.reshape(-1, 1))
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        return encoded
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   include_validation: bool = True) -> Tuple:
        """
        Split data into train, test, and optionally validation sets
        
        Args:
            X: Features DataFrame
            y: Labels Series
            include_validation: Whether to create validation set
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) or 
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        test_size = self.config['test_size']
        val_size = self.config['validation_size']
        random_state = self.config['random_state']
        
        logger.info(f"Splitting data - Test: {test_size*100}%, Val: {val_size*100 if include_validation else 0}%")
        
        if include_validation:
            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Second split: separate validation from training
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, 
                random_state=random_state, stratify=y_temp
            )
            
            logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            self.preprocessing_stats['split_sizes'] = {
                'train': len(X_train),
                'validation': len(X_val),
                'test': len(X_test)
            }
            
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
            
            self.preprocessing_stats['split_sizes'] = {
                'train': len(X_train),
                'test': len(X_test)
            }
            
            return X_train, X_test, y_train, y_test
    
    def full_pipeline(self, df: pd.DataFrame, label_column: str,
                      include_validation: bool = True, max_genes: int = 100) -> Tuple:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input DataFrame
            label_column: Name of the label column
            include_validation: Whether to include validation set
            max_genes: Maximum number of genes to keep (default: 100 for faster processing)
        
        Returns:
            Processed train, validation (optional), and test sets
        """
        logger.info("=" * 50)
        logger.info("Starting Full Preprocessing Pipeline")
        logger.info("=" * 50)
        
        # Step 1: Clean data
        df_clean = self.clean_data(df)
        
        # Step 2: Select top variable genes (NEW - for speed optimization)
        df_selected = self.select_top_variable_genes(df_clean, label_column, n_genes=max_genes)
        
        # Step 3: Handle missing values
        df_imputed = self.handle_missing_values(df_selected)
        
        # Step 4: Remove outliers
        df_no_outliers = self.remove_outliers(df_imputed, label_column=label_column)
        
        # Step 4: Separate features and labels
        if label_column not in df_no_outliers.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset")
        
        y = df_no_outliers[label_column]
        X = df_no_outliers.drop(columns=[label_column])
        
        # Remove non-numeric columns from X (keep only gene expression)
        X = X.select_dtypes(include=[np.number])
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Labels distribution:\n{y.value_counts()}")
        
        # Step 5: Normalize features
        X_normalized = self.normalize_data(X)
        
        # Step 6: Encode labels
        y_encoded = self.encode_labels(y)
        
        # Step 7: Split data
        if include_validation:
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
                X_normalized, y_encoded, include_validation=True
            )
            result = (X_train, X_val, X_test, y_train, y_val, y_test)
        else:
            X_train, X_test, y_train, y_test = self.split_data(
                X_normalized, y_encoded, include_validation=False
            )
            result = (X_train, X_test, y_train, y_test)
        
        logger.info("=" * 50)
        logger.info("Preprocessing Pipeline Completed Successfully!")
        logger.info("=" * 50)
        
        return result
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor objects for future use"""
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'config': self.config,
            'stats': self.preprocessing_stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Load saved preprocessor objects"""
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.scaler = preprocessor_data['scaler']
        self.label_encoder = preprocessor_data['label_encoder']
        self.imputer = preprocessor_data['imputer']
        self.feature_names = preprocessor_data['feature_names']
        self.config = preprocessor_data['config']
        self.preprocessing_stats = preprocessor_data['stats']
        
        logger.info(f"Preprocessor loaded from {filepath}")
    
    def transform_new_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor (for real-time prediction)
        
        Args:
            df: New data to transform
        
        Returns:
            Transformed data ready for prediction
        """
        if self.scaler is None:
            raise ValueError("Preprocessor not fitted. Run full_pipeline first.")
        
        df_transformed = df.copy()
        
        # Select only numeric columns
        numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
        
        # Handle missing values
        if self.imputer:
            df_transformed[numeric_cols] = self.imputer.transform(df_transformed[numeric_cols])
        
        # Normalize
        df_transformed[numeric_cols] = self.scaler.transform(df_transformed[numeric_cols])
        
        return df_transformed[numeric_cols].values
    
    def get_preprocessing_report(self) -> str:
        """Generate a comprehensive preprocessing report"""
        report = "\n" + "=" * 60 + "\n"
        report += "PREPROCESSING REPORT\n"
        report += "=" * 60 + "\n\n"
        
        for key, value in self.preprocessing_stats.items():
            report += f"{key.replace('_', ' ').title()}:\n"
            if isinstance(value, dict):
                for k, v in value.items():
                    report += f"  {k}: {v}\n"
            else:
                report += f"  {value}\n"
            report += "\n"
        
        report += "=" * 60 + "\n"
        
        return report


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'gene_1': np.random.randn(100),
        'gene_2': np.random.randn(100),
        'gene_3': np.random.randn(100),
        'gene_4': np.random.randn(100),
        'disease_type': np.random.choice(['breast_cancer', 'lung_cancer'], 100)
    })
    
    # Add some missing values
    sample_data.loc[0:5, 'gene_1'] = np.nan
    
    preprocessor = GeneDataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.full_pipeline(
        sample_data, label_column='disease_type'
    )
    
    print(preprocessor.get_preprocessing_report())
    print(f"\nTrain set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
