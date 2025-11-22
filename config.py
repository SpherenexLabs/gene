"""
Configuration file for Disease Gene Detection System
"""
import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, UPLOADS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Disease categories
DISEASE_CATEGORIES = [
    'breast_cancer',
    'lung_cancer',
    'prostate_cancer',
    'alzheimers',
    'parkinsons'
]

# Public repositories configuration
PUBLIC_REPOS = {
    'GEO': 'https://www.ncbi.nlm.nih.gov/geo/',
    'TCGA': 'https://portal.gdc.cancer.gov/',
    'GTEx': 'https://gtexportal.org/'
}

# Preprocessing parameters
PREPROCESSING_CONFIG = {
    'missing_value_strategy': 'mean',  # 'mean', 'median', 'drop', 'knn'
    'outlier_method': 'iqr',  # 'iqr', 'zscore', 'isolation_forest'
    'normalization_method': 'zscore',  # 'zscore', 'minmax', 'robust'
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42
}

# Supported file formats
ALLOWED_EXTENSIONS = {'csv', 'txt', 'xlsx', 'xls', 'tsv'}

# API configuration
API_HOST = '0.0.0.0'
API_PORT = int(os.environ.get('PORT', 5000))  # Use environment PORT or default to 5000
DEBUG_MODE = os.environ.get('DEBUG', 'False').lower() == 'true'

# Maximum file size (500MB)
MAX_FILE_SIZE = 500 * 1024 * 1024
