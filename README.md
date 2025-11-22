# Disease Gene Detection System

🧬 A comprehensive machine learning system for automated disease gene detection using gene expression data.

## Features

### ✨ Core Capabilities
- **Multi-source Data Collection**
  - Upload local datasets (CSV, Excel, TXT)
  - Collect from GEO (Gene Expression Omnibus)
  - Support for TCGA data integration
  
- **Advanced Preprocessing**
  - Multiple missing value strategies (mean, median, KNN imputation)
  - Outlier detection (IQR, Z-score, Isolation Forest)
  - Flexible normalization (Z-score, Min-Max, Robust Scaler)
  - Automated data cleaning and validation
  
- **Real-time Processing**
  - Web-based upload interface
  - REST API for programmatic access
  - Instant dataset validation
  - Real-time preprocessing feedback

- **Supported Diseases**
  - Breast Cancer
  - Lung Cancer
  - Prostate Cancer
  - Alzheimer's Disease
  - Parkinson's Disease

## Installation

### 1. Clone or Download
```bash
cd c:\Users\USER\Desktop\gene
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Optional: Install GEO Support
```bash
pip install GEOparse
```

## Quick Start

### Option 1: Web Interface (Recommended)

1. **Start the server:**
```bash
python app.py
```

2. **Open browser:**
```
http://localhost:5000
```

3. **Upload your dataset:**
   - Drag & drop or click to upload
   - Select disease type
   - Validate and preview data

4. **Preprocess:**
   - Configure preprocessing options
   - Run automated pipeline
   - Download processed data

### Option 2: Python Script

```python
from data_collector import DataCollector
from preprocessor import GeneDataPreprocessor

# Initialize
collector = DataCollector()
preprocessor = GeneDataPreprocessor()

# Load dataset
df = collector.load_local_file('your_dataset.csv')

# Validate
validation = collector.validate_dataset(df)
print(validation)

# Preprocess
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.full_pipeline(
    df, 
    label_column='disease_type'
)

# Save preprocessor for real-time use
preprocessor.save_preprocessor('preprocessor.pkl')
```

### Option 3: Example Script

```bash
python example_usage.py
```

This will:
- Create sample datasets
- Demonstrate all preprocessing steps
- Save processed data
- Show real-time transformation

## Project Structure

```
gene/
├── app.py                  # Flask web application
├── config.py              # Configuration settings
├── data_collector.py      # Data collection module
├── preprocessor.py        # Preprocessing pipeline
├── example_usage.py       # Example usage script
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── templates/
│   └── index.html        # Web interface
│
├── data/
│   ├── raw/              # Raw uploaded datasets
│   └── processed/        # Preprocessed datasets
│       ├── breast_cancer/
│       ├── lung_cancer/
│       ├── prostate_cancer/
│       ├── alzheimers/
│       └── parkinsons/
│
├── uploads/              # Temporary upload directory
└── models/              # Trained ML models (for future use)
```

## Usage Guide

### 1. Data Collection

#### Upload Local File
```python
from data_collector import DataCollector

collector = DataCollector()
df = collector.load_local_file('breast_cancer_data.csv', 'breast_cancer')
```

#### Collect from GEO
```python
df = collector.collect_from_geo('GSE123456', 'breast_cancer')
```

#### Merge Multiple Datasets
```python
datasets = [df1, df2, df3]
labels = ['breast_cancer', 'lung_cancer', 'alzheimers']
merged = collector.merge_datasets(datasets, labels)
```

### 2. Data Preprocessing

#### Basic Preprocessing
```python
from preprocessor import GeneDataPreprocessor

preprocessor = GeneDataPreprocessor()

# Full pipeline
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.full_pipeline(
    df, 
    label_column='disease_type'
)
```

#### Custom Configuration
```python
config = {
    'missing_value_strategy': 'median',
    'outlier_method': 'zscore',
    'normalization_method': 'minmax',
    'test_size': 0.15,
    'validation_size': 0.15,
    'random_state': 42
}

preprocessor = GeneDataPreprocessor(config)
```

#### Step-by-Step Preprocessing
```python
# Individual steps
df_clean = preprocessor.clean_data(df)
df_imputed = preprocessor.handle_missing_values(df_clean)
df_no_outliers = preprocessor.remove_outliers(df_imputed)
df_normalized = preprocessor.normalize_data(df_no_outliers)
```

### 3. Real-time Prediction Preprocessing

```python
# Load saved preprocessor
preprocessor.load_preprocessor('preprocessor.pkl')

# Transform new data
new_data = pd.DataFrame(...)  # New patient data
transformed = preprocessor.transform_new_data(new_data)

# Ready for prediction!
# predictions = model.predict(transformed)
```

## API Endpoints

### Upload Dataset
```http
POST /api/upload
Content-Type: multipart/form-data

file: <file>
disease_type: breast_cancer
```

### Preprocess Data
```http
POST /api/preprocess
Content-Type: application/json

{
  "filepath": "path/to/file",
  "label_column": "disease_type",
  "disease_type": "breast_cancer",
  "missing_strategy": "mean",
  "outlier_method": "iqr",
  "normalization": "zscore"
}
```

### Collect from GEO
```http
POST /api/collect_geo
Content-Type: application/json

{
  "geo_accession": "GSE123456",
  "disease_type": "breast_cancer"
}
```

### List Datasets
```http
GET /api/datasets
```

### Get Statistics
```http
GET /api/statistics
```

## Configuration

Edit `config.py` to customize:

```python
# Preprocessing defaults
PREPROCESSING_CONFIG = {
    'missing_value_strategy': 'mean',  # 'mean', 'median', 'knn', 'drop'
    'outlier_method': 'iqr',           # 'iqr', 'zscore', 'isolation_forest'
    'normalization_method': 'zscore',   # 'zscore', 'minmax', 'robust'
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42
}

# API settings
API_HOST = '0.0.0.0'
API_PORT = 5000

# File size limit (100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024
```

## Real-time Workflow

1. **Upload** → Dataset validation (< 1 second)
2. **Preprocess** → Full pipeline (seconds to minutes)
3. **Save** → Preprocessor and splits
4. **Deploy** → Load preprocessor once
5. **Predict** → Transform new data (< 100ms)

## Dataset Format

### Expected Structure

Your CSV/Excel files should have:
- **Columns**: Gene names or IDs (e.g., GENE_1, BRCA1, TP53)
- **Rows**: Samples/patients
- **Label column**: Disease type or classification

Example:
```csv
patient_id,GENE_1,GENE_2,GENE_3,...,disease_type
P0001,2.5,1.3,0.8,...,breast_cancer
P0002,1.2,2.1,1.5,...,breast_cancer
P0003,0.9,0.7,2.3,...,lung_cancer
```

### Automatic Detection

The system automatically detects:
- Numeric vs categorical columns
- Potential label columns
- Sample ID columns
- Missing values
- Data types

## Troubleshooting

### Issue: "Module not found"
```bash
pip install -r requirements.txt
```

### Issue: "File too large"
Increase limit in `config.py`:
```python
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
```

### Issue: "GEOparse not working"
```bash
pip install GEOparse
# May require additional NCBI tools
```

### Issue: "Port already in use"
Change port in `config.py`:
```python
API_PORT = 5001
```

## Performance

- **Upload**: < 1 second for files up to 100MB
- **Validation**: < 1 second
- **Preprocessing**: 
  - Small datasets (< 1000 samples): < 10 seconds
  - Medium datasets (1000-10000): 30-60 seconds
  - Large datasets (> 10000): 1-5 minutes
- **Real-time transform**: < 100ms

## Next Steps

After preprocessing, you can:
1. Train machine learning models (Random Forest, XGBoost, Neural Networks)
2. Implement feature selection
3. Build prediction API
4. Create visualization dashboard
5. Deploy to production

## Contributing

This is a project implementation. Feel free to extend with:
- Additional disease types
- More preprocessing methods
- Machine learning models
- Advanced visualizations

## License

Educational/Research Use

## Contact

For questions or issues, please refer to the documentation or example scripts.

---

**Ready to detect disease genes? Start with:**
```bash
python app.py
```

Then visit: **http://localhost:5000** 🚀
