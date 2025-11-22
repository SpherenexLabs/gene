# 📋 Quick Start Guide - ML Analysis

## ⚠️ Important: Dataset Requirements

### The ML pipeline requires datasets with LABELS

**What you need:**
- Gene expression values (numeric data)
- **Disease/Class labels** (categorical column indicating the condition)

### Current Issue with GEO Data:

The file `breast_cancer_GSE39582.csv` downloaded from GEO contains:
- ✅ Gene expression values (54,675 genes × 585 samples)
- ❌ **NO disease labels** - it's raw expression data only

**This type of data requires manual labeling before ML analysis can work.**

---

## 🎯 Two Options to Proceed:

### Option 1: Use the Sample Dataset (Recommended for Testing)

I've created a sample dataset with labels for you:

**File:** `data/raw/sample_gene_expression_with_labels.csv`

**Contains:**
- 200 samples
- 100 genes
- 3 classes: Breast_Cancer, Lung_Cancer, Healthy
- Label column: `disease_type`

**How to use:**
1. Go to "Upload Data" tab
2. Upload `sample_gene_expression_with_labels.csv`
3. Select disease type: "Breast Cancer"
4. Go to "Preprocess" tab
5. Select the uploaded file
6. Label column will auto-detect as `disease_type`
7. Click "🚀 Run Complete ML Analysis"

### Option 2: Add Labels to GEO Data Manually

You need to:
1. Know which samples correspond to which conditions (from GEO metadata)
2. Add a column with disease labels
3. Re-upload the modified file

**Example structure needed:**
```csv
ID_REF,GSM1681353,GSM1681354,...,disease_type
1007_s_at,9.77,9.90,...,Breast_Cancer
1053_at,6.26,5.84,...,Breast_Cancer
...
```

---

## 🚀 Quick Test with Sample Data

### Step-by-Step:

1. **Upload the sample file:**
   ```
   Navigate to: Upload Data tab
   Select file: C:\Users\USER\Desktop\gene\data\raw\sample_gene_expression_with_labels.csv
   Disease type: Breast Cancer
   Click: Upload & Validate
   ```

2. **Run ML Analysis:**
   ```
   Navigate to: Preprocess tab
   Select file: sample_gene_expression_with_labels.csv (auto-populated)
   Label column: disease_type (auto-detected)
   Click: 🚀 Run Complete ML Analysis
   ```

3. **Wait for Results:**
   - Processing: 2-5 minutes
   - Results will show:
     - Best model and accuracy
     - Performance comparison table
     - Top 10 important genes
     - Visualizations
     - Export files

4. **View Outputs:**
   ```
   Location: ml_results/TIMESTAMP/
   - visualizations/ (PNG files)
   - results/ (CSV, Excel, PDF)
   - models/ (Trained models)
   ```

---

## 📊 Expected Results

With the sample dataset, you should see:

**Model Performance:**
- Random Forest: ~85-95% accuracy
- SVM: ~80-90% accuracy
- Gradient Boosting: ~85-92% accuracy
- ANN: ~75-85% accuracy

**Top Genes:**
- GENE_0 through GENE_30 (for Breast Cancer)
- GENE_30 through GENE_60 (for Lung Cancer)
- These have the strongest discriminative power

**Visualizations:**
- Feature importance bar chart
- Confusion matrix (3×3 for 3 classes)
- ROC curves (one for each class)
- Model comparison chart
- PCA visualization

**Export Files:**
- PDF report with all results
- Excel workbook with multiple sheets
- CSV files for further analysis

---

## 🔧 Troubleshooting

### "No label column found"
➡️ **Solution:** Use the sample dataset OR add labels to your data manually

### "Preprocessing failed"
➡️ **Solution:** 
- Make sure file has disease labels
- Check that label column name is correct
- Restart Flask server: `python app.py`

### "File not found"
➡️ **Solution:** Upload the file first via "Upload Data" tab

### Server crashes during processing
➡️ **Solution:**
- Dataset might be too large
- Reduce `n_features` to 50 in the code
- Set `tune_hyperparameters: false`

---

## 📁 File Locations

**Sample Data:**
```
C:\Users\USER\Desktop\gene\data\raw\sample_gene_expression_with_labels.csv
```

**Uploaded Files:**
```
C:\Users\USER\Desktop\gene\uploads\
```

**ML Results:**
```
C:\Users\USER\Desktop\gene\ml_results\TIMESTAMP\
```

**GEO Data (no labels):**
```
C:\Users\USER\Desktop\gene\data\raw\breast_cancer_GSE39582.csv
```

---

## ✅ Next Steps

1. **Upload the sample dataset** to test the system
2. **Run ML analysis** and review results
3. **Download reports** from ml_results folder
4. For real analysis, prepare datasets with proper labels

---

## 💡 Tips

- **Faster Processing:** Set fewer models initially (just Random Forest)
- **Better Results:** Use `tune_hyperparameters: true` (but slower)
- **More Features:** Increase `n_features` to 150-200 for comprehensive analysis
- **Class Balance:** Ensure roughly equal samples per disease class

---

**Created:** Sample dataset ready to use!
**Test it now:** Upload → Preprocess → Run ML Analysis
