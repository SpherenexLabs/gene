@echo off
echo ========================================
echo Disease Gene Detection System
echo Installation and First Run
echo ========================================
echo.

echo [1/3] Installing dependencies...
echo.
pip install -r requirements.txt
echo.

echo [2/3] Testing system...
echo.
python test_system.py
echo.

echo [3/3] System ready!
echo.
echo ========================================
echo QUICK START OPTIONS
echo ========================================
echo.
echo Option 1 - Web Interface (Recommended):
echo    python app.py
echo    Then open: http://localhost:5000
echo.
echo Option 2 - Interactive Menu:
echo    python quick_start.py
echo.
echo Option 3 - Run Examples:
echo    python example_usage.py
echo.
echo Option 4 - Complete ML Pipeline:
echo    python complete_pipeline.py
echo.
echo ========================================
echo NEW ML FEATURES AVAILABLE:
echo ========================================
echo - 7 Feature Selection Methods
echo - 6 ML Classifiers (SVM, RF, ANN, KNN, GB, LR)
echo - Hyperparameter Tuning
echo - 8+ Visualization Types
echo - Export to CSV/Excel/PDF
echo See ML_PIPELINE_GUIDE.md for details
echo ========================================
echo.

pause
